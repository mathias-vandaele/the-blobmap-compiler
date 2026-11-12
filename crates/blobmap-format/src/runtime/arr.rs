//! Arrays the runtime reads as slices: in memory, or, when the runtime has a scratch
//! directory, written to a file there and mapped, so that a continent's vocabulary and its
//! largest components live in the page cache rather than in the process (SPEC §10). A mapped
//! file is unlinked as soon as it is mapped: nothing is left behind. Small arrays stay in memory.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::marker::PhantomData;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result};
use bytemuck::Pod;
use memmap2::Mmap;

enum Inner<T> {
    Mem(Vec<T>),
    Map(Mmap, usize, PhantomData<T>),
}

pub struct Arr<T: Pod>(Inner<T>);

impl<T: Pod> Arr<T> {
    /// `v`, moved to a mapped file under `scratch` if there is one and it is not small.
    pub fn new(v: Vec<T>, scratch: Option<&Path>) -> Result<Arr<T>> {
        if scratch.is_none() || std::mem::size_of_val(&v[..]) < SMALL {
            return Ok(Arr(Inner::Mem(v)));
        }
        let mut w = ArrWriter::new(scratch)?;
        w.extend_from_slice(&v)?;
        drop(v);
        w.finish()
    }
}

impl<T: Pod> Default for Arr<T> {
    fn default() -> Arr<T> {
        Arr(Inner::Mem(Vec::new()))
    }
}

impl<T: Pod> Deref for Arr<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        match &self.0 {
            Inner::Mem(v) => v,
            Inner::Map(m, n, _) => &bytemuck::cast_slice(&m[..])[..*n],
        }
    }
}

impl<'a, T: Pod> IntoIterator for &'a Arr<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;
    fn into_iter(self) -> std::slice::Iter<'a, T> {
        self.iter()
    }
}

/// Distinct file names under one scratch directory, across threads.
static NEXT: AtomicU64 = AtomicU64::new(0);

/// An array below this many bytes stays in memory: a continent has hundreds of thousands of
/// small components, and a file for each of their arrays costs more than it saves.
const SMALL: usize = 1 << 20;

/// Builds an `Arr` by appending: to a `Vec`, which goes to a file under the scratch directory
/// once it reaches `SMALL` bytes, and the rest straight to that file.
pub struct ArrWriter<T: Pod> {
    mem: Vec<T>,
    scratch: Option<PathBuf>,
    file: Option<(BufWriter<File>, PathBuf)>,
    len: usize,
}

impl<T: Pod> ArrWriter<T> {
    pub fn new(scratch: Option<&Path>) -> Result<ArrWriter<T>> {
        Ok(ArrWriter { mem: Vec::new(), scratch: scratch.map(Path::to_path_buf), file: None, len: 0 })
    }

    pub fn push(&mut self, x: T) -> Result<()> {
        self.extend_from_slice(std::slice::from_ref(&x))
    }

    pub fn extend_from_slice(&mut self, xs: &[T]) -> Result<()> {
        self.len += xs.len();
        if let Some((w, _)) = &mut self.file {
            w.write_all(bytemuck::cast_slice(xs))?;
            return Ok(());
        }
        self.mem.extend_from_slice(xs);
        if let Some(dir) = &self.scratch
            && std::mem::size_of_val(&self.mem[..]) >= SMALL
        {
            std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
            let path = dir.join(format!("arr-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
            // Read as well as written: it is mapped once finished.
            let f = File::options().read(true).write(true).create_new(true).open(&path).with_context(|| format!("creating {}", path.display()))?;
            let mut w = BufWriter::with_capacity(1 << 20, f);
            w.write_all(bytemuck::cast_slice(&self.mem))?;
            self.mem = Vec::new();
            self.file = Some((w, path));
        }
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn finish(self) -> Result<Arr<T>> {
        let Some((w, path)) = self.file else {
            return Ok(Arr(Inner::Mem(self.mem)));
        };
        let f = w.into_inner().map_err(|e| e.into_error()).with_context(|| format!("writing {}", path.display()))?;
        // SAFETY: the file is private to this array, unlinked below, and never written again.
        let out = Arr(Inner::Map(unsafe { Mmap::map(&f) }.with_context(|| format!("mapping {}", path.display()))?, self.len, PhantomData));
        std::fs::remove_file(&path).with_context(|| format!("removing {}", path.display()))?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_in_memory_large_mapped() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/arr-test");
        let mut small = ArrWriter::new(Some(&dir)).unwrap();
        small.extend_from_slice(&[1u32, 2, 3]).unwrap();
        let small = small.finish().unwrap();
        assert!(matches!(small.0, Inner::Mem(_)) && small[..] == [1, 2, 3]);
        // Across the switch to a file: pushes, then slices, both sides of SMALL.
        let want: Vec<u32> = (0..(SMALL as u32 / 4) * 3 + 7).collect();
        let mut w = ArrWriter::new(Some(&dir)).unwrap();
        for &x in &want[..1000] {
            w.push(x).unwrap();
        }
        for c in want[1000..].chunks(70_001) {
            w.extend_from_slice(c).unwrap();
        }
        assert_eq!(w.len(), want.len());
        let big = w.finish().unwrap();
        assert!(matches!(big.0, Inner::Map(..)) && big[..] == want[..]);
        assert!(matches!(Arr::new(want.clone(), Some(&dir)).unwrap().0, Inner::Map(..)));
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0, "every file is unlinked");
    }
}
