//! Arrays the runtime reads as slices: in memory, or, when the runtime has a scratch
//! directory, written to a file there and mapped, so that a continent's vocabulary and its
//! largest components live in the page cache rather than in the process (SPEC §10). A mapped
//! file is unlinked as soon as it is mapped: nothing is left behind.

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
    /// `v`, moved to a mapped file under `scratch` if there is one.
    pub fn new(v: Vec<T>, scratch: Option<&Path>) -> Result<Arr<T>> {
        if scratch.is_none() {
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

/// Builds an `Arr` by appending: to a `Vec`, or straight to its file.
pub enum ArrWriter<T: Pod> {
    Mem(Vec<T>),
    File { w: BufWriter<File>, path: PathBuf, len: usize },
}

impl<T: Pod> ArrWriter<T> {
    pub fn new(scratch: Option<&Path>) -> Result<ArrWriter<T>> {
        let Some(dir) = scratch else {
            return Ok(ArrWriter::Mem(Vec::new()));
        };
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        let path = dir.join(format!("arr-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
        // Read as well as written: it is mapped once finished.
        let f = File::options().read(true).write(true).create_new(true).open(&path).with_context(|| format!("creating {}", path.display()))?;
        Ok(ArrWriter::File { w: BufWriter::with_capacity(1 << 20, f), path, len: 0 })
    }

    pub fn push(&mut self, x: T) -> Result<()> {
        self.extend_from_slice(std::slice::from_ref(&x))
    }

    pub fn extend_from_slice(&mut self, xs: &[T]) -> Result<()> {
        match self {
            ArrWriter::Mem(v) => v.extend_from_slice(xs),
            ArrWriter::File { w, len, .. } => {
                w.write_all(bytemuck::cast_slice(xs))?;
                *len += xs.len();
            }
        }
        Ok(())
    }

    pub fn len(&self) -> usize {
        match self {
            ArrWriter::Mem(v) => v.len(),
            ArrWriter::File { len, .. } => *len,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn finish(self) -> Result<Arr<T>> {
        match self {
            ArrWriter::Mem(v) => Ok(Arr(Inner::Mem(v))),
            ArrWriter::File { w, path, len } => {
                let f = w.into_inner().map_err(|e| e.into_error()).with_context(|| format!("writing {}", path.display()))?;
                let out = if len == 0 {
                    Arr(Inner::Mem(Vec::new()))
                } else {
                    // SAFETY: the file is private to this array, unlinked below, and never
                    // written again.
                    Arr(Inner::Map(unsafe { Mmap::map(&f) }.with_context(|| format!("mapping {}", path.display()))?, len, PhantomData))
                };
                std::fs::remove_file(&path).with_context(|| format!("removing {}", path.display()))?;
                Ok(out)
            }
        }
    }
}
