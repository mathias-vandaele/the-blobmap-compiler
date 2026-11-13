//! Columns of plain values (SPEC §10, memory): held in memory, or written to a file of a
//! work directory as they are made and then mapped read-only, so that a family's pieces
//! and ways need not stay resident whatever the size of the input. Both read as slices.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::marker::PhantomData;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use memmap2::{Mmap, UncheckedAdvice};

/// Values stored as their bytes: no padding, no pointers, every bit pattern written by
/// this module read back as written.
///
/// # Safety
/// Implementors must be `Copy` types without padding or invalid bit patterns among those
/// a `Column` of them ever writes.
pub unsafe trait Plain: Copy + Send + Sync + 'static {}

// SAFETY: plain numbers and arrays of them; `bool` columns only ever hold 0 or 1 bytes.
unsafe impl Plain for u8 {}
unsafe impl Plain for bool {}
unsafe impl Plain for u32 {}
unsafe impl Plain for u64 {}
unsafe impl Plain for i64 {}
unsafe impl Plain for f32 {}
unsafe impl Plain for f64 {}
unsafe impl Plain for [f32; 2] {}
unsafe impl Plain for [f64; 2] {}
unsafe impl Plain for [u32; 2] {}
unsafe impl Plain for [u64; 2] {}
unsafe impl Plain for u16 {}
unsafe impl Plain for [u8; 16] {}

const _: () = assert!(cfg!(target_endian = "little"), "columns are read in place as little-endian");

enum Inner<T> {
    Mem(Vec<T>),
    Map(Arc<Mmap>, usize),
}

/// A column: a `Vec` or a mapped file, read as `[T]`.
pub struct Column<T: Plain>(Inner<T>);

impl<T: Plain> Column<T> {
    /// Appends to an in-memory column. A mapped column is read-only.
    pub fn push(&mut self, x: T) {
        match &mut self.0 {
            Inner::Mem(v) => v.push(x),
            Inner::Map(..) => panic!("a mapped column is read-only"),
        }
    }

    pub fn extend(&mut self, xs: impl IntoIterator<Item = T>) {
        match &mut self.0 {
            Inner::Mem(v) => v.extend(xs),
            Inner::Map(..) => panic!("a mapped column is read-only"),
        }
    }

    /// Moves an in-memory column to `dir`'s file `name`, mapped; a mapped column, or one
    /// without a directory, stays as it is.
    pub fn spill(&mut self, dir: Option<&Path>, name: &str) -> Result<()> {
        if let (Inner::Mem(v), Some(_)) = (&self.0, dir) {
            let mut w = ColumnWriter::new(dir, name)?;
            w.push_slice(v)?;
            *self = Column::default();
            *self = w.finish()?;
        }
        Ok(())
    }

    /// Drops a mapped column's pages from the resident set (they stay in the page cache).
    pub fn release(&self) {
        if let Inner::Map(m, _) = &self.0
            && !m.is_empty()
        {
            // SAFETY: the map is a read-only view of a file nobody changes; dropped pages
            // fault back in from the file on the next read.
            let _ = unsafe { m.unchecked_advise_range(UncheckedAdvice::DontNeed, 0, m.len()) };
        }
    }
}

impl<T: Plain> Deref for Column<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        match &self.0 {
            Inner::Mem(v) => v,
            Inner::Map(_, 0) => &[],
            // SAFETY: the file holds `n` values of `T` written by `ColumnWriter`, and the map
            // is page-aligned, which is at least `T`'s alignment.
            Inner::Map(m, n) => unsafe { std::slice::from_raw_parts(m.as_ptr() as *const T, *n) },
        }
    }
}

impl<'a, T: Plain> IntoIterator for &'a Column<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T: Plain> Default for Column<T> {
    fn default() -> Self {
        Column(Inner::Mem(Vec::new()))
    }
}

impl<T: Plain> Clone for Column<T> {
    fn clone(&self) -> Self {
        match &self.0 {
            Inner::Mem(v) => Column(Inner::Mem(v.clone())),
            Inner::Map(m, n) => Column(Inner::Map(m.clone(), *n)),
        }
    }
}

impl<T: Plain> std::fmt::Debug for Column<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Column({} values)", self.len())
    }
}

impl<T: Plain> From<Vec<T>> for Column<T> {
    fn from(v: Vec<T>) -> Self {
        Column(Inner::Mem(v))
    }
}

/// A column being made: in memory, or appended to a file.
pub enum ColumnWriter<T: Plain> {
    Mem(Vec<T>),
    File { out: BufWriter<File>, path: PathBuf, n: usize, _t: PhantomData<T> },
}

impl<T: Plain> ColumnWriter<T> {
    /// A column in `dir`'s file `name`, or in memory without a directory.
    pub fn new(dir: Option<&Path>, name: &str) -> Result<ColumnWriter<T>> {
        Ok(match dir {
            None => ColumnWriter::Mem(Vec::new()),
            Some(d) => {
                std::fs::create_dir_all(d).with_context(|| format!("creating {}", d.display()))?;
                let path = d.join(name);
                let out = BufWriter::with_capacity(1 << 20, File::create(&path).with_context(|| format!("creating {}", path.display()))?);
                ColumnWriter::File { out, path, n: 0, _t: PhantomData }
            }
        })
    }

    pub fn push(&mut self, x: T) -> Result<()> {
        match self {
            ColumnWriter::Mem(v) => v.push(x),
            ColumnWriter::File { out, n, .. } => {
                // SAFETY: `T: Plain` has no padding, so all its bytes are initialised.
                let bytes = unsafe { std::slice::from_raw_parts(&x as *const T as *const u8, size_of::<T>()) };
                out.write_all(bytes)?;
                *n += 1;
            }
        }
        Ok(())
    }

    pub fn push_slice(&mut self, xs: &[T]) -> Result<()> {
        match self {
            ColumnWriter::Mem(v) => v.extend_from_slice(xs),
            ColumnWriter::File { out, n, .. } => {
                // SAFETY: as in `push`.
                let bytes = unsafe { std::slice::from_raw_parts(xs.as_ptr() as *const u8, std::mem::size_of_val(xs)) };
                out.write_all(bytes)?;
                *n += xs.len();
            }
        }
        Ok(())
    }

    pub fn len(&self) -> usize {
        match self {
            ColumnWriter::Mem(v) => v.len(),
            ColumnWriter::File { n, .. } => *n,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn finish(self) -> Result<Column<T>> {
        Ok(match self {
            ColumnWriter::Mem(v) => Column(Inner::Mem(v)),
            ColumnWriter::File { mut out, path, n, .. } => {
                out.flush()?;
                drop(out);
                let f = File::open(&path).with_context(|| format!("opening {}", path.display()))?;
                // SAFETY: the file was just written by this writer, lives in a work
                // directory of this process, and is never changed afterwards.
                let m = unsafe { Mmap::map(&f)? };
                Column(Inner::Map(Arc::new(m), n))
            }
        })
    }
}
