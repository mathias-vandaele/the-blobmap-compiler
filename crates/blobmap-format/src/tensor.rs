//! An owned logical tensor and a unit of tensors (FORMAT §2.4).

use std::collections::BTreeMap;

use anyhow::{Result, ensure};

use crate::dtype::{Dtype, Element};

/// A logical tensor: dtype, shape and little-endian data.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    pub data: Vec<u8>,
}

impl Tensor {
    /// A tensor of the given shape. The element count must match.
    pub fn new<T: Element>(shape: &[u64], data: &[T]) -> Tensor {
        let n: u64 = shape.iter().product();
        assert_eq!(n as usize, data.len(), "tensor shape {shape:?} does not hold {} elements", data.len());
        Tensor { dtype: T::DTYPE, shape: shape.to_vec(), data: bytemuck::cast_slice(data).to_vec() }
    }

    /// A 1-D tensor.
    pub fn vec<T: Element>(data: &[T]) -> Tensor {
        Tensor::new(&[data.len() as u64], data)
    }

    /// A tensor of rows of `width` elements (`[n, width]`).
    pub fn rows<T: Element>(width: u64, data: &[T]) -> Tensor {
        assert!(width > 0 && (data.len() as u64).is_multiple_of(width), "data is not a whole number of rows of {width}");
        Tensor::new(&[data.len() as u64 / width, width], data)
    }

    /// A tensor of `n` rows of `W` elements, row `i` being `f(i)`, made in place in its
    /// bytes (no second copy of the data). `shape` must hold `n * W` elements.
    pub fn from_rows<T: Element, const W: usize>(shape: &[u64], n: usize, f: impl Fn(usize) -> [T; W] + Sync) -> Tensor {
        use rayon::prelude::*;
        let row = W * std::mem::size_of::<T>();
        assert_eq!(shape.iter().product::<u64>(), (n * W) as u64, "tensor shape {shape:?} does not hold {n} rows of {W}");
        let mut data = vec![0u8; n * row];
        if row > 0 {
            data.par_chunks_mut(row).enumerate().for_each(|(i, b)| b.copy_from_slice(bytemuck::cast_slice(&f(i)[..])));
        }
        Tensor { dtype: T::DTYPE, shape: shape.to_vec(), data }
    }

    /// A `U8` tensor that takes its bytes without copying them.
    pub fn of_bytes(shape: &[u64], data: Vec<u8>) -> Tensor {
        assert_eq!(shape.iter().product::<u64>(), data.len() as u64, "tensor shape {shape:?} does not hold {} bytes", data.len());
        Tensor { dtype: Dtype::U8, shape: shape.to_vec(), data }
    }

    /// Bytes per index of dimension 0 (FORMAT §2.2).
    pub fn row_bytes(&self) -> u64 {
        self.shape.iter().skip(1).product::<u64>() * self.dtype.size() as u64
    }

    pub fn rows_count(&self) -> u64 {
        self.shape.first().copied().unwrap_or(1)
    }

    pub fn check(&self) -> Result<()> {
        ensure!(!self.shape.is_empty(), "a tensor has at least one dimension");
        let n: u64 = self.shape.iter().product();
        ensure!(n * self.dtype.size() as u64 == self.data.len() as u64, "shape {:?} does not match {} bytes", self.shape, self.data.len());
        Ok(())
    }
}

/// One unit: the tensors that are written together and never share a shard with another
/// unit (FORMAT §2.4), by full logical name.
#[derive(Default, Debug)]
pub struct Unit {
    pub tensors: BTreeMap<String, Tensor>,
}

impl Unit {
    pub fn put(&mut self, name: impl Into<String>, t: Tensor) -> &mut Self {
        let name = name.into();
        assert!(!name.contains('@'), "{name}: '@' is reserved for chunks");
        let old = self.tensors.insert(name.clone(), t);
        assert!(old.is_none(), "tensor {name} written twice");
        self
    }

    /// A tensor under `<prefix>/<name>`.
    pub fn put_in(&mut self, prefix: &str, name: &str, t: Tensor) -> &mut Self {
        self.put(format!("{prefix}/{name}"), t)
    }
}
