//! The dtypes of FORMAT §1, with their safetensors names.

use anyhow::{Result, bail};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Dtype {
    U8,
    U16,
    U32,
    U64,
    I64,
    F32,
}

impl Dtype {
    pub fn name(self) -> &'static str {
        match self {
            Dtype::U8 => "U8",
            Dtype::U16 => "U16",
            Dtype::U32 => "U32",
            Dtype::U64 => "U64",
            Dtype::I64 => "I64",
            Dtype::F32 => "F32",
        }
    }

    /// Bytes per element.
    pub fn size(self) -> usize {
        match self {
            Dtype::U8 => 1,
            Dtype::U16 => 2,
            Dtype::U32 | Dtype::F32 => 4,
            Dtype::U64 | Dtype::I64 => 8,
        }
    }

    pub fn parse(name: &str) -> Result<Dtype> {
        Ok(match name {
            "U8" => Dtype::U8,
            "U16" => Dtype::U16,
            "U32" => Dtype::U32,
            "U64" => Dtype::U64,
            "I64" => Dtype::I64,
            "F32" => Dtype::F32,
            _ => bail!("dtype {name} is not part of the format"),
        })
    }
}

/// A Rust element type of a dtype.
pub trait Element: bytemuck::Pod {
    const DTYPE: Dtype;
}

impl Element for u8 {
    const DTYPE: Dtype = Dtype::U8;
}
impl Element for u16 {
    const DTYPE: Dtype = Dtype::U16;
}
impl Element for u32 {
    const DTYPE: Dtype = Dtype::U32;
}
impl Element for u64 {
    const DTYPE: Dtype = Dtype::U64;
}
impl Element for i64 {
    const DTYPE: Dtype = Dtype::I64;
}
impl Element for f32 {
    const DTYPE: Dtype = Dtype::F32;
}
