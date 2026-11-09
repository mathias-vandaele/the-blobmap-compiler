//! The byte-level contract of FORMAT.md: dtypes and tensors (§1), the deterministic shard
//! writer and the index (§2), the manifest (§3), hashes (§9) and the validating reader
//! (§10); and the CPU reference runtime (`runtime`), which answers SPEC §9 from those bytes
//! alone. Nothing here depends on the compiler.

pub mod dtype;
pub mod hash;
pub mod json;
pub mod manifest;
pub mod reader;
pub mod runtime;
pub mod tensor;
pub mod writer;

pub use hash::Hash;
pub use reader::Model;
pub use tensor::{Tensor, Unit};
pub use writer::{ModelWriter, ShardLimits};

/// The format name and version this crate reads and writes (FORMAT §12).
pub const FORMAT: &str = "blobmap-model";
pub const FORMAT_VERSION: &str = "3.0.0";
pub const FORMAT_MAJOR: u64 = 3;

pub const MANIFEST_FILE: &str = "manifest.json";
pub const INDEX_FILE: &str = "model.safetensors.index.json";
pub const LICENSES_FILE: &str = "LICENSES";

/// `NONE` in a `U32` index, `NONE16` in a `U16` taxonomy index, `INF` in a `U16` time
/// (FORMAT §1).
pub const NONE: u32 = u32::MAX;
pub const NONE16: u16 = u16::MAX;
pub const INF: u16 = u16::MAX;

/// `a ⊕ b`: saturating addition of two times (FORMAT §1, SPEC §7).
#[inline]
pub fn sat(a: u16, b: u16) -> u16 {
    a.saturating_add(b)
}

/// `rnd(x)`: the nearest integer, ties away from zero, for x ≥ 0 (FORMAT §1).
#[inline]
pub fn rnd(x: f64) -> u64 {
    debug_assert!(x >= 0.0);
    x.round() as u64
}

/// The major version of a `MAJOR.MINOR.PATCH` string.
pub fn major(version: &str) -> Option<u64> {
    version.split('.').next()?.parse().ok()
}
