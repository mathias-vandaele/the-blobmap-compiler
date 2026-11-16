//! `blobmap-runtime`: the GPU serving library (SPEC §9 on a GPU). For now, the CubeCL trial:
//! the research car sweep ported to CubeCL, against the research CUDA kernels.

pub mod layered;
#[cfg(feature = "cubecl")]
pub mod cube;
#[cfg(feature = "cuda")]
pub mod research;
