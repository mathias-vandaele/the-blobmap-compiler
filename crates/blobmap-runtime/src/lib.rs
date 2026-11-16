//! blobmap-runtime: answers every operation of SPEC §9 from a compiled model, exactly as
//! `blobmap_model::reference` does, with the downward sweeps of its columns on a GPU
//! (CubeCL: its CUDA runtime or its wgpu one) or on the CPU.
//!
//! - `vocab`: the vocabulary and the family tables (FORMAT §5.1, §7);
//! - `graph`: a component's hierarchy, renumbered by level, and the upward half on the host;
//! - `sweep`: the downward half's semantics, pruning included, and the CPU sweep;
//! - `gpu`: the CubeCL kernels and the device layout; `engine`: the GPU worker and batching;
//! - `runtime`: columns, pairs, heads (FORMAT §8); `ops`: the operations and golden JSON.

pub mod engine;
pub mod gpu;
pub mod graph;
pub mod ops;
pub mod runtime;
pub mod sweep;
pub mod vocab;

pub use ops::{Answer, Best, Expr, Kind as MatchKind, Matches, Refusal, Source, Target, Term, Time};
pub use runtime::{Backend, Options, Runtime};
pub use vocab::{Category, Dir, Id, Reason};
