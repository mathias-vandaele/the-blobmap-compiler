//! The CPU reference runtime (SPEC §1, §9). It reads only the model's bytes (FORMAT.md):
//! components are loaded on demand, queries are full sweeps over one component's hierarchy
//! (upward by decreasing layout id, downward by increasing id, FORMAT §8), and the
//! operations of SPEC §9 are evaluated exactly, in integers.

pub mod model;
pub mod ops;
pub mod query;
pub mod recustomize;

pub use model::{Dir, Component, Fam, Faults, Reason, Runtime};
