//! The BlobMap world-model compiler (SPEC.md), one module per part of the pipeline, in build
//! order: `profile`, `osm`, `graph`, `order`, `cch`, `overture`, `snap`, `build`; then
//! `verify`, the conformance suite (SPEC §11), and `fixtures`, the tiny tier's synthetic
//! inputs. The bytes of a model are `blobmap_model`'s.

pub mod build;
pub mod cch;
pub mod fixtures;
pub mod graph;
pub mod order;
pub mod osm;
pub mod overture;
pub mod profile;
pub mod snap;
pub mod verify;
