//! `manifest.json` (FORMAT §3). Reading ignores keys it does not know (SPEC §8).

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::json;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Manifest {
    pub format: String,
    pub format_version: String,
    pub model_id: String,
    pub inputs_hash: String,
    pub spec_version: String,
    pub compiler: Compiler,
    pub inputs: Inputs,
    pub config: Config,
    pub build: Build,
    pub families: BTreeMap<String, Family>,
    pub modes: BTreeMap<String, Mode>,
    pub experts: Vec<Entry>,
    pub parts: Parts,
    pub attribution: Vec<String>,
    pub licenses_sha256: String,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Compiler {
    pub name: String,
    pub version: String,
    pub commit: String,
    pub dirty: bool,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct NamedFile {
    pub name: String,
    pub sha256: String,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Pbf {
    pub name: String,
    pub bytes: u64,
    pub sha256: String,
    pub timestamp: Option<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Overture {
    pub release: String,
    pub files: Vec<NamedFile>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Taxonomy {
    pub version: String,
    pub files: Vec<NamedFile>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Inputs {
    pub pbf: Pbf,
    pub overture: Overture,
    pub taxonomy: Taxonomy,
    pub extent: Option<NamedFile>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Profiles {
    pub car: NamedFile,
    pub active: NamedFile,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Config {
    pub profiles: Profiles,
    pub place_rules: NamedFile,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Build {
    pub shard_target_bytes: u64,
    pub shard_max_bytes: u64,
    pub memory_budget_bytes: u64,
    pub group_max_nodes: u64,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Family {
    pub graph: String,
    pub metrics: Vec<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Mode {
    pub family: String,
    pub metric: String,
    pub snap_radius_m: u64,
}

/// A manifest entry: one expert, or an expert group (FORMAT §3).
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Entry {
    pub family: String,
    pub kind: String,
    pub experts: [u64; 2],
    pub components: u64,
    pub nodes: u64,
    pub links: u64,
    pub levels: u64,
    pub cells: u64,
    pub packs: [u64; 2],
    pub shards: Vec<String>,
    pub hash: String,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct RoadPart {
    pub hash: String,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct VocabPart {
    pub hash: String,
    pub road_hash: String,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Parts {
    pub road: RoadPart,
    pub vocabulary: VocabPart,
}

impl Manifest {
    pub fn write(&self, path: &Path) -> Result<()> {
        json::write_file(path, &serde_json::to_value(self)?)
    }

    pub fn from_value(v: &serde_json::Value) -> Result<Manifest> {
        Ok(serde_json::from_value(v.clone())?)
    }
}
