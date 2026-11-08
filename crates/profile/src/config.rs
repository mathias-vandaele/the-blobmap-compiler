//! A config directory (`blobc build --config`, SPEC §10): the two profiles, the place rules
//! and the build options, each a versioned TOML file. The first three change answers, so
//! their hashes go into the manifest and into `inputs_hash`; `build.toml` only shapes the
//! output (FORMAT §3 `build`).

use std::path::Path;

use anyhow::{Context, Result, ensure};
use serde::Deserialize;

use crate::{ActiveFile, CarFile, Loaded};

/// `places.toml` (SPEC §3).
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PlaceRules {
    pub rules: PlaceHeader,
    pub confidence: Confidence,
    pub duplicates: Duplicates,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PlaceHeader {
    pub name: String,
    pub version: u32,
    pub closed_statuses: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Confidence {
    pub default: f64,
    /// (taxonomy node, minimum): the first whose node is the primary category or one of its
    /// ancestors applies.
    pub under: Vec<(String, f64)>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Duplicates {
    pub radius_m: f64,
}

/// `build.toml`: options that never change an answer.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BuildFile {
    pub build: BuildOptions,
    #[serde(default)]
    pub order: FixedOrder,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BuildOptions {
    pub shard_target_bytes: u64,
    pub shard_max_bytes: u64,
    pub memory_budget_bytes: u64,
    pub group_max_nodes: u64,
}

/// A hand-made node order for the ACTIVE family, as OSM node ids from the first contracted
/// to the last (the toy town, SPEC §11). Empty: nested dissection.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FixedOrder {
    #[serde(default)]
    pub active: Vec<i64>,
}

/// Everything in a config directory.
#[derive(Clone, Debug)]
pub struct Config {
    pub car: Loaded<CarFile>,
    pub active: Loaded<ActiveFile>,
    pub places: Loaded<PlaceRules>,
    pub build: BuildFile,
}

pub const CAR_FILE: &str = "car.toml";
pub const ACTIVE_FILE: &str = "active.toml";
pub const PLACES_FILE: &str = "places.toml";
pub const BUILD_FILE: &str = "build.toml";

impl Config {
    pub fn load(dir: &Path) -> Result<Config> {
        let car = crate::load::<CarFile>(&dir.join(CAR_FILE))?;
        let active = crate::load::<ActiveFile>(&dir.join(ACTIVE_FILE))?;
        let path = dir.join(PLACES_FILE);
        let text = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        let places = parse_places(&text, &path.display().to_string())?;
        let path = dir.join(BUILD_FILE);
        let text = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        let build: BuildFile = toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        let b = &build.build;
        ensure!(b.shard_target_bytes > 0 && b.shard_target_bytes <= b.shard_max_bytes, "build: shard target must be in 1..=max");
        ensure!(b.memory_budget_bytes > 0, "build: memory_budget_bytes must be positive");
        Ok(Config { car, active, places, build })
    }
}

pub fn parse_places(text: &str, source: &str) -> Result<Loaded<PlaceRules>> {
    use sha2::Digest;
    let config: PlaceRules = toml::from_str(text).with_context(|| format!("parsing place rules {source}"))?;
    ensure!(
        config.rules.name == "places" && config.rules.version == 1,
        "{source}: place rules version {} (this compiler reads 1)",
        config.rules.version
    );
    ensure!(config.duplicates.radius_m >= 0.0, "{source}: duplicates.radius_m must not be negative");
    Ok(Loaded { config, sha256: hex::encode(sha2::Sha256::digest(text.as_bytes())), source: source.to_string() })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_repository_config_loads() {
        let c = Config::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../../config")).unwrap();
        assert_eq!(c.places.config.confidence.default, 0.6);
        assert_eq!(c.places.config.confidence.under, [("health_care".to_string(), 0.5)]);
        assert_eq!(c.places.config.duplicates.radius_m, 30.0);
        assert_eq!(c.build.build.shard_target_bytes, 2 << 30);
        assert!(c.build.order.active.is_empty());
        assert_eq!(c.car.sha256.len(), 64);
    }
}
