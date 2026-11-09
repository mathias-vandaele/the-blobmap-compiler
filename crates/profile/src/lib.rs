//! Routing profiles from versioned config files (`config/*.toml`): the car profile
//! (OSRM v6.0.0 car.lua speed rules, obstacle delays, turn costs) and the ACTIVE profile
//! (bike and walk speeds, node costs). The rules' evaluation order is code; the numbers and
//! tag lists are config, and the file's hash goes into the manifest (`Loaded::sha256`).

pub mod active;
pub mod config;
pub mod profile;
pub mod ways;

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use sha2::{Digest, Sha256};

/// A profile file as loaded: the parsed config, the file's bytes' sha256 and its path.
#[derive(Clone, Debug)]
pub struct Loaded<C> {
    pub config: C,
    pub sha256: String,
    pub source: String,
}

/// What every profile file starts with.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Header {
    pub name: String,
    pub version: u32,
    pub family: String,
    #[serde(default)]
    pub modes: Vec<String>,
    pub reference: String,
}

/// A profile config: its header and the checks a file must pass after parsing.
pub trait ProfileConfig: for<'de> Deserialize<'de> {
    const FAMILY: &'static str;
    const VERSION: u32;
    fn header(&self) -> &Header;
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

/// Parse and check a profile from its text; `source` names it in errors and in `Loaded`.
pub fn parse<C: ProfileConfig>(text: &str, source: &str) -> Result<Loaded<C>> {
    let config: C = toml::from_str(text).with_context(|| format!("parsing profile {source}"))?;
    let h = config.header();
    ensure!(h.family == C::FAMILY, "{source}: family {} is not {}", h.family, C::FAMILY);
    ensure!(h.version == C::VERSION, "{source}: profile version {} (this compiler reads {})", h.version, C::VERSION);
    config.validate().with_context(|| format!("checking profile {source}"))?;
    Ok(Loaded { config, sha256: hex::encode(Sha256::digest(text.as_bytes())), source: source.to_string() })
}

pub fn load<C: ProfileConfig>(path: &Path) -> Result<Loaded<C>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    parse(&text, &path.display().to_string())
}

/// The car profile (`config/car.toml`).
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CarFile {
    pub profile: Header,
    pub speeds: CarSpeeds,
    pub ways: CarWays,
    pub access: CarAccess,
    pub obstacles: CarObstacles,
    pub turns: CarTurns,
    pub snap: SnapRules,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CarSpeeds {
    pub maxspeed_factor: f32,
    pub maxspeed_keys: Vec<String>,
    pub movable_bridge_kmh: f32,
    pub highway: BTreeMap<String, f32>,
    pub zone_defaults: BTreeMap<String, f32>,
    pub maxspeed_table: BTreeMap<String, f32>,
    /// Caps by tag key (`surface`, `tracktype`, `smoothness`), then value.
    pub caps: BTreeMap<String, BTreeMap<String, f32>>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CarWays {
    pub exclude: Vec<(String, String)>,
    pub implied_oneway_highways: Vec<String>,
    pub implied_oneway_junctions: Vec<String>,
    pub oneway_forward: Vec<String>,
    pub oneway_backward: Vec<String>,
    pub oneway_none: Vec<String>,
    pub oneway_drop: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CarAccess {
    pub keys: Vec<String>,
    pub blacklist: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CarObstacles {
    pub traffic_signals_s: f32,
    pub stop_s: f32,
    pub stop_minor_s: f32,
    pub give_way_s: f32,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CarTurns {
    pub turn_penalty_s: f64,
    pub turn_bias: f64,
    pub u_turn_penalty_s: f64,
}

/// Snapping (SPEC §6): the radius, Δ, and which pieces an ID may stand on. A piece is
/// ineligible if its `highway` is in `exclude_highways`, or if it has a key of
/// `exclude_keys` with a value not in `keep_values` (`bridge=no` is no bridge). A servable
/// piece of a larger component at most `larger_within_m` (Δ) beyond the nearest one wins.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SnapRules {
    pub radius_m: u64,
    pub larger_within_m: f64,
    pub exclude_highways: Vec<String>,
    pub exclude_keys: Vec<String>,
    pub keep_values: Vec<String>,
}

impl SnapRules {
    pub fn eligible<'a>(&self, mut tags: impl Iterator<Item = (&'a str, &'a str)>) -> bool {
        !tags.any(|(k, v)| {
            (k == "highway" && self.exclude_highways.iter().any(|h| h == v))
                || (self.exclude_keys.iter().any(|x| x == k) && !self.keep_values.iter().any(|x| x == v))
        })
    }
}

/// The three tag keys car.lua caps speeds by, in its order.
pub const CAR_CAP_KEYS: [&str; 3] = ["surface", "tracktype", "smoothness"];

impl ProfileConfig for CarFile {
    const FAMILY: &'static str = "car";
    const VERSION: u32 = 1;

    fn header(&self) -> &Header {
        &self.profile
    }

    fn validate(&self) -> Result<()> {
        let s = &self.speeds;
        ensure!(s.maxspeed_factor > 0.0, "speeds.maxspeed_factor must be positive");
        ensure!(s.highway.values().all(|&v| v > 0.0), "every speeds.highway value must be positive");
        for k in s.caps.keys() {
            ensure!(CAR_CAP_KEYS.contains(&k.as_str()), "speeds.caps.{k}: not one of {CAR_CAP_KEYS:?}");
        }
        ensure!(s.maxspeed_table.keys().all(|k| *k == k.to_lowercase()), "speeds.maxspeed_table keys must be lower case");
        ensure!(self.snap.radius_m > 0, "snap.radius_m must be positive");
        ensure!(self.snap.larger_within_m >= 0.0, "snap.larger_within_m must not be negative");
        Ok(())
    }
}

/// The ACTIVE profile (`config/active.toml`).
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ActiveFile {
    pub profile: Header,
    pub never: ActiveNever,
    pub access: ActiveAccess,
    pub walk: Walk,
    pub bike: Bike,
    pub snap: SnapRules,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ActiveNever {
    pub highway: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ActiveAccess {
    pub walk_closed: Vec<String>,
    pub bike_closed: Vec<String>,
    pub open: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Walk {
    pub kmh: f32,
    pub default_highways: Vec<String>,
    pub signal_s: f32,
    pub crossing_s: f32,
    pub surface_factor: BTreeMap<String, f32>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Bike {
    pub kmh: f32,
    pub track_kmh: f32,
    pub path_kmh: f32,
    pub ride_highways: Vec<String>,
    pub signal_s: f32,
    pub crossing_s: f32,
    pub push: bool,
    pub push_kmh: f32,
    pub push_steps_kmh: f32,
    pub surface_cap: BTreeMap<String, f32>,
}

impl ProfileConfig for ActiveFile {
    const FAMILY: &'static str = "active";
    const VERSION: u32 = 1;

    fn header(&self) -> &Header {
        &self.profile
    }

    fn validate(&self) -> Result<()> {
        ensure!(self.profile.modes == ["bike", "walk"], "profile.modes must be [\"bike\", \"walk\"] (the packed pair's order)");
        ensure!(self.walk.kmh > 0.0 && self.bike.kmh > 0.0, "walk.kmh and bike.kmh must be positive");
        ensure!(self.walk.surface_factor.values().all(|&f| f > 0.0 && f <= 1.0), "walk.surface_factor values must be in (0, 1]");
        ensure!(self.snap.radius_m > 0, "snap.radius_m must be positive");
        ensure!(self.snap.larger_within_m >= 0.0, "snap.larger_within_m must not be negative");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn root() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../config")
    }

    #[test]
    fn car_profile_loads_with_the_research_values() {
        let p = load::<CarFile>(&root().join("car.toml")).unwrap();
        let s = &p.config.speeds;
        assert_eq!(s.maxspeed_factor, 0.8);
        assert_eq!(s.highway["motorway"], 90.0);
        assert_eq!(s.highway.len(), 14);
        assert_eq!(s.maxspeed_table.len(), 44);
        // The research code computes the British values in f32: mph * 1609 / 1000.
        assert_eq!(s.maxspeed_table["gb:nsl_single"], 60.0f32 * 1609.0 / 1000.0);
        assert_eq!(s.maxspeed_table["gb:nsl_dual"], 70.0f32 * 1609.0 / 1000.0);
        assert_eq!(s.maxspeed_table["uk:motorway"], 70.0f32 * 1609.0 / 1000.0);
        assert_eq!(s.caps["surface"].len() + s.caps["tracktype"].len() + s.caps["smoothness"].len(), 34);
        assert_eq!((p.config.turns.turn_penalty_s, p.config.turns.turn_bias, p.config.turns.u_turn_penalty_s), (7.5, 1.075, 20.0));
        assert_eq!(p.sha256.len(), 64);
    }

    #[test]
    fn active_profile_loads_with_the_research_values() {
        let p = load::<ActiveFile>(&root().join("active.toml")).unwrap();
        let c = &p.config;
        assert_eq!((c.bike.kmh, c.bike.track_kmh, c.bike.path_kmh, c.walk.kmh), (15.0, 12.0, 13.0, 5.0));
        assert_eq!((c.bike.push_kmh, c.bike.push_steps_kmh, c.bike.signal_s, c.walk.signal_s), (4.0, 2.0, 2.0, 2.0));
        assert_eq!(c.bike.surface_cap.len(), 19);
        assert_eq!(c.walk.default_highways.len(), 18);
        assert_eq!(c.snap.radius_m, 500);
    }

    #[test]
    fn snapping_eligibility() {
        let car = load::<CarFile>(&root().join("car.toml")).unwrap().config.snap;
        assert_eq!(car.radius_m, 2000);
        let ok = |t: &[(&str, &str)]| car.eligible(t.iter().copied());
        assert!(ok(&[("highway", "residential")]) && ok(&[("highway", "service")]) && ok(&[("highway", "primary"), ("bridge", "no")]));
        assert!(!ok(&[("highway", "trunk_link")]) && !ok(&[("highway", "motorway")]) && !ok(&[("highway", "trunk")]));
        assert!(!ok(&[("highway", "residential"), ("bridge", "yes")]) && !ok(&[("highway", "residential"), ("tunnel", "culvert")]));
        assert!(!ok(&[("highway", "footway"), ("covered", "yes")]));
    }

    #[test]
    fn a_wrong_family_or_version_is_refused() {
        let text = std::fs::read_to_string(root().join("car.toml")).unwrap();
        assert!(parse::<ActiveFile>(&text, "car.toml").is_err());
        let v2 = text.replace("version = 1", "version = 2");
        assert!(parse::<CarFile>(&v2, "v2").unwrap_err().to_string().contains("version"));
        let unknown = format!("{text}\n[extra]\nx = 1\n");
        assert!(parse::<CarFile>(&unknown, "extra").is_err());
    }
}
