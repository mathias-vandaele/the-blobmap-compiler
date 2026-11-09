//! ACTIVE component profile (modes phase, Step 4; OSRM's values since the France ACTIVE phase,
//! Step 2): bike and walk on one node-based graph, the union of both networks, with one
//! metric per mode (∞ where a mode may not go). Speeds are flat (no elevation in this phase).
//!
//! The speeds and node costs are an `ActiveConfig`, and the tag lists an `ActiveProfile`,
//! both read from the profile file (`config/active.toml`, `ActiveFile`), whose values are
//! OSRM v6.0.0's bicycle.lua and foot.lua. Another config (an e-bike, a slow walker) is
//! another metric on the same kind of graph. The lists below are that file's.
//!
//! Rules, per OSM way:
//! - **Never** (both modes): `highway=proposed|construction|abandoned|motorway|motorway_link`,
//!   `impassable=yes`, `status=impassable`, a `proposed=*` tag, ferries (out of scope), and
//!   any highway not listed below. Area outlines (`area=yes`) are kept, as OSRM does.
//! - **Walk** (foot.lua): access `foot`, then `access`; `no`, `private`, `agricultural`,
//!   `forestry`, `delivery` close the way. Used by default: `primary`, `secondary`,
//!   `tertiary` (and their `_link`), `unclassified`, `residential`, `road`,
//!   `living_street`, `service`, `track`, `path`, `steps`, `pedestrian`, `footway`,
//!   `platform`, `pier`; any other highway (`trunk`, `cycleway`, `bridleway`, ...) when an
//!   access tag opens it (`yes`, `designated`, `permissive`). Speed `walk_kmh` everywhere,
//!   steps included; surface `fine_gravel|gravel|pebblestone` x0.75, `mud|sand` x0.5. Both
//!   directions, except `oneway:foot`.
//! - **Bike** (bicycle.lua): access `bicycle`, then `vehicle`, then `access`; `no`,
//!   `private`, `agricultural`, `forestry`, `delivery`, `use_sidepath` close the way (no
//!   pushing either). Riding speed: `bike_kmh` on `cycleway`, `primary`, `secondary`,
//!   `tertiary` (and `_link`), `residential`, `unclassified`, `living_street`, `road`,
//!   `service`; `track` 12, `path` 13 (`bike_track_kmh`, `bike_path_kmh`); any other way
//!   opened by an access tag at `bike_kmh`. One-way: `oneway:bicycle`, else `oneway`, else
//!   implied by `junction=roundabout|circular`; on a one-way road, an `opposite*` cycleway
//!   (`cycleway`, `cycleway:left|right`) rides the reverse direction too.
//! - **Pushing** (bicycle.lua `bike_push_handler`, `push`): a direction the bike may not
//!   ride is pushed at `push_kmh` (OSRM's `walking_speed`, 4) unless `foot=no`: `footway`
//!   and `pedestrian` (`steps` at `push_steps_kmh`, 2), `man_made=pier`, ways with
//!   `foot=yes` (`foot:forward|backward=yes` in that direction), and the reverse of a
//!   one-way road the bike rides (not of an implied one-way). `bicycle=dismount` pushes
//!   both directions.
//! - Bike speeds are capped by `maxspeed` (km/h) and by bicycle.lua's `surface_speeds`
//!   (e.g. `paving_stones` 10, `cobblestone` 7, `gravel` 6, `sand` 3).
//! - **Node costs**, charged to the arc arriving at the node (`extract::Network::pieces`):
//!   `highway=traffic_signals` costs `bike_signal_s` / `walk_signal_s` (OSRM's
//!   `traffic_signal_penalty`, 2 s; for the bike only in its `traffic_signals:direction`,
//!   as lib/traffic_signal.lua); `highway=crossing` costs `bike_crossing_s` /
//!   `walk_crossing_s`, 0 by default: neither OSRM profile charges crossings.
//! - Not modelled: turn costs (node-based graph: bicycle.lua's angle-squared turn
//!   penalty and U-turn penalty, foot.lua's 2 s U-turn), barriers, elevation, ferries,
//!   public transport, movable bridges and parking amenities without a highway tag.

use std::collections::BTreeMap;

use crate::ActiveFile;
use crate::profile::{Obstacle, ObstacleDir, ObstacleKind, WayAttrs};
use crate::ways::{NetworkMode, Speeds, WayProfile};

/// Speeds (km/h) and node costs (s) of the ACTIVE modes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ActiveConfig {
    /// bicycle.lua `default_speed`: cycleways, roads, service, living streets.
    pub bike_kmh: f32,
    pub bike_track_kmh: f32,
    pub bike_path_kmh: f32,
    /// Push the bike where it may not ride (bicycle.lua `bike_push_handler`).
    pub push: bool,
    /// bicycle.lua `walking_speed`, and `pedestrian_speeds.steps`.
    pub push_kmh: f32,
    pub push_steps_kmh: f32,
    /// foot.lua `walking_speed`.
    pub walk_kmh: f32,
    /// `traffic_signal_penalty` of bicycle.lua / foot.lua.
    pub bike_signal_s: f32,
    pub walk_signal_s: f32,
    /// `highway=crossing` (no OSRM profile charges it).
    pub bike_crossing_s: f32,
    pub walk_crossing_s: f32,
}

impl Default for ActiveConfig {
    /// OSRM v6.0.0 bicycle.lua and foot.lua.
    fn default() -> Self {
        ActiveConfig {
            bike_kmh: 15.0,
            bike_track_kmh: 12.0,
            bike_path_kmh: 13.0,
            push: true,
            push_kmh: 4.0,
            push_steps_kmh: 2.0,
            walk_kmh: 5.0,
            bike_signal_s: 2.0,
            walk_signal_s: 2.0,
            bike_crossing_s: 0.0,
            walk_crossing_s: 0.0,
        }
    }
}

impl ActiveConfig {
    pub fn new(f: &ActiveFile) -> ActiveConfig {
        ActiveConfig {
            bike_kmh: f.bike.kmh,
            bike_track_kmh: f.bike.track_kmh,
            bike_path_kmh: f.bike.path_kmh,
            push: f.bike.push,
            push_kmh: f.bike.push_kmh,
            push_steps_kmh: f.bike.push_steps_kmh,
            walk_kmh: f.walk.kmh,
            bike_signal_s: f.bike.signal_s,
            walk_signal_s: f.walk.signal_s,
            bike_crossing_s: f.bike.crossing_s,
            walk_crossing_s: f.walk.crossing_s,
        }
    }

    /// The config as JSON (for reports and the metric files' metadata).
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "bike_kmh": self.bike_kmh, "bike_track_kmh": self.bike_track_kmh, "bike_path_kmh": self.bike_path_kmh,
            "push": self.push, "push_kmh": self.push_kmh, "push_steps_kmh": self.push_steps_kmh, "walk_kmh": self.walk_kmh,
            "bike_signal_s": self.bike_signal_s, "walk_signal_s": self.walk_signal_s,
            "bike_crossing_s": self.bike_crossing_s, "walk_crossing_s": self.walk_crossing_s,
        })
    }
}

/// The two modes of a way: `WayAttrs` per mode (forward / backward allowed, speeds).
#[derive(Clone, Copy, Debug)]
pub struct ActiveAttrs {
    pub bike: WayAttrs,
    pub walk: WayAttrs,
    /// `highway=steps` (for snapping rules and reports).
    pub steps: bool,
}

impl ActiveAttrs {
    pub fn mode(&self, m: Mode) -> &WayAttrs {
        match m {
            Mode::Bike => &self.bike,
            Mode::Walk => &self.walk,
        }
    }
}

/// The modes of the ACTIVE family; their index is the metric's index in the pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Mode {
    Bike = 0,
    Walk = 1,
}

impl Mode {
    pub const ALL: [Mode; 2] = [Mode::Bike, Mode::Walk];

    pub fn name(self) -> &'static str {
        match self {
            Mode::Bike => "bike",
            Mode::Walk => "walk",
        }
    }

    pub fn by_name(s: &str) -> Option<Mode> {
        Mode::ALL.into_iter().find(|m| m.name() == s)
    }
}

/// One ACTIVE mode with its config, as a `NetworkMode` of the ACTIVE network: the mode's
/// attributes of a way, and its node costs.
#[derive(Clone, Copy, Debug)]
pub struct ActiveMode {
    pub mode: Mode,
    pub cfg: ActiveConfig,
}

impl NetworkMode<ActiveAttrs> for ActiveMode {
    fn attrs(&self, a: &ActiveAttrs) -> WayAttrs {
        *a.mode(self.mode)
    }

    fn delay(&self, o: &Obstacle, forward: bool) -> Option<f64> {
        let c = &self.cfg;
        let s = match (o.kind, self.mode) {
            (ObstacleKind::TrafficSignals, Mode::Bike) => match o.dir {
                ObstacleDir::Any => c.bike_signal_s,
                ObstacleDir::Forward if forward => c.bike_signal_s,
                ObstacleDir::Backward if !forward => c.bike_signal_s,
                _ => return None,
            },
            // foot.lua: a signal applies in every direction.
            (ObstacleKind::TrafficSignals, Mode::Walk) => c.walk_signal_s,
            (ObstacleKind::Crossing, Mode::Bike) => c.bike_crossing_s,
            (ObstacleKind::Crossing, Mode::Walk) => c.walk_crossing_s,
            _ => return None,
        };
        (s > 0.0).then_some(s as f64)
    }
}

const NONE: WayAttrs = WayAttrs { speed_fwd_kmh: 0.0, speed_bwd_kmh: 0.0, forward: false, backward: false };

/// `maxspeed` in km/h when it is a plain number (lib/maxspeed.lua's numeric case).
fn maxspeed_kmh(v: Option<&str>) -> Option<f32> {
    v?.trim().parse::<f32>().ok().filter(|&x| x > 0.0)
}

/// The first access tag present (most specific first), if any.
fn first<'a>(tags: &[Option<&'a str>]) -> Option<&'a str> {
    tags.iter().flatten().next().copied()
}

/// The repository's ACTIVE profile file, built in (tests and defaults).
pub const ACTIVE_TOML: &str = include_str!("../../../config/active.toml");

/// The ACTIVE network's ways and node obstacles under a config.
#[derive(Clone, Debug)]
pub struct ActiveProfile {
    pub cfg: ActiveConfig,
    never_highways: Vec<String>,
    walk_closed: Vec<String>,
    bike_closed: Vec<String>,
    open: Vec<String>,
    walk_default_highways: Vec<String>,
    bike_ride_highways: Vec<String>,
    /// foot.lua surface factors (1 for a surface not listed).
    walk_surface_factor: BTreeMap<String, f32>,
    /// bicycle.lua `surface_speeds` below its `default_speed`, used as caps.
    bike_surface_cap: BTreeMap<String, f32>,
}

impl Default for ActiveProfile {
    fn default() -> Self {
        ActiveProfile::osrm_v6()
    }
}

fn has(list: &[String], v: &str) -> bool {
    list.iter().any(|x| x == v)
}

impl ActiveProfile {
    pub fn new(f: &ActiveFile) -> ActiveProfile {
        ActiveProfile {
            cfg: ActiveConfig::new(f),
            never_highways: f.never.highway.clone(),
            walk_closed: f.access.walk_closed.clone(),
            bike_closed: f.access.bike_closed.clone(),
            open: f.access.open.clone(),
            walk_default_highways: f.walk.default_highways.clone(),
            bike_ride_highways: f.bike.ride_highways.clone(),
            walk_surface_factor: f.walk.surface_factor.clone(),
            bike_surface_cap: f.bike.surface_cap.clone(),
        }
    }

    /// The built-in `config/active.toml`.
    pub fn osrm_v6() -> ActiveProfile {
        ActiveProfile::new(&crate::parse::<ActiveFile>(ACTIVE_TOML, "config/active.toml").expect("built-in active profile").config)
    }

    pub fn way_attrs<'a>(&self, tags: impl Iterator<Item = (&'a str, &'a str)>) -> Option<ActiveAttrs> {
        let cfg = &self.cfg;
        let tags: Vec<(&str, &str)> = tags.collect();
        let get = |k: &str| tags.iter().find(|(tk, _)| *tk == k).map(|&(_, v)| v);
        if get("impassable") == Some("yes")
            || get("status") == Some("impassable")
            || get("route") == Some("ferry")
            || get("proposed").is_some()
        {
            return None;
        }
        let highway = get("highway")?;
        if has(&self.never_highways, highway) {
            return None;
        }
        let surface = get("surface").unwrap_or("");
        let walk = self.walk(cfg, &get, highway, surface);
        let bike = self.bike(cfg, &get, highway, surface);
        let any = |a: &WayAttrs| a.forward || a.backward;
        (any(&walk) || any(&bike)).then_some(ActiveAttrs { bike, walk, steps: highway == "steps" })
    }

    fn walk<'a>(&self, cfg: &ActiveConfig, get: &impl Fn(&str) -> Option<&'a str>, highway: &str, surface: &str) -> WayAttrs {
        let by_default = has(&self.walk_default_highways, highway);
        let ok = match first(&[get("foot"), get("access")]) {
            Some(v) if has(&self.walk_closed, v) => false,
            Some(v) if has(&self.open, v) => true,
            _ => by_default,
        };
        if !ok {
            return NONE;
        }
        let kmh = cfg.walk_kmh * self.walk_surface_factor.get(surface).copied().unwrap_or(1.0);
        let (f, b) = match get("oneway:foot") {
            Some("yes" | "1" | "true") => (true, false),
            Some("-1") => (false, true),
            _ => (true, true),
        };
        WayAttrs { speed_fwd_kmh: kmh, speed_bwd_kmh: kmh, forward: f, backward: b }
    }

    fn bike<'a>(&self, cfg: &ActiveConfig, get: &impl Fn(&str) -> Option<&'a str>, highway: &str, surface: &str) -> WayAttrs {
        let access = first(&[get("bicycle"), get("vehicle"), get("access")]);
        if access.is_some_and(|v| has(&self.bike_closed, v)) {
            return NONE;
        }
        // Riding speed by way type (bicycle.lua `speed_handler`).
        let ride = match highway {
            _ if has(&self.bike_ride_highways, highway) => Some(cfg.bike_kmh),
            "track" => Some(cfg.bike_track_kmh),
            "path" => Some(cfg.bike_path_kmh),
            _ if access.is_some_and(|v| has(&self.open, v)) => Some(cfg.bike_kmh),
            _ => None,
        };
        let allows_pushing = ride.is_some();
        // One-way (bicycle.lua `oneway_handler`) and opposite cycleways (`cycleway_handler`).
        let implied = matches!(get("junction"), Some("roundabout" | "circular"));
        let (mut f, mut b) = match (get("oneway:bicycle"), get("oneway")) {
            (Some("yes" | "1" | "true"), _) => (true, false),
            (Some("no" | "0" | "false"), _) => (true, true),
            (Some("-1"), _) => (false, true),
            (_, Some("yes" | "1" | "true")) => (true, false),
            (_, Some("no" | "0" | "false")) => (true, true),
            (_, Some("-1")) => (false, true),
            _ if implied => (true, false),
            _ => (true, true),
        };
        let opposite = |k: &str| get(k).is_some_and(|v| matches!(v, "opposite" | "opposite_lane" | "opposite_track"));
        if (f != b) && (opposite("cycleway") || opposite("cycleway:left") || opposite("cycleway:right")) {
            (f, b) = (true, true);
        }
        let (mut sf, mut sb) = match ride {
            Some(k) => (if f { k } else { 0.0 }, if b { k } else { 0.0 }),
            None => (0.0, 0.0),
        };
        // Pushing where the bike may not ride (bicycle.lua `bike_push_handler`).
        if cfg.push && (sf == 0.0 || sb == 0.0) && get("foot") != Some("no") {
            let p = Some(cfg.push_kmh);
            let (pf, pb) = match highway {
                "footway" | "pedestrian" => (p, p),
                "steps" => (Some(cfg.push_steps_kmh), Some(cfg.push_steps_kmh)),
                _ if get("man_made") == Some("pier") => (p, p),
                _ if get("foot") == Some("yes") => (p, p.filter(|_| !implied)),
                _ if get("foot:forward") == Some("yes") => (p, None),
                _ if get("foot:backward") == Some("yes") => (None, p),
                _ if allows_pushing => (p, p.filter(|_| !implied)),
                _ => (None, None),
            };
            if sf == 0.0 {
                sf = pf.unwrap_or(0.0);
            }
            if sb == 0.0 {
                sb = pb.unwrap_or(0.0);
            }
        }
        if cfg.push && get("bicycle") == Some("dismount") {
            (sf, sb) = (cfg.push_kmh, cfg.push_kmh);
        }
        // Caps: maxspeed, then the surface (WayHandlers.surface runs after the bicycle tags).
        let cap = |x: f32| {
            let x = maxspeed_kmh(get("maxspeed")).map_or(x, |m| x.min(m));
            self.bike_surface_cap.get(surface).map_or(x, |&c| x.min(c))
        };
        let (sf, sb) = (cap(sf), cap(sb));
        WayAttrs { speed_fwd_kmh: sf, speed_bwd_kmh: sb, forward: sf > 0.0, backward: sb > 0.0 }
    }

    /// Node obstacles: traffic signals (with their `traffic_signals:direction`) and crossings.
    pub fn obstacle<'a>(&self, tags: impl Iterator<Item = (&'a str, &'a str)>) -> Option<Obstacle> {
        let (mut highway, mut dir) = (None, None);
        for (k, v) in tags {
            match k {
                "highway" => highway = Some(v),
                "traffic_signals:direction" => dir = Some(v),
                _ => {}
            }
        }
        let kind = match highway? {
            "traffic_signals" => ObstacleKind::TrafficSignals,
            "crossing" => ObstacleKind::Crossing,
            _ => return None,
        };
        let dir = match (kind, dir) {
            (ObstacleKind::TrafficSignals, Some("forward")) => ObstacleDir::Forward,
            (ObstacleKind::TrafficSignals, Some("backward")) => ObstacleDir::Backward,
            _ => ObstacleDir::Any,
        };
        // The duration is the mode's (`ActiveMode::delay`).
        Some(Obstacle { kind, dir, duration_s: 0.0 })
    }
}

impl Speeds for ActiveAttrs {
    fn speeds(&self) -> (f32, f32) {
        (self.bike.speed_fwd_kmh.max(self.walk.speed_fwd_kmh), self.bike.speed_bwd_kmh.max(self.walk.speed_bwd_kmh))
    }
}

impl WayProfile for ActiveProfile {
    type Attrs = ActiveAttrs;
    fn way_attrs<'a>(&self, tags: impl Iterator<Item = (&'a str, &'a str)>) -> Option<ActiveAttrs> {
        ActiveProfile::way_attrs(self, tags)
    }
    fn obstacle<'a>(&self, tags: impl Iterator<Item = (&'a str, &'a str)>) -> Option<Obstacle> {
        ActiveProfile::obstacle(self, tags)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn attrs(tags: &[(&str, &str)]) -> Option<ActiveAttrs> {
        ActiveProfile::default().way_attrs(tags.iter().copied())
    }

    #[test]
    fn the_file_config_is_the_research_default() {
        assert_eq!(ActiveProfile::osrm_v6().cfg, ActiveConfig::default());
    }

    #[test]
    fn rules() {
        let c = ActiveConfig::default();
        // A residential street: both modes, both directions.
        let a = attrs(&[("highway", "residential")]).unwrap();
        assert!(a.walk.forward && a.walk.backward && a.bike.forward && a.bike.backward);
        assert_eq!((a.walk.speed_fwd_kmh, a.bike.speed_fwd_kmh), (5.0, 15.0));
        // A one-way street: walk both ways; the bike rides one way and pushes the other,
        // unless a contraflow lane opens it.
        let a = attrs(&[("highway", "residential"), ("oneway", "yes")]).unwrap();
        assert!(a.walk.backward && a.bike.forward && a.bike.backward);
        assert_eq!((a.bike.speed_fwd_kmh, a.bike.speed_bwd_kmh), (15.0, c.push_kmh));
        let a = attrs(&[("highway", "residential"), ("oneway", "yes"), ("cycleway:left", "opposite_lane")]).unwrap();
        assert_eq!(a.bike.speed_bwd_kmh, 15.0);
        let a = attrs(&[("highway", "residential"), ("oneway", "yes"), ("oneway:bicycle", "no")]).unwrap();
        assert_eq!(a.bike.speed_bwd_kmh, 15.0);
        // A roundabout: no pushing against it.
        let a = attrs(&[("highway", "residential"), ("junction", "roundabout")]).unwrap();
        assert!(a.bike.forward && !a.bike.backward);
        // A footway: walk; the bike pushes, unless bicycles are allowed.
        let a = attrs(&[("highway", "footway")]).unwrap();
        assert!(a.walk.forward && a.bike.forward && a.bike.speed_fwd_kmh == c.push_kmh);
        assert_eq!(attrs(&[("highway", "footway"), ("bicycle", "yes")]).unwrap().bike.speed_fwd_kmh, 15.0);
        assert!(!attrs(&[("highway", "footway"), ("bicycle", "no")]).unwrap().bike.forward);
        // A cycleway: bike only, unless foot is allowed.
        let a = attrs(&[("highway", "cycleway")]).unwrap();
        assert!(!a.walk.forward && a.bike.forward);
        assert!(attrs(&[("highway", "cycleway"), ("foot", "designated")]).unwrap().walk.forward);
        // Steps: walk at full speed (foot.lua), the bike pushed at 2 km/h.
        let a = attrs(&[("highway", "steps")]).unwrap();
        assert!(a.steps && a.walk.speed_fwd_kmh == 5.0 && a.bike.speed_fwd_kmh == c.push_steps_kmh);
        // Without pushing, the bike cannot use steps or footways (the Step 4 rule).
        let no_push = ActiveConfig { push: false, ..c };
        assert!(
            !ActiveProfile { cfg: no_push, ..ActiveProfile::default() }.way_attrs([("highway", "steps")].into_iter()).unwrap().bike.forward
        );
        // Motorways and closed ways: nothing; trunks only for those an access tag opens.
        assert!(attrs(&[("highway", "motorway")]).is_none());
        assert!(attrs(&[("highway", "trunk")]).is_none());
        assert!(attrs(&[("highway", "trunk"), ("foot", "yes")]).unwrap().walk.forward);
        assert!(attrs(&[("highway", "service"), ("access", "private")]).is_none());
        assert!(attrs(&[("highway", "service"), ("access", "private"), ("foot", "yes")]).unwrap().walk.forward);
        // Surfaces and maxspeed.
        assert_eq!(attrs(&[("highway", "track"), ("surface", "gravel")]).unwrap().bike.speed_fwd_kmh, 6.0);
        assert_eq!(attrs(&[("highway", "track"), ("surface", "gravel")]).unwrap().walk.speed_fwd_kmh, 5.0 * 0.75);
        assert_eq!(attrs(&[("highway", "living_street"), ("maxspeed", "10")]).unwrap().bike.speed_fwd_kmh, 10.0);
    }

    #[test]
    fn node_costs() {
        let tags = |t: &[(&'static str, &'static str)]| ActiveProfile::default().obstacle(t.iter().copied());
        let bike = ActiveMode { mode: Mode::Bike, cfg: ActiveConfig::default() };
        let walk = ActiveMode { mode: Mode::Walk, cfg: ActiveConfig::default() };
        let s = tags(&[("highway", "traffic_signals"), ("traffic_signals:direction", "forward")]).unwrap();
        assert_eq!((bike.delay(&s, true), bike.delay(&s, false)), (Some(2.0), None));
        assert_eq!((walk.delay(&s, true), walk.delay(&s, false)), (Some(2.0), Some(2.0)));
        // Crossings: 0 s by default (no OSRM profile charges them), configurable.
        let x = tags(&[("highway", "crossing")]).unwrap();
        assert_eq!(walk.delay(&x, true), None);
        let w = ActiveMode { mode: Mode::Walk, cfg: ActiveConfig { walk_crossing_s: 5.0, ..ActiveConfig::default() } };
        assert_eq!(w.delay(&x, false), Some(5.0));
        assert!(tags(&[("highway", "stop")]).is_none());
    }
}
