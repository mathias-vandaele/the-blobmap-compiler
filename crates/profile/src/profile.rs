//! Car profile modelled on OSRM's `car.lua` (v6.0.0), with every number and tag list from
//! the profile file (`config/car.toml`, `CarFile`): default speeds by highway class,
//! maxspeed (scaled by `maxspeed_factor`, 0.8 like OSRM), the access blacklist, one-way
//! handling, surface / tracktype / smoothness caps, `bridge=movable`, and OSRM's obstacle
//! delays (traffic signals, stops, give-ways). The rules' order is car.lua's (research
//! `profile.rs`, `SpeedRules::Osrm`).
//!
//! - maxspeed: the first of `maxspeed:advisory` / `maxspeed` / `source:maxspeed` /
//!   `maxspeed:type` (with :forward/:backward variants), parsed like `parse_maxspeed`:
//!   leading digits (mph scaled), else `maxspeed_table`, else the `xx:<class>` zone defaults.
//! - Not modelled here (other stages, or not at all): turn costs (the `edge` crate),
//!   barriers, ferries, routability weights, the 10 km/h speed of otherwise-untyped ways
//!   with explicit car access.

use std::collections::BTreeMap;

use crate::{CAR_CAP_KEYS, CarFile};

#[derive(Clone, Copy, Debug)]
pub struct WayAttrs {
    pub speed_fwd_kmh: f32,
    pub speed_bwd_kmh: f32,
    pub forward: bool,
    pub backward: bool,
}

/// OSRM v6 obstacle kinds with a delay (lib/obstacles.lua; barriers are not delays).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ObstacleKind {
    TrafficSignals,
    Stop,
    /// `stop=minor`: only for traffic entering by a slower road than the junction's fastest other road.
    StopMinor,
    GiveWay,
    /// `highway=crossing` (the ACTIVE modes; no OSRM profile charges it).
    Crossing,
}

/// Direction relative to the way's node order; `Any` = both ways (no or unknown tag).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ObstacleDir {
    Any,
    Forward,
    Backward,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Obstacle {
    pub kind: ObstacleKind,
    pub dir: ObstacleDir,
    pub duration_s: f32,
}

/// The car profile, ready for lookups.
#[derive(Clone, Debug)]
pub struct CarProfile {
    pub maxspeed_factor: f32,
    maxspeed_keys: Vec<String>,
    movable_bridge_kmh: f32,
    highway: BTreeMap<String, f32>,
    zone_defaults: BTreeMap<String, f32>,
    maxspeed_table: BTreeMap<String, f32>,
    /// Caps in `CAR_CAP_KEYS` order.
    caps: Vec<(String, BTreeMap<String, f32>)>,
    exclude: Vec<(String, String)>,
    implied_oneway_highways: Vec<String>,
    implied_oneway_junctions: Vec<String>,
    oneway_forward: Vec<String>,
    oneway_backward: Vec<String>,
    oneway_none: Vec<String>,
    oneway_drop: Vec<String>,
    access_keys: Vec<String>,
    access_blacklist: Vec<String>,
    traffic_signals_s: f32,
    stop_s: f32,
    stop_minor_s: f32,
    give_way_s: f32,
}

/// The repository's car profile file, built in (tests and defaults).
pub const CAR_TOML: &str = include_str!("../../../config/car.toml");

impl Default for CarProfile {
    fn default() -> Self {
        CarProfile::osrm_v6()
    }
}

impl CarProfile {
    pub fn new(f: &CarFile) -> CarProfile {
        let s = &f.speeds;
        CarProfile {
            maxspeed_factor: s.maxspeed_factor,
            maxspeed_keys: s.maxspeed_keys.clone(),
            movable_bridge_kmh: s.movable_bridge_kmh,
            highway: s.highway.clone(),
            zone_defaults: s.zone_defaults.clone(),
            maxspeed_table: s.maxspeed_table.clone(),
            caps: CAR_CAP_KEYS.iter().map(|k| (k.to_string(), s.caps.get(*k).cloned().unwrap_or_default())).collect(),
            exclude: f.ways.exclude.clone(),
            implied_oneway_highways: f.ways.implied_oneway_highways.clone(),
            implied_oneway_junctions: f.ways.implied_oneway_junctions.clone(),
            oneway_forward: f.ways.oneway_forward.clone(),
            oneway_backward: f.ways.oneway_backward.clone(),
            oneway_none: f.ways.oneway_none.clone(),
            oneway_drop: f.ways.oneway_drop.clone(),
            access_keys: f.access.keys.clone(),
            access_blacklist: f.access.blacklist.clone(),
            traffic_signals_s: f.obstacles.traffic_signals_s,
            stop_s: f.obstacles.stop_s,
            stop_minor_s: f.obstacles.stop_minor_s,
            give_way_s: f.obstacles.give_way_s,
        }
    }

    /// The built-in `config/car.toml`.
    pub fn osrm_v6() -> CarProfile {
        CarProfile::new(&crate::parse::<CarFile>(CAR_TOML, "config/car.toml").expect("built-in car profile").config)
    }

    /// The obstacle at a node with these tags, as OSRM v6 `Obstacles.process_node`:
    /// traffic_signals (direction from `traffic_signals:direction`, else `direction`),
    /// stop (`stop=minor` -> StopMinor), give_way.
    pub fn obstacle<'a>(&self, tags: impl Iterator<Item = (&'a str, &'a str)>) -> Option<Obstacle> {
        let (mut highway, mut direction, mut ts_direction, mut stop) = (None, None, None, None);
        for (k, v) in tags {
            match k {
                "highway" => highway = Some(v),
                "direction" => direction = Some(v),
                "traffic_signals:direction" => ts_direction = Some(v),
                "stop" => stop = Some(v),
                _ => {}
            }
        }
        let (kind, duration_s, dir) = match highway? {
            "traffic_signals" => (ObstacleKind::TrafficSignals, self.traffic_signals_s, ts_direction.or(direction)),
            "stop" if stop == Some("minor") => (ObstacleKind::StopMinor, self.stop_minor_s, direction),
            "stop" => (ObstacleKind::Stop, self.stop_s, direction),
            "give_way" => (ObstacleKind::GiveWay, self.give_way_s, direction),
            _ => return None,
        };
        let dir = match dir {
            Some("forward") => ObstacleDir::Forward,
            Some("backward") => ObstacleDir::Backward,
            _ => ObstacleDir::Any,
        };
        Some(Obstacle { kind, dir, duration_s })
    }

    /// car.lua v6.0.0 `WayHandlers.parse_maxspeed`; 0 = no maxspeed.
    fn maxspeed(&self, v: &str) -> f32 {
        // Measure.parse_value_speed: tonumber(source:match("%d*")), mph scaled.
        let digits: String = v.chars().take_while(|c| c.is_ascii_digit()).collect();
        if let Ok(n) = digits.parse::<f32>() {
            return if v.contains("mph") || v.contains("mp/h") { n * 1.609 } else { n };
        }
        let lower = v.to_lowercase();
        if let Some(&n) = self.maxspeed_table.get(&lower) {
            return n;
        }
        // string.match(source, "%a%a:(%a+)"): two letters, a colon, then letters.
        let b = lower.as_bytes();
        let class = (0..b.len().saturating_sub(2))
            .find(|&i| b[i].is_ascii_alphabetic() && b[i + 1].is_ascii_alphabetic() && b[i + 2] == b':')
            .map(|i| lower[i + 3..].chars().take_while(|c| c.is_ascii_alphabetic()).collect::<String>());
        class.and_then(|c| self.zone_defaults.get(&c).copied()).unwrap_or(0.0)
    }

    /// car.lua v6.0.0 speed of one way, (forward, backward) km/h. `tag` looks a key up.
    fn speed<'a>(&self, highway_speed: f32, tag: impl Fn(&str) -> Option<&'a str>) -> (f32, f32) {
        let (mut fwd, mut bwd) = (highway_speed, highway_speed);
        // WayHandlers.movables (before speed, which then keeps it).
        if tag("bridge") == Some("movable") && tag("capacity:car") != Some("0") {
            (fwd, bwd) = (self.movable_bridge_kmh, self.movable_bridge_kmh);
        }
        // WayHandlers.maxspeed via Tags.get_forward_backward_by_set.
        let (mut f, mut b): (Option<&str>, Option<&str>) = (None, None);
        for key in &self.maxspeed_keys {
            if f.is_none() {
                f = tag(&format!("{key}:forward"));
            }
            if b.is_none() {
                b = tag(&format!("{key}:backward"));
            }
            if f.is_none() || b.is_none() {
                let common = tag(key);
                f = f.or(common);
                b = b.or(common);
            }
            if f.is_some() && b.is_some() {
                break;
            }
        }
        let (mf, mb) = (f.map_or(0.0, |v| self.maxspeed(v)), b.map_or(0.0, |v| self.maxspeed(v)));
        if mf > 0.0 {
            fwd = mf * self.maxspeed_factor;
        }
        if mb > 0.0 {
            bwd = mb * self.maxspeed_factor;
        }
        // WayHandlers.surface.
        for (key, caps) in &self.caps {
            if let Some(&cap) = tag(key).and_then(|v| caps.get(v)) {
                fwd = fwd.min(cap);
                bwd = bwd.min(cap);
            }
        }
        (fwd, bwd)
    }

    pub fn way_attrs<'a>(&self, tags: impl Iterator<Item = (&'a str, &'a str)>) -> Option<WayAttrs> {
        let tags: Vec<(&'a str, &'a str)> = tags.collect();
        if tags.iter().any(|&(k, v)| self.exclude.iter().any(|(ek, ev)| ek == k && ev == v)) {
            return None;
        }
        let get = |k: &str| tags.iter().find(|(tk, _)| *tk == k).map(|&(_, v)| v);
        let highway = get("highway")?;
        let base = *self.highway.get(highway)?;
        // Most specific access tag wins.
        if let Some(a) = self.access_keys.iter().find_map(|k| get(k))
            && self.access_blacklist.iter().any(|b| b == a)
        {
            return None;
        }
        let (speed_fwd, speed_bwd) = self.speed(base, get);
        let in_list = |v: Option<&str>, l: &[String]| v.is_some_and(|v| l.iter().any(|x| x == v));
        let implied = self.implied_oneway_highways.iter().any(|h| h == highway) || in_list(get("junction"), &self.implied_oneway_junctions);
        let oneway = get("oneway");
        let (forward, backward) = if in_list(oneway, &self.oneway_forward) {
            (true, false)
        } else if in_list(oneway, &self.oneway_backward) {
            (false, true)
        } else if in_list(oneway, &self.oneway_drop) {
            return None;
        } else if in_list(oneway, &self.oneway_none) {
            (true, true)
        } else if implied {
            (true, false)
        } else {
            (true, true)
        };
        // A zero speed (smoothness=impassable) makes that direction unusable, as in OSRM.
        let (forward, backward) = (forward && speed_fwd > 0.0, backward && speed_bwd > 0.0);
        (forward || backward).then_some(WayAttrs { speed_fwd_kmh: speed_fwd, speed_bwd_kmh: speed_bwd, forward, backward })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The research `profile.rs` tables (commit 9f8c060), as it hard-codes them.
    const RESEARCH_HIGHWAY: &[(&str, f32)] = &[
        ("motorway", 90.0),
        ("motorway_link", 45.0),
        ("trunk", 85.0),
        ("trunk_link", 40.0),
        ("primary", 65.0),
        ("primary_link", 30.0),
        ("secondary", 55.0),
        ("secondary_link", 25.0),
        ("tertiary", 40.0),
        ("tertiary_link", 20.0),
        ("unclassified", 25.0),
        ("residential", 25.0),
        ("living_street", 10.0),
        ("service", 15.0),
    ];
    const RESEARCH_MAXSPEED_TABLE: &[(&str, f32)] = &[
        ("at:rural", 100.0),
        ("at:trunk", 100.0),
        ("be:motorway", 120.0),
        ("be-bru:rural", 70.0),
        ("be-bru:urban", 30.0),
        ("be-vlg:rural", 70.0),
        ("bg:motorway", 140.0),
        ("by:urban", 60.0),
        ("by:motorway", 110.0),
        ("ca-on:rural", 80.0),
        ("ch:rural", 80.0),
        ("ch:trunk", 100.0),
        ("ch:motorway", 120.0),
        ("cz:trunk", 0.0),
        ("cz:motorway", 0.0),
        ("de:living_street", 7.0),
        ("de:rural", 100.0),
        ("de:motorway", 0.0),
        ("dk:rural", 80.0),
        ("es:trunk", 90.0),
        ("fr:rural", 80.0),
        ("gb:nsl_single", 60.0 * 1609.0 / 1000.0),
        ("gb:nsl_dual", 70.0 * 1609.0 / 1000.0),
        ("gb:motorway", 70.0 * 1609.0 / 1000.0),
        ("nl:rural", 80.0),
        ("nl:trunk", 100.0),
        ("no:rural", 80.0),
        ("no:motorway", 110.0),
        ("ph:urban", 40.0),
        ("ph:rural", 80.0),
        ("ph:motorway", 100.0),
        ("pl:rural", 100.0),
        ("pl:expressway", 120.0),
        ("pl:motorway", 140.0),
        ("ro:trunk", 100.0),
        ("ru:living_street", 20.0),
        ("ru:urban", 60.0),
        ("ru:motorway", 110.0),
        ("uk:nsl_single", 60.0 * 1609.0 / 1000.0),
        ("uk:nsl_dual", 70.0 * 1609.0 / 1000.0),
        ("uk:motorway", 70.0 * 1609.0 / 1000.0),
        ("za:urban", 60.0),
        ("za:rural", 100.0),
        ("none", 140.0),
    ];
    const RESEARCH_CAPS: &[(&str, &[&str], f32)] = &[
        ("surface", &["cement", "compacted", "fine_gravel"], 80.0),
        ("surface", &["paving_stones", "metal", "bricks"], 60.0),
        ("surface", &["grass", "wood", "sett", "grass_paver", "gravel", "unpaved", "ground", "dirt", "pebblestone", "tartan"], 40.0),
        ("surface", &["cobblestone", "clay"], 30.0),
        ("surface", &["earth", "stone", "rocky", "sand"], 20.0),
        ("surface", &["mud"], 10.0),
        ("tracktype", &["grade1"], 60.0),
        ("tracktype", &["grade2"], 40.0),
        ("tracktype", &["grade3"], 30.0),
        ("tracktype", &["grade4"], 25.0),
        ("tracktype", &["grade5"], 20.0),
        ("smoothness", &["intermediate"], 80.0),
        ("smoothness", &["bad"], 40.0),
        ("smoothness", &["very_bad"], 20.0),
        ("smoothness", &["horrible"], 10.0),
        ("smoothness", &["very_horrible"], 5.0),
        ("smoothness", &["impassable"], 0.0),
    ];

    fn strs(v: &[String]) -> Vec<&str> {
        v.iter().map(String::as_str).collect()
    }

    #[test]
    fn the_file_profile_is_the_research_profile() {
        let p = CarProfile::osrm_v6();
        assert_eq!(p.maxspeed_factor, 0.8);
        assert_eq!(p.movable_bridge_kmh, 5.0);
        assert_eq!(strs(&p.maxspeed_keys), ["maxspeed:advisory", "maxspeed", "source:maxspeed", "maxspeed:type"]);
        let highway: BTreeMap<String, f32> = RESEARCH_HIGHWAY.iter().map(|&(k, v)| (k.to_string(), v)).collect();
        assert_eq!(p.highway, highway);
        let table: BTreeMap<String, f32> = RESEARCH_MAXSPEED_TABLE.iter().map(|&(k, v)| (k.to_string(), v)).collect();
        assert_eq!(table.len(), RESEARCH_MAXSPEED_TABLE.len());
        // Bit for bit, including the f32 British values.
        assert!(p.maxspeed_table.len() == table.len() && table.iter().all(|(k, v)| p.maxspeed_table[k].to_bits() == v.to_bits()));
        let zones: BTreeMap<String, f32> =
            [("urban", 50.0), ("rural", 90.0), ("trunk", 110.0), ("motorway", 130.0)].iter().map(|&(k, v)| (k.to_string(), v)).collect();
        assert_eq!(p.zone_defaults, zones);
        let mut caps: Vec<(String, BTreeMap<String, f32>)> = CAR_CAP_KEYS.iter().map(|k| (k.to_string(), BTreeMap::new())).collect();
        for &(key, values, cap) in RESEARCH_CAPS {
            let m = &mut caps.iter_mut().find(|(k, _)| k == key).unwrap().1;
            for v in values {
                assert!(m.insert(v.to_string(), cap).is_none());
            }
        }
        assert_eq!(p.caps, caps);
        let exclude: Vec<(String, String)> = p.exclude.clone();
        assert_eq!(
            exclude,
            [("area", "yes"), ("service", "emergency_access"), ("impassable", "yes")].map(|(k, v)| (k.to_string(), v.to_string()))
        );
        assert_eq!(strs(&p.implied_oneway_highways), ["motorway"]);
        assert_eq!(strs(&p.implied_oneway_junctions), ["roundabout", "circular"]);
        assert_eq!(strs(&p.oneway_forward), ["yes", "1", "true"]);
        assert_eq!(strs(&p.oneway_backward), ["-1"]);
        assert_eq!(strs(&p.oneway_drop), ["reversible", "alternating"]);
        assert_eq!(strs(&p.oneway_none), ["no", "0", "false"]);
        assert_eq!(strs(&p.access_keys), ["motorcar", "motor_vehicle", "vehicle", "access"]);
        assert_eq!(strs(&p.access_blacklist), ["no", "agricultural", "forestry", "emergency", "psv", "customers", "private", "delivery"]);
        assert_eq!((p.traffic_signals_s, p.stop_s, p.stop_minor_s, p.give_way_s), (2.0, 2.0, 2.0, 1.0));
    }
}
