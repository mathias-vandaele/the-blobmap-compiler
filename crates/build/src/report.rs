//! The data report (SPEC §3), written next to the model directory: place counts per country
//! and basic category, compared with `--previous-report`; served counts and reasons per mode;
//! every trapped component; the fragments IDs were moved away from, to a larger component
//! within Δ (SPEC §6); restrictions whose prefix copy has no exit; places without a
//! category.

use std::collections::BTreeMap;

use anyhow::{Context, Result};
use blobmap_overture::Catalog;
use serde_json::{Value, json};

use crate::Args;
use crate::family::Family;
use blobmap_snap::modes::{active_expert, car_expert};

use crate::vocab::{Families, Ids, MODES, ModeSnap, Served};

/// A change above this fraction is flagged (SPEC §3).
pub const FLAG_CHANGE: f64 = 0.2;

pub fn data(a: &Args, f: &Families, c: &Catalog, snaps: &[ModeSnap], ids: &Ids, served: &Served) -> Result<Value> {
    let mut places: BTreeMap<String, BTreeMap<String, u64>> = BTreeMap::new();
    for &p in &ids.places {
        let p = &c.places[p];
        if p.status == 0 {
            let country = if p.country.is_empty() { "none".to_string() } else { p.country.clone() };
            *places.entry(country).or_default().entry(p.basic_key.clone().unwrap_or_else(|| "none".into())).or_default() += 1;
        }
    }
    let mut trapped = serde_json::Map::new();
    for fam in [f.car as &dyn Family, f.active] {
        let e = fam.experts();
        let b = if fam.name() == "car" { f.car_built } else { f.active_built };
        let mut ids_in: BTreeMap<u32, BTreeMap<&str, u64>> = BTreeMap::new();
        for (m, (mode, family)) in MODES.iter().enumerate() {
            if *family != fam.name() {
                continue;
            }
            let kept = ids.addresses.iter().copied().chain(ids.places.iter().map(|&p| c.addresses.len() + p));
            for i in kept {
                let x = snaps[m].at[i].trapped;
                if x != blobmap_format::NONE {
                    *ids_in.entry(x).or_default().entry(mode).or_default() += 1;
                }
            }
        }
        let list: Vec<Value> = (0..e.len() as u32)
            .filter(|&x| e.trapped[x as usize])
            .map(|x| {
                let way = b.experts[x as usize].nodes.iter().map(|&v| fam.road_key(v)).min().map(|k| k.0);
                json!({"expert": x, "nodes": e.nodes[x as usize], "ids": ids_in.get(&x).cloned().unwrap_or_default(), "way": way})
            })
            .collect();
        trapped.insert(fam.name().into(), Value::Array(list));
    }
    // Per mode, the components IDs were moved away from (likely OSM errors), most IDs first,
    // with one OSM way each.
    let mut moved = serde_json::Map::new();
    for (m, (mode, family)) in MODES.iter().enumerate() {
        let car = *family == "car";
        let (p, e) = if car { (f.car.p, &f.car.e) } else { (f.active.p, &f.active.e) };
        let mut by: BTreeMap<u32, (u64, i64)> = BTreeMap::new();
        let kept = ids.addresses.iter().copied().chain(ids.places.iter().map(|&p| c.addresses.len() + p));
        for i in kept {
            let k = snaps[m].at[i].moved_from;
            if k == blobmap_format::NONE {
                continue;
            }
            let x = if car { car_expert(f.car.g, e, k as usize) } else { active_expert(p, e, k as usize) }.context("a moved-from piece is servable")?;
            let b = by.entry(x).or_insert((0, p.way[k as usize]));
            b.0 += 1;
            b.1 = b.1.min(p.way[k as usize]);
        }
        let mut list: Vec<(u32, u64, i64)> = by.into_iter().map(|(x, (n, w))| (x, n, w)).collect();
        list.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let rows = list.into_iter().map(|(x, n, w)| json!({"expert": x, "nodes": e.nodes[x as usize], "ids_moved": n, "way": w}));
        moved.insert(mode.to_string(), Value::Array(rows.collect()));
    }
    let previous = match &a.previous_report {
        None => Value::Null,
        Some(p) => {
            let prev = blobmap_format::json::read_file(p).with_context(|| format!("reading the previous report {}", p.display()))?;
            json!({"file": p.file_name().map(|n| n.to_string_lossy().into_owned()), "flagged": compare(&prev["places"], &places)})
        }
    };
    // Restrictions whose prefix copy has no turn out (SPEC §5): likely mis-tagged.
    let g = f.car.g;
    let dead_ends: Vec<Value> = g
        .dead_ends
        .iter()
        .map(|&(rel, c)| {
            let c = c as usize;
            json!({"relation": rel, "way": g.way[c], "way_seq": g.seq[c], "way_dir": g.dir[c]})
        })
        .collect();
    let s = &c.stats;
    Ok(json!({
        "inputs": {"pbf": a.pbf.file_name().map(|n| n.to_string_lossy().into_owned()), "overture": c.release},
        "ids": {
            "addresses": ids.addresses.len(),
            "places": ids.places.len(),
            "aliases": ids.aliases.len(),
            "outside_every_radius": (c.addresses.len() + c.places.len()) - (ids.addresses.len() + ids.places.len()),
        },
        "catalog": {
            "addresses_read": s.addresses,
            "places_read": s.places,
            "filtered": s.filtered,
            "merged": s.merged,
            "primary_none": s.primary_none,
            "missing_keys": s.missing_keys,
        },
        "places": places,
        "served": served,
        "trapped": trapped,
        "moved_from": moved,
        "dead_end_restrictions": dead_ends,
        "previous": previous,
    }))
}

/// Every (country, basic category) whose count changed by more than `FLAG_CHANGE`.
fn compare(prev: &Value, cur: &BTreeMap<String, BTreeMap<String, u64>>) -> Vec<Value> {
    let mut keys: BTreeMap<(String, String), (u64, u64)> = BTreeMap::new();
    if let Some(p) = prev.as_object() {
        for (country, cats) in p {
            for (cat, n) in cats.as_object().into_iter().flatten() {
                keys.entry((country.clone(), cat.clone())).or_default().0 = n.as_u64().unwrap_or(0);
            }
        }
    }
    for (country, cats) in cur {
        for (cat, &n) in cats {
            keys.entry((country.clone(), cat.clone())).or_default().1 = n;
        }
    }
    keys.into_iter()
        .filter(|(_, (p, n))| if *p == 0 { *n > 0 } else { (*n as f64 - *p as f64).abs() / *p as f64 > FLAG_CHANGE })
        .map(|((country, cat), (p, n))| json!({"country": country, "basic": cat, "previous": p, "current": n}))
        .collect()
}
