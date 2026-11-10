//! The data report (SPEC §3), written next to the model directory: place counts per country
//! and basic category, compared with `--previous-report`; served counts and reasons per mode;
//! every trapped component; the small components IDs were moved away from, the components
//! other than the largest that still serve IDs (SPEC §6), and the real islands emptied, with
//! the island each stands on; restrictions whose prefix copy has no exit; places without a
//! category.

use std::collections::BTreeMap;

use anyhow::{Context, Result};
use serde_json::{Value, json};

use crate::build::Args;
use crate::build::family::Family;
use crate::build::vocab::{Families, Ids, MODES, ModeSnap, Served, membership};
use crate::overture::Catalog;
use crate::snap::islands::Islands;
use crate::snap::modes::{active_component, car_component};

/// A change above this fraction is flagged (SPEC §3).
pub const FLAG_CHANGE: f64 = 0.2;

pub fn data(a: &Args, f: &Families, c: &Catalog, snaps: &[ModeSnap], ids: &Ids, served: &Served, islands: &Islands) -> Result<Value> {
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
        let e = fam.components();
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
                let way = b.components[x as usize].nodes.iter().map(|&v| fam.road_key(v)).min().map(|k| k.0);
                json!({"component": x, "nodes": e.nodes[x as usize], "ids": ids_in.get(&x).cloned().unwrap_or_default(), "way": way})
            })
            .collect();
        trapped.insert(fam.name().into(), Value::Array(list));
    }
    // Per mode (SPEC §6): the small components IDs were moved away from, most IDs first, with
    // the IDs left on each; and every component other than the largest still serving IDs, the
    // real islands and the fragments with no larger component in reach (likely OSM errors);
    // and the real islands the rule emptied. One OSM way each, and the real island it stands
    // on; filtered places are not served and not counted.
    let (mut moved, mut left, mut emptied) = (serde_json::Map::new(), serde_json::Map::new(), serde_json::Map::new());
    for (m, (mode, family)) in MODES.iter().enumerate() {
        let car = *family == "car";
        let (p, e) = if car { (f.car.p, &f.car.e) } else { (f.active.p, &f.active.e) };
        let component = |k: u32| if car { car_component(f.car.g, e, k as usize) } else { active_component(p, e, k as usize) };
        // Per component: IDs moved away, IDs served, and the least way and least piece of the
        // pieces involved.
        let mut from: BTreeMap<u32, (u64, i64, u32)> = BTreeMap::new();
        let mut on: BTreeMap<u32, (u64, i64, u32)> = BTreeMap::new();
        let count = |by: &mut BTreeMap<u32, (u64, i64, u32)>, x: u32, k: u32| {
            let b = by.entry(x).or_insert((0, p.way[k as usize], k));
            b.0 += 1;
            b.1 = b.1.min(p.way[k as usize]);
            b.2 = b.2.min(k);
        };
        let kept = ids.addresses.iter().copied().chain(ids.places.iter().filter(|&&q| c.places[q].status == 0).map(|&q| c.addresses.len() + q));
        for i in kept {
            let s = &snaps[m].at[i];
            if s.moved_from != blobmap_format::NONE {
                count(&mut from, component(s.moved_from).context("a moved-from piece is servable")?, s.moved_from);
            }
            if let Ok((k, _)) = s.outcome() {
                count(&mut on, component(k).context("a served piece is servable")?, k);
            }
        }
        let sorted = |by: &BTreeMap<u32, (u64, i64, u32)>| {
            let mut v: Vec<(u32, u64, i64)> = by.iter().map(|(&x, &(n, w, _))| (x, n, w)).collect();
            v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            v
        };
        let mut asked: BTreeMap<u32, u32> = BTreeMap::new();
        for (&x, &(_, _, k)) in from.iter().chain(on.iter()).filter(|(x, _)| **x != 0) {
            let a = asked.entry(x).or_insert(k);
            *a = (*a).min(k);
        }
        let real = islands.real(p, &membership(f, mode), &e.nodes, &asked);
        let island = |x: u32| real.get(&x).copied().flatten().map(|j| islands.all[j as usize].label.clone());
        let ids_on = |x: u32| on.get(&x).map_or(0, |b| b.0);
        let rows: Vec<Value> = sorted(&from)
            .into_iter()
            .map(|(x, n, w)| json!({"component": x, "nodes": e.nodes[x as usize], "ids_moved": n, "ids_left": ids_on(x), "way": w, "island": island(x)}))
            .collect();
        emptied.insert(mode.to_string(), Value::Array(rows.iter().filter(|r| r["ids_left"] == 0 && !r["island"].is_null()).cloned().collect()));
        moved.insert(mode.to_string(), Value::Array(rows));
        let rows = sorted(&on).into_iter().filter(|r| r.0 != 0).map(|(x, n, w)| json!({"component": x, "nodes": e.nodes[x as usize], "ids": n, "way": w, "island": island(x)}));
        left.insert(mode.to_string(), Value::Array(rows.collect()));
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
        "served_components": left,
        "islands_emptied": emptied,
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
