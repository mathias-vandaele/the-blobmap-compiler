//! The vocabulary part (SPEC §6, FORMAT §7): which IDs are in the model, their entries and
//! reasons per mode, the places' categories and the taxonomy.

use std::collections::BTreeMap;

use std::path::Path;

use anyhow::{Context, Result};
use blobmap_format::{INF, ModelWriter, NONE, Tensor, Unit, UnitWritten};
use rayon::prelude::*;

use crate::build::family::{Active, Car};
use crate::build::RoadOut;
use crate::graph::active::METRICS;
use crate::osm::column::{Column, ColumnWriter};
use crate::overture::{Catalog, Id, Taxonomy};
use crate::snap::modes::{active_component, car_component};
use crate::snap::{Grid, Membership, Sets, Snapped, parts};

/// The modes, in manifest order, with their family.
pub const MODES: [(&str, &str); 3] = [("car", "car"), ("bike", "active"), ("walk", "active")];

/// One mode's snapping of every point: the outcome, and the trapped component holding the
/// nearest eligible piece, if any (for the data report, SPEC §3).
pub struct ModeSnap {
    pub at: Column<Snapped>,
}

/// Points snapped at once, before their outcomes go to the column.
const SNAP_BATCH: usize = 1 << 22;

pub struct Families<'a> {
    pub car: &'a Car<'a>,
    pub car_road: &'a RoadOut,
    pub active: &'a Active<'a>,
    pub active_road: &'a RoadOut,
    pub car_radius_m: f64,
    pub active_radius_m: f64,
    /// The small-component size per family (SPEC §6).
    pub car_small_max: u32,
    pub active_small_max: u32,
}

/// The three sets of SPEC §6 for a mode.
pub fn membership(f: &Families, mode: &str) -> Membership {
    match mode {
        "car" => Membership::car(f.car.p, f.car.g, &f.car.e),
        m => Membership::active(f.active.p, f.active.g, &f.active.e, METRICS.iter().position(|x| *x == m).unwrap()),
    }
}

/// Snaps the points 0 .. n per mode (car, bike, walk), each mode's outcomes into a column of
/// `dir` (in memory without one).
///
/// The points are snapped in a spatial order, so that a continent's piece coordinates, mapped
/// from the work directory, are read a neighbourhood at a time rather than at random (SPEC
/// §10, memory); each mode's outcomes are put back in index order once its family's grid is
/// dropped. The outcomes do not depend on the order.
pub fn snap(f: &Families, n: usize, point: &(dyn Fn(usize) -> (f64, f64) + Sync), dir: Option<&Path>) -> Result<Vec<ModeSnap>> {
    let (car, act) = (f.car, f.active);
    let order = spatial_order(n, point)?;
    // One family's grid at a time: the modes of a family are consecutive.
    let mut grid: Option<(bool, Grid)> = None;
    let mut sorted: Vec<(&str, Column<Snapped>)> = Vec::new();
    let mut out = Vec::new();
    for (mode, _) in MODES {
        let sets = membership(f, mode);
        let is_car = mode == "car";
        let (p, radius, small_max) = match mode {
            "car" => (car.p, f.car_radius_m, f.car_small_max),
            _ => (act.p, f.active_radius_m, f.active_small_max),
        };
        if grid.as_ref().map(|g| g.0) != Some(is_car) {
            // The old grid goes before the new one is built.
            drop(grid.take());
            in_index_order(&mut sorted, &order, dir, &mut out)?;
            grid = Some((is_car, Grid::new(p)));
        }
        let grid = &grid.as_ref().unwrap().1;
        let trapped = |k: usize| -> Option<u32> {
            if sets.servable(k as u32) {
                return None;
            }
            if mode == "car" {
                car.g.seg_of_piece[k].iter().filter(|&&s| s != NONE).map(|&s| car.e.of[s as usize]).filter(|&x| car.e.trapped[x as usize]).min()
            } else {
                [act.p.u[k], act.p.v[k]].iter().map(|&v| act.e.of[v as usize]).filter(|&x| act.e.trapped[x as usize]).min()
            }
        };
        let mut at = ColumnWriter::new(dir, &format!("{mode}.spatial"))?;
        for batch in order.chunks(SNAP_BATCH) {
            let done: Vec<Snapped> = batch
                .par_iter()
                .map(|&i| {
                    let (lat, lon) = point(i as usize);
                    let fd = grid.find(p, lat, lon, radius, small_max, &sets as &dyn Sets);
                    let t = fd.eligible.and_then(|h| trapped(h.piece as usize)).unwrap_or(NONE);
                    Snapped::new(&fd, t)
                })
                .collect();
            at.push_slice(&done)?;
        }
        sorted.push((mode, at.finish()?));
    }
    drop(grid);
    in_index_order(&mut sorted, &order, dir, &mut out)?;
    Ok(out)
}

/// The points 0 .. n by the Morton code of their 0.01° tile, then by index.
fn spatial_order(n: usize, point: &(dyn Fn(usize) -> (f64, f64) + Sync)) -> Result<Vec<u32>> {
    anyhow::ensure!(n <= u32::MAX as usize, "{n} points to snap, beyond U32 indices");
    // Bit k of x goes to bit 2k.
    let spread = |x: u32| -> u64 {
        let mut x = x as u64;
        x = (x | (x << 16)) & 0x0000_ffff_0000_ffff;
        x = (x | (x << 8)) & 0x00ff_00ff_00ff_00ff;
        x = (x | (x << 4)) & 0x0f0f_0f0f_0f0f_0f0f;
        x = (x | (x << 2)) & 0x3333_3333_3333_3333;
        (x | (x << 1)) & 0x5555_5555_5555_5555
    };
    let mut keys: Vec<u64> = (0..n)
        .into_par_iter()
        .map(|i| {
            let (lat, lon) = point(i);
            // 18 000 rows and 36 000 columns: 16 bits each, a 32-bit code.
            let row = ((lat + 90.0) * 100.0).clamp(0.0, 17_999.0) as u32;
            let col = ((lon + 180.0).rem_euclid(360.0) * 100.0).min(35_999.0) as u32;
            (spread(row) << 1 | spread(col)) << 32 | i as u64
        })
        .collect();
    keys.par_sort_unstable();
    Ok(keys.into_iter().map(|k| k as u32).collect())
}

/// Each mode's outcomes in spatial order, put back in index order as its column, its spatial
/// column then removed.
fn in_index_order(sorted: &mut Vec<(&str, Column<Snapped>)>, order: &[u32], dir: Option<&Path>, out: &mut Vec<ModeSnap>) -> Result<()> {
    if sorted.is_empty() {
        return Ok(());
    }
    let mut rank = vec![0u32; order.len()];
    for (j, &i) in order.iter().enumerate() {
        rank[i as usize] = j as u32;
    }
    for (mode, col) in sorted.drain(..) {
        let mut at = ColumnWriter::new(dir, mode)?;
        for start in (0..rank.len()).step_by(SNAP_BATCH) {
            let done: Vec<Snapped> = rank[start..rank.len().min(start + SNAP_BATCH)].par_iter().map(|&j| col[j as usize]).collect();
            at.push_slice(&done)?;
        }
        drop(col);
        if let Some(d) = dir {
            let path = d.join(format!("{mode}.spatial"));
            std::fs::remove_file(&path).with_context(|| format!("removing {}", path.display()))?;
        }
        out.push(ModeSnap { at: at.finish()? });
    }
    Ok(())
}

/// What the build keeps of the catalog: points in internal-index order.
pub struct Ids {
    pub addresses: Vec<usize>,
    pub places: Vec<usize>,
    /// (alias, internal index of its kept place).
    pub aliases: Vec<(Id, u32)>,
    /// Indices into `Catalog::aliases` of the aliases kept.
    pub alias_rows: Vec<usize>,
}

/// Without `--extent`, an ID is in the model when some mode's family graph has a piece
/// within that mode's radius (SPEC §3); aliases follow their kept place.
pub fn select(c: &Catalog, snaps: &[ModeSnap], polygon: bool) -> Ids {
    let na = c.addresses.len();
    let inside = |i: usize| polygon || snaps.iter().any(|s| s.at[i].any);
    let addresses: Vec<usize> = (0..na).filter(|&i| inside(i)).collect();
    let places: Vec<usize> = (0..c.places.len()).filter(|&p| inside(na + p)).collect();
    let mut new_place = vec![NONE; c.places.len()];
    for (k, &p) in places.iter().enumerate() {
        new_place[p] = (addresses.len() + k) as u32;
    }
    let alias_rows: Vec<usize> = (0..c.aliases.len()).filter(|&k| new_place[c.aliases[k].1 as usize] != NONE).collect();
    let aliases = alias_rows.iter().map(|&k| (c.aliases[k].0, new_place[c.aliases[k].1 as usize])).collect();
    Ids { addresses, places, aliases, alias_rows }
}

/// Per mode, how many IDs are served and how many have each reason (data report).
pub type Served = BTreeMap<String, BTreeMap<String, u64>>;

pub const REASONS: [&str; 5] = ["", "NoEligibleRoad", "NoRoad", "TrappedComponent", "FilteredPlace"];

/// Writes the vocabulary part (FORMAT §7) as one unit, a group of tensors at a time in
/// full-name order (each mode's, the IDs, the places, the road hash, the taxonomy), so that
/// one group at a time is in memory.
pub fn write(f: &Families, c: &Catalog, snaps: &[ModeSnap], ids: &Ids, road_hash: &[u8; 32], w: &mut ModelWriter) -> Result<(UnitWritten, Served)> {
    let put = |w: &mut ModelWriter, u: Unit| -> Result<()> {
        for (name, t) in u.tensors {
            w.put(name, t)?;
        }
        Ok(())
    };
    w.begin_unit()?;
    let na = c.addresses.len();
    // Point i of the model in internal-index order: its catalog point index, and whether it
    // is a filtered place.
    let pt = |i: usize| match ids.addresses.get(i) {
        Some(&a) => (a, false),
        None => {
            let p = ids.places[i - ids.addresses.len()];
            (na + p, c.places[p].status != 0)
        }
    };
    let n = ids.addresses.len() + ids.places.len();
    let mut served: Served = BTreeMap::new();
    let mut modes: Vec<usize> = (0..MODES.len()).collect();
    modes.sort_by_key(|&m| format!("vocab/{}/{}.", MODES[m].1, MODES[m].0));
    for m in modes {
        let (mode, family) = MODES[m];
        let counts = served.entry(mode.to_string()).or_default();
        let road_rows = if mode == "car" { 0 } else { n };
        let (mut node, mut road, mut off) = (vec![NONE; 2 * n], vec![NONE; road_rows], vec![INF; 4 * n]);
        let (mut rid, mut rcode, mut rm) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..n {
            let (pt, filtered) = pt(i);
            if filtered {
                *counts.entry("FilteredPlace".into()).or_default() += 1;
                continue;
            }
            let (piece, frac) = match snaps[m].at[pt].outcome() {
                Ok(h) => h,
                Err((code, dist)) => {
                    rid.push(i as u32);
                    rcode.push(code);
                    rm.push(dist);
                    *counts.entry(REASONS[code as usize].into()).or_default() += 1;
                    continue;
                }
            };
            *counts.entry("served".into()).or_default() += 1;
            let k = piece as usize;
            if mode == "car" {
                let (car, b) = (f.car, f.car_road);
                let x = car_component(car.g, &car.e, k).expect("servable");
                for d in 0..2 {
                    let s = car.g.seg_of_piece[k][d];
                    if s == NONE || car.e.of[s as usize] != x {
                        continue;
                    }
                    node[2 * i + d] = b.layout_of[s as usize];
                    let g = if d == 0 { frac } else { 1.0 - frac };
                    let (before, after) = parts(car.g.w[s as usize], g);
                    off[4 * i + 2 * d] = before;
                    off[4 * i + 2 * d + 1] = after;
                }
            } else {
                let (act, b) = (f.active, f.active_road);
                let metric = m - 1;
                debug_assert!(active_component(act.p, &act.e, k).is_some(), "a served piece is servable");
                let [along, against] = act.g.road_of_piece[k];
                let (uu, vv, fu, ruv, rvu) = if along != NONE {
                    (act.p.u[k], act.p.v[k], frac, along, against)
                } else {
                    (act.p.v[k], act.p.u[k], 1.0 - frac, against, along)
                };
                node[2 * i] = b.layout_of[uu as usize];
                node[2 * i + 1] = b.layout_of[vv as usize];
                road[i] = b.arc_local[ruv as usize];
                let (u_id, id_v) = parts(act.g.w[metric][ruv as usize], fu);
                let (v_id, id_u) = if rvu != NONE { parts(act.g.w[metric][rvu as usize], 1.0 - fu) } else { (INF, INF) };
                off[4 * i..4 * i + 4].copy_from_slice(&[u_id, id_u, v_id, id_v]);
            }
        }
        // In full-name order, each vector dropped as it becomes a tensor.
        let p = format!("vocab/{family}/{mode}.");
        if mode == "car" {
            w.put(format!("{p}off"), Tensor::new(&[n as u64, 2, 2], &std::mem::take(&mut off)))?;
        } else {
            w.put(format!("{p}node"), Tensor::rows(2, &std::mem::take(&mut node)))?;
            w.put(format!("{p}off"), Tensor::rows(4, &std::mem::take(&mut off)))?;
        }
        w.put(format!("{p}reason_code"), Tensor::vec(&rcode))?;
        w.put(format!("{p}reason_id"), Tensor::vec(&rid))?;
        w.put(format!("{p}reason_m"), Tensor::vec(&rm))?;
        if mode == "car" {
            w.put(format!("{p}seg"), Tensor::rows(2, &std::mem::take(&mut node)))?;
        } else {
            w.put(format!("{p}road"), Tensor::vec(&std::mem::take(&mut road)))?;
        }
    }
    let id_rows = |v: &mut dyn Iterator<Item = &Id>| {
        let b: Vec<u8> = v.flatten().copied().collect();
        Tensor::of_bytes(&[b.len() as u64 / 16, 16], b)
    };
    w.put("vocab/ids/address", id_rows(&mut ids.addresses.iter().map(|&i| &c.addresses[i].id)))?;
    w.put("vocab/ids/alias", id_rows(&mut ids.aliases.iter().map(|a| &a.0)))?;
    w.put("vocab/ids/alias_to", Tensor::vec(&ids.aliases.iter().map(|a| a.1).collect::<Vec<u32>>()))?;
    w.put("vocab/ids/place", id_rows(&mut ids.places.iter().map(|&p| &c.places[p].id)))?;
    let mut u = Unit::default();
    let places: Vec<&crate::overture::Place> = ids.places.iter().map(|&p| &c.places[p]).collect();
    u.put("vocab/places/status", Tensor::vec(&places.iter().map(|p| p.status).collect::<Vec<u8>>()));
    u.put("vocab/places/primary", Tensor::vec(&places.iter().map(|p| p.primary).collect::<Vec<u16>>()));
    u.put("vocab/places/basic", Tensor::vec(&places.iter().map(|p| p.basic).collect::<Vec<u16>>()));
    let mut alt_first = vec![0u32];
    let mut alt = Vec::new();
    for p in &places {
        alt.extend_from_slice(c.alt(p));
        alt_first.push(alt.len() as u32);
    }
    u.put("vocab/places/alt_first", Tensor::vec(&alt_first));
    u.put("vocab/places/alt", Tensor::vec(&alt));
    put(w, u)?;
    let mut u = Unit::default();
    u.put("vocab/road_hash", Tensor::vec(road_hash));
    put(w, u)?;
    let mut u = Unit::default();
    taxonomy(&c.taxonomy, &mut u);
    put(w, u)?;
    Ok((w.end_unit()?, served))
}

fn strings(u: &mut Unit, name: &str, s: &[String]) {
    let mut first = vec![0u32];
    let mut bytes = Vec::new();
    for x in s {
        bytes.extend_from_slice(x.as_bytes());
        first.push(bytes.len() as u32);
    }
    u.put(format!("{name}.bytes"), Tensor::vec(&bytes));
    u.put(format!("{name}.first"), Tensor::vec(&first));
}

fn taxonomy(t: &Taxonomy, u: &mut Unit) {
    strings(u, "vocab/taxonomy/name", &t.name);
    u.put("vocab/taxonomy/parent", Tensor::vec(&t.parent));
    u.put("vocab/taxonomy/basic", Tensor::vec(&t.basic));
    u.put("vocab/taxonomy/is_basic", Tensor::vec(&t.is_basic.iter().map(|&b| b as u8).collect::<Vec<u8>>()));
    strings(u, "vocab/taxonomy/redirect", &t.redirect);
    u.put("vocab/taxonomy/redirect_to", Tensor::vec(&t.redirect_to));
}
