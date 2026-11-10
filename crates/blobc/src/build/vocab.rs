//! The vocabulary part (SPEC §6, FORMAT §7): which IDs are in the model, their entries and
//! reasons per mode, the places' categories and the taxonomy.

use std::collections::BTreeMap;

use blobmap_format::{INF, NONE, Tensor, Unit};
use rayon::prelude::*;

use crate::build::family::{Active, Car};
use crate::build::hier::Built;
use crate::build::layout::Layout;
use crate::graph::active::METRICS;
use crate::overture::{Catalog, Id, Taxonomy};
use crate::snap::modes::{active_component, car_component};
use crate::snap::{Grid, Membership, Sets, Snapped, parts};

/// The modes, in manifest order, with their family.
pub const MODES: [(&str, &str); 3] = [("car", "car"), ("bike", "active"), ("walk", "active")];

/// One mode's snapping of every point: the outcome, and the trapped component holding the
/// nearest eligible piece, if any (for the data report, SPEC §3).
pub struct ModeSnap {
    pub at: Vec<Snapped>,
}

pub struct Families<'a> {
    pub car: &'a Car<'a>,
    pub car_built: &'a Built,
    pub car_layout: &'a Layout,
    pub active: &'a Active<'a>,
    pub active_built: &'a Built,
    pub active_layout: &'a Layout,
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

/// Snaps points per mode (car, bike, walk).
pub fn snap(f: &Families, points: &[(f64, f64)]) -> Vec<ModeSnap> {
    let (car, act) = (f.car, f.active);
    let cgrid = Grid::new(car.p);
    let agrid = Grid::new(act.p);
    let mut out = Vec::new();
    for (mode, _) in MODES {
        let sets = membership(f, mode);
        let (grid, p, radius, small_max) = match mode {
            "car" => (&cgrid, car.p, f.car_radius_m, f.car_small_max),
            _ => (&agrid, act.p, f.active_radius_m, f.active_small_max),
        };
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
        let at = points
            .par_iter()
            .map(|&(lat, lon)| {
                let fd = grid.find(p, lat, lon, radius, small_max, &sets as &dyn Sets);
                let t = fd.eligible.and_then(|h| trapped(h.piece as usize)).unwrap_or(NONE);
                Snapped::new(&fd, t)
            })
            .collect();
        out.push(ModeSnap { at });
    }
    out
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

pub fn unit(f: &Families, c: &Catalog, snaps: &[ModeSnap], ids: &Ids, road_hash: &[u8; 32]) -> (Unit, Served) {
    let mut u = Unit::default();
    let na = c.addresses.len();
    // Points of the model in internal-index order: catalog point index, and whether filtered.
    let pts: Vec<(usize, bool)> =
        ids.addresses.iter().map(|&i| (i, false)).chain(ids.places.iter().map(|&p| (na + p, c.places[p].status != 0))).collect();
    let n = pts.len();
    let id_rows = |v: Vec<&Id>| Tensor::rows(16, &v.into_iter().flatten().copied().collect::<Vec<u8>>());
    u.put("vocab/ids/address", id_rows(ids.addresses.iter().map(|&i| &c.addresses[i].id).collect()));
    u.put("vocab/ids/place", id_rows(ids.places.iter().map(|&p| &c.places[p].id).collect()));
    u.put("vocab/ids/alias", id_rows(ids.aliases.iter().map(|a| &a.0).collect()));
    u.put("vocab/ids/alias_to", Tensor::vec(&ids.aliases.iter().map(|a| a.1).collect::<Vec<u32>>()));
    u.put("vocab/road_hash", Tensor::vec(road_hash));
    let mut served: Served = BTreeMap::new();
    for (m, (mode, family)) in MODES.iter().enumerate() {
        let counts = served.entry(mode.to_string()).or_default();
        let (mut node, mut road, mut off) = (vec![NONE; 2 * n], vec![NONE; n], Vec::new());
        off.resize(4 * n, INF);
        let (mut rid, mut rcode, mut rm) = (Vec::new(), Vec::new(), Vec::new());
        for (i, &(pt, filtered)) in pts.iter().enumerate() {
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
            if *mode == "car" {
                let (car, b) = (f.car, f.car_built);
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
                let (act, b, l) = (f.active, f.active_built, f.active_layout);
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
                road[i] = l.arc_local[ruv as usize];
                let (u_id, id_v) = parts(act.g.w[metric][ruv as usize], fu);
                let (v_id, id_u) = if rvu != NONE { parts(act.g.w[metric][rvu as usize], 1.0 - fu) } else { (INF, INF) };
                off[4 * i..4 * i + 4].copy_from_slice(&[u_id, id_u, v_id, id_v]);
            }
        }
        let p = format!("vocab/{family}/{mode}.");
        if *mode == "car" {
            u.put(format!("{p}seg"), Tensor::rows(2, &node));
            u.put(format!("{p}off"), Tensor::new(&[n as u64, 2, 2], &off));
        } else {
            u.put(format!("{p}node"), Tensor::rows(2, &node));
            u.put(format!("{p}road"), Tensor::vec(&road));
            u.put(format!("{p}off"), Tensor::rows(4, &off));
        }
        u.put(format!("{p}reason_id"), Tensor::vec(&rid));
        u.put(format!("{p}reason_code"), Tensor::vec(&rcode));
        u.put(format!("{p}reason_m"), Tensor::vec(&rm));
    }
    let places: Vec<&crate::overture::Place> = ids.places.iter().map(|&p| &c.places[p]).collect();
    u.put("vocab/places/status", Tensor::vec(&places.iter().map(|p| p.status).collect::<Vec<u8>>()));
    u.put("vocab/places/primary", Tensor::vec(&places.iter().map(|p| p.primary).collect::<Vec<u16>>()));
    u.put("vocab/places/basic", Tensor::vec(&places.iter().map(|p| p.basic).collect::<Vec<u16>>()));
    let mut alt_first = vec![0u32];
    let mut alt = Vec::new();
    for p in &places {
        alt.extend_from_slice(&p.alt);
        alt_first.push(alt.len() as u32);
    }
    u.put("vocab/places/alt_first", Tensor::vec(&alt_first));
    u.put("vocab/places/alt", Tensor::vec(&alt));
    taxonomy(&c.taxonomy, &mut u);
    (u, served)
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
