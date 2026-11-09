//! What the small-component rule moves (SPEC §6): `fragments <tier file> <inputs name> <data dir>`.
//!
//! Every ID of the inputs is snapped per mode with the rule off (the nearest servable piece)
//! and on (the model's choice). Per mode: the IDs moved and how much farther they stand; the
//! served IDs on components other than 0, before and after, on islands and on the mainland;
//! and every component the rule empties that served more than `LISTED` IDs.
//!
//! A component is on a real island when the smallest island polygon around one of its
//! pieces (OSM `place=island|islet`, or a closed `natural=coastline` way) holds no servable
//! piece of a larger component of the family. An island with a road to the rest holds the
//! larger component; a fragment on it is no island. IDs moved off a real island are what the
//! rule must not do.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, Ordering};

use anyhow::{Context, Result, bail, ensure};
use blobmap_conformance::reference::{Inputs, Reference};
use blobmap_conformance::tier::{TierFile, sha256_file};
use blobmap_osm::Pieces;
use blobmap_snap::modes::{active_component, car_component};
use blobmap_snap::{Grid, Sets};
use osmpbf::{BlobDecode, BlobReader, RelMemberType};
use rayon::prelude::*;

/// Emptied components serving more IDs than this before are listed.
const LISTED: u64 = 20;
/// Cell size of the polygon index, in degrees.
const CELL: f64 = 0.05;

/// An island polygon: its outer ring as (lat, lon), bounding box, a planar area for
/// choosing the smallest, and what OSM calls it.
struct Island {
    ring: Vec<(f64, f64)>,
    bbox: (f64, f64, f64, f64),
    area: f64,
    label: String,
}

impl Island {
    fn new(ring: Vec<(f64, f64)>, label: String) -> Island {
        let mut b = (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
        for &(la, lo) in &ring {
            b = (b.0.min(la), b.1.min(lo), b.2.max(la), b.3.max(lo));
        }
        let c = ((b.0 + b.2) / 2.0).to_radians().cos();
        let area = ring.windows(2).map(|w| (w[0].1 * w[1].0 - w[1].1 * w[0].0) * c).sum::<f64>().abs() / 2.0;
        Island { ring, bbox: b, area, label }
    }

    fn contains(&self, (la, lo): (f64, f64)) -> bool {
        let b = self.bbox;
        if la < b.0 || la > b.2 || lo < b.1 || lo > b.3 {
            return false;
        }
        let mut inside = false;
        for w in self.ring.windows(2) {
            let ((ya, xa), (yb, xb)) = (w[0], w[1]);
            if (ya > la) != (yb > la) && lo < xa + (la - ya) / (yb - ya) * (xb - xa) {
                inside = !inside;
            }
        }
        inside
    }
}

fn cell(la: f64, lo: f64) -> (i32, i32) {
    ((la / CELL).floor() as i32, (lo / CELL).floor() as i32)
}

/// The islands of a `.pbf`, with an index by cell (looked up only, never iterated).
struct Islands {
    all: Vec<Island>,
    by_cell: HashMap<(i32, i32), Vec<u32>>,
}

/// Joins ways (node lists) into closed rings by their end nodes; open leftovers are dropped.
fn rings(mut parts: Vec<Vec<i64>>) -> Vec<Vec<i64>> {
    parts.reverse();
    let mut out = Vec::new();
    while let Some(mut cur) = parts.pop() {
        loop {
            if cur.len() >= 4 && cur.first() == cur.last() {
                out.push(cur);
                break;
            }
            let end = *cur.last().unwrap();
            let Some(j) = parts.iter().position(|w| w[0] == end || *w.last().unwrap() == end) else { break };
            let mut w = parts.remove(j);
            if w[0] != end {
                w.reverse();
            }
            cur.extend_from_slice(&w[1..]);
        }
    }
    out
}

impl Islands {
    fn read(pbf: &Path) -> Result<Islands> {
        let is_island = |k: &str, v: &str| k == "place" && (v == "island" || v == "islet");
        // Pass 1: closed island and coastline ways; island multipolygons' outer ways.
        let found: Mutex<(Vec<(i64, String, Vec<i64>)>, Vec<(i64, String, Vec<i64>)>)> = Mutex::new(Default::default());
        BlobReader::from_path(pbf)?.par_bridge().try_for_each(|blob| -> Result<()> {
            let (mut ways, mut rels) = (Vec::new(), Vec::new());
            if let BlobDecode::OsmData(block) = blob?.decode()? {
                for g in block.groups() {
                    for w in g.ways() {
                        let (mut island, mut coast, mut name) = (false, false, None);
                        for (k, v) in w.tags() {
                            island |= is_island(k, v);
                            coast |= k == "natural" && v == "coastline";
                            if k == "name" {
                                name = Some(v.to_string());
                            }
                        }
                        if island || coast {
                            let refs: Vec<i64> = w.refs().collect();
                            if refs.len() >= 4 && refs.first() == refs.last() {
                                ways.push((w.id(), format!("{} way {}", name.as_deref().unwrap_or("coastline"), w.id()), refs));
                            }
                        }
                    }
                    for r in g.relations() {
                        let (mut island, mut name) = (false, None);
                        for (k, v) in r.tags() {
                            island |= is_island(k, v);
                            if k == "name" {
                                name = Some(v.to_string());
                            }
                        }
                        if island {
                            let outer = r
                                .members()
                                .filter(|m| m.member_type == RelMemberType::Way && matches!(m.role(), Ok("outer") | Ok("")))
                                .map(|m| m.member_id)
                                .collect();
                            rels.push((r.id(), format!("{} relation {}", name.as_deref().unwrap_or("unnamed"), r.id()), outer));
                        }
                    }
                }
            }
            let mut f = found.lock().unwrap();
            f.0.extend(ways);
            f.1.extend(rels);
            Ok(())
        })?;
        let (mut ways, mut rels) = found.into_inner().unwrap();
        ways.sort_by_key(|w| w.0);
        rels.sort_by_key(|r| r.0);
        // Pass 2: the outer ways' nodes.
        let mut want: Vec<i64> = rels.iter().flat_map(|r| r.2.iter().copied()).collect();
        want.sort_unstable();
        want.dedup();
        let members: Mutex<Vec<(i64, Vec<i64>)>> = Mutex::new(Vec::new());
        BlobReader::from_path(pbf)?.par_bridge().try_for_each(|blob| -> Result<()> {
            let mut local = Vec::new();
            if let BlobDecode::OsmData(block) = blob?.decode()? {
                for g in block.groups() {
                    for w in g.ways() {
                        if want.binary_search(&w.id()).is_ok() {
                            local.push((w.id(), w.refs().collect()));
                        }
                    }
                }
            }
            members.lock().unwrap().extend(local);
            Ok(())
        })?;
        let mut members = members.into_inner().unwrap();
        members.sort_by_key(|m| m.0);
        let refs_of = |id: i64| members.binary_search_by_key(&id, |m| m.0).ok().map(|i| members[i].1.clone());
        let mut polys: Vec<(String, Vec<i64>)> = ways.into_iter().map(|(_, l, r)| (l, r)).collect();
        for (_, label, outer) in rels {
            let parts: Vec<Vec<i64>> = outer.into_iter().filter_map(refs_of).filter(|r| r.len() >= 2).collect();
            polys.extend(rings(parts).into_iter().map(|r| (label.clone(), r)));
        }
        // Pass 3: node coordinates.
        let mut nodes: Vec<i64> = polys.iter().flat_map(|p| p.1.iter().copied()).collect();
        nodes.par_sort_unstable();
        nodes.dedup();
        let at: Vec<AtomicU32> = (0..2 * nodes.len()).map(|_| AtomicU32::new(u32::MAX)).collect();
        let store = |id: i64, la: i32, lo: i32| {
            if let Ok(i) = nodes.binary_search(&id) {
                at[2 * i].store(la as u32, Ordering::Relaxed);
                at[2 * i + 1].store(lo as u32, Ordering::Relaxed);
            }
        };
        BlobReader::from_path(pbf)?.par_bridge().try_for_each(|blob| -> Result<()> {
            if let BlobDecode::OsmData(block) = blob?.decode()? {
                for g in block.groups() {
                    for n in g.dense_nodes() {
                        store(n.id(), n.decimicro_lat(), n.decimicro_lon());
                    }
                    for n in g.nodes() {
                        store(n.id(), n.decimicro_lat(), n.decimicro_lon());
                    }
                }
            }
            Ok(())
        })?;
        let deg = |a: &AtomicU32| a.load(Ordering::Relaxed) as i32 as f64 * 1e-7;
        let mut all = Vec::new();
        for (label, refs) in polys {
            let ring: Option<Vec<(f64, f64)>> = refs
                .iter()
                .map(|r| {
                    let i = nodes.binary_search(r).ok()?;
                    (at[2 * i].load(Ordering::Relaxed) != u32::MAX).then(|| (deg(&at[2 * i]), deg(&at[2 * i + 1])))
                })
                .collect();
            if let Some(ring) = ring {
                all.push(Island::new(ring, label));
            }
        }
        let mut by_cell: HashMap<(i32, i32), Vec<u32>> = HashMap::new();
        for (k, p) in all.iter().enumerate() {
            let (a, b) = (cell(p.bbox.0, p.bbox.1), cell(p.bbox.2, p.bbox.3));
            for y in a.0..=b.0 {
                for x in a.1..=b.1 {
                    by_cell.entry((y, x)).or_default().push(k as u32);
                }
            }
        }
        Ok(Islands { all, by_cell })
    }

    /// The islands around a point.
    fn around(&self, pt: (f64, f64)) -> impl Iterator<Item = u32> + '_ {
        self.by_cell.get(&cell(pt.0, pt.1)).into_iter().flatten().copied().filter(move |&k| self.all[k as usize].contains(pt))
    }

    /// The smallest island around a point, ties to the first read.
    fn smallest(&self, pt: (f64, f64)) -> Option<u32> {
        self.around(pt).min_by(|&a, &b| self.all[a as usize].area.total_cmp(&self.all[b as usize].area).then(a.cmp(&b)))
    }
}

fn quantiles(v: &mut [f64]) -> String {
    if v.is_empty() {
        return "–".into();
    }
    v.sort_by(f64::total_cmp);
    let q = |p: f64| v[((p * v.len() as f64).ceil() as usize).clamp(1, v.len()) - 1];
    format!("p50 {:.1}  p90 {:.1}  p99 {:.1}  max {:.1}", q(0.5), q(0.9), q(0.99), v[v.len() - 1])
}

fn first_point(p: &Pieces, k: u32) -> (f64, f64) {
    let i = p.points(k as usize).start;
    (p.pt_lat[i], p.pt_lon[i])
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let [tier_file, name, data] = &a[..] else {
        bail!("usage: fragments <tier file> <inputs name> <data dir>");
    };
    let data = Path::new(data);
    let tf = TierFile::read(Path::new(tier_file))?;
    let set = tf.inputs.iter().find(|s| &s.name == name).with_context(|| format!("the tier lists no inputs named {name}"))?;
    for (p, want) in set.files.iter().filter(|(p, _)| p.starts_with(&format!("{}/", set.config))) {
        ensure!(&sha256_file(&data.join(p))? == want, "{p}: not the file the tier lists");
    }
    let inputs = Inputs {
        pbf: data.join(&set.pbf),
        overture: data.join(&set.overture),
        config: data.join(&set.config),
        extent: set.extent.as_ref().map(|e| data.join(e)),
    };
    let islands = Islands::read(&inputs.pbf)?;
    println!("{} island polygons", islands.all.len());
    let r = Reference::build(&inputs)?;
    let (cgrid, agrid) = (Grid::new(&r.car_pieces), Grid::new(&r.active_pieces));
    let mut real_emptied = 0;
    for (m, mode) in r.modes.iter().enumerate() {
        let (p, e, grid) = if mode.car { (&r.car_pieces, &r.car_components, &cgrid) } else { (&r.active_pieces, &r.active_components, &agrid) };
        let sets: &dyn Sets = &r.sets[m];
        let component = |k: u32| if mode.car { car_component(&r.car, e, k as usize) } else { active_component(p, e, k as usize) }.expect("servable");
        // Per served ID: (piece before, piece after, extra metres).
        let moves: Vec<Option<(u32, u32, f64)>> = (0..r.n())
            .into_par_iter()
            .map(|i| {
                let en = mode.at[i].ok()?;
                let (lat, lon) = r.points[i];
                let f = grid.find(p, lat, lon, r.radius[m], r.small_max[m], sets);
                let (b, a) = (f.nearest.expect("served"), f.servable.expect("served"));
                assert_eq!(a.piece, en.piece, "the re-snap differs from the reference");
                Some((b.piece, a.piece, a.dist_m - b.dist_m))
            })
            .collect();
        // Per component: IDs before and after, the least piece of each (a point on it), and
        // where the IDs moved away from it went.
        let mut before: BTreeMap<u32, (u64, u32)> = BTreeMap::new();
        let mut after: BTreeMap<u32, (u64, u32)> = BTreeMap::new();
        let mut went: BTreeMap<u32, (BTreeMap<u32, u64>, Vec<f64>)> = BTreeMap::new();
        let mut extra = Vec::new();
        for &(b, a, d) in moves.iter().flatten() {
            let (xb, xa) = (component(b), component(a));
            for (map, x, k) in [(&mut before, xb, b), (&mut after, xa, a)] {
                let s = map.entry(x).or_insert((0, k));
                s.0 += 1;
                s.1 = s.1.min(k);
            }
            if b != a {
                extra.push(d);
                let w = went.entry(xb).or_default();
                *w.0.entry(xa).or_default() += 1;
                w.1.push(d);
            }
        }
        // The island around each component serving IDs, and per island the largest component
        // with a servable piece in it.
        let island_of: BTreeMap<u32, Option<u32>> =
            before.iter().chain(after.iter()).map(|(&x, &(_, k))| (x, islands.smallest(first_point(p, k)))).collect();
        let largest: Vec<AtomicU32> = (0..islands.all.len()).map(|_| AtomicU32::new(0)).collect();
        let wanted: Vec<bool> = {
            let mut w = vec![false; islands.all.len()];
            island_of.values().flatten().for_each(|&k| w[k as usize] = true);
            w
        };
        (0..p.len() as u32).into_par_iter().filter(|&k| sets.servable(k)).for_each(|k| {
            let size = sets.size(k);
            let pt = first_point(p, k);
            for j in islands.by_cell.get(&cell(pt.0, pt.1)).into_iter().flatten() {
                let j = *j as usize;
                if wanted[j] && size > largest[j].load(Ordering::Relaxed) && islands.all[j].contains(pt) {
                    largest[j].fetch_max(size, Ordering::Relaxed);
                }
            }
        });
        let on_island = |x: u32| island_of[&x].filter(|&j| largest[j as usize].load(Ordering::Relaxed) <= e.nodes[x as usize]);
        let label = |x: u32| island_of[&x].map_or("–".to_string(), |j| islands.all[j as usize].label.clone());
        let sum = |map: &BTreeMap<u32, (u64, u32)>, island: bool| -> u64 {
            map.iter().filter(|(x, _)| **x != 0 && on_island(**x).is_some() == island).map(|(_, s)| s.0).sum()
        };
        println!("{} ({}): {} IDs served, {} moved; extra metres {}", mode.name, if mode.car { "car" } else { "active" }, moves.iter().flatten().count(), extra.len(), quantiles(&mut extra));
        println!("  served on components other than 0, mainland: {} before, {} after", sum(&before, false), sum(&after, false));
        println!("  served on components other than 0, islands:  {} before, {} after", sum(&before, true), sum(&after, true));
        let emptied: Vec<(u32, u64)> = before.iter().filter(|(x, s)| s.0 > LISTED && !after.contains_key(x)).map(|(&x, s)| (x, s.0)).collect();
        let mut rows: Vec<(u32, u64)> = emptied.clone();
        rows.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let real: Vec<u32> = rows.iter().map(|r| r.0).filter(|&x| on_island(x).is_some()).collect();
        real_emptied += real.len();
        println!("  {} components emptied that served more than {LISTED} IDs, {} of them on a real island:", rows.len(), real.len());
        println!("    {:>7} {:>5} {:>6} {:>8} {:>9}  {:>20}  {:>11}  {:<7} island", "component", "nodes", "IDs", "to", "median m", "point", "OSM way", "verdict");
        for (x, n) in rows {
            let (to, mut d) = went.remove(&x).unwrap_or_default();
            let to = to.iter().max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0))).map_or(u32::MAX, |t| *t.0);
            let med = { d.sort_by(f64::total_cmp); d.get(d.len() / 2).copied().unwrap_or(f64::NAN) };
            let k = before[&x].1;
            let pt = first_point(p, k);
            let verdict = if on_island(x).is_some() { "ISLAND" } else { "fragment" };
            println!("    {x:>7} {:>5} {n:>6} {to:>8} {med:>9.1}  {:>9.5}, {:>9.5}  {:>11}  {verdict:<8} {}", e.nodes[x as usize], pt.0, pt.1, p.way[k as usize], label(x));
        }
        // The real islands still served, for checking the island test against known ones.
        let mut isl: Vec<(u32, u64)> = after.iter().filter(|(x, _)| **x != 0 && on_island(**x).is_some()).map(|(&x, s)| (x, s.0)).collect();
        isl.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        println!("  the 15 islands serving the most IDs after:");
        for (x, n) in isl.into_iter().take(15) {
            println!("    {x:>7} {:>8} nodes {n:>7} IDs  {}", e.nodes[x as usize], label(x));
        }
    }
    println!("{real_emptied} components on real islands emptied");
    Ok(())
}
