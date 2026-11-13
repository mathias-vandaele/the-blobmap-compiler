//! Islands (SPEC §3, §6): the island polygons of a `.pbf`, and which small components stand on
//! a real island.
//!
//! A component is on a real island when the island polygon (OSM `place=island|islet`, or a
//! closed `natural=coastline` way) holding most of its pieces holds no servable piece of a
//! larger component of the family. An island with a road to the rest holds the larger
//! component; a fragment on it is no island.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, Ordering};

use anyhow::Result;
use osmpbf::{BlobDecode, BlobReader, RelMemberType};
use rayon::prelude::*;

use crate::osm::column::{Column, ColumnWriter};
use crate::osm::nodes::MISSING;
use crate::osm::{Osm, Pieces};
use crate::osm::geo::{lon_shifts, unwrap_ring};
use crate::snap::Sets;

/// A piece in no island polygon.
const NO_ISLAND: u32 = u32::MAX;

/// Cell size of the polygon index, in degrees.
const CELL: f64 = 0.05;

/// An island polygon: its outer ring (`Islands::ring`), bounding box, a planar area for
/// choosing the smallest, and what OSM calls it.
pub struct Island {
    first: u64,
    len: u32,
    bbox: (f64, f64, f64, f64),
    area: f64,
    pub label: String,
}

/// A ring unwrapped across the antimeridian (`geo::unwrap_ring`): its bounding box and area.
fn shape(ring: &mut [(f64, f64)]) -> ((f64, f64, f64, f64), f64) {
    unwrap_ring(ring);
    let mut b = (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    for &(la, lo) in ring.iter() {
        b = (b.0.min(la), b.1.min(lo), b.2.max(la), b.3.max(lo));
    }
    let c = ((b.0 + b.2) / 2.0).to_radians().cos();
    let area = ring.windows(2).map(|w| (w[0].1 * w[1].0 - w[1].1 * w[0].0) * c).sum::<f64>().abs() / 2.0;
    (b, area)
}

fn contains_at(ring: &[[f64; 2]], la: f64, lo: f64) -> bool {
    let mut inside = false;
    for w in ring.windows(2) {
        let ([ya, xa], [yb, xb]) = (w[0], w[1]);
        if (ya > la) != (yb > la) && lo < xa + (la - ya) / (yb - ya) * (xb - xa) {
            inside = !inside;
        }
    }
    inside
}

/// Index cells per turn of longitude.
const TURN: i32 = (360.0 / CELL) as i32;

fn cell(la: f64, lo: f64) -> (i32, i32) {
    ((la / CELL).floor() as i32, (lo / CELL).floor() as i32)
}

/// The index cell of a point.
fn cell_of(pt: (f64, f64)) -> (i32, i32) {
    let (y, x) = cell(pt.0, pt.1);
    (y, wrap_col(x))
}

/// A cell's column brought into the turn of [-180, 180), for rings past the antimeridian.
fn wrap_col(x: i32) -> i32 {
    (x + TURN / 2).rem_euclid(TURN) - TURN / 2
}

/// The islands of a `.pbf`, with an index by cell (looked up only, never iterated). The
/// rings' points are a column of the work directory (SPEC §10, memory).
pub struct Islands {
    pub all: Vec<Island>,
    points: Column<[f64; 2]>,
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
    /// Reads the island polygons, their points to `dir` if given.
    pub fn read(osm: &Osm, dir: Option<&Path>) -> Result<Islands> {
        let pbf = osm.pbf.as_path();
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
        // The rings' node coordinates.
        let mut nodes: Vec<i64> = polys.iter().flat_map(|p| p.1.iter().copied()).collect();
        nodes.par_sort_unstable();
        nodes.dedup();
        let at = osm.coords(&nodes);
        let deg = |x: u32| x as i32 as f64 * 1e-7;
        let mut all = Vec::new();
        let mut points = ColumnWriter::new(dir, "island-points")?;
        let mut ring = Vec::new();
        for (label, refs) in polys {
            ring.clear();
            for r in &refs {
                let c = at[nodes.binary_search(r).expect("ring node")];
                if c == MISSING {
                    break;
                }
                ring.push((deg((c >> 32) as u32), deg(c as u32)));
            }
            if ring.len() == refs.len() {
                let (bbox, area) = shape(&mut ring);
                all.push(Island { first: points.len() as u64, len: u32::try_from(ring.len())?, bbox, area, label });
                for &(la, lo) in &ring {
                    points.push([la, lo])?;
                }
            }
        }
        drop((at, nodes));
        let points = points.finish()?;
        let mut by_cell: HashMap<(i32, i32), Vec<u32>> = HashMap::new();
        for (k, p) in all.iter().enumerate() {
            let (a, b) = (cell(p.bbox.0, p.bbox.1), cell(p.bbox.2, p.bbox.3));
            for y in a.0..=b.0 {
                for x in a.1..=b.1 {
                    by_cell.entry((y, wrap_col(x))).or_default().push(k as u32);
                }
            }
        }
        Ok(Islands { all, points, by_cell })
    }

    fn ring(&self, j: usize) -> &[[f64; 2]] {
        let x = &self.all[j];
        &self.points[x.first as usize..][..x.len as usize]
    }

    fn contains(&self, j: usize, (la, lo): (f64, f64)) -> bool {
        let b = self.all[j].bbox;
        lon_shifts(lo, b.1, b.3).any(|lo| la >= b.0 && la <= b.2 && contains_at(self.ring(j), la, lo))
    }

    /// The islands around a point.
    fn around(&self, pt: (f64, f64)) -> impl Iterator<Item = u32> + '_ {
        self.by_cell.get(&cell_of(pt)).into_iter().flatten().copied().filter(move |&k| self.contains(k as usize, pt))
    }

    /// The smallest island around a point, ties to the first read.
    pub fn smallest(&self, pt: (f64, f64)) -> Option<u32> {
        self.around(pt).min_by(|&a, &b| self.all[a as usize].area.total_cmp(&self.all[b as usize].area).then(a.cmp(&b)))
    }
    /// Per component asked, the island it stands on if that is a real one. Each servable
    /// piece of the component counts for the smallest island polygon around its first point,
    /// or for none; the polygon holding the most of them wins (ties to none, then to the first
    /// polygon read), and is a real island when it holds no servable piece of a larger
    /// component. `component` gives a servable piece's component.
    pub fn real(&self, p: &Pieces, sets: &dyn Sets, nodes: &[u32], asked: &BTreeSet<u32>, component: &(dyn Fn(u32) -> u32 + Sync)) -> BTreeMap<u32, Option<u32>> {
        let counts: BTreeMap<(u32, u32), u64> = (0..p.len() as u32)
            .into_par_iter()
            .filter(|&k| sets.servable(k))
            .filter_map(|k| {
                let x = component(k);
                asked.contains(&x).then(|| (x, self.smallest(first_point(p, k)).unwrap_or(NO_ISLAND)))
            })
            .fold(BTreeMap::new, |mut m, key| {
                *m.entry(key).or_insert(0) += 1;
                m
            })
            .reduce(BTreeMap::new, |mut a, b| {
                for (key, n) in b {
                    *a.entry(key).or_insert(0) += n;
                }
                a
            });
        let mut most: BTreeMap<u32, (u64, u32)> = BTreeMap::new();
        for (&(x, j), &n) in &counts {
            let m = most.entry(x).or_insert((n, j));
            if n > m.0 || (n == m.0 && j == NO_ISLAND) {
                *m = (n, j);
            }
        }
        let around: BTreeMap<u32, Option<u32>> = asked.iter().map(|&x| (x, most.get(&x).map(|m| m.1).filter(|&j| j != NO_ISLAND))).collect();
        let mut wanted = vec![false; self.all.len()];
        around.values().flatten().for_each(|&j| wanted[j as usize] = true);
        // Per island polygon asked about: the largest component with a servable piece in it.
        let largest: Vec<AtomicU32> = (0..self.all.len()).map(|_| AtomicU32::new(0)).collect();
        (0..p.len() as u32).into_par_iter().filter(|&k| sets.servable(k)).for_each(|k| {
            let (size, pt) = (sets.size(k), first_point(p, k));
            for &j in self.by_cell.get(&cell_of(pt)).into_iter().flatten() {
                let j = j as usize;
                if wanted[j] && size > largest[j].load(Ordering::Relaxed) && self.contains(j, pt) {
                    largest[j].fetch_max(size, Ordering::Relaxed);
                }
            }
        });
        around.into_iter().map(|(x, j)| (x, j.filter(|&j| largest[j as usize].load(Ordering::Relaxed) <= nodes[x as usize]))).collect()
    }
}

/// The first point of a piece.
pub fn first_point(p: &Pieces, k: u32) -> (f64, f64) {
    let i = p.points(k as usize).start;
    (p.pt_lat[i], p.pt_lon[i])
}
