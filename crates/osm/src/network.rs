//! A family's ways and their pieces (ported from the old `osm/extract.rs`).
//!
//! Two passes over the file: the ways the profile keeps, then the coordinates and obstacles
//! of the nodes they reference. A node is a routing node when it ends a way or is used at
//! least twice; ways are cut into pieces between consecutive routing nodes. Blobs are
//! decoded in parallel and put back in file order, so nothing depends on the thread count.

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::Result;
use osmpbf::{BlobDecode, BlobReader};
use rayon::prelude::*;

use blobmap_profile::SnapRules;
use blobmap_profile::profile::{Obstacle, ObstacleDir, ObstacleKind};
use blobmap_profile::ways::{NetworkMode, Speeds, WayProfile};

use crate::{arc_time, cheap_ruler_m, degrees};

pub struct Way<A> {
    pub id: i64,
    pub attrs: A,
    pub refs: Vec<i64>,
    /// Snapping eligibility from the way's tags (SPEC §6).
    pub eligible: bool,
}

const MISSING: u64 = u64::MAX;

/// The ways of one family and the nodes they reference.
pub struct Network<A> {
    pub ways: Vec<Way<A>>,
    /// Sorted OSM ids of the referenced nodes; `coords[i]` packs (lat, lon) of `ids[i]`.
    pub ids: Vec<i64>,
    coords: Vec<u64>,
    routing: Vec<bool>,
    /// Obstacles by node index, sorted.
    obstacles: Vec<(usize, Obstacle)>,
}

/// A family's pieces: each a stretch of one way between two routing nodes, with both
/// directions' master times per metric.
#[derive(Clone, Debug, Default)]
pub struct Pieces {
    /// Routing nodes that end a piece, by OSM id ascending.
    pub node_osm: Vec<i64>,
    pub node_lat: Vec<f64>,
    pub node_lon: Vec<f64>,
    /// Per piece, in (way, seq) order.
    pub way: Vec<i64>,
    /// Position of the piece on its way, 0-based in node order.
    pub seq: Vec<u32>,
    /// Routing nodes at the start and end of the piece in the way's node order (equal for a
    /// closed piece).
    pub u: Vec<u32>,
    pub v: Vec<u32>,
    pub len_m: Vec<f64>,
    pub eligible: Vec<bool>,
    /// The piece's points from u to v (CSR by piece).
    pub pt_first: Vec<u32>,
    pub pt_lat: Vec<f64>,
    pub pt_lon: Vec<f64>,
    /// Per metric, per piece: [along, against] the way's node order; +∞ where the metric
    /// may not go.
    pub time: Vec<Vec<[f32; 2]>>,
    /// Pieces dropped because a node had no coordinates.
    pub missing_coords: usize,
}

impl Pieces {
    pub fn len(&self) -> usize {
        self.way.len()
    }

    pub fn is_empty(&self) -> bool {
        self.way.is_empty()
    }

    pub fn points(&self, p: usize) -> std::ops::Range<usize> {
        self.pt_first[p] as usize..self.pt_first[p + 1] as usize
    }
}

fn read_ways<P: WayProfile>(path: &Path, profile: &P, rules: &SnapRules) -> Result<Vec<Way<P::Attrs>>> {
    let mut blocks: Vec<(usize, Vec<Way<P::Attrs>>)> = BlobReader::from_path(path)?
        .enumerate()
        .par_bridge()
        .map(|(k, blob)| -> Result<(usize, Vec<Way<P::Attrs>>)> {
            let mut out = Vec::new();
            if let BlobDecode::OsmData(block) = blob?.decode()? {
                for group in block.groups() {
                    for w in group.ways() {
                        let tags: Vec<(&str, &str)> = w.tags().collect();
                        if let Some(attrs) = profile.way_attrs(tags.iter().copied()) {
                            let mut refs: Vec<i64> = w.refs().collect();
                            refs.dedup();
                            if refs.len() >= 2 {
                                out.push(Way { id: w.id(), attrs, refs, eligible: rules.eligible(tags.iter().copied()) });
                            }
                        }
                    }
                }
            }
            Ok((k, out))
        })
        .collect::<Result<_>>()?;
    blocks.sort_unstable_by_key(|b| b.0);
    Ok(blocks.into_iter().flat_map(|b| b.1).collect())
}

fn read_coords<P: WayProfile>(path: &Path, ids: &[i64], coords: &[AtomicU64], profile: &P) -> Result<Vec<(usize, Obstacle)>> {
    let store = |id: i64, lat: i32, lon: i32| -> Option<usize> {
        let i = ids.binary_search(&id).ok()?;
        coords[i].store(((lat as u32 as u64) << 32) | lon as u32 as u64, Ordering::Relaxed);
        Some(i)
    };
    let found = std::sync::Mutex::new(Vec::new());
    BlobReader::from_path(path)?.par_bridge().try_for_each(|blob| -> Result<()> {
        let mut local = Vec::new();
        if let BlobDecode::OsmData(block) = blob?.decode()? {
            for group in block.groups() {
                for n in group.dense_nodes() {
                    if let Some(i) = store(n.id(), n.decimicro_lat(), n.decimicro_lon())
                        && let Some(o) = profile.obstacle(n.tags())
                    {
                        local.push((i, o));
                    }
                }
                for n in group.nodes() {
                    if let Some(i) = store(n.id(), n.decimicro_lat(), n.decimicro_lon())
                        && let Some(o) = profile.obstacle(n.tags())
                    {
                        local.push((i, o));
                    }
                }
            }
        }
        found.lock().unwrap().extend(local);
        Ok(())
    })?;
    let mut found = found.into_inner().unwrap();
    found.sort_unstable_by_key(|x| x.0);
    found.dedup_by_key(|x| x.0);
    Ok(found)
}

fn unpack(c: u64) -> (f64, f64) {
    (degrees((c >> 32) as u32 as i32), degrees(c as u32 as i32))
}

impl<A: Copy + Send + Sync + Speeds> Network<A> {
    /// The profile's ways of the file and their nodes.
    pub fn read<P: WayProfile<Attrs = A>>(path: &Path, profile: &P, rules: &SnapRules) -> Result<Network<A>> {
        let mut ways = read_ways(path, profile, rules)?;
        ways.par_sort_by_key(|w| w.id);
        ways.dedup_by_key(|w| w.id);
        let mut ids: Vec<i64> = ways.iter().flat_map(|w| w.refs.iter().copied()).collect();
        ids.par_sort_unstable();
        ids.dedup();
        let coords: Vec<AtomicU64> = (0..ids.len()).map(|_| AtomicU64::new(MISSING)).collect();
        let obstacles = read_coords(path, &ids, &coords, profile)?;
        let coords: Vec<u64> = coords.into_iter().map(AtomicU64::into_inner).collect();
        let idx = |id: i64| ids.binary_search(&id).unwrap();
        let mut uses = vec![0u8; ids.len()];
        let mut routing = vec![false; ids.len()];
        for w in &ways {
            for &r in &w.refs {
                let i = idx(r);
                uses[i] = uses[i].saturating_add(1);
            }
            routing[idx(w.refs[0])] = true;
            routing[idx(*w.refs.last().unwrap())] = true;
        }
        for (r, &u) in routing.iter_mut().zip(&uses) {
            *r |= u >= 2;
        }
        Ok(Network { ways, ids, coords, routing, obstacles })
    }

    fn index(&self, id: i64) -> usize {
        self.ids.binary_search(&id).unwrap()
    }

    fn coord(&self, i: usize) -> Option<(f64, f64)> {
        (self.coords[i] != MISSING).then(|| unpack(self.coords[i]))
    }

    fn obstacle(&self, i: usize) -> Option<&Obstacle> {
        self.obstacles.binary_search_by_key(&i, |x| x.0).ok().map(|k| &self.obstacles[k].1)
    }

    /// Every piece in (way, seq) order: `f(way, seq, node indices of the piece, length)`.
    /// A piece with a node without coordinates is skipped (its seq is still counted).
    fn walk(&self, mut f: impl FnMut(&Way<A>, u32, &[usize], f64), mut missing: impl FnMut()) {
        let mut pts: Vec<usize> = Vec::new();
        for w in &self.ways {
            let first = self.index(w.refs[0]);
            pts.clear();
            pts.push(first);
            let mut len = 0.0f64;
            let mut broken = self.coord(first).is_none();
            let mut seq = 0u32;
            for pair in w.refs.windows(2) {
                let (a, b) = (*pts.last().unwrap(), self.index(pair[1]));
                pts.push(b);
                match (self.coord(a), self.coord(b)) {
                    (Some((la, oa)), Some((lb, ob))) => len += cheap_ruler_m(la, oa, lb, ob),
                    _ => broken = true,
                }
                if self.routing[b] {
                    if broken {
                        missing();
                    } else {
                        f(w, seq, &pts, len);
                    }
                    seq += 1;
                    pts.clear();
                    pts.push(b);
                    len = 0.0;
                    broken = self.coord(b).is_none();
                }
            }
        }
    }

    /// The pieces, with one time per mode and direction.
    ///
    /// Delays are the obstacles passed when travelling the piece, priced by the mode
    /// (`NetworkMode::delay`), charged to the piece *arriving* at the obstacle node in the
    /// directions where they apply (OSRM `process_turn` and lib/obstacles.lua):
    /// - a directed obstacle only in its direction;
    /// - `stop=minor` only when the arriving road is slower than the fastest other road at
    ///   the junction (never on a node inside a piece);
    /// - an undirected `stop` at a node with two roads is skipped when it lies < 20 m after
    ///   the previous junction and > 20 m before the next (OSRM's direction heuristic).
    pub fn pieces<M: NetworkMode<A>>(&self, modes: &[M]) -> Pieces {
        // Roads at routing nodes that carry an obstacle: (way, first point, length, speed).
        let mut legs: HashMap<usize, Vec<(i64, usize, f64, f32)>> = HashMap::new();
        if !self.obstacles.is_empty() {
            self.walk(
                |w, _, pts, len| {
                    let (sf, sb) = w.attrs.speeds();
                    for &end in [pts[0], *pts.last().unwrap()].iter() {
                        if self.obstacle(end).is_some() {
                            legs.entry(end).or_default().push((w.id, pts[0], len, sf.max(sb)));
                        }
                    }
                },
                || {},
            );
        }
        let mut out = Pieces { time: vec![Vec::new(); modes.len()], pt_first: vec![0], ..Pieces::default() };
        let mut ends: Vec<usize> = Vec::new();
        let mut cum: Vec<f64> = Vec::new();
        let mut missing = 0usize;
        self.walk(
            |w, seq, pts, len| {
                out.way.push(w.id);
                out.seq.push(seq);
                ends.push(pts[0]);
                ends.push(*pts.last().unwrap());
                out.len_m.push(len);
                out.eligible.push(w.eligible);
                for &p in pts {
                    let (lat, lon) = self.coord(p).unwrap();
                    out.pt_lat.push(lat);
                    out.pt_lon.push(lon);
                }
                out.pt_first.push(out.pt_lat.len() as u32);
                let has_obstacle = !self.obstacles.is_empty() && pts.iter().any(|&p| self.obstacle(p).is_some());
                if has_obstacle {
                    cum.clear();
                    cum.push(0.0);
                    for k in 1..pts.len() {
                        let ((la, oa), (lb, ob)) = (self.coord(pts[k - 1]).unwrap(), self.coord(pts[k]).unwrap());
                        cum.push(cum[k - 1] + cheap_ruler_m(la, oa, lb, ob));
                    }
                }
                for (m, mode) in modes.iter().enumerate() {
                    let a = mode.attrs(&w.attrs);
                    let delay = if has_obstacle { self.delays(w, pts, len, &cum, &legs, mode) } else { [0.0; 2] };
                    let fwd = if a.forward { arc_time(len, a.speed_fwd_kmh, delay[0]) } else { f32::INFINITY };
                    let bwd = if a.backward { arc_time(len, a.speed_bwd_kmh, delay[1]) } else { f32::INFINITY };
                    out.time[m].push([fwd, bwd]);
                }
            },
            || missing += 1,
        );
        out.missing_coords = missing;
        // Routing nodes in OSM id order; ids are sorted, so node indices are too.
        let mut used = ends.clone();
        used.sort_unstable();
        used.dedup();
        for &i in &used {
            let (lat, lon) = self.coord(i).unwrap();
            out.node_osm.push(self.ids[i]);
            out.node_lat.push(lat);
            out.node_lon.push(lon);
        }
        let local = |i: usize| used.binary_search(&i).unwrap() as u32;
        for e in ends.chunks(2) {
            out.u.push(local(e[0]));
            out.v.push(local(e[1]));
        }
        out
    }

    fn delays<M: NetworkMode<A>>(
        &self,
        w: &Way<A>,
        pts: &[usize],
        len: f64,
        cum: &[f64],
        legs: &HashMap<usize, Vec<(i64, usize, f64, f32)>>,
        mode: &M,
    ) -> [f64; 2] {
        let mut delay = [0.0f64; 2];
        let last = pts.len() - 1;
        for (d, fwd) in [(0usize, true), (1, false)] {
            let speed = if fwd { w.attrs.speeds().0 } else { w.attrs.speeds().1 };
            let arrivals: Vec<usize> = if fwd { (1..=last).collect() } else { (0..last).rev().collect() };
            for j in arrivals {
                let Some(o) = self.obstacle(pts[j]) else { continue };
                let Some(cost) = mode.delay(o, fwd) else { continue };
                let src = if fwd { cum[j] } else { len - cum[j] };
                let is_end = j == if fwd { last } else { 0 };
                let others: Vec<&(i64, usize, f64, f32)> = if is_end {
                    legs.get(&pts[j]).into_iter().flatten().filter(|l| (l.0, l.1) != (w.id, pts[0])).collect()
                } else {
                    Vec::new()
                };
                let (roads, tgt) =
                    if is_end { (others.len() + 1, if others.len() == 1 { others[0].2 } else { f64::INFINITY }) } else { (2, len - src) };
                let skip = match o.kind {
                    ObstacleKind::StopMinor => !others.iter().any(|l| l.3 > speed),
                    ObstacleKind::Stop => o.dir == ObstacleDir::Any && roads == 2 && src < 20.0 && tgt > 20.0,
                    _ => false,
                };
                if !skip {
                    delay[d] += cost;
                }
            }
        }
        delay
    }
}

#[cfg(test)]
mod tests {
    use blobmap_profile::active::{ActiveMode, ActiveProfile, Mode};
    use blobmap_profile::profile::{CarProfile, WayAttrs};

    use super::*;

    fn town() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("blobmap-osm-{}", std::process::id()));
        blobmap_fixtures::by_name("town").unwrap().write(&dir).unwrap().pbf
    }

    #[test]
    fn town_pieces() {
        let path = town();
        let cfg = blobmap_profile::config::Config::load(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../config")).unwrap();
        let car = CarProfile::new(&cfg.car.config);
        let net = Network::read(&path, &car, &cfg.car.config.snap).unwrap();
        let p = net.pieces(&[|a: &WayAttrs| *a]);
        // Row 0 of the main grid is way 1: five pieces of 100 m.
        let row0: Vec<usize> = (0..p.len()).filter(|&i| p.way[i] == 1).collect();
        assert_eq!(row0.iter().map(|&i| p.seq[i]).collect::<Vec<_>>(), [0, 1, 2, 3, 4]);
        for &i in &row0 {
            assert!((p.len_m[i] - 100.0).abs() < 0.02, "{}", p.len_m[i]);
            assert!((p.time[0][i][0] - 14.4).abs() < 0.01);
        }
        // Row 2 is one-way; the motorway is ineligible.
        assert!(p.time[0].iter().zip(&p.way).any(|(t, &w)| w == 5 && t[1].is_infinite()));
        assert!((0..p.len()).any(|i| !p.eligible[i]));
        // The traffic signals at (2, 3) delay the pieces arriving at it.
        assert!(p.time[0].iter().any(|t| (t[0] - 16.4).abs() < 0.01));
        assert_eq!(p.missing_coords, 0);
        // The pocket is a closed piece.
        assert!((0..p.len()).any(|i| p.u[i] == p.v[i]));

        let active = ActiveProfile::new(&cfg.active.config);
        let net = Network::read(&path, &active, &cfg.active.config.snap).unwrap();
        let modes: Vec<ActiveMode> = [Mode::Bike, Mode::Walk].map(|mode| ActiveMode { mode, cfg: active.cfg }).to_vec();
        let p = net.pieces(&modes);
        // The footbridge: walk both ways, no bike, not eligible.
        let fb: Vec<usize> = (0..p.len()).filter(|&i| !p.eligible[i] && p.time[0][i][0].is_infinite()).collect();
        assert!(!fb.is_empty() && fb.iter().all(|&i| p.time[1][i][0].is_finite() && p.time[1][i][1].is_finite()));
    }
}
