//! A family's ways and their pieces (ported from the old `osm/extract.rs`).
//!
//! The ways the profile keeps are read from the file, their node ids streamed to a column;
//! coordinates and obstacles come from the node locations on disk (`Osm`). A node is a
//! routing node when it ends a way or is used at least twice; ways are cut into pieces
//! between consecutive routing nodes. Blobs are decoded in parallel and put back in file
//! order, so nothing depends on the thread count. With a work directory, the way nodes and
//! the pieces live in files there (SPEC §10, memory), so that no part of the input needs to
//! be held in memory whole but the ways' attributes and a few bits per node.

use std::collections::HashMap;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Result, bail, ensure};
use osmpbf::{BlobDecode, BlobReader};
use rayon::prelude::*;

use crate::osm::column::{Column, ColumnWriter};
use crate::osm::geo::{lon_diff, lon_wrap};
use crate::osm::nodes::{MISSING, Osm, unpack};
use crate::osm::{MAX_PIECE_S, arc_time, cheap_ruler_m};
use crate::profile::SnapRules;
use crate::profile::car::{Obstacle, ObstacleDir, ObstacleKind};
use crate::profile::ways::{NetworkMode, Speeds, WayProfile};

/// Blobs decoded at once, in file order.
const BATCH: usize = 1024;
/// Way nodes per run of ways walked by one task, and runs per batch.
const CHUNK_NODES: usize = 1 << 16;
const BATCH_RUNS: usize = 1024;

#[derive(Clone, Copy)]
pub struct Way<A> {
    pub id: i64,
    pub attrs: A,
    /// Snapping eligibility from the way's tags (SPEC §6).
    pub eligible: bool,
    /// Traffic keeps left (OSRM `WayHandlers.driving_side`): from the way's `driving_side`
    /// tag, else, once coordinates are read, from where its last node lies.
    pub left: bool,
    by_location: bool,
}

/// The distinct nodes of a family's ways, numbered in id order: a sorted list, or for an
/// input whose nodes fill much of the id range, a bitmap over that range with the count of
/// set bits before each word.
enum NodeSet {
    Sorted(Vec<i64>),
    Bits { lo: i64, words: Vec<u64>, before: Vec<u32> },
}

/// Words of a bitset of `n` bits, as atomics for setting bits from many threads.
fn atomic_bits(n: usize) -> Vec<AtomicU64> {
    (0..n.div_ceil(64)).map(|_| AtomicU64::new(0)).collect()
}

fn get_bit(w: &[AtomicU64], i: usize) -> bool {
    w[i / 64].load(Ordering::Relaxed) >> (i % 64) & 1 == 1
}

/// Sets bit i, and tells whether it was set before.
fn set_bit(w: &[AtomicU64], i: usize) -> bool {
    w[i / 64].fetch_or(1 << (i % 64), Ordering::Relaxed) >> (i % 64) & 1 == 1
}

impl NodeSet {
    /// By a bitmap when it is smaller than a sorted copy of the way nodes.
    fn new(refs: &[i64]) -> Result<NodeSet> {
        let (Some(&lo), Some(&hi)) = (refs.par_iter().min(), refs.par_iter().max()) else { return Ok(NodeSet::Sorted(Vec::new())) };
        let span = (hi - lo) as u64 + 1;
        let set = if span / 8 < refs.len() as u64 * 8 {
            let bits = atomic_bits(span as usize);
            refs.par_iter().for_each(|&r| {
                set_bit(&bits, (r - lo) as usize);
            });
            let words: Vec<u64> = bits.into_iter().map(AtomicU64::into_inner).collect();
            let mut before = Vec::with_capacity(words.len());
            let mut n = 0u64;
            for w in &words {
                ensure!(n < u32::MAX as u64, "more than a u32 of nodes in one family");
                before.push(n as u32);
                n += w.count_ones() as u64;
            }
            NodeSet::Bits { lo, words, before }
        } else {
            let mut ids = refs.to_vec();
            ids.par_sort_unstable();
            ids.dedup();
            NodeSet::Sorted(ids)
        };
        ensure!(set.len() < u32::MAX as usize, "{} nodes in one family: more than a u32 index", set.len());
        Ok(set)
    }

    fn len(&self) -> usize {
        match self {
            NodeSet::Sorted(ids) => ids.len(),
            NodeSet::Bits { words, before, .. } => words.last().map_or(0, |w| *before.last().unwrap() as usize + w.count_ones() as usize),
        }
    }

    /// The number of a node of the set.
    fn index(&self, id: i64) -> usize {
        match self {
            NodeSet::Sorted(ids) => ids.binary_search(&id).expect("a way node"),
            NodeSet::Bits { lo, words, before } => {
                let b = (id - lo) as usize;
                let w = words[b / 64];
                debug_assert!(w >> (b % 64) & 1 == 1);
                before[b / 64] as usize + (w & ((1u64 << (b % 64)) - 1)).count_ones() as usize
            }
        }
    }

    /// The ids of the nodes whose numbers have their bit set in `marks`, in id order.
    fn ids_marked(&self, marks: &[AtomicU64]) -> Vec<i64> {
        let mut out = Vec::new();
        match self {
            NodeSet::Sorted(ids) => out.extend(ids.iter().enumerate().filter(|(i, _)| get_bit(marks, *i)).map(|(_, &id)| id)),
            NodeSet::Bits { lo, words, before } => {
                for (k, &w) in words.iter().enumerate() {
                    let mut x = w;
                    let mut i = before[k] as usize;
                    while x != 0 {
                        let t = x.trailing_zeros();
                        if get_bit(marks, i) {
                            out.push(lo + (k as i64) * 64 + t as i64);
                        }
                        i += 1;
                        x &= x - 1;
                    }
                }
            }
        }
        out
    }
}

/// The ways of one family and the nodes they reference.
pub struct Network<'o, A> {
    osm: &'o Osm,
    pub ways: Vec<Way<A>>,
    /// Each way's node ids (CSR by way: `first`).
    refs: Column<i64>,
    first: Vec<u64>,
    nodes: NodeSet,
    /// Routing nodes, by node number.
    routing: Vec<AtomicU64>,
    /// Obstacles by OSM id, sorted.
    obstacles: Vec<(i64, Obstacle)>,
    /// Where the pieces' columns go; in memory without one.
    dir: Option<PathBuf>,
}

/// A family's pieces: each a stretch of one way between two routing nodes, with both
/// directions' master times per metric.
#[derive(Clone, Debug, Default)]
pub struct Pieces {
    /// Routing nodes that end a piece: the first `osm_nodes` by OSM id ascending, then the
    /// nodes made by cutting long pieces (SPEC §7), with OSM id 0, in piece order.
    pub node_osm: Column<i64>,
    pub osm_nodes: usize,
    pub node_lat: Column<f64>,
    pub node_lon: Column<f64>,
    /// Per piece, in (way, seq) order.
    pub way: Column<i64>,
    /// Position of the piece on its way, 0-based in node order; the parts of a cut piece
    /// count as pieces.
    pub seq: Column<u32>,
    /// Routing nodes at the start and end of the piece in the way's node order (equal for a
    /// closed piece).
    pub u: Column<u32>,
    pub v: Column<u32>,
    pub len_m: Column<f64>,
    pub eligible: Column<bool>,
    /// Traffic keeps left on the piece's way (SPEC §4, turns).
    pub left: Column<bool>,
    /// The piece's points from u to v (CSR by piece).
    pub pt_first: Column<u32>,
    pub pt_lat: Column<f64>,
    pub pt_lon: Column<f64>,
    /// Per metric, per piece: [along, against] the way's node order; +∞ where the metric
    /// may not go.
    pub time: Vec<Column<[f32; 2]>>,
    /// Pieces dropped because a node had no coordinates.
    pub missing_coords: usize,
    /// Pieces cut because a master time exceeded the cap (SPEC §7), and the parts made.
    pub cut: usize,
    pub cut_parts: usize,
}

impl Pieces {
    pub fn len(&self) -> usize {
        self.way.len()
    }

    pub fn points(&self, p: usize) -> std::ops::Range<usize> {
        self.pt_first[p] as usize..self.pt_first[p + 1] as usize
    }

    /// The routing node of an OSM node id.
    pub fn osm_node(&self, id: i64) -> Option<u32> {
        self.node_osm[..self.osm_nodes].binary_search(&id).ok().map(|i| i as u32)
    }
}

/// The part (of k equal parts of a piece of `len` metres) that arrives at a point `at`
/// metres from the piece's start, travelling along (`fwd`) or against the node order.
fn part_of(at: f64, len: f64, k: usize, fwd: bool) -> usize {
    let x = at / (len / k as f64);
    let i = if fwd { x.ceil() as i64 - 1 } else { x.floor() as i64 };
    i.clamp(0, k as i64 - 1) as usize
}

/// The point `d` metres along a polyline with cumulative lengths `cum`.
fn point_at(lat: &[f64], lon: &[f64], cum: &[f64], d: f64) -> (f64, f64) {
    let j = cum.partition_point(|&c| c <= d).clamp(1, cum.len() - 1);
    let span = cum[j] - cum[j - 1];
    let f = if span > 0.0 { ((d - cum[j - 1]) / span).clamp(0.0, 1.0) } else { 0.0 };
    (lat[j - 1] + f * (lat[j] - lat[j - 1]), lon_wrap(lon[j - 1] + f * lon_diff(lon[j], lon[j - 1])))
}

impl<A> Network<'_, A> {
    /// Nodes of all ways, counted along each way.
    pub fn way_nodes(&self) -> usize {
        self.refs.len()
    }

    /// Distinct nodes of the ways.
    pub fn nodes(&self) -> usize {
        self.nodes.len()
    }

    fn refs_of(&self, k: usize) -> &[i64] {
        &self.refs[self.first[k] as usize..self.first[k + 1] as usize]
    }
}

/// One blob's ways, and their node ids end to end with each way's count.
type Block<A> = (Vec<Way<A>>, Vec<i64>, Vec<u32>);

/// The profile's ways in file order, which must be by id, with their node ids as one
/// column (CSR: `first`), each node as it routes (`Osm::joined`, across the antimeridian).
fn read_ways<P: WayProfile>(osm: &Osm, profile: &P, rules: &SnapRules, dir: Option<&Path>) -> Result<(Vec<Way<P::Attrs>>, Column<i64>, Vec<u64>)> {
    let path = &osm.pbf;
    let (mut ways, mut first) = (Vec::new(), vec![0u64]);
    let mut refs = ColumnWriter::new(dir, "way_nodes.bin")?;
    let mut blobs = BlobReader::from_path(path)?;
    let mut last = i64::MIN;
    loop {
        let batch: Vec<_> = blobs.by_ref().take(BATCH).collect::<std::result::Result<_, _>>()?;
        if batch.is_empty() {
            break;
        }
        let blocks: Vec<Block<P::Attrs>> = batch
            .into_par_iter()
            .map(|blob| -> Result<Block<P::Attrs>> {
                let (mut out, mut refs, mut lens) = (Vec::new(), Vec::new(), Vec::new());
                if let BlobDecode::OsmData(block) = blob.decode()? {
                    for group in block.groups() {
                        for w in group.ways() {
                            let tags: Vec<(&str, &str)> = w.tags().collect();
                            if let Some(attrs) = profile.way_attrs(tags.iter().copied()) {
                                let start = refs.len();
                                for r in w.refs().map(|r| osm.joined(r)) {
                                    if refs.len() == start || refs.last() != Some(&r) {
                                        refs.push(r);
                                    }
                                }
                                if refs.len() - start >= 2 {
                                    let (left, by_location) = match profile.driving_side(&tags) {
                                        Some(left) => (left, false),
                                        None => (false, true),
                                    };
                                    out.push(Way { id: w.id(), attrs, eligible: rules.eligible(tags.iter().copied()), left, by_location });
                                    lens.push((refs.len() - start) as u32);
                                } else {
                                    refs.truncate(start);
                                }
                            }
                        }
                    }
                }
                Ok((out, refs, lens))
            })
            .collect::<Result<_>>()?;
        for (w, r, l) in blocks {
            for x in &w {
                if x.id <= last {
                    bail!("{}: ways are not sorted by id (way {} after {last}); sort the file first (osmium sort)", path.display(), x.id);
                }
                last = x.id;
            }
            ways.extend(w);
            for x in r {
                refs.push(x)?;
            }
            for n in l {
                first.push(first.last().unwrap() + n as u64);
            }
        }
    }
    Ok((ways, refs.finish()?, first))
}

/// The pieces of a run of ways, as `Network::pieces` writes them.
#[derive(Default)]
struct Run {
    way: Vec<i64>,
    seq: Vec<u32>,
    ends: Vec<[u64; 2]>,
    len_m: Vec<f64>,
    /// [eligible, left].
    flags: Vec<[bool; 2]>,
    time: Vec<Vec<[f32; 2]>>,
    pt_first: Vec<u32>,
    pt: Vec<(f64, f64)>,
    cut_nodes: Vec<(f64, f64)>,
    cut: usize,
    cut_parts: usize,
    missing: usize,
}

/// Where a piece ends: a node of the set (its number), or the k-th node made by cutting.
#[derive(Clone, Copy)]
enum End {
    Node(usize),
    Cut(usize),
}

impl<'o, A: Copy + Send + Sync + Speeds> Network<'o, A> {
    /// The profile's ways of the file and their nodes, with the profile's obstacles (one of
    /// the lists of `osm`). With `dir`, the way nodes and the pieces go to files there.
    pub fn read<P: WayProfile<Attrs = A>>(osm: &'o Osm, obstacles: &[(i64, Obstacle)], profile: &P, rules: &SnapRules, dir: Option<&Path>) -> Result<Network<'o, A>> {
        let (mut ways, refs, first) = read_ways(osm, profile, rules, dir)?;
        let nodes = NodeSet::new(&refs)?;
        let n = nodes.len();
        // A node is routing when it ends a way or is met twice along the ways.
        let (once, routing) = (atomic_bits(n), atomic_bits(n));
        (0..ways.len()).into_par_iter().for_each(|k| {
            let r = &refs[first[k] as usize..first[k + 1] as usize];
            for &id in r {
                let i = nodes.index(id);
                if set_bit(&once, i) {
                    set_bit(&routing, i);
                }
            }
            set_bit(&routing, nodes.index(r[0]));
            set_bit(&routing, nodes.index(r[r.len() - 1]));
        });
        drop(once);
        ways.par_iter_mut().enumerate().filter(|(_, w)| w.by_location).for_each(|(k, w)| {
            let c = osm.coord(refs[first[k + 1] as usize - 1]);
            w.left = c != MISSING && {
                let (lat, lon) = unpack(c);
                profile.keeps_left_at(lat, lon)
            };
        });
        let in_set = |id: i64| match &nodes {
            NodeSet::Sorted(ids) => ids.binary_search(&id).is_ok(),
            NodeSet::Bits { lo, words, .. } => id >= *lo && ((id - lo) as usize) < words.len() * 64 && words[((id - lo) / 64) as usize] >> ((id - lo) % 64) & 1 == 1,
        };
        let obstacles: Vec<(i64, Obstacle)> = obstacles.iter().copied().filter(|&(id, _)| in_set(id)).collect();
        refs.release();
        osm.release_all();
        Ok(Network { osm, ways, refs, first, nodes, routing, obstacles, dir: dir.map(Path::to_path_buf) })
    }

    fn obstacle(&self, id: i64) -> Option<&Obstacle> {
        self.obstacles.binary_search_by_key(&id, |x| x.0).ok().map(|k| &self.obstacles[k].1)
    }

    /// The pieces of ways `ways`, in (way, seq) order: `f(way, seq, the piece's nodes (id,
    /// packed coordinates), length)`. A piece with a node without coordinates is skipped
    /// (its seq is still counted) and `missing` called.
    fn walk(&self, ways: Range<usize>, mut f: impl FnMut(&Way<A>, u32, &[(i64, u64)], f64), mut missing: impl FnMut()) {
        let mut pts: Vec<(i64, u64)> = Vec::new();
        for k in ways {
            let (w, refs) = (&self.ways[k], self.refs_of(k));
            let first = (refs[0], self.osm.coord(refs[0]));
            pts.clear();
            pts.push(first);
            let mut len = 0.0f64;
            let mut broken = first.1 == MISSING;
            let mut seq = 0u32;
            for &id in &refs[1..] {
                let a = *pts.last().unwrap();
                let b = (id, self.osm.coord(id));
                pts.push(b);
                if a.1 != MISSING && b.1 != MISSING {
                    let ((la, oa), (lb, ob)) = (unpack(a.1), unpack(b.1));
                    len += cheap_ruler_m(la, oa, lb, ob);
                } else {
                    broken = true;
                }
                if get_bit(&self.routing, self.nodes.index(id)) {
                    if broken {
                        missing();
                    } else {
                        f(w, seq, &pts, len);
                    }
                    seq += 1;
                    pts.clear();
                    pts.push(b);
                    len = 0.0;
                    broken = b.1 == MISSING;
                }
            }
        }
    }

    /// The ways in runs of about `CHUNK_NODES` way nodes, then the runs in batches, so that a
    /// batch's runs are walked in parallel and their results used in order; the mapped pages
    /// read are released after each batch.
    fn in_batches<T: Send>(&self, run: impl Fn(Range<usize>) -> T + Sync, mut each: impl FnMut(T) -> Result<()>) -> Result<()> {
        let mut runs = Vec::new();
        let mut start = 0;
        for k in 0..self.ways.len() {
            if self.first[k + 1] - self.first[start] >= CHUNK_NODES as u64 || k + 1 == self.ways.len() {
                runs.push(start..k + 1);
                start = k + 1;
            }
        }
        for batch in runs.chunks(BATCH_RUNS) {
            let outs: Vec<T> = batch.par_iter().map(|r| run(r.clone())).collect();
            for o in outs {
                each(o)?;
            }
            self.refs.release();
            self.osm.release_all();
        }
        Ok(())
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
    pub fn pieces<M: NetworkMode<A> + Sync>(&self, modes: &[M]) -> Result<Pieces> {
        // Roads at routing nodes that carry an obstacle: (way, first point, length, speed),
        // in walk order.
        let mut legs: HashMap<i64, Vec<(i64, i64, f64, f32)>> = HashMap::new();
        if !self.obstacles.is_empty() {
            self.in_batches(
                |r| {
                    let mut out = Vec::new();
                    self.walk(
                        r,
                        |w, _, pts, len| {
                            let (sf, sb) = w.attrs.speeds();
                            for &(end, _) in [pts[0], *pts.last().unwrap()].iter() {
                                if self.obstacle(end).is_some() {
                                    out.push((end, (w.id, pts[0].0, len, sf.max(sb))));
                                }
                            }
                        },
                        || {},
                    );
                    out
                },
                |out| {
                    for (end, l) in out {
                        legs.entry(end).or_default().push(l);
                    }
                    Ok(())
                },
            )?;
        }
        let dir = self.dir.as_deref();
        let col = |name: &str| dir.map(|d| d.join(name));
        let (mut way, mut seq, mut len_m) = (ColumnWriter::new(dir, "way.bin")?, ColumnWriter::new(dir, "seq.bin")?, ColumnWriter::new(dir, "len_m.bin")?);
        let (mut eligible, mut left) = (ColumnWriter::new(dir, "eligible.bin")?, ColumnWriter::new(dir, "left.bin")?);
        let (mut pt_first, mut pt_lat, mut pt_lon) = (ColumnWriter::new(dir, "pt_first.bin")?, ColumnWriter::new(dir, "pt_lat.bin")?, ColumnWriter::new(dir, "pt_lon.bin")?);
        let mut time: Vec<ColumnWriter<[f32; 2]>> = (0..modes.len()).map(|m| ColumnWriter::new(dir, &format!("time{m}.bin"))).collect::<Result<_>>()?;
        // Piece ends, by node number or as cut nodes (`n + k`), until the numbers of the
        // routing nodes that end a piece are known.
        let n = self.nodes.len() as u64;
        let (mut ends_u, mut ends_v) = (ColumnWriter::<u64>::new(dir, "ends_u.bin")?, ColumnWriter::<u64>::new(dir, "ends_v.bin")?);
        let used = atomic_bits(self.nodes.len());
        pt_first.push(0u32)?;
        let (mut cut, mut cut_parts, mut missing) = (0usize, 0usize, 0usize);
        let mut cut_nodes: Vec<(f64, f64)> = Vec::new();
        self.in_batches(
            |r| self.piece_run(r, modes, &legs, &used),
            |o| {
                let base = cut_nodes.len() as u64;
                let end = |x: u64| if x < n { x } else { x + base };
                for p in 0..o.way.len() {
                    way.push(o.way[p])?;
                    seq.push(o.seq[p])?;
                    ends_u.push(end(o.ends[p][0]))?;
                    ends_v.push(end(o.ends[p][1]))?;
                    len_m.push(o.len_m[p])?;
                    eligible.push(o.flags[p][0])?;
                    left.push(o.flags[p][1])?;
                    for t in time.iter_mut().zip(&o.time[p]) {
                        t.0.push(*t.1)?;
                    }
                    let pts = o.pt_first[p] as usize..o.pt_first[p + 1] as usize;
                    for j in pts {
                        pt_lat.push(o.pt[j].0)?;
                        pt_lon.push(o.pt[j].1)?;
                    }
                    ensure!(pt_lat.len() < u32::MAX as usize, "more than a u32 of piece points in one family");
                    pt_first.push(pt_lat.len() as u32)?;
                }
                cut_nodes.extend(o.cut_nodes);
                cut += o.cut;
                cut_parts += o.cut_parts;
                missing += o.missing;
                Ok(())
            },
        )?;
        // Routing nodes in OSM id order, then cut nodes; piece ends renumbered to them.
        let ids = self.nodes.ids_marked(&used);
        let (mut node_osm, mut node_lat, mut node_lon) = (ColumnWriter::new(dir, "node_osm.bin")?, ColumnWriter::new(dir, "node_lat.bin")?, ColumnWriter::new(dir, "node_lon.bin")?);
        for &id in &ids {
            let (la, lo) = unpack(self.osm.coord(id));
            node_osm.push(id)?;
            node_lat.push(la)?;
            node_lon.push(lo)?;
        }
        self.osm.release_all();
        for &(la, lo) in &cut_nodes {
            node_osm.push(0)?;
            node_lat.push(la)?;
            node_lon.push(lo)?;
        }
        // Per word of node numbers, the routing nodes before it that end a piece.
        let mut before = Vec::with_capacity(used.len());
        let mut c = 0u32;
        for w in &used {
            before.push(c);
            c += w.load(Ordering::Relaxed).count_ones();
        }
        let local = |x: u64| -> u32 {
            if x < n {
                let x = x as usize;
                before[x / 64] + (used[x / 64].load(Ordering::Relaxed) & ((1u64 << (x % 64)) - 1)).count_ones()
            } else {
                ids.len() as u32 + (x - n) as u32
            }
        };
        let (ends_u, ends_v) = (ends_u.finish()?, ends_v.finish()?);
        let (mut u, mut v) = (ColumnWriter::new(dir, "u.bin")?, ColumnWriter::new(dir, "v.bin")?);
        for (&a, &b) in ends_u.iter().zip(ends_v.iter()) {
            u.push(local(a))?;
            v.push(local(b))?;
        }
        drop((ends_u, ends_v));
        for p in [col("ends_u.bin"), col("ends_v.bin")].into_iter().flatten() {
            std::fs::remove_file(p)?;
        }
        Ok(Pieces {
            osm_nodes: ids.len(),
            node_osm: node_osm.finish()?,
            node_lat: node_lat.finish()?,
            node_lon: node_lon.finish()?,
            way: way.finish()?,
            seq: seq.finish()?,
            u: u.finish()?,
            v: v.finish()?,
            len_m: len_m.finish()?,
            eligible: eligible.finish()?,
            left: left.finish()?,
            pt_first: pt_first.finish()?,
            pt_lat: pt_lat.finish()?,
            pt_lon: pt_lon.finish()?,
            time: time.into_iter().map(ColumnWriter::finish).collect::<Result<_>>()?,
            missing_coords: missing,
            cut,
            cut_parts,
        })
    }

    /// The pieces of one run of ways, with their ends as node numbers or as `n + k` for the
    /// run's k-th cut node.
    fn piece_run<M: NetworkMode<A>>(&self, ways: Range<usize>, modes: &[M], legs: &HashMap<i64, Vec<(i64, i64, f64, f32)>>, used: &[AtomicU64]) -> Run {
        let mut o = Run { pt_first: vec![0], ..Run::default() };
        let n = self.nodes.len() as u64;
        let (mut lat, mut lon, mut cum): (Vec<f64>, Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new(), Vec::new());
        // Parts added on the current way so far, which shift the seq of its later pieces.
        let (mut way_at, mut shift) = (i64::MIN, 0u32);
        let mut missing = 0;
        self.walk(
            ways,
            |w, seq, pts, len| {
                if w.id != way_at {
                    (way_at, shift) = (w.id, 0);
                }
                (lat, lon) = pts.iter().map(|&(_, c)| unpack(c)).unzip();
                cum.clear();
                cum.push(0.0);
                for k in 1..pts.len() {
                    cum.push(cum[k - 1] + cheap_ruler_m(lat[k - 1], lon[k - 1], lat[k], lon[k]));
                }
                let has_obstacle = !self.obstacles.is_empty() && pts.iter().any(|&(p, _)| self.obstacle(p).is_some());
                // Per mode: the attributes and the delays [along, against] with where they are.
                let per_mode: Vec<_> = modes
                    .iter()
                    .map(|mode| {
                        let delays = if has_obstacle { self.delays(w, pts, len, &cum, legs, mode) } else { [Vec::new(), Vec::new()] };
                        (mode.attrs(&w.attrs), delays)
                    })
                    .collect();
                // Master times of part i of k, per mode: [along, against].
                let times = |k: usize, i: usize| -> Vec<[f32; 2]> {
                    per_mode
                        .iter()
                        .map(|(a, delays)| {
                            let delay = |d: usize, fwd: bool| -> f64 {
                                let mut s = 0.0;
                                for &(at, cost) in &delays[d] {
                                    if part_of(at, len, k, fwd) == i {
                                        s += cost;
                                    }
                                }
                                s
                            };
                            let part = len / k as f64;
                            let fwd = if a.forward { arc_time(part, a.speed_fwd_kmh, delay(0, true)) } else { f32::INFINITY };
                            let bwd = if a.backward { arc_time(part, a.speed_bwd_kmh, delay(1, false)) } else { f32::INFINITY };
                            [fwd, bwd]
                        })
                        .collect()
                };
                let fits = |t: &[[f32; 2]]| t.iter().flatten().all(|&x| !x.is_finite() || x <= MAX_PIECE_S);
                let mut k = 1;
                while !(0..k).all(|i| fits(&times(k, i))) {
                    k += 1;
                    assert!(k < 1 << 20, "way {}: a delay alone exceeds the cap", w.id);
                }
                if k > 1 {
                    o.cut += 1;
                    o.cut_parts += k;
                }
                // Cut points: ends, and their distances along the piece.
                let node_end = |p: (i64, u64)| End::Node(self.nodes.index(p.0));
                let mut at: Vec<(End, f64)> = vec![(node_end(pts[0]), 0.0)];
                for i in 1..k {
                    let d = len * i as f64 / k as f64;
                    at.push((End::Cut(o.cut_nodes.len()), d));
                    o.cut_nodes.push(point_at(&lat, &lon, &cum, d));
                }
                at.push((node_end(*pts.last().unwrap()), len));
                let number = |e: End| match e {
                    End::Node(x) => {
                        set_bit(used, x);
                        x as u64
                    }
                    End::Cut(c) => n + c as u64,
                };
                for i in 0..k {
                    let ((a, da), (b, db)) = (at[i], at[i + 1]);
                    o.way.push(w.id);
                    o.seq.push(seq + shift + i as u32);
                    o.ends.push([number(a), number(b)]);
                    o.len_m.push(if k == 1 { len } else { len / k as f64 });
                    o.flags.push([w.eligible, w.left]);
                    let node = |e: End, j: usize, d: f64| match e {
                        End::Node(_) => (lat[j], lon[j]),
                        End::Cut(_) => point_at(&lat, &lon, &cum, d),
                    };
                    o.pt.push(node(a, 0, da));
                    for j in 1..pts.len() - 1 {
                        if k == 1 || (cum[j] > da && cum[j] < db) {
                            o.pt.push((lat[j], lon[j]));
                        }
                    }
                    o.pt.push(node(b, pts.len() - 1, db));
                    o.pt_first.push(o.pt.len() as u32);
                    o.time.push(times(k, i));
                }
                shift += k as u32 - 1;
            },
            || missing += 1,
        );
        o.missing = missing;
        o
    }

    fn delays<M: NetworkMode<A>>(
        &self,
        w: &Way<A>,
        pts: &[(i64, u64)],
        len: f64,
        cum: &[f64],
        legs: &HashMap<i64, Vec<(i64, i64, f64, f32)>>,
        mode: &M,
    ) -> [Vec<(f64, f64)>; 2] {
        // Per direction, in the order of travel: (metres from the piece's start, seconds).
        let mut delay = [Vec::new(), Vec::new()];
        let last = pts.len() - 1;
        for (d, fwd) in [(0usize, true), (1, false)] {
            let speed = if fwd { w.attrs.speeds().0 } else { w.attrs.speeds().1 };
            let arrivals: Vec<usize> = if fwd { (1..=last).collect() } else { (0..last).rev().collect() };
            for j in arrivals {
                let Some(o) = self.obstacle(pts[j].0) else { continue };
                let Some(cost) = mode.delay(o, fwd) else { continue };
                let src = if fwd { cum[j] } else { len - cum[j] };
                let is_end = j == if fwd { last } else { 0 };
                let others: Vec<&(i64, i64, f64, f32)> = if is_end {
                    legs.get(&pts[j].0).into_iter().flatten().filter(|l| (l.0, l.1) != (w.id, pts[0].0)).collect()
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
                    delay[d].push((cum[j], cost));
                }
            }
        }
        delay
    }
}

#[cfg(test)]
mod tests {
    use crate::profile::active::{ActiveMode, ActiveProfile, Mode};
    use crate::profile::car::{CarProfile, WayAttrs};

    use super::*;

    fn town(cfg: &crate::profile::config::Config) -> Osm {
        let dir = std::env::temp_dir().join(format!("blobmap-osm-{}", std::process::id()));
        let pbf = crate::fixtures::by_name("town").unwrap().write(&dir).unwrap().pbf;
        Osm::open(&pbf, &dir.join("nodes"), cfg).unwrap()
    }

    #[test]
    fn town_pieces() {
        let cfg = crate::profile::config::Config::load(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../config")).unwrap();
        let osm = town(&cfg);
        let car = CarProfile::new(&cfg.car.config);
        let net = Network::read(&osm, &osm.car_obstacles, &car, &cfg.car.config.snap, None).unwrap();
        let p = net.pieces(&[|a: &WayAttrs| *a]).unwrap();
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
        let net = Network::read(&osm, &osm.active_obstacles, &active, &cfg.active.config.snap, None).unwrap();
        let modes: Vec<ActiveMode> = [Mode::Bike, Mode::Walk].map(|mode| ActiveMode { mode, cfg: active.cfg }).to_vec();
        let p = net.pieces(&modes).unwrap();
        // The footbridge: walk both ways, no bike, not eligible.
        let fb: Vec<usize> = (0..p.len()).filter(|&i| !p.eligible[i] && p.time[0][i][0].is_infinite()).collect();
        assert!(!fb.is_empty() && fb.iter().all(|&i| p.time[1][i][0].is_finite() && p.time[1][i][1].is_finite()));
    }
}
