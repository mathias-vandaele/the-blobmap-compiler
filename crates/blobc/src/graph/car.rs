//! The CAR family's graph (SPEC §4, FORMAT §5.3): nodes are directed road segments, arcs
//! are allowed turns. Ported from the old `edge/edge.rs` (OSRM v6 turn costs, u-turns,
//! node-via restrictions) and `edge/restrict.rs` (via-way copies as Aho–Corasick states),
//! with `only_*` rewriting, two or more via ways, and copy rows added.

use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::path::Path;

use anyhow::{Result, ensure};
use blobmap_model::NONE;
use rayon::prelude::*;

use crate::graph::components::{Adjacency, Components, scc};
use crate::graph::round::round_columns;
use crate::osm::column::{Column, ColumnMut, ColumnWriter};
use crate::osm::geo::{Local, bearing_deg};
use crate::osm::{Pieces, Restriction, Via, cheap_ruler_m};
use crate::profile::CarTurns;

/// OSRM looks 10 m along a road for the turn angle (coordinate_extractor
/// LOOKAHEAD_DISTANCE_WITHOUT_LANES).
pub const LOOKAHEAD_M: f64 = 10.0;

/// A turn from segment a (arriving at the via node) onto segment b.
#[derive(Clone, Copy, Debug)]
pub struct Turn {
    /// Degrees in (-180, 180]: 0 straight, + right, - left.
    pub angle: f64,
    /// Roads (piece ends) at the via node.
    pub roads: usize,
    pub u_turn: bool,
    /// Traffic keeps left on the road arriving (OSRM takes the incoming edge's flag).
    pub left: bool,
}

/// OSRM v6.0.0 car.lua `process_turn`, the duration part: a sigmoid of the angle when the
/// node has more than two roads or the turn is a u-turn, plus the u-turn penalty. Where
/// traffic keeps left the bias is inverted, so right turns cost what left ones do elsewhere.
pub fn turn_seconds(c: &CarTurns, t: &Turn) -> f64 {
    if !(t.roads > 2 || t.u_turn) {
        return 0.0;
    }
    let (p, b) = (c.turn_penalty_s, if t.left { 1.0 / c.turn_bias } else { c.turn_bias });
    let mut s = if t.angle >= 0.0 {
        p / (1.0 + (-((13.0 / b) * t.angle / 180.0 - 6.5 * b)).exp())
    } else {
        p / (1.0 + (-((13.0 * b) * -t.angle / 180.0 - 6.5 / b)).exp())
    };
    if t.u_turn {
        s += c.u_turn_penalty_s;
    }
    s
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CarStats {
    pub segments: usize,
    /// Original segments on roads where traffic keeps left.
    pub left_segments: usize,
    pub copies: usize,
    pub turns: usize,
    pub u_turns_dropped: usize,
    /// Restrictions turned into forbidden sequences, by kind.
    pub node_via_applied: usize,
    pub way_via_applied: usize,
    /// Restrictions whose ways, nodes or chain are not in the graph, or whose sequence a
    /// one-way already makes impossible.
    pub unresolved: usize,
    pub forbidden_sequences: usize,
    /// Via-way copies with no turn out (SPEC §5): every exit completes a forbidden sequence.
    pub dead_end_copies: usize,
}

/// The turn graph. Nodes `0 .. originals` are the road segments in (way, seq, dir) order;
/// the rest are via-way copies, ordered by the prefix they stand for.
#[derive(Clone, Debug, Default)]
pub struct CarGraph {
    pub originals: usize,
    pub piece: Column<u32>,
    pub way: Column<i64>,
    pub dir: Column<u8>,
    pub seq: Column<u32>,
    /// Routing nodes at the segment's start and end.
    pub tail: Column<u32>,
    pub head: Column<u32>,
    pub time: Column<f32>,
    pub w: Column<u16>,
    /// For a copy, its original; `NONE` for an original.
    pub copy_of: Column<u32>,
    /// Per piece, the original segments [along, against] its node order, or `NONE`.
    pub seg_of_piece: Column<[u32; 2]>,
    /// Turns, sorted by (from, to).
    pub turn_from: Column<u32>,
    pub turn_to: Column<u32>,
    pub turn_cost: Column<u16>,
    /// Each node's turns are `turn_first[v] .. turn_first[v + 1]`.
    pub turn_first: Column<u64>,
    /// The turns before any restriction, by the original segment they leave, and the
    /// restrictions as forbidden sequences of original segments: what an independent
    /// turn-aware Dijkstra needs (SPEC §11).
    pub base: BaseTurns,
    pub forbidden: Vec<Vec<u32>>,
    /// Restrictions with a prefix copy that has no turn out, likely mis-tagged (data
    /// report): (relation id, the copy), sorted.
    pub dead_ends: Vec<(i64, u32)>,
    pub stats: CarStats,
}

/// Turns by the original segment they leave, CSR: the segment entered and the cost.
#[derive(Clone, Debug, Default)]
pub struct BaseTurns {
    first: Column<u64>,
    to: Column<u32>,
    cost: Column<u16>,
}

impl BaseTurns {
    /// The segments they leave.
    pub fn len(&self) -> usize {
        self.first.len().saturating_sub(1)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The turns leaving segment a: (segment entered, cost).
    pub fn of(&self, a: usize) -> impl Iterator<Item = (u32, u16)> + '_ {
        let r = self.first[a] as usize..self.first[a + 1] as usize;
        self.to[r.clone()].iter().copied().zip(self.cost[r].iter().copied())
    }
}

/// The point at distance `d` along piece p.
fn point_at(p: &Pieces, piece: usize, cum: &[f64], d: f64) -> (f64, f64) {
    let r = p.points(piece);
    let d = d.clamp(0.0, *cum.last().unwrap());
    let k = cum.partition_point(|&c| c < d).clamp(1, cum.len() - 1);
    let (i, j) = (r.start + k - 1, r.start + k);
    let seg = cum[k] - cum[k - 1];
    let f = if seg > 0.0 { (d - cum[k - 1]) / seg } else { 0.0 };
    let l = Local::at(p.pt_lat[i], p.pt_lon[i]);
    let (x, y) = l.xy(p.pt_lat[j], p.pt_lon[j]);
    l.latlon(f * x, f * y)
}

/// (bearing leaving the tail, bearing arriving at the head) of each direction of a piece.
fn bearings(p: &Pieces, piece: usize) -> [(f64, f64); 2] {
    let r = p.points(piece);
    let mut cum = vec![0.0];
    for i in r.start + 1..r.end {
        cum.push(cum.last().unwrap() + cheap_ruler_m(p.pt_lat[i - 1], p.pt_lon[i - 1], p.pt_lat[i], p.pt_lon[i]));
    }
    let len = *cum.last().unwrap();
    let pt = |d: f64| point_at(p, piece, &cum, d);
    let b = |a: f64, c: f64| bearing_deg(pt(a), pt(c));
    let l = LOOKAHEAD_M;
    [(b(0.0, l.min(len)), b((len - l).max(0.0), len)), (b(len, (len - l).max(0.0)), b(l.min(len), 0.0))]
}

/// The automaton of FORMAT §5.3: forbidden sequences, and their proper prefixes of length
/// at least 2 as states.
struct Automaton {
    forbidden: HashSet<Vec<u32>>,
    prefixes: HashSet<Vec<u32>>,
    forbidden_pairs: HashSet<(u32, u32)>,
    prefix_pairs: HashSet<(u32, u32)>,
}

impl Automaton {
    fn new(seqs: &[Vec<u32>]) -> Automaton {
        let mut a = Automaton {
            forbidden: HashSet::new(),
            prefixes: HashSet::new(),
            forbidden_pairs: HashSet::new(),
            prefix_pairs: HashSet::new(),
        };
        for s in seqs {
            if s.len() == 2 {
                a.forbidden_pairs.insert((s[0], s[1]));
            }
            for j in 2..s.len() {
                if j == 2 {
                    a.prefix_pairs.insert((s[0], s[1]));
                }
                a.prefixes.insert(s[..j].to_vec());
            }
            a.forbidden.insert(s.clone());
        }
        a
    }

    /// The state reached from state `p` by a turn onto `b`: `Ok(None)` for the original b,
    /// `Ok(Some(prefix))` for a copy, `Err(())` if the turn completes a forbidden sequence.
    fn step(&self, p: &[u32], b: u32) -> Result<Option<Vec<u32>>, ()> {
        if p.len() == 1 {
            let pair = (p[0], b);
            if self.forbidden_pairs.contains(&pair) {
                return Err(());
            }
            return Ok(self.prefix_pairs.contains(&pair).then(|| vec![p[0], b]));
        }
        let mut q = p.to_vec();
        q.push(b);
        if (0..q.len() - 1).any(|i| self.forbidden.contains(&q[i..])) {
            return Err(());
        }
        Ok((0..q.len() - 1).find(|&i| self.prefixes.contains(&q[i..])).map(|i| q[i..].to_vec()))
    }
}

/// `f` over `0..n` in ranges, in parallel, each range's result given to `each` in order.
fn in_ranges<T: Send>(n: usize, f: impl Fn(Range<usize>) -> T + Sync, mut each: impl FnMut(T) -> Result<()>) -> Result<()> {
    const RANGE: usize = 1 << 16;
    const BATCH: usize = 256;
    let ranges: Vec<Range<usize>> = (0..n).step_by(RANGE).map(|a| a..(a + RANGE).min(n)).collect();
    for b in ranges.chunks(BATCH) {
        let outs: Vec<T> = b.par_iter().map(|r| f(r.clone())).collect();
        for o in outs {
            each(o)?;
        }
    }
    Ok(())
}

/// Each way's pieces, ascending: found by binary search where the pieces come sorted by way
/// (a `.pbf`'s ways are), else from a list of (way, piece) sorted.
pub enum ByWay<'a> {
    Sorted(&'a [i64]),
    Listed(Vec<(i64, u32)>),
}

impl<'a> ByWay<'a> {
    pub fn new(way: &'a [i64]) -> ByWay<'a> {
        if way.par_windows(2).all(|w| w[0] <= w[1]) {
            ByWay::Sorted(way)
        } else {
            let mut l: Vec<(i64, u32)> = way.iter().enumerate().map(|(i, &w)| (w, i as u32)).collect();
            l.par_sort_unstable();
            ByWay::Listed(l)
        }
    }

    pub fn pieces(&self, w: i64) -> Vec<usize> {
        match self {
            ByWay::Sorted(way) => (way.partition_point(|&x| x < w)..way.partition_point(|&x| x <= w)).collect(),
            ByWay::Listed(l) => {
                let lo = l.partition_point(|x| x.0 < w);
                l[lo..].iter().take_while(|x| x.0 == w).map(|x| x.1 as usize).collect()
            }
        }
    }
}

/// The turn graph on its original segments, a turn into or out of a copy counting as one into
/// or out of its original: a node's turns, then its copies' (in copy order), as `Csr` over
/// those arcs would list them.
struct OnOriginals<'a> {
    g: &'a CarGraph,
    /// (original, copy), sorted.
    copies: &'a [(u32, u32)],
}

impl OnOriginals<'_> {
    fn orig(&self, v: u32) -> u32 {
        if (v as usize) < self.g.originals { v } else { self.g.copy_of[v as usize] }
    }

    fn own(&self, v: usize) -> Range<u64> {
        self.g.turn_first[v]..self.g.turn_first[v + 1]
    }

    fn copies_of(&self, v: usize) -> &[(u32, u32)] {
        let lo = self.copies.partition_point(|x| (x.0 as usize) < v);
        let hi = self.copies.partition_point(|x| (x.0 as usize) <= v);
        &self.copies[lo..hi]
    }
}

impl Adjacency for OnOriginals<'_> {
    fn len(&self) -> usize {
        self.g.originals
    }
    fn degree(&self, v: usize) -> u64 {
        let own = self.own(v);
        own.end - own.start + self.copies_of(v).iter().map(|&(_, c)| self.own(c as usize).end - self.own(c as usize).start).sum::<u64>()
    }
    fn head(&self, v: usize, mut i: u64) -> u32 {
        let own = self.own(v);
        let k = own.end - own.start;
        if i < k {
            return self.orig(self.g.turn_to[(own.start + i) as usize]);
        }
        i -= k;
        for &(_, c) in self.copies_of(v) {
            let r = self.own(c as usize);
            if i < r.end - r.start {
                return self.orig(self.g.turn_to[(r.start + i) as usize]);
            }
            i -= r.end - r.start;
        }
        unreachable!("an arc past the node's degree")
    }
}

impl CarGraph {
    /// The turn graph of the pieces `p`. With `dir`, its columns are written to files there
    /// as they are made, then mapped (SPEC §10, memory): while it is built, memory holds a few
    /// bytes per routing node and per segment, the restrictions' sequences and the copies.
    pub fn build(p: &Pieces, turns: &CarTurns, restrictions: &[Restriction], dir: Option<&Path>) -> Result<CarGraph> {
        let phase = crate::build::mem::phase;
        let mut g = CarGraph::default();
        let nodes = p.node_osm.len();
        // The original segments, in piece order.
        let mut piece = ColumnWriter::<u32>::new(dir, "piece0")?;
        let mut way = ColumnWriter::<i64>::new(dir, "way0")?;
        let mut sdir = ColumnWriter::<u8>::new(dir, "dir0")?;
        let mut seq = ColumnWriter::<u32>::new(dir, "seq0")?;
        let mut tail = ColumnWriter::<u32>::new(dir, "tail0")?;
        let mut head = ColumnWriter::<u32>::new(dir, "head0")?;
        let mut time = ColumnWriter::<f32>::new(dir, "time0")?;
        let mut seg_of_piece = ColumnWriter::<[u32; 2]>::new(dir, "seg_of_piece")?;
        let mut left = 0usize;
        for i in 0..p.len() {
            let mut sp = [NONE; 2];
            for d in 0..2 {
                let t = p.time[0][i][d];
                if t.is_finite() {
                    ensure!(piece.len() < NONE as usize, "more than a u32 of CAR segments");
                    sp[d] = piece.len() as u32;
                    piece.push(i as u32)?;
                    way.push(p.way[i])?;
                    sdir.push(d as u8)?;
                    seq.push(p.seq[i])?;
                    let (a, b) = if d == 0 { (p.u[i], p.v[i]) } else { (p.v[i], p.u[i]) };
                    tail.push(a)?;
                    head.push(b)?;
                    time.push(t)?;
                    left += p.left[i] as usize;
                }
            }
            seg_of_piece.push(sp)?;
        }
        (g.piece, g.way, g.dir, g.seq) = (piece.finish()?, way.finish()?, sdir.finish()?, seq.finish()?);
        (g.tail, g.head, g.time, g.seg_of_piece) = (tail.finish()?, head.finish()?, time.finish()?, seg_of_piece.finish()?);
        let s = g.piece.len();
        g.originals = s;
        g.stats.segments = s;
        g.stats.left_segments = left;
        let mut w = round_columns(&g.way, &g.dir, &g.seq, &|k| g.time[k]);
        phase("segments and rounding");

        // Roads at each node: piece ends, a closed piece counting twice; saturated, as a turn
        // only compares it with 2.
        let mut roads = vec![0u8; nodes];
        for i in 0..p.len() {
            for x in [p.u[i], p.v[i]] {
                roads[x as usize] = roads[x as usize].saturating_add(1);
            }
        }
        // Each piece's bearings, [along: (leaving, arriving), against: (leaving, arriving)].
        let mut bw = ColumnWriter::<[f64; 4]>::new(dir, "bearings")?;
        in_ranges(p.len(), |r| r.map(|i| bearings(p, i)).map(|[a, b]| [a.0, a.1, b.0, b.1]).collect::<Vec<_>>(), |b| bw.push_slice(&b))?;
        let bear = bw.finish()?;
        let bear = |k: usize, leaving: bool| -> f64 {
            let x = bear[g.piece[k] as usize];
            let d = 2 * g.dir[k] as usize;
            if leaving { x[d] } else { x[d + 1] }
        };
        // The segments leaving each node, CSR, in segment order.
        let mut out_first = vec![0u32; nodes + 1];
        for &t in g.tail.iter() {
            out_first[t as usize + 1] += 1;
        }
        for i in 0..nodes {
            out_first[i + 1] += out_first[i];
        }
        let mut out = ColumnMut::<u32>::zeroed(dir, "out", s)?;
        for k in 0..s {
            let t = g.tail[k] as usize;
            out[out_first[t] as usize] = k as u32;
            out_first[t] += 1;
        }
        for i in (1..=nodes).rev() {
            out_first[i] = out_first[i - 1];
        }
        out_first[0] = 0;
        let out = out.freeze()?;
        let out_of = |x: usize| &out[out_first[x] as usize..out_first[x + 1] as usize];
        // Base turns, by the segment they leave.
        let (mut first, mut to, mut cost) = (ColumnWriter::<u64>::new(dir, "base_first")?, ColumnWriter::<u32>::new(dir, "base_to")?, ColumnWriter::<u16>::new(dir, "base_cost")?);
        first.push(0)?;
        in_ranges(
            s,
            |r| {
                let (mut t, mut c, mut n, mut dropped) = (Vec::new(), Vec::new(), Vec::with_capacity(r.len()), 0usize);
                for a in r {
                    let x = g.head[a] as usize;
                    for &b in out_of(x) {
                        let b_ = b as usize;
                        let u_turn = g.piece[a] == g.piece[b_] && g.dir[a] != g.dir[b_];
                        if u_turn && roads[x] == 2 {
                            dropped += 1;
                            continue;
                        }
                        let mut angle = bear(b_, true) - bear(a, false);
                        while angle > 180.0 {
                            angle -= 360.0;
                        }
                        while angle <= -180.0 {
                            angle += 360.0;
                        }
                        if u_turn {
                            angle = 180.0;
                        }
                        let s = turn_seconds(turns, &Turn { angle, roads: roads[x] as usize, u_turn, left: p.left[g.piece[a] as usize] }).round();
                        t.push(b);
                        c.push(s.min(65534.0) as u16);
                    }
                    n.push(t.len());
                }
                (t, c, n, dropped)
            },
            |(t, c, n, dropped)| {
                let base = to.len() as u64;
                for k in n {
                    first.push(base + k as u64)?;
                }
                to.push_slice(&t)?;
                cost.push_slice(&c)?;
                g.stats.u_turns_dropped += dropped;
                Ok(())
            },
        )?;
        drop((out, out_first, roads, bear));
        let base = BaseTurns { first: first.finish()?, to: to.finish()?, cost: cost.finish()? };
        phase("base turns");

        let ways = ByWay::new(&p.way);
        let (seqs, of_relation) = g.forbidden_sequences(p, &base, &ways, restrictions);
        drop(ways);
        g.stats.forbidden_sequences = seqs.len();
        let auto = Automaton::new(&seqs);
        phase("forbidden sequences");

        // The originals a restriction begins at: only their turns can lead to a copy.
        let mut restricted = vec![0u64; s.div_ceil(64)];
        for &(a, _) in auto.forbidden_pairs.iter().chain(&auto.prefix_pairs) {
            restricted[a as usize / 64] |= 1 << (a % 64);
        }
        let is_restricted = |a: usize| restricted[a / 64] >> (a % 64) & 1 == 1;
        // The copies: the states reached from the originals, breadth-first, with their turns.
        let mut copies: HashMap<Vec<u32>, u32> = HashMap::new();
        let mut copy_keys: Vec<Vec<u32>> = Vec::new();
        let mut state = |q: Vec<u32>, copy_keys: &mut Vec<Vec<u32>>| -> u32 {
            let next = (s + copy_keys.len()) as u32;
            *copies.entry(q.clone()).or_insert_with(|| {
                copy_keys.push(q);
                next
            })
        };
        for a in (0..s).filter(|&a| is_restricted(a)) {
            for (b, _) in base.of(a) {
                if let Ok(Some(q)) = auto.step(&[a as u32], b) {
                    state(q, &mut copy_keys);
                }
            }
        }
        let mut trans: Vec<(u32, u32, u16)> = Vec::new();
        let mut k = 0;
        while k < copy_keys.len() {
            let pkey = copy_keys[k].clone();
            let from = (s + k) as u32;
            let x = *pkey.last().unwrap() as usize;
            for (b, c) in base.of(x) {
                match auto.step(&pkey, b) {
                    Err(()) => {}
                    Ok(None) => trans.push((from, b, c)),
                    Ok(Some(q)) => {
                        let id = state(q, &mut copy_keys);
                        trans.push((from, id, c));
                    }
                }
            }
            k += 1;
        }
        phase("copies");
        // Copies renumbered by their prefix.
        let mut order: Vec<usize> = (0..copy_keys.len()).collect();
        order.sort_by(|&a, &b| copy_keys[a].cmp(&copy_keys[b]));
        let mut renum = vec![0u32; copy_keys.len()];
        for (new, &old) in order.iter().enumerate() {
            renum[old] = (s + new) as u32;
        }
        let map = |v: u32| if (v as usize) < s { v } else { renum[v as usize - s] };
        let of: Vec<usize> = order.iter().map(|&old| *copy_keys[old].last().unwrap() as usize).collect();
        g.stats.copies = copy_keys.len();
        let mut trans: Vec<(u32, u32, u16)> = trans.into_iter().map(|(a, b, c)| (map(a), map(b), c)).collect();
        trans.sort_unstable();
        // Every turn, sorted by (from, to): the originals' in order, then the copies'.
        let (mut tf, mut tt, mut tc) = (ColumnWriter::<u32>::new(dir, "turn_from")?, ColumnWriter::<u32>::new(dir, "turn_to")?, ColumnWriter::<u16>::new(dir, "turn_cost")?);
        let mut turn_first = ColumnWriter::<u64>::new(dir, "turn_first")?;
        turn_first.push(0)?;
        in_ranges(
            s,
            |r| {
                let (mut f, mut t, mut c, mut n) = (Vec::new(), Vec::new(), Vec::new(), Vec::with_capacity(r.len()));
                let mut row: Vec<(u32, u16)> = Vec::new();
                for a in r {
                    row.clear();
                    if is_restricted(a) {
                        for (b, x) in base.of(a) {
                            match auto.step(&[a as u32], b) {
                                Err(()) => {}
                                Ok(None) => row.push((b, x)),
                                Ok(Some(q)) => row.push((map(copies[&q]), x)),
                            }
                        }
                        row.sort_unstable();
                    } else {
                        row.extend(base.of(a));
                    }
                    for &(b, x) in &row {
                        f.push(a as u32);
                        t.push(b);
                        c.push(x);
                    }
                    n.push(f.len());
                }
                (f, t, c, n)
            },
            |(f, t, c, n)| {
                let at = tf.len() as u64;
                for k in n {
                    turn_first.push(at + k as u64)?;
                }
                tf.push_slice(&f)?;
                tt.push_slice(&t)?;
                tc.push_slice(&c)
            },
        )?;
        let mut exits = vec![false; copy_keys.len()];
        let mut i = 0;
        for c in s..s + copy_keys.len() {
            while i < trans.len() && trans[i].0 as usize == c {
                let (a, b, x) = trans[i];
                tf.push(a)?;
                tt.push(b)?;
                tc.push(x)?;
                exits[c - s] = true;
                i += 1;
            }
            turn_first.push(tf.len() as u64)?;
        }
        g.stats.turns = tf.len();
        (g.turn_from, g.turn_to, g.turn_cost, g.turn_first) = (tf.finish()?, tt.finish()?, tc.finish()?, turn_first.finish()?);
        // Copies with no turn out, and the restrictions whose sequences begin with their prefix.
        for (k, key) in copy_keys.iter().enumerate() {
            let c = renum[k];
            if exits[c as usize - s] {
                continue;
            }
            g.stats.dead_end_copies += 1;
            let lo = of_relation.partition_point(|x| x.0.as_slice() < key.as_slice());
            for (_, rel) in of_relation[lo..].iter().take_while(|x| x.0.starts_with(key)) {
                g.dead_ends.push((*rel, c));
            }
        }
        g.dead_ends.sort_unstable();
        g.dead_ends.dedup();
        // The copies' rows: their original's.
        let at = |c: &Column<u32>| of.iter().map(|&x| c[x]).collect::<Vec<u32>>();
        let (cp, ct, ch, cs) = (at(&g.piece), at(&g.tail), at(&g.head), at(&g.seq));
        let cw: Vec<i64> = of.iter().map(|&x| g.way[x]).collect();
        let cd: Vec<u8> = of.iter().map(|&x| g.dir[x]).collect();
        let cti: Vec<f32> = of.iter().map(|&x| g.time[x]).collect();
        w.extend(of.iter().map(|&x| w[x]).collect::<Vec<u16>>());
        g.piece = std::mem::take(&mut g.piece).append(dir, "piece", &cp)?;
        g.way = std::mem::take(&mut g.way).append(dir, "way", &cw)?;
        g.dir = std::mem::take(&mut g.dir).append(dir, "dir", &cd)?;
        g.seq = std::mem::take(&mut g.seq).append(dir, "seq", &cs)?;
        g.tail = std::mem::take(&mut g.tail).append(dir, "tail", &ct)?;
        g.head = std::mem::take(&mut g.head).append(dir, "head", &ch)?;
        g.time = std::mem::take(&mut g.time).append(dir, "time", &cti)?;
        g.w = Column::from(w);
        g.w.spill(dir, "w")?;
        let mut copy_of = ColumnWriter::<u32>::new(dir, "copy_of")?;
        for _ in 0..s {
            copy_of.push(NONE)?;
        }
        copy_of.push_slice(&of.iter().map(|&x| x as u32).collect::<Vec<u32>>())?;
        g.copy_of = copy_of.finish()?;
        g.base = base;
        g.forbidden = seqs;
        phase("turns");
        Ok(g)
    }

    pub fn len(&self) -> usize {
        self.piece.len()
    }

    /// (original, copy) of every copy, sorted.
    pub fn copies(&self) -> Vec<(u32, u32)> {
        let mut c: Vec<(u32, u32)> = (self.originals..self.len()).map(|c| (self.copy_of[c], c as u32)).collect();
        c.sort_unstable();
        c
    }

    /// Drops what only the order, the hierarchy and the road part read, once the family's road
    /// units are written: the vocabulary and the data report read the rest.
    /// Moves the graph's arrays to files of `dir`, mapped (SPEC §10, memory).
    pub fn spill(&mut self, dir: &Path) -> Result<()> {
        let d = Some(dir);
        self.piece.spill(d, "piece")?;
        self.way.spill(d, "way")?;
        self.dir.spill(d, "dir")?;
        self.seq.spill(d, "seq")?;
        self.tail.spill(d, "tail")?;
        self.head.spill(d, "head")?;
        self.time.spill(d, "time")?;
        self.w.spill(d, "w")?;
        self.copy_of.spill(d, "copy_of")?;
        self.seg_of_piece.spill(d, "seg_of_piece")?;
        self.turn_from.spill(d, "turn_from")?;
        self.turn_to.spill(d, "turn_to")?;
        self.turn_cost.spill(d, "turn_cost")?;
        self.turn_first.spill(d, "turn_first")?;
        let b = &mut self.base;
        b.first.spill(d, "base_first")?;
        b.to.spill(d, "base_to")?;
        b.cost.spill(d, "base_cost")?;
        Ok(())
    }

    pub fn shed(&mut self) {
        (self.tail, self.head, self.time) = Default::default();
        (self.turn_from, self.turn_to, self.turn_cost, self.turn_first) = Default::default();
        (self.base, self.forbidden) = Default::default();
    }

    /// The restrictions as forbidden sequences of original segments (FORMAT §5.3): length 2
    /// for a via node, at least 3 for via ways; `only_*` rewritten as every other exit.
    /// The forbidden sequences, sorted, and every (sequence, relation id), sorted.
    fn forbidden_sequences(&mut self, p: &Pieces, base: &BaseTurns, by_way: &ByWay, restrictions: &[Restriction]) -> (Vec<Vec<u32>>, Vec<(Vec<u32>, i64)>) {
        let node = |osm: i64| p.osm_node(osm);
        let mut out: Vec<Vec<u32>> = Vec::new();
        let mut of_relation: Vec<(Vec<u32>, i64)> = Vec::new();
        for r in restrictions {
            let mut seqs: Vec<Vec<u32>> = Vec::new();
            // (prefix, allowed last segments) per from way: the sequences end with one turn.
            let mut prefixes: Vec<(Vec<u32>, Vec<u32>)> = Vec::new();
            match &r.via {
                Via::Node(n) => {
                    let Some(n) = node(*n) else {
                        self.stats.unresolved += 1;
                        continue;
                    };
                    let ins = |w: &i64| -> Vec<u32> {
                        self.segments_of(by_way, *w).into_iter().filter(|&s| self.head[s as usize] == n).collect()
                    };
                    let outs = |w: &i64| -> Vec<u32> {
                        self.segments_of(by_way, *w).into_iter().filter(|&s| self.tail[s as usize] == n).collect()
                    };
                    let to: Vec<u32> = r.to.iter().flat_map(outs).collect();
                    for a in r.from.iter().flat_map(ins) {
                        prefixes.push((vec![a], to.clone()));
                    }
                }
                Via::Ways(vias) => {
                    for f in &r.from {
                        for t in &r.to {
                            if let Some((pre, last)) = self.via_chain(p, by_way, *f, vias, *t) {
                                prefixes.push((pre, vec![last]));
                            }
                        }
                    }
                }
            }
            for (pre, to) in prefixes {
                let x = *pre.last().unwrap() as usize;
                for (b, _) in base.of(x) {
                    if to.contains(&b) != r.only {
                        let mut s = pre.clone();
                        s.push(b);
                        seqs.push(s);
                    }
                }
            }
            if seqs.is_empty() {
                self.stats.unresolved += 1;
                continue;
            }
            match r.via {
                Via::Node(_) => self.stats.node_via_applied += 1,
                Via::Ways(_) => self.stats.way_via_applied += 1,
            }
            of_relation.extend(seqs.iter().map(|q| (q.clone(), r.id)));
            out.extend(seqs);
        }
        out.sort();
        out.dedup();
        of_relation.sort();
        of_relation.dedup();
        (out, of_relation)
    }

    fn segments_of(&self, by_way: &ByWay, way: i64) -> Vec<u32> {
        by_way.pieces(way).into_iter().flat_map(|i| self.seg_of_piece[i]).filter(|&s| s != NONE).collect()
    }

    /// The segments of `from` into the via chain, along the via ways, and the segment of
    /// `to` out of it: (from and via segments, to segment). None if the ways do not chain
    /// or a direction is not drivable.
    fn via_chain(&self, p: &Pieces, by_way: &ByWay, from: i64, vias: &[i64], to: i64) -> Option<(Vec<u32>, u32)> {
        let pieces = |w: i64| -> Option<Vec<usize>> { Some(by_way.pieces(w)).filter(|v| !v.is_empty()) };
        let ends = |w: i64| -> Option<(u32, u32)> {
            let v = pieces(w)?;
            Some((p.u[v[0]], p.v[*v.last().unwrap()]))
        };
        let (fa, fb) = ends(from)?;
        for start in [fb, fa] {
            // The from way's piece at `start`, arriving there.
            let fp = pieces(from)?;
            let first = if start == fb { self.seg_of_piece[*fp.last().unwrap()][0] } else { self.seg_of_piece[fp[0]][1] };
            if first == NONE {
                continue;
            }
            let mut seq = vec![first];
            let mut at = start;
            let mut left: Vec<i64> = vias.to_vec();
            let mut ok = true;
            while !left.is_empty() {
                let Some(k) = left.iter().position(|&w| ends(w).is_some_and(|(a, b)| a == at || b == at)) else {
                    ok = false;
                    break;
                };
                let w = left.remove(k);
                let (a, b) = ends(w)?;
                let ps = pieces(w)?;
                if a == at {
                    seq.extend(ps.iter().map(|&i| self.seg_of_piece[i][0]));
                    at = b;
                } else {
                    seq.extend(ps.iter().rev().map(|&i| self.seg_of_piece[i][1]));
                    at = a;
                }
            }
            if !ok || seq.contains(&NONE) {
                continue;
            }
            let (ta, tb) = ends(to)?;
            let tp = pieces(to)?;
            let last = if ta == at {
                self.seg_of_piece[tp[0]][0]
            } else if tb == at {
                self.seg_of_piece[*tp.last().unwrap()][1]
            } else {
                continue;
            };
            if last != NONE {
                return Some((seq, last));
            }
        }
        None
    }

    /// The components of the turn graph (SPEC §5), determined by the original segments only:
    /// the strongly connected components of the originals, a turn into or out of a copy
    /// counting as one into or out of its original; each copy is in its original's component,
    /// whatever its own connectivity. Ties go to the smallest (way, way_seq, way_dir), then
    /// originals before copies.
    pub fn components(&self) -> Components {
        let copies = self.copies();
        let mut comp = scc(&OnOriginals { g: self, copies: &copies });
        let of_copies: Vec<u32> = (self.originals..self.len()).map(|c| comp[self.copy_of[c] as usize]).collect();
        comp.extend(of_copies);
        let joined = self
            .turn_from
            .iter()
            .copied()
            .zip(self.turn_to.iter().copied())
            .chain(self.seg_of_piece.iter().filter(|s| s[0] != NONE && s[1] != NONE).map(|s| (s[0], s[1])))
            .chain((self.originals..self.len()).map(|c| (c as u32, self.copy_of[c])));
        Components::new(&comp, |v| (self.way[v], self.seq[v], self.dir[v], v), joined)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile::car::{CarProfile, WayAttrs};

    fn graph(name: &str) -> (Pieces, CarGraph) {
        let dir = std::env::temp_dir().join(format!("blobmap-graph-car-{}-{name}", std::process::id()));
        let w = crate::fixtures::by_name(name).unwrap().write(&dir).unwrap();
        let cfg = crate::profile::config::Config::load(&w.configs[0].1).unwrap();
        let osm = crate::osm::Osm::open(&w.pbf, &dir.join("nodes"), &cfg).unwrap();
        let net = crate::osm::Network::read(&osm, &osm.car_obstacles, &CarProfile::new(&cfg.car.config), &cfg.car.config.snap, None).unwrap();
        let p = net.pieces(&[|a: &WayAttrs| *a]).unwrap();
        let (r, _) = osm.restrictions().unwrap();
        let g = CarGraph::build(&p, &cfg.car.config.turns, &r, None).unwrap();
        (p, g)
    }

    #[test]
    fn left_hand_turns_mirror() {
        let c = crate::profile::CarTurns { turn_penalty_s: 7.5, turn_bias: 1.075, u_turn_penalty_s: 20.0 };
        let t = |angle: f64, left: bool| turn_seconds(&c, &Turn { angle, roads: 3, u_turn: false, left });
        for a in [10.0, 45.0, 90.0, 135.0, 170.0] {
            assert!((t(a, false) - t(-a, true)).abs() < 1e-12 && (t(-a, false) - t(a, true)).abs() < 1e-12);
        }
        // Keeping right, a right turn is the cheaper; keeping left, a left one.
        assert!(t(90.0, false) < t(-90.0, false) && t(-90.0, true) < t(90.0, true));
    }

    #[test]
    fn viaway_copies() {
        let (_, g) = graph("viaway");
        assert_eq!(g.stats.way_via_applied, 4);
        assert_eq!(g.stats.unresolved, 0);
        assert!(g.stats.copies >= 4, "{:?}", g.stats);
        // Copies take their original's weight and key.
        for c in g.originals..g.len() {
            let o = g.copy_of[c] as usize;
            assert_eq!((g.w[c], g.way[c], g.seq[c], g.dir[c]), (g.w[o], g.way[o], g.seq[o], g.dir[o]));
        }
        let e = g.components();
        assert_eq!(e.nodes[0] as usize, g.len(), "one component: {:?}", e.nodes);
    }

    #[test]
    fn town_turns_and_components() {
        let (p, g) = graph("town");
        assert_eq!(g.stats.node_via_applied, 2);
        assert_eq!(g.stats.copies, 0);
        // Row 0 (way 1): 14.4 s pieces, rounded per way to 15, 15, 14, 14, 14.
        let row0: Vec<u16> = (0..g.originals).filter(|&k| g.way[k] == 1 && g.dir[k] == 0).map(|k| g.w[k]).collect();
        assert_eq!((row0.len(), row0.iter().sum::<u16>()), (5, 72));
        assert!(row0.iter().all(|&w| w == 14 || w == 15));
        let e = g.components();
        assert!(e.trapped.iter().filter(|&&t| t).count() >= 2);
        assert!(!e.trapped[0]);
        // The main grid is component 0, the island and the east bank are served islands.
        assert!(e.nodes[0] > 100);
        let _ = p;
    }
}
