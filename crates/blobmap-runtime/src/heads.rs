//! Heads (FORMAT §8): a column's time at each ID it is asked for, from its sweep's labels.
//! Two streaming passes over one component:
//!
//! - **nodes**, in layout order: h(v) is, for CAR `From`, the least label over v's arrivals
//!   (v and its copies); for CAR `To`, the least exit turn of v ⊕ its head's label; for
//!   ACTIVE, v's label;
//! - **IDs**, in internal-index order, only those in the wanted range: the least of
//!   h(node k) ⊕ offset `2k` (going from) or `2k + 1` (going to), the direct term from or to
//!   a source on the same piece, and a source's own start.
//!
//! Saturating addition is monotone and associative, so the sums are the reference's.
//! When only times up to a limit are wanted, an ID is skipped unread unless one of its
//! nodes has h at most that limit (a bit per node) or it shares a source's nodes.

use std::sync::Arc;

use blobmap_model::{INF, NONE};
use rayon::prelude::*;

use crate::graph::Graph;
use crate::runtime::{Src, direct};
use crate::sweep::DEAD;
use crate::vocab::{Dir, Fam, Mode};

/// The IDs a column answers: internal indices `lo .. hi`, and of those only the times at
/// most `max` (`INF`: every time, dense).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Want {
    pub lo: u32,
    pub hi: u32,
    pub max: u16,
}

impl Want {
    pub fn all(n: usize) -> Want {
        Want { lo: 0, hi: n as u32, max: INF }
    }

    pub fn range(lo: usize, hi: usize) -> Want {
        Want { lo: lo as u32, hi: hi as u32, max: INF }
    }

    /// Only the times at most t (every finite time when t is not below `INF`).
    pub fn within(self, t: u64) -> Want {
        Want { max: t.min(INF as u64 - 1) as u16, ..self }
    }

    pub fn dense(&self) -> bool {
        self.max == INF
    }
}

/// A column's answer: dense over `lo ..` (`INF` for BEYOND), or the (index, time) pairs at
/// most the wanted limit, ascending by index.
#[derive(Clone, Debug)]
pub enum Times {
    Dense { lo: u32, t: Vec<u16> },
    Sparse(Vec<(u32, u16)>),
}

impl Times {
    pub fn get(&self, i: u32) -> u16 {
        match self {
            Times::Dense { lo, t } => i.checked_sub(*lo).and_then(|k| t.get(k as usize)).copied().unwrap_or(INF),
            Times::Sparse(v) => v.binary_search_by_key(&i, |e| e.0).map_or(INF, |k| v[k].1),
        }
    }

    /// The dense times; a sparse answer has none.
    pub fn into_dense(self) -> Vec<u16> {
        match self {
            Times::Dense { t, .. } => t,
            Times::Sparse(_) => panic!("a sparse column has no dense times"),
        }
    }

    /// The indices with a finite time, ascending.
    pub fn finite(&self) -> Vec<u32> {
        match self {
            Times::Dense { lo, t } => t.iter().enumerate().filter(|e| *e.1 != INF).map(|(k, _)| lo + k as u32).collect(),
            Times::Sparse(v) => v.iter().filter(|e| e.1 != INF).map(|e| e.0).collect(),
        }
    }
}

/// A sweep's labels by new id: one column's own, or one column of a batch read back from a
/// device (u16 pairs, `p` words a node, column `c`).
pub enum Labels {
    Plain(Vec<u16>),
    Packed { words: Arc<Vec<u32>>, p: usize, c: usize },
}

/// One column on one component: what the heads read.
pub struct Part<'a> {
    pub g: &'a Graph,
    pub mv: &'a Mode,
    pub x: u32,
    /// The component's IDs, ascending.
    pub ids: &'a [u32],
    pub srcs: &'a [Src],
    pub dir: Dir,
    /// The sweep's pruning bound (`DEAD` or above: none).
    pub bound: u16,
}

/// Rows of the ID pass: this many indices a task.
const CHUNK: usize = 1 << 15;

/// The heads of one part. Dense wants write `out` (indices `want.lo ..`); sparse ones
/// return their pairs.
pub fn heads(p: &Part, lab: &Labels, want: Want, out: Option<&mut [u16]>) -> Vec<(u32, u16)> {
    match lab {
        Labels::Plain(l) => run(p, |v| l[v], want, out),
        Labels::Packed { words, p: pw, c } => {
            let w: &[u32] = words;
            let (q, sh) = (c / 2, 16 * (c % 2));
            run(p, |v| (w[v * pw + q] >> sh) as u16, want, out)
        }
    }
}

fn run<L: Fn(usize) -> u16 + Sync>(p: &Part, lab: L, want: Want, out: Option<&mut [u16]>) -> Vec<(u32, u16)> {
    let (g, mv) = (p.g, p.mv);
    let h = node_pass(g, p.dir, &lab);
    let first = mv.first(p.x);
    let koff = (p.dir == Dir::To) as usize;
    let pruned = p.bound < DEAD;
    // Sources by piece: (layout ids, road, source).
    let mut by_piece: Vec<([u32; 2], u32, usize)> = p.srcs.iter().enumerate().map(|(j, s)| (mv.raw_node(s.id as usize), mv.road(s.id as usize), j)).collect();
    by_piece.sort_unstable();
    let time = |i: usize| -> u16 {
        let (nd, off) = (mv.raw_node(i), mv.off(i));
        let mut t = INF;
        for k in 0..2 {
            if nd[k] != NONE {
                t = t.min(h[(nd[k] - first) as usize].saturating_add(off[2 * k + koff]));
            }
        }
        if !by_piece.is_empty() {
            let key = (nd, mv.road(i));
            let from = by_piece.partition_point(|b| (b.0, b.1) < key);
            for &(_, _, j) in by_piece[from..].iter().take_while(|b| (b.0, b.1) == key) {
                let s = &p.srcs[j];
                let si = s.id as usize;
                if si == i {
                    t = t.min(s.start);
                } else {
                    let d = if p.dir == Dir::From { direct(mv, si, i) } else { direct(mv, i, si) };
                    t = t.min(s.start.saturating_add(d));
                }
            }
        }
        if pruned && t > p.bound { INF } else { t }
    };
    let span = |lo: u32, hi: u32| &p.ids[p.ids.partition_point(|&i| i < lo)..p.ids.partition_point(|&i| i < hi)];
    if let Some(out) = out {
        out.par_chunks_mut(CHUNK).enumerate().for_each(|(c, o)| {
            let lo = want.lo + (c * CHUNK) as u32;
            for &i in span(lo, lo + o.len() as u32) {
                o[(i - lo) as usize] = time(i as usize);
            }
        });
        return Vec::new();
    }
    // Sparse: a bit per node with h at most the limit; the IDs sharing a source's nodes.
    let max = want.max;
    let mut near = vec![0u64; g.n.div_ceil(64)];
    near.par_iter_mut().enumerate().for_each(|(w, b)| {
        let v0 = w * 64;
        for (j, &x) in h[v0..(v0 + 64).min(g.n)].iter().enumerate() {
            *b |= ((x <= max) as u64) << j;
        }
    });
    let is_near = |v: u32| v != NONE && (near[((v - first) >> 6) as usize] >> ((v - first) & 63)) & 1 == 1;
    let on_source = |nd: [u32; 2]| {
        let k = by_piece.partition_point(|b| b.0 < nd);
        k < by_piece.len() && by_piece[k].0 == nd
    };
    let ids = span(want.lo, want.hi);
    let parts: Vec<Vec<(u32, u16)>> = ids
        .par_chunks(CHUNK)
        .map(|ch| {
            ch.iter()
                .filter_map(|&i| {
                    let nd = mv.raw_node(i as usize);
                    if !is_near(nd[0]) && !is_near(nd[1]) && !on_source(nd) {
                        return None;
                    }
                    let t = time(i as usize);
                    (t <= max).then_some((i, t))
                })
                .collect()
        })
        .collect();
    parts.concat()
}

/// h(v) for every layout node of the component (numbered from 0).
fn node_pass<L: Fn(usize) -> u16 + Sync>(g: &Graph, dir: Dir, lab: &L) -> Vec<u16> {
    let mut h = vec![INF; g.n];
    let at = |v: u32| lab(g.new[v as usize] as usize);
    h.par_chunks_mut(CHUNK).enumerate().for_each(|(c, o)| {
        let v0 = (c * CHUNK) as u32;
        for (j, x) in o.iter_mut().enumerate() {
            let v = v0 + j as u32;
            *x = match (g.fam, dir) {
                (Fam::Car, Dir::From) => g.arrivals(v).map(at).min().unwrap_or(INF),
                (Fam::Car, Dir::To) => g.arcs(v).map(|a| g.turn_cost[a].saturating_add(at(g.arc_head[a]))).min().unwrap_or(INF),
                (Fam::Active, _) => at(v),
            };
        }
    });
    h
}
