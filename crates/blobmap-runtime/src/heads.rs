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

/// Dense times read back from a device: per chunk, its first index and `p` words an index.
#[derive(Debug)]
pub struct PackedOut {
    pub per: u32,
    pub p: usize,
    pub chunks: Vec<(u32, Vec<u32>)>,
}

/// A column's answer: dense over `lo ..` (`INF` for BEYOND), as its own vector or as column
/// `c` of a device's read-back over `lo .. hi` (with one index's time replaced, `patch`); or
/// the (index, time) pairs at most the wanted limit, ascending by index.
#[derive(Clone, Debug)]
pub enum Times {
    Dense { lo: u32, t: Vec<u16> },
    Packed { lo: u32, hi: u32, out: Arc<PackedOut>, c: usize, patch: Option<(u32, u16)> },
    Sparse(Vec<(u32, u16)>),
}

impl Times {
    pub fn get(&self, i: u32) -> u16 {
        match self {
            Times::Dense { lo, t } => i.checked_sub(*lo).and_then(|k| t.get(k as usize)).copied().unwrap_or(INF),
            Times::Packed { lo, hi, out, c, patch } => {
                if let Some((j, t)) = patch
                    && *j == i
                {
                    return *t;
                }
                if i < *lo || i >= *hi {
                    return INF;
                }
                let k = (i / out.per - out.chunks[0].0 / out.per) as usize;
                let (s, w) = &out.chunks[k];
                (w[(i - s) as usize * out.p + c / 2] >> (16 * (c % 2))) as u16
            }
            Times::Sparse(v) => v.binary_search_by_key(&i, |e| e.0).map_or(INF, |k| v[k].1),
        }
    }

    /// Replaces index i's time (in a dense answer, or a sparse one).
    pub fn set(&mut self, i: u32, t: u16) {
        match self {
            Times::Dense { lo, t: v } => v[(i - *lo) as usize] = t,
            Times::Packed { patch, .. } => *patch = Some((i, t)),
            Times::Sparse(v) => match v.binary_search_by_key(&i, |e| e.0) {
                Ok(k) => v[k].1 = t,
                Err(k) => v.insert(k, (i, t)),
            },
        }
    }

    /// Every time of indices `0 .. n` in order (`INF` outside a dense range).
    pub fn for_each(&self, n: u32, mut f: impl FnMut(u16)) {
        match self {
            Times::Packed { lo, hi, out, c, patch } => {
                let (q, sh) = (c / 2, 16 * (c % 2));
                for _ in 0..(*lo).min(n) {
                    f(INF);
                }
                for (s, w) in &out.chunks {
                    for (k, x) in w.chunks_exact(out.p).enumerate() {
                        let j = s + k as u32;
                        if j >= n {
                            return;
                        }
                        f(match patch {
                            Some((pi, pt)) if *pi == j => *pt,
                            _ => (x[q] >> sh) as u16,
                        });
                    }
                }
                for _ in (*hi).max(*lo).min(n)..n {
                    f(INF);
                }
            }
            _ => (0..n).for_each(|i| f(self.get(i))),
        }
    }

    /// The indices with a finite time, ascending.
    pub fn finite(&self) -> Vec<u32> {
        match self {
            Times::Dense { lo, t } => t.iter().enumerate().filter(|e| *e.1 != INF).map(|(k, _)| lo + k as u32).collect(),
            Times::Packed { hi, .. } => {
                let mut v = Vec::new();
                let mut i = 0u32;
                self.for_each(*hi, |t| {
                    if t != INF {
                        v.push(i);
                    }
                    i += 1;
                });
                v
            }
            Times::Sparse(v) => v.iter().filter(|e| e.1 != INF).map(|e| e.0).collect(),
        }
    }

    /// The (index, time) pairs at most `max` in `lo .. hi`.
    pub fn sparse(&self, lo: u32, hi: u32, max: u16) -> Vec<(u32, u16)> {
        let mut v = Vec::new();
        let mut i = 0u32;
        self.for_each(hi, |t| {
            if i >= lo && i < hi && t <= max {
                v.push((i, t));
            }
            i += 1;
        });
        v
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
    /// The component's IDs, ascending, for a small component; else every ID whose layout
    /// ids are the component's.
    pub ids: Option<&'a [u32]>,
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
    // The component's IDs in lo .. hi: from its list, or those whose layout id is in it.
    let n = g.n as u32;
    let member = |i: u32| {
        let nd = mv.raw_node(i as usize);
        let v = if nd[0] != NONE { nd[0] } else { nd[1] };
        v != NONE && v >= first && v - first < n
    };
    let each = |lo: u32, hi: u32, f: &mut dyn FnMut(u32)| match p.ids {
        Some(ids) => ids[ids.partition_point(|&i| i < lo)..ids.partition_point(|&i| i < hi)].iter().for_each(|&i| f(i)),
        None => (lo..hi).filter(|&i| member(i)).for_each(f),
    };
    if let Some(out) = out {
        out.par_chunks_mut(CHUNK).enumerate().for_each(|(c, o)| {
            let lo = want.lo + (c * CHUNK) as u32;
            each(lo, lo + o.len() as u32, &mut |i| o[(i - lo) as usize] = time(i as usize));
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
    let chunks = (want.hi.saturating_sub(want.lo) as usize).div_ceil(CHUNK);
    let parts: Vec<Vec<(u32, u16)>> = (0..chunks)
        .into_par_iter()
        .map(|c| {
            let lo = want.lo + (c * CHUNK) as u32;
            let hi = (lo + CHUNK as u32).min(want.hi);
            let mut v = Vec::new();
            each(lo, hi, &mut |i| {
                let nd = mv.raw_node(i as usize);
                if !is_near(nd[0]) && !is_near(nd[1]) && !on_source(nd) {
                    return;
                }
                let t = time(i as usize);
                if t <= max {
                    v.push((i, t));
                }
            });
            v
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
