//! The downward half of a sweep, given the upward half's labels: the column's job a backend
//! runs. Shared semantics, the CPU's and the GPU's alike:
//!
//! - a node's label is the least of its upward label (when the upward half reached it) and
//!   of each upper neighbour's label ⊕ the link's weight, saturating at `INF` (FORMAT §8);
//! - a column may carry a bound T < `DEAD` (pruning, for answers that only compare times
//!   with T): every label above T is then `INF`, and an unreached node whose entry bound
//!   (`Graph::entry_bounds`) is above T, and every node below it in the elimination tree, is
//!   `DEAD`, without reading its row. A head of a `DEAD` or `INF` label is above T, so every
//!   answer that compares times with T is the unpruned one.

use anyhow::Result;
use blobmap_model::INF;
use rayon::prelude::*;

use crate::graph::Graph;
use crate::vocab::Dir;

/// The label of a pruned node; also the bound that turns pruning off.
pub const DEAD: u16 = 0xFFFE;
/// Levels at least this wide are swept in parallel on the CPU.
const PAR_LEVEL: u32 = 1 << 14;

/// One column of a sweep: the upward half's labels (new id, label), ascending, and the bound.
#[derive(Clone, Debug)]
pub struct Column {
    pub entries: Vec<(u32, u16)>,
    pub bound: u16,
}

impl Column {
    pub fn pruned(&self) -> bool {
        self.bound < DEAD
    }
}

/// What a sweep runs on: a component's graph and one metric and direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Key {
    pub fam: u8,
    pub component: u32,
    pub metric: u8,
    pub dir: u8,
}

/// The entry bounds of a graph, per metric and direction (index `2 * metric + dir`).
pub type Bounds = Vec<Vec<u16>>;

pub fn dir_index(d: Dir) -> usize {
    (d == Dir::To) as usize
}

pub fn all_bounds(g: &Graph) -> Result<Bounds> {
    let mut out = Vec::new();
    for m in 0..g.down.len() {
        for d in [Dir::From, Dir::To] {
            out.push(g.entry_bounds(m, d)?);
        }
    }
    Ok(out)
}

/// The downward half on the CPU: labels by new id.
pub fn cpu_down(g: &Graph, metric: usize, dir: Dir, c: &Column, bounds: Option<&[u16]>) -> Vec<u16> {
    let w = g.going_down(metric, dir);
    let mut lab = vec![INF; g.n];
    let mut seeded = vec![false; g.n];
    for &(v, t) in &c.entries {
        lab[v as usize] = t;
        seeded[v as usize] = true;
    }
    let prune = c.pruned() && bounds.is_some();
    let t = c.bound;
    for l in 0..g.levels() {
        let (a, b) = (g.level_first[l] as usize, g.level_first[l + 1] as usize);
        let (done, rest) = lab.split_at_mut(a);
        let cur = &mut rest[..b - a];
        let node = |i: usize, out: &mut u16| {
            let v = a + i;
            if prune {
                let p = g.parent[v];
                if p != blobmap_model::NONE && done[p as usize] == DEAD {
                    *out = DEAD;
                    return;
                }
            }
            let mut acc = *out;
            let mut lb = INF;
            for k in g.row(v as u32) {
                let x = done[g.up_head[k] as usize];
                acc = acc.min(x.saturating_add(w[k]));
                if prune {
                    lb = lb.min(x.saturating_add(bounds.unwrap()[k]));
                }
            }
            if prune {
                if acc > t {
                    acc = INF;
                }
                if !seeded[v] && lb > t {
                    acc = DEAD;
                }
            }
            *out = acc;
        };
        if (b - a) as u32 >= PAR_LEVEL {
            cur.par_iter_mut().enumerate().with_min_len(4096).for_each(|(i, o)| node(i, o));
        } else {
            cur.iter_mut().enumerate().for_each(|(i, o)| node(i, o));
        }
    }
    lab
}
