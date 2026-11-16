//! The trial's GPU layout of one component and metric, shared by both backends so that they
//! sweep the same bytes: nodes renumbered top-first by longest-path level (as the research
//! kernels and the milestone 5 smoke test require), rows sorted, weights for both directions.
//! Also the CPU upward search that turns a column's seeds into the entries a GPU sweep starts
//! from, and the degree split tables of the wide levels.

use std::collections::BinaryHeap;

use blobmap_model::INF;
use blobmap_model::runtime::{Component, Dir};

/// Levels of at most this many nodes, from the top, form the narrow top (research default).
pub const TOP_CAP: u32 = 512;
/// Shared memory a resident top may use: the CUDA opt-in limit of the research driver, and
/// the 48 KiB of static shared memory a CubeCL kernel gets.
pub const RESIDENT_OPTIN: usize = 101_376;
pub const RESIDENT_STATIC: usize = 48 * 1024;

pub struct Layered {
    pub n: usize,
    pub level_first: Vec<u32>,
    pub up_first: Vec<u32>,
    pub up_head: Vec<u32>,
    /// Weights going down: `w_fwd` for `From` (the metric's `down`), `w_rev` for `To` (`up`).
    pub w_fwd: Vec<u16>,
    pub w_rev: Vec<u16>,
    /// New id of each component-local id, and back.
    pub new: Vec<u32>,
    pub old: Vec<u32>,
    /// Levels of the narrow top, fitted to both kernels' shared memory.
    pub top: usize,
}

impl Layered {
    pub fn new(e: &Component, metric: usize) -> Layered {
        let n = e.n;
        let mut depth = vec![0u32; n];
        for v in 0..n {
            let mut d = 0;
            for k in e.row(v as u32) {
                d = d.max(depth[e.up_head[k] as usize] + 1);
            }
            depth[v] = d;
        }
        let levels = depth.iter().max().map_or(0, |&d| d as usize + 1);
        let mut level_first = vec![0u32; levels + 1];
        for &d in &depth {
            level_first[d as usize + 1] += 1;
        }
        for l in 0..levels {
            level_first[l + 1] += level_first[l];
        }
        let mut next = level_first.clone();
        let (mut new, mut old) = (vec![0u32; n], vec![0u32; n]);
        for v in 0..n {
            let d = depth[v] as usize;
            new[v] = next[d];
            old[next[d] as usize] = v as u32;
            next[d] += 1;
        }
        let (down, up) = (&e.down[metric], &e.up[metric]);
        let m = e.up_head.len();
        assert!(m < u32::MAX as usize, "links exceed u32");
        let mut up_first = Vec::with_capacity(n + 1);
        let (mut up_head, mut w_fwd, mut w_rev) = (Vec::with_capacity(m), Vec::with_capacity(m), Vec::with_capacity(m));
        up_first.push(0u32);
        let mut row: Vec<(u32, u16, u16)> = Vec::new();
        for i in 0..n {
            let v = old[i];
            row.clear();
            for k in e.row(v) {
                row.push((new[e.up_head[k] as usize], down[k], up[k]));
            }
            row.sort_unstable();
            for &(h, f, r) in &row {
                debug_assert!((h as usize) < i);
                up_head.push(h);
                w_fwd.push(f);
                w_rev.push(r);
            }
            up_first.push(up_head.len() as u32);
        }
        let mut c = Layered { n, level_first, up_first, up_head, w_fwd, w_rev, new, old, top: 0 };
        let mut top = (0..levels).take_while(|&l| c.width(l) <= TOP_CAP).count();
        while top > 0 && (c.resident_bytes(top) > RESIDENT_OPTIN.min(RESIDENT_STATIC) || c.level_first[top] > 65536) {
            top -= 1;
        }
        c.top = top;
        c
    }

    pub fn levels(&self) -> usize {
        self.level_first.len() - 1
    }

    pub fn width(&self, l: usize) -> u32 {
        self.level_first[l + 1] - self.level_first[l]
    }

    pub fn max_top_level(&self) -> u32 {
        (0..self.top).map(|l| self.width(l)).max().unwrap_or(0)
    }

    /// The research kernel's resident shared memory for a top of `top` levels.
    pub fn resident_bytes(&self, top: usize) -> usize {
        let maxw = (0..top).map(|l| self.width(l) as usize).max().unwrap_or(0);
        let nodes = self.level_first[top] as usize;
        (2 * (top + 1) + 2 * maxw) * 4 + (nodes * 2).div_ceil(4) * 4
    }

    /// Top arcs packed as `head | tail's index in its level << 16` (research `top_arc`).
    pub fn top_arcs(&self) -> Vec<u32> {
        let top_nodes = self.level_first[self.top];
        let arcs = self.up_first[top_nodes as usize] as usize;
        let mut packed = vec![0u32; arcs.max(1)];
        for l in 0..self.top {
            for v in self.level_first[l]..self.level_first[l + 1] {
                for k in self.up_first[v as usize]..self.up_first[v as usize + 1] {
                    packed[k as usize] = self.up_head[k as usize] | ((v - self.level_first[l]) << 16);
                }
            }
        }
        packed
    }

    /// The research degree split of the wide levels at lane width `lp` and threshold `t`:
    /// per wide level (first heavy node, heavy nodes per class of 32, 16, 8, 4, 2 groups), and
    /// the heavy nodes by decreasing degree.
    pub fn split(&self, lp: u32, t: u32) -> (Vec<(u32, [u32; 5])>, Vec<u32>) {
        let mut heavy = Vec::new();
        let mut out = Vec::new();
        for l in self.top..self.levels() {
            let first = heavy.len() as u32;
            let mut count = [0u32; 5];
            if t != 0 && (self.width(l) as u64) * (lp as u64) <= 65536 {
                let mut h: Vec<(u32, u32)> = (self.level_first[l]..self.level_first[l + 1])
                    .map(|v| (self.up_first[v as usize + 1] - self.up_first[v as usize], v))
                    .filter(|&(d, _)| d > t)
                    .collect();
                h.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
                for &(d, _) in &h {
                    let mut groups = 2u32;
                    while groups < 32 && d.div_ceil(groups) > t {
                        groups <<= 1;
                    }
                    count[(32 / groups).trailing_zeros() as usize] += 1;
                }
                heavy.extend(h.iter().map(|x| x.1));
            }
            out.push((first, count));
        }
        (out, heavy)
    }

    /// The CPU downward sweep on this layout, from seed entries (new ids): the check of a GPU
    /// sweep.
    pub fn cpu_down(&self, entries: &[Entry], dir: Dir) -> Vec<u16> {
        let w = if dir == Dir::From { &self.w_fwd } else { &self.w_rev };
        let mut lab = vec![INF; self.n];
        for e in entries {
            lab[e.node as usize] = e.value.min(INF as u32) as u16;
        }
        for v in 0..self.n {
            let mut l = lab[v];
            for k in self.up_first[v] as usize..self.up_first[v + 1] as usize {
                l = l.min(lab[self.up_head[k] as usize].saturating_add(w[k]));
            }
            lab[v] = l;
        }
        lab
    }
}

/// A seed of a GPU sweep (research `Entry`): new id, column, label.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Entry {
    pub node: u32,
    pub col: u32,
    pub value: u32,
}

/// The upward half of a sweep on the CPU, from seeds (component-local id, label): the finite
/// labels of the nodes the seeds reach over upper links, as entries of column `col` in the
/// new ids.
pub fn upward(e: &Component, c: &Layered, metric: usize, seeds: &[(u32, u16)], dir: Dir, col: u32) -> Vec<Entry> {
    let go_up = match dir {
        Dir::From => &e.up[metric],
        Dir::To => &e.down[metric],
    };
    let mut lab: std::collections::HashMap<u32, u16> = std::collections::HashMap::new();
    let mut heap = BinaryHeap::new();
    for &(v, t) in seeds {
        let l = lab.entry(v).or_insert(INF);
        if t < *l {
            if *l == INF {
                heap.push(v);
            }
            *l = t;
        }
    }
    let mut out = Vec::new();
    while let Some(v) = heap.pop() {
        let lv = lab[&v];
        out.push(Entry { node: c.new[v as usize], col, value: lv as u32 });
        for k in e.row(v) {
            let u = e.up_head[k];
            let x = lv.saturating_add(go_up[k]);
            let l = lab.entry(u).or_insert(INF);
            if x < *l {
                if *l == INF {
                    heap.push(u);
                }
                *l = x;
            }
        }
    }
    out.sort_by_key(|e| e.node);
    out
}

/// Pseudo-random nodes of a component, fixed seed: the trial's sources.
pub fn sources(n: usize, count: usize) -> Vec<u32> {
    let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
    (0..count)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x % n as u64) as u32
        })
        .collect()
}
