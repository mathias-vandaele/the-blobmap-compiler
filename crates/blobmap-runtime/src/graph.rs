//! One component of a family, as the runtime sweeps it: its hierarchy (hot) renumbered
//! top-first by the format's levels (FORMAT §5.2), so that a level is a set of nodes a
//! downward sweep can settle at once, with rows sorted; and what the heads read (warm): CAR exit
//! turns and via-way copies, in the component's own layout ids (FORMAT §5, §8).

use std::collections::BinaryHeap;

use anyhow::{Result, ensure};
use blobmap_model::{INF, Model, NONE};

use crate::vocab::{Dir, Fam, Family};

pub struct Graph {
    pub fam: Fam,
    pub n: usize,
    /// Level l is new ids `level_first[l] .. level_first[l + 1]`; level 0 has no upper
    /// neighbour, and every upper neighbour of a node is in a smaller level.
    pub level_first: Vec<u32>,
    pub up_first: Vec<u64>,
    pub up_head: Vec<u32>,
    /// Per metric, the weights of each link: `down` (going down `From`, up `To`) and `up`.
    pub down: Vec<Vec<u16>>,
    pub up: Vec<Vec<u16>>,
    /// The new id of each layout id of the component (numbered from 0).
    pub new: Vec<u32>,
    /// The elimination-tree parent of each new id (the last of its row in layout ids), or
    /// `NONE` for a root.
    pub parent: Vec<u32>,
    /// CAR: exit turns by layout id, their heads and costs; copies by layout id.
    pub arc_first: Vec<u64>,
    pub arc_head: Vec<u32>,
    pub turn_cost: Vec<u16>,
    pub copy_first: Vec<u32>,
    pub copies: Vec<u32>,
    /// The top block's level count (0 for a whole component), and each block's first layout
    /// id (FORMAT §5.2): a cell's links stay in the cell or go to the top, and its level l is
    /// layer `l_top + l`.
    pub l_top: u32,
    pub block_first: Vec<u32>,
}

impl Graph {
    /// A component, layered by the format's own levels: the top block's, then every cell's
    /// level l together.
    pub fn load(m: &Model, t: &Family, x: u32) -> Result<Graph> {
        let fam = t.fam;
        let f = fam.name();
        ensure!((x as usize) < t.components(), "{f} component {x} does not exist");
        let n = t.nodes(x);
        let base = t.first[x as usize];
        let local = |v: u32, what: &str| -> Result<u32> {
            ensure!(v >= base && v - base < n as u32, "{f} component {x}: {what} {v} is outside its layout ids");
            Ok(v - base)
        };
        let word = if fam == Fam::Car { "turn" } else { "road" };
        let nm = t.metrics.len();
        // Exact sizes first: grown by doubling, a continent's largest component would hold
        // up to twice its links, more than the renumbered copy.
        let (mut links, mut arcs, mut ncopies) = (0usize, 0usize, 0usize);
        for b in t.block_first[x as usize] as usize..t.block_first[x as usize + 1] as usize {
            let p = format!("road/{f}/p{}/", t.b_pack[b]);
            let (r0, r1) = (t.b_row_first[b] as usize, (t.b_row_first[b] + t.b_nodes[b]) as usize);
            let span = |name: &str| -> Result<usize> {
                let v = m.get::<u32>(&format!("{p}{name}"))?;
                ensure!(r1 < v.len(), "{p}{name}: rows out of range");
                Ok((v[r1] - v[r0]) as usize)
            };
            links += span("up_first")?;
            if fam == Fam::Car {
                arcs += span(&format!("{word}_first"))?;
                ncopies += span("copy_first")?;
            }
        }
        let mut up_first = Vec::with_capacity(n + 1);
        up_first.push(0u64);
        let mut up_head = Vec::with_capacity(links);
        let (mut up, mut down): (Vec<Vec<u16>>, Vec<Vec<u16>>) = ((0..nm).map(|_| Vec::with_capacity(links)).collect(), (0..nm).map(|_| Vec::with_capacity(links)).collect());
        let (mut arc_first, mut arc_head, mut turn_cost) = (Vec::with_capacity(if fam == Fam::Car { n + 1 } else { 1 }), Vec::with_capacity(arcs), Vec::with_capacity(arcs));
        arc_first.push(0u64);
        let (mut copy_first, mut copies) = (Vec::with_capacity(if fam == Fam::Car { n + 1 } else { 1 }), Vec::with_capacity(ncopies));
        copy_first.push(0u32);
        for b in t.block_first[x as usize] as usize..t.block_first[x as usize + 1] as usize {
            let (pack, row_first, id_first, nodes) = (t.b_pack[b], t.b_row_first[b], t.b_id_first[b], t.b_nodes[b]);
            ensure!(id_first == base + up_first.len() as u32 - 1, "{f} component {x}: blocks do not follow layout ids");
            let p = format!("road/{f}/p{pack}/");
            let u32s = |name: &str| m.get::<u32>(&format!("{p}{name}"));
            let uf = u32s("up_first")?;
            let uh = u32s("up_head")?;
            let rows = row_first as usize..(row_first + nodes) as usize;
            ensure!(rows.end < uf.len(), "{p}: rows out of range");
            let links = uf[rows.start] as usize..uf[rows.end] as usize;
            for r in rows.clone() {
                let last = *up_first.last().unwrap();
                up_first.push(last + (uf[r + 1] - uf[r]) as u64);
            }
            for &h in &uh[links.clone()] {
                up_head.push(local(h, "upper neighbour")?);
            }
            for (k, name) in t.metrics.iter().enumerate() {
                up[k].extend_from_slice(&m.get::<u16>(&format!("{p}{name}.up"))?[links.clone()]);
                down[k].extend_from_slice(&m.get::<u16>(&format!("{p}{name}.down"))?[links.clone()]);
            }
            if fam == Fam::Car {
                let af = u32s(&format!("{word}_first"))?;
                let ah = u32s(&format!("{word}_head"))?;
                ensure!(rows.end < af.len(), "{p}: turn rows out of range");
                let arcs = af[rows.start] as usize..af[rows.end] as usize;
                for r in rows.clone() {
                    let last = *arc_first.last().unwrap();
                    arc_first.push(last + (af[r + 1] - af[r]) as u64);
                }
                for &h in &ah[arcs.clone()] {
                    arc_head.push(local(h, "turn head")?);
                }
                turn_cost.extend_from_slice(&m.get::<u16>(&format!("{p}car.turn_cost"))?[arcs]);
                let cf = u32s("copy_first")?;
                let cs = u32s("copies")?;
                for r in rows {
                    for &c in &cs[cf[r] as usize..cf[r + 1] as usize] {
                        copies.push(local(c, "copy")?);
                    }
                    copy_first.push(copies.len() as u32);
                }
            }
        }
        ensure!(up_first.len() == n + 1, "{f} component {x}: blocks cover {} of {n} nodes", up_first.len() - 1);
        let layers = format_layers(m, t, x)?;
        let blocks = t.block_first[x as usize] as usize..t.block_first[x as usize + 1] as usize;
        let l_top = blocks.clone().find(|&b| t.b_kind[b] == 1).map_or(0, |b| t.b_levels[b]);
        let block_first = blocks.map(|b| t.b_id_first[b] - base).collect();
        let mut g = renumber(fam, n, up_first, up_head, up, down, arc_first, arc_head, turn_cost, copy_first, copies, layers)?;
        g.l_top = l_top;
        g.block_first = block_first;
        Ok(g)
    }

    pub fn levels(&self) -> usize {
        self.level_first.len() - 1
    }

    pub fn width(&self, l: usize) -> u32 {
        self.level_first[l + 1] - self.level_first[l]
    }

    pub fn row(&self, v: u32) -> std::ops::Range<usize> {
        self.up_first[v as usize] as usize..self.up_first[v as usize + 1] as usize
    }

    pub fn links(&self) -> usize {
        self.up_head.len()
    }

    /// The weights of the downward half of a sweep in `dir` (FORMAT §8: `To` sweeps the
    /// reversed hierarchy), and of the upward half.
    pub fn going_down(&self, metric: usize, dir: Dir) -> &[u16] {
        if dir == Dir::From { &self.down[metric] } else { &self.up[metric] }
    }

    pub fn going_up(&self, metric: usize, dir: Dir) -> &[u16] {
        if dir == Dir::From { &self.up[metric] } else { &self.down[metric] }
    }

    /// A CAR segment's arrivals: itself and its copies (FORMAT §8 `arr`), layout ids.
    pub fn arrivals(&self, s: u32) -> impl Iterator<Item = u32> + '_ {
        let c: &[u32] = if self.fam == Fam::Car { &self.copies[self.copy_first[s as usize] as usize..self.copy_first[s as usize + 1] as usize] } else { &[] };
        std::iter::once(s).chain(c.iter().copied())
    }

    pub fn arcs(&self, s: u32) -> std::ops::Range<usize> {
        self.arc_first[s as usize] as usize..self.arc_first[s as usize + 1] as usize
    }

    /// Host bytes held.
    pub fn bytes(&self) -> usize {
        self.up_first.len() * 8
            + self.up_head.len() * 4
            + (self.up.len() + self.down.len()) * self.up_head.len() * 2
            + self.new.len() * 4
            + self.parent.len() * 4
            + self.arc_first.len() * 8
            + self.arc_head.len() * 6
            + self.copy_first.len() * 4
            + self.copies.len() * 4
    }

    /// The entry bounds of the downward half in `dir` (the pruning study's §5.5): for each
    /// link v → u, the cheapest link from u into v's elimination-tree subtree. Every link into
    /// the subtree from outside leaves an upper neighbour of v (filled-graph property), so a
    /// label in the subtree is at least min over v's row of label(u) ⊕ bound. One bottom-up
    /// merge of each row into its parent's.
    pub fn entry_bounds(&self, metric: usize, dir: Dir) -> Result<Vec<u16>> {
        let mut e = self.going_down(metric, dir).to_vec();
        // Children before parents: by decreasing level.
        for v in (0..self.n as u32).rev() {
            let p = self.parent[v as usize];
            if p == NONE {
                continue;
            }
            let (pr, vr) = (self.row(p), self.row(v));
            let heads = &self.up_head[pr.clone()];
            for k in vr {
                let u = self.up_head[k];
                if u == p {
                    continue;
                }
                let j = heads.binary_search(&u).map_err(|_| anyhow::anyhow!("{} component: node {v}'s upper neighbour {u} is not one of its parent {p}'s (not a filled graph)", self.fam.name()))?;
                let at = pr.start + j;
                e[at] = e[at].min(e[k]);
            }
        }
        Ok(e)
    }
}

/// The format's layer of each node of component x (numbered from 0): its level in the top
/// or whole block, or the top's level count plus its level in its cell.
pub fn format_layers(m: &Model, t: &Family, x: u32) -> Result<Vec<u32>> {
    let base = t.first[x as usize];
    let mut layer = vec![u32::MAX; t.nodes(x)];
    let mut l_top = 0u32;
    for b in t.block_first[x as usize] as usize..t.block_first[x as usize + 1] as usize {
        let lf = m.get::<u32>(&format!("road/{}/p{}/level_first", t.fam.name(), t.b_pack[b]))?;
        let (lb, nl) = (t.b_level_base[b] as usize, t.b_levels[b] as usize);
        ensure!(lb + nl < lf.len(), "road/{}/p{}/level_first: block {b}'s levels out of range", t.fam.name(), t.b_pack[b]);
        let add = if t.b_kind[b] == 2 { l_top } else { 0 };
        for l in 0..nl {
            for v in lf[lb + l]..lf[lb + l + 1] {
                ensure!(v >= base && ((v - base) as usize) < layer.len(), "block {b}'s level {l} names layout id {v}, outside its component");
                layer[(v - base) as usize] = add + l as u32;
            }
        }
        if t.b_kind[b] != 2 {
            l_top = nl as u32;
        }
    }
    ensure!(layer.iter().all(|&l| l != u32::MAX), "{} component {x}: the blocks' levels do not cover it", t.fam.name());
    Ok(layer)
}

/// Renumbers by layer, top first; every upper neighbour of a node must be in a smaller layer.
#[allow(clippy::too_many_arguments)]
pub(crate) fn renumber(
    fam: Fam,
    n: usize,
    up_first: Vec<u64>,
    up_head: Vec<u32>,
    up: Vec<Vec<u16>>,
    down: Vec<Vec<u16>>,
    arc_first: Vec<u64>,
    arc_head: Vec<u32>,
    turn_cost: Vec<u16>,
    copy_first: Vec<u32>,
    copies: Vec<u32>,
    depth: Vec<u32>,
) -> Result<Graph> {
    let row = |v: usize| up_first[v] as usize..up_first[v + 1] as usize;
    ensure!(depth.len() == n, "{} levels: {} layers for {n} nodes", fam.name(), depth.len());
    for v in 0..n {
        for k in row(v) {
            ensure!(depth[up_head[k] as usize] < depth[v], "{} levels: node {v}'s upper neighbour {} is not in a smaller level", fam.name(), up_head[k]);
        }
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
    drop(depth);
    // Heads first, with each new link's offset in its old row; then the old heads go, and
    // each weight array is permuted and its old one dropped, so a continent's largest
    // component never holds two full copies.
    let m = up_head.len();
    let mut g_first = Vec::with_capacity(n + 1);
    g_first.push(0u64);
    let mut g_head = Vec::with_capacity(m);
    let mut offs: Vec<u16> = Vec::with_capacity(m);
    let mut parent = vec![NONE; n];
    let mut r: Vec<(u32, u16)> = Vec::new();
    for i in 0..n {
        let v = old[i] as usize;
        let rv = row(v);
        ensure!(rv.len() <= 1 << 16, "a row of {} links", rv.len());
        r.clear();
        r.extend(rv.clone().map(|k| (new[up_head[k] as usize], (k - rv.start) as u16)));
        r.sort_unstable();
        if let Some(k) = rv.last() {
            parent[i] = new[up_head[k] as usize];
        }
        for &(h, o) in &r {
            g_head.push(h);
            offs.push(o);
        }
        g_first.push(g_head.len() as u64);
    }
    drop(up_head);
    let permute = |w: Vec<u16>| -> Vec<u16> {
        let mut out = Vec::with_capacity(m);
        for i in 0..n {
            let base = up_first[old[i] as usize] as usize;
            out.extend(offs[g_first[i] as usize..g_first[i + 1] as usize].iter().map(|&o| w[base + o as usize]));
        }
        out
    };
    let g_up: Vec<Vec<u16>> = up.into_iter().map(permute).collect();
    let g_down: Vec<Vec<u16>> = down.into_iter().map(permute).collect();
    Ok(Graph { fam, n, level_first, up_first: g_first, up_head: g_head, down: g_down, up: g_up, new, parent, arc_first, arc_head, turn_cost, copy_first, copies, l_top: 0, block_first: vec![0] })
}

thread_local! {
    static SCRATCH: std::cell::RefCell<(Vec<u16>, Vec<u64>)> = const { std::cell::RefCell::new((Vec::new(), Vec::new())) };
}

/// The upward half of a sweep from seeds (new id, label): every node the seeds reach over
/// upper links, with its label (`INF` when only saturated sums reach it), ascending by new id.
pub fn upward(g: &Graph, metric: usize, dir: Dir, seeds: &[(u32, u16)]) -> Vec<(u32, u16)> {
    let w = g.going_up(metric, dir);
    SCRATCH.with(|s| {
        let (label, seen) = &mut *s.borrow_mut();
        if label.len() < g.n {
            label.resize(g.n, INF);
            seen.resize(g.n.div_ceil(64), 0);
        }
        let mut heap = BinaryHeap::new();
        let mut visit = |v: u32, t: u16, heap: &mut BinaryHeap<u32>, label: &mut Vec<u16>| {
            let (wd, b) = (v as usize / 64, 1u64 << (v % 64));
            if seen[wd] & b == 0 {
                seen[wd] |= b;
                heap.push(v);
            }
            let l = &mut label[v as usize];
            *l = (*l).min(t);
        };
        for &(v, t) in seeds {
            visit(v, t, &mut heap, label);
        }
        let mut out = Vec::new();
        while let Some(v) = heap.pop() {
            let lv = label[v as usize];
            out.push((v, lv));
            for k in g.row(v) {
                visit(g.up_head[k], lv.saturating_add(w[k]), &mut heap, label);
            }
        }
        for &(v, _) in &out {
            label[v as usize] = INF;
            seen[v as usize / 64] = 0;
        }
        out.reverse();
        out
    })
}

/// The least sum of two upward labels over the nodes both reach (both ascending by id).
pub fn meet(a: &[(u32, u16)], b: &[(u32, u16)]) -> u16 {
    let (mut i, mut j, mut t) = (0, 0, INF);
    while i < a.len() && j < b.len() {
        match a[i].0.cmp(&b[j].0) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                t = t.min(a[i].1.saturating_add(b[j].1));
                i += 1;
                j += 1;
            }
        }
    }
    t
}
