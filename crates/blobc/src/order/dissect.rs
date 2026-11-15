//! The coarsening fallback's dissection (SPEC §5; milestone 5, attempt 1: flow-based
//! refinement of the lifted separators). A region larger than a cell takes the top separator
//! of its coarse graph's nested-dissection order, lifted to the fine graph; a minimum vertex
//! cut in a band of `BAND` hops around it, between the two sides beyond the band, replaces
//! it; the region's fine nodes left, split into connected pieces, are dissected in turn.
//! Regions no larger than a cell are cells, each ordered on its own; the separators, deepest
//! first, are the top block. A separator is then as thin as the fine graph allows near the
//! coarse one, rather than whole clusters thick.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};

use rayon::prelude::*;

use crate::order::cells::{Adj, Block, Kind, order_induced};
use crate::order::elim::etree;
use crate::order::{Graph, InertialFlow, Orderer, ranks};

const NONE: u32 = u32::MAX;

/// Hops on either side of a lifted separator in which the flow may move it (on HdF, forced:
/// 4 hops give 1.08× the one-piece sweep layers for CAR, 8 give 1.04×, 16 and 32 give 1.00×).
const BAND: u32 = 16;

/// The share of a region's weight a second part must hold for the coarse order to split it
/// there (smaller subtrees off the separator are dangling pieces, not a side).
const SPLIT: f64 = 0.1;

enum Tree {
    Leaf(Vec<u32>),
    /// A separator (empty when the region was disconnected) and the regions it leaves.
    Sep(Vec<u32>, Vec<Tree>),
}

struct Dissect<'a> {
    g: &'a Graph<'a>,
    fine: &'a Adj,
    /// Each fine node's cluster.
    cluster: &'a [u32],
    max_cell: usize,
    /// The region each fine node is in now, and its index there. Regions are disjoint, so
    /// parallel regions write disjoint entries.
    owner: Vec<AtomicU32>,
    index: Vec<AtomicU32>,
    next: AtomicU32,
}

/// The blocks of a component from the dissection: the top block (the separators, deepest
/// first), then the cells, the regions packed in dissection order up to `max_cell` nodes.
/// `whole` orders the component in one piece, for a dissection that does not split it.
pub(crate) fn refined(g: &Graph, fine: &Adj, cluster: &[u32], whole: &dyn Fn() -> Vec<u32>, orderer: &dyn Orderer, max_cell: usize) -> Vec<Block> {
    let d = Dissect {
        g,
        fine,
        cluster,
        max_cell,
        owner: (0..g.n).map(|_| AtomicU32::new(NONE)).collect(),
        index: (0..g.n).map(|_| AtomicU32::new(NONE)).collect(),
        next: AtomicU32::new(0),
    };
    let tree = d.dissect((0..g.n as u32).collect());
    drop(d);
    crate::build::mem::phase("dissected");
    let (mut top, mut leaves) = (Vec::new(), Vec::new());
    walk(tree, &mut top, &mut leaves);
    let mut packs: Vec<Vec<u32>> = Vec::new();
    for leaf in leaves {
        match packs.last_mut() {
            Some(p) if !p.is_empty() && p.len() + leaf.len() <= max_cell => p.extend(leaf),
            _ => packs.push(leaf),
        }
    }
    if top.is_empty() || packs.len() < 2 {
        return vec![Block { kind: Kind::Whole, nodes: whole() }];
    }
    let mut out = vec![Block { kind: Kind::Top, nodes: top }];
    for mut p in packs {
        p.sort_unstable();
        out.push(Block { kind: Kind::Cell, nodes: order_induced(g, fine, &p, orderer) });
    }
    crate::build::mem::phase("cells ordered");
    out
}

/// The separators in post-order (each after the regions it splits), and the leaves in order.
fn walk(t: Tree, top: &mut Vec<u32>, leaves: &mut Vec<Vec<u32>>) {
    match t {
        Tree::Leaf(nodes) => leaves.push(nodes),
        Tree::Sep(sep, children) => {
            for c in children {
                walk(c, top, leaves);
            }
            top.extend(sep);
        }
    }
}

impl Dissect<'_> {
    fn owned(&self, v: u32, id: u32) -> bool {
        self.owner[v as usize].load(Ordering::Relaxed) == id
    }

    fn at(&self, v: u32) -> usize {
        self.index[v as usize].load(Ordering::Relaxed) as usize
    }

    /// Takes the nodes (ascending) as a new region: its id.
    fn claim(&self, nodes: &[u32]) -> u32 {
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        for (i, &v) in nodes.iter().enumerate() {
            self.owner[v as usize].store(id, Ordering::Relaxed);
            self.index[v as usize].store(i as u32, Ordering::Relaxed);
        }
        id
    }

    /// The connected pieces of region `id`'s nodes, each ascending, in order of their least node.
    fn pieces(&self, nodes: &[u32], id: u32) -> Vec<Vec<u32>> {
        let mut seen = vec![false; nodes.len()];
        let mut out = Vec::new();
        for (s, &v) in nodes.iter().enumerate() {
            if seen[s] || !self.owned(v, id) {
                continue;
            }
            seen[s] = true;
            let mut piece = vec![v];
            let mut k = 0;
            while k < piece.len() {
                let x = piece[k];
                k += 1;
                for e in self.fine.range(x as usize) {
                    let u = self.fine.to[e];
                    if self.owned(u, id) && !seen[self.at(u)] {
                        seen[self.at(u)] = true;
                        piece.push(u);
                    }
                }
            }
            piece.sort_unstable();
            out.push(piece);
        }
        out
    }

    fn dissect(&self, nodes: Vec<u32>) -> Tree {
        if nodes.len() <= self.max_cell.max(1) {
            return Tree::Leaf(nodes);
        }
        let id = self.claim(&nodes);
        let pieces = self.pieces(&nodes, id);
        if pieces.len() > 1 {
            return Tree::Sep(Vec::new(), pieces.into_par_iter().map(|p| self.dissect(p)).collect());
        }
        let Some(sep) = self.separator(&nodes, id) else {
            return Tree::Leaf(nodes);
        };
        // The separator leaves the region; the rest splits into pieces.
        let gone = self.next.fetch_add(1, Ordering::Relaxed);
        for &v in &sep {
            self.owner[v as usize].store(gone, Ordering::Relaxed);
        }
        let pieces = self.pieces(&nodes, id);
        Tree::Sep(sep, pieces.into_par_iter().map(|p| self.dissect(p)).collect())
    }

    /// A fine separator of the connected region: the coarse order's top separator, refined by
    /// a minimum vertex cut in its band. None if the coarse order does not split the region.
    fn separator(&self, nodes: &[u32], id: u32) -> Option<Vec<u32>> {
        // The region's coarse graph: its nodes' clusters, numbered as first met.
        let mut local: HashMap<u32, u32> = HashMap::new();
        let mut of = Vec::with_capacity(nodes.len());
        let (mut lat, mut lon, mut w): (Vec<f64>, Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new(), Vec::new());
        for &v in nodes {
            let next = local.len() as u32;
            let c = *local.entry(self.cluster[v as usize]).or_insert(next);
            if c == next {
                lat.push(0.0);
                lon.push(0.0);
                w.push(0.0);
            }
            lat[c as usize] += self.g.lat[v as usize] as f64;
            lon[c as usize] += self.g.lon[v as usize] as f64;
            w[c as usize] += 1.0;
            of.push(c);
        }
        drop(local);
        let nc = w.len();
        if nc < 3 {
            return None;
        }
        let mut edges: Vec<(u32, u32)> = Vec::new();
        for (i, &v) in nodes.iter().enumerate() {
            for e in self.fine.range(v as usize) {
                let u = self.fine.to[e];
                if self.owned(u, id) {
                    let (a, b) = (of[i], of[self.at(u)]);
                    if a < b {
                        edges.push((a, b));
                    }
                }
            }
        }
        edges.sort_unstable();
        edges.dedup();
        let (tail, head): (Vec<u32>, Vec<u32>) = edges.iter().copied().unzip();
        let clat: Vec<f32> = (0..nc).map(|c| (lat[c] / w[c]) as f32).collect();
        let clon: Vec<f32> = (0..nc).map(|c| (lon[c] / w[c]) as f32).collect();
        let order = InertialFlow.order(&Graph { n: nc, tail: &tail, head: &head, lat: &clat, lon: &clon });
        let rank = ranks(&order);
        let parent = etree(nc, edges.iter().map(|&(a, b)| (rank[a as usize], rank[b as usize])));
        drop((edges, tail, head));
        let mut kids: Vec<Vec<u32>> = vec![Vec::new(); nc];
        for (r, p) in parent.iter().enumerate() {
            if let Some(p) = p {
                kids[*p as usize].push(r as u32);
            }
        }
        // Fine nodes per coarse rank, and per subtree.
        let mut sub = vec![0u64; nc];
        for c in 0..nc {
            sub[rank[c] as usize] = w[c] as u64;
        }
        for r in 0..nc {
            if let Some(p) = parent[r] {
                sub[p as usize] += sub[r];
            }
        }
        // The top separator: down the heavy path from the root, to the first node where a
        // second subtree holds at least `SPLIT` of the weight below it; the light subtrees
        // hanging off the path on the way are parts of their own.
        let mut part = vec![NONE; nc];
        let mut x = (nc - 1) as u32;
        if parent[x as usize].is_some() {
            return None;
        }
        let mut tops: Vec<u32> = Vec::new();
        loop {
            part[x as usize] = 0;
            let mut k = kids[x as usize].clone();
            if k.is_empty() {
                return None;
            }
            k.sort_by(|&i, &j| sub[j as usize].cmp(&sub[i as usize]).then(i.cmp(&j)));
            if k.len() >= 2 && sub[k[1] as usize] as f64 >= SPLIT * sub[x as usize] as f64 {
                tops.extend(k);
                break;
            }
            tops.extend_from_slice(&k[1..]);
            x = k[0];
        }
        // The parts below it, numbered from 1, and their fine sizes.
        let mut size = vec![0u64; tops.len() + 1];
        for (k, &c) in tops.iter().enumerate() {
            let mut stack = vec![c];
            while let Some(r) = stack.pop() {
                part[r as usize] = k as u32 + 1;
                stack.extend_from_slice(&kids[r as usize]);
            }
        }
        let side: Vec<u32> = of.iter().map(|&c| part[rank[c as usize] as usize]).collect();
        for &s in &side {
            size[s as usize] += 1;
        }
        // The flow separates the largest part from the others.
        let a = (1..size.len()).max_by(|&i, &j| size[i].cmp(&size[j]).then(j.cmp(&i))).unwrap() as u32;
        let lifted: Vec<u32> = (0..nodes.len()).filter(|&i| side[i] == 0).map(|i| nodes[i]).collect();
        // The band: within BAND hops of the lifted separator.
        let mut dist = vec![NONE; nodes.len()];
        let mut queue: Vec<u32> = Vec::new();
        for (i, &s) in side.iter().enumerate() {
            if s == 0 {
                dist[i] = 0;
                queue.push(i as u32);
            }
        }
        let mut k = 0;
        while k < queue.len() {
            let i = queue[k] as usize;
            k += 1;
            if dist[i] == BAND {
                continue;
            }
            for e in self.fine.range(nodes[i] as usize) {
                let u = self.fine.to[e];
                if self.owned(u, id) {
                    let j = self.at(u);
                    if dist[j] == NONE {
                        dist[j] = dist[i] + 1;
                        queue.push(j as u32);
                    }
                }
            }
        }
        let band = queue;
        let cut = min_cut(self, nodes, id, &band, &dist, &side, a);
        Some(cut.unwrap_or(lifted))
    }
}

/// A minimum vertex cut in the band between the nodes beyond it on side `a` and those on the
/// other sides: max-flow (Dinic) with unit node capacities. None if one side does not reach
/// the band.
fn min_cut(d: &Dissect, nodes: &[u32], id: u32, band: &[u32], dist: &[u32], side: &[u32], a: u32) -> Option<Vec<u32>> {
    let nb = band.len();
    let mut bidx = HashMap::with_capacity(nb);
    for (b, &i) in band.iter().enumerate() {
        bidx.insert(i, b as u32);
    }
    // Nodes 2b (in) and 2b + 1 (out) per band node, then the source and the sink.
    let (s, t) = ((2 * nb) as u32, (2 * nb + 1) as u32);
    let mut arcs: Vec<(u32, u32, u32)> = Vec::new();
    let (mut from_a, mut to_b) = (false, false);
    const INF: u32 = u32::MAX / 2;
    for (b, &i) in band.iter().enumerate() {
        let b = b as u32;
        arcs.push((2 * b, 2 * b + 1, 1));
        let (mut src, mut snk) = (false, false);
        for e in d.fine.range(nodes[i as usize] as usize) {
            let u = d.fine.to[e];
            if !d.owned(u, id) {
                continue;
            }
            let j = d.at(u) as u32;
            if dist[j as usize] != NONE {
                arcs.push((2 * b + 1, 2 * bidx[&j], INF));
            } else if side[j as usize] == a {
                src = true;
            } else {
                snk = true;
            }
        }
        if src {
            arcs.push((s, 2 * b, INF));
            from_a = true;
        }
        if snk {
            arcs.push((2 * b + 1, t, INF));
            to_b = true;
        }
    }
    if !(from_a && to_b) {
        return None;
    }
    let mut f = Flow::new(2 * nb + 2, &arcs);
    drop(arcs);
    f.max_flow(s, t);
    let reach = f.reachable(s);
    let mut cut: Vec<u32> = (0..nb).filter(|&b| reach[2 * b] && !reach[2 * b + 1]).map(|b| nodes[band[b] as usize]).collect();
    cut.sort_unstable();
    Some(cut)
}

/// A flow network, CSR with paired residual arcs.
struct Flow {
    first: Vec<u32>,
    to: Vec<u32>,
    cap: Vec<u32>,
    rev: Vec<u32>,
}

impl Flow {
    fn new(n: usize, arcs: &[(u32, u32, u32)]) -> Flow {
        // Arc positions are u32: a band of a separator, far below 2^32 arcs.
        assert!(2 * arcs.len() < u32::MAX as usize, "{} arcs in a flow network", arcs.len());
        let mut deg = vec![0u32; n + 1];
        for &(a, b, _) in arcs {
            deg[a as usize + 1] += 1;
            deg[b as usize + 1] += 1;
        }
        for i in 0..n {
            deg[i + 1] += deg[i];
        }
        let m = 2 * arcs.len();
        let (mut to, mut cap, mut rev) = (vec![0u32; m], vec![0u32; m], vec![0u32; m]);
        let mut at = deg.clone();
        for &(a, b, c) in arcs {
            let (i, j) = (at[a as usize], at[b as usize]);
            at[a as usize] += 1;
            at[b as usize] += 1;
            (to[i as usize], cap[i as usize], rev[i as usize]) = (b, c, j);
            (to[j as usize], cap[j as usize], rev[j as usize]) = (a, 0, i);
        }
        Flow { first: deg, to, cap, rev }
    }

    fn n(&self) -> usize {
        self.first.len() - 1
    }

    fn levels(&self, s: u32) -> Vec<u32> {
        let mut level = vec![NONE; self.n()];
        level[s as usize] = 0;
        let mut queue = vec![s];
        let mut k = 0;
        while k < queue.len() {
            let v = queue[k] as usize;
            k += 1;
            for e in self.first[v] as usize..self.first[v + 1] as usize {
                let u = self.to[e] as usize;
                if self.cap[e] > 0 && level[u] == NONE {
                    level[u] = level[v] + 1;
                    queue.push(u as u32);
                }
            }
        }
        level
    }

    fn reachable(&self, s: u32) -> Vec<bool> {
        self.levels(s).into_iter().map(|l| l != NONE).collect()
    }

    /// Dinic: blocking flows along BFS levels, each path found by an iterative DFS.
    fn max_flow(&mut self, s: u32, t: u32) -> u64 {
        let mut total = 0u64;
        loop {
            let level = self.levels(s);
            if level[t as usize] == NONE {
                return total;
            }
            let mut it: Vec<u32> = self.first[..self.n()].to_vec();
            loop {
                // One augmenting path: the arcs taken, from s.
                let mut path: Vec<u32> = Vec::new();
                let mut v = s;
                while v != t {
                    let vi = v as usize;
                    let mut advanced = false;
                    while it[vi] < self.first[vi + 1] {
                        let e = it[vi] as usize;
                        let u = self.to[e];
                        if self.cap[e] > 0 && level[u as usize] == level[vi] + 1 {
                            path.push(e as u32);
                            v = u;
                            advanced = true;
                            break;
                        }
                        it[vi] += 1;
                    }
                    if !advanced {
                        // A dead end: retreat, and skip the arc that led here.
                        match path.pop() {
                            None => break,
                            Some(e) => {
                                let back = self.to[self.rev[e as usize] as usize];
                                it[back as usize] += 1;
                                v = back;
                            }
                        }
                    }
                }
                if v != t {
                    break;
                }
                let push = path.iter().map(|&e| self.cap[e as usize]).min().unwrap_or(0);
                for &e in &path {
                    self.cap[e as usize] -= push;
                    let r = self.rev[e as usize] as usize;
                    self.cap[r] += push;
                }
                total += push as u64;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_vertex_cut() {
        // Two triangles joined through one node 3: s side {0,1,2}, t side {4,5,6}.
        let arcs = [(0, 1, 5), (1, 2, 1), (0, 3, 1), (3, 2, 1)];
        let mut f = Flow::new(4, &arcs);
        assert_eq!(f.max_flow(0, 2), 2);
        let r = f.reachable(0);
        assert!(r[0] && r[1] && !r[2] && !r[3]);
    }
}
