//! Cells (SPEC §5): an expert too large to build at once is coarsened, its coarse graph is
//! ordered by nested dissection, and the coarse elimination tree is cut where subtrees fit
//! the budget. The subtrees below the cut become cells; the coarse nodes above it, lifted
//! to the fine graph, are the top block. Different subtrees of an elimination tree share no
//! arc, so no fine arc joins two cells. Each block is then ordered on its own.

use crate::elim::eliminate;
use crate::{Graph, InertialFlow, Orderer, ranks};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Whole = 0,
    Top = 1,
    Cell = 2,
}

/// A block of an expert: its nodes in contraction order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Block {
    pub kind: Kind,
    pub nodes: Vec<u32>,
}

/// Undirected adjacency without loops or parallel arcs, with arc multiplicities.
struct Adj {
    first: Vec<u32>,
    to: Vec<u32>,
    weight: Vec<u32>,
}

impl Adj {
    fn new(n: usize, edges: impl Iterator<Item = (u32, u32, u32)>) -> Adj {
        let mut e: Vec<(u32, u32, u32)> = edges.filter(|e| e.0 != e.1).flat_map(|(a, b, w)| [(a, b, w), (b, a, w)]).collect();
        e.sort_unstable();
        let mut merged: Vec<(u32, u32, u32)> = Vec::with_capacity(e.len());
        for (a, b, w) in e {
            match merged.last_mut() {
                Some(l) if l.0 == a && l.1 == b => l.2 += w,
                _ => merged.push((a, b, w)),
            }
        }
        let mut first = vec![0u32; n + 1];
        for &(a, _, _) in &merged {
            first[a as usize + 1] += 1;
        }
        for i in 0..n {
            first[i + 1] += first[i];
        }
        Adj { first, to: merged.iter().map(|e| e.1).collect(), weight: merged.iter().map(|e| e.2).collect() }
    }

    fn range(&self, v: usize) -> std::ops::Range<usize> {
        self.first[v] as usize..self.first[v + 1] as usize
    }
}

/// The induced subgraph on `nodes` (sorted), ordered, mapped back to node ids.
fn order_induced(g: &Graph, adj: &Adj, nodes: &[u32], orderer: &dyn Orderer) -> Vec<u32> {
    let local = |v: u32| nodes.binary_search(&v).ok();
    let (mut tail, mut head) = (Vec::new(), Vec::new());
    for (i, &v) in nodes.iter().enumerate() {
        for k in adj.range(v as usize) {
            if let Some(j) = local(adj.to[k])
                && i < j
            {
                tail.push(i as u32);
                head.push(j as u32);
            }
        }
    }
    let lat: Vec<f32> = nodes.iter().map(|&v| g.lat[v as usize]).collect();
    let lon: Vec<f32> = nodes.iter().map(|&v| g.lon[v as usize]).collect();
    let sub = Graph { n: nodes.len(), tail: &tail, head: &head, lat: &lat, lon: &lon };
    orderer.order(&sub).into_iter().map(|i| nodes[i as usize]).collect()
}

/// One round of heavy-edge matching: the cluster of each node, and the cluster count.
fn matching(adj: &Adj, w: &[u64], cap: u64) -> (Vec<u32>, usize) {
    const NONE: u32 = u32::MAX;
    let n = w.len();
    let mut cluster = vec![NONE; n];
    let mut k = 0u32;
    for v in 0..n {
        if cluster[v] != NONE {
            continue;
        }
        let mut best: Option<(u32, u64, u32)> = None;
        for e in adj.range(v) {
            let u = adj.to[e] as usize;
            if cluster[u] != NONE || w[u] + w[v] > cap {
                continue;
            }
            // Heaviest arc, then the lightest pair, then the smallest node.
            let cand = (adj.weight[e], w[u] + w[v], u as u32);
            if best.is_none_or(|b| cand.0 > b.0 || (cand.0 == b.0 && (cand.1, cand.2) < (b.1, b.2))) {
                best = Some(cand);
            }
        }
        cluster[v] = k;
        if let Some((_, _, u)) = best {
            cluster[u as usize] = k;
        }
        k += 1;
    }
    (cluster, k as usize)
}

/// The blocks of an expert: one whole block if it has at most `max_cell` nodes, else a top
/// block and cells of at most `max_cell` nodes each.
pub fn plan(g: &Graph, orderer: &dyn Orderer, max_cell: usize) -> Vec<Block> {
    if g.n <= max_cell.max(1) {
        return vec![Block { kind: Kind::Whole, nodes: orderer.order(g) }];
    }
    let fine = Adj::new(g.n, g.tail.iter().zip(g.head).map(|(&a, &b)| (a, b, 1)));
    // Coarsen to clusters of at most a quarter cell, until about 16 fine nodes per cluster.
    let target = (g.n / 16).max(1024);
    let cap = (max_cell as u64 / 4).max(1);
    let mut map: Vec<u32> = (0..g.n as u32).collect();
    let mut w: Vec<u64> = vec![1; g.n];
    let mut adj = Adj::new(g.n, g.tail.iter().zip(g.head).map(|(&a, &b)| (a, b, 1)));
    let (mut lat, mut lon): (Vec<f64>, Vec<f64>) = (g.lat.iter().map(|&x| x as f64).collect(), g.lon.iter().map(|&x| x as f64).collect());
    while w.len() > target {
        let (cluster, k) = matching(&adj, &w, cap);
        if k * 10 > w.len() * 9 {
            break;
        }
        let mut nw = vec![0u64; k];
        let (mut nlat, mut nlon) = (vec![0.0f64; k], vec![0.0f64; k]);
        for v in 0..w.len() {
            let c = cluster[v] as usize;
            nw[c] += w[v];
            nlat[c] += lat[v] * w[v] as f64;
            nlon[c] += lon[v] * w[v] as f64;
        }
        for c in 0..k {
            nlat[c] /= nw[c] as f64;
            nlon[c] /= nw[c] as f64;
        }
        let edges: Vec<(u32, u32, u32)> = (0..w.len())
            .flat_map(|v| adj.range(v).map(move |e| (v, e)))
            .map(|(v, e)| (cluster[v], cluster[adj.to[e] as usize], adj.weight[e]))
            .collect();
        adj = Adj::new(k, edges.into_iter().filter(|e| e.0 < e.1));
        for m in &mut map {
            *m = cluster[*m as usize];
        }
        (w, lat, lon) = (nw, nlat, nlon);
    }
    // Order the coarse graph and cut its elimination tree.
    let nc = w.len();
    let (mut tail, mut head) = (Vec::new(), Vec::new());
    for v in 0..nc {
        for e in adj.range(v) {
            if (v as u32) < adj.to[e] {
                tail.push(v as u32);
                head.push(adj.to[e]);
            }
        }
    }
    let (flat, flon): (Vec<f32>, Vec<f32>) = (lat.iter().map(|&x| x as f32).collect(), lon.iter().map(|&x| x as f32).collect());
    let order = InertialFlow.order(&Graph { n: nc, tail: &tail, head: &head, lat: &flat, lon: &flon });
    let rank = ranks(&order);
    let el = eliminate(nc, tail.iter().zip(&head).map(|(&a, &b)| (rank[a as usize], rank[b as usize])));
    let parent: Vec<Option<u32>> = el.up.iter().map(|u| u.first().copied()).collect();
    let mut sub = vec![0u64; nc];
    for r in 0..nc {
        sub[r] += w[order[r] as usize];
        if let Some(p) = parent[r] {
            sub[p as usize] += sub[r];
        }
    }
    let top: Vec<bool> = sub.iter().map(|&s| s > max_cell as u64).collect();
    // Units: subtrees hanging below the top, by their root; packed into cells in root order.
    const NONE: u32 = u32::MAX;
    let mut unit = vec![NONE; nc];
    for r in (0..nc).rev() {
        if !top[r] {
            unit[r] = match parent[r] {
                Some(p) if !top[p as usize] => unit[p as usize],
                _ => r as u32,
            };
        }
    }
    let mut cell_of_unit = vec![NONE; nc];
    let mut size: Vec<u64> = Vec::new();
    for r in 0..nc {
        if !top[r] && unit[r] == r as u32 {
            match size.last_mut() {
                Some(s) if *s + sub[r] <= max_cell as u64 => *s += sub[r],
                _ => size.push(sub[r]),
            }
            cell_of_unit[r] = (size.len() - 1) as u32;
        }
    }
    // Lift to the fine graph, then let top nodes whose arcs all stay in one cell join it.
    let mut block: Vec<u32> = (0..g.n)
        .map(|v| {
            let r = rank[map[v] as usize] as usize;
            if top[r] { NONE } else { cell_of_unit[unit[r] as usize] }
        })
        .collect();
    let mut count: Vec<u64> = vec![0; size.len()];
    for &b in &block {
        if b != NONE {
            count[b as usize] += 1;
        }
    }
    for _ in 0..3 {
        let mut moved = false;
        for v in 0..g.n {
            if block[v] != NONE {
                continue;
            }
            let mut only = NONE;
            let mut ok = true;
            for e in fine.range(v) {
                let b = block[fine.to[e] as usize];
                if b != NONE {
                    if only != NONE && only != b {
                        ok = false;
                        break;
                    }
                    only = b;
                }
            }
            if ok && only != NONE && count[only as usize] < max_cell as u64 {
                block[v] = only;
                count[only as usize] += 1;
                moved = true;
            }
        }
        if !moved {
            break;
        }
    }
    let top_nodes: Vec<u32> = (0..g.n as u32).filter(|&v| block[v as usize] == NONE).collect();
    if top_nodes.is_empty() || size.len() < 2 {
        return vec![Block { kind: Kind::Whole, nodes: orderer.order(g) }];
    }
    let mut out = vec![Block { kind: Kind::Top, nodes: order_induced(g, &fine, &top_nodes, orderer) }];
    for c in 0..size.len() as u32 {
        let nodes: Vec<u32> = (0..g.n as u32).filter(|&v| block[v as usize] == c).collect();
        if !nodes.is_empty() {
            out.push(Block { kind: Kind::Cell, nodes: order_induced(g, &fine, &nodes, orderer) });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(k: u32) -> (Vec<u32>, Vec<u32>, Vec<f32>, Vec<f32>) {
        let (mut tail, mut head) = (Vec::new(), Vec::new());
        for i in 0..k {
            for j in 0..k {
                let v = i * k + j;
                if j + 1 < k {
                    tail.push(v);
                    head.push(v + 1);
                }
                if i + 1 < k {
                    tail.push(v);
                    head.push(v + k);
                }
            }
        }
        let lat = (0..k * k).map(|v| (v / k) as f32 * 0.001).collect();
        let lon = (0..k * k).map(|v| (v % k) as f32 * 0.001).collect();
        (tail, head, lat, lon)
    }

    #[test]
    fn cells_partition_and_separate() {
        let (tail, head, lat, lon) = grid(40);
        let g = Graph { n: 1600, tail: &tail, head: &head, lat: &lat, lon: &lon };
        let blocks = plan(&g, &InertialFlow, 300);
        assert_eq!(blocks[0].kind, Kind::Top);
        assert!(blocks.len() > 3);
        let mut of = vec![usize::MAX; 1600];
        for (b, bl) in blocks.iter().enumerate() {
            for &v in &bl.nodes {
                assert_eq!(of[v as usize], usize::MAX);
                of[v as usize] = b;
            }
            assert!(bl.kind == Kind::Top || bl.nodes.len() <= 300);
        }
        assert!(of.iter().all(|&b| b != usize::MAX));
        for (&a, &b) in tail.iter().zip(&head) {
            let (x, y) = (of[a as usize], of[b as usize]);
            assert!(x == y || x == 0 || y == 0, "arc {a}-{b} joins cells {x} and {y}");
        }
        assert!(blocks[0].nodes.len() < 400, "top block of {}", blocks[0].nodes.len());
        assert_eq!(blocks, plan(&g, &InertialFlow, 300));
        assert_eq!(plan(&g, &InertialFlow, 2000)[0].kind, Kind::Whole);
    }
}
