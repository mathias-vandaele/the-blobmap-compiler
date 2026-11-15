//! The hierarchy of one component (SPEC §5, FORMAT §4, §5.2, §6). Each cell is contracted on
//! its own, with its top-block neighbours as external nodes; the cliques it leaves among
//! them are contracted with the top block, last. Layout ids number the top block (or the
//! whole component) first, then the cells; inside a block, levels top first.

use blobmap_model::{INF, sat};

use crate::order::elim::eliminate;
use crate::order::{Block, Kind};

/// One block of the layout (FORMAT §5.1).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayoutBlock {
    pub kind: Kind,
    pub id_first: u32,
    pub nodes: u32,
    /// L + 1 ascending layout ids: level l is `level_first[l] .. level_first[l + 1]`.
    pub level_first: Vec<u32>,
}

impl LayoutBlock {
    pub fn levels(&self) -> u32 {
        (self.level_first.len() - 1) as u32
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hierarchy {
    /// Node of the input graph → layout id, and back.
    pub layout_of: Vec<u32>,
    pub node_of: Vec<u32>,
    pub blocks: Vec<LayoutBlock>,
    /// Upper neighbours by layout id (CSR), ascending; a link is a position in `up_head`.
    pub up_first: Vec<u64>,
    pub up_head: Vec<u32>,
}

impl Hierarchy {
    /// `edges` are node pairs in any direction, run twice (to count, then to fill); `plan` is
    /// `crate::order::plan`'s. The arcs and the upper neighbours are flat arrays, a few words
    /// per node and per link, so that a component of any size fits (SPEC §10).
    pub fn build<I: Iterator<Item = (u32, u32)>>(n: usize, edges: impl Fn() -> I, plan: &[Block]) -> Hierarchy {
        const NONE: u32 = u32::MAX;
        let mut block_of = vec![NONE; n];
        let mut rank_in = vec![0u32; n];
        for (b, bl) in plan.iter().enumerate() {
            for (r, &v) in bl.nodes.iter().enumerate() {
                assert_eq!(block_of[v as usize], NONE, "node {v} in two blocks");
                block_of[v as usize] = b as u32;
                rank_in[v as usize] = r as u32;
            }
        }
        assert!(block_of.iter().all(|&b| b != NONE), "a node in no block");
        let top = plan.iter().position(|b| b.kind == Kind::Top);
        // Arcs by node, both directions, in the order given (CSR).
        let mut adj_first = vec![0u64; n + 1];
        for (a, b) in edges().filter(|e| e.0 != e.1) {
            adj_first[a as usize + 1] += 1;
            adj_first[b as usize + 1] += 1;
        }
        for i in 0..n {
            adj_first[i + 1] += adj_first[i];
        }
        let mut adj_to = vec![0u32; adj_first[n] as usize];
        let mut at = adj_first[..n].to_vec();
        for (a, b) in edges().filter(|e| e.0 != e.1) {
            for (x, y) in [(a, b), (b, a)] {
                adj_to[at[x as usize] as usize] = y;
                at[x as usize] += 1;
            }
        }
        drop(at);
        let adj = |v: u32| &adj_to[adj_first[v as usize] as usize..adj_first[v as usize + 1] as usize];
        // Upper neighbours of each node, as nodes, by the block's rank (externals last, by
        // their top-block rank): node v's are `up_to[up_at[v] ..][.. up_len[v]]`.
        let (mut up_at, mut up_len) = (vec![0u64; n], vec![0u32; n]);
        let mut up_to: Vec<u32> = Vec::new();
        let mut top_fill: Vec<(u32, u32)> = Vec::new();
        for (b, bl) in plan.iter().enumerate() {
            if Some(b) == top {
                continue;
            }
            let k = bl.nodes.len() as u32;
            let mut ext: Vec<u32> = Vec::new();
            for &v in &bl.nodes {
                for &u in adj(v) {
                    if block_of[u as usize] != b as u32 {
                        assert_eq!(Some(block_of[u as usize] as usize), top, "an arc joins two cells");
                        ext.push(u);
                    }
                }
            }
            ext.sort_unstable_by_key(|&u| rank_in[u as usize]);
            ext.dedup();
            let ext_rank = |u: u32| k + rank_in[u as usize];
            let local = |v: u32| if block_of[v as usize] == b as u32 { rank_in[v as usize] } else { ext_rank(v) };
            let node_of_rank = |r: u32| {
                if r < k { bl.nodes[r as usize] } else { ext[ext.partition_point(|&u| ext_rank(u) < r)] }
            };
            let el = eliminate(k as usize, bl.nodes.iter().flat_map(|&v| adj(v).iter().map(move |&u| (local(v), local(u)))));
            for (r, list) in el.up.into_iter().enumerate() {
                let v = bl.nodes[r] as usize;
                (up_at[v], up_len[v]) = (up_to.len() as u64, list.len() as u32);
                up_to.extend(list.into_iter().map(node_of_rank));
            }
            top_fill.extend(el.fill.into_iter().map(|(a, c)| (node_of_rank(a), node_of_rank(c))));
        }
        if let Some(t) = top {
            let bl = &plan[t];
            let k = bl.nodes.len();
            let r = |v: u32| rank_in[v as usize];
            let tb = t as u32;
            let block_of = &block_of;
            let edges = bl
                .nodes
                .iter()
                .flat_map(move |&v| adj(v).iter().filter(move |&&u| block_of[u as usize] == tb).map(move |&u| (r(v), r(u))))
                .chain(top_fill.iter().map(|&(a, c)| (r(a), r(c))));
            let el = eliminate(k, edges);
            assert!(el.fill.is_empty());
            for (rk, list) in el.up.into_iter().enumerate() {
                let v = bl.nodes[rk] as usize;
                (up_at[v], up_len[v]) = (up_to.len() as u64, list.len() as u32);
                up_to.extend(list.into_iter().map(|x| bl.nodes[x as usize]));
            }
        }
        drop((top_fill, adj_to, adj_first));
        // Heights inside each block: the parent is the lowest-ranked upper neighbour.
        let mut height = vec![0u32; n];
        for bl in plan {
            for &v in &bl.nodes {
                if up_len[v as usize] > 0 {
                    let p = up_to[up_at[v as usize] as usize];
                    if block_of[p as usize] == block_of[v as usize] {
                        height[p as usize] = height[p as usize].max(height[v as usize] + 1);
                    }
                }
            }
        }
        // Layout: the top (or whole) block first, then the cells in plan order.
        let mut order: Vec<usize> = Vec::new();
        order.extend(top);
        order.extend((0..plan.len()).filter(|&b| Some(b) != top));
        let mut layout_of = vec![NONE; n];
        let mut node_of = Vec::with_capacity(n);
        let mut blocks = Vec::new();
        for &b in &order {
            let bl = &plan[b];
            let l = bl.nodes.iter().map(|&v| height[v as usize]).max().unwrap_or(0) + 1;
            let mut nodes = bl.nodes.clone();
            // Level (top first), then rank descending.
            nodes.sort_by_key(|&v| (l - 1 - height[v as usize], std::cmp::Reverse(rank_in[v as usize])));
            let id_first = node_of.len() as u32;
            let mut level_first = vec![id_first];
            for lv in 0..l {
                let end = nodes.partition_point(|&v| l - 1 - height[v as usize] <= lv);
                level_first.push(id_first + end as u32);
            }
            for &v in &nodes {
                layout_of[v as usize] = node_of.len() as u32;
                node_of.push(v);
            }
            blocks.push(LayoutBlock { kind: bl.kind, id_first, nodes: nodes.len() as u32, level_first });
        }
        drop((block_of, rank_in, height));
        let mut up_first = Vec::with_capacity(n + 1);
        up_first.push(0u64);
        let mut up_head = Vec::with_capacity(up_to.len());
        let mut row: Vec<u32> = Vec::new();
        for &v in &node_of {
            let id = layout_of[v as usize];
            let at = up_at[v as usize] as usize;
            row.clear();
            row.extend(up_to[at..at + up_len[v as usize] as usize].iter().map(|&u| layout_of[u as usize]));
            row.sort_unstable();
            assert!(row.last().is_none_or(|&x| x < id), "an upper neighbour after its node");
            up_head.extend_from_slice(&row);
            up_first.push(up_head.len() as u64);
        }
        Hierarchy { layout_of, node_of, blocks, up_first, up_head }
    }

    pub fn len(&self) -> usize {
        self.node_of.len()
    }

    pub fn links(&self) -> u64 {
        self.up_head.len() as u64
    }

    pub fn row(&self, v: u32) -> std::ops::Range<usize> {
        self.up_first[v as usize] as usize..self.up_first[v as usize + 1] as usize
    }

    /// The link between layout ids a and b (a ≠ b), stored with the larger one.
    pub fn link(&self, a: u32, b: u32) -> Option<u64> {
        let (lo, hi) = (a.max(b), a.min(b));
        let r = self.row(lo);
        self.up_head[r.clone()].binary_search(&hi).ok().map(|i| (r.start + i) as u64)
    }

    /// The block holding layout id v.
    pub fn block_of(&self, v: u32) -> usize {
        self.blocks.partition_point(|b| b.id_first <= v) - 1
    }

    /// The longest top-down chain of levels (FORMAT §3, §5.1): the top block's levels plus
    /// the most of any cell, or the whole block's.
    pub fn levels(&self) -> u32 {
        let top = self.blocks.iter().filter(|b| b.kind != Kind::Cell).map(LayoutBlock::levels).sum::<u32>();
        top + self.blocks.iter().filter(|b| b.kind == Kind::Cell).map(LayoutBlock::levels).max().unwrap_or(0)
    }

    /// The elimination tree's height in levels: the nodes on its longest root-to-leaf path
    /// (FORMAT §5.2: the parent of v is its largest upper neighbour). It equals `levels`
    /// for a component built in one piece, and for one built as cells from the fine order.
    pub fn tree_levels(&self) -> u32 {
        let mut h = vec![1u32; self.len()];
        for v in (0..self.len() as u32).rev() {
            if let Some(&p) = self.up_head[self.row(v)].last() {
                h[p as usize] = h[p as usize].max(h[v as usize] + 1);
            }
        }
        h.into_iter().max().unwrap_or(0)
    }

    pub fn cells(&self) -> usize {
        self.blocks.iter().filter(|b| b.kind == Kind::Cell).count()
    }

    /// FORMAT §6.3: the triangles of every node, from the last layout id to the first (cells
    /// before the top, bottom levels first), relax the links above them.
    pub fn customize(&self, up: &mut [u16], down: &mut [u16]) {
        for v in (0..self.len() as u32).rev() {
            let r = self.row(v);
            for i in 1..r.len() {
                let (a, x) = (r.start + i, self.up_head[r.start + i]);
                for j in 0..i {
                    let (b, y) = (r.start + j, self.up_head[r.start + j]);
                    let t = self.link(x, y).expect("chordal") as usize;
                    up[t] = up[t].min(sat(down[a], up[b]));
                    down[t] = down[t].min(sat(down[b], up[a]));
                }
            }
        }
    }

    /// Fresh link weights: `INF` in both directions.
    pub fn unset(&self) -> (Vec<u16>, Vec<u16>) {
        (vec![INF; self.up_head.len()], vec![INF; self.up_head.len()])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The research toy town: 17 links, 5 levels, and its distances from node 0.
    #[test]
    fn toy_town() {
        let two_way =
            [(0, 1, 2), (3, 4, 4), (4, 5, 1), (6, 7, 5), (7, 8, 2), (0, 3, 1), (3, 6, 4), (1, 4, 2), (4, 7, 3), (2, 5, 3), (5, 8, 1)];
        let mut arcs: Vec<(u32, u32, u16)> = two_way.iter().flat_map(|&(u, v, t)| [(u, v, t), (v, u, t)]).collect();
        arcs.push((1, 2, 4));
        let edges: Vec<(u32, u32)> = arcs.iter().map(|a| (a.0, a.1)).collect();
        let plan = [Block { kind: Kind::Whole, nodes: vec![0, 6, 3, 2, 8, 5, 1, 7, 4] }];
        let h = Hierarchy::build(9, || edges.iter().copied(), &plan);
        assert_eq!((h.links(), h.levels()), (17, 5));
        let (mut up, mut down) = h.unset();
        for &(a, b, t) in &arcs {
            let (la, lb) = (h.layout_of[a as usize], h.layout_of[b as usize]);
            let k = h.link(la, lb).unwrap() as usize;
            if la > lb { up[k] = up[k].min(t) } else { down[k] = down[k].min(t) }
        }
        h.customize(&mut up, &mut down);
        // Upward sweep from node 0 by decreasing id, then downward by increasing id.
        let n = h.len();
        let mut d = vec![INF; n];
        d[h.layout_of[0] as usize] = 0;
        for v in (0..n as u32).rev() {
            for k in h.row(v) {
                let u = h.up_head[k] as usize;
                d[u] = d[u].min(sat(d[v as usize], up[k]));
            }
        }
        for v in 0..n as u32 {
            for k in h.row(v) {
                let u = h.up_head[k] as usize;
                d[v as usize] = d[v as usize].min(sat(d[u], down[k]));
            }
        }
        let by_node: Vec<u16> = (0..9).map(|x| d[h.layout_of[x] as usize]).collect();
        assert_eq!(by_node, [0, 2, 6, 1, 4, 5, 5, 7, 6]);
    }

    #[test]
    fn cells_link_only_inside_or_to_the_top() {
        // A 30 × 30 grid split into cells: every upper neighbour of a cell node is in its
        // cell or the top block.
        let k = 30u32;
        let mut edges = Vec::new();
        for i in 0..k {
            for j in 0..k {
                let v = i * k + j;
                if j + 1 < k {
                    edges.push((v, v + 1));
                }
                if i + 1 < k {
                    edges.push((v, v + k));
                }
            }
        }
        let tail: Vec<u32> = edges.iter().map(|e| e.0).collect();
        let head: Vec<u32> = edges.iter().map(|e| e.1).collect();
        let lat: Vec<f32> = (0..k * k).map(|v| (v / k) as f32 * 0.001).collect();
        let lon: Vec<f32> = (0..k * k).map(|v| (v % k) as f32 * 0.001).collect();
        let g = crate::order::Graph { n: (k * k) as usize, tail: &tail, head: &head, lat: &lat, lon: &lon };
        // From the fine order (the one-piece hierarchy), then from a coarsened graph.
        let whole = Hierarchy::build(g.n, || edges.iter().copied(), &crate::order::plan(&g, &crate::order::InertialFlow, usize::MAX, 0));
        let pairs = |h: &Hierarchy| {
            let mut p: Vec<(u32, u32)> =
                (0..h.len() as u32).flat_map(|v| h.row(v).map(move |kk| (v, kk))).map(|(v, kk)| (h.node_of[v as usize], h.node_of[h.up_head[kk] as usize])).collect();
            p.sort_unstable();
            p
        };
        for nd_max in [usize::MAX, 0] {
        let plan = crate::order::plan(&g, &crate::order::InertialFlow, 200, nd_max);
        let h = Hierarchy::build(g.n, || edges.iter().copied(), &plan);
        assert!(h.cells() >= 3);
        if nd_max == usize::MAX {
            assert_eq!(pairs(&h), pairs(&whole), "the links of the one-piece hierarchy");
            assert_eq!(h.tree_levels(), whole.tree_levels());
        }
        assert_eq!(whole.tree_levels(), whole.levels());
        for v in 0..h.len() as u32 {
            let bv = h.block_of(v);
            for kk in h.row(v) {
                let bu = h.block_of(h.up_head[kk]);
                assert!(bu == bv || h.blocks[bu].kind == Kind::Top);
                if bu == bv {
                    // Inside a block, an upper neighbour is in a smaller level.
                    let lvl = |x: u32| h.blocks[bv].level_first.partition_point(|&f| f <= x);
                    assert!(lvl(h.up_head[kk]) < lvl(v));
                }
            }
            // Chordality.
            let r = h.row(v);
            for i in r.clone() {
                for j in r.start..i {
                    assert!(h.link(h.up_head[i], h.up_head[j]).is_some());
                }
            }
        }
        }
    }
}
