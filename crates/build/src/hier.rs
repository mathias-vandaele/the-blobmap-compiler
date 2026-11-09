//! One hierarchy per expert (SPEC §5): the expert's induced graph, its plan (whole or
//! cells), contraction, link initialisation (FORMAT §6.2) and customization (§6.3).

use anyhow::{Result, ensure};
use blobmap_cch::Hierarchy;
use blobmap_format::NONE;
use blobmap_order::{Block, Fixed, Graph, InertialFlow, Orderer, plan};
use rayon::prelude::*;

use crate::family::Family;

/// The memory budget per node of a cell under construction (SPEC §5): a cell has at most
/// `memory_budget_bytes / BYTES_PER_NODE` nodes.
pub const BYTES_PER_NODE: u64 = 1024;

/// The memory per node of nested dissection on an expert's whole fine graph (measured: about
/// 330 bytes for France's CAR expert, 450 for ACTIVE, 770 for Hauts-de-France's, rounded up
/// for margin): an expert of at most
/// `MEMORY_LIMIT_BYTES / ND_BYTES_PER_NODE` nodes is cut into cells from its own order, a
/// larger one from a coarsened graph (SPEC §5).
pub const ND_BYTES_PER_NODE: u64 = 1024;

/// The peak RSS of a build (SPEC §10).
pub const MEMORY_LIMIT_BYTES: u64 = 62_000_000_000;

pub struct ExpertBuilt {
    /// The expert's nodes, ascending; a node's position here is its index in `h`.
    pub nodes: Vec<u32>,
    pub h: Hierarchy,
    /// Per metric, the link weights.
    pub up: Vec<Vec<u16>>,
    pub down: Vec<Vec<u16>>,
}

impl ExpertBuilt {
    /// The node at layout id l.
    pub fn node(&self, l: u32) -> u32 {
        self.nodes[self.h.node_of[l as usize] as usize]
    }
}

pub struct Built {
    pub experts: Vec<ExpertBuilt>,
    /// Each node's layout id in its expert.
    pub layout_of: Vec<u32>,
    /// The arcs inside one expert, CSR by tail node (`arc_count` order inside a node).
    pub out_first: Vec<u32>,
    pub out: Vec<u32>,
}

/// A node order fixed by the config (FORMAT: `build.toml [order]`): each node's rank, or
/// `NONE` if the list does not name it.
pub type FixedRanks = Vec<u32>;

/// A family split by expert: each expert's nodes and the arcs inside it.
pub struct Split {
    /// The nodes of expert x, ascending, are `members[first[x] .. first[x + 1]]`.
    members: Vec<u32>,
    first: Vec<usize>,
    /// A node's position among its expert's nodes.
    local_of: Vec<u32>,
    /// The arcs inside expert x are `by_expert[arc_first[x] .. arc_first[x + 1]]`.
    by_expert: Vec<u32>,
    arc_first: Vec<usize>,
    /// The arcs inside one expert, CSR by tail node.
    out_first: Vec<u32>,
    out: Vec<u32>,
}

impl Split {
    fn new(f: &dyn Family) -> Split {
        let e = f.experts();
        let n = f.nodes();
        let mut members: Vec<(u32, u32)> = (0..n as u32).map(|v| (e.of[v as usize], v)).collect();
        members.sort_unstable();
        let mut first = vec![0usize; e.len() + 1];
        for &(x, _) in &members {
            first[x as usize + 1] += 1;
        }
        for i in 0..e.len() {
            first[i + 1] += first[i];
        }
        let members: Vec<u32> = members.into_iter().map(|m| m.1).collect();
        let mut local_of = vec![0u32; n];
        for x in 0..e.len() {
            for (i, &v) in members[first[x]..first[x + 1]].iter().enumerate() {
                local_of[v as usize] = i as u32;
            }
        }
        // Arcs inside one expert, by expert and by tail.
        let inner: Vec<u32> = (0..f.arc_count() as u32)
            .filter(|&a| {
                let (t, h) = f.arc(a as usize);
                e.of[t as usize] == e.of[h as usize]
            })
            .collect();
        let mut by_expert: Vec<(u32, u32)> = inner.iter().map(|&a| (e.of[f.arc(a as usize).0 as usize], a)).collect();
        by_expert.sort_unstable();
        let mut arc_first = vec![0usize; e.len() + 1];
        for &(x, _) in &by_expert {
            arc_first[x as usize + 1] += 1;
        }
        for i in 0..e.len() {
            arc_first[i + 1] += arc_first[i];
        }
        let by_expert: Vec<u32> = by_expert.into_iter().map(|x| x.1).collect();
        let mut out_first = vec![0u32; n + 1];
        for &a in &inner {
            out_first[f.arc(a as usize).0 as usize + 1] += 1;
        }
        for i in 0..n {
            out_first[i + 1] += out_first[i];
        }
        let mut fill = out_first.clone();
        let mut out = vec![0u32; inner.len()];
        for &a in &inner {
            let t = f.arc(a as usize).0 as usize;
            out[fill[t] as usize] = a;
            fill[t] += 1;
        }
        Split { members, first, local_of, by_expert, arc_first, out_first, out }
    }

    /// Expert x: its nodes, its arcs, and its arcs as local node pairs without loops.
    fn expert(&self, f: &dyn Family, x: usize) -> (Vec<u32>, &[u32], Vec<(u32, u32)>) {
        let nodes = self.members[self.first[x]..self.first[x + 1]].to_vec();
        let arcs = &self.by_expert[self.arc_first[x]..self.arc_first[x + 1]];
        let edges = arcs.iter().map(|&a| self.local(f, a)).filter(|(t, h)| t != h).collect();
        (nodes, arcs, edges)
    }

    fn local(&self, f: &dyn Family, a: u32) -> (u32, u32) {
        let (t, h) = f.arc(a as usize);
        (self.local_of[t as usize], self.local_of[h as usize])
    }
}

/// Every expert's blocks (SPEC §5), in their contraction order.
pub struct Ordered {
    split: Split,
    plans: Vec<Vec<Block>>,
}

/// Orders every expert: in one piece if it has at most `max_cell` nodes, else as cells, cut
/// from its fine graph's order if it has at most `nd_max` nodes.
pub fn order(f: &dyn Family, max_cell: usize, nd_max: usize, fixed: Option<&FixedRanks>) -> Result<Ordered> {
    let split = Split::new(f);
    let plans: Vec<Result<Vec<Block>>> = (0..f.experts().len())
        .into_par_iter()
        .map(|x| {
            let (nodes, _, edges) = split.expert(f, x);
            let k = nodes.len();
            let tail: Vec<u32> = edges.iter().map(|e| e.0).collect();
            let head: Vec<u32> = edges.iter().map(|e| e.1).collect();
            let (lat, lon): (Vec<f32>, Vec<f32>) = nodes.iter().map(|&v| f.coord(v)).unzip();
            let g = Graph { n: k, tail: &tail, head: &head, lat: &lat, lon: &lon };
            Ok(match fixed {
                Some(rank) => {
                    ensure!(nodes.iter().all(|&v| rank[v as usize] != NONE), "{} expert {x}: the fixed order misses a node", f.name());
                    ensure!(k <= max_cell, "{} expert {x}: a fixed order cannot be built as cells", f.name());
                    let mut order: Vec<u32> = (0..k as u32).collect();
                    order.sort_by_key(|&i| rank[nodes[i as usize] as usize]);
                    plan(&g, &Fixed(order), max_cell, nd_max)
                }
                None => plan(&g, &InertialFlow as &dyn Orderer, max_cell, nd_max),
            })
        })
        .collect();
    Ok(Ordered { split, plans: plans.into_iter().collect::<Result<_>>()? })
}

/// Contracts and customizes every expert in its order.
pub fn build(f: &dyn Family, o: Ordered) -> Result<Built> {
    let Ordered { split, plans } = o;
    let n = f.nodes();
    let metrics = f.metrics().len();
    let experts: Vec<Result<ExpertBuilt>> = plans
        .into_par_iter()
        .enumerate()
        .map(|(x, blocks)| {
            let (nodes, arcs, edges) = split.expert(f, x);
            let k = nodes.len();
            let local = |a: u32| split.local(f, a);
            let h = Hierarchy::build(k, &edges, &blocks);
            let (mut up, mut down) = (Vec::with_capacity(metrics), Vec::with_capacity(metrics));
            for m in 0..metrics {
                let (mut u, mut d) = h.unset();
                for &a in arcs {
                    let (t, hd) = local(a);
                    if t == hd {
                        continue;
                    }
                    let (lt, lh) = (h.layout_of[t as usize], h.layout_of[hd as usize]);
                    let l = h.link(lt, lh).expect("an arc has a link") as usize;
                    let w = f.arc_weight(m, a as usize);
                    if lt > lh {
                        u[l] = u[l].min(w);
                    } else {
                        d[l] = d[l].min(w);
                    }
                }
                h.customize(&mut u, &mut d);
                up.push(u);
                down.push(d);
            }
            Ok(ExpertBuilt { nodes, h, up, down })
        })
        .collect();
    let experts: Vec<ExpertBuilt> = experts.into_iter().collect::<Result<_>>()?;
    let mut layout_of = vec![NONE; n];
    for x in &experts {
        for (i, &v) in x.nodes.iter().enumerate() {
            layout_of[v as usize] = x.h.layout_of[i];
        }
    }
    Ok(Built { experts, layout_of, out_first: split.out_first, out: split.out })
}
