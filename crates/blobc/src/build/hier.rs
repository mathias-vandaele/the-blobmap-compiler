//! The family's hierarchy (SPEC §5), built one component at a time: each component's induced
//! graph, its plan (whole or cells), contraction, link initialisation (FORMAT §6.2) and
//! customization (§6.3). Each component is one tree of the family's elimination forest.

use std::path::Path;

use anyhow::{Result, ensure};
use blobmap_format::NONE;
use rayon::prelude::*;

use crate::build::family::Family;
use crate::cch::Hierarchy;
use crate::osm::column::Column;
use crate::order::{Block, Fixed, Graph, InertialFlow, Orderer, plan};

/// The memory budget per node of a cell under construction (SPEC §5): a cell has at most
/// `memory_budget_bytes / BYTES_PER_NODE` nodes.
pub const BYTES_PER_NODE: u64 = 1024;

/// The memory per node of nested dissection on a component's whole fine graph, at the order
/// stage's peak (measured on France: 10.9 GB for 20.5 M CAR nodes, about 535 bytes): a
/// component of at most `order_memory_bytes / ND_BYTES_PER_NODE` nodes (build.toml) is cut
/// into cells from its own order, a larger one from a coarsened graph (SPEC §5).
pub const ND_BYTES_PER_NODE: u64 = 535;

pub struct ComponentBuilt {
    /// The component's nodes, ascending; a node's position here is its index in `h`.
    pub nodes: Vec<u32>,
    pub h: Hierarchy,
    /// Per metric, the link weights.
    pub up: Vec<Vec<u16>>,
    pub down: Vec<Vec<u16>>,
}

impl ComponentBuilt {
    /// The node at layout id l.
    pub fn node(&self, l: u32) -> u32 {
        self.nodes[self.h.node_of[l as usize] as usize]
    }
}

pub struct Built {
    pub components: Vec<ComponentBuilt>,
    /// The family's layout id of each component's local id 0: components are consecutive
    /// ranges of layout ids, in index order (FORMAT §4).
    pub first: Vec<u32>,
    /// Each node's layout id in the family.
    pub layout_of: Vec<u32>,
    /// The arcs inside one component, CSR by tail node (`arc_count` order inside a node). Arc
    /// indices are the family's, u64: a family's turns or roads may pass 2^32.
    pub out_first: Column<u64>,
    pub out: Column<u64>,
}

/// A node order fixed by the config (FORMAT: `build.toml [order]`): each node's rank, or
/// `NONE` if the list does not name it.
pub type FixedRanks = Vec<u32>;

/// A family split by component: each component's nodes and the arcs inside it, in columns
/// mapped from the work directory.
pub struct Split {
    /// The nodes of component x, ascending, are `members[first[x] .. first[x + 1]]`.
    members: Column<u32>,
    first: Vec<usize>,
    /// A node's position among its component's nodes.
    local_of: Column<u32>,
    /// The arcs inside component x are `by_component[arc_first[x] .. arc_first[x + 1]]`.
    by_component: Column<u64>,
    arc_first: Vec<usize>,
    /// The arcs inside one component, CSR by tail node.
    out_first: Column<u64>,
    out: Column<u64>,
}

impl Split {
    fn new(f: &dyn Family, dir: &Path) -> Result<Split> {
        let e = f.components();
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
        // Arcs inside one component, by component and by tail.
        let inner: Vec<u64> = (0..f.arc_count() as u64)
            .filter(|&a| {
                let (t, h) = f.arc(a as usize);
                e.of[t as usize] == e.of[h as usize]
            })
            .collect();
        let mut by_component: Vec<(u32, u64)> = inner.iter().map(|&a| (e.of[f.arc(a as usize).0 as usize], a)).collect();
        by_component.sort_unstable();
        let mut arc_first = vec![0usize; e.len() + 1];
        for &(x, _) in &by_component {
            arc_first[x as usize + 1] += 1;
        }
        for i in 0..e.len() {
            arc_first[i + 1] += arc_first[i];
        }
        let by_component: Vec<u64> = by_component.into_iter().map(|x| x.1).collect();
        let mut out_first = vec![0u64; n + 1];
        for &a in &inner {
            out_first[f.arc(a as usize).0 as usize + 1] += 1;
        }
        for i in 0..n {
            out_first[i + 1] += out_first[i];
        }
        let mut fill = out_first.clone();
        let mut out = vec![0u64; inner.len()];
        for &a in &inner {
            let t = f.arc(a as usize).0 as usize;
            out[fill[t] as usize] = a;
            fill[t] += 1;
        }
        fn col<T: crate::osm::column::Plain>(v: Vec<T>, dir: &Path, name: &str) -> Result<Column<T>> {
            let mut c = Column::from(v);
            c.spill(Some(dir), name)?;
            Ok(c)
        }
        let (members, local_of, by_component) = (col(members, dir, "members")?, col(local_of, dir, "local_of")?, col(by_component, dir, "by_component")?);
        Ok(Split { members, first, local_of, by_component, arc_first, out_first: col(out_first, dir, "out_first")?, out: col(out, dir, "out")? })
    }

    /// Component x: its nodes and its arcs.
    fn component(&self, x: usize) -> (Vec<u32>, &[u64]) {
        (self.members[self.first[x]..self.first[x + 1]].to_vec(), &self.by_component[self.arc_first[x]..self.arc_first[x + 1]])
    }

    /// Arcs as local node pairs without loops.
    fn edges<'a>(&'a self, f: &'a dyn Family, arcs: &'a [u64]) -> impl Iterator<Item = (u32, u32)> + 'a {
        arcs.iter().map(move |&a| self.local(f, a)).filter(|(t, h)| t != h)
    }

    fn local(&self, f: &dyn Family, a: u64) -> (u32, u32) {
        let (t, h) = f.arc(a as usize);
        (self.local_of[t as usize], self.local_of[h as usize])
    }
}

/// Every component's blocks (SPEC §5), in their contraction order.
pub struct Ordered {
    split: Split,
    plans: Vec<Vec<Block>>,
}

/// Orders every component: in one piece if it has at most `max_cell` nodes, else as cells, cut
/// from its fine graph's order if it has at most `nd_max` nodes. The family's split by component
/// is kept in columns under `dir`.
pub fn order(f: &dyn Family, max_cell: usize, nd_max: usize, fixed: Option<&FixedRanks>, dir: &Path) -> Result<Ordered> {
    let split = Split::new(f, dir)?;
    let plans: Vec<Result<Vec<Block>>> = (0..f.components().len())
        .into_par_iter()
        .map(|x| {
            let (nodes, arcs) = split.component(x);
            let k = nodes.len();
            let tail: Vec<u32> = split.edges(f, arcs).map(|e| e.0).collect();
            let head: Vec<u32> = split.edges(f, arcs).map(|e| e.1).collect();
            let (lat, mut lon): (Vec<f32>, Vec<f32>) = nodes.iter().map(|&v| f.coord(v)).unzip();
            // A component across the antimeridian is cut as one region (SPEC §4).
            crate::osm::geo::unwrap_lons(&mut lon);
            let g = Graph { n: k, tail: &tail, head: &head, lat: &lat, lon: &lon };
            Ok(match fixed {
                Some(rank) => {
                    ensure!(nodes.iter().all(|&v| rank[v as usize] != NONE), "{} component {x}: the fixed order misses a node", f.name());
                    ensure!(k <= max_cell, "{} component {x}: a fixed order cannot be built as cells", f.name());
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

/// Contracts and customizes every component in its order.
pub fn build(f: &dyn Family, o: Ordered) -> Result<Built> {
    let Ordered { split, plans } = o;
    let n = f.nodes();
    let metrics = f.metrics().len();
    let components: Vec<Result<ComponentBuilt>> = plans
        .into_par_iter()
        .enumerate()
        .map(|(x, blocks)| {
            let (nodes, arcs) = split.component(x);
            let k = nodes.len();
            let local = |a: u64| split.local(f, a);
            let h = Hierarchy::build(k, || split.edges(f, arcs), &blocks);
            drop(blocks);
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
            Ok(ComponentBuilt { nodes, h, up, down })
        })
        .collect();
    let components: Vec<ComponentBuilt> = components.into_iter().collect::<Result<_>>()?;
    // Layout ids are U32, NONE excluded (FORMAT §1, §4).
    ensure!((n as u64) < NONE as u64, "{}: {n} layout nodes, beyond the U32 layout ids (FORMAT §4)", f.name());
    let (mut first, mut layout_of) = (Vec::with_capacity(components.len()), vec![NONE; n]);
    let mut next = 0u32;
    for x in &components {
        first.push(next);
        for (i, &v) in x.nodes.iter().enumerate() {
            layout_of[v as usize] = next + x.h.layout_of[i];
        }
        next += x.nodes.len() as u32;
    }
    Ok(Built { components, first, layout_of, out_first: split.out_first, out: split.out })
}
