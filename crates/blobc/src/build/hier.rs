//! The family's order (SPEC §5): the family split by component, and each component's plan
//! (whole or cells). `blocks` contracts, customizes and writes them; each component is one
//! tree of the family's elimination forest.

use std::path::Path;

use anyhow::{Result, ensure};
use blobmap_model::NONE;
use rayon::prelude::*;

use crate::build::family::Family;
use crate::osm::column::{Column, ColumnMut};
use crate::order::{Block, Fixed, Graph, InertialFlow, Orderer, plan};

/// The memory budget per node of a cell under construction (SPEC §5): a cell has at most
/// `memory_budget_bytes / BYTES_PER_NODE` nodes.
pub const BYTES_PER_NODE: u64 = 1024;

/// The memory per node of nested dissection on a component's whole fine graph, at the order
/// stage's peak (measured on France: 10.9 GB for 20.5 M CAR nodes, about 535 bytes): a
/// component of at most `order_memory_bytes / ND_BYTES_PER_NODE` nodes (build.toml) is cut
/// into cells from its own order, a larger one from a coarsened graph (SPEC §5).
pub const ND_BYTES_PER_NODE: u64 = 535;

/// A node order fixed by the config (FORMAT: `build.toml [order]`): each node's rank, or
/// `NONE` if the list does not name it.
pub type FixedRanks = Vec<u32>;

/// A family split by component: each component's nodes and the arcs inside it, in columns
/// mapped from the work directory.
pub struct Split {
    /// The nodes of component x, ascending, are `members[first[x] .. first[x + 1]]`.
    members: Column<u32>,
    pub(crate) first: Vec<usize>,
    /// A node's position among its component's nodes.
    local_of: Column<u32>,
    /// The arcs inside component x are `by_component[arc_first[x] .. arc_first[x + 1]]`.
    by_component: Column<u64>,
    arc_first: Vec<usize>,
    /// The arcs inside one component, CSR by tail node.
    pub(crate) out_first: Column<u64>,
    pub(crate) out: Column<u64>,
}

impl Split {
    /// Counting sorts into columns of `dir`: memory holds two words per component.
    fn new(f: &dyn Family, dir: &Path) -> Result<Split> {
        let e = f.components();
        let (n, k) = (f.nodes(), e.len());
        let d = Some(dir);
        // Nodes by component, ascending inside one; a node's position there.
        let mut first = vec![0usize; k + 1];
        for &x in &e.of {
            first[x as usize + 1] += 1;
        }
        for i in 0..k {
            first[i + 1] += first[i];
        }
        let (mut members, mut local_of) = (ColumnMut::<u32>::zeroed(d, "members", n)?, ColumnMut::<u32>::zeroed(d, "local_of", n)?);
        let mut at = first.clone();
        for v in 0..n {
            let x = e.of[v] as usize;
            members[at[x]] = v as u32;
            local_of[v] = (at[x] - first[x]) as u32;
            at[x] += 1;
        }
        // Arcs inside one component, by component (ascending inside one) and by tail.
        let inside = |a: usize| {
            let (t, h) = f.arc(a);
            (e.of[t as usize] == e.of[h as usize]).then_some((t, e.of[t as usize] as usize))
        };
        let mut arc_first = vec![0usize; k + 1];
        let mut out_first = ColumnMut::<u64>::zeroed(d, "out_first", n + 1)?;
        for a in 0..f.arc_count() {
            if let Some((t, x)) = inside(a) {
                arc_first[x + 1] += 1;
                out_first[t as usize + 1] += 1;
            }
        }
        for i in 0..k {
            arc_first[i + 1] += arc_first[i];
        }
        for i in 0..n {
            out_first[i + 1] += out_first[i];
        }
        let m = arc_first[k];
        let (mut by_component, mut out) = (ColumnMut::<u64>::zeroed(d, "by_component", m)?, ColumnMut::<u64>::zeroed(d, "out", m)?);
        // Each start moved along as its arcs are placed, then shifted back.
        at.copy_from_slice(&arc_first);
        for a in 0..f.arc_count() {
            if let Some((t, x)) = inside(a) {
                by_component[at[x]] = a as u64;
                at[x] += 1;
                let p = &mut out_first[t as usize];
                out[*p as usize] = a as u64;
                *p += 1;
            }
        }
        drop(at);
        for i in (1..=n).rev() {
            out_first[i] = out_first[i - 1];
        }
        out_first[0] = 0;
        Ok(Split {
            members: members.freeze()?,
            first,
            local_of: local_of.freeze()?,
            by_component: by_component.freeze()?,
            arc_first,
            out_first: out_first.freeze()?,
            out: out.freeze()?,
        })
    }

    /// Component x: its nodes and its arcs.
    pub(crate) fn component(&self, x: usize) -> (Vec<u32>, &[u64]) {
        (self.members[self.first[x]..self.first[x + 1]].to_vec(), &self.by_component[self.arc_first[x]..self.arc_first[x + 1]])
    }

    /// Arcs as local node pairs without loops.
    pub(crate) fn edges<'a>(&'a self, f: &'a dyn Family, arcs: &'a [u64]) -> impl Iterator<Item = (u32, u32)> + 'a {
        arcs.iter().map(move |&a| self.local(f, a)).filter(|(t, h)| t != h)
    }

    pub(crate) fn local(&self, f: &dyn Family, a: u64) -> (u32, u32) {
        let (t, h) = f.arc(a as usize);
        (self.local_of[t as usize], self.local_of[h as usize])
    }
}

/// Every component's blocks (SPEC §5), in their contraction order.
pub struct Ordered {
    pub(crate) split: Split,
    pub(crate) plans: Vec<Vec<Block>>,
}

/// Orders every component: in one piece if it has at most `max_cell` nodes, else as cells, cut
/// from its fine graph's order if it has at most `nd_max` nodes. The family's split by component
/// is kept in columns under `dir`.
pub fn order(f: &dyn Family, max_cell: usize, nd_max: usize, fixed: Option<&FixedRanks>, dir: &Path) -> Result<Ordered> {
    let split = Split::new(f, dir)?;
    crate::build::mem::phase("split");
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
    crate::build::mem::phase("plans");
    Ok(Ordered { split, plans: plans.into_iter().collect::<Result<_>>()? })
}

