//! Packs (SPEC §8; FORMAT §4): where every block's rows, links and level bounds go, and the
//! pack-local index of every arc.

use anyhow::{Result, ensure};
use blobmap_format::NONE;
use blobmap_order::Kind;

use crate::family::Family;
use crate::hier::Built;

pub struct PackPlan {
    /// (component, block) in row order.
    pub blocks: Vec<(u32, u32)>,
    /// Nodes in row order.
    pub rows: Vec<u32>,
    pub links: u64,
    /// Arcs in stored order, CSR by row.
    pub arcs: Vec<u32>,
    pub arc_first: Vec<u32>,
}

/// Where one block is stored.
#[derive(Clone, Copy, Debug)]
pub struct Place {
    pub pack: u32,
    pub row_first: u32,
    /// Pack-local index of the block's first link.
    pub link_first: u64,
    pub level_base: u32,
}

pub struct Layout {
    pub packs: Vec<PackPlan>,
    /// Per component, per block.
    pub place: Vec<Vec<Place>>,
    /// Pack-local index of every arc inside a component, `NONE` for the others.
    pub arc_local: Vec<u32>,
}

impl Layout {
    /// Pack-local index of the link between layout ids a and c of component x, in the pack of
    /// its lower node (FORMAT §4).
    pub fn link(&self, b: &Built, x: u32, a: u32, c: u32) -> u32 {
        let e = &b.components[x as usize];
        let (a, c) = (a - b.first[x as usize], c - b.first[x as usize]);
        let lo = a.max(c);
        let blk = e.h.block_of(lo);
        let k = e.h.link(a, c).expect("a link");
        let p = self.place[x as usize][blk];
        (p.link_first + k - e.h.up_first[e.h.blocks[blk].id_first as usize]) as u32
    }
}

const LIMIT: u64 = 1 << 32;

/// Every block of a component built as cells, or of one with at least `pack_max_nodes` nodes,
/// is a pack of its own. The other components share packs, in index order, of at most
/// `pack_max_nodes` nodes each.
pub fn plan(f: &dyn Family, b: &Built, pack_max_nodes: u64) -> Result<Layout> {
    let n = b.components.len();
    let small = |x: usize| (b.components[x].nodes.len() as u64) < pack_max_nodes && b.components[x].h.blocks.len() == 1;
    // The packs' blocks, (component, block) in row order.
    let mut members: Vec<Vec<(u32, u32)>> = Vec::new();
    let mut x = 0;
    while x < n {
        if small(x) {
            let (mut end, mut sum) = (x + 1, b.components[x].nodes.len() as u64);
            while end < n && small(end) && sum + b.components[end].nodes.len() as u64 <= pack_max_nodes {
                sum += b.components[end].nodes.len() as u64;
                end += 1;
            }
            members.push((x as u32..end as u32).map(|c| (c, 0)).collect());
            x = end;
        } else {
            members.extend((0..b.components[x].h.blocks.len() as u32).map(|blk| vec![(x as u32, blk)]));
            x += 1;
        }
    }
    let mut place: Vec<Vec<Place>> = b.components.iter().map(|e| Vec::with_capacity(e.h.blocks.len())).collect();
    let mut packs: Vec<PackPlan> = Vec::new();
    let mut arc_local = vec![NONE; f.arc_count()];
    for blocks in members {
        let pk = packs.len() as u32;
        let mut p = PackPlan { blocks: blocks.clone(), rows: Vec::new(), links: 0, arcs: Vec::new(), arc_first: vec![0] };
        let mut levels = 0u32;
        for &(x, blk) in &blocks {
            let e = &b.components[x as usize];
            let lb = &e.h.blocks[blk as usize];
            place[x as usize].push(Place { pack: pk, row_first: p.rows.len() as u32, link_first: p.links, level_base: levels });
            levels += lb.level_first.len() as u32;
            for l in lb.id_first..lb.id_first + lb.nodes {
                let v = e.node(l);
                p.rows.push(v);
                p.links += e.h.row(l).len() as u64;
                let mut arcs: Vec<u32> = b.out[b.out_first[v as usize] as usize..b.out_first[v as usize + 1] as usize].to_vec();
                arcs.sort_by_key(|&a| (b.layout_of[f.arc(a as usize).1 as usize], f.arc_key(a as usize)));
                for a in arcs {
                    arc_local[a as usize] = p.arcs.len() as u32;
                    p.arcs.push(a);
                }
                p.arc_first.push(p.arcs.len() as u32);
            }
        }
        let (rows, arcs) = (p.rows.len() as u64, p.arcs.len() as u64);
        ensure!(
            rows < LIMIT && p.links < LIMIT && arcs < LIMIT,
            "{} pack {pk}: {rows} rows, {} links, {arcs} arcs: a count reaches 2^32 (FORMAT §4)",
            f.name(),
            p.links
        );
        packs.push(p);
    }
    debug_assert!(b.components.iter().all(|e| e.h.blocks.iter().all(|bl| bl.kind == Kind::Whole) || e.h.blocks[0].kind == Kind::Top));
    Ok(Layout { packs, place, arc_local })
}
