//! Manifest entries, expert groups and packs (SPEC §2, §8; FORMAT §3, §4): where every
//! block's rows, links and level bounds go, and the pack-local index of every arc.

use anyhow::{Result, ensure};
use blobmap_format::NONE;
use blobmap_order::Kind;

use crate::family::Family;
use crate::hier::Built;

pub struct EntryPlan {
    pub group: bool,
    /// Its experts, `first .. first + count`.
    pub first: u32,
    pub count: u32,
    pub pack_first: u32,
    pub packs: u32,
}

pub struct PackPlan {
    pub entry: u32,
    /// (expert, block) in row order.
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
    pub entries: Vec<EntryPlan>,
    pub packs: Vec<PackPlan>,
    /// Per expert, per block.
    pub place: Vec<Vec<Place>>,
    pub entry_of: Vec<u32>,
    /// Pack-local index of every arc inside an expert, `NONE` for the others.
    pub arc_local: Vec<u32>,
}

impl Layout {
    /// Pack-local index of the link between layout ids a and b of expert x, in the pack of
    /// its lower node (FORMAT §4).
    pub fn link(&self, b: &Built, x: u32, a: u32, c: u32) -> u32 {
        let e = &b.experts[x as usize];
        let lo = a.max(c);
        let blk = e.h.block_of(lo);
        let k = e.h.link(a, c).expect("a link");
        let p = self.place[x as usize][blk];
        (p.link_first + k - e.h.up_first[e.h.blocks[blk].id_first as usize]) as u32
    }
}

const LIMIT: u64 = 1 << 32;

/// Experts below `group_max_nodes` nodes that are built in one piece are packed, in index
/// order, into groups of at most `group_max_nodes` nodes; a run of one stays an expert.
pub fn plan(f: &dyn Family, b: &Built, group_max_nodes: u64) -> Result<Layout> {
    let n = b.experts.len();
    let mut entries: Vec<EntryPlan> = Vec::new();
    let small = |x: usize| (b.experts[x].nodes.len() as u64) < group_max_nodes && b.experts[x].h.blocks.len() == 1;
    let mut x = 0;
    while x < n {
        let mut end = x + 1;
        if small(x) {
            let mut sum = b.experts[x].nodes.len() as u64;
            while end < n && small(end) && sum + b.experts[end].nodes.len() as u64 <= group_max_nodes {
                sum += b.experts[end].nodes.len() as u64;
                end += 1;
            }
        }
        let group = end - x > 1;
        let packs = if group { 1 } else { b.experts[x].h.blocks.len() as u32 };
        let pack_first = entries.last().map_or(0, |e| e.pack_first + e.packs);
        entries.push(EntryPlan { group, first: x as u32, count: (end - x) as u32, pack_first, packs });
        x = end;
    }
    let mut entry_of = vec![0u32; n];
    let mut place: Vec<Vec<Place>> = b.experts.iter().map(|e| Vec::with_capacity(e.h.blocks.len())).collect();
    let mut packs: Vec<PackPlan> = Vec::new();
    let mut arc_local = vec![NONE; f.arc_count()];
    for (k, en) in entries.iter().enumerate() {
        let mut members: Vec<Vec<(u32, u32)>> = Vec::new();
        for x in en.first..en.first + en.count {
            entry_of[x as usize] = k as u32;
            for blk in 0..b.experts[x as usize].h.blocks.len() as u32 {
                if en.group && !members.is_empty() {
                    members[0].push((x, blk));
                } else {
                    members.push(vec![(x, blk)]);
                }
            }
        }
        for blocks in members {
            let pk = packs.len() as u32;
            let mut p = PackPlan { entry: k as u32, blocks: blocks.clone(), rows: Vec::new(), links: 0, arcs: Vec::new(), arc_first: vec![0] };
            let mut levels = 0u32;
            for &(x, blk) in &blocks {
                let e = &b.experts[x as usize];
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
    }
    debug_assert!(entries.iter().zip(entries.iter().skip(1)).all(|(a, c)| a.pack_first + a.packs == c.pack_first));
    debug_assert!(b.experts.iter().all(|e| e.h.blocks.iter().all(|bl| bl.kind == Kind::Whole) || e.h.blocks[0].kind == Kind::Top));
    Ok(Layout { entries, packs, place, entry_of, arc_local })
}
