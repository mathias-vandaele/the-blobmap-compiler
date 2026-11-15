//! A family's hierarchy built and written block by block (SPEC §5, §10 memory; FORMAT §4–§6):
//! each pack is written as soon as its blocks are customized, and only what the family
//! tables, the vocabulary and the reports read stays.
//!
//! A component built in one piece is contracted and customized at once (`Hierarchy`). A
//! component built as cells goes in four phases, so that at most a few cells are held at a
//! time: (1) each cell is eliminated with its top-block neighbours as external nodes, its
//! upper neighbours kept in files; (2) the top block is eliminated with the cliques the cells
//! leave, and laid out; (3) each cell's rows are read back in layout ids, its links set from
//! its arcs and customized, its triangles among top nodes relaxing the top's links, and its
//! pack written; (4) the top block is customized and its pack written. The links, levels and
//! weights are those of contracting and customizing the whole component in layout order:
//! every link's weight is the minimum over its arcs and the triangles below it, and each
//! node's links are final before its own triangles are taken, whatever the order of the cells.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::atomic::{AtomicU16, Ordering::Relaxed};

use anyhow::{Result, ensure};
use blobmap_model::{Hash, INF, ModelWriter, NONE, Tensor, Unit, sat};
use rayon::prelude::*;
use serde_json::json;

use crate::build::family::Family;
use crate::build::hier::{Ordered, Split};
use crate::cch::{Hierarchy, customize_rows, row_link};
use crate::order::elim::eliminate;
use crate::order::{Block, Kind};
use crate::osm::column::{Column, ColumnMut, ColumnWriter};

/// One block, contracted and customized, in its component's layout ids (FORMAT §4).
pub struct BlockOut {
    /// Its position in the component's layout: the top (or whole) block 0, then the cells.
    pub index: u32,
    pub kind: Kind,
    pub id_first: u32,
    /// L + 1 ascending layout ids (FORMAT §5.1).
    pub level_first: Vec<u32>,
    /// Its rows' nodes (family node ids), in layout order.
    pub rows: Vec<u32>,
    /// Upper neighbours, ascending, CSR by row.
    pub up_first: Vec<u64>,
    pub up_head: Vec<u32>,
    /// Per metric, the link weights.
    pub up: Vec<Vec<u16>>,
    pub down: Vec<Vec<u16>>,
}

impl BlockOut {
    fn levels(&self) -> u32 {
        (self.level_first.len() - 1) as u32
    }

    fn links(&self) -> u64 {
        self.up_head.len() as u64
    }

    /// The position among the block's links of the link from layout id `lo` (a row of the
    /// block) up to `hi`.
    fn link(&self, lo: u32, hi: u32) -> Option<u64> {
        row_link(&self.up_first, &self.up_head, lo - self.id_first, hi)
    }

    fn contains(&self, l: u32) -> bool {
        l >= self.id_first && l < self.id_first + self.rows.len() as u32
    }
}

/// Where one block is stored (FORMAT §5.1, the block table).
#[derive(Clone, Copy, Debug, Default)]
struct Place {
    kind: u8,
    id_first: u32,
    nodes: u32,
    levels: u32,
    pack: u32,
    row_first: u32,
    level_base: u32,
}

/// The largest pack of a family (FORMAT §4 limits), for the build report.
#[derive(Clone, Debug, Default)]
pub struct PackSize {
    pub pack: u32,
    pub rows: u64,
    pub links: u64,
    pub arcs: u64,
}

/// What a family's road units leave for the vocabulary, the manifest and the reports.
pub struct Written {
    /// Each node's layout id in the family.
    pub layout_of: Column<u32>,
    /// Each arc's index among its pack's arcs (FORMAT §5.4), `NONE` for an arc between two
    /// components; empty unless asked for.
    pub arc_local: Column<u32>,
    pub hashes: BTreeMap<String, Hash>,
    pub largest: PackSize,
    pub links: u64,
    pub components: u64,
    pub packs: u64,
    pub bytes: u64,
    pub hierarchies: Vec<serde_json::Value>,
}

const LIMIT: u64 = 1 << 32;

/// The family's packs and tables, as they are built.
struct FamilyWriter<'a> {
    f: &'a dyn Family,
    split: &'a Split,
    /// The family layout id of each component's id 0.
    first: Vec<u32>,
    layout_of: ColumnMut<u32>,
    arc_local: ColumnMut<u32>,
    writer: &'a mut ModelWriter,
    pack_slot: u64,
    /// Per component, its blocks' places, by layout position.
    places: Vec<Vec<Place>>,
    hashes: BTreeMap<String, Hash>,
    largest: PackSize,
    links: u64,
    packs: u64,
    bytes: u64,
}

impl FamilyWriter<'_> {
    /// Writes pack k of `blocks` (component, block), in row order. `outside(arc)` is the
    /// pack-local index of an arc's link that lives in another pack (FORMAT §4: the pack of
    /// its lower node).
    fn pack(&mut self, k: u32, blocks: &[(u32, &BlockOut)], outside: &dyn Fn(u64) -> u32) -> Result<()> {
        let (f, split) = (self.f, self.split);
        let mut u = Unit::default();
        let p = format!("road/{}/p{k}", f.name());
        let metrics = f.metrics().len();
        let (mut level_first, mut up_first, mut up_head) = (Vec::new(), vec![0u32], Vec::new());
        let (mut up, mut down): (Vec<Vec<u16>>, Vec<Vec<u16>>) = (vec![Vec::new(); metrics], vec![Vec::new(); metrics]);
        let (mut rows, mut arcs, mut arc_first) = (Vec::new(), Vec::new(), vec![0u32]);
        // Each block's first family layout id and its first link in the pack.
        let mut starts: Vec<(u32, u64)> = Vec::with_capacity(blocks.len());
        let mut levels = 0u32;
        for &(x, b) in blocks {
            let base = self.first[x as usize];
            let pl = &mut self.places[x as usize][b.index as usize];
            (pl.pack, pl.row_first, pl.level_base) = (k, rows.len() as u32, levels);
            levels += b.level_first.len() as u32;
            starts.push((base + b.id_first, up_head.len() as u64));
            level_first.extend(b.level_first.iter().map(|&l| base + l));
            up_head.extend(b.up_head.iter().map(|&h| base + h));
            for m in 0..metrics {
                up[m].extend_from_slice(&b.up[m]);
                down[m].extend_from_slice(&b.down[m]);
            }
            for i in 0..b.rows.len() {
                up_first.push((starts.last().unwrap().1 + b.up_first[i + 1]) as u32);
                let v = b.rows[i];
                rows.push(v);
                let mut out: Vec<u64> = split.out[split.out_first[v as usize] as usize..split.out_first[v as usize + 1] as usize].to_vec();
                out.sort_by_key(|&a| (self.layout_of[f.arc(a as usize).1 as usize], f.arc_key(a as usize)));
                for a in out {
                    if !self.arc_local.is_empty() {
                        self.arc_local[a as usize] = arcs.len() as u32;
                    }
                    arcs.push(a);
                }
                arc_first.push(arcs.len() as u32);
            }
        }
        let links = up_head.len() as u64;
        ensure!(
            (rows.len() as u64) < LIMIT && links < LIMIT && (arcs.len() as u64) < LIMIT,
            "{} pack {k}: {} rows, {links} links, {} arcs: a count reaches 2^32 (FORMAT §4)",
            f.name(),
            rows.len(),
            arcs.len()
        );
        let layout_of = &self.layout_of;
        let heads: Vec<u32> = arcs.iter().map(|&a| layout_of[f.arc(a as usize).1 as usize]).collect();
        let link_of: Vec<u32> = arcs
            .iter()
            .map(|&a| {
                let (t, h) = f.arc(a as usize);
                if t == h {
                    return NONE;
                }
                let (lt, lh) = (layout_of[t as usize], layout_of[h as usize]);
                let (lo, hi) = (lt.max(lh), lt.min(lh));
                let at = starts.partition_point(|s| s.0 <= lo);
                if at > 0 {
                    let (x, b) = blocks[at - 1];
                    let base = self.first[x as usize];
                    if b.contains(lo - base) {
                        return (starts[at - 1].1 + b.link(lo - base, hi - base).expect("an arc has a link")) as u32;
                    }
                }
                outside(a)
            })
            .collect();
        u.put_in(&p, "level_first", Tensor::vec(&level_first));
        u.put_in(&p, "up_first", Tensor::vec(&up_first));
        u.put_in(&p, "up_head", Tensor::vec(&up_head));
        for (m, name) in f.metrics().iter().enumerate() {
            u.put_in(&p, &format!("{name}.up"), Tensor::vec(&up[m]));
            u.put_in(&p, &format!("{name}.down"), Tensor::vec(&down[m]));
        }
        let w = f.arc_word();
        u.put_in(&p, &format!("{w}_first"), Tensor::vec(&arc_first));
        u.put_in(&p, &format!("{w}_head"), Tensor::vec(&heads));
        u.put_in(&p, &format!("{w}_link"), Tensor::vec(&link_of));
        f.row_tensors(&rows, &layout_of[..], &mut u, &p);
        f.arc_tensors(&arcs, &mut u, &p);
        drop((up, down, up_head, level_first));
        let size = PackSize { pack: k, rows: rows.len() as u64, links, arcs: arcs.len() as u64 };
        if (size.rows, size.links) > (self.largest.rows, self.largest.links) {
            self.largest = size;
        }
        self.writer.set_slot(self.pack_slot + k as u64)?;
        let wr = self.writer.write_unit(u)?;
        self.bytes += wr.bytes;
        self.hashes.extend(wr.hashes);
        self.links += links;
        self.packs += 1;
        Ok(())
    }

    /// Records a component's blocks and their layout ids (`layout_of`).
    fn lay_out(&mut self, x: usize, b: &BlockOut) {
        let base = self.first[x];
        for (i, &v) in b.rows.iter().enumerate() {
            self.layout_of[v as usize] = base + b.id_first + i as u32;
        }
        let pl = &mut self.places[x][b.index as usize];
        (pl.kind, pl.id_first, pl.nodes, pl.levels) = (b.kind as u8, base + b.id_first, b.rows.len() as u32, b.levels());
    }
}

/// A component built in one piece: its single block.
fn whole(f: &dyn Family, split: &Split, x: usize, blocks: &[Block]) -> Result<BlockOut> {
    ensure!(blocks.len() == 1, "{} component {x}: {} blocks without a top block", f.name(), blocks.len());
    let (nodes, arcs) = split.component(x);
    let k = nodes.len();
    let h = Hierarchy::build(k, || split.edges(f, arcs), blocks);
    let (mut up, mut down) = (Vec::new(), Vec::new());
    for m in 0..f.metrics().len() {
        let (mut u, mut d) = h.unset();
        for &a in arcs {
            let (t, hd) = split.local(f, a);
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
    let Hierarchy { node_of, mut blocks, up_first, up_head, .. } = h;
    let bl = blocks.remove(0);
    Ok(BlockOut { index: 0, kind: bl.kind, id_first: 0, level_first: bl.level_first, rows: node_of.iter().map(|&i| nodes[i as usize]).collect(), up_first, up_head, up, down })
}

/// The levels of a block (FORMAT §5.1): each node's height in the block (its parent the
/// lowest-ranked upper neighbour inside the block), then the nodes by level, top first, and
/// rank descending. `up[r]` are rank r's upper neighbours, ascending by rank; ranks ≥ k are
/// outside. Returns each rank's position and the level bounds from `id_first`.
fn levels(k: usize, up: &[Vec<u32>], id_first: u32) -> (Vec<u32>, Vec<u32>) {
    let mut height = vec![0u32; k];
    for r in 0..k {
        if let Some(&p) = up[r].first()
            && (p as usize) < k
        {
            height[p as usize] = height[p as usize].max(height[r] + 1);
        }
    }
    let l = height.iter().copied().max().unwrap_or(0) + 1;
    let mut by: Vec<u32> = (0..k as u32).collect();
    by.sort_by_key(|&r| (l - 1 - height[r as usize], std::cmp::Reverse(r)));
    let mut level_first = vec![id_first];
    for lv in 0..l {
        let end = by.partition_point(|&r| l - 1 - height[r as usize] <= lv);
        level_first.push(id_first + end as u32);
    }
    let mut pos = vec![0u32; k];
    for (p, &r) in by.iter().enumerate() {
        pos[r as usize] = p as u32;
    }
    (pos, level_first)
}

/// A cell after phase 1: its layout and its upper neighbours (in rank space) on disk.
struct Eliminated {
    /// The cell's block index in the plan, and its layout position.
    b: usize,
    index: u32,
    id_first: u32,
    pos: Vec<u32>,
    level_first: Vec<u32>,
    up_first: Column<u64>,
    up: Column<u32>,
}

/// A component built as cells (the four phases above). `k0` is its top block's pack; its
/// cells' packs follow in layout order. Returns the component's hierarchy report.
fn cells(fw: &mut FamilyWriter, x: usize, plan: Vec<Block>, k0: u32, in_flight: usize, dir: &Path) -> Result<serde_json::Value> {
    let (f, split) = (fw.f, fw.split);
    let metrics = f.metrics().len();
    let (nodes, arcs) = split.component(x);
    let n = nodes.len();
    let t = plan.iter().position(|b| b.kind == Kind::Top).expect("a top block");
    let d = Some(dir);
    // Each node's block and rank in it.
    let (mut block_of, mut rank_in) = (ColumnMut::<u32>::filled(d, "block_of", n, NONE)?, ColumnMut::<u32>::zeroed(d, "rank_in", n)?);
    for (b, bl) in plan.iter().enumerate() {
        for (r, &v) in bl.nodes.iter().enumerate() {
            ensure!(block_of[v as usize] == NONE, "node {v} in two blocks");
            (block_of[v as usize], rank_in[v as usize]) = (b as u32, r as u32);
        }
    }
    ensure!(block_of.iter().all(|&b| b != NONE), "a node in no block");
    let (block_of, rank_in) = (block_of.freeze()?, rank_in.freeze()?);
    // Each block's arcs: those with an end in it, loops dropped, in arc order.
    let mut lists: Vec<ColumnWriter<u64>> = (0..plan.len()).map(|b| ColumnWriter::new(d, &format!("arcs{b}"))).collect::<Result<_>>()?;
    for &a in arcs {
        let (lt, lh) = split.local(f, a);
        if lt == lh {
            continue;
        }
        let (bt, bh) = (block_of[lt as usize] as usize, block_of[lh as usize] as usize);
        lists[bt].push(a)?;
        if bh != bt {
            ensure!(bt == t || bh == t, "an arc joins two cells");
            lists[bh].push(a)?;
        }
    }
    let lists: Vec<Column<u64>> = lists.into_iter().map(ColumnWriter::finish).collect::<Result<_>>()?;
    let top_n = plan[t].nodes.len() as u32;
    let cell_ids: Vec<usize> = (0..plan.len()).filter(|&b| b != t).collect();
    let mut id_first = Vec::with_capacity(cell_ids.len());
    let mut next = top_n;
    for &b in &cell_ids {
        id_first.push(next);
        next += plan[b].nodes.len() as u32;
    }
    crate::build::mem::phase("blocks' arcs");

    // Phase 1: the cells eliminated, a few at a time.
    let batches = in_batches(&cell_ids.iter().map(|&b| plan[b].nodes.len()).collect::<Vec<_>>(), in_flight);
    let mut done: Vec<Eliminated> = Vec::with_capacity(cell_ids.len());
    let mut top_fill: Vec<(u32, u32)> = Vec::new();
    for batch in &batches {
        let out: Vec<Result<(Eliminated, Vec<(u32, u32)>)>> = batch
            .clone()
            .into_par_iter()
            .map(|i| {
                let b = cell_ids[i];
                let bl = &plan[b];
                let k = bl.nodes.len() as u32;
                let local = |v: u32| if block_of[v as usize] == b as u32 { rank_in[v as usize] } else { k + rank_in[v as usize] };
                let el = eliminate(k as usize, lists[b].iter().map(|&a| {
                    let (lt, lh) = split.local(f, a);
                    (local(lt), local(lh))
                }));
                let (pos, level_first) = levels(k as usize, &el.up, id_first[i]);
                let fill: Vec<(u32, u32)> = el.fill.iter().map(|&(a, c)| (a - k, c - k)).collect();
                let (mut first, mut up) = (ColumnWriter::<u64>::new(d, &format!("up_first{b}"))?, ColumnWriter::<u32>::new(d, &format!("up{b}"))?);
                first.push(0)?;
                for l in &el.up {
                    up.push_slice(l)?;
                    first.push(up.len() as u64)?;
                }
                drop(el);
                let e = Eliminated { b, index: i as u32 + 1, id_first: id_first[i], pos, level_first, up_first: first.finish()?, up: up.finish()? };
                Ok((e, fill))
            })
            .collect();
        for o in out {
            let (e, fill) = o?;
            done.push(e);
            top_fill.extend(fill);
        }
    }
    crate::build::mem::phase("cells eliminated");

    // Phase 2: the top block, eliminated with the cells' cliques; its layout and its rows.
    let tk = top_n as usize;
    let top_in = |v: u32| block_of[v as usize] == t as u32;
    let el = eliminate(
        tk,
        lists[t]
            .iter()
            .filter_map(|&a| {
                let (lt, lh) = split.local(f, a);
                (top_in(lt) && top_in(lh)).then(|| (rank_in[lt as usize], rank_in[lh as usize]))
            })
            .chain(top_fill.iter().copied()),
    );
    ensure!(el.fill.is_empty(), "the top block leaves fill");
    drop(top_fill);
    let (top_pos, top_levels) = levels(tk, &el.up, 0);
    let mut top_rank = vec![0u32; tk];
    for r in 0..tk {
        top_rank[top_pos[r] as usize] = r as u32;
    }
    let (mut top_first, mut top_head) = (Vec::with_capacity(tk + 1), Vec::new());
    top_first.push(0u64);
    let mut row: Vec<u32> = Vec::new();
    for p in 0..tk {
        row.clear();
        row.extend(el.up[top_rank[p] as usize].iter().map(|&r| top_pos[r as usize]));
        row.sort_unstable();
        ensure!(row.last().is_none_or(|&h| h < p as u32), "an upper neighbour after its node");
        top_head.extend_from_slice(&row);
        top_first.push(top_head.len() as u64);
    }
    drop(el);
    // The layout ids of the cells and of the top.
    let top_block = BlockOut {
        index: 0,
        kind: Kind::Top,
        id_first: 0,
        level_first: top_levels,
        rows: top_rank.iter().map(|&r| nodes[plan[t].nodes[r as usize] as usize]).collect(),
        up_first: top_first,
        up_head: top_head,
        up: Vec::new(),
        down: Vec::new(),
    };
    fw.lay_out(x, &top_block);
    for e in &done {
        let bl = &plan[e.b];
        let mut rows = vec![0u32; bl.nodes.len()];
        for (r, &v) in bl.nodes.iter().enumerate() {
            rows[e.pos[r] as usize] = nodes[v as usize];
        }
        let shell = BlockOut { index: e.index, kind: Kind::Cell, id_first: e.id_first, level_first: e.level_first.clone(), rows, up_first: Vec::new(), up_head: Vec::new(), up: Vec::new(), down: Vec::new() };
        fw.lay_out(x, &shell);
    }
    // The top's link weights, relaxed by the cells' triangles as they come.
    let atomics = || (0..top_block.up_head.len()).map(|_| AtomicU16::new(INF)).collect::<Vec<_>>();
    let (top_up, top_down): (Vec<Vec<AtomicU16>>, Vec<Vec<AtomicU16>>) = ((0..metrics).map(|_| atomics()).collect(), (0..metrics).map(|_| atomics()).collect());
    let cell_of: Vec<usize> = {
        let mut c = vec![usize::MAX; plan.len()];
        for (i, &b) in cell_ids.iter().enumerate() {
            c[b] = i;
        }
        c
    };
    let layout = |v: u32| -> u32 {
        let (b, r) = (block_of[v as usize] as usize, rank_in[v as usize] as usize);
        if b == t { top_pos[r] } else { done[cell_of[b]].id_first + done[cell_of[b]].pos[r] }
    };
    for &a in lists[t].iter() {
        let (lt, lh) = split.local(f, a);
        if top_in(lt) && top_in(lh) {
            let (a_, c_) = (layout(lt), layout(lh));
            let l = top_block.link(a_.max(c_), a_.min(c_)).expect("an arc has a link") as usize;
            for m in 0..metrics {
                let w = f.arc_weight(m, a as usize);
                if a_ > c_ { top_up[m][l].fetch_min(w, Relaxed) } else { top_down[m][l].fetch_min(w, Relaxed) };
            }
        }
    }
    crate::build::mem::phase("top laid out");

    // Phase 3: each cell's rows, links and pack; the links from the top down into it.
    let mut boundary: Vec<(u64, u32)> = Vec::new();
    let mut top_height = vec![1u32; tk];
    let mut tree = 0u32;
    let (mut levels_max, mut links) = (0u32, top_block.links());
    for batch in &batches {
        let out: Vec<Result<(BlockOut, Vec<(u64, u32)>, Vec<(u32, u32)>, u32)>> = batch
            .clone()
            .into_par_iter()
            .map(|i| {
                let e = &done[i];
                let bl = &plan[e.b];
                let k = bl.nodes.len();
                let mut rank_at = vec![0u32; k];
                for r in 0..k {
                    rank_at[e.pos[r] as usize] = r as u32;
                }
                let (mut up_first, mut up_head) = (Vec::with_capacity(k + 1), Vec::new());
                up_first.push(0u64);
                let mut row: Vec<u32> = Vec::new();
                for p in 0..k {
                    let r = rank_at[p] as usize;
                    row.clear();
                    row.extend(e.up[e.up_first[r] as usize..e.up_first[r + 1] as usize].iter().map(|&u| {
                        if (u as usize) < k { e.id_first + e.pos[u as usize] } else { top_pos[(u - k as u32) as usize] }
                    }));
                    row.sort_unstable();
                    ensure!(row.last().is_none_or(|&h| h < e.id_first + p as u32), "an upper neighbour after its node");
                    up_head.extend_from_slice(&row);
                    up_first.push(up_head.len() as u64);
                }
                let rows: Vec<u32> = rank_at.iter().map(|&r| nodes[bl.nodes[r as usize] as usize]).collect();
                let mut cell = BlockOut { index: e.index, kind: Kind::Cell, id_first: e.id_first, level_first: e.level_first.clone(), rows, up_first, up_head, up: Vec::new(), down: Vec::new() };
                let m_links = cell.up_head.len();
                for m in 0..metrics {
                    let (mut u, mut dn) = (vec![INF; m_links], vec![INF; m_links]);
                    for &a in lists[e.b].iter() {
                        let (lt, lh) = split.local(f, a);
                        let (a_, c_) = (layout(lt), layout(lh));
                        let l = cell.link(a_.max(c_), a_.min(c_)).expect("an arc has a link") as usize;
                        let w = f.arc_weight(m, a as usize);
                        if a_ > c_ {
                            u[l] = u[l].min(w);
                        } else {
                            dn[l] = dn[l].min(w);
                        }
                    }
                    // FORMAT §6.3 for the cell's nodes, last first.
                    for p in (0..k).rev() {
                        let r = cell.up_first[p] as usize..cell.up_first[p + 1] as usize;
                        for i in 1..r.len() {
                            let (a, xh) = (r.start + i, cell.up_head[r.start + i]);
                            for j in 0..i {
                                let (b, yh) = (r.start + j, cell.up_head[r.start + j]);
                                let (wu, wd) = (sat(dn[a], u[b]), sat(dn[b], u[a]));
                                if xh >= e.id_first {
                                    let t = cell.link(xh, yh).expect("chordal") as usize;
                                    u[t] = u[t].min(wu);
                                    dn[t] = dn[t].min(wd);
                                } else {
                                    let t = top_block.link(xh, yh).expect("chordal") as usize;
                                    top_up[m][t].fetch_min(wu, Relaxed);
                                    top_down[m][t].fetch_min(wd, Relaxed);
                                }
                            }
                        }
                    }
                    cell.up.push(u);
                    cell.down.push(dn);
                }
                // The links of arcs from the top down into the cell, in the cell's pack.
                let mut down_in: Vec<(u64, u32)> = Vec::new();
                for &a in lists[e.b].iter() {
                    let (lt, lh) = split.local(f, a);
                    if top_in(lt) {
                        let (a_, c_) = (layout(lt), layout(lh));
                        down_in.push((a, cell.link(c_, a_).expect("an arc has a link") as u32));
                    }
                }
                // The elimination tree's heights (FORMAT §5.2: the parent is the largest upper
                // neighbour): inside the cell, and what it passes to the top.
                let mut h = vec![1u32; k];
                let mut to_top: Vec<(u32, u32)> = Vec::new();
                for p in (0..k).rev() {
                    if let Some(&par) = cell.up_head[cell.up_first[p] as usize..cell.up_first[p + 1] as usize].last() {
                        if par >= e.id_first {
                            let q = (par - e.id_first) as usize;
                            h[q] = h[q].max(h[p] + 1);
                        } else {
                            to_top.push((par, h[p] + 1));
                        }
                    }
                }
                Ok((cell, down_in, to_top, h.into_iter().max().unwrap_or(0)))
            })
            .collect();
        for o in out {
            let (cell, down_in, to_top, hmax) = o?;
            boundary.extend(down_in);
            for (p, hh) in to_top {
                top_height[p as usize] = top_height[p as usize].max(hh);
            }
            tree = tree.max(hmax);
            levels_max = levels_max.max(cell.levels());
            links += cell.links();
            fw.pack(k0 + cell.index, &[(x as u32, &cell)], &|_| unreachable!("a cell's arcs have their links in it"))?;
        }
    }
    drop(done);
    crate::build::mem::phase("cells customized and written");

    // Phase 4: the top block customized and written.
    let mut top = top_block;
    top.up = top_up.into_iter().map(|v| v.into_iter().map(AtomicU16::into_inner).collect()).collect();
    top.down = top_down.into_iter().map(|v| v.into_iter().map(AtomicU16::into_inner).collect()).collect();
    for m in 0..metrics {
        customize_rows(&top.up_first, &top.up_head, &mut top.up[m], &mut top.down[m]);
    }
    for p in (0..tk).rev() {
        if let Some(&par) = top.up_head[top.up_first[p] as usize..top.up_first[p + 1] as usize].last() {
            top_height[par as usize] = top_height[par as usize].max(top_height[p] + 1);
        }
    }
    tree = tree.max(top_height.into_iter().max().unwrap_or(0));
    boundary.sort_unstable();
    let outside = |a: u64| -> u32 {
        let i = boundary.binary_search_by_key(&a, |x| x.0).expect("an arc into a cell has its link there");
        boundary[i].1
    };
    let report = json!({
        "family": f.name(), "component": x, "nodes": n, "links": links, "cells": cell_ids.len(),
        "sweep_layers": top.levels() + levels_max, "tree_levels": tree,
    });
    fw.pack(k0, &[(x as u32, &top)], &outside)?;
    crate::build::mem::phase("top customized and written");
    Ok(report)
}

/// Consecutive batches of items (by size), each of at most `budget` in total but for an item
/// larger alone.
fn in_batches(sizes: &[usize], budget: usize) -> Vec<std::ops::Range<usize>> {
    let mut out = Vec::new();
    let (mut start, mut sum) = (0, 0);
    for (i, &s) in sizes.iter().enumerate() {
        if i > start && sum + s > budget {
            out.push(start..i);
            (start, sum) = (i, 0);
        }
        sum += s;
    }
    if start < sizes.len() {
        out.push(start..sizes.len());
    }
    out
}

/// Builds and writes a family's road units (FORMAT §2.4): its packs in pack slots from
/// `slots.1` up, its tables in `slots.0`. Components are built in index order, in batches of
/// at most `in_flight` nodes; every block of a component built as cells, or of one with at
/// least `pack_max_nodes` nodes, is a pack of its own, the other components share packs of at
/// most `pack_max_nodes` nodes. Scratch files go to `dir`.
#[allow(clippy::too_many_arguments)]
pub fn write(f: &dyn Family, o: Ordered, pack_max_nodes: u64, in_flight: usize, writer: &mut ModelWriter, slots: (u64, u64), arc_local: bool, dir: &Path) -> Result<Written> {
    let Ordered { split, mut plans } = o;
    let n = f.nodes();
    ensure!((n as u64) < NONE as u64, "{}: {n} layout nodes, beyond the U32 layout ids (FORMAT §4)", f.name());
    let k = plans.len();
    let size = |x: usize| split.first[x + 1] - split.first[x];
    let mut first = Vec::with_capacity(k);
    let mut next = 0u32;
    for x in 0..k {
        first.push(next);
        next += size(x) as u32;
    }
    let d = Some(dir);
    let mut fw = FamilyWriter {
        f,
        split: &split,
        first,
        layout_of: ColumnMut::filled(d, "layout_of", n, NONE)?,
        arc_local: ColumnMut::filled(d, "arc_local", if arc_local { f.arc_count() } else { 0 }, NONE)?,
        writer,
        pack_slot: slots.1,
        places: plans.iter().map(|p| vec![Place::default(); p.len()]).collect(),
        hashes: BTreeMap::new(),
        largest: PackSize::default(),
        links: 0,
        packs: 0,
        bytes: 0,
    };
    let mut hierarchies = Vec::new();
    // The packs: (first component, end) of components sharing one, or a component's blocks.
    let small = |x: usize, plans: &[Vec<Block>]| (size(x) as u64) < pack_max_nodes && plans[x].len() == 1;
    let mut x = 0;
    let mut k_next = 0u32;
    while x < k {
        if plans[x].len() > 1 {
            let plan = std::mem::take(&mut plans[x]);
            let blocks = plan.len() as u32;
            hierarchies.push(cells(&mut fw, x, plan, k_next, in_flight, dir)?);
            k_next += blocks;
            x += 1;
            continue;
        }
        // Whole components, packed, in a batch of at most `in_flight` nodes.
        let mut groups: Vec<std::ops::Range<usize>> = Vec::new();
        let mut nodes = 0usize;
        while x < k && plans[x].len() == 1 && (groups.is_empty() || nodes + size(x) <= in_flight) {
            let end = if small(x, &plans) {
                let (mut end, mut sum) = (x + 1, size(x) as u64);
                while end < k && small(end, &plans) && sum + size(end) as u64 <= pack_max_nodes {
                    sum += size(end) as u64;
                    end += 1;
                }
                end
            } else {
                x + 1
            };
            nodes += (x..end).map(size).sum::<usize>();
            groups.push(x..end);
            x = end;
        }
        let (lo, hi) = (groups[0].start, groups.last().unwrap().end);
        let built: Vec<Result<BlockOut>> = (lo..hi).into_par_iter().map(|c| whole(f, &split, c, &plans[c])).collect();
        let built: Vec<BlockOut> = built.into_iter().collect::<Result<_>>()?;
        for (c, b) in (lo..hi).zip(&built) {
            fw.lay_out(c, b);
            plans[c] = Vec::new();
            if c == 0 {
                hierarchies.push(json!({
                    "family": f.name(), "component": 0, "nodes": b.rows.len(), "links": b.links(), "cells": 0,
                    "sweep_layers": b.levels(), "tree_levels": tree_levels(b),
                }));
            }
        }
        for g in groups {
            let blocks: Vec<(u32, &BlockOut)> = g.clone().map(|c| (c as u32, &built[c - lo])).collect();
            fw.pack(k_next, &blocks, &|_| unreachable!("a component's arcs have their links in its pack"))?;
            k_next += 1;
        }
    }
    crate::build::mem::phase("packs");
    // The family tables (FORMAT §5.1, §5.4).
    let mut u = Unit::default();
    let mut component = Vec::with_capacity(n);
    for x in 0..k {
        component.extend(std::iter::repeat_n(x as u32, size(x)));
    }
    u.put_in(&format!("road/{}/nodes", f.name()), "component", Tensor::vec(&component));
    drop(component);
    let places: Vec<&Place> = fw.places.iter().flatten().collect();
    let p = format!("road/{}/blocks", f.name());
    u.put_in(&p, "kind", Tensor::vec(&places.iter().map(|x| x.kind).collect::<Vec<u8>>()));
    u.put_in(&p, "id_first", Tensor::vec(&places.iter().map(|x| x.id_first).collect::<Vec<u32>>()));
    u.put_in(&p, "nodes", Tensor::vec(&places.iter().map(|x| x.nodes).collect::<Vec<u32>>()));
    u.put_in(&p, "pack", Tensor::vec(&places.iter().map(|x| x.pack).collect::<Vec<u32>>()));
    u.put_in(&p, "row_first", Tensor::vec(&places.iter().map(|x| x.row_first).collect::<Vec<u32>>()));
    u.put_in(&p, "levels", Tensor::vec(&places.iter().map(|x| x.levels).collect::<Vec<u32>>()));
    u.put_in(&p, "level_base", Tensor::vec(&places.iter().map(|x| x.level_base).collect::<Vec<u32>>()));
    f.table_tensors(&mut u);
    fw.writer.set_slot(slots.0)?;
    let wr = fw.writer.write_unit(u)?;
    fw.bytes += wr.bytes;
    fw.hashes.extend(wr.hashes);
    let FamilyWriter { layout_of, arc_local: al, hashes, largest, links, packs, bytes, .. } = fw;
    Ok(Written {
        layout_of: layout_of.freeze()?,
        arc_local: al.freeze()?,
        hashes,
        largest,
        links,
        components: k as u64,
        packs,
        bytes,
        hierarchies,
    })
}

/// The elimination tree's height of a block built in one piece (`Hierarchy::tree_levels`).
fn tree_levels(b: &BlockOut) -> u32 {
    let mut h = vec![1u32; b.rows.len()];
    for v in (0..b.rows.len()).rev() {
        if let Some(&p) = b.up_head[b.up_first[v] as usize..b.up_first[v + 1] as usize].last() {
            h[p as usize] = h[p as usize].max(h[v] + 1);
        }
    }
    h.into_iter().max().unwrap_or(0)
}
