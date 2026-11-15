//! Cells (SPEC §5): a component too large to build at once is ordered by the same nested
//! dissection as in one piece, on its fine graph, and the elimination tree of that order is
//! cut where subtrees fit the budget. The subtrees below the cut, packed together, are the
//! cells; the nodes above it are the top block. Different subtrees of an elimination tree
//! share no arc, and each block keeps the order's ranks, so the contraction order is a
//! reordering of the same elimination tree and the hierarchy is the one-piece hierarchy (its
//! links, its fill).
//!
//! When the fine graph is too large to order within the budget, the component is coarsened
//! instead: the coarse graph is ordered and its elimination tree cut the same way, the
//! coarse nodes above the cut, lifted to the fine graph, are the top block, and each block
//! is ordered on its own (the top block by the coarse order, which is the separator
//! hierarchy).

use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::Path;

use anyhow::{Context, Result};
use blobmap_model::NONE;
use rayon::prelude::*;

use crate::osm::column::{Column, ColumnWriter};
use crate::order::elim::etree;
use crate::order::{Graph, Orderer, ranks};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Whole = 0,
    Top = 1,
    Cell = 2,
}

/// A block of a component: its nodes in contraction order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Block {
    pub kind: Kind,
    pub nodes: Vec<u32>,
}

/// Undirected adjacency without loops or parallel arcs, with arc multiplicities. Arc
/// positions are u64: a whole component's arcs, both ways, may pass 2^32.
pub(crate) struct Adj {
    first: Column<u64>,
    pub to: Column<u32>,
    weight: Column<u32>,
}

/// Records per bucket of a spilled adjacency (`Adj::spilled`): 12 bytes each, sorted in memory.
const BUCKET_RECORDS: u64 = 1 << 25;
/// Nodes per counting span when buckets are drawn.
const SPAN: usize = 1 << 12;

impl Adj {
    /// Each edge both ways, loops dropped, parallel arcs merged with their weights summed, the
    /// arcs of a node by head. `edges` runs twice, to count and to fill, so the list is never
    /// held: the CSR is built in place and each row sorted and merged there.
    fn new<I: Iterator<Item = (u32, u32, u32)>>(n: usize, edges: impl Fn() -> I) -> Adj {
        let mut first = vec![0u64; n + 1];
        for (a, b, _) in edges().filter(|e| e.0 != e.1) {
            first[a as usize + 1] += 1;
            first[b as usize + 1] += 1;
        }
        for i in 0..n {
            first[i + 1] += first[i];
        }
        let m = first[n] as usize;
        let (mut to, mut weight) = (vec![0u32; m], vec![0u32; m]);
        let mut at = first[..n].to_vec();
        for (a, b, w) in edges().filter(|e| e.0 != e.1) {
            for (x, y) in [(a, b), (b, a)] {
                let k = at[x as usize] as usize;
                (to[k], weight[k]) = (y, w);
                at[x as usize] += 1;
            }
        }
        drop(at);
        let (mut row, mut out) = (Vec::new(), 0usize);
        for v in 0..n {
            let (start, end) = (first[v] as usize, first[v + 1] as usize);
            row.clear();
            row.extend((start..end).map(|k| (to[k], weight[k])));
            row.sort_unstable();
            first[v] = out as u64;
            for &(b, w) in &row {
                if out > first[v] as usize && to[out - 1] == b {
                    weight[out - 1] += w;
                } else {
                    (to[out], weight[out]) = (b, w);
                    out += 1;
                }
            }
        }
        first[n] = out as u64;
        to.truncate(out);
        weight.truncate(out);
        Adj { first: first.into(), to: to.into(), weight: weight.into() }
    }

    /// `Adj::new` in memory without a directory; with one, `Adj::spilled` into its files `name.*`.
    pub(crate) fn build<I: Iterator<Item = (u32, u32, u32)>>(n: usize, edges: impl Fn() -> I, dir: Option<&Path>, name: &str) -> Result<Adj> {
        match dir {
            None => Ok(Adj::new(n, edges)),
            Some(d) => Adj::spilled(n, edges, d, name, SPAN, BUCKET_RECORDS),
        }
    }

    /// The same adjacency as `Adj::new`, built on disk: each arc, both ways, is appended to the
    /// bucket file of its row (buckets are runs of rows of at most about `BUCKET_RECORDS` arcs);
    /// each bucket is then read, sorted and merged in memory, and its rows appended to the
    /// columns, which are mapped from `dir`. Memory holds one bucket. Buckets are drawn from
    /// counts per `span` rows.
    fn spilled<I: Iterator<Item = (u32, u32, u32)>>(n: usize, edges: impl Fn() -> I, dir: &Path, name: &str, span: usize, bucket_records: u64) -> Result<Adj> {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        let spans = n.div_ceil(span);
        let mut count = vec![0u64; spans];
        for (a, b, _) in edges().filter(|e| e.0 != e.1) {
            count[a as usize / span] += 1;
            count[b as usize / span] += 1;
        }
        // Each span's bucket, and each bucket's first row.
        let (mut bucket, mut start, mut held) = (vec![0u32; spans], vec![0usize], 0u64);
        for s in 0..spans {
            if held > 0 && held + count[s] > bucket_records {
                start.push(s * span);
                held = 0;
            }
            held += count[s];
            bucket[s] = (start.len() - 1) as u32;
        }
        start.push(n);
        drop(count);
        let paths: Vec<_> = (0..start.len() - 1).map(|b| dir.join(format!("{name}.b{b}"))).collect();
        let mut files = paths
            .iter()
            .map(|p| Ok(BufWriter::with_capacity(1 << 20, File::create(p).with_context(|| format!("creating {}", p.display()))?)))
            .collect::<Result<Vec<_>>>()?;
        for (a, b, w) in edges().filter(|e| e.0 != e.1) {
            for (x, y) in [(a, b), (b, a)] {
                let mut r = [0u8; 12];
                r[..4].copy_from_slice(&x.to_le_bytes());
                r[4..8].copy_from_slice(&y.to_le_bytes());
                r[8..].copy_from_slice(&w.to_le_bytes());
                files[bucket[x as usize / span] as usize].write_all(&r)?;
            }
        }
        for f in &mut files {
            f.flush()?;
        }
        drop((files, bucket));
        let d = Some(dir);
        let (mut first, mut to, mut weight) = (
            ColumnWriter::<u64>::new(d, &format!("{name}.first"))?,
            ColumnWriter::<u32>::new(d, &format!("{name}.to"))?,
            ColumnWriter::<u32>::new(d, &format!("{name}.weight"))?,
        );
        let mut out = 0u64;
        for (b, p) in paths.iter().enumerate() {
            let mut bytes = Vec::new();
            File::open(p).with_context(|| format!("opening {}", p.display()))?.read_to_end(&mut bytes)?;
            std::fs::remove_file(p).with_context(|| format!("removing {}", p.display()))?;
            let word = |r: &[u8], i: usize| u32::from_le_bytes(r[4 * i..4 * i + 4].try_into().unwrap());
            let mut recs: Vec<[u32; 3]> = bytes.as_chunks::<12>().0.iter().map(|r| [word(r, 0), word(r, 1), word(r, 2)]).collect();
            drop(bytes);
            recs.par_sort_unstable();
            let mut k = 0;
            for v in start[b]..start[b + 1] {
                first.push(out)?;
                while k < recs.len() && recs[k][0] as usize == v {
                    let (y, mut w) = (recs[k][1], recs[k][2]);
                    k += 1;
                    while k < recs.len() && recs[k][0] as usize == v && recs[k][1] == y {
                        w += recs[k][2];
                        k += 1;
                    }
                    to.push(y)?;
                    weight.push(w)?;
                    out += 1;
                }
            }
        }
        first.push(out)?;
        Ok(Adj { first: first.finish()?, to: to.finish()?, weight: weight.finish()? })
    }

    pub fn range(&self, v: usize) -> std::ops::Range<usize> {
        self.first[v] as usize..self.first[v + 1] as usize
    }
}

/// The induced subgraph on `nodes` (sorted), ordered, mapped back to node ids.
pub(crate) fn order_induced(g: &Graph, adj: &Adj, nodes: &[u32], orderer: &dyn Orderer) -> Vec<u32> {
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

/// The blocks of a component: one whole block if it has at most `max_cell` nodes, else a top
/// block and cells of at most `max_cell` nodes each: from the fine graph's order when the
/// component has at most `nd_max` nodes, else from a coarsened graph.
pub fn plan(g: &Graph, orderer: &dyn Orderer, max_cell: usize, nd_max: usize) -> Vec<Block> {
    if g.n <= max_cell.max(1) {
        return vec![Block { kind: Kind::Whole, nodes: orderer.order(g) }];
    }
    if g.n <= nd_max {
        return fine(g, orderer, max_cell);
    }
    let edges = || Box::new(g.tail.iter().zip(g.head).map(|(&a, &b)| (a, b))) as Edges<'_>;
    coarse(g, &edges, &|| orderer.order(g), orderer, max_cell, None).expect("an adjacency in memory")
}

/// Whether `plan` coarsens a component of `n` nodes.
pub fn coarsened(n: usize, max_cell: usize, nd_max: usize) -> bool {
    n > max_cell.max(1) && n > nd_max
}

/// A component's arcs, as node pairs in any direction.
pub type Edges<'a> = Box<dyn Iterator<Item = (u32, u32)> + 'a>;

/// The cut of an elimination tree over ranks, with `sub[r]` the weight of r's subtree: the
/// ranks whose subtree weighs more than `max_cell` are the top (`NONE`); the subtrees below
/// them are packed into cells in root order. Returns each rank's cell and the cell count.
fn cut(parent: &[Option<u32>], sub: &[u64], max_cell: usize) -> (Vec<u32>, usize) {
    let n = parent.len();
    let top: Vec<bool> = sub.iter().map(|&s| s > max_cell as u64).collect();
    // Units: subtrees hanging below the top, by their root.
    let mut unit = vec![NONE; n];
    for r in (0..n).rev() {
        if !top[r] {
            unit[r] = match parent[r] {
                Some(p) if !top[p as usize] => unit[p as usize],
                _ => r as u32,
            };
        }
    }
    let mut cell_of_unit = vec![NONE; n];
    let mut size: Vec<u64> = Vec::new();
    for r in 0..n {
        if !top[r] && unit[r] == r as u32 {
            match size.last_mut() {
                Some(s) if *s + sub[r] <= max_cell as u64 => *s += sub[r],
                _ => size.push(sub[r]),
            }
            cell_of_unit[r] = (size.len() - 1) as u32;
        }
    }
    ((0..n).map(|r| if top[r] { NONE } else { cell_of_unit[unit[r] as usize] }).collect(), size.len())
}

/// Cells cut from the fine graph's own order.
fn fine(g: &Graph, orderer: &dyn Orderer, max_cell: usize) -> Vec<Block> {
    let order = orderer.order(g);
    let rank = ranks(&order);
    let parent = etree(g.n, g.tail.iter().zip(g.head).map(|(&a, &b)| (rank[a as usize], rank[b as usize])));
    let mut sub = vec![1u64; g.n];
    for r in 0..g.n {
        if let Some(p) = parent[r] {
            sub[p as usize] += sub[r];
        }
    }
    let (cell, k) = cut(&parent, &sub, max_cell);
    let mut top = Vec::new();
    let mut cells: Vec<Vec<u32>> = vec![Vec::new(); k];
    for (r, &v) in order.iter().enumerate() {
        match cell[r] {
            NONE => top.push(v),
            c => cells[c as usize].push(v),
        }
    }
    if top.is_empty() || k < 2 {
        return vec![Block { kind: Kind::Whole, nodes: order }];
    }
    let mut out = vec![Block { kind: Kind::Top, nodes: top }];
    out.extend(cells.into_iter().map(|nodes| Block { kind: Kind::Cell, nodes }));
    out
}

/// Cells from a coarsened graph, for a component too large to order at once: the fine nodes
/// clustered, then dissected with the clusters' orders as a guide (`dissect`). `g` gives the
/// node count and coordinates (its arcs are not read); `edges` gives the arcs each time it is
/// called, and `whole` the order in one piece should the dissection not split the component.
/// With `spill`, the fine and coarse adjacencies are built in files there (`Adj::spilled`);
/// the blocks are the same either way.
pub fn coarse<'e>(g: &Graph, edges: &dyn Fn() -> Edges<'e>, whole: &dyn Fn() -> Vec<u32>, orderer: &dyn Orderer, max_cell: usize, spill: Option<&Path>) -> Result<Vec<Block>> {
    let fine = Adj::build(g.n, || edges().map(|(a, b)| (a, b, 1)), spill, "fine")?;
    crate::build::mem::phase("fine adjacency");
    // Coarsen to clusters of at most a quarter cell, until about 16 fine nodes per cluster.
    let target = (g.n / 16).max(1024);
    let cap = (max_cell as u64 / 4).max(1);
    let mut map: Vec<u32> = (0..g.n as u32).collect();
    let mut w: Vec<u64> = vec![1; g.n];
    // The graph of the current clusters; the fine graph itself until the first round.
    let mut adj: Option<Adj> = None;
    // The clusters' centres; the fine coordinates themselves until the first round.
    let (mut lat, mut lon): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
    while w.len() > target {
        let cur = adj.as_ref().unwrap_or(&fine);
        let (cluster, k) = matching(cur, &w, cap);
        if k * 10 > w.len() * 9 {
            break;
        }
        let mut nw = vec![0u64; k];
        let (mut nlat, mut nlon) = (vec![0.0f64; k], vec![0.0f64; k]);
        for v in 0..w.len() {
            let c = cluster[v] as usize;
            nw[c] += w[v];
            let (la, lo) = if adj.is_none() { (g.lat[v] as f64, g.lon[v] as f64) } else { (lat[v], lon[v]) };
            nlat[c] += la * w[v] as f64;
            nlon[c] += lo * w[v] as f64;
        }
        for c in 0..k {
            nlat[c] /= nw[c] as f64;
            nlon[c] /= nw[c] as f64;
        }
        let cluster = &cluster;
        let next = Adj::build(
            k,
            || {
                (0..w.len())
                    .flat_map(move |v| cur.range(v).map(move |e| (cluster[v], cluster[cur.to[e] as usize], cur.weight[e])))
                    .filter(|e| e.0 < e.1)
            },
            spill,
            "coarse",
        )?;
        adj = Some(next);
        for m in &mut map {
            *m = cluster[*m as usize];
        }
        (w, lat, lon) = (nw, nlat, nlon);
    }
    drop((adj, w, lat, lon));
    crate::build::mem::phase("coarsened");
    Ok(crate::order::dissect::refined(g, &fine, &map, whole, orderer, max_cell))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::order::InertialFlow;

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
        // From the fine order, then from a coarsened graph.
        for nd_max in [usize::MAX, 0] {
            let blocks = plan(&g, &InertialFlow, 300, nd_max);
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
            assert_eq!(blocks, plan(&g, &InertialFlow, 300, nd_max));
        }
        assert_eq!(plan(&g, &InertialFlow, 2000, 0)[0].kind, Kind::Whole);
    }

    fn same(a: &Adj, b: &Adj) {
        assert_eq!((&a.first[..], &a.to[..], &a.weight[..]), (&b.first[..], &b.to[..], &b.weight[..]));
    }

    #[test]
    fn spilled_adjacency_is_the_same() {
        // Loops, parallel arcs both ways, weights, an isolated node at the end, many buckets.
        let mut edges: Vec<(u32, u32, u32)> = Vec::new();
        let mut x = 7u64;
        for _ in 0..5000 {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let (a, b, w) = ((x >> 33) as u32 % 900, (x >> 13) as u32 % 900, (x >> 50) as u32 % 5 + 1);
            edges.push((a, b, w));
        }
        let dir = std::env::temp_dir().join(format!("blobc-adj-{}", std::process::id()));
        let mem = Adj::new(901, || edges.iter().copied());
        for (span, bucket) in [(1, 1), (3, 100), (64, 2000), (4096, 1 << 25)] {
            same(&mem, &Adj::spilled(901, || edges.iter().copied(), &dir, "t", span, bucket).unwrap());
        }
        assert!(std::fs::read_dir(&dir).unwrap().next().is_none(), "bucket files left");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn spilled_coarsening_gives_the_same_blocks() {
        let (tail, head, lat, lon) = grid(40);
        let g = Graph { n: 1600, tail: &tail, head: &head, lat: &lat, lon: &lon };
        let dir = std::env::temp_dir().join(format!("blobc-coarse-{}", std::process::id()));
        let edges = || Box::new(tail.iter().zip(&head).map(|(&a, &b)| (a, b))) as Edges<'_>;
        let spilled = coarse(&g, &edges, &|| InertialFlow.order(&g), &InertialFlow, 300, Some(&dir)).unwrap();
        assert_eq!(spilled, plan(&g, &InertialFlow, 300, 0));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn fine_cells_keep_the_order() {
        // Each block lists its nodes in the one-piece order.
        let (tail, head, lat, lon) = grid(40);
        let g = Graph { n: 1600, tail: &tail, head: &head, lat: &lat, lon: &lon };
        let rank = ranks(&InertialFlow.order(&g));
        for b in plan(&g, &InertialFlow, 300, usize::MAX) {
            assert!(b.nodes.windows(2).all(|w| rank[w[0] as usize] < rank[w[1] as usize]));
        }
    }
}
