//! `runtime-prune <model> [--family car|active] [--metric k] [--dir from|to] [--sources n]
//! [--bounds t,t,...]`: the two pruning variants of PLAN.md on the CPU, one thread, on a
//! family's largest component from pseudo-random served IDs (fixed seed):
//!
//! - (a) the runtime's: a per-node `DEAD` flag passed down level by level (`cpu_down`);
//! - (b) a depth-first renumbering of the elimination tree, so that every subtree is a range
//!   and an unreached node whose entry bound is above T skips its subtree in one jump.
//!
//! Per bound: the mean milliseconds a column of each, against the unpruned sweep, the nodes
//! (b) reads, and a check that both give the same labels up to the bound.

use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use blobmap_model::{INF, NONE};
use blobmap_runtime::graph::upward;
use blobmap_runtime::sweep::{Column, DEAD, all_bounds, cpu_down, dir_index};
use blobmap_runtime::{Backend, Dir, Options, Runtime};

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    ensure!(!a.is_empty(), "usage: runtime-prune <model> [options]");
    let (mut fam, mut metric, mut dir, mut sources) = (0usize, 0usize, Dir::From, 20usize);
    let mut bounds_t: Vec<u16> = vec![900, 1800, 3600];
    let mut i = 1;
    while i < a.len() {
        let v = || a.get(i + 1).context("option value");
        match a[i].as_str() {
            "--family" => fam = (v()? == "active") as usize,
            "--metric" => metric = v()?.parse()?,
            "--dir" => dir = Dir::parse(v()?)?,
            "--sources" => sources = v()?.parse()?,
            "--bounds" => bounds_t = v()?.split(',').map(|x| x.parse()).collect::<Result<_, _>>()?,
            x => bail!("unknown option {x}"),
        }
        i += 2;
    }
    let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build()?;
    let rt = Runtime::open(Path::new(&a[0]), Options { backend: Backend::Cpu, ..Options::default() })?;
    let f = &rt.fams[fam];
    let x = (0..f.components() as u32).max_by_key(|&x| f.nodes(x)).context("no component")?;
    let c = rt.comp(f.fam, x)?;
    let g = &c.g;
    let n = g.n;
    let t0 = Instant::now();
    let bounds = all_bounds(g)?;
    let eb = &bounds[2 * metric + dir_index(dir)];
    let w = g.going_down(metric, dir);
    // (b): depth-first preorder of the elimination tree, children by ascending new id.
    let mut kid_first = vec![0u32; n + 1];
    for v in 0..n {
        if g.parent[v] != NONE {
            kid_first[g.parent[v] as usize + 1] += 1;
        }
    }
    for v in 0..n {
        kid_first[v + 1] += kid_first[v];
    }
    let mut kids = vec![0u32; n];
    let mut at = kid_first.clone();
    for v in 0..n {
        let p = g.parent[v];
        if p != NONE {
            kids[at[p as usize] as usize] = v as u32;
            at[p as usize] += 1;
        }
    }
    drop(at);
    let mut order = Vec::with_capacity(n);
    let mut stack: Vec<u32> = (0..n as u32).filter(|&v| g.parent[v as usize] == NONE).rev().collect();
    while let Some(v) = stack.pop() {
        order.push(v);
        stack.extend(kids[kid_first[v as usize] as usize..kid_first[v as usize + 1] as usize].iter().rev());
    }
    ensure!(order.len() == n, "the elimination tree does not cover the component");
    drop((kids, kid_first));
    let mut pos = vec![0u32; n];
    for (i, &v) in order.iter().enumerate() {
        pos[v as usize] = i as u32;
    }
    let mut size = vec![1u32; n];
    for i in (0..n).rev() {
        let p = g.parent[order[i] as usize];
        if p != NONE {
            size[pos[p as usize] as usize] += size[i];
        }
    }
    let mut b_first = Vec::with_capacity(n + 1);
    b_first.push(0u64);
    let mut b_link: Vec<(u32, u16, u16)> = Vec::with_capacity(g.links());
    for &v in &order {
        for k in g.row(v) {
            let h = pos[g.up_head[k] as usize];
            ensure!(h < pos[v as usize], "an upper neighbour after its node in depth-first order");
            b_link.push((h, w[k], eb[k]));
        }
        b_first.push(b_link.len() as u64);
    }
    eprintln!("{} component {x}: {n} nodes, {} links; bounds and depth-first copy in {:.1} s", f.fam.name(), g.links(), t0.elapsed().as_secs_f64());
    let sweep_b = |c: &Column, read: &mut u64| -> Vec<u16> {
        let t = c.bound;
        let mut lab = vec![DEAD; n];
        let mut seeded = vec![false; n];
        for &(v, x) in &c.entries {
            let p = pos[v as usize] as usize;
            lab[p] = x;
            seeded[p] = true;
        }
        let mut i = 0;
        while i < n {
            *read += 1;
            let s = seeded[i];
            let mut acc = if s { lab[i] } else { INF };
            let mut lb = INF;
            for &(h, wk, ek) in &b_link[b_first[i] as usize..b_first[i + 1] as usize] {
                let y = lab[h as usize];
                acc = acc.min(y.saturating_add(wk));
                lb = lb.min(y.saturating_add(ek));
            }
            if !s && lb > t {
                // Already DEAD, the whole subtree.
                i += size[i] as usize;
                continue;
            }
            lab[i] = if acc > t { INF } else { acc };
            i += 1;
        }
        lab
    };
    // Seeds: a node of each pseudo-random ID of the component in a mode of the family.
    let mode = rt.vocab.modes.iter().position(|m| m.fam == f.fam && m.metric == metric).context("no mode")?;
    let mv = &rt.vocab.modes[mode];
    let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut picks = Vec::new();
    while picks.len() < sources {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let i = (s % mv.len() as u64) as usize;
        if mv.component(i) == x {
            let v = mv.node(i).iter().copied().find(|&v| v != NONE).unwrap();
            picks.push(g.new[v as usize]);
        }
    }
    let entries: Vec<Vec<(u32, u16)>> = picks.iter().map(|&v| upward(g, metric, dir, &[(v, 0)])).collect();
    pool.install(|| -> Result<()> {
        let t = Instant::now();
        for e in &entries {
            std::hint::black_box(cpu_down(g, metric, dir, &Column { entries: e.clone(), bound: INF }, None));
        }
        let full = 1e3 * t.elapsed().as_secs_f64() / entries.len() as f64;
        println!("{} {:?} metric {metric}: unpruned {full:.1} ms a column ({} columns, one thread)", f.fam.name(), dir, entries.len());
        for &tb in &bounds_t {
            let cols: Vec<Column> = entries.iter().map(|e| Column { entries: e.clone(), bound: tb }).collect();
            let t = Instant::now();
            let la: Vec<Vec<u16>> = cols.iter().map(|c| cpu_down(g, metric, dir, c, Some(eb))).collect();
            let ta = 1e3 * t.elapsed().as_secs_f64() / cols.len() as f64;
            let mut read = 0u64;
            let t = Instant::now();
            let lb: Vec<Vec<u16>> = cols.iter().map(|c| sweep_b(c, &mut read)).collect();
            let tbm = 1e3 * t.elapsed().as_secs_f64() / cols.len() as f64;
            for (k, (xa, xb)) in la.iter().zip(&lb).enumerate() {
                for v in 0..n {
                    let (ya, yb) = (xa[v], xb[pos[v] as usize]);
                    ensure!((ya <= tb) == (yb <= tb) && (ya > tb || ya == yb), "bound {tb}, column {k}: node {v} {ya} (a) against {yb} (b)");
                }
            }
            let dead_a: usize = la.iter().map(|l| l.iter().filter(|&&y| y == DEAD).count()).sum();
            println!(
                "  T {tb} s: (a) {ta:.1} ms, (b) {tbm:.1} ms a column; (a) marks {:.1}% DEAD, (b) reads {:.1}% of nodes; labels agree up to T",
                100.0 * dead_a as f64 / (n * cols.len()) as f64,
                100.0 * read as f64 / (n * cols.len()) as f64
            );
        }
        Ok(())
    })
}
