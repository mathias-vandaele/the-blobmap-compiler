//! `runtime-mm <model> [--n N] [--mode m] [--dir from|to] [--seed s]`: the shape of an N×N
//! matrix's restricted sweep on a family's largest component: N distinct served IDs of the
//! mode in it; the nodes their times read (heads); the union of those nodes' elimination-tree
//! chains (every node a restricted sweep labels), its links and levels; the sources' upward
//! search spaces.

use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use blobmap_model::NONE;
use blobmap_runtime::graph::upward;
use blobmap_runtime::heads::{self, Part};
use blobmap_runtime::{Backend, Dir, Options, Runtime};

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    ensure!(!a.is_empty(), "usage: runtime-mm <model> [options]");
    let (mut n, mut mode, mut dir, mut seed) = (1000usize, "car".to_string(), Dir::From, 1u64);
    let mut i = 1;
    while i < a.len() {
        let v = a.get(i + 1).context("option value")?;
        match a[i].as_str() {
            "--n" => n = v.parse()?,
            "--mode" => mode = v.clone(),
            "--dir" => dir = Dir::parse(v)?,
            "--seed" => seed = v.parse()?,
            x => bail!("unknown option {x}"),
        }
        i += 2;
    }
    let rt = Runtime::open(Path::new(&a[0]), Options { backend: Backend::Cpu, ..Options::default() })?;
    let m = rt.mode(&mode)?;
    let mv = &rt.vocab.modes[m];
    let f = rt.fams.iter().find(|f| f.fam == mv.fam).context("family")?;
    let x = (0..f.components() as u32).max_by_key(|&x| f.nodes(x)).context("no component")?;
    let c = rt.comp(f.fam, x)?;
    let g = &c.g;
    let mut s = 0x2545_F491_4F6C_DD1Du64 ^ seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let mut ids = std::collections::BTreeSet::new();
    while ids.len() < n {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let i = (s % mv.len() as u64) as usize;
        if mv.component(i) == x {
            ids.insert(i as u32);
        }
    }
    let ids: Vec<u32> = ids.into_iter().collect();
    let p = Part { g, mv, x, ids: None, srcs: &[], dir, bound: u16::MAX };
    let t = Instant::now();
    let need = heads::needed(&p, &ids);
    let mut mark = vec![false; g.n];
    let mut chain_sum = 0usize;
    for &v in &need {
        let mut u = v;
        loop {
            chain_sum += 1;
            if mark[u as usize] {
                break;
            }
            mark[u as usize] = true;
            u = g.parent[u as usize];
            if u == NONE {
                break;
            }
        }
    }
    let union: Vec<u32> = (0..g.n as u32).filter(|&v| mark[v as usize]).collect();
    let links: usize = union.iter().map(|&v| g.row(v).len()).sum();
    let level = |v: u32| g.level_first.partition_point(|&f| f <= v) - 1;
    let mut lv: Vec<usize> = union.iter().map(|&v| level(v)).collect();
    lv.dedup();
    let built = t.elapsed().as_secs_f64();
    // Check: every upper neighbour of a union node is in the union.
    ensure!(union.iter().all(|&v| g.row(v).all(|k| mark[g.up_head[k] as usize])), "a union node's upper neighbour outside the union");
    let srcs: Vec<u32> = ids.iter().map(|&i| mv.node(i as usize).iter().copied().find(|&v| v != NONE).unwrap()).map(|v| g.new[v as usize]).collect();
    let t = Instant::now();
    let ups: Vec<usize> = srcs.iter().map(|&v| upward(g, mv.metric, dir, &[(v, 0)]).len()).collect();
    let up_s = t.elapsed().as_secs_f64();
    let in_union: usize = srcs.iter().map(|&v| upward(g, mv.metric, dir, &[(v, 0)]).iter().filter(|e| mark[e.0 as usize]).count()).sum();
    println!(
        "{} component {x}: {} nodes, {} links, {} levels. {n} IDs: {} head nodes; union of their chains {} nodes ({:.2}%), {links} links ({:.2}%), {} levels (widest {}), built in {:.1} ms on the host; chains {:.0} nodes on average",
        f.fam.name(),
        g.n,
        g.links(),
        g.levels(),
        need.len(),
        union.len(),
        100.0 * union.len() as f64 / g.n as f64,
        100.0 * links as f64 / g.links() as f64,
        lv.len(),
        {
            let mut w = std::collections::BTreeMap::new();
            for &v in &union {
                *w.entry(level(v)).or_insert(0usize) += 1;
            }
            w.values().copied().max().unwrap_or(0)
        },
        built * 1e3,
        chain_sum as f64 / need.len() as f64,
    );
    println!(
        "sources' upward spaces: {:.0} nodes on average (max {}), {:.0} in the union; {:.1} ms for all {n} on one host thread",
        ups.iter().sum::<usize>() as f64 / n as f64,
        ups.iter().max().unwrap(),
        in_union as f64 / n as f64,
        up_s * 1e3
    );
    println!("restricted sweep at B = {n}: {:.2} G label-word reads ({:.1} GB at 4 B per word)", links as f64 * n.div_ceil(2) as f64 / 1e9, links as f64 * n.div_ceil(2) as f64 * 4.0 / 1e9);
    Ok(())
}
