//! `runtime-bench <model> --backend cuda|wgpu [--family car|active] [--metric k] [--dir
//! from|to] [--sources n] [--rounds r] [--b k] [--bound t] [--no-graphs]`: GPU sweeps of a
//! family's largest component from pseudo-random served IDs (fixed seed), B columns each:
//! the seconds of seeding and launches up to a device sync, and of the read-back, as p50
//! and p99 per sweep. The trial's measure is the first.

use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use blobmap_model::{INF, NONE};
use blobmap_runtime::gpu::{Gpu, Kind};
use blobmap_runtime::graph::upward;
use blobmap_runtime::sweep::{Column, all_bounds};
use blobmap_runtime::{Backend, Dir, Options, Runtime};

fn pct(v: &mut [f64], p: f64) -> f64 {
    v.sort_by(f64::total_cmp);
    v[((v.len() as f64 - 1.0) * p).round() as usize]
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    ensure!(!a.is_empty(), "usage: runtime-bench <model> --backend cuda|wgpu [options]");
    let (mut kind, mut fam, mut metric, mut dir, mut sources, mut rounds, mut b, mut bound, mut graphs) = (Kind::Cuda, 0usize, 0usize, Dir::From, 1000usize, 2usize, 1usize, INF, true);
    let mut i = 1;
    while i < a.len() {
        let v = || a.get(i + 1).context("option value");
        match a[i].as_str() {
            "--backend" => {
                kind = match v()?.as_str() {
                    "cuda" => Kind::Cuda,
                    "wgpu" => Kind::Wgpu,
                    x => bail!("backend {x}"),
                }
            }
            "--family" => fam = (v()? == "active") as usize,
            "--metric" => metric = v()?.parse()?,
            "--dir" => dir = Dir::parse(v()?)?,
            "--sources" => sources = v()?.parse()?,
            "--rounds" => rounds = v()?.parse()?,
            "--b" => b = v()?.parse()?,
            "--bound" => bound = v()?.parse()?,
            "--no-graphs" => {
                graphs = false;
                i -= 1;
            }
            x => bail!("unknown option {x}"),
        }
        i += 2;
    }
    let rt = Runtime::open(Path::new(&a[0]), Options { backend: Backend::Cpu, ..Options::default() })?;
    let f = &rt.fams[fam];
    let x = (0..f.components() as u32).max_by_key(|&x| f.nodes(x)).context("no component")?;
    let t = Instant::now();
    let c = rt.comp(f.fam, x)?;
    let g = &c.g;
    eprintln!("{} component {x}: {} nodes, {} links, {} levels, loaded in {:.1} s", f.fam.name(), g.n, g.links(), g.levels(), t.elapsed().as_secs_f64());
    let mut gpu = Gpu::open(kind)?;
    gpu.use_graphs &= graphs;
    gpu.timing = true;
    let bounds = if bound < INF { Some(all_bounds(g)?) } else { None };
    let mut d = gpu.load(g, bounds.as_ref(), (b as u32).div_ceil(2))?;
    eprintln!("{}; on the device {:.2} GiB", gpu.describe(), d.bytes as f64 / (1u64 << 30) as f64);
    // Seeds: a node of each pseudo-random ID of the component in a mode of the family.
    let mode = rt.vocab.modes.iter().position(|m| m.fam == f.fam && m.metric == metric).context("no mode")?;
    let mv = &rt.vocab.modes[mode];
    let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut picks = Vec::new();
    while picks.len() < sources {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let i = (s % mv.component.len() as u64) as usize;
        if mv.component[i] == x {
            let v = mv.node[i].iter().copied().find(|&v| v != NONE).unwrap();
            picks.push(g.new[v as usize]);
        }
    }
    let cols: Vec<Column> = picks.iter().map(|&v| Column { entries: upward(g, metric, dir, &[(v, 0)]), bound }).collect();
    let (mut down, mut read) = (Vec::new(), Vec::new());
    for r in 0..rounds {
        for chunk in cols.chunks(b) {
            let refs: Vec<&Column> = chunk.iter().collect();
            gpu.sweep(&mut d, metric, dir, &refs)?;
            let [x, y] = gpu.times.get();
            if r > 0 || rounds == 1 {
                down.push(x * 1e3);
                read.push(y * 1e3);
            }
        }
    }
    println!(
        "{:?} {} B={b} bound {bound}: {} sweeps, sweep p50 {:.3} ms p99 {:.3} ms; read-back p50 {:.3} ms p99 {:.3} ms",
        kind,
        f.fam.name(),
        down.len(),
        pct(&mut down, 0.5),
        pct(&mut down, 0.99),
        pct(&mut read, 0.5),
        pct(&mut read, 0.99)
    );
    Ok(())
}
