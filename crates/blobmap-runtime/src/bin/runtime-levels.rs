//! `runtime-levels <model> [--family car|active] [--component x]`: a component's sweep
//! levels, the format's own (FORMAT §5.2: per block; the top block's levels, then every
//! cell's level l run together): level count, launches one per block level, widths, and
//! runs of consecutive levels by width class.

use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use blobmap_runtime::{Backend, Options, Runtime};

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    ensure!(!a.is_empty(), "usage: runtime-levels <model> [--family car|active] [--component x]");
    let (mut fam, mut comp) = (0usize, None);
    let mut i = 1;
    while i < a.len() {
        let v = a.get(i + 1).context("option value")?;
        match a[i].as_str() {
            "--family" => fam = (v == "active") as usize,
            "--component" => comp = Some(v.parse::<u32>()?),
            x => bail!("unknown option {x}"),
        }
        i += 2;
    }
    let rt = Runtime::open(Path::new(&a[0]), Options { backend: Backend::Cpu, ..Options::default() })?;
    let f = &rt.fams[fam];
    let x = comp.unwrap_or(0);
    let c = rt.comp(f.fam, x)?;
    let g = &c.g;
    let (b0, b1) = (f.block_first[x as usize] as usize, f.block_first[x as usize + 1] as usize);
    let launches: u64 = (b0..b1).map(|b| f.b_levels[b] as u64).sum();
    let cells = (b0..b1).filter(|&b| f.b_kind[b] == 2).count();
    let w: Vec<u64> = (0..g.levels()).map(|l| g.width(l) as u64).collect();
    let narrow = w.iter().filter(|&&x| x < 256).count();
    let wide = w.iter().filter(|&&x| x >= 1 << 14).count();
    println!("{} component {x}: {} nodes, {} links; {} blocks ({cells} cells)", f.fam.name(), g.n, g.links(), b1 - b0);
    println!("{} levels ({narrow} under 256 nodes, {wide} of 16384 or more; widest {}); {launches} level launches one per block level", w.len(), w.iter().max().unwrap_or(&0));
    // Runs of consecutive levels by width class: (class, first level, levels, nodes).
    let class = |w: u64| if w <= 512 { "≤512" } else if w < 4096 { "<4096" } else if w < 16384 { "<16384" } else { "≥16384" };
    let mut runs: Vec<(&str, usize, usize, u64)> = Vec::new();
    for (l, &w) in w.iter().enumerate() {
        match runs.last_mut() {
            Some(r) if r.0 == class(w) => {
                r.2 += 1;
                r.3 += w;
            }
            _ => runs.push((class(w), l, 1, w)),
        }
    }
    println!("runs (width class, first level, levels, nodes):");
    for r in &runs {
        println!("  {:>7} from {:5}: {:5} levels, {:9} nodes", r.0, r.1, r.2, r.3);
    }
    Ok(())
}
