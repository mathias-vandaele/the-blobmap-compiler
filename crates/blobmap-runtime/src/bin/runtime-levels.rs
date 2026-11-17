//! `runtime-levels <model> [--family car|active] [--component x]`: a component's sweep
//! layers from the format's own levels (FORMAT §5.2: per block; the top block's levels, then
//! every cell's level l run together) against the runtime's relayering (longest path from
//! the top). Prints level counts, launches, widths, and the nodes whose layer differs.

use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use blobmap_model::Model;
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
    let m: &Model = &rt.model;
    let base = f.first[x as usize];
    // Format layer per node (layout id - base): top / whole levels, then L_top + a cell's level.
    let mut layer = vec![u32::MAX; g.n];
    let (b0, b1) = (f.block_first[x as usize] as usize, f.block_first[x as usize + 1] as usize);
    let mut l_top = 0u32;
    let (mut launches, mut cells, mut max_cell) = (0u64, 0u32, 0u32);
    for b in b0..b1 {
        let p = format!("road/{}/p{}/level_first", f.fam.name(), f.b_pack[b]);
        let lf = m.get::<u32>(&p)?;
        let (lb, nl) = (f.b_level_base[b] as usize, f.b_levels[b] as usize);
        let add = if f.b_kind[b] == 2 { l_top } else { 0 };
        for l in 0..nl {
            for v in lf[lb + l]..lf[lb + l + 1] {
                layer[(v - base) as usize] = add + l as u32;
            }
        }
        launches += nl as u64;
        match f.b_kind[b] {
            2 => {
                cells += 1;
                max_cell = max_cell.max(nl as u32);
            }
            _ => l_top = nl as u32,
        }
    }
    ensure!(layer.iter().all(|&l| l != u32::MAX), "the blocks' levels do not cover the component");
    let merged = l_top + max_cell;
    // Relayered layer per node.
    let depth = |v: usize| -> u32 { (g.level_first.partition_point(|&s| s <= g.new[v]) - 1) as u32 };
    let mut differ = 0usize;
    let mut shallower = 0usize;
    for v in 0..g.n {
        let d = depth(v);
        if d != layer[v] {
            differ += 1;
            if layer[v] > d {
                shallower += 1;
            }
        }
    }
    let widths = |w: &[u64]| {
        let narrow = w.iter().filter(|&&x| x < 256).count();
        let wide = w.iter().filter(|&&x| x >= 1 << 14).count();
        format!("{} layers ({narrow} under 256 nodes, {wide} of 16384 or more; widest {})", w.len(), w.iter().max().unwrap_or(&0))
    };
    let mut fw = vec![0u64; merged as usize];
    for &l in &layer {
        fw[l as usize] += 1;
    }
    let rw: Vec<u64> = (0..g.levels()).map(|l| g.width(l) as u64).collect();
    println!("{} component {x}: {} nodes, {} links; {} blocks ({cells} cells)", f.fam.name(), g.n, g.links(), b1 - b0);
    println!("format: top {l_top} levels + deepest cell {max_cell} = {}; {launches} level launches one per block level", widths(&fw));
    println!("relayered (longest path): {}", widths(&rw));
    println!("nodes in a different layer: {differ} of {} ({shallower} deeper in the format)", g.n);
    // Runs of consecutive layers by width class, relayered: (class, first layer, layers, nodes).
    let class = |w: u64| if w <= 512 { "≤512" } else if w < 4096 { "<4096" } else if w < 16384 { "<16384" } else { "≥16384" };
    let mut runs: Vec<(&str, usize, usize, u64)> = Vec::new();
    for (l, &w) in rw.iter().enumerate() {
        match runs.last_mut() {
            Some(r) if r.0 == class(w) => {
                r.2 += 1;
                r.3 += w;
            }
            _ => runs.push((class(w), l, 1, w)),
        }
    }
    println!("relayered runs (width class, first layer, layers, nodes):");
    for r in &runs {
        println!("  {:>7} from {:5}: {:5} layers, {:9} nodes", r.0, r.1, r.2, r.3);
    }
    Ok(())
}
