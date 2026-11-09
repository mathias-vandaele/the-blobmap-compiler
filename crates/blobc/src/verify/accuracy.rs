//! Accuracy against cached OSRM times (PLAN, milestone 3), never a live service:
//! `blobc accuracy <model> <pairs.csv> <mode> <origin column> <dest column> <osrm column>`.
//!
//! Each pair is asked `travel_time` on the model; one column per origin, in parallel. The
//! answered pairs give the MAE, the p95 of |diff|, the bias (mean of ours − OSRM) and the
//! median relative error |diff| / OSRM, in all and by OSRM duration; the others are counted
//! by answer. With `--components`, no time is computed: the pairs whose ends are served in
//! different components are counted by their two components.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use blobmap_format::INF;
use blobmap_format::runtime::query::Source;
use blobmap_format::runtime::{Dir, Runtime};
use rayon::prelude::*;

const BUCKETS: [(&str, f64); 6] = [("0–15 min", 900.0), ("15–30 min", 1800.0), ("30–60 min", 3600.0), ("1–2 h", 7200.0), ("2–4 h", 14400.0), ("> 4 h", f64::INFINITY)];

fn stats(d: &[(f64, f64)]) -> String {
    if d.is_empty() {
        return "–".into();
    }
    let n = d.len() as f64;
    let mut abs: Vec<f64> = d.iter().map(|(ours, osrm)| (ours - osrm).abs()).collect();
    let mut rel: Vec<f64> = d.iter().map(|(ours, osrm)| (ours - osrm).abs() / osrm.max(1.0)).collect();
    abs.sort_by(f64::total_cmp);
    rel.sort_by(f64::total_cmp);
    let mae = abs.iter().sum::<f64>() / n;
    let bias = d.iter().map(|(ours, osrm)| ours - osrm).sum::<f64>() / n;
    let p95 = abs[((0.95 * n).ceil() as usize).clamp(1, abs.len()) - 1];
    format!("{:>6} pairs  MAE {mae:7.1} s  p95 {p95:7.1} s  bias {bias:+7.1} s  median rel {:.2} %", d.len(), 100.0 * rel[rel.len() / 2])
}

pub fn main(a: Vec<String>) -> Result<()> {
    let by_components = a.iter().any(|x| x == "--components");
    let a: Vec<&String> = a.iter().filter(|x| *x != "--components").collect();
    let [model, pairs, mode, oc, dc, tc] = a[..] else {
        bail!("usage: blobc accuracy <model> <pairs.csv> <mode> <origin column> <dest column> <osrm column> [--components]");
    };
    let rt = Runtime::open(Path::new(model))?;
    let m = rt.mode(mode)?;
    let text = std::fs::read_to_string(pairs).with_context(|| format!("reading {pairs}"))?;
    let mut lines = text.lines();
    let head: Vec<&str> = lines.next().context("empty pairs file")?.split(',').collect();
    let col = |name: &str| head.iter().position(|h| *h == name).with_context(|| format!("no column {name}"));
    let (oi, di, ti) = (col(oc)?, col(dc)?, col(tc)?);
    let rows: Vec<(String, String, f64)> = lines
        .filter(|l| !l.is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split(',').collect();
            Ok((f[oi].to_string(), f[di].to_string(), f[ti].parse::<f64>()?))
        })
        .collect::<Result<_>>()?;
    if by_components {
        let mv = &rt.vocab.modes[m];
        let t = &rt.tables[&mv.info.fam];
        let served = |id: &str| rt.resolve(id).and_then(|i| rt.served(i, m)).ok();
        let mut apart: BTreeMap<(u32, u32), u64> = BTreeMap::new();
        for (o, d, _) in &rows {
            if let (Some(x), Some(y)) = (served(o), served(d))
                && x != y
            {
                *apart.entry((x, y)).or_default() += 1;
            }
        }
        println!("{mode}: {} pairs served in different components", apart.values().sum::<u64>());
        // One OSM way of each component but the largest, which would take long to load.
        let way = |x: u32| -> Result<String> { Ok(if x == 0 { "–".into() } else { rt.component(mv.info.fam, x)?.way[0].to_string() }) };
        println!("  {:>7} {:>9} {:>11}  {:>7} {:>9} {:>11}  {:>6}", "origin", "nodes", "OSM way", "dest", "nodes", "OSM way", "pairs");
        for ((x, y), k) in apart {
            println!("  {x:>7} {:>9} {:>11}  {y:>7} {:>9} {:>11}  {k:>6}", t.nodes(x), way(x)?, t.nodes(y), way(y)?);
        }
        return Ok(());
    }
    let mut by_origin: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (k, r) in rows.iter().enumerate() {
        by_origin.entry(&r.0).or_default().push(k);
    }
    let groups: Vec<(&str, Vec<usize>)> = by_origin.into_iter().collect();
    // Per pair: Ok(seconds) or the answer's reason.
    let answers: Vec<Vec<(usize, Result<f64, String>)>> = groups
        .par_iter()
        .map(|(o, ks)| -> Result<_> {
            let src = match rt.resolve(o).and_then(|i| rt.served(i, m).map(|_| i)) {
                Ok(i) => i,
                Err(r) => return Ok(ks.iter().map(|&k| (k, Err(format!("origin {}", r.name())))).collect()),
            };
            let times = rt.column(&[Source { id: src, start: 0 }], m, Dir::From)?;
            let mv = &rt.vocab.modes[m];
            Ok(ks
                .iter()
                .map(|&k| {
                    let d = match rt.resolve(&rows[k].1).and_then(|i| rt.served(i, m).map(|_| i)) {
                        Ok(i) => i,
                        Err(r) => return (k, Err(format!("destination {}", r.name()))),
                    };
                    if mv.component[d as usize] != mv.component[src as usize] {
                        return (k, Err("DifferentComponent".into()));
                    }
                    let t = times[d as usize];
                    (k, if t == INF { Err("BEYOND".into()) } else { Ok(t as f64) })
                })
                .collect())
        })
        .collect::<Result<_>>()?;
    let mut ok: Vec<(f64, f64)> = Vec::new();
    let mut missed: BTreeMap<String, u64> = BTreeMap::new();
    for (k, r) in answers.into_iter().flatten() {
        match r {
            Ok(t) => ok.push((t, rows[k].2)),
            Err(e) => *missed.entry(e).or_default() += 1,
        }
    }
    println!("{mode}: {} pairs, {} answered, {} origins", rows.len(), ok.len(), groups.len());
    let mut lo = 0.0;
    for (name, hi) in BUCKETS {
        let d: Vec<(f64, f64)> = ok.iter().copied().filter(|p| p.1 >= lo && p.1 < hi).collect();
        println!("  {name:>10}  {}", stats(&d));
        lo = hi;
    }
    println!("  {:>10}  {}", "all", stats(&ok));
    for (why, n) in &missed {
        println!("  not answered: {n} {why}");
    }
    Ok(())
}
