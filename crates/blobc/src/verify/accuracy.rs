//! Accuracy against cached OSRM times (PLAN, milestone 3), never a live service:
//! `blobc accuracy <model> <pairs.csv> <mode> <origin column> <dest column> <osrm column>`.
//!
//! Each pair is asked `travel_time` on the model; one column per origin, in parallel. The
//! answered pairs give the MAE, the p95 of |diff|, the bias (mean of ours − OSRM) and the
//! median relative error |diff| / OSRM, in all and by OSRM duration; the others are counted
//! by answer. With `--components`, no time is computed: the pairs whose ends are served in
//! different components are counted by their two components, each with the real island it
//! stands on (from the data report next to the model) and, with `--pbf`, whether one of its
//! ways shares a node with a `route=ferry` way. A pair is explained when one of its ends is
//! on a real island or at a ferry (SPEC §13, milestone 3); the others are counted apart.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Mutex;

use anyhow::{Context, Result, bail};
use blobmap_model::INF;
use blobmap_model::runtime::query::Source;
use blobmap_model::runtime::{Dir, Runtime};
use osmpbf::{BlobDecode, BlobReader};
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
    let pbf = a.iter().position(|x| x == "--pbf").map(|i| a.get(i + 1).cloned().context("--pbf takes a file")).transpose()?;
    let a: Vec<&String> = a.iter().filter(|x| *x != "--components" && Some(*x) != pbf.as_ref() && *x != "--pbf").collect();
    let [model, pairs, mode, oc, dc, tc] = a[..] else {
        bail!("usage: blobc accuracy <model> <pairs.csv> <mode> <origin column> <dest column> <osrm column> [--components [--pbf <file>]]");
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
        // The ways of each component but the largest, which would take long to load.
        let mut ways: BTreeMap<u32, Vec<i64>> = BTreeMap::new();
        for &x in apart.keys().flat_map(|(x, y)| [x, y]).filter(|&&x| x != 0) {
            let mut w = rt.component(mv.info.fam, x)?.way.to_vec();
            w.sort_unstable();
            w.dedup();
            ways.insert(x, w);
        }
        let report = format!("{}.data-report.json", model.trim_end_matches('/'));
        let report = blobmap_model::json::read_file(Path::new(&report)).with_context(|| format!("reading {report}"))?;
        let island: BTreeMap<u64, String> = report["served_components"][mode.as_str()]
            .as_array()
            .context("the data report has no served components for the mode")?
            .iter()
            .filter_map(|r| Some((r["component"].as_u64()?, r["island"].as_str()?.to_string())))
            .collect();
        let ferry = match &pbf {
            Some(f) => at_ferry(Path::new(f), &ways)?,
            None => Default::default(),
        };
        let tag = |x: u32| match island.get(&(x as u64)) {
            Some(l) => l.clone(),
            None if ferry.contains(&x) => "ferry".into(),
            None => "–".into(),
        };
        println!("  {:>7} {:>9} {:>11}  {:>7} {:>9} {:>11}  {:>6}  why", "origin", "nodes", "OSM way", "dest", "nodes", "OSM way", "pairs");
        let way = |x: u32| ways.get(&x).map_or("–".into(), |w| w[0].to_string());
        let mut unexplained = 0;
        for ((x, y), k) in apart {
            let why = [x, y].into_iter().filter(|&x| x != 0).map(tag).filter(|t| t != "–").collect::<Vec<_>>().join("; ");
            if why.is_empty() {
                unexplained += k;
            }
            println!("  {x:>7} {:>9} {:>11}  {y:>7} {:>9} {:>11}  {k:>6}  {}", t.nodes(x), way(x), t.nodes(y), way(y), if why.is_empty() { "UNEXPLAINED" } else { &why });
        }
        println!("  {unexplained} pairs neither on a real island nor at a ferry");
        return Ok(());
    }
    let mut by_origin: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (k, r) in rows.iter().enumerate() {
        by_origin.entry(&r.0).or_default().push(k);
    }
    let groups: Vec<(&str, Vec<usize>)> = by_origin.into_iter().collect();
    // Per pair: Ok(seconds) or the answer's reason.
    let answers: Vec<Vec<(usize, Result<f64, String>)>> = crate::verify::columns_at_once(&rt, m, &groups, |(o, ks)| -> Result<_> {
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
    .into_iter()
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

/// The components, of those given with their ways, one of whose ways shares a node with a
/// `route=ferry` way of the `.pbf`.
fn at_ferry(pbf: &Path, ways: &BTreeMap<u32, Vec<i64>>) -> Result<BTreeSet<u32>> {
    let found: Mutex<(Vec<i64>, Vec<(i64, Vec<i64>)>)> = Mutex::new(Default::default());
    BlobReader::from_path(pbf)?.par_bridge().try_for_each(|blob| -> Result<()> {
        let (mut ferry, mut mine) = (Vec::new(), Vec::new());
        if let BlobDecode::OsmData(block) = blob?.decode()? {
            for g in block.groups() {
                for w in g.ways() {
                    if w.tags().any(|(k, v)| k == "route" && v == "ferry") {
                        ferry.extend(w.refs());
                    }
                    if ways.values().any(|v| v.binary_search(&w.id()).is_ok()) {
                        mine.push((w.id(), w.refs().collect()));
                    }
                }
            }
        }
        let mut f = found.lock().unwrap();
        f.0.extend(ferry);
        f.1.extend(mine);
        Ok(())
    })?;
    let (mut ferry, mut mine) = found.into_inner().unwrap();
    ferry.sort_unstable();
    mine.sort_unstable();
    let nodes: BTreeMap<i64, Vec<i64>> = mine.into_iter().collect();
    Ok(ways
        .iter()
        .filter(|(_, w)| w.iter().filter_map(|id| nodes.get(id)).flatten().any(|n| ferry.binary_search(n).is_ok()))
        .map(|(&x, _)| x)
        .collect())
}
