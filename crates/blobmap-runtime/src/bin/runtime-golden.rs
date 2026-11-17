//! `runtime-golden <model> <golden dir> [options]`: replays a golden file (FORMAT §11)
//! through blobmap-runtime and counts the answers that differ. Options: `--backend
//! auto|cpu|cuda|wgpu`, `--threads n` (queries at once), `--min-nodes n` (components below
//! sweep on the CPU), `--b-max n`, `--budget-gib x`, `--no-prune`, `--format-levels`, `--ops a,b`,
//! `--gpu-family car|active` (the other family sweeps on the CPU), `--family
//! car|active|mixed` (only the queries of that family's modes, or of both).

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use blobmap_runtime::{Backend, Options, Runtime};
use rayon::prelude::*;
use serde_json::Value;
use sha2::{Digest, Sha256};

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    ensure!(a.len() >= 2, "usage: runtime-golden <model> <golden dir> [options]");
    let mut o = Options { gpu_min_nodes: 0, ..Options::default() };
    let mut threads = 8usize;
    let mut ops: Option<Vec<String>> = None;
    let mut family: Option<String> = None;
    let mut i = 2;
    while i < a.len() {
        let v = || a.get(i + 1).context("option value");
        match a[i].as_str() {
            "--backend" => o.backend = Backend::parse(v()?)?,
            "--threads" => threads = v()?.parse()?,
            "--min-nodes" => o.gpu_min_nodes = v()?.parse()?,
            "--b-max" => o.b_max = v()?.parse()?,
            "--budget-gib" => o.vram_budget = (v()?.parse::<f64>()? * (1u64 << 30) as f64) as u64,
            "--no-prune" => {
                o.prune = false;
                i -= 1;
            }
            "--format-levels" => {
                o.format_levels = true;
                i -= 1;
            }
            "--ops" => ops = Some(v()?.split(',').map(String::from).collect()),
            "--gpu-family" => o.gpu_families = [v()? == "car", v()? == "active"],
            "--family" => family = Some(v()?.clone()),
            x => bail!("unknown option {x}"),
        }
        i += 2;
    }
    let dir = Path::new(&a[1]);
    let t0 = Instant::now();
    let rt = Runtime::open(Path::new(&a[0]), o.clone())?;
    eprintln!("opened in {:.1} s; backend {}; options {o:?}", t0.elapsed().as_secs_f64(), rt.backend());
    let header: Value = serde_json::from_slice(&std::fs::read(dir.join("golden.json"))?)?;
    ensure!(header["format"] == "blobmap-golden", "not a golden header");
    ensure!(header["inputs_hash"] == rt.manifest.inputs_hash.as_str(), "golden for another inputs_hash");
    let body = std::fs::read(dir.join("queries.jsonl"))?;
    ensure!(header["queries_sha256"] == hex::encode(Sha256::digest(&body)).as_str(), "queries_sha256 differs");
    let mut queries: Vec<Value> = body.split(|&b| b == b'\n').filter(|l| !l.is_empty()).map(serde_json::from_slice).collect::<Result<_, _>>()?;
    if let Some(ops) = &ops {
        queries.retain(|q| ops.iter().any(|o| q["op"] == o.as_str()));
    }
    if let Some(f) = &family {
        // The families a query's modes belong to: car, active, or both (mixed).
        let of = |q: &Value| -> &str {
            let s = q.to_string();
            let car = s.contains("\"mode\":\"car\"") || q["op"].as_str().is_some_and(|o| o.starts_with("chained"));
            let act = s.contains("\"mode\":\"bike\"") || s.contains("\"mode\":\"walk\"") || q["op"].as_str().is_some_and(|o| o.starts_with("chained"));
            match (car, act) {
                (true, false) => "car",
                (false, true) => "active",
                _ => "mixed",
            }
        };
        queries.retain(|q| of(q) == f);
    }
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build()?;
    let per_op: Mutex<BTreeMap<String, (usize, usize, f64)>> = Mutex::default();
    let bad: Mutex<Vec<String>> = Mutex::default();
    let t1 = Instant::now();
    pool.install(|| {
        queries.par_iter().with_max_len(1).for_each(|q| {
            let op = q["op"].as_str().unwrap_or("?").to_string();
            let t = Instant::now();
            let got = rt.answer(&op, &q["args"]);
            let dt = t.elapsed().as_secs_f64();
            let ok = matches!(&got, Ok(x) if *x == q["answer"]);
            if !ok {
                let msg = match got {
                    Ok(x) => format!("{op} {}: got {x}, want {}", q["args"], q["answer"]),
                    Err(e) => format!("{op} {}: error {e:#}", q["args"]),
                };
                bad.lock().unwrap().push(msg);
            }
            let mut m = per_op.lock().unwrap();
            let e = m.entry(op).or_default();
            e.0 += 1;
            e.1 += !ok as usize;
            e.2 += dt;
        });
    });
    let wall = t1.elapsed().as_secs_f64();
    let per_op = per_op.into_inner().unwrap();
    let mut total = (0, 0);
    for (op, (n, b, s)) in &per_op {
        println!("{op:>20} {n:6} queries {b:5} mismatches  {:9.2} ms mean", 1e3 * s / *n as f64);
        total.0 += n;
        total.1 += b;
    }
    let bad = bad.into_inner().unwrap();
    for m in bad.iter().take(10) {
        println!("MISMATCH {}", if m.len() > 600 { &m[..600] } else { m });
    }
    if let Some(s) = rt.gpu_stats() {
        println!("gpu: {s:?}");
    }
    let ms = |a: &std::sync::atomic::AtomicU64| a.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e6;
    let pr = &rt.prof;
    println!(
        "columns: {} calls; summed ms: component load {:.0}, upward {:.0}, gpu wait {:.0}, cpu sweep {:.0}, heads {:.0}",
        pr.calls.load(std::sync::atomic::Ordering::Relaxed),
        ms(&pr.comp_load),
        ms(&pr.upward),
        ms(&pr.gpu_wait),
        ms(&pr.cpu_sweep),
        ms(&pr.heads)
    );
    let ld = |a: &std::sync::atomic::AtomicU64| a.load(std::sync::atomic::Ordering::Relaxed);
    println!(
        "heads: output vectors {:.0} ms, passes {:.0} ms; page faults {} minor, {} major; pool of {} threads",
        ms(&pr.heads_init),
        ms(&pr.heads_ids),
        ld(&pr.heads_minflt),
        ld(&pr.heads_majflt),
        ld(&pr.heads_threads)
    );
    println!("{}: {} queries, {} mismatches, {wall:.1} s on {threads} threads, backend {}", dir.display(), total.0, total.1, rt.backend());
    if total.1 > 0 {
        std::process::exit(1);
    }
    Ok(())
}
