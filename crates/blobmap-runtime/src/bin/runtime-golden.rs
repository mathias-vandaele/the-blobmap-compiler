//! `runtime-golden <model> <golden dir> [options]`: replays a golden file (FORMAT §11)
//! through blobmap-runtime and counts the answers that differ. Options: `--backend
//! auto|cpu|cuda|wgpu`, `--threads n` (queries at once), `--min-nodes n` (components below
//! sweep on the CPU), `--b-max n`, `--budget-gib x`, `--no-prune`, `--format-levels`, `--ops a,b`,
//! `--gpu-family car|active` (the other family sweeps on the CPU), `--family
//! car|active|mixed` (only the queries of that family's modes, or of both), `--repeat r`
//! (replay r times; timings from the last pass), `--sequential` (one query at a time, on
//! a pool of `--threads` threads). Per query type (operation and family): p50, p99 and mean
//! of each query's time, answer included; one query at a time, its phases too (component
//! load, upward, GPU wait, CPU sweep, heads).

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
    let mut repeat = 1usize;
    let mut sequential = false;
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
            "--repeat" => repeat = v()?.parse::<usize>()?.max(1),
            "--sequential" => {
                sequential = true;
                i -= 1;
            }
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
    // The families a query's modes belong to: car, active, or both (mixed).
    let of = |q: &Value| -> &'static str {
        let s = q.to_string();
        let car = s.contains("\"mode\":\"car\"") || q["op"].as_str().is_some_and(|o| o.starts_with("chained"));
        let act = s.contains("\"mode\":\"bike\"") || s.contains("\"mode\":\"walk\"") || q["op"].as_str().is_some_and(|o| o.starts_with("chained"));
        match (car, act) {
            (true, false) => "car",
            (false, true) => "active",
            _ => "mixed",
        }
    };
    if let Some(f) = &family {
        queries.retain(|q| of(q) == f);
    }
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build()?;
    // Per query type: mismatches, each query's seconds, and (one thread) its phases' summed ns.
    type Row = (usize, Vec<f64>, [u64; 5]);
    let phases = || -> [u64; 5] {
        let l = |a: &std::sync::atomic::AtomicU64| a.load(std::sync::atomic::Ordering::Relaxed);
        let p = &rt.prof;
        [l(&p.comp_load), l(&p.upward), l(&p.gpu_wait), l(&p.cpu_sweep), l(&p.heads)]
    };
    let mut per_op: BTreeMap<String, Row> = BTreeMap::new();
    let bad: Mutex<Vec<String>> = Mutex::default();
    let mut wall = 0.0;
    for pass in 0..repeat {
        let rows: Mutex<BTreeMap<String, Row>> = Mutex::default();
        bad.lock().unwrap().clear();
        let t1 = Instant::now();
        let one = |q: &Value| {
                let op = q["op"].as_str().unwrap_or("?").to_string();
                // The size too: an isochrone's or reachable's limit, a column's target count.
                let a = &q["args"];
                let size = match (a.get("t"), a.get("targets")) {
                    (Some(t), _) => format!(" t={t}"),
                    (_, Some(v)) => format!(" N={}", v.as_array().map_or(0, |v| v.len())),
                    _ => String::new(),
                };
                let key = format!("{op}{size} {}", of(q));
                let p0 = phases();
                let t = Instant::now();
                let got = rt.answer(&op, &q["args"]);
                let dt = t.elapsed().as_secs_f64();
                let p1 = phases();
                let ok = matches!(&got, Ok(x) if *x == q["answer"]);
                if !ok {
                    let msg = match got {
                        Ok(x) => format!("{op} {}: got {x}, want {}", q["args"], q["answer"]),
                        Err(e) => format!("{op} {}: error {e:#}", q["args"]),
                    };
                    bad.lock().unwrap().push(msg);
                }
                let mut m = rows.lock().unwrap();
                let e = m.entry(key).or_default();
                e.0 += !ok as usize;
                e.1.push(dt);
                for k in 0..5 {
                    e.2[k] += p1[k] - p0[k];
                }
        };
        pool.install(|| {
            if sequential {
                queries.iter().for_each(one);
            } else {
                queries.par_iter().with_max_len(1).for_each(one);
            }
        });
        wall = t1.elapsed().as_secs_f64();
        if pass + 1 == repeat {
            per_op = rows.into_inner().unwrap();
        }
    }
    let mut total = (0, 0);
    println!("{:>28} {:>7} {:>6} {:>10} {:>10} {:>10}{}", "query type", "queries", "bad", "p50 ms", "p99 ms", "mean ms", if threads == 1 || sequential { "   phases, mean ms: load / upward / gpu wait / cpu sweep / heads" } else { "" });
    for (op, (b, v, ph)) in per_op.iter_mut() {
        v.sort_by(f64::total_cmp);
        let n = v.len();
        let pct = |p: f64| 1e3 * v[((n as f64 - 1.0) * p).round() as usize];
        let mean = 1e3 * v.iter().sum::<f64>() / n as f64;
        let phase = if threads == 1 || sequential { ph.iter().map(|&x| format!("{:.2}", x as f64 / 1e6 / n as f64)).collect::<Vec<_>>().join(" / ") } else { String::new() };
        println!("{op:>28} {n:7} {b:6} {:10.2} {:10.2} {mean:10.2}   {phase}", pct(0.5), pct(0.99));
        total.0 += n;
        total.1 += *b;
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
    println!("{}: {} queries, {} mismatches, {wall:.1} s on {threads} threads{}, backend {}", dir.display(), total.0, total.1, if sequential { " one query at a time" } else { "" }, rt.backend());
    memory_report();
    if total.1 > 0 {
        std::process::exit(1);
    }
    Ok(())
}

/// The process's page faults since it started, and its cgroup's memory: peak, file pages
/// evicted and read again (workingset refaults), and the current mapped and anonymous bytes.
fn memory_report() {
    let (minor, major) = blobmap_runtime::runtime::faults();
    let cg = std::fs::read_to_string("/proc/self/cgroup").unwrap_or_default();
    let dir = format!("/sys/fs/cgroup{}", cg.lines().find_map(|l| l.strip_prefix("0::")).unwrap_or(""));
    let stat = std::fs::read_to_string(format!("{dir}/memory.stat")).unwrap_or_default();
    let field = |k: &str| stat.lines().find_map(|l| l.strip_prefix(k).and_then(|v| v.trim().parse::<u64>().ok())).unwrap_or(0);
    let peak = std::fs::read_to_string(format!("{dir}/memory.peak")).ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(0);
    let max = std::fs::read_to_string(format!("{dir}/memory.max")).unwrap_or_default();
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    println!(
        "memory: page faults {minor} minor, {major} major; cgroup max {}, peak {:.1} GiB, file refaults {} ({:.1} GiB of 4 KiB pages), now file {:.1} GiB, mapped {:.1} GiB, anon {:.1} GiB",
        max.trim(),
        gib(peak),
        field("workingset_refault_file "),
        gib(field("workingset_refault_file ") * 4096),
        gib(field("file ")),
        gib(field("file_mapped ")),
        gib(field("anon ")),
    );
}
