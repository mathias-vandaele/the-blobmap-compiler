//! The CubeCL trial (PLAN.md, Runtime): the research car sweep ported to CubeCL against the
//! research CUDA kernels, on the same GPU, the same layout and the same seeds.
//!
//! `cubecl-trial golden <model> <golden dir> cubecl|cuda`: replays the golden queries that
//! use car, every car column's downward sweep on the GPU (the reference does the rest).
//! `cubecl-trial time <model> <runs> <out.csv>`: times both on the largest car component.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use blobmap_model::INF;
use blobmap_model::runtime::query::DownSweep;
use blobmap_model::runtime::{Component, Dir, Fam, Runtime};
use blobmap_runtime::cube::{self, CubeSweep};
use blobmap_runtime::layered::{self, Entry, Layered};
use blobmap_runtime::research::{self, ResearchGpu};
use serde_json::Value;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Backend {
    Cube,
    Cuda,
}

enum Gpu {
    Cube(Box<CubeSweep>),
    Cuda(ResearchGpu),
}

impl Gpu {
    fn new(b: Backend, c: &Layered) -> Result<Gpu> {
        Ok(match b {
            Backend::Cube => Gpu::Cube(Box::new(CubeSweep::new(c, 1)?)),
            Backend::Cuda => Gpu::Cuda(ResearchGpu::new(c, 1)?),
        })
    }

    /// One column; graph schedule for both (each's best at B = 1).
    fn sweep(&mut self, entries: &[Entry], reversed: bool, graph: bool) -> Result<()> {
        match self {
            Gpu::Cube(g) => g.sweep(1, entries, reversed, if graph { cube::Schedule::Graph } else { cube::Schedule::PerLevel }),
            Gpu::Cuda(g) => g.sweep(1, entries, reversed, if graph { research::Schedule::Graph } else { research::Schedule::PerLevel }).map(|_| ()),
        }
    }

    fn download(&mut self) -> Result<Vec<u16>> {
        match self {
            Gpu::Cube(g) => g.download(1),
            Gpu::Cuda(g) => g.download(),
        }
    }
}

struct Loaded {
    c: Layered,
    gpu: Gpu,
}

/// The car columns' downward sweep on the GPU: the upward search on the CPU from the placed
/// seeds, then the GPU sweep, then the labels back in component-local ids.
struct GpuDown {
    backend: Backend,
    loaded: Mutex<BTreeMap<(u32, usize), Arc<Mutex<Loaded>>>>,
    sweeps: Mutex<u64>,
}

impl DownSweep for GpuDown {
    fn sweep(&self, e: &Component, fam: Fam, x: u32, metric: usize, label: &mut [u16], dir: Dir) -> Result<bool> {
        if fam != Fam::Car {
            return Ok(false);
        }
        let l = {
            let mut m = self.loaded.lock().unwrap();
            if let Some(l) = m.get(&(x, metric)) {
                l.clone()
            } else {
                let c = Layered::new(e, metric);
                let gpu = Gpu::new(self.backend, &c)?;
                let l = Arc::new(Mutex::new(Loaded { c, gpu }));
                m.insert((x, metric), l.clone());
                l
            }
        };
        let mut l = l.lock().unwrap();
        let seeds: Vec<(u32, u16)> = label.iter().enumerate().filter(|(_, t)| **t != INF).map(|(v, &t)| (v as u32, t)).collect();
        let entries = layered::upward(e, &l.c, metric, &seeds, dir, 0);
        l.gpu.sweep(&entries, dir == Dir::To, true)?;
        let out = l.gpu.download()?;
        for (v, t) in label.iter_mut().enumerate() {
            *t = out[l.c.new[v] as usize];
        }
        *self.sweeps.lock().unwrap() += 1;
        Ok(true)
    }
}

fn uses_car(q: &Value) -> bool {
    let op = q["op"].as_str().unwrap_or("");
    op.starts_with("chained") || q["args"].to_string().contains("\"car\"")
}

fn golden(model: &str, dir: &str, backend: Backend) -> Result<()> {
    let rt = Runtime::open(Path::new(model))?;
    rt.set_queries_only(true);
    let down = Arc::new(GpuDown { backend, loaded: Mutex::new(BTreeMap::new()), sweeps: Mutex::new(0) });
    if rt.down.set(down.clone()).is_err() {
        bail!("a sweep is already installed");
    }
    let header: Value = serde_json::from_slice(&std::fs::read(Path::new(dir).join("golden.json"))?)?;
    ensure!(header["inputs_hash"] == rt.manifest.inputs_hash.as_str(), "golden for another inputs_hash");
    let body = std::fs::read_to_string(Path::new(dir).join("queries.jsonl"))?;
    let (mut per_op, mut bad, mut n): (BTreeMap<String, (u64, u64)>, u64, u64) = (BTreeMap::new(), 0, 0);
    let t0 = Instant::now();
    for (i, line) in body.lines().enumerate() {
        let q: Value = serde_json::from_str(line)?;
        if !uses_car(&q) {
            continue;
        }
        let op = q["op"].as_str().context("op")?.to_string();
        let got = rt.answer(&op, &q["args"]);
        let ok = matches!(&got, Ok(a) if *a == q["answer"]);
        let e = per_op.entry(op.clone()).or_default();
        e.0 += 1;
        n += 1;
        if !ok {
            e.1 += 1;
            bad += 1;
            if bad <= 10 {
                eprintln!("MISMATCH line {}: {op} {} -> {:?}, golden {}", i + 1, q["args"], got.as_ref().map(|v| v.to_string()), q["answer"]);
            }
        }
    }
    for (op, (c, b)) in &per_op {
        println!("{backend:?} golden {op:>20} {c:>5} queries {b:>3} mismatches");
    }
    println!("{backend:?} golden: {n} car queries, {bad} mismatches, {} GPU sweeps, {:.1} s", down.sweeps.lock().unwrap(), t0.elapsed().as_secs_f64());
    ensure!(bad == 0, "{bad} mismatches");
    Ok(())
}

fn pct(v: &[f64], p: f64) -> f64 {
    let mut v = v.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    v[((v.len() - 1) as f64 * p).round() as usize]
}

fn time(model: &str, runs: usize, out: &str) -> Result<()> {
    let (free, total) = research::mem_info()?;
    eprintln!("GPU memory: {} MiB free of {} MiB", free >> 20, total >> 20);
    let rt = Runtime::open(Path::new(model))?;
    rt.set_queries_only(true);
    let metric = rt.vocab.modes[rt.mode("car")?].info.metric;
    let e = rt.component(Fam::Car, 0)?;
    let c = Layered::new(&e, metric);
    let warm = 20;
    let srcs = layered::sources(c.n, runs + warm);
    let seeds: Vec<Vec<Entry>> = srcs.iter().map(|&s| layered::upward(&e, &c, metric, &[(s, 0)], Dir::From, 0)).collect();
    let top_nodes = c.level_first[c.top];
    eprintln!(
        "car component 0: {} nodes, {} links, {} levels, top {} levels ({} nodes, {} B resident), {} wide levels",
        c.n,
        c.up_head.len(),
        c.levels(),
        c.top,
        top_nodes,
        c.resident_bytes(c.top),
        c.levels() - c.top
    );
    // Both backends' labels against the CPU sweep, on the first sources.
    let mut cubegpu = Gpu::new(Backend::Cube, &c)?;
    let mut cudagpu = Gpu::new(Backend::Cuda, &c)?;
    for (i, s) in seeds.iter().take(10).enumerate() {
        let cpu = c.cpu_down(s, Dir::From);
        for (name, g) in [("cubecl", &mut cubegpu), ("cuda", &mut cudagpu)] {
            for graph in [false, true] {
                g.sweep(s, false, graph)?;
                let got = g.download()?;
                let bad = (0..c.n).filter(|&v| got[v] != cpu[v]).count();
                ensure!(bad == 0, "{name} graph={graph} source {i}: {bad} labels differ from the CPU sweep");
            }
        }
    }
    eprintln!("labels: cubecl and cuda equal the CPU sweep on 10 sources, both schedules");
    let mut csv = std::fs::File::create(out)?;
    use std::io::Write;
    writeln!(csv, "backend,schedule,round,source,wall_ms")?;
    let mut res: BTreeMap<(&str, &str), Vec<f64>> = BTreeMap::new();
    // Two rounds, alternating backends, so that a drift of the GPU's clock or of the
    // desktop's use shows as a round difference rather than a backend difference.
    for round in 0..2 {
        for (name, g) in [("cubecl", &mut cubegpu), ("cuda", &mut cudagpu)] {
            for (sname, graph) in [("PerLevel", false), ("Graph", true)] {
                for (k, s) in seeds.iter().enumerate() {
                    let t = Instant::now();
                    g.sweep(s, false, graph)?;
                    let ms = t.elapsed().as_secs_f64() * 1e3;
                    if k >= warm {
                        res.entry((name, sname)).or_default().push(ms);
                        writeln!(csv, "{name},{sname},{round},{k},{ms:.4}")?;
                    }
                }
            }
        }
    }
    println!("backend  schedule   sweeps    p50 ms   p99 ms   mean ms");
    for ((name, sname), v) in &res {
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        println!("{name:<8} {sname:<9} {:>7} {:>9.3} {:>8.3} {:>9.3}", v.len(), pct(v, 0.5), pct(v, 0.99), mean);
    }
    let (free, _) = research::mem_info()?;
    eprintln!("GPU memory after: {} MiB free", free >> 20);
    Ok(())
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    match a.get(1).map(String::as_str) {
        Some("golden") if a.len() == 5 => golden(&a[2], &a[3], if a[4] == "cuda" { Backend::Cuda } else { Backend::Cube }),
        Some("time") if a.len() == 5 => time(&a[2], a[3].parse()?, &a[4]),
        _ => bail!("usage: cubecl-trial golden <model> <golden dir> cubecl|cuda | time <model> <runs> <out.csv>"),
    }
}
