//! `runtime-matrix <model> --n N --mode m [--dir from|to] [--seed s] [--backend
//! cpu|cuda|wgpu|reference] [--threads t] [--b-max B] [--budget-gib g]`: an N×N matrix
//! built from SPEC §9 operations: N columns at listed targets, one per source, over the
//! same N served IDs (sources = targets). The columns go in t at a time, so that the GPU
//! worker batches them up to B a sweep. `reference` builds the same matrix from
//! `blobmap_model::reference` columns. Prints the wall time, the time per pair and the
//! matrix's SHA-256 (row by row, u16 little-endian, 65535 for BEYOND) for comparison.

use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use blobmap_model::INF;
use blobmap_runtime::ops::Source;
use blobmap_runtime::vocab::{Dir, id_string};
use blobmap_runtime::{Backend, Options, Runtime};
use rayon::prelude::*;
use sha2::{Digest, Sha256};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    ensure!(!a.is_empty(), "usage: runtime-matrix <model> --n N --mode m [options]");
    let mut o = Options { gpu_min_nodes: 0, ..Options::default() };
    let (mut n, mut mode, mut dir, mut seed, mut threads, mut reference) = (100usize, "car".to_string(), Dir::From, 1u64, 32usize, false);
    let mut i = 1;
    while i < a.len() {
        let v = a.get(i + 1).context("option value")?;
        match a[i].as_str() {
            "--n" => n = v.parse()?,
            "--mode" => mode = v.clone(),
            "--dir" => dir = if v == "to" { Dir::To } else { Dir::From },
            "--seed" => seed = v.parse()?,
            "--threads" => threads = v.parse()?,
            "--backend" if v == "reference" => reference = true,
            "--backend" => o.backend = Backend::parse(v)?,
            "--b-max" => o.b_max = v.parse()?,
            "--budget-gib" => o.vram_budget = (v.parse::<f64>()? * (1u64 << 30) as f64) as u64,
            x => bail!("unknown option {x}"),
        }
        i += 2;
    }
    let t0 = Instant::now();
    let rt = Runtime::open(Path::new(&a[0]), o.clone())?;
    let m = rt.mode(&mode)?;
    // N distinct served IDs, by a fixed seed.
    let total = rt.vocab.n();
    let mut r = Rng(0x2545_F491_4F6C_DD1D ^ seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let mut idx: Vec<u32> = Vec::with_capacity(n);
    let mut seen = std::collections::BTreeSet::new();
    while idx.len() < n {
        let i = (r.next() % total as u64) as u32;
        if rt.served(i, m).is_ok() && seen.insert(i) {
            idx.push(i);
        }
    }
    let ids: Vec<String> = idx.iter().map(|&i| id_string(&rt.vocab.id(i))).collect();
    let opened = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    let rows: Vec<Vec<u16>> = if reference {
        drop(rt);
        let rf = blobmap_model::reference::Runtime::open(Path::new(&a[0]))?;
        rf.set_queries_only(true);
        let rd = if dir == Dir::From { blobmap_model::reference::Dir::From } else { blobmap_model::reference::Dir::To };
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build()?;
        pool.install(|| {
            idx.par_iter()
                .with_max_len(1)
                .map(|&s| {
                    let col = rf.column(&[blobmap_model::reference::query::Source { id: s, start: 0 }], m, rd)?;
                    Ok(idx.iter().map(|&t| col[t as usize]).collect())
                })
                .collect::<Result<Vec<Vec<u16>>>>()
        })?
    } else {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build()?;
        pool.install(|| {
            ids.par_iter()
                .with_max_len(1)
                .map(|s| match rt.column_at(&[Source { id: s.clone(), start: 0 }], &mode, dir, &ids)? {
                    Ok(t) => Ok(t),
                    Err(e) => bail!("column from {s}: {} {}", e.reason.name(), e.id),
                })
                .collect::<Result<Vec<Vec<u16>>>>()
        })?
    };
    let wall = t1.elapsed().as_secs_f64();
    let mut h = Sha256::new();
    let mut beyond = 0usize;
    for row in &rows {
        for &t in row {
            beyond += (t == INF) as usize;
            h.update(t.to_le_bytes());
        }
    }
    let pairs = (n * n) as f64;
    println!(
        "matrix {n}x{n} {mode} {}: {wall:.2} s wall, {:.3} µs a pair, {beyond} BEYOND; sha256 {}; backend {}, {threads} threads, opened in {opened:.1} s",
        if dir == Dir::From { "from" } else { "to" },
        1e6 * wall / pairs,
        hex::encode(h.finalize()),
        if reference { "reference".to_string() } else { format!("{:?}, B max {}", o.backend, o.b_max) },
    );
    Ok(())
}
