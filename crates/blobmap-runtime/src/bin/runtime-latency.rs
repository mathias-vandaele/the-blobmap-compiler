//! `runtime-latency <model> <golden dir> <out dir> [--seed s]`: the R2 latency set, a golden
//! directory for `runtime-golden` answered by `blobmap_model::reference`:
//!
//! - isochrones from one source, at 15, 30 and 60 min, every mode and direction (4 each);
//! - columns answered only at a list of N target IDs (1×N), N = 100, 1,000 and 10,000,
//!   every mode and direction (2 each);
//! - the golden directory's queries of the other types at their golden sizes, with their
//!   golden answers: travel times, nearest, isochrones, reachable, multi-criteria and
//!   chained trips (not its full columns and chained columns, which carry no limit).
//!
//! Fixed seed, so the same model gives the same file.

use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result, ensure};
use blobmap_model::INF;
use blobmap_model::reference::ops::id_string;
use blobmap_model::reference::query::Source;
use blobmap_model::reference::{Dir, Runtime};
use rayon::prelude::*;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
}

/// Golden operations kept with their answers.
const KEEP: [&str; 7] = ["travel_time", "nearest", "isochrone", "reachable_from_all", "reachable_from_any", "multi_criteria", "chained"];

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    ensure!(a.len() >= 3, "usage: runtime-latency <model> <golden dir> <out dir> [--seed s]");
    let mut seed = 1u64;
    let mut i = 3;
    while i < a.len() {
        let v = a.get(i + 1).context("option value")?;
        match a[i].as_str() {
            "--seed" => seed = v.parse()?,
            x => anyhow::bail!("unknown option {x}"),
        }
        i += 2;
    }
    let out = Path::new(&a[2]);
    std::fs::create_dir_all(out)?;
    let scratch = out.join("scratch");
    let rt = Runtime::open_in(Path::new(&a[0]), Some(&scratch))?;
    rt.set_queries_only(true);
    let v = &rt.vocab;
    let n = v.n();
    let mut r = Rng(0x2545_F491_4F6C_DD1D ^ seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let modes: Vec<String> = v.modes.iter().map(|m| m.info.name.clone()).collect();
    let served = |r: &mut Rng, m: usize| -> u32 {
        loop {
            let i = r.below(n) as u32;
            if rt.served(i, m).is_ok() {
                return i;
            }
        }
    };
    // (op, args, how to answer): isochrones and columns at targets answered here.
    enum Job {
        Iso { m: usize, dir: Dir, src: u32, t: u64 },
        At { m: usize, dir: Dir, src: u32, targets: Vec<u32> },
    }
    let mut jobs: Vec<(String, Value, Job)> = Vec::new();
    for (m, mode) in modes.iter().enumerate() {
        for dir in [Dir::From, Dir::To] {
            for t in [900u64, 1800, 3600] {
                for _ in 0..4 {
                    let src = served(&mut r, m);
                    let args = json!({"sources": [{"id": id_string(&v.id(src))}], "t": t, "mode": mode, "dir": if *&dir == Dir::From { "from" } else { "to" }});
                    jobs.push(("isochrone".into(), args, Job::Iso { m, dir, src, t }));
                }
            }
            for k in [100usize, 1000, 10000] {
                for _ in 0..2 {
                    let src = served(&mut r, m);
                    // Targets: any IDs (served or not, any component), as a client would list them.
                    let targets: Vec<u32> = (0..k).map(|_| r.below(n) as u32).collect();
                    let args = json!({"sources": [{"id": id_string(&v.id(src))}], "mode": mode, "dir": if *&dir == Dir::From { "from" } else { "to" }, "targets": targets.iter().map(|&i| id_string(&v.id(i))).collect::<Vec<_>>()});
                    jobs.push(("column".into(), args, Job::At { m, dir, src, targets }));
                }
            }
        }
    }
    let t0 = std::time::Instant::now();
    let lines: Vec<Result<String>> = jobs
        .par_iter()
        .with_max_len(1)
        .map(|(op, args, job)| {
            let answer = match job {
                Job::Iso { m, dir, src, t } => {
                    let col = rt.column(&[Source { id: *src, start: 0 }], *m, *dir)?;
                    let (_, sha) = rt.id_set((0..col.len() as u32).filter(|&i| col[i as usize] != INF && col[i as usize] as u64 <= *t));
                    let count = col.iter().filter(|&&x| x != INF && x as u64 <= *t).count();
                    json!({"count": count, "ids_sha256": sha})
                }
                Job::At { m, dir, src, targets } => {
                    let col = rt.column(&[Source { id: *src, start: 0 }], *m, *dir)?;
                    json!({"times": targets.iter().map(|&i| { let x = col[i as usize]; if x == INF { json!("BEYOND") } else { json!(x) } }).collect::<Vec<_>>()})
                }
            };
            Ok(serde_json::to_string(&json!({"answer": answer, "args": args, "op": op}))?)
        })
        .collect();
    let mut body = Vec::new();
    for l in lines {
        body.extend_from_slice(l?.as_bytes());
        body.push(b'\n');
    }
    let made = jobs.len();
    // The golden queries kept, as they are.
    let golden = std::fs::read_to_string(Path::new(&a[1]).join("queries.jsonl"))?;
    let mut kept = 0;
    for l in golden.lines() {
        let q: Value = serde_json::from_str(l)?;
        if KEEP.contains(&q["op"].as_str().unwrap_or("")) {
            body.extend_from_slice(l.as_bytes());
            body.push(b'\n');
            kept += 1;
        }
    }
    std::fs::File::create(out.join("queries.jsonl"))?.write_all(&body)?;
    let header = json!({
        "format": "blobmap-golden",
        "format_version": "2.0.0",
        "inputs_hash": rt.manifest.inputs_hash,
        "queries_sha256": hex::encode(Sha256::digest(&body)),
        "tier": "latency",
    });
    std::fs::write(out.join("golden.json"), serde_json::to_string_pretty(&header)?)?;
    let _ = std::fs::remove_dir_all(&scratch);
    eprintln!("{made} queries answered by the reference in {:.1} s, {kept} golden queries kept", t0.elapsed().as_secs_f64());
    Ok(())
}
