//! `runtime-random <model> <out dir> [--seed s] [--count n] [--named ids.txt]`: random
//! queries of every operation, answered by `blobmap_model::reference`, written as a golden
//! directory for `runtime-golden` to replay through blobmap-runtime. Columns, isochrones
//! (bounds from a minute to beyond the cap), nearest, multi-criteria (sources and category
//! targets, and/or/not, with and without k), chained trips and columns, pairs, reachable;
//! both directions, start times, same-piece pairs, and the named IDs (border sources) among
//! the sources. Fixed seed, so the same model gives the same file.

use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result, ensure};
use blobmap_model::NONE16;
use blobmap_model::reference::Runtime;
use blobmap_model::reference::ops::id_string;
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
    fn pick<'a, T>(&mut self, v: &'a [T]) -> &'a T {
        &v[self.below(v.len())]
    }
    fn chance(&mut self, pct: u64) -> bool {
        self.next() % 100 < pct
    }
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    ensure!(a.len() >= 2, "usage: runtime-random <model> <out dir> [--seed s] [--count n] [--named ids.txt]");
    let (mut seed, mut count, mut named) = (1u64, 300usize, Vec::<String>::new());
    let mut i = 2;
    while i < a.len() {
        let v = a.get(i + 1).context("option value")?;
        match a[i].as_str() {
            "--seed" => seed = v.parse()?,
            "--count" => count = v.parse()?,
            "--named" => named = std::fs::read_to_string(v)?.lines().map(|l| l.trim().to_string()).filter(|l| !l.is_empty()).collect(),
            x => anyhow::bail!("unknown option {x}"),
        }
        i += 2;
    }
    let out = Path::new(&a[1]);
    std::fs::create_dir_all(out)?;
    let scratch = out.join("scratch");
    let rt = Runtime::open_in(Path::new(&a[0]), Some(&scratch))?;
    rt.set_queries_only(true);
    let v = &rt.vocab;
    let n = v.n();
    let na = v.address.len();
    let mut r = Rng(0x2545_F491_4F6C_DD1D ^ seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let modes: Vec<String> = v.modes.iter().map(|m| m.info.name.clone()).collect();
    // Pools of IDs per mode: random served IDs, members of same-piece pairs, named IDs.
    let mut pool: Vec<Vec<String>> = vec![Vec::new(); modes.len()];
    let mut pairs: Vec<Vec<(String, String)>> = vec![Vec::new(); modes.len()];
    for (m, mv) in v.modes.iter().enumerate() {
        for _ in 0..4000 {
            let i = r.below(n) as u32;
            if rt.served(i, m).is_ok() {
                pool[m].push(id_string(&v.id(i)));
            }
            if pool[m].len() >= 60 {
                break;
            }
        }
        let mut keys: Vec<(u32, [u32; 2], u32, u32)> = (0..200_000).map(|_| r.below(n) as u32).filter(|&i| rt.served(i, m).is_ok()).map(|i| (mv.component[i as usize], mv.node[i as usize], mv.road[i as usize], i)).collect();
        keys.sort_unstable();
        for w in keys.windows(2) {
            if (w[0].0, w[0].1, w[0].2) == (w[1].0, w[1].1, w[1].2) && w[0].3 != w[1].3 && pairs[m].len() < 30 {
                pairs[m].push((id_string(&v.id(w[0].3)), id_string(&v.id(w[1].3))));
            }
        }
        for (x, y) in pairs[m].clone().iter().take(10) {
            pool[m].push(x.clone());
            pool[m].push(y.clone());
        }
        pool[m].extend(named.iter().cloned());
    }
    // Categories from places: primary, its parent, basic.
    let mut cats: Vec<Value> = Vec::new();
    for _ in 0..400 {
        if v.place.is_empty() {
            break;
        }
        let p = r.below(v.place.len());
        if v.status[p] != 0 || v.primary[p] == NONE16 {
            continue;
        }
        let c = v.primary[p] as usize;
        cats.push(json!({"category": v.tax_name[c]}));
        let up = v.tax_parent[c];
        if up != NONE16 {
            cats.push(json!({"category": v.tax_name[up as usize]}));
        }
        if v.basic[p] != NONE16 {
            cats.push(json!({"basic": v.tax_name[v.basic[p] as usize]}));
        }
        if cats.len() >= 40 {
            break;
        }
    }
    ensure!(!cats.is_empty(), "no categorised place");
    let dirs = ["from", "to"];
    let mut qs: Vec<(String, Value)> = Vec::new();
    let srcs = |r: &mut Rng, m: usize, k: usize| -> Value {
        (0..k)
            .map(|_| {
                let id = r.pick(&pool[m]).clone();
                if r.chance(40) { json!({"id": id, "start": r.below(900)}) } else { json!({"id": id}) }
            })
            .collect()
    };
    let car = modes.iter().position(|m| m == "car");
    let walk = modes.iter().position(|m| m == "walk");
    while qs.len() < count {
        let m = r.below(modes.len());
        if pool[m].is_empty() {
            continue;
        }
        let mode = &modes[m];
        let dir = *r.pick(&dirs);
        let k = 1 + r.below(3);
        let q = match r.below(100) {
            0..10 => ("column", json!({"sources": srcs(&mut r, m, k), "mode": mode, "dir": dir})),
            10..35 => {
                let t = *r.pick(&[60u64, 300, 900, 1800, 3600, 7200, 20000, 65534, 70000]);
                ("isochrone", json!({"sources": srcs(&mut r, m, k), "t": t, "mode": mode, "dir": dir, "ids": r.chance(10)}))
            }
            35..50 => ("nearest", json!({"category": r.pick(&cats), "id": r.pick(&pool[m]), "mode": mode, "dir": dir})),
            50..65 => {
                let term = |r: &mut Rng| -> Value {
                    let tm = r.below(modes.len());
                    let target = if r.chance(60) || pool[tm].is_empty() {
                        r.pick(&cats).clone()
                    } else {
                        let ids: Vec<Value> = (0..1 + r.below(2)).map(|_| json!(r.pick(&pool[tm]))).collect();
                        let starts: Vec<u64> = ids.iter().map(|_| r.below(600) as u64).collect();
                        if r.chance(50) { json!({"sources": ids, "start": starts}) } else { json!({"sources": ids}) }
                    };
                    let max = *r.pick(&[120u64, 300, 900, 1800, 3600, 70000]);
                    json!({"term": {"target": target, "mode": modes[tm], "dir": r.pick(&dirs), "max": max, "weight": 1 + r.below(5)}})
                };
                let expr = match r.below(4) {
                    0 => term(&mut r),
                    1 => json!({"and": [term(&mut r), term(&mut r)]}),
                    2 => json!({"or": [term(&mut r), {"not": term(&mut r)}]}),
                    _ => json!({"and": [term(&mut r), {"or": [term(&mut r), term(&mut r)]}]}),
                };
                let mut a = json!({"expr": expr, "kind": r.pick(&["addresses", "places", "both"])});
                if r.chance(40) {
                    a["k"] = json!(1 + r.below(10));
                }
                ("multi_criteria", a)
            }
            65..75 => {
                let (Some(c), Some(w)) = (car, walk) else { continue };
                if pool[c].is_empty() || pool[w].is_empty() {
                    continue;
                }
                let pen = *r.pick(&[0u64, 60, 300]);
                if r.chance(60) {
                    ("chained", json!({"a": r.pick(&pool[c]), "b": r.pick(&pool[w]), "penalty": pen, "transfer_category": r.pick(&cats)}))
                } else {
                    let first = if dir == "from" { c } else { w };
                    ("chained_column", json!({"id": r.pick(&pool[first]), "penalty": pen, "transfer_category": r.pick(&cats), "dir": dir}))
                }
            }
            75..90 => {
                let (x, y) = if !pairs[m].is_empty() && r.chance(50) {
                    let p = r.pick(&pairs[m]).clone();
                    if r.chance(50) { (p.0, p.1) } else { (p.1, p.0) }
                } else {
                    (r.pick(&pool[m]).clone(), r.pick(&pool[m]).clone())
                };
                ("travel_time", json!({"a": x, "b": y, "mode": mode}))
            }
            _ => {
                let op = if r.chance(50) { "reachable_from_all" } else { "reachable_from_any" };
                let k = 1 + r.below(3);
                let t = *r.pick(&[300u64, 900, 1800, 70000]);
                (op, json!({"sources": srcs(&mut r, m, k), "t": t, "mode": mode}))
            }
        };
        qs.push((q.0.to_string(), q.1));
    }
    let _ = na;
    let t0 = std::time::Instant::now();
    let lines: Vec<Result<String>> = qs
        .par_iter()
        .with_max_len(1)
        .map(|(op, args)| {
            let answer = rt.answer(op, args).with_context(|| format!("{op} {args}"))?;
            Ok(serde_json::to_string(&json!({"answer": answer, "args": args, "op": op}))?)
        })
        .collect();
    let mut body = Vec::new();
    let mut errors = 0;
    for l in lines {
        match l {
            Ok(l) => {
                body.extend_from_slice(l.as_bytes());
                body.push(b'\n');
            }
            Err(e) => {
                errors += 1;
                eprintln!("reference error: {e:#}");
            }
        }
    }
    std::fs::File::create(out.join("queries.jsonl"))?.write_all(&body)?;
    let header = json!({
        "format": "blobmap-golden",
        "format_version": "2.0.0",
        "inputs_hash": rt.manifest.inputs_hash,
        "queries_sha256": hex::encode(Sha256::digest(&body)),
        "tier": "random",
    });
    std::fs::write(out.join("golden.json"), serde_json::to_string_pretty(&header)?)?;
    let _ = std::fs::remove_dir_all(&scratch);
    eprintln!("{} queries ({errors} reference errors dropped) answered by the reference in {:.1} s", qs.len() - errors, t0.elapsed().as_secs_f64());
    Ok(())
}
