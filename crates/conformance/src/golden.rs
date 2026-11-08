//! Golden queries (FORMAT §11): generated from the reference, written per tier and
//! scenario, and replayed against any runtime with the same `inputs_hash`.

use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result, ensure};
use rayon::prelude::*;
use blobmap_runtime::Runtime;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::Samples;
use crate::expected::{Expected, uuid};
use crate::report::Report;

pub const HEADER: &str = "golden.json";
pub const QUERIES: &str = "queries.jsonl";

/// IDs a scenario names, plus an even sample of the model's IDs (at most `k`).
pub fn sample(e: &Expected, named: &[String], k: usize) -> Vec<String> {
    let n = e.r.n();
    let step = n.div_ceil(k.max(1)).max(1);
    let mut out: Vec<String> = (0..n).step_by(step).map(|i| uuid(&e.r.ids[i])).collect();
    for s in named {
        if !out.contains(s) {
            out.push(s.clone());
        }
    }
    out
}

/// The golden queries of a scenario, with their reference answers.
pub fn queries(e: &Expected, named: &[String], n: &Samples) -> Vec<Value> {
    let keys = category_keys(e, n.categories);
    let step = e.r.aliases.len().div_ceil(n.aliases.min(e.r.aliases.len()).max(1)).max(1);
    let aliases: Vec<&(blobmap_overture::Id, u32)> = e.r.aliases.iter().step_by(step).collect();
    let ids = sample(e, named, 24);
    let mut q: Vec<(String, Value)> = Vec::new();
    let modes = ["car", "bike", "walk"];
    for m in modes {
        for a in &ids {
            for b in &ids {
                q.push(("travel_time".into(), json!({"a": a, "b": b, "mode": m})));
            }
        }
        q.push(("travel_time".into(), json!({"a": "00000000-0000-4000-8000-000000000000", "b": ids[0], "mode": m})));
        for (alias, kept) in aliases.iter().copied() {
            q.push(("travel_time".into(), json!({"a": uuid(alias), "b": uuid(&e.r.ids[*kept as usize]), "mode": m})));
            q.push(("travel_time".into(), json!({"a": ids[0], "b": uuid(alias), "mode": m})));
        }
        let mi = e.r.modes.iter().position(|x| x.name == m).unwrap();
        let served: Vec<&String> = ids
            .iter()
            .filter(|s| blobmap_overture::parse_uuid(s).ok().and_then(|id| e.r.lookup(&id)).is_some_and(|i| e.r.modes[mi].at[i as usize].is_ok()))
            .collect();
        for dir in ["from", "to"] {
            for w in served.windows(3).step_by(3).take(3) {
                let src = json!([{"id": w[0]}, {"id": w[1], "start": 30}, {"id": w[2], "start": 400}]);
                q.push(("column".into(), json!({"sources": src, "mode": m, "dir": dir})));
                for t in [60, 300, 3600] {
                    q.push(("isochrone".into(), json!({"sources": src, "t": t, "mode": m, "dir": dir})));
                }
            }
            for s in served.iter().take(6) {
                for key in &keys {
                    q.push(("nearest".into(), json!({"category": key, "id": s, "mode": m, "dir": dir})));
                }
            }
        }
        for w in served.windows(2).step_by(2).take(3) {
            for op in ["reachable_from_all", "reachable_from_any"] {
                q.push((op.into(), json!({"sources": [{"id": w[0]}, {"id": w[1], "start": 20}], "t": 600, "mode": m})));
            }
        }
        if let [a, b, ..] = served[..] {
            let term = |target: Value, mode: &str, dir: &str, max: u64, weight: u64| {
                json!({"term": {"target": target, "mode": mode, "dir": dir, "max": max, "weight": weight}})
            };
            let s1 = term(json!({"sources": [a], "start": [0]}), m, "from", 900, 2);
            let s2 = term(json!({"sources": [b]}), m, "to", 1200, 3);
            let mut exprs = vec![json!({"and": [s1.clone(), {"not": s2.clone()}]}), json!({"or": [s1.clone(), s2.clone()]})];
            for key in keys.iter().take(2) {
                exprs.push(json!({"and": [term(key.clone(), m, "from", 1800, 1), s1.clone()]}));
            }
            for expr in exprs {
                for kind in ["addresses", "places", "both"] {
                    q.push(("multi_criteria".into(), json!({"expr": expr, "kind": kind})));
                    q.push(("multi_criteria".into(), json!({"expr": expr, "kind": kind, "k": 5})));
                }
            }
        }
    }
    // Chained trips (SPEC §9): car, a penalty, then walk; no mode. Besides the sample, an
    // unknown ID and the first IDs not served for car and for walk.
    let unserved = |mode: &str| -> Option<String> {
        let mi = e.r.modes.iter().position(|x| x.name == mode).unwrap();
        (0..e.r.n()).find(|&i| e.r.modes[mi].at[i].is_err()).map(|i| uuid(&e.r.ids[i]))
    };
    let mut odd = vec!["00000000-0000-4000-8000-000000000000".to_string()];
    odd.extend(unserved("car"));
    odd.extend(unserved("walk"));
    for key in keys.iter().take(3) {
        for x in &odd {
            q.push(("chained".into(), json!({"a": x, "b": ids[0], "transfer_category": key, "penalty": 60})));
            q.push(("chained".into(), json!({"a": ids[0], "b": x, "transfer_category": key, "penalty": 60})));
            for dir in ["from", "to"] {
                q.push(("chained_column".into(), json!({"id": x, "transfer_category": key, "penalty": 60, "dir": dir})));
            }
        }
        for penalty in [0, 300] {
            for a in ids.iter().take(8) {
                for b in ids.iter().take(8) {
                    q.push(("chained".into(), json!({"a": a, "b": b, "transfer_category": key, "penalty": penalty})));
                }
            }
            for dir in ["from", "to"] {
                for id in ids.iter().take(4) {
                    q.push(("chained_column".into(), json!({"id": id, "transfer_category": key, "penalty": penalty, "dir": dir})));
                }
            }
        }
    }
    q.into_par_iter().map(|(op, args)| json!({"op": op, "answer": e.answer(&op, &args), "args": args})).collect()
}

/// Category arguments: every taxonomy node a kept place uses, and its basic categories;
/// at most `max`, evenly sampled, plus a redirected key.
fn category_keys(e: &Expected, max: usize) -> Vec<Value> {
    let t = &e.r.catalog.taxonomy;
    let mut nodes: Vec<u16> = e.r.catalog.places.iter().filter(|p| p.status == 0 && p.primary != blobmap_format::NONE16).map(|p| p.primary).collect();
    nodes.sort_unstable();
    nodes.dedup();
    let mut out: Vec<Value> = Vec::new();
    for &n in &nodes {
        out.push(json!({"category": t.name[n as usize]}));
        if t.parent[n as usize] != blobmap_format::NONE16 {
            let p = &t.name[t.parent[n as usize] as usize];
            if !out.contains(&json!({"category": p})) {
                out.push(json!({"category": p}));
            }
        }
        if t.basic[n as usize] != blobmap_format::NONE16 {
            let b = json!({"basic": t.name[t.basic[n as usize] as usize]});
            if !out.contains(&b) {
                out.push(b);
            }
        }
    }
    let step = out.len().div_ceil(max.min(out.len()).max(1)).max(1);
    let mut out: Vec<Value> = out.into_iter().step_by(step).collect();
    // A redirected key, when the taxonomy has one with a target.
    if let Some(k) = (0..t.redirect.len()).find(|&k| t.redirect_to[k] != blobmap_format::NONE16) {
        out.push(json!({"category": t.redirect[k]}));
    }
    out
}

pub fn write(dir: &Path, tier: &str, inputs_hash: &str, lines: &[Value]) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let compact: Vec<u8> = lines
        .iter()
        .flat_map(|l| {
            let mut s = serde_json::to_vec(l).unwrap();
            s.push(b'\n');
            s
        })
        .collect();
    let mut f = std::fs::File::create(dir.join(QUERIES))?;
    f.write_all(&compact)?;
    let header = json!({
        "format": "blobmap-golden",
        "format_version": blobmap_format::FORMAT_VERSION,
        "tier": tier,
        "inputs_hash": inputs_hash,
        "queries_sha256": hex::encode(Sha256::digest(&compact)),
    });
    blobmap_format::json::write_file(&dir.join(HEADER), &header)
}

/// Replays a golden directory: every answer must match exactly.
pub fn replay(rt: &Runtime, dir: &Path, rep: &mut Report) -> Result<()> {
    let header = blobmap_format::json::read_file(&dir.join(HEADER))?;
    ensure!(header["format"] == "blobmap-golden", "{}: not a golden header", dir.display());
    ensure!(header["inputs_hash"] == rt.manifest.inputs_hash.as_str(), "{}: golden for another inputs_hash (FORMAT §11)", dir.display());
    let body = std::fs::read(dir.join(QUERIES)).with_context(|| format!("reading {}", dir.display()))?;
    ensure!(header["queries_sha256"] == hex::encode(Sha256::digest(&body)).as_str(), "{}: queries_sha256 differs", dir.display());
    let o = rep.check("golden replay");
    for line in body.split(|&b| b == b'\n').filter(|l| !l.is_empty()) {
        let q: Value = serde_json::from_slice(line)?;
        o.count();
        let op = q["op"].as_str().context("op")?;
        match rt.answer(op, &q["args"]) {
            Ok(a) if a == q["answer"] => {}
            Ok(a) => o.fail(format!("{op} {}: {a} != {}", q["args"], q["answer"])),
            Err(e) => o.fail(format!("{op} {}: {e:#}", q["args"])),
        }
    }
    Ok(())
}
