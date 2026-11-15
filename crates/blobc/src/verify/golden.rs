//! Golden queries (FORMAT §11): generated from the reference, written per tier and
//! scenario, and replayed against any runtime with the same `inputs_hash`.

use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result, ensure};
use blobmap_model::runtime::Runtime;
use rayon::prelude::*;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::verify::Samples;
use crate::verify::expected::{Expected, uuid};
use crate::verify::report::Report;

pub const HEADER: &str = "golden.json";
pub const QUERIES: &str = "queries.jsonl";
/// The golden format's own version (FORMAT §11), independent of the model format's.
pub const GOLDEN_VERSION: &str = "2.0.0";

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
    let aliases: Vec<&(crate::overture::Id, u32)> = e.r.aliases.iter().step_by(step).collect();
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
            .filter(|s| crate::overture::parse_uuid(s).ok().and_then(|id| e.r.lookup(&id)).is_some_and(|i| e.r.modes[mi].at(i as usize).is_ok()))
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
        (0..e.r.n()).find(|&i| e.r.modes[mi].at(i).is_err()).map(|i| uuid(&e.r.ids[i]))
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
    let mut nodes: Vec<u16> = e.r.catalog.places.iter().filter(|p| p.status == 0 && p.primary != blobmap_model::NONE16).map(|p| p.primary).collect();
    nodes.sort_unstable();
    nodes.dedup();
    let mut out: Vec<Value> = Vec::new();
    for &n in &nodes {
        out.push(json!({"category": t.name[n as usize]}));
        if t.parent[n as usize] != blobmap_model::NONE16 {
            let p = &t.name[t.parent[n as usize] as usize];
            if !out.contains(&json!({"category": p})) {
                out.push(json!({"category": p}));
            }
        }
        if t.basic[n as usize] != blobmap_model::NONE16 {
            let b = json!({"basic": t.name[t.basic[n as usize] as usize]});
            if !out.contains(&b) {
                out.push(b);
            }
        }
    }
    let step = out.len().div_ceil(max.min(out.len()).max(1)).max(1);
    let mut out: Vec<Value> = out.into_iter().step_by(step).collect();
    // A redirected key, when the taxonomy has one with a target.
    if let Some(k) = (0..t.redirect.len()).find(|&k| t.redirect_to[k] != blobmap_model::NONE16) {
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
        "format_version": GOLDEN_VERSION,
        "tier": tier,
        "inputs_hash": inputs_hash,
        "queries_sha256": hex::encode(Sha256::digest(&compact)),
    });
    blobmap_model::json::write_file(&dir.join(HEADER), &header)
}

/// Replays a golden directory: every answer must match exactly. The queries run grouped by
/// the component of their first ID, so that each component is loaded once and its queries
/// run together, on as many threads as the memory left holds the working state of (one
/// `Work` per thread, reused from query to query). Neither the order nor the thread count
/// changes an answer; failures are reported in file order.
pub fn replay(rt: &Runtime, dir: &Path, rep: &mut Report) -> Result<()> {
    replay_on(rt, dir, rep, None)
}

/// `blobc replay <model> <golden dir> [--threads n]`: the replay alone, in a work directory
/// next to the model as verify's; true when every answer matches.
pub fn main(a: Vec<String>) -> Result<bool> {
    let (model, dir, threads) = match &a[..] {
        [m, d] => (m, d, None),
        [m, d, t, n] if t == "--threads" => (m, d, Some(n.parse::<usize>()?)),
        _ => anyhow::bail!("usage: blobc replay <model> <golden dir> [--threads n]"),
    };
    let work = crate::verify::work_dir(Path::new(model), "replay");
    std::fs::create_dir_all(&work)?;
    let mut rep = Report::default();
    let r = Runtime::open_in(Path::new(model), Some(&work.join("runtime"))).and_then(|rt| {
        rt.set_queries_only(true);
        replay_on(&rt, Path::new(dir), &mut rep, threads)
    });
    let _ = std::fs::remove_dir_all(&work);
    r?;
    print!("{rep}");
    println!("{} mismatches", rep.mismatches());
    Ok(rep.mismatches() == 0)
}

fn replay_on(rt: &Runtime, dir: &Path, rep: &mut Report, threads: Option<usize>) -> Result<()> {
    let header = blobmap_model::json::read_file(&dir.join(HEADER))?;
    ensure!(header["format"] == "blobmap-golden" && header["format_version"] == GOLDEN_VERSION, "{}: not a {GOLDEN_VERSION} golden header", dir.display());
    ensure!(header["inputs_hash"] == rt.manifest.inputs_hash.as_str(), "{}: golden for another inputs_hash (FORMAT §11)", dir.display());
    let body = std::fs::read(dir.join(QUERIES)).with_context(|| format!("reading {}", dir.display()))?;
    ensure!(header["queries_sha256"] == hex::encode(Sha256::digest(&body)).as_str(), "{}: queries_sha256 differs", dir.display());
    let lines: Vec<&[u8]> = body.split(|&b| b == b'\n').filter(|l| !l.is_empty()).collect();
    let queries: Vec<Result<Value>> = lines.par_iter().map(|l| Ok(serde_json::from_slice(l)?)).collect();
    let mut order: Vec<usize> = (0..queries.len()).collect();
    order.sort_by_key(|&k| (queries[k].as_ref().map_or((u8::MAX, u32::MAX), |q| component_of(rt, q)), k));
    // A pool of a thread per physical core (or `threads`), and as many queries at once as
    // half the free memory holds working states for: each query's own loops use the pool.
    let cores = crate::build::mem::physical_cores().unwrap_or(usize::MAX).min(rayon::current_num_threads());
    let threads = threads.unwrap_or(cores).max(1);
    let lanes = replay_lanes(rt).min(threads);
    let t0 = std::time::Instant::now();
    let c0 = crate::build::mem::cpu_seconds();
    let works = std::sync::Mutex::new(Vec::new());
    let times = std::sync::Mutex::new(Vec::new());
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build()?;
    let mut results: Vec<Option<Result<Option<String>>>> = (0..queries.len()).map(|_| None).collect();
    // One group (component) at a time, its queries in parallel.
    let mut at = 0;
    while at < order.len() {
        let key = |k: usize| queries[k].as_ref().map_or((u8::MAX, u32::MAX), |q| component_of(rt, q));
        let g = key(order[at]);
        let len = order[at..].iter().take_while(|&&k| key(k) == g).count();
        let group = &order[at..at + len];
        let answers: Vec<Result<Option<String>>> = crate::verify::in_lanes(Some(&pool), lanes, group, |&k| {
            let q = queries[k].as_ref().map_err(|e| anyhow::anyhow!("{e:#}"))?;
            let op = q["op"].as_str().context("op")?;
            let mut w = works.lock().unwrap().pop().unwrap_or_default();
            let t = std::time::Instant::now();
            let got = rt.answer_in(&mut w, op, &q["args"]);
            times.lock().unwrap().push((t.elapsed().as_secs_f64(), k));
            works.lock().unwrap().push(w);
            Ok(match got {
                Ok(a) if a == q["answer"] => None,
                Ok(a) => Some(format!("{op} {}: {a} != {}", q["args"], q["answer"])),
                Err(e) => Some(format!("{op} {}: {e:#}", q["args"])),
            })
        });
        for (&k, a) in group.iter().zip(answers) {
            results[k] = Some(a);
        }
        at += len;
    }
    let held: usize = works.lock().unwrap().iter().map(|w| w.bytes()).sum();
    eprintln!(
        "golden replay: {} queries, {lanes} at once on {threads} threads, {:.1} s, {:.0} core-seconds, working state {} MiB",
        queries.len(),
        t0.elapsed().as_secs_f64(),
        crate::build::mem::cpu_seconds() - c0,
        held >> 20
    );
    // The slowest queries, and the time per op: what bounds the replay's wall time.
    let mut times = times.into_inner().unwrap();
    times.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut per_op: std::collections::BTreeMap<&str, (usize, f64)> = Default::default();
    for &(t, k) in &times {
        let op = queries[k].as_ref().ok().and_then(|q| q["op"].as_str()).unwrap_or("?");
        let e = per_op.entry(op).or_default();
        e.0 += 1;
        e.1 += t;
    }
    for (op, (n, t)) in &per_op {
        eprintln!("golden replay: {op:>20} {n:>5} queries {t:>9.1} s");
    }
    for &(t, k) in times.iter().take(5) {
        let q = queries[k].as_ref().ok();
        eprintln!("golden replay: slowest {t:.1} s: line {} {}", k + 1, q.and_then(|q| q["op"].as_str()).unwrap_or("?"));
    }
    let results: Vec<Result<Option<String>>> = results.into_iter().map(|r| r.expect("every query answered")).collect();
    let o = rep.check("golden replay");
    for res in results {
        o.count();
        if let Some(f) = res? {
            o.fail(f);
        }
    }
    Ok(())
}

/// The family and component of a query's first ID, for grouping (`u8::MAX` when it has none:
/// an unknown ID or an unserved one, answered by a reason without a component).
fn component_of(rt: &Runtime, q: &Value) -> (u8, u32) {
    let a = &q["args"];
    let mode = a["mode"].as_str().or_else(|| first_term(&a["expr"]).and_then(|t| t["mode"].as_str()));
    let mode = match q["op"].as_str() {
        Some("chained") | Some("chained_column") => Some("car"),
        _ => mode,
    };
    let id = a["a"].as_str().or(a["id"].as_str()).or_else(|| a["sources"][0]["id"].as_str()).or_else(|| first_term(&a["expr"]).and_then(|t| t["target"]["sources"][0].as_str()));
    let (Some(mode), Some(id)) = (mode, id) else { return (u8::MAX, u32::MAX) };
    let (Ok(m), Ok(i)) = (rt.mode(mode), rt.resolve(id)) else { return (u8::MAX, u32::MAX) };
    let mv = &rt.vocab.modes[m];
    match mv.component[i as usize] {
        blobmap_model::NONE => (u8::MAX, u32::MAX),
        x => (mv.info.fam as u8, x),
    }
}

fn first_term(e: &Value) -> Option<&Value> {
    if let Some(t) = e.get("term") {
        return Some(t);
    }
    e.get("and").or_else(|| e.get("or")).and_then(|x| x.as_array()).and_then(|x| x.first()).and_then(first_term).or_else(|| e.get("not").and_then(first_term))
}

/// Queries at once in the replay: as many as half the memory left holds one query's working state
/// (four columns of every ID, a column's IDs by component, and a sweep and two ancestor
/// searches over the largest component), at least one. The pool has a thread per physical core: an SMT sibling adds
/// little here and slows its twin (FR+5: 1.24x the serial core-seconds on 12 threads, 2.0x
/// on 23, and a longer wall time).
fn replay_lanes(rt: &Runtime) -> usize {
    use blobmap_model::runtime::Fam;
    let largest = [Fam::Car, Fam::Active]
        .iter()
        .filter(|f| rt.tables.contains_key(f))
        .flat_map(|&f| (0..rt.components(f) as u32).map(move |x| rt.tables[&f].nodes(x) as usize))
        .max()
        .unwrap_or(0);
    let per = (4 * 2 + 4) * rt.vocab.n() + largest * 2 + 2 * (largest * 2 + largest.div_ceil(64) * 8);
    let free = crate::build::mem::free().unwrap_or(0) as usize / 2;
    (free / per.max(1)).max(1)
}
