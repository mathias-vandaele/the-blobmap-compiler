//! The checks of SPEC §11 that compare a model with the reference: experts, the snapping
//! re-check, re-customization, distances against Dijkstra, and the scenario's own facts.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use anyhow::Result;
use blobmap_fixtures::Check;
use blobmap_format::{INF, NONE};
use blobmap_runtime::query::Source;
use blobmap_runtime::recustomize::{RoadRef, customize};
use blobmap_runtime::{Dir, Expert, Fam, Runtime};
use blobmap_snap::parts;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde_json::{Value, json};

use crate::expected::{Expected, reason_name, uuid};
use crate::Samples;
use crate::reference::Reference;
use crate::report::Report;

pub fn experts(rt: &Runtime, r: &Reference, rep: &mut Report) {
    let o = rep.check("experts against the reference (SPEC §5)");
    for (fam, e) in [(Fam::Car, &r.car_experts), (Fam::Active, &r.active_experts)] {
        let t = &rt.tables[&fam];
        o.count();
        o.expect(t.nodes == e.nodes, || format!("{}: expert sizes {:?} != {:?}", fam.name(), t.nodes, e.nodes));
        let tr: Vec<bool> = t.trapped.iter().map(|&x| x == 1).collect();
        o.expect(tr == e.trapped, || format!("{}: trapped flags differ", fam.name()));
    }
}

/// Expected stored entry of an ID: per slot the road key (way, seq, dir), and the offsets
/// in the runtime's common shape.
fn expected_entry(r: &Reference, m: usize, piece: usize, frac: f64) -> ([Option<(i64, u32, u8)>; 2], [u16; 4]) {
    let mode = &r.modes[m];
    if mode.car {
        let g = &r.car;
        let e = r.modes[m].at.iter().flatten().next().map(|_| ()).is_some();
        let _ = e;
        let x = blobmap_snap::modes::car_expert(g, &r.car_experts, piece).unwrap();
        let mut keys = [None, None];
        let mut off = [INF; 4];
        for d in 0..2 {
            let s = g.seg_of_piece[piece][d];
            if s == NONE || r.car_experts.of[s as usize] != x {
                continue;
            }
            keys[d] = Some((g.way[s as usize], g.seq[s as usize], g.dir[s as usize]));
            let (a, b) = parts(g.w[s as usize], if d == 0 { frac } else { 1.0 - frac });
            off[2 * d] = a;
            off[2 * d + 1] = b;
        }
        (keys, off)
    } else {
        let g = &r.active;
        let [along, against] = g.road_of_piece[piece];
        let (fu, ruv, rvu) = if along != NONE { (frac, along, against) } else { (1.0 - frac, against, along) };
        let mi = mode.metric;
        let (u_id, id_v) = parts(g.w[mi][ruv as usize], fu);
        let (v_id, id_u) = if rvu != NONE { parts(g.w[mi][rvu as usize], 1.0 - fu) } else { (INF, INF) };
        let key = |k: u32| (g.way[k as usize], g.seq[k as usize], g.dir[k as usize]);
        ([Some(key(ruv)), (rvu != NONE).then(|| key(rvu))], [u_id, id_u, v_id, id_v])
    }
}

/// Re-snapping from the inputs reproduces the stored entries exactly (SPEC §11).
/// The reference's grid snapping against a search of every segment (SPEC §6), on every ID
/// up to `max` per mode, evenly sampled beyond.
pub fn snapping_brute(r: &Reference, max: usize, rep: &mut Report) {
    let o = rep.check("snapping without the grid");
    let step = r.n().div_ceil(max.max(1)).max(1);
    for (m, mode) in r.modes.iter().enumerate() {
        let p = if mode.car { &r.car_pieces } else { &r.active_pieces };
        let picked: Vec<usize> = (0..r.n()).step_by(step).filter(|&i| !matches!(mode.at[i], Err((4, _)))).collect();
        let results: Vec<Option<String>> = picked
            .par_iter()
            .map(|&i| {
                let (lat, lon) = r.points[i];
                let want = blobmap_snap::find_brute(p, lat, lon, r.radius[m], &r.sets[m]).outcome();
                let got = mode.at[i].map(|e| (e.piece, e.frac.to_bits())).map_err(|(c, d)| (c, d.to_bits()));
                let want = want.map(|h| (h.piece, h.frac.to_bits())).map_err(|(c, d)| (c, d.to_bits()));
                (got != want).then(|| format!("{} {}: grid {got:?}, every segment {want:?}", mode.name, uuid(&r.ids[i])))
            })
            .collect();
        for f in results {
            o.count();
            if let Some(f) = f {
                o.fail(f);
            }
        }
    }
}

pub fn snapping(rt: &Runtime, r: &Reference, rep: &mut Report) -> Result<()> {
    let o = rep.check("snapping re-check");
    let v = &rt.vocab;
    let ids: Vec<[u8; 16]> = v.address.iter().chain(&v.place).copied().collect();
    o.expect(ids == r.ids, || format!("ID tables differ: {} stored, {} expected", ids.len(), r.ids.len()));
    let mut al: Vec<([u8; 16], u32)> = v.alias.iter().copied().zip(v.alias_to.iter().copied()).collect();
    al.sort();
    let mut ral = r.aliases.clone();
    ral.sort();
    o.expect(al == ral, || "aliases differ".into());
    if ids != r.ids {
        return Ok(());
    }
    for (m, mode) in r.modes.iter().enumerate() {
        let k = rt.mode(mode.name)?;
        let mv = &v.modes[k];
        for i in 0..r.n() {
            o.count();
            let stored_reason = mv.reasons.binary_search_by_key(&(i as u32), |x| x.0).ok().map(|j| (mv.reasons[j].1, mv.reasons[j].2));
            match mode.at[i] {
                Err((4, _)) => o.expect(mv.expert[i] == NONE && v.status_of(i as u32) == 4, || format!("{} {}: not a filtered place", mode.name, uuid(&r.ids[i]))),
                Err((c, d)) => o.expect(
                    mv.expert[i] == NONE && stored_reason.is_some_and(|(sc, sd)| sc == c && sd.to_bits() == d.to_bits()),
                    || format!("{} {}: reason {stored_reason:?}, expected ({c}, {d})", mode.name, uuid(&r.ids[i])),
                ),
                Ok(e) => {
                    if mv.expert[i] != e.expert {
                        o.fail(format!("{} {}: expert {} != {}", mode.name, uuid(&r.ids[i]), mv.expert[i], e.expert));
                        continue;
                    }
                    let x = rt.expert(mv.info.fam, e.expert)?;
                    let (keys, off) = expected_entry(r, m, e.piece as usize, e.frac);
                    let stored: [Option<(i64, u32, u8)>; 2] = if mode.car {
                        [0, 1].map(|d| {
                            let s = mv.node[i][d];
                            (s != NONE).then(|| (x.way[s as usize], x.way_seq[s as usize], x.way_dir[s as usize]))
                        })
                    } else {
                        let [uu, vv] = mv.node[i];
                        let uv = x.arcs(uu).find(|&a| x.arc_local[a] == mv.road[i] && x.arc_head[a] == vv);
                        let key = |a: usize| (x.way[a], x.way_seq[a], x.way_dir[a]);
                        let vu = uv.and_then(|uv| x.arcs(vv).find(|&a| x.arc_head[a] == uu && key(a).0 == key(uv).0 && key(a).1 == key(uv).1 && key(a).2 != key(uv).2));
                        [uv.map(key), vu.map(key)]
                    };
                    o.expect(stored == keys && mv.off[i] == off, || {
                        format!("{} {}: entry {stored:?} {:?}, expected {keys:?} {off:?}", mode.name, uuid(&r.ids[i]), mv.off[i])
                    });
                }
            }
        }
    }
    Ok(())
}

/// The CCH sweep of FORMAT §8 with given link weights, from one node.
fn sweep(e: &Expert, up: &[u16], down: &[u16], src: u32) -> Vec<u16> {
    let mut l = vec![INF; e.n];
    l[src as usize] = 0;
    for v in (0..e.n as u32).rev() {
        if l[v as usize] == INF {
            continue;
        }
        for k in e.row(v) {
            let u = e.up_head[k] as usize;
            l[u] = l[u].min(l[v as usize].saturating_add(up[k]));
        }
    }
    for v in 0..e.n as u32 {
        for k in e.row(v) {
            l[v as usize] = l[v as usize].min(l[e.up_head[k] as usize].saturating_add(down[k]));
        }
    }
    l
}

/// Dijkstra over an expert's own arcs with road weights w (CAR start view).
fn dijkstra(e: &Expert, w: &[u16], src: u32) -> Vec<u16> {
    let mut d = vec![u32::MAX; e.n];
    d[src as usize] = 0;
    let mut h = BinaryHeap::from([Reverse((0u32, src))]);
    while let Some(Reverse((t, v))) = h.pop() {
        if t > d[v as usize] {
            continue;
        }
        for a in e.arcs(v) {
            let x = match e.fam {
                Fam::Car if w[v as usize] == INF => continue,
                Fam::Car => t + w[v as usize] as u32 + e.turn_cost[a] as u32,
                Fam::Active if w[a] == INF => continue,
                Fam::Active => t + w[a] as u32,
            };
            let hd = e.arc_head[a] as usize;
            if x < d[hd] {
                d[hd] = x;
                h.push(Reverse((x, hd as u32)));
            }
        }
    }
    d.into_iter().map(|x| x.min(INF as u32) as u16).collect()
}

/// Re-customization (FORMAT §6): the stored master times reproduce the stored weights and
/// links; random master times give the same distances as Dijkstra (SPEC §11).
pub fn recustomization(rt: &Runtime, rep: &mut Report) -> Result<()> {
    for fam in [Fam::Car, Fam::Active] {
        let o = rep.check(&format!("{}: re-customization from the bytes", fam.name()));
        let stored = rt.reround(fam, &|r, m| rt.stored_time(fam, r, m))?;
        for x in 0..rt.experts(fam) as u32 {
            let e = rt.expert(fam, x)?;
            for m in 0..e.up.len() {
                o.count();
                let w = &stored.w[m][x as usize];
                o.expect(*w == e.w[m], || format!("expert {x} metric {m}: re-rounded weights differ"));
                let (up, down) = customize(&e, w)?;
                o.expect(up == e.up[m] && down == e.down[m], || format!("expert {x} metric {m}: re-customized links differ"));
            }
        }
        if fam == Fam::Active {
            let cw = rt.model.get::<u16>("road/active/cross/walk.w")?;
            o.expect(stored.cross[1] == *cw, || "cross roads: re-rounded walk weights differ".into());
        }
        let o = rep.check(&format!("{}: random metrics against Dijkstra", fam.name()));
        let random = |r: RoadRef, m: usize| -> f32 {
            let seed = ((r.expert as u64) << 40) ^ ((r.index as u64) << 8) ^ m as u64;
            let mut g = rand::rngs::StdRng::seed_from_u64(seed);
            if g.random_bool(0.05) { f32::INFINITY } else { g.random_range(0.0f32..400.0) }
        };
        let rw = rt.reround(fam, &random)?;
        for x in 0..rt.experts(fam) as u32 {
            let e = rt.expert(fam, x)?;
            for m in 0..e.up.len() {
                let w = &rw.w[m][x as usize];
                let (up, down) = customize(&e, w)?;
                for src in (0..e.n as u32).step_by((e.n / 8).max(1)) {
                    o.count();
                    let a = sweep(&e, &up, &down, src);
                    let b = dijkstra(&e, w, src);
                    o.expect(a == b, || format!("expert {x} metric {m} from {src}: CCH and Dijkstra differ"));
                }
            }
        }
    }
    Ok(())
}

/// Columns from and to `n.columns` sampled sources, multi-source columns with start times,
/// and point-to-point answers with reasons between `n.pairs` sampled IDs, against the
/// reference Dijkstra. The reference columns are computed in parallel.
pub fn distances(rt: &Runtime, ex: &Expected, n: &Samples, rep: &mut Report) -> Result<()> {
    let r = ex.r;
    for (m, mode) in r.modes.iter().enumerate() {
        let k = rt.mode(mode.name)?;
        let o = rep.check(&format!("{}: columns against Dijkstra (from, to, starts)", mode.name));
        let served: Vec<usize> = (0..r.n()).filter(|&i| mode.at[i].is_ok()).collect();
        let step = served.len().div_ceil(n.columns).max(1);
        let mut cases: Vec<(Vec<(usize, u16)>, Dir, &str)> = Vec::new();
        for &s in served.iter().step_by(step) {
            cases.push((vec![(s, 0)], Dir::From, "from"));
            cases.push((vec![(s, 0)], Dir::To, "to"));
        }
        for w in served.chunks(3).step_by(step.max(1)).take(6) {
            let src: Vec<(usize, u16)> = w.iter().enumerate().map(|(j, &s)| (s, [0u16, 45, 700][j])).collect();
            cases.push((src.clone(), Dir::From, "from"));
            cases.push((src, Dir::To, "to"));
        }
        let results: Vec<Result<Option<String>>> = cases
            .par_iter()
            .map(|(src, dir, name)| {
                let got = rt.column(&src.iter().map(|&(s, st)| Source { id: s as u32, start: st }).collect::<Vec<_>>(), k, *dir)?;
                let want = ex.column(m, src, name);
                Ok((got != *want).then(|| {
                    let i = (0..got.len()).find(|&i| got[i] != want[i]).unwrap();
                    let from: Vec<String> = src.iter().map(|x| format!("{}+{}", uuid(&r.ids[x.0]), x.1)).collect();
                    format!("{name} {from:?} to {}: {} != {}", uuid(&r.ids[i]), got[i], want[i])
                }))
            })
            .collect();
        for res in results {
            o.count();
            if let Some(f) = res? {
                o.fail(f);
            }
        }
        let o = rep.check(&format!("{}: travel_time and reasons", mode.name));
        let sample: Vec<usize> = (0..r.n()).step_by(r.n().div_ceil(n.pairs).max(1)).collect();
        let results: Vec<Result<Vec<Option<String>>>> = sample
            .par_iter()
            .map(|&a| {
                sample
                    .iter()
                    .map(|&b| {
                        let (sa, sb) = (uuid(&r.ids[a]), uuid(&r.ids[b]));
                        let got = rt.travel_time(&sa, &sb, k)?;
                        let want = ex.travel_time(&sa, &sb, mode.name);
                        Ok((got != want).then(|| format!("{sa} -> {sb}: {got} != {want}")))
                    })
                    .collect()
            })
            .collect();
        for res in results {
            for f in res? {
                o.count();
                if let Some(f) = f {
                    o.fail(f);
                }
            }
        }
    }
    Ok(())
}

fn tt(rt: &Runtime, a: &str, b: &str, mode: &str) -> Value {
    rt.mode(mode).and_then(|m| rt.travel_time(a, b, m)).unwrap_or_else(|e| json!({"error": e.to_string()}))
}

/// The scenario's own facts (fixtures `Check`).
pub fn scenario(rt: &Runtime, ex: &Expected, checks: &[Check], cells_variant: bool, rep: &mut Report) -> Result<()> {
    let o = rep.check("scenario facts");
    let reason = |id: &str, mode: &str| -> Option<String> {
        let i = rt.resolve(id).ok()?;
        rt.served(i, rt.mode(mode).ok()?).err().map(|r| r.name().to_string())
    };
    for c in checks {
        o.count();
        match c {
            Check::Hierarchy { family, expert, links, levels } => {
                let fam = if *family == "car" { Fam::Car } else { Fam::Active };
                let e = rt.expert(fam, *expert)?;
                o.expect(e.up_head.len() as u64 == *links && rt.tables[&fam].levels[*expert as usize] == *levels, || {
                    format!("{family} expert {expert}: {} links, {} levels", e.up_head.len(), rt.tables[&fam].levels[*expert as usize])
                });
            }
            Check::Grid { mode, ids, times } => {
                for (i, a) in ids.iter().enumerate() {
                    for (j, b) in ids.iter().enumerate() {
                        let got = tt(rt, a, b, mode);
                        o.expect(got == json!({"time": times[i][j]}), || format!("grid {i}->{j}: {got}, expected {}", times[i][j]));
                    }
                }
            }
            Check::Cap { mode, a, b, under } => {
                let got = tt(rt, a, b, mode);
                let want = ex.travel_time(a, b, mode);
                let ok = got == want && (got["time"].is_u64() == *under);
                o.expect(ok, || format!("cap {a}->{b}: {got}, expected {} {want}", if *under { "under" } else { "BEYOND" }));
            }
            Check::Reason { id, mode, reason: want } => {
                let got = reason(id, mode);
                o.expect(got.as_deref() == Some(*want), || format!("{id} {mode}: {got:?}, expected {want}"));
            }
            Check::Served { id, mode } => o.expect(reason(id, mode).is_none() && rt.resolve(id).is_ok(), || format!("{id} not served for {mode}")),
            Check::Alias { alias, kept } => {
                o.expect(rt.resolve(alias).is_ok() && rt.resolve(alias) == rt.resolve(kept), || format!("alias {alias} does not resolve to {kept}"));
                o.expect(tt(rt, alias, kept, "walk") == json!({"time": 0}), || "alias and kept place are not the same ID".into());
            }
            Check::Unknown { id } => o.expect(rt.resolve(id) == Err(blobmap_runtime::Reason::UnknownId), || format!("{id} is known")),
            Check::DifferentComponent { mode, a, b } => {
                let got = tt(rt, a, b, mode);
                o.expect(got["reason"] == "DifferentComponent", || format!("{a}->{b} {mode}: {got}"));
            }
            Check::Beyond { mode, a, b } => {
                let got = tt(rt, a, b, mode);
                o.expect(got == json!({"time": "BEYOND"}), || format!("{a}->{b} {mode}: {got}, expected BEYOND"));
            }
            Check::Reachable { mode, a, b } => {
                let got = tt(rt, a, b, mode);
                o.expect(got["time"].is_u64(), || format!("{a}->{b} {mode}: {got}"));
            }
            Check::Tie { category, id, mode, place } => {
                let m = rt.mode(mode)?;
                let got = rt.nearest(rt.category(&json!({"category": category}))?, id, m, Dir::From)?;
                o.expect(got["place"] == place.as_str(), || format!("tie {category} from {id}: {got}, expected {place}"));
                let want = ex.answer("nearest", &json!({"category": {"category": category}, "id": id, "mode": mode, "dir": "from"}));
                o.expect(got == want, || format!("tie {category}: {got} != {want}"));
            }
            Check::Matches { category, place } => {
                let c = rt.category(&json!({"category": category}))?;
                let p = rt.resolve(place).map(|i| i as usize - rt.vocab.address.len());
                o.expect(p.is_ok_and(|p| rt.matches(p, c)), || format!("{place} does not match {category}"));
            }
            Check::SamePiece { mode, a, b } => {
                let m = rt.mode(mode)?;
                let (ia, ib) = (rt.resolve(a), rt.resolve(b));
                let same = match (ia, ib) {
                    (Ok(x), Ok(y)) => blobmap_runtime::query::same_piece(&rt.vocab.modes[m], x as usize, y as usize),
                    _ => false,
                };
                o.expect(same, || format!("{a} and {b} are not on one piece for {mode}"));
                for (x, y) in [(a, b), (b, a)] {
                    let got = tt(rt, x, y, mode);
                    let want = ex.travel_time(x, y, mode);
                    o.expect(got == want, || format!("same piece {x}->{y} {mode}: {got} != {want}"));
                }
            }
            Check::AtNode { id, mode } => {
                let m = rt.mode(mode)?;
                let ok = rt.resolve(id).is_ok_and(|i| {
                    let off = rt.vocab.modes[m].off[i as usize];
                    (off[0] == 0 && off[1] == 0) || (off[2] == 0 && off[3] == 0)
                });
                o.expect(ok, || format!("{id} does not stand at a node for {mode}"));
            }
            Check::Trapped { family, at_least } => {
                let fam = if *family == "car" { Fam::Car } else { Fam::Active };
                let n = rt.tables[&fam].trapped.iter().filter(|&&t| t == 1).count();
                o.expect(n >= *at_least, || format!("{family}: {n} trapped, expected at least {at_least}"));
            }
            Check::Restrictions { at_least } => {
                let n = ex.r.car.stats.way_via_applied;
                o.expect(n >= *at_least, || format!("{n} via-way restrictions applied"));
            }
            Check::OnVia { id } => {
                let m = rt.mode("car")?;
                let ok = rt.resolve(id).is_ok_and(|i| {
                    let mv = &rt.vocab.modes[m];
                    let x = mv.expert[i as usize];
                    x != NONE
                        && rt.expert(Fam::Car, x).is_ok_and(|e| {
                            mv.node[i as usize].iter().any(|&s| s != NONE && e.copy_first[s as usize + 1] > e.copy_first[s as usize])
                        })
                });
                o.expect(ok, || format!("{id} is not on a segment with copies"));
                for other in [&ex.r.ids[0]] {
                    let b = uuid(other);
                    for (x, y) in [(id.as_str(), b.as_str()), (b.as_str(), id.as_str())] {
                        let got = tt(rt, x, y, "car");
                        let want = ex.travel_time(x, y, "car");
                        o.expect(got == want, || format!("on via {x}->{y}: {got} != {want}"));
                    }
                }
            }
            Check::Cells { family } => {
                if cells_variant {
                    let fam = if *family == "car" { Fam::Car } else { Fam::Active };
                    let e = rt.expert(fam, 0)?;
                    o.expect(e.blocks.iter().filter(|b| b.kind == 2).count() >= 2, || format!("{family} expert 0 is not built as cells"));
                }
            }
        }
    }
    let _ = reason_name;
    Ok(())
}
