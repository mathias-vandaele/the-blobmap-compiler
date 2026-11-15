//! The checks of SPEC §11 that compare a model with the reference: components, the snapping
//! re-check, re-customization, distances against Dijkstra, and the scenario's own facts.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap};

use anyhow::Result;
use blobmap_model::runtime::query::{Source, labels_at};
use blobmap_model::runtime::recustomize::{RoadRef, customize};
use blobmap_model::runtime::{Dir, Component, Fam, Runtime};
use blobmap_model::{INF, NONE, NONE64};
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde_json::{Value, json};

use crate::fixtures::Check;
use crate::snap::{FILTERED_PLACE, parts};
use crate::verify::Samples;
use crate::verify::expected::{Expected, reason_name, uuid};
use crate::verify::reference::Reference;
use crate::verify::report::{Outcome, Report};

pub fn components(rt: &Runtime, r: &Reference, rep: &mut Report) {
    let o = rep.check("components against the reference (SPEC §5)");
    for (fam, e) in [(Fam::Car, &r.car_components), (Fam::Active, &r.active_components)].into_iter().filter(|f| r.has(f.0)) {
        let t = &rt.tables[&fam];
        o.count();
        let nodes: Vec<u32> = (0..t.components() as u32).map(|x| t.nodes(x)).collect();
        o.expect(nodes == e.nodes, || format!("{}: component sizes {nodes:?} != {:?}", fam.name(), e.nodes));
    }
}

/// The antimeridian's twins (SPEC §4): in each family held, no twin but the smallest of its
/// group is a routing node; the twin nodes that join roads from both sides are counted.
pub fn twins(r: &Reference, rep: &mut Report) {
    for (fam, p) in [(Fam::Car, &r.car_pieces), (Fam::Active, &r.active_pieces)].into_iter().filter(|f| r.has(f.0)) {
        let ids = &p.node_osm[..p.osm_nodes];
        let mut wrong = Vec::new();
        let mut joined: BTreeMap<u32, [bool; 2]> = BTreeMap::new();
        for g in &r.twins {
            let routing: Vec<i64> = g.iter().copied().filter(|id| ids.binary_search(id).is_ok()).collect();
            if routing.iter().any(|&id| id != g[0]) {
                wrong.push(format!("twins {g:?} route as {routing:?}, not as {} alone", g[0]));
            }
            if let Ok(k) = ids.binary_search(&g[0]) {
                joined.insert(k as u32, [false; 2]);
            }
        }
        // The side of the meridian each piece at a twin node leaves towards.
        for k in 0..p.len() {
            let pts = p.points(k);
            for (end, next) in [(p.u[k], pts.start + 1), (p.v[k], pts.end - 2)] {
                if let Some(side) = joined.get_mut(&end) {
                    side[(p.pt_lon[next] > 0.0) as usize] = true;
                }
            }
        }
        let both = joined.values().filter(|s| s[0] && s[1]).count();
        let o = rep.check(&format!(
            "{}: antimeridian twins, one routing node per group (SPEC §4; {} groups in the .pbf, {both} of the {} routing twin nodes join roads from both sides)",
            fam.name(),
            r.twins.len(),
            joined.len()
        ));
        for _ in &r.twins {
            o.count();
        }
        for w in wrong {
            o.fail(w);
        }
    }
}

/// Expected stored entry of an ID: per slot the road key (way, seq, dir), and the offsets
/// in the runtime's common shape.
fn expected_entry(r: &Reference, m: usize, piece: usize, frac: f64) -> ([Option<(i64, u32, u8)>; 2], [u16; 4]) {
    let mode = &r.modes[m];
    if mode.car {
        let g = &r.car;
        let x = crate::snap::modes::car_component(g, &r.car_components, piece).unwrap();
        let mut keys = [None, None];
        let mut off = [INF; 4];
        for d in 0..2 {
            let s = g.seg_of_piece[piece][d];
            if s == NONE || r.car_components.of[s as usize] != x {
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
        let (fu, ruv, rvu) = if along != NONE64 { (frac, along, against) } else { (1.0 - frac, against, along) };
        let mi = mode.metric;
        let (u_id, id_v) = parts(g.w[mi][ruv as usize], fu);
        let (v_id, id_u) = if rvu != NONE64 { parts(g.w[mi][rvu as usize], 1.0 - fu) } else { (INF, INF) };
        let key = |k: u64| (g.way[k as usize], g.seq[k as usize], g.dir[k as usize]);
        ([Some(key(ruv)), (rvu != NONE64).then(|| key(rvu))], [u_id, id_u, v_id, id_v])
    }
}

/// Re-snapping from the inputs reproduces the stored entries exactly (SPEC §11).
/// The reference's grid snapping against a search of every segment (SPEC §6), on every ID
/// up to `max` per mode, evenly sampled beyond.
pub fn snapping_brute(r: &Reference, max: usize, rep: &mut Report) {
    let o = rep.check("snapping without the grid");
    let step = r.n().div_ceil(max.max(1)).max(1);
    for mode in &r.modes {
        let p = if mode.car { &r.car_pieces } else { &r.active_pieces };
        let picked: Vec<usize> = (0..r.n()).step_by(step).filter(|&i| !matches!(mode.at(i), Err((FILTERED_PLACE, _)))).collect();
        let results: Vec<Option<String>> = picked
            .par_iter()
            .map(|&i| {
                let [lat, lon] = r.points[i];
                let want = crate::snap::find_brute(p, lat, lon, mode.radius, mode.small_max, &mode.sets).outcome();
                let got = mode.at(i).map(|e| (e.piece, e.frac.to_bits())).map_err(|(c, d)| (c, d.to_bits()));
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
    let same_ids = v.address.len() + v.place.len() == r.ids.len() && v.address.iter().chain(&v.place).eq(r.ids.iter());
    o.expect(same_ids, || format!("ID tables differ: {} stored, {} expected", v.address.len() + v.place.len(), r.ids.len()));
    let mut al: Vec<([u8; 16], u32)> = v.alias.iter().copied().zip(v.alias_to.iter().copied()).collect();
    al.sort();
    let mut ral = r.aliases.clone();
    ral.sort();
    o.expect(al == ral, || "aliases differ".into());
    if !same_ids {
        return Ok(());
    }
    for mode in &r.modes {
        let k = rt.mode(mode.name)?;
        let mv = &v.modes[k];
        for i in 0..r.n() {
            o.count();
            let stored_reason = mv.reasons.binary_search_by_key(&(i as u32), |x| x.0).ok().map(|j| (mv.reasons[j].1, mv.reasons[j].2));
            match mode.at(i) {
                Err((FILTERED_PLACE, _)) => o.expect(mv.component[i] == NONE && v.status_of(i as u32) == 4, || {
                    format!("{} {}: not a filtered place", mode.name, uuid(&r.ids[i]))
                }),
                Err((c, d)) => o
                    .expect(mv.component[i] == NONE && stored_reason.is_some_and(|(sc, sd)| sc == c && sd.to_bits() == d.to_bits()), || {
                        format!("{} {}: reason {stored_reason:?}, expected ({c}, {d})", mode.name, uuid(&r.ids[i]))
                    }),
                Ok(e) => {
                    if mv.component[i] != e.component {
                        o.fail(format!("{} {}: component {} != {}", mode.name, uuid(&r.ids[i]), mv.component[i], e.component));
                    }
                }
            }
        }
    }
    // Served IDs whose component matches, checked one component at a time from the reference
    // split per component: by family, component, mode and ID.
    let mut fams: Vec<Fam> = Vec::new();
    for mode in &r.modes {
        let f = v.modes[rt.mode(mode.name)?].info.fam;
        if !fams.contains(&f) {
            fams.push(f);
        }
    }
    fams.sort_unstable();
    for fam in fams {
        let modes: Vec<usize> = (0..r.modes.len()).filter(|&m| rt.mode(r.modes[m].name).is_ok_and(|k| v.modes[k].info.fam == fam)).collect();
        for c in 0..r.modes[modes[0]].components() {
            if modes.iter().all(|&m| r.modes[m].in_component(c).is_empty()) {
                continue;
            }
            let x = rt.component(fam, c)?;
            for (m, i) in modes.iter().flat_map(|&m| r.modes[m].in_component(c).iter().map(move |&i| (m, i))) {
                let (mode, i) = (&r.modes[m], i as usize);
                let mv = &v.modes[rt.mode(mode.name)?];
                if mv.component[i] != c {
                    continue;
                }
                let Ok(e) = mode.at(i) else { unreachable!("a served ID") };
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
                    let vu = uv.and_then(|uv| {
                        x.arcs(vv)
                            .find(|&a| x.arc_head[a] == uu && key(a).0 == key(uv).0 && key(a).1 == key(uv).1 && key(a).2 != key(uv).2)
                    });
                    [uv.map(key), vu.map(key)]
                };
                o.expect(stored == keys && mv.off[i] == off, || {
                    format!("{} {}: entry {stored:?} {:?}, expected {keys:?} {off:?}", mode.name, uuid(&r.ids[i]), mv.off[i])
                });
            }
            drop(x);
            rt.release();
        }
    }
    Ok(())
}

/// The CCH sweep of FORMAT §8 with given link weights, from one node.
fn sweep(e: &Component, up: &[u16], down: &[u16], src: u32) -> Vec<u16> {
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

/// Dijkstra over a component's own arcs with road weights w (CAR start view).
fn dijkstra(e: &Component, w: &[u16], src: u32) -> Vec<u16> {
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

/// A component of at least this many nodes is checked alone, in parallel inside; smaller ones
/// are checked in parallel with each other.
const LARGE_COMPONENT: u32 = 1 << 20;

/// Small components checked between two releases of the runtime's component cache.
const SMALL_AT_ONCE: usize = 4096;

/// Runs `f` on every component of the family and absorbs its outcomes in component order.
/// `f` gets whether it is checking a large component alone, and may then work in parallel.
fn each_component(rt: &Runtime, fam: Fam, o: &mut Outcome, f: impl Fn(u32, &Component, bool) -> Result<Outcome> + Sync) -> Result<()> {
    let t = &rt.tables[&fam];
    let n = rt.components(fam) as u32;
    // Components are numbered by node count descending (FORMAT §5.1).
    let large = (0..n).take_while(|&x| t.nodes(x) >= LARGE_COMPONENT).count() as u32;
    for x in 0..large {
        let e = rt.component(fam, x)?;
        o.absorb(f(x, &e, true)?);
        drop(e);
        rt.release();
    }
    let small: Vec<u32> = (large..n).collect();
    for chunk in small.chunks(SMALL_AT_ONCE) {
        let parts: Vec<Result<Outcome>> = chunk.par_iter().map(|&x| f(x, &*rt.component(fam, x)?, false)).collect();
        for part in parts {
            o.absorb(part?);
        }
        rt.release();
    }
    Ok(())
}

/// Re-customization (FORMAT §6): the stored master times reproduce the stored weights and
/// links; random master times give the same distances as Dijkstra (SPEC §11).
pub fn recustomization(rt: &Runtime, rep: &mut Report) -> Result<()> {
    for fam in [Fam::Car, Fam::Active] {
        let cross = rt.stored_cross_times(fam)?;
        let stored = rt.reround(fam, &|e, r, m| match e {
            Some(e) => e.time_s[m][r.index as usize],
            None => cross[m][r.index as usize],
        })?;
        let o = rep.check(&format!("{}: re-customization from the bytes", fam.name()));
        each_component(rt, fam, o, |x, e, alone| {
            let metric = |m: usize| -> Result<Outcome> {
                let mut o = Outcome::default();
                o.count();
                let w = &stored.w[m][x as usize];
                o.expect(w[..] == e.w[m][..], || format!("component {x} metric {m}: re-rounded weights differ"));
                let (up, down) = customize(e, w)?;
                o.expect(up[..] == e.up[m][..] && down[..] == e.down[m][..], || format!("component {x} metric {m}: re-customized links differ"));
                Ok(o)
            };
            in_order(e.up.len(), alone, metric)
        })?;
        if fam == Fam::Active {
            let cw = rt.model.get::<u16>("road/active/cross/walk.w")?;
            let o = rep.check(&format!("{}: re-customization from the bytes", fam.name()));
            o.expect(stored.cross[1] == *cw, || "cross roads: re-rounded walk weights differ".into());
        }
        drop(stored);
        let random = |_: Option<&Component>, r: RoadRef, m: usize| -> f32 {
            let seed = ((r.component as u64) << 40) ^ (r.index << 8) ^ m as u64;
            let mut g = rand::rngs::StdRng::seed_from_u64(seed);
            if g.random_bool(0.05) { f32::INFINITY } else { g.random_range(0.0f32..400.0) }
        };
        let rw = rt.reround(fam, &random)?;
        let o = rep.check(&format!("{}: random metrics against Dijkstra", fam.name()));
        each_component(rt, fam, o, |x, e, alone| {
            let metric = |m: usize| -> Result<Outcome> {
                let w = &rw.w[m][x as usize];
                let (up, down) = customize(e, w)?;
                let srcs: Vec<u32> = (0..e.n as u32).step_by((e.n / 8).max(1)).collect();
                in_order(srcs.len(), alone, |k| {
                    let src = srcs[k];
                    let mut o = Outcome::default();
                    o.count();
                    let a = sweep(e, &up, &down, src);
                    let b = dijkstra(e, w, src);
                    o.expect(a == b, || format!("component {x} metric {m} from {src}: CCH and Dijkstra differ"));
                    Ok(o)
                })
            };
            // A metric at a time: its eight searches at once already fill the threads.
            in_order(e.up.len(), false, metric)
        })?;
    }
    Ok(())
}

/// The outcomes of `f(0 .. n)`, in parallel when asked, absorbed in order.
fn in_order(n: usize, parallel: bool, f: impl Fn(usize) -> Result<Outcome> + Sync) -> Result<Outcome> {
    let parts: Vec<Result<Outcome>> = if parallel { (0..n).into_par_iter().map(&f).collect() } else { (0..n).map(&f).collect() };
    let mut o = Outcome::default();
    for part in parts {
        o.absorb(part?);
    }
    Ok(o)
}

/// Columns from and to `n.columns` sampled sources, multi-source columns with start times,
/// and point-to-point answers with reasons between `n.pairs` sampled IDs, against the
/// reference Dijkstra. The reference columns are computed in parallel.
pub fn distances(rt: &Runtime, ex: &Expected, n: &Samples, rep: &mut Report) -> Result<()> {
    let r = ex.r;
    for (m, mode) in r.modes.iter().enumerate() {
        let k = rt.mode(mode.name)?;
        let o = rep.check(&format!("{}: columns against Dijkstra (from, to, starts)", mode.name));
        let (n_served, step) = (mode.served(), mode.served().div_ceil(n.columns).max(1));
        // The served IDs at the positions sampled below (every step-th, and three in a row at
        // six of every third step), without listing every served ID.
        let wanted = |k: usize| k.is_multiple_of(step) || (k % (3 * step) < 3 && k / (3 * step) < 6);
        let at_pos: BTreeMap<usize, usize> = (0..r.n()).filter(|&i| mode.at(i).is_ok()).enumerate().filter(|&(k, _)| wanted(k)).collect();
        let mut cases: Vec<(Vec<(usize, u16)>, Dir, &str)> = Vec::new();
        for k in (0..n_served).step_by(step) {
            cases.push((vec![(at_pos[&k], 0)], Dir::From, "from"));
            cases.push((vec![(at_pos[&k], 0)], Dir::To, "to"));
        }
        for base in (0..n_served).step_by(3 * step).take(6) {
            let src: Vec<(usize, u16)> = (base..(base + 3).min(n_served)).enumerate().map(|(j, k)| (at_pos[&k], [0u16, 45, 700][j])).collect();
            cases.push((src.clone(), Dir::From, "from"));
            cases.push((src, Dir::To, "to"));
        }
        let results: Vec<Result<Option<String>>> = crate::verify::columns_at_once(rt, k, &cases, |(src, dir, name)| {
            let got = rt.column(&src.iter().map(|&(s, st)| Source { id: s as u32, start: st }).collect::<Vec<_>>(), k, *dir)?;
            let want = ex.column(m, src, name);
            Ok((got != *want).then(|| {
                let i = (0..got.len()).find(|&i| got[i] != want[i]).unwrap();
                let from: Vec<String> = src.iter().map(|x| format!("{}+{}", uuid(&r.ids[x.0]), x.1)).collect();
                format!("{name} {from:?} to {}: {} != {}", uuid(&r.ids[i]), got[i], want[i])
            }))
        });
        for res in results {
            o.count();
            if let Some(f) = res? {
                o.fail(f);
            }
        }
        let o = rep.check(&format!("{}: travel_time and reasons", mode.name));
        let sample: Vec<usize> = (0..r.n()).step_by(r.n().div_ceil(n.pairs).max(1)).collect();
        let results: Vec<Result<Vec<Option<String>>>> = crate::verify::columns_at_once(rt, k, &sample, |&a| {
            sample
                .iter()
                .map(|&b| {
                    let (sa, sb) = (uuid(&r.ids[a]), uuid(&r.ids[b]));
                    let got = rt.travel_time(&sa, &sb, k)?;
                    let want = ex.travel_time(&sa, &sb, mode.name);
                    Ok((got != want).then(|| format!("{sa} -> {sb}: {got} != {want}")))
                })
                .collect()
        });
        for res in results {
            for f in res? {
                o.count();
                if let Some(f) = f {
                    o.fail(f);
                }
            }
        }
        rt.release();
    }
    Ok(())
}

/// The proper prefixes (length ≥ 2) of the forbidden sequences, sorted, and how many there
/// are: all of them, or `at_most` drawn with a fixed seed (SPEC §11, world tier), sorted.
fn via_prefixes(forbidden: &[Vec<u32>], at_most: usize) -> (Vec<&[u32]>, usize) {
    let mut prefixes: Vec<&[u32]> = forbidden.iter().flat_map(|f| (2..f.len()).map(move |j| &f[..j])).collect();
    prefixes.sort_unstable();
    prefixes.dedup();
    let total = prefixes.len();
    if total > at_most {
        let mut g = rand::rngs::StdRng::seed_from_u64(0x5649_4157_4159);
        let mut keep = rand::seq::index::sample(&mut g, total, at_most).into_vec();
        keep.sort_unstable();
        prefixes = keep.into_iter().map(|i| prefixes[i]).collect();
    }
    (prefixes, total)
}

/// How a via-way check names the prefixes it covers.
fn via_scope(checked: usize, total: usize) -> String {
    if checked == total { format!("every via-way prefix ({total})") } else { format!("{checked} of {total} via-way prefixes, sampled") }
}

/// Every via-way copy (SPEC §11): for every proper prefix of every forbidden sequence, an
/// ID on its last segment, reached from an ID on its first segment (or, when that has none,
/// the nearest segment within three turns before it), in both directions, against a
/// point-to-point reference Dijkstra. Prefixes whose last segment carries no ID are counted.
pub fn via_copies(rt: &Runtime, ex: &Expected, at_most: usize, rep: &mut Report) -> Result<()> {
    let r = ex.r;
    if !r.has(Fam::Car) {
        return Ok(());
    }
    let (g, mode) = (&r.car, r.mode("car"));
    let k = rt.mode("car")?;
    let (prefixes, total) = via_prefixes(&g.forbidden, at_most);
    // IDs by the segment of each slot, in the ID's component.
    let mut on: Vec<(u32, u32)> = Vec::new();
    for (i, at) in mode.iter().enumerate() {
        if let Ok(e) = at {
            for s in g.seg_of_piece[e.piece as usize] {
                if s != NONE && r.car_components.of[s as usize] == e.component {
                    on.push((s, i as u32));
                }
            }
        }
    }
    on.sort_unstable();
    let first_on = |s: u32| -> Option<usize> {
        let k = on.partition_point(|x| x.0 < s);
        (k < on.len() && on[k].0 == s).then(|| on[k].1 as usize)
    };
    let mut into: Vec<(u32, u32)> = (0..g.base.len()).flat_map(|a| g.base.of(a).map(move |(b, _)| (b, a as u32))).collect();
    into.sort_unstable();
    // The nearest ID upstream of s, breadth first over turns in segment order.
    let upstream = |s: u32| -> Option<usize> {
        let mut level = vec![s];
        let mut seen = vec![s];
        for _ in 0..=3 {
            if let Some(i) = level.iter().find_map(|&x| first_on(x)) {
                return Some(i);
            }
            let mut next = Vec::new();
            for &x in &level {
                let lo = into.partition_point(|y| y.0 < x);
                for &(_, a) in into[lo..].iter().take_while(|y| y.0 == x) {
                    if !seen.contains(&a) {
                        seen.push(a);
                        next.push(a);
                    }
                }
            }
            next.sort_unstable();
            level = next;
        }
        None
    };
    let mut cases: Vec<(usize, usize)> = Vec::new();
    let (mut no_target, mut no_source) = (0, 0);
    for p in &prefixes {
        let Some(t) = first_on(*p.last().unwrap()) else {
            no_target += 1;
            continue;
        };
        match upstream(p[0]) {
            Some(s) if s != t => cases.push((s, t)),
            _ => no_source += 1,
        }
    }
    // One value of each column, from the two IDs' ancestors (SPEC §11): the `from` column of
    // s at t and the `to` column of t at s.
    let results: Vec<Result<Option<String>>> = cases
        .par_iter()
        .map(|&(s, t)| {
            let want = ex.car_pair(s, t);
            let from = rt.column_at(s as u32, t as u32, k, Dir::From)?;
            let to = rt.column_at(t as u32, s as u32, k, Dir::To)?;
            Ok((from != want || to != want)
                .then(|| format!("{} -> {}: from {from}, to {to}, reference {want}", uuid(&r.ids[s]), uuid(&r.ids[t]))))
        })
        .collect();
    let o = rep.check(&format!(
        "car: {}, both directions ({no_target} with no ID on the copied segment, {no_source} with no ID before it)",
        via_scope(prefixes.len(), total)
    ));
    for res in results {
        o.count();
        if let Some(f) = res? {
            o.fail(f);
        }
    }
    Ok(())
}

/// Every via-way copy at segment level (SPEC §5): from the start of the first segment of
/// each forbidden sequence to the arrival at the original of each of its copies (the least
/// over the original and its copies), from and to, against the reference.
pub fn via_segments(rt: &Runtime, ex: &Expected, at_most: usize, rep: &mut Report) -> Result<()> {
    let r = ex.r;
    if !r.has(Fam::Car) {
        return Ok(());
    }
    let g = &r.car;
    let k = rt.mode("car")?;
    let (metric, f) = (rt.vocab.modes[k].info.metric, &rt.faults);
    let (prefixes, total) = via_prefixes(&g.forbidden, at_most);
    let mut pairs: Vec<(u32, u32)> = prefixes.iter().map(|p| (p[0], *p.last().unwrap())).collect();
    pairs.sort_unstable();
    pairs.dedup();
    let x_of = |s: u32| r.car_components.of[s as usize];
    let elsewhere = pairs.iter().filter(|p| x_of(p.0) != x_of(p.1)).count();
    pairs.retain(|p| x_of(p.0) == x_of(p.1));
    // Per component, the layout id of each original segment by its road key.
    let key = |s: u32| (g.way[s as usize], g.seq[s as usize], g.dir[s as usize]);
    let mut layout: BTreeMap<u32, Vec<((i64, u32, u8), u32)>> = BTreeMap::new();
    for &(_, b) in &pairs {
        if let std::collections::btree_map::Entry::Vacant(v) = layout.entry(x_of(b)) {
            let e = rt.component(Fam::Car, x_of(b))?;
            let mut l: Vec<_> = (0..e.n as u32)
                .filter(|&v| e.kind[v as usize] != 2)
                .map(|v| ((e.way[v as usize], e.way_seq[v as usize], e.way_dir[v as usize]), v))
                .collect();
            l.sort_unstable();
            v.insert(l);
        }
    }
    let node = |s: u32| -> Option<u32> {
        let l = &layout[&x_of(s)];
        l.binary_search_by_key(&key(s), |e| e.0).ok().map(|i| l[i].1)
    };
    let want: Vec<u16> = pairs.par_iter().map(|&(a, b)| ex.car_segments(a, b)).collect();
    let missing = |s: u32| anyhow::anyhow!("car segment {s} (way {}) has no node in its component", g.way[s as usize]);
    // Per pair, the labels a full sweep would give, from ancestors alone (SPEC §11): from the
    // first segment to the arrivals at the original, and to the first segment from them.
    let both: Vec<Result<(u16, u16)>> = pairs
        .par_iter()
        .map(|&(a, b)| {
            let e = rt.component(Fam::Car, x_of(b))?;
            let (na, nb) = (node(a).ok_or_else(|| missing(a))?, node(b).ok_or_else(|| missing(b))?);
            let arr: Vec<u32> = e.arrivals(nb, f).collect();
            let from = labels_at(&e, metric, &[(na, 0)], &arr, Dir::From, f).into_iter().min().unwrap();
            let seeds: Vec<(u32, u16)> = arr.iter().map(|&y| (y, 0)).collect();
            let to = labels_at(&e, metric, &seeds, &[na], Dir::To, f)[0];
            Ok((from, to))
        })
        .collect();
    let both: Vec<(u16, u16)> = both.into_iter().collect::<Result<_>>()?;
    let o = rep.check(&format!(
        "car: via-way copies segment to segment, {}, both directions ({} first and last segment pairs; {elsewhere} with the first segment in another component)",
        via_scope(prefixes.len(), total),
        pairs.len() + elsewhere
    ));
    for (side, pick) in [("from", 0), ("to", 1)] {
        for (i, t) in both.iter().map(|t| if pick == 0 { t.0 } else { t.1 }).enumerate() {
            o.count();
            let (a, b) = pairs[i];
            o.expect(t == want[i], || {
                format!("{side}: way {} seg {a} -> way {} seg {b}: {t}, reference {}", g.way[a as usize], g.way[b as usize], want[i])
            });
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
            Check::Hierarchy { family, component, links, levels } => {
                let fam = if *family == "car" { Fam::Car } else { Fam::Active };
                let e = rt.component(fam, *component)?;
                o.expect(e.up_head.len() as u64 == *links && rt.tables[&fam].levels(*component) == *levels, || {
                    format!("{family} component {component}: {} links, {} levels", e.up_head.len(), rt.tables[&fam].levels(*component))
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
            Check::Served { id, mode } => {
                o.expect(reason(id, mode).is_none() && rt.resolve(id).is_ok(), || format!("{id} not served for {mode}"))
            }
            Check::Alias { alias, kept } => {
                o.expect(rt.resolve(alias).is_ok() && rt.resolve(alias) == rt.resolve(kept), || {
                    format!("alias {alias} does not resolve to {kept}")
                });
                o.expect(tt(rt, alias, kept, "walk") == json!({"time": 0}), || "alias and kept place are not the same ID".into());
            }
            Check::Unknown { id } => o.expect(rt.resolve(id) == Err(blobmap_model::runtime::Reason::UnknownId), || format!("{id} is known")),
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
                    (Ok(x), Ok(y)) => blobmap_model::runtime::query::same_piece(&rt.vocab.modes[m], x as usize, y as usize),
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
                let e = if *family == "car" { &ex.r.car_components } else { &ex.r.active_components };
                let n = e.trapped.iter().filter(|&&t| t).count();
                o.expect(n >= *at_least, || format!("{family}: {n} trapped, expected at least {at_least}"));
            }
            Check::Restrictions { at_least } => {
                let n = ex.r.car.stats.way_via_applied;
                o.expect(n >= *at_least, || format!("{n} via-way restrictions applied"));
            }
            Check::DeadEnds { at_least } => {
                let n = ex.r.car.dead_ends.len();
                o.expect(n >= *at_least, || format!("{n} restrictions with a dead-end prefix copy"));
            }
            Check::OnVia { id } => {
                let m = rt.mode("car")?;
                let ok = rt.resolve(id).is_ok_and(|i| {
                    let mv = &rt.vocab.modes[m];
                    let x = mv.component[i as usize];
                    x != NONE
                        && rt.component(Fam::Car, x).is_ok_and(|e| {
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
                    let e = rt.component(fam, 0)?;
                    o.expect(e.blocks.iter().filter(|b| b.kind == 2).count() >= 2, || format!("{family} component 0 is not built as cells"));
                }
            }
        }
    }
    let _ = reason_name;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn via_prefixes_all_or_a_fixed_sample() {
        let forbidden: Vec<Vec<u32>> = (0..500u32).map(|i| vec![i, i + 1, i + 2, i + 3]).collect();
        let (all, total) = via_prefixes(&forbidden, usize::MAX);
        assert_eq!((all.len(), total), (1000, 1000));
        let (some, total) = via_prefixes(&forbidden, 100);
        assert_eq!((some.len(), total), (100, 1000));
        assert!(some.windows(2).all(|w| w[0] < w[1]) && some.iter().all(|p| all.contains(p)));
        assert_eq!(some, via_prefixes(&forbidden, 100).0);
    }
}
