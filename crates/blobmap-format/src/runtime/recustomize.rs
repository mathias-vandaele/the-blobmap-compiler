//! Re-customization from the bytes alone (FORMAT §6): new master times → integer road
//! weights by rounding group → link initialisation → triangles. With the stored master
//! times it must reproduce the stored metric tensors exactly.

use std::sync::Arc;
use std::sync::atomic::AtomicU16;
use std::sync::atomic::Ordering::Relaxed;

use anyhow::{Context, Result};
use rayon::prelude::*;

use crate::runtime::model::{Component, Fam, Runtime};
use crate::{INF, NONE, NONE64, rnd};

/// A road of a family: (component, index) — a CAR node or an ACTIVE arc — or an ACTIVE cross
/// road (`component = NONE`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct RoadRef {
    pub component: u32,
    pub index: u64,
}

/// The integer weights of every road (FORMAT §6.1) for one metric. `roads` are (key way,
/// way_dir, way_seq, member, time). The rounding groups are found by sorting and rounded in
/// parallel; each group's result does not depend on the others.
pub fn round(roads: &[(i64, u8, u32, bool, f32)]) -> Vec<u16> {
    let mut w = vec![INF; roads.len()];
    // The finite members by group (way, direction), along the way, ties by road index.
    let mut idx: Vec<u64> = (0..roads.len() as u64).into_par_iter().filter(|&k| roads[k as usize].3 && roads[k as usize].4.is_finite()).collect();
    idx.par_sort_unstable_by_key(|&k| {
        let r = &roads[k as usize];
        (r.0, r.1, r.2, k)
    });
    let group = |k: u64| (roads[k as usize].0, roads[k as usize].1);
    let mut starts: Vec<usize> = (0..idx.len()).into_par_iter().filter(|&i| i == 0 || group(idx[i]) != group(idx[i - 1])).collect();
    starts.push(idx.len());
    let vals: Vec<u16> = starts
        .par_windows(2)
        .flat_map_iter(|b| {
            let mut g: Vec<u64> = idx[b[0]..b[1]].to_vec();
            if roads[g[0] as usize].1 == 1 {
                g.reverse();
            }
            let t: Vec<f64> = g.iter().map(|&k| roads[k as usize].4 as f64).collect();
            let total = rnd(t.iter().sum::<f64>());
            let fl: Vec<u64> = t.iter().map(|x| x.floor() as u64).collect();
            let extra = total - fl.iter().sum::<u64>();
            let mut by: Vec<usize> = (0..g.len()).collect();
            by.sort_by(|&a, &b| (t[b] - fl[b] as f64).partial_cmp(&(t[a] - fl[a] as f64)).unwrap().then(a.cmp(&b)));
            let mut v = fl;
            for &i in by.iter().take(extra as usize) {
                v[i] += 1;
            }
            let mut out = vec![0u16; g.len()];
            // Back in the sorted order of `idx`.
            let flip = roads[g[0] as usize].1 == 1;
            for (i, x) in v.into_iter().enumerate() {
                out[if flip { g.len() - 1 - i } else { i }] = x.min(INF as u64) as u16;
            }
            out
        })
        .collect();
    for (&k, &x) in idx.iter().zip(&vals) {
        w[k as usize] = x;
    }
    // Copies take their original's weight: the member with the same key (the last one, if
    // several have it).
    let mut members: Vec<u64> = (0..roads.len() as u64).into_par_iter().filter(|&k| roads[k as usize].3).collect();
    let key = |k: u64| (roads[k as usize].0, roads[k as usize].1, roads[k as usize].2);
    members.par_sort_unstable_by_key(|&k| (key(k), k));
    let copied: Vec<(usize, u16)> = (0..roads.len() as u64)
        .into_par_iter()
        .filter(|&k| !roads[k as usize].3)
        .filter_map(|k| {
            let end = members.partition_point(|&m| key(m) <= key(k));
            (end > 0 && key(members[end - 1]) == key(k)).then(|| (k as usize, w[members[end - 1] as usize]))
        })
        .collect();
    for (k, x) in copied {
        w[k] = x;
    }
    w
}

/// Link weights of one component from its road weights (FORMAT §6.2, §6.3). `w` is per node for
/// CAR and per arc for ACTIVE.
///
/// In parallel, with the same result as the pass from the last layout id to the first: every
/// relaxation is a minimum. A node's triangles run in a round after those of every node whose row
/// holds it, so its own links are final when they are read; two nodes of one round are never
/// in each other's rows, so neither writes a link the other reads.
pub fn customize(e: &Component, w: &[u16]) -> Result<(Vec<u16>, Vec<u16>)> {
    let links = e.up_head.len();
    let unset = || -> Vec<AtomicU16> { (0..links).into_par_iter().map(|_| AtomicU16::new(INF)).collect() };
    let (up, down) = (unset(), unset());
    (0..e.n as u32).into_par_iter().for_each(|v| {
        for a in e.arcs(v) {
            let l = e.arc_link[a];
            if l == NONE64 {
                continue;
            }
            let x = match e.fam {
                Fam::Car => {
                    let c = e.turn_cost[a];
                    if c == INF {
                        continue;
                    }
                    w[v as usize].saturating_add(c)
                }
                Fam::Active => w[a],
            };
            if v > e.arc_head[a] {
                up[l as usize].fetch_min(x, Relaxed);
            } else {
                down[l as usize].fetch_min(x, Relaxed);
            }
        }
    });
    // The round of each node: one more than that of every node whose row holds it.
    let mut round = vec![0u32; e.n];
    for v in (0..e.n as u32).rev() {
        let next = round[v as usize] + 1;
        for k in e.row(v) {
            let u = e.up_head[k] as usize;
            round[u] = round[u].max(next);
        }
    }
    let rounds = round.iter().max().map_or(0, |&r| r as usize + 1);
    let mut first = vec![0usize; rounds + 1];
    for &r in &round {
        first[r as usize + 1] += 1;
    }
    for r in 0..rounds {
        first[r + 1] += first[r];
    }
    let mut fill = first.clone();
    let mut by_round = vec![0u32; e.n];
    for (v, &r) in round.iter().enumerate() {
        by_round[fill[r as usize]] = v as u32;
        fill[r as usize] += 1;
    }
    drop((round, fill));
    // The triangles of v with upper corner x = row[i]: links (x, row[j]) for j < i.
    let triangles = |v: u32, i: usize| -> Result<()> {
        let r = e.row(v);
        let (a, x) = (r.start + i, e.up_head[r.start + i]);
        let (da, ua) = (down[a].load(Relaxed), up[a].load(Relaxed));
        for b in r.start..a {
            let y = e.up_head[b];
            let t = e.link(x, y).with_context(|| format!("node {v}: no link {x}-{y}, the hierarchy is not chordal"))?;
            up[t].fetch_min(da.saturating_add(up[b].load(Relaxed)), Relaxed);
            down[t].fetch_min(down[b].load(Relaxed).saturating_add(ua), Relaxed);
        }
        Ok(())
    };
    for k in 0..rounds {
        by_round[first[k]..first[k + 1]].par_iter().try_for_each(|&v| {
            let n = e.row(v).len();
            // A long row's triangles in parallel too: each i writes links of its own x.
            if n >= 512 { (1..n).into_par_iter().try_for_each(|i| triangles(v, i)) } else { (1..n).try_for_each(|i| triangles(v, i)) }
        })?;
    }
    let done = |a: Vec<AtomicU16>| -> Vec<u16> { a.into_par_iter().map(AtomicU16::into_inner).collect() };
    Ok((done(up), done(down)))
}

pub struct FamilyWeights {
    /// Per metric, per component, the road weights (CAR per node, ACTIVE per arc).
    pub w: Vec<Vec<Vec<u16>>>,
    /// Per metric, the cross roads' weights.
    pub cross: Vec<Vec<u16>>,
}

impl Runtime {
    /// Re-rounds a whole family from master times given per road and metric: `time` gets the
    /// road's component, or `None` for an ACTIVE cross road.
    pub fn reround(&self, fam: Fam, time: &(dyn Fn(Option<&Component>, RoadRef, usize) -> f32 + Sync)) -> Result<FamilyWeights> {
        let metrics = self.tables[&fam].metrics.len();
        let ne = self.components(fam) as u32;
        let comps: Vec<Arc<Component>> = (0..ne).into_par_iter().map(|x| self.component(fam, x)).collect::<Result<_>>()?;
        let (mut way, mut dir, mut seq) = (Vec::new(), Vec::new(), Vec::new());
        if fam == Fam::Active {
            let p = "road/active/cross/";
            way = self.model.get::<i64>(&format!("{p}way"))?.into_owned();
            dir = self.model.get::<u8>(&format!("{p}way_dir"))?.into_owned();
            seq = self.model.get::<u32>(&format!("{p}way_seq"))?.into_owned();
        }
        let mut out = FamilyWeights { w: Vec::new(), cross: Vec::new() };
        for m in 0..metrics {
            let mut roads: Vec<(i64, u8, u32, bool, f32)> = comps
                .par_iter()
                .enumerate()
                .flat_map_iter(|(x, e)| {
                    (0..e.way.len()).map(move |k| (e.way[k], e.way_dir[k], e.way_seq[k], e.kind[k] != 2, time(Some(e), RoadRef { component: x as u32, index: k as u64 }, m)))
                })
                .collect();
            roads.extend((0..way.len()).map(|k| (way[k], dir[k], seq[k], true, time(None, RoadRef { component: NONE, index: k as u64 }, m))));
            let w = round(&roads);
            drop(roads);
            let mut per = Vec::with_capacity(comps.len());
            let mut at = 0;
            for e in &comps {
                per.push(w[at..at + e.way.len()].to_vec());
                at += e.way.len();
            }
            out.w.push(per);
            out.cross.push(w[at..].to_vec());
        }
        Ok(out)
    }

    /// The stored master time of every ACTIVE cross road, per metric.
    pub fn stored_cross_times(&self, fam: Fam) -> Result<Vec<Vec<f32>>> {
        if fam != Fam::Active {
            return Ok(Vec::new());
        }
        self.tables[&fam].metrics.iter().map(|name| Ok(self.model.get::<f32>(&format!("road/active/cross/{name}.time_s"))?.into_owned())).collect()
    }
}
