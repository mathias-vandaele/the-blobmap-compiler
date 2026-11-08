//! Re-customization from the bytes alone (FORMAT §6): new master times → integer road
//! weights by rounding group → link initialisation → triangles. With the stored master
//! times it must reproduce the stored metric tensors exactly.

use std::collections::BTreeMap;

use anyhow::Result;
use blobmap_format::{INF, NONE, rnd};

use crate::model::{Expert, Fam, Runtime};

/// A road of a family: (expert, index) — a CAR node or an ACTIVE arc — or an ACTIVE cross
/// road (`expert = NONE`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct RoadRef {
    pub expert: u32,
    pub index: u32,
}

/// The integer weights of every road (FORMAT §6.1) for one metric. `roads` are (key way,
/// way_dir, way_seq, member, time).
pub fn round(roads: &[(i64, u8, u32, bool, f32)]) -> Vec<u16> {
    let mut w = vec![INF; roads.len()];
    let mut groups: BTreeMap<(i64, u8), Vec<usize>> = BTreeMap::new();
    for (k, r) in roads.iter().enumerate() {
        if r.3 && r.4.is_finite() {
            groups.entry((r.0, r.1)).or_default().push(k);
        }
    }
    for ((_, dir), mut g) in groups {
        g.sort_by_key(|&k| roads[k].2);
        if dir == 1 {
            g.reverse();
        }
        let t: Vec<f64> = g.iter().map(|&k| roads[k].4 as f64).collect();
        let total = rnd(t.iter().sum::<f64>());
        let fl: Vec<u64> = t.iter().map(|x| x.floor() as u64).collect();
        let extra = total - fl.iter().sum::<u64>();
        let mut by: Vec<usize> = (0..g.len()).collect();
        by.sort_by(|&a, &b| (t[b] - fl[b] as f64).partial_cmp(&(t[a] - fl[a] as f64)).unwrap().then(a.cmp(&b)));
        let mut v = fl.clone();
        for &i in by.iter().take(extra as usize) {
            v[i] += 1;
        }
        for (i, &k) in g.iter().enumerate() {
            w[k] = v[i].min(INF as u64) as u16;
        }
    }
    // Copies take their original's weight: the member with the same key.
    let member: BTreeMap<(i64, u8, u32), usize> = (0..roads.len()).filter(|&k| roads[k].3).map(|k| ((roads[k].0, roads[k].1, roads[k].2), k)).collect();
    for k in 0..roads.len() {
        if !roads[k].3
            && let Some(&o) = member.get(&(roads[k].0, roads[k].1, roads[k].2))
        {
            w[k] = w[o];
        }
    }
    w
}

/// Link weights of one expert from its road weights (FORMAT §6.2, §6.3). `w` is per node for
/// CAR and per arc for ACTIVE.
pub fn customize(e: &Expert, w: &[u16]) -> (Vec<u16>, Vec<u16>) {
    let links = e.up_head.len();
    let (mut up, mut down) = (vec![INF; links], vec![INF; links]);
    for v in 0..e.n as u32 {
        for a in e.arcs(v) {
            let l = e.arc_link[a];
            if l == NONE {
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
            let l = l as usize;
            if v > e.arc_head[a] {
                up[l] = up[l].min(x);
            } else {
                down[l] = down[l].min(x);
            }
        }
    }
    for v in (0..e.n as u32).rev() {
        let r = e.row(v);
        for i in r.clone() {
            let x = e.up_head[i];
            for j in r.start..i {
                let y = e.up_head[j];
                let t = e.link(x, y).expect("chordal");
                up[t] = up[t].min(down[i].saturating_add(up[j]));
                down[t] = down[t].min(down[j].saturating_add(up[i]));
            }
        }
    }
    (up, down)
}

pub struct FamilyWeights {
    /// Per metric, per expert, the road weights (CAR per node, ACTIVE per arc).
    pub w: Vec<Vec<Vec<u16>>>,
    /// Per metric, the cross roads' weights.
    pub cross: Vec<Vec<u16>>,
}

impl Runtime {
    /// Re-rounds a whole family from master times given per road and metric.
    pub fn reround(&self, fam: Fam, time: &dyn Fn(RoadRef, usize) -> f32) -> Result<FamilyWeights> {
        let metrics = self.tables[&fam].metrics.len();
        let mut refs: Vec<RoadRef> = Vec::new();
        let mut keys: Vec<(i64, u8, u32, bool)> = Vec::new();
        let ne = self.experts(fam) as u32;
        let mut sizes = Vec::new();
        for x in 0..ne {
            let e = self.expert(fam, x)?;
            sizes.push(e.way.len());
            for k in 0..e.way.len() {
                refs.push(RoadRef { expert: x, index: k as u32 });
                keys.push((e.way[k], e.way_dir[k], e.way_seq[k], e.kind[k] != 2));
            }
        }
        let mut cross_n = 0;
        if fam == Fam::Active {
            let p = "road/active/cross/";
            let way = self.model.get::<i64>(&format!("{p}way"))?;
            let dir = self.model.get::<u8>(&format!("{p}way_dir"))?;
            let seq = self.model.get::<u32>(&format!("{p}way_seq"))?;
            cross_n = way.len();
            for k in 0..way.len() {
                refs.push(RoadRef { expert: NONE, index: k as u32 });
                keys.push((way[k], dir[k], seq[k], true));
            }
        }
        let mut out = FamilyWeights { w: Vec::new(), cross: Vec::new() };
        for m in 0..metrics {
            let roads: Vec<(i64, u8, u32, bool, f32)> = refs.iter().zip(&keys).map(|(&r, k)| (k.0, k.1, k.2, k.3, time(r, m))).collect();
            let w = round(&roads);
            let mut per = Vec::new();
            let mut at = 0;
            for &s in &sizes {
                per.push(w[at..at + s].to_vec());
                at += s;
            }
            out.w.push(per);
            out.cross.push(w[at..at + cross_n].to_vec());
        }
        Ok(out)
    }

    /// The stored master time of a road.
    pub fn stored_time(&self, fam: Fam, r: RoadRef, m: usize) -> f32 {
        if r.expert == NONE {
            let name = &self.tables[&fam].metrics[m];
            return self.model.get::<f32>(&format!("road/active/cross/{name}.time_s")).map(|v| v[r.index as usize]).unwrap_or(f32::NAN);
        }
        self.expert(fam, r.expert).map(|e| e.time_s[m][r.index as usize]).unwrap_or(f32::NAN)
    }
}
