//! Columns (FORMAT §8): seeds from the sources' offsets, a full sweep of the expert, heads
//! at every ID, and the direct term on a shared piece.

use anyhow::Result;
use blobmap_format::{INF, NONE};

use crate::model::{Dir, Expert, Fam, Faults, ModeVocab, Reason, Runtime};

impl Faults {
    #[inline]
    pub fn add(&self, a: u16, b: u16) -> u16 {
        if self.wrapping { a.wrapping_add(b) } else { a.saturating_add(b) }
    }
}

/// A source of a column: internal index and start time.
#[derive(Clone, Copy, Debug)]
pub struct Source {
    pub id: u32,
    pub start: u16,
}

impl Runtime {
    /// Whether internal index i is served for the mode, or why not.
    pub fn served(&self, i: u32, mode: usize) -> Result<u32, Reason> {
        if self.vocab.status_of(i) == 4 {
            return Err(Reason::FilteredPlace);
        }
        let mv = &self.vocab.modes[mode];
        match mv.expert[i as usize] {
            NONE => {
                let k = mv.reasons.binary_search_by_key(&i, |r| r.0).map_err(|_| Reason::NoRoad)?;
                Err(Reason::from_code(mv.reasons[k].1).unwrap_or(Reason::NoRoad))
            }
            x => Ok(x),
        }
    }

    /// For each ID in internal-index order, the time to (`To`) or from (`From`) the nearest
    /// source plus its start; `INF` for BEYOND, unreachable or unserved. Every source must be
    /// served for the mode.
    pub fn column(&self, sources: &[Source], mode: usize, dir: Dir) -> Result<Vec<u16>> {
        let mv = &self.vocab.modes[mode];
        let f = &self.faults;
        let n = self.vocab.n();
        let mut out = vec![INF; n];
        let mut by_expert: std::collections::BTreeMap<u32, Vec<Source>> = std::collections::BTreeMap::new();
        for s in sources {
            let x = mv.expert[s.id as usize];
            assert_ne!(x, NONE, "a column source must be served");
            let start = if f.no_start { 0 } else { s.start };
            by_expert.entry(x).or_default().push(Source { id: s.id, start });
        }
        for (x, srcs) in by_expert {
            let e = self.expert(mv.info.fam, x)?;
            let mut label = vec![INF; e.n];
            for s in &srcs {
                seed(&e, mv, s, dir, f, &mut label);
            }
            sweep(&e, mv.info.metric, &mut label, dir, f);
            for i in 0..n {
                if mv.expert[i] != x {
                    continue;
                }
                let mut t = head(&e, mv, i, &label, dir, f);
                for s in &srcs {
                    let si = s.id as usize;
                    if si == i {
                        t = t.min(s.start);
                    } else if !f.no_direct && same_piece(mv, si, i) {
                        let d = if dir == Dir::From { direct(mv, si, i) } else { direct(mv, i, si) };
                        t = t.min(f.add(s.start, d));
                    }
                }
                out[i] = out[i].min(t);
            }
        }
        Ok(out)
    }
}

/// Upward by decreasing layout id, then downward by increasing id (FORMAT §8). `To`
/// sweeps the reversed hierarchy: `down` weights going up, `up` going down.
pub fn sweep(e: &Expert, metric: usize, label: &mut [u16], dir: Dir, f: &Faults) {
    let (go_up, go_down) = match dir {
        Dir::From => (&e.up[metric], &e.down[metric]),
        Dir::To => (&e.down[metric], &e.up[metric]),
    };
    for v in (0..e.n as u32).rev() {
        let lv = label[v as usize];
        if lv == INF {
            continue;
        }
        for k in e.row(v) {
            let u = e.up_head[k] as usize;
            label[u] = label[u].min(f.add(lv, go_up[k]));
        }
    }
    for v in 0..e.n as u32 {
        let mut lv = label[v as usize];
        for k in e.row(v) {
            lv = lv.min(f.add(label[e.up_head[k] as usize], go_down[k]));
        }
        label[v as usize] = lv;
    }
}

fn seed(e: &Expert, mv: &ModeVocab, s: &Source, dir: Dir, f: &Faults, label: &mut [u16]) {
    let i = s.id as usize;
    let (nodes, off) = (mv.node[i], mv.off[i]);
    let mut put = |v: u32, t: u16| label[v as usize] = label[v as usize].min(t);
    match (mv.info.fam, dir) {
        (Fam::Car, Dir::From) => {
            for k in 0..2 {
                if nodes[k] == NONE {
                    continue;
                }
                for a in e.arcs(nodes[k]) {
                    put(e.arc_head[a], f.add(f.add(s.start, off[2 * k + 1]), e.turn_cost[a]));
                }
            }
        }
        (Fam::Car, Dir::To) => {
            for k in 0..2 {
                if nodes[k] == NONE {
                    continue;
                }
                for y in e.arrivals(nodes[k], f) {
                    put(y, f.add(s.start, off[2 * k]));
                }
            }
        }
        (Fam::Active, Dir::From) => {
            put(nodes[0], f.add(s.start, off[1]));
            put(nodes[1], f.add(s.start, off[3]));
        }
        (Fam::Active, Dir::To) => {
            put(nodes[0], f.add(s.start, off[0]));
            put(nodes[1], f.add(s.start, off[2]));
        }
    }
}

fn head(e: &Expert, mv: &ModeVocab, i: usize, label: &[u16], dir: Dir, f: &Faults) -> u16 {
    let (nodes, off) = (mv.node[i], mv.off[i]);
    let mut t = INF;
    match (mv.info.fam, dir) {
        (Fam::Car, Dir::From) => {
            for k in 0..2 {
                if nodes[k] == NONE {
                    continue;
                }
                for y in e.arrivals(nodes[k], f) {
                    t = t.min(f.add(label[y as usize], off[2 * k]));
                }
            }
        }
        (Fam::Car, Dir::To) => {
            for k in 0..2 {
                if nodes[k] == NONE {
                    continue;
                }
                for a in e.arcs(nodes[k]) {
                    t = t.min(f.add(f.add(off[2 * k + 1], e.turn_cost[a]), label[e.arc_head[a] as usize]));
                }
            }
        }
        (Fam::Active, Dir::From) => {
            t = f.add(label[nodes[0] as usize], off[0]).min(f.add(label[nodes[1] as usize], off[2]));
        }
        (Fam::Active, Dir::To) => {
            t = f.add(off[1], label[nodes[0] as usize]).min(f.add(off[3], label[nodes[1] as usize]));
        }
    }
    t
}

/// Two IDs on the same piece (FORMAT §8 "Direct drive"): same segments, or same ends and road.
pub fn same_piece(mv: &ModeVocab, a: usize, b: usize) -> bool {
    mv.expert[a] == mv.expert[b] && mv.node[a] == mv.node[b] && mv.road[a] == mv.road[b]
}

/// The direct term from a to b on their shared piece: per shared directed road with a
/// finite weight, the difference of the parts before when b is not behind a.
pub fn direct(mv: &ModeVocab, a: usize, b: usize) -> u16 {
    let (oa, ob) = (mv.off[a], mv.off[b]);
    let before: [(usize, bool); 2] = match mv.info.fam {
        Fam::Car => [(0, mv.node[a][0] != NONE), (2, mv.node[a][1] != NONE)],
        Fam::Active => [(0, true), (2, true)],
    };
    let mut t = INF;
    for (k, present) in before {
        if present && oa[k] != INF && ob[k] >= oa[k] {
            t = t.min(ob[k] - oa[k]);
        }
    }
    t
}
