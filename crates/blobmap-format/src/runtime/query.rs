//! Columns (FORMAT §8): seeds from the sources' offsets, a full sweep of the component, heads
//! at every ID, and the direct term on a shared piece; and pairs, over two IDs' ancestors. One
//! value of a column, or the labels of a few nodes, from ancestors alone (`column_at`,
//! `labels_at`), for the checks that need one value per case (SPEC §11, via-way restrictions).

use anyhow::Result;

use crate::runtime::model::{Dir, Component, Fam, Faults, ModeVocab, Reason, Runtime};
use crate::{INF, NONE};

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
        match mv.component[i as usize] {
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
        let mut by_component: std::collections::BTreeMap<u32, Vec<Source>> = std::collections::BTreeMap::new();
        for s in sources {
            let x = mv.component[s.id as usize];
            anyhow::ensure!(x != NONE, "a column source must be served (internal index {})", s.id);
            let start = if f.no_start { 0 } else { s.start };
            by_component.entry(x).or_default().push(Source { id: s.id, start });
        }
        for (x, srcs) in by_component {
            let e = self.component(mv.info.fam, x)?;
            let mut label = vec![INF; e.n];
            for s in &srcs {
                seed(&e, mv, s, dir, f, &mut |v, t| label[v as usize] = label[v as usize].min(t));
            }
            sweep(&e, mv.info.metric, &mut label, dir, f);
            for i in 0..n {
                if mv.component[i] != x {
                    continue;
                }
                let mut t = head(&e, mv, i, |v| label[v as usize], dir, f);
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

    /// The time from o to d, o ≠ d, both served for the mode in one component: the value of
    /// o's `From` column at d, from the ancestors of o's seeds and of d's alone (FORMAT §8,
    /// hierarchy property). d's labels are those of a `To` column from d.
    pub fn pair(&self, o: u32, d: u32, mode: usize) -> Result<u16> {
        let mv = &self.vocab.modes[mode];
        let f = &self.faults;
        let x = mv.component[o as usize];
        anyhow::ensure!(x != NONE && x == mv.component[d as usize], "a pair must be served in one component (internal indices {o}, {d})");
        let e = self.component(mv.info.fam, x)?;
        let up = |s: u32, dir: Dir| Ancestors::up(&e, mv, &Source { id: s, start: 0 }, dir, f);
        let mut t = up(o, Dir::From).meet(&up(d, Dir::To), f);
        if !f.no_direct && same_piece(mv, o as usize, d as usize) {
            t = t.min(f.add(0, direct(mv, o as usize, d as usize)));
        }
        Ok(t)
    }

    /// `column(&[Source { id: o, start: 0 }], mode, dir)[d]`, from the ancestors of o's seeds and
    /// of the nodes d's head reads, not the whole component: on a valid hierarchy every label
    /// those nodes take in the sweep comes from their ancestors (FORMAT §8). o and d must be
    /// served in one component.
    pub fn column_at(&self, o: u32, d: u32, mode: usize, dir: Dir) -> Result<u16> {
        let mv = &self.vocab.modes[mode];
        let f = &self.faults;
        let x = mv.component[o as usize];
        anyhow::ensure!(x != NONE && x == mv.component[d as usize], "column_at: both must be served in one component (internal indices {o}, {d})");
        if o == d {
            return Ok(0);
        }
        let e = self.component(mv.info.fam, x)?;
        let mut seeds = Vec::new();
        seed(&e, mv, &Source { id: o, start: 0 }, dir, f, &mut |v, t| seeds.push((v, t)));
        let mut targets = Vec::new();
        head(&e, mv, d as usize, |v| {
            targets.push(v);
            INF
        }, dir, f);
        let at = labels_at(&e, mv.info.metric, &seeds, &targets, dir, f);
        let mut next = at.iter();
        let mut t = head(&e, mv, d as usize, |_| *next.next().unwrap(), dir, f);
        if !f.no_direct && same_piece(mv, o as usize, d as usize) {
            let dd = if dir == Dir::From { direct(mv, o as usize, d as usize) } else { direct(mv, d as usize, o as usize) };
            t = t.min(f.add(0, dd));
        }
        Ok(t)
    }
}

/// The labels a full `sweep` from `seeds` gives `targets`, from ancestors alone: the upward
/// half over the seeds' ancestors, then the downward half over the targets' ancestors, by
/// increasing layout id (every upper neighbour of an ancestor is an ancestor).
pub fn labels_at(e: &Component, metric: usize, seeds: &[(u32, u16)], targets: &[u32], dir: Dir, f: &Faults) -> Vec<u16> {
    let up = Ancestors::from_seeds(e, seeds.iter().copied(), metric, dir, f);
    let go_down = match dir {
        Dir::From => &e.down[metric],
        Dir::To => &e.up[metric],
    };
    let below = Ancestors::closure(e, targets.iter().copied());
    let mut label = vec![INF; below.ids.len()];
    for (i, &v) in below.ids.iter().enumerate() {
        let mut lv = up.label(v);
        for k in e.row(v) {
            let j = below.ids.binary_search(&e.up_head[k]).expect("an ancestor's upper neighbour is an ancestor");
            lv = lv.min(f.add(label[j], go_down[k]));
        }
        label[i] = lv;
    }
    targets.iter().map(|v| label[below.ids.binary_search(v).unwrap()]).collect()
}

/// A source's labels after the upward half of the sweep, on the nodes its seeds reach over
/// upper links (on a valid hierarchy, their ancestors): every other node keeps `INF`. The
/// arrays span the component but start zeroed, so only the pages of the reached nodes are
/// ever touched.
struct Ancestors {
    /// The reached nodes, ascending.
    ids: Vec<u32>,
    /// Whether each node is reached, a bit per node.
    reached: Vec<u64>,
    /// `!label`, so that the zeroed array reads `INF` everywhere.
    not_label: Vec<u16>,
}

impl Ancestors {
    fn up(e: &Component, mv: &ModeVocab, s: &Source, dir: Dir, f: &Faults) -> Ancestors {
        let mut seeds = Vec::new();
        seed(e, mv, s, dir, f, &mut |v, t| seeds.push((v, t)));
        Ancestors::from_seeds(e, seeds, mv.info.metric, dir, f)
    }

    /// The nodes `from` reach over upper links, ascending, every label `INF`.
    fn closure(e: &Component, from: impl IntoIterator<Item = u32>) -> Ancestors {
        let mut a = Ancestors { ids: Vec::new(), reached: vec![0; e.n.div_ceil(64)], not_label: vec![0; e.n] };
        let mut stack: Vec<u32> = from.into_iter().collect();
        while let Some(v) = stack.pop() {
            let (w, bit) = (v as usize / 64, 1u64 << (v % 64));
            if a.reached[w] & bit == 0 {
                a.reached[w] |= bit;
                a.ids.push(v);
                stack.extend(e.row(v).map(|k| e.up_head[k]));
            }
        }
        a.ids.sort_unstable();
        a
    }

    fn from_seeds(e: &Component, seeds: impl IntoIterator<Item = (u32, u16)>, metric: usize, dir: Dir, f: &Faults) -> Ancestors {
        let seeds: Vec<(u32, u16)> = seeds.into_iter().collect();
        let mut a = Ancestors::closure(e, seeds.iter().map(|s| s.0));
        for (v, t) in seeds {
            a.put(v, t);
        }
        let go_up = match dir {
            Dir::From => &e.up[metric],
            Dir::To => &e.down[metric],
        };
        for i in (0..a.ids.len()).rev() {
            let v = a.ids[i];
            let lv = a.label(v);
            if lv == INF {
                continue;
            }
            for k in e.row(v) {
                a.put(e.up_head[k], f.add(lv, go_up[k]));
            }
        }
        a
    }

    fn label(&self, v: u32) -> u16 {
        !self.not_label[v as usize]
    }

    fn put(&mut self, v: u32, t: u16) {
        let x = &mut self.not_label[v as usize];
        *x = (*x).max(!t);
    }

    fn reaches(&self, v: u32) -> bool {
        self.reached[v as usize / 64] & (1u64 << (v % 64)) != 0
    }

    /// The least sum of the two labels over the nodes both reach.
    fn meet(&self, to: &Ancestors, f: &Faults) -> u16 {
        let mut t = INF;
        for &z in &self.ids {
            if to.reaches(z) {
                t = t.min(f.add(self.label(z), to.label(z)));
            }
        }
        t
    }
}

/// Upward by decreasing layout id, then downward by increasing id (FORMAT §8). `To`
/// sweeps the reversed hierarchy: `down` weights going up, `up` going down.
pub fn sweep(e: &Component, metric: usize, label: &mut [u16], dir: Dir, f: &Faults) {
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

/// Puts each seed of a source: a node and its label.
fn seed(e: &Component, mv: &ModeVocab, s: &Source, dir: Dir, f: &Faults, put: &mut impl FnMut(u32, u16)) {
    let i = s.id as usize;
    let (nodes, off) = (mv.node[i], mv.off[i]);
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

/// The head of ID i (FORMAT §8) from `label`, called on its nodes in a fixed order.
fn head(e: &Component, mv: &ModeVocab, i: usize, mut label: impl FnMut(u32) -> u16, dir: Dir, f: &Faults) -> u16 {
    let (nodes, off) = (mv.node[i], mv.off[i]);
    let mut t = INF;
    match (mv.info.fam, dir) {
        (Fam::Car, Dir::From) => {
            for k in 0..2 {
                if nodes[k] == NONE {
                    continue;
                }
                for y in e.arrivals(nodes[k], f) {
                    t = t.min(f.add(label(y), off[2 * k]));
                }
            }
        }
        (Fam::Car, Dir::To) => {
            for k in 0..2 {
                if nodes[k] == NONE {
                    continue;
                }
                for a in e.arcs(nodes[k]) {
                    t = t.min(f.add(f.add(off[2 * k + 1], e.turn_cost[a]), label(e.arc_head[a])));
                }
            }
        }
        (Fam::Active, Dir::From) => {
            t = f.add(label(nodes[0]), off[0]).min(f.add(label(nodes[1]), off[2]));
        }
        (Fam::Active, Dir::To) => {
            t = f.add(off[1], label(nodes[0])).min(f.add(off[3], label(nodes[1])));
        }
    }
    t
}

/// Two IDs on the same piece (FORMAT §8 "Direct drive"): same segments, or same ends and road.
pub fn same_piece(mv: &ModeVocab, a: usize, b: usize) -> bool {
    mv.component[a] == mv.component[b] && mv.node[a] == mv.node[b] && mv.road[a] == mv.road[b]
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
