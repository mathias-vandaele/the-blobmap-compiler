//! Columns (FORMAT §8): seeds from the sources' offsets, a full sweep of the component, heads
//! at every ID, and the direct term on a shared piece; and pairs, over two IDs' ancestors. One
//! value of a column, or the labels of a few nodes, from ancestors alone (`column_at`,
//! `labels_at`), for the checks that need one value per case (SPEC §11, via-way restrictions).

use anyhow::Result;
use rayon::prelude::*;

use crate::runtime::model::{Dir, Component, Fam, Faults, ModeVocab, Reason, Runtime};
use crate::{INF, NONE};

/// IDs a task takes in a column's parallel loops.
const PAR_IDS: usize = 1 << 14;
/// Components at least this large are swept level by level in parallel.
const PAR_NODES: usize = 1 << 16;

impl Faults {
    #[inline]
    pub fn add(&self, a: u16, b: u16) -> u16 {
        if self.wrapping { a.wrapping_add(b) } else { a.saturating_add(b) }
    }
}

/// A query's working state, kept from one query to the next so that a thread refills a
/// component's arrays instead of allocating them anew: an allocation the size of a component
/// costs page faults and zeroing, more than the sweep itself on a sparse search, and many
/// threads doing it at once contend in the kernel. Answers never depend on it; its arrays
/// only grow, so `bytes` is what it holds.
#[derive(Default)]
pub struct Work {
    /// Columns (one time per ID) not in use.
    cols: Vec<Vec<u16>>,
    /// The sweep's labels (one per node of a component).
    label: Vec<u16>,
    /// `Ancestors` arrays not in use, every entry zero.
    anc: Vec<(Vec<u64>, Vec<u16>)>,
}

impl Work {
    /// A column of `n` times, every one `INF`.
    pub fn col(&mut self, n: usize) -> Vec<u16> {
        let mut v = self.cols.pop().unwrap_or_default();
        v.clear();
        v.resize(n, INF);
        v
    }

    /// A column back, for a later query.
    pub fn give(&mut self, v: Vec<u16>) {
        self.cols.push(v);
    }

    pub fn bytes(&self) -> usize {
        let cols: usize = self.cols.iter().map(|v| v.capacity() * 2).sum();
        let anc: usize = self.anc.iter().map(|(r, l)| r.capacity() * 8 + l.capacity() * 2).sum();
        cols + self.label.capacity() * 2 + anc
    }
}

/// Another implementation of a column's sweep (FORMAT §8): given the labels of component
/// x with the seeds placed, it leaves the swept labels, exactly those of `sweep`. A column
/// uses it when one is installed in `Runtime::down` and the runtime has no faults; it may
/// decline (`Ok(false)`), and the column then sweeps itself.
pub trait DownSweep: Send + Sync {
    fn sweep(&self, e: &Component, fam: Fam, x: u32, metric: usize, label: &mut [u16], dir: Dir) -> Result<bool>;
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
        self.column_in(&mut Work::default(), sources, mode, dir)
    }

    /// `column` with the working state `w`; the column comes from `w` (`Work::give` it back).
    pub fn column_in(&self, w: &mut Work, sources: &[Source], mode: usize, dir: Dir) -> Result<Vec<u16>> {
        let mv = &self.vocab.modes[mode];
        let f = &self.faults;
        let n = self.vocab.n();
        let mut out = w.col(n);
        let mut by_component: std::collections::BTreeMap<u32, Vec<Source>> = std::collections::BTreeMap::new();
        for s in sources {
            let x = mv.component[s.id as usize];
            anyhow::ensure!(x != NONE, "a column source must be served (internal index {})", s.id);
            let start = if f.no_start { 0 } else { s.start };
            by_component.entry(x).or_default().push(Source { id: s.id, start });
        }
        // The IDs of each source component, from one counting pass over the IDs with a slot
        // per component of the family: a pass per component would cost components × IDs,
        // hours for a column of places in many components.
        let keys: Vec<u32> = by_component.keys().copied().collect();
        let (mut first, mut members) = (Vec::new(), Vec::new());
        if keys.len() > 1 {
            let mut slot = vec![NONE; self.components(mv.info.fam)];
            for (k, &x) in keys.iter().enumerate() {
                slot[x as usize] = k as u32;
            }
            let slot_of = |i: usize| match mv.component[i] {
                NONE => NONE,
                x => slot[x as usize],
            };
            first = vec![0usize; keys.len() + 1];
            for i in 0..n {
                let k = slot_of(i);
                if k != NONE {
                    first[k as usize + 1] += 1;
                }
            }
            for k in 0..keys.len() {
                first[k + 1] += first[k];
            }
            members = vec![0u32; first[keys.len()]];
            let mut at = first.clone();
            for i in 0..n {
                let k = slot_of(i);
                if k != NONE {
                    members[at[k as usize]] = i as u32;
                    at[k as usize] += 1;
                }
            }
        }
        for (k, (x, srcs)) in by_component.into_iter().enumerate() {
            let e = self.component(mv.info.fam, x)?;
            let mut label = std::mem::take(&mut w.label);
            label.clear();
            label.resize(e.n, INF);
            for s in &srcs {
                seed(&e, mv, s, dir, f, &mut |v, t| label[v as usize] = label[v as usize].min(t));
            }
            let faultless = !(f.no_copies || f.no_direct || f.wrapping || f.largest_id || f.no_start);
            let done = match self.down.get() {
                Some(d) if faultless => d.sweep(&e, mv.info.fam, x, mv.info.metric, &mut label, dir)?,
                _ => false,
            };
            if !done {
                sweep(&e, mv.info.metric, &mut label, dir, f);
            }
            // The sources by piece (the component is x's), so that an ID meets only the
            // sources on its own piece, not every source.
            let mut by_piece: Vec<([u32; 2], u32, usize)> = srcs.iter().enumerate().map(|(j, s)| (mv.node[s.id as usize], mv.road[s.id as usize], j)).collect();
            by_piece.sort_unstable();
            let lab = &label[..];
            let time = |i: usize| -> u16 {
                let mut t = head(&e, mv, i, |v| lab[v as usize], dir, f);
                let key = (mv.node[i], mv.road[i]);
                let from = by_piece.partition_point(|p| (p.0, p.1) < key);
                for &(_, _, j) in by_piece[from..].iter().take_while(|p| (p.0, p.1) == key) {
                    let s = &srcs[j];
                    let si = s.id as usize;
                    if si == i {
                        t = t.min(s.start);
                    } else if !f.no_direct {
                        let d = if dir == Dir::From { direct(mv, si, i) } else { direct(mv, i, si) };
                        t = t.min(f.add(s.start, d));
                    }
                }
                t
            };
            // The heads, in parallel over the IDs: each writes its own entry.
            if keys.len() > 1 {
                let ids = &members[first[k]..first[k + 1]];
                if ids.len() >= PAR_IDS {
                    let ts: Vec<u16> = ids.par_iter().with_min_len(PAR_IDS).map(|&i| time(i as usize)).collect();
                    for (&i, t) in ids.iter().zip(ts) {
                        out[i as usize] = out[i as usize].min(t);
                    }
                } else {
                    for &i in ids {
                        out[i as usize] = out[i as usize].min(time(i as usize));
                    }
                }
            } else {
                out.par_chunks_mut(PAR_IDS).enumerate().for_each(|(c, part)| {
                    for (j, o) in part.iter_mut().enumerate() {
                        let i = c * PAR_IDS + j;
                        if mv.component[i] == x {
                            *o = (*o).min(time(i));
                        }
                    }
                });
            }
            w.label = label;
        }
        Ok(out)
    }

    /// The time from o to d, o ≠ d, both served for the mode in one component: the value of
    /// o's `From` column at d, from the ancestors of o's seeds and of d's alone (FORMAT §8,
    /// hierarchy property). d's labels are those of a `To` column from d.
    pub fn pair(&self, o: u32, d: u32, mode: usize) -> Result<u16> {
        self.pair_in(&mut Work::default(), o, d, mode)
    }

    pub fn pair_in(&self, w: &mut Work, o: u32, d: u32, mode: usize) -> Result<u16> {
        let mv = &self.vocab.modes[mode];
        let f = &self.faults;
        let x = mv.component[o as usize];
        anyhow::ensure!(x != NONE && x == mv.component[d as usize], "a pair must be served in one component (internal indices {o}, {d})");
        let e = self.component(mv.info.fam, x)?;
        let from = Ancestors::up(&e, mv, &Source { id: o, start: 0 }, Dir::From, f, w);
        let to = Ancestors::up(&e, mv, &Source { id: d, start: 0 }, Dir::To, f, w);
        let mut t = from.meet(&to, f);
        from.done(w);
        to.done(w);
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
        self.column_at_in(&mut Work::default(), o, d, mode, dir)
    }

    pub fn column_at_in(&self, w: &mut Work, o: u32, d: u32, mode: usize, dir: Dir) -> Result<u16> {
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
        let at = labels_at_in(w, &e, mv.info.metric, &seeds, &targets, dir, f);
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
    labels_at_in(&mut Work::default(), e, metric, seeds, targets, dir, f)
}

pub fn labels_at_in(w: &mut Work, e: &Component, metric: usize, seeds: &[(u32, u16)], targets: &[u32], dir: Dir, f: &Faults) -> Vec<u16> {
    let up = Ancestors::from_seeds(e, seeds.iter().copied(), metric, dir, f, w);
    let go_down = match dir {
        Dir::From => &e.down[metric],
        Dir::To => &e.up[metric],
    };
    let below = Ancestors::closure(e, targets.iter().copied(), w);
    let mut label = vec![INF; below.ids.len()];
    for (i, &v) in below.ids.iter().enumerate() {
        let mut lv = up.label(v);
        for k in e.row(v) {
            let j = below.ids.binary_search(&e.up_head[k]).expect("an ancestor's upper neighbour is an ancestor");
            lv = lv.min(f.add(label[j], go_down[k]));
        }
        label[i] = lv;
    }
    let at = targets.iter().map(|v| label[below.ids.binary_search(v).unwrap()]).collect();
    up.done(w);
    below.done(w);
    at
}

/// A source's labels after the upward half of the sweep, on the nodes its seeds reach over
/// upper links (on a valid hierarchy, their ancestors): every other node keeps `INF`. The
/// arrays span the component; they come from a `Work`, all zero, and go back to it with only
/// the reached entries cleared (`done`), so a search costs what it reaches.
struct Ancestors {
    /// The reached nodes, ascending.
    ids: Vec<u32>,
    /// Whether each node is reached, a bit per node.
    reached: Vec<u64>,
    /// `!label`, so that the zeroed array reads `INF` everywhere.
    not_label: Vec<u16>,
}

impl Ancestors {
    fn up(e: &Component, mv: &ModeVocab, s: &Source, dir: Dir, f: &Faults, w: &mut Work) -> Ancestors {
        let mut seeds = Vec::new();
        seed(e, mv, s, dir, f, &mut |v, t| seeds.push((v, t)));
        Ancestors::from_seeds(e, seeds, mv.info.metric, dir, f, w)
    }

    /// The nodes `from` reach over upper links, ascending, every label `INF`.
    fn closure(e: &Component, from: impl IntoIterator<Item = u32>, w: &mut Work) -> Ancestors {
        let (mut reached, mut not_label) = w.anc.pop().unwrap_or_default();
        if reached.len() < e.n.div_ceil(64) {
            reached.resize(e.n.div_ceil(64), 0);
        }
        if not_label.len() < e.n {
            not_label.resize(e.n, 0);
        }
        let mut a = Ancestors { ids: Vec::new(), reached, not_label };
        let mut stack: Vec<u32> = from.into_iter().collect();
        while let Some(v) = stack.pop() {
            let (word, bit) = (v as usize / 64, 1u64 << (v % 64));
            if a.reached[word] & bit == 0 {
                a.reached[word] |= bit;
                a.ids.push(v);
                stack.extend(e.row(v).map(|k| e.up_head[k]));
            }
        }
        a.ids.sort_unstable();
        a
    }

    fn from_seeds(e: &Component, seeds: impl IntoIterator<Item = (u32, u16)>, metric: usize, dir: Dir, f: &Faults, w: &mut Work) -> Ancestors {
        let seeds: Vec<(u32, u16)> = seeds.into_iter().collect();
        let mut a = Ancestors::closure(e, seeds.iter().map(|s| s.0), w);
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

    /// Clears the reached entries and gives the arrays back to `w`.
    fn done(mut self, w: &mut Work) {
        for &v in &self.ids {
            self.reached[v as usize / 64] = 0;
            self.not_label[v as usize] = 0;
        }
        w.anc.push((self.reached, self.not_label));
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
    if e.n >= PAR_NODES && rayon::current_num_threads() > 1 {
        sweep_levels(e, metric, label, dir, f);
    } else {
        sweep_ids(e, metric, label, dir, f);
    }
}

/// `sweep` in layout-id order, on one thread.
fn sweep_ids(e: &Component, metric: usize, label: &mut [u16], dir: Dir, f: &Faults) {
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

/// `sweep` level by level (FORMAT §5.2). Upward: the cells, each from its bottom level, then
/// the other blocks in decreasing ids; downward: the other blocks in increasing ids, each from
/// its top level, then the cells. Both ends of a link are never in one level, so the nodes of
/// a level go in parallel, and so do the cells, which share no links. A node is final before
/// it is read or pushes, and writes are `min`, so the labels are the sequential sweep's. They
/// are atomics, so a model that breaks the level rule gives wrong labels, never a data race.
fn sweep_levels(e: &Component, metric: usize, label: &mut [u16], dir: Dir, f: &Faults) {
    use std::sync::atomic::{AtomicU16, Ordering::Relaxed};
    let (go_up, go_down) = match dir {
        Dir::From => (&e.up[metric], &e.down[metric]),
        Dir::To => (&e.down[metric], &e.up[metric]),
    };
    // SAFETY: `AtomicU16` has the size, alignment and bit validity of `u16`, and `label` is
    // borrowed mutably, so only these atomics touch it until the function returns.
    let lab: &[AtomicU16] = unsafe { &*(label as *mut [u16] as *const [AtomicU16]) };
    let push = |v: u32| {
        let lv = lab[v as usize].load(Relaxed);
        if lv == INF {
            return;
        }
        for k in e.row(v) {
            let (u, t) = (e.up_head[k] as usize, f.add(lv, go_up[k]));
            if t < lab[u].load(Relaxed) {
                lab[u].fetch_min(t, Relaxed);
            }
        }
    };
    let pull = |v: u32| {
        let mut lv = lab[v as usize].load(Relaxed);
        for k in e.row(v) {
            lv = lv.min(f.add(lab[e.up_head[k] as usize].load(Relaxed), go_down[k]));
        }
        lab[v as usize].store(lv, Relaxed);
    };
    let level = |a: u32, b: u32, g: &(dyn Fn(u32) + Sync)| {
        if ((b - a) as usize) < PAR_IDS {
            (a..b).for_each(g);
        } else {
            (a..b).into_par_iter().with_min_len(PAR_IDS / 4).for_each(g);
        }
    };
    let up = |b: usize| {
        let l = &e.levels[b];
        (0..l.len() - 1).rev().for_each(|i| level(l[i], l[i + 1], &push));
    };
    let down = |b: usize| {
        let l = &e.levels[b];
        (0..l.len() - 1).for_each(|i| level(l[i], l[i + 1], &pull));
    };
    let (cells, rest): (Vec<usize>, Vec<usize>) = (0..e.blocks.len()).partition(|&b| e.blocks[b].kind == 2);
    cells.par_iter().for_each(|&b| up(b));
    rest.iter().rev().for_each(|&b| up(b));
    rest.iter().for_each(|&b| down(b));
    cells.par_iter().for_each(|&b| down(b));
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::arr::Arr;
    use crate::runtime::model::BlockAt;

    /// A hierarchy of a top block and two cells with levels of a few to tens of thousands of
    /// nodes, random upper links obeying FORMAT §5.2's level rules, random weights (some near
    /// saturation) and labels (mostly `INF`).
    fn hierarchy(seed: u64) -> Component {
        let mut x = seed;
        let mut rnd = move |m: u64| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x % m
        };
        let blocks: [(u8, &[u32]); 3] = [(1, &[200, 300, 500, 20000]), (2, &[100, 1000, 30000]), (2, &[50, 500, 25000])];
        let (mut bl, mut levels, mut level_of, mut block_of) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut id = 0u32;
        for (b, (kind, sizes)) in blocks.iter().enumerate() {
            bl.push(BlockAt { kind: *kind, id_first: id, pack_link_first: 0 });
            let mut bounds = vec![id];
            for (l, &k) in sizes.iter().enumerate() {
                for _ in 0..k {
                    level_of.push(l);
                    block_of.push(b);
                }
                id += k;
                bounds.push(id);
            }
            levels.push(bounds);
        }
        let n = id as usize;
        let (mut up_first, mut up_head): (Vec<u64>, Vec<u32>) = (vec![0], Vec::new());
        for v in 0..n {
            let (b, l) = (block_of[v], level_of[v]);
            let mut row: Vec<u32> = (0..rnd(5))
                .filter_map(|_| {
                    // An upper neighbour: a smaller level of v's block, or (from a cell) the top.
                    let to_top = b != 0 && rnd(3) == 0;
                    let (tb, below) = if to_top { (0, levels[0].len() - 1) } else { (b, l) };
                    (below > 0).then(|| {
                        let tl = rnd(below as u64) as usize;
                        let (a, z) = (levels[tb][tl], levels[tb][tl + 1]);
                        a + rnd((z - a) as u64) as u32
                    })
                })
                .collect();
            row.sort_unstable();
            row.dedup();
            up_head.extend(&row);
            up_first.push(up_head.len() as u64);
        }
        let weights = |rnd: &mut dyn FnMut(u64) -> u64| -> Vec<u16> { (0..up_head.len()).map(|_| if rnd(50) == 0 { 65000 } else { rnd(3000) as u16 }).collect() };
        let (up, down) = (weights(&mut rnd), weights(&mut rnd));
        fn a<T: bytemuck::Pod>(v: Vec<T>) -> Arr<T> {
            Arr::new(v, None).unwrap()
        }
        Component {
            fam: Fam::Active,
            n,
            blocks: bl,
            levels,
            up_first: a(up_first),
            up_head: a(up_head),
            up: vec![a(up)],
            down: vec![a(down)],
            arc_first: Arr::default(),
            arc_head: Arr::default(),
            arc_link: Arr::default(),
            arc_local: Arr::default(),
            turn_cost: Arr::default(),
            way: Arr::default(),
            way_dir: Arr::default(),
            way_seq: Arr::default(),
            kind: Arr::default(),
            time_s: Vec::new(),
            w: Vec::new(),
            copy_of: Arr::default(),
            copy_first: Arr::default(),
            copies: Arr::default(),
        }
    }

    #[test]
    fn levels_in_parallel_same_labels() {
        for seed in [1u64, 7, 99] {
            let e = hierarchy(seed);
            let mut x = seed * 31 + 5;
            let seeds: Vec<u16> = (0..e.n)
                .map(|_| {
                    x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    if (x >> 33) % 100 < 3 { ((x >> 40) % 30000) as u16 } else { INF }
                })
                .collect();
            for f in [Faults::default(), Faults { wrapping: true, ..Faults::default() }] {
                for dir in [Dir::From, Dir::To] {
                    let (mut a, mut b) = (seeds.clone(), seeds.clone());
                    sweep_ids(&e, 0, &mut a, dir, &f);
                    sweep_levels(&e, 0, &mut b, dir, &f);
                    assert!(a == b, "seed {seed}, {dir:?}, wrapping {}: labels differ", f.wrapping);
                    assert!(a.iter().filter(|&&t| t != INF).count() > seeds.iter().filter(|&&t| t != INF).count());
                }
            }
        }
    }
}
