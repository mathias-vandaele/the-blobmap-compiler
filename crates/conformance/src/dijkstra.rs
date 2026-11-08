//! Dijkstra on the original graphs, with IDs inserted at their offsets (FORMAT §8, last
//! paragraph). CAR runs on the turns before restrictions and tracks the segments driven:
//! a state is a segment with the longest suffix of the drive that begins some forbidden
//! sequence, and a turn completing a forbidden sequence is never taken. This is
//! independent of the via-way copies. Times are summed in u32 and capped at the end.
//!
//! The states and their turns are built once, forward and backward, so a column in either
//! direction is one multi-source Dijkstra per expert of its sources.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet};

use blobmap_format::{INF, NONE};
use blobmap_snap::parts;

use crate::reference::{Entry, Mode, Reference};

/// A time in u32, capped to the model's u16 (SPEC §7).
fn cap(t: u32) -> u16 {
    t.min(INF as u32) as u16
}

/// The parts of a road of weight w around fraction g, or None where w is `INF`.
fn split(w: u16, g: f64) -> Option<(u32, u32)> {
    (w != INF).then(|| {
        let (a, b) = parts(w, g);
        (a as u32, b as u32)
    })
}

/// Directed roads of an entry's piece: (road weight, fraction from its tail, tail state key,
/// slot). For CAR, the slots are the piece's two segments; for ACTIVE, its two roads.
struct Side {
    w: u16,
    g: f64,
    /// CAR: the segment; ACTIVE: the tail node, then the head node.
    tail: u32,
    head: u32,
}

fn sides(r: &Reference, m: &Mode, e: &Entry) -> Vec<Option<Side>> {
    let k = e.piece as usize;
    if m.car {
        let g = &r.car;
        (0..2)
            .map(|d| {
                let s = g.seg_of_piece[k][d];
                (s != NONE && r.car_experts.of[s as usize] == e.expert).then(|| Side {
                    w: g.w[s as usize],
                    g: if d == 0 { e.frac } else { 1.0 - e.frac },
                    tail: s,
                    head: s,
                })
            })
            .collect()
    } else {
        let (g, p) = (&r.active, &r.active_pieces);
        (0..2)
            .map(|d| {
                let road = g.road_of_piece[k][d];
                (road != NONE).then(|| {
                    let (t, h) = if d == 0 { (p.u[k], p.v[k]) } else { (p.v[k], p.u[k]) };
                    Side { w: g.w[m.metric][road as usize], g: if d == 0 { e.frac } else { 1.0 - e.frac }, tail: t, head: h }
                })
            })
            .collect()
    }
}

/// The direct term from o to d on their shared piece (SPEC §9).
fn direct(r: &Reference, m: &Mode, o: &Entry, d: &Entry) -> u32 {
    if o.piece != d.piece || o.expert != d.expert {
        return u32::MAX;
    }
    let (so, sd) = (sides(r, m, o), sides(r, m, d));
    let mut t = u32::MAX;
    for k in 0..2 {
        if let (Some(a), Some(b)) = (&so[k], &sd[k])
            && let (Some((pa, _)), Some((pb, _))) = (split(a.w, a.g), split(b.w, b.g))
            && pb >= pa
        {
            t = t.min(pb - pa);
        }
    }
    t
}

/// A graph in CSR form: per node, its arcs (head, cost).
#[derive(Default)]
struct Csr {
    first: Vec<u32>,
    head: Vec<u32>,
    cost: Vec<u16>,
}

impl Csr {
    /// From arcs (tail, head, cost) of n nodes, in their order within each tail.
    fn new(n: usize, arcs: &[(u32, u32, u16)]) -> Csr {
        let mut first = vec![0u32; n + 1];
        for &(t, _, _) in arcs {
            first[t as usize + 1] += 1;
        }
        for i in 0..n {
            first[i + 1] += first[i];
        }
        let mut at = first.clone();
        let (mut head, mut cost) = (vec![0u32; arcs.len()], vec![0u16; arcs.len()]);
        for &(t, h, c) in arcs {
            let k = at[t as usize] as usize;
            head[k] = h;
            cost[k] = c;
            at[t as usize] += 1;
        }
        Csr { first, head, cost }
    }

    fn arcs(&self, v: u32) -> impl Iterator<Item = (u32, u16)> + '_ {
        let r = self.first[v as usize] as usize..self.first[v as usize + 1] as usize;
        self.head[r.clone()].iter().copied().zip(self.cost[r].iter().copied())
    }

    fn reversed(&self) -> Csr {
        let n = self.first.len() - 1;
        let mut arcs = Vec::with_capacity(self.head.len());
        for v in 0..n as u32 {
            arcs.extend(self.arcs(v).map(|(h, c)| (h, v, c)));
        }
        Csr::new(n, &arcs)
    }
}

/// The CAR states: `0 .. originals` are the segments alone; the others are the proper
/// prefixes of forbidden sequences of length two or more, sorted. A state's turns lead to
/// the state after driving on, with the turn cost.
struct CarStates {
    last: Vec<u32>,
    fwd: Csr,
    rev: Csr,
    /// (segment, state) of the states of length two or more, sorted.
    by_last: Vec<(u32, u32)>,
}

impl CarStates {
    fn new(r: &Reference) -> CarStates {
        let g = &r.car;
        let s = g.base.len();
        let forbidden: HashSet<&[u32]> = g.forbidden.iter().map(Vec::as_slice).collect();
        assert!(g.forbidden.iter().all(|f| f.len() >= 2), "a forbidden sequence of one segment");
        let mut long: Vec<&[u32]> = g.forbidden.iter().flat_map(|f| (2..f.len()).map(move |j| &f[..j])).collect();
        long.sort_unstable();
        long.dedup();
        let id: HashMap<&[u32], u32> = long.iter().enumerate().map(|(k, p)| (*p, (s + k) as u32)).collect();
        let firsts: HashSet<u32> = g.forbidden.iter().map(|f| f[0]).collect();
        let mut last: Vec<u32> = (0..s as u32).collect();
        last.extend(long.iter().map(|p| *p.last().unwrap()));
        let mut arcs: Vec<(u32, u32, u16)> = Vec::new();
        let mut d: Vec<u32> = Vec::new();
        for k in 0..last.len() {
            let a = last[k];
            if k < s && !firsts.contains(&a) {
                arcs.extend(g.base[a as usize].iter().map(|&(b, c)| (k as u32, b, c)));
                continue;
            }
            for &(b, c) in &g.base[a as usize] {
                d.clear();
                if k < s {
                    d.push(a);
                } else {
                    d.extend_from_slice(long[k - s]);
                }
                d.push(b);
                if (0..d.len()).any(|i| forbidden.contains(&d[i..])) {
                    continue;
                }
                let next = (0..d.len() - 1).find_map(|i| id.get(&d[i..]).copied()).unwrap_or(b);
                arcs.push((k as u32, next, c));
            }
        }
        let fwd = Csr::new(last.len(), &arcs);
        let rev = fwd.reversed();
        let mut by_last: Vec<(u32, u32)> = (s..last.len()).map(|k| (last[k], k as u32)).collect();
        by_last.sort_unstable();
        CarStates { last, fwd, rev, by_last }
    }

    /// Every state whose segment is a.
    fn of(&self, a: u32) -> impl Iterator<Item = u32> + '_ {
        let lo = self.by_last.partition_point(|x| x.0 < a);
        let hi = self.by_last.partition_point(|x| x.0 <= a);
        std::iter::once(a).chain(self.by_last[lo..hi].iter().map(|x| x.1))
    }
}

/// Dijkstra from seeded labels: the arc v → h costs its cost plus `extra(v, h)`.
fn run(g: &Csr, dist: &mut [u32], extra: impl Fn(u32, u32) -> u32) {
    let mut heap: BinaryHeap<Reverse<(u32, u32)>> = (0..dist.len() as u32).filter(|&v| dist[v as usize] != u32::MAX).map(|v| Reverse((dist[v as usize], v))).collect();
    while let Some(Reverse((t, v))) = heap.pop() {
        if t > dist[v as usize] {
            continue;
        }
        for (h, c) in g.arcs(v) {
            let x = t.saturating_add(c as u32 + extra(v, h));
            if x < dist[h as usize] {
                dist[h as usize] = x;
                heap.push(Reverse((x, h)));
            }
        }
    }
}

/// The reference graphs, built once: the CAR states and, per ACTIVE metric, its roads.
pub struct Graphs {
    car: CarStates,
    active: Vec<(Csr, Csr)>,
}

impl Graphs {
    pub fn new(r: &Reference) -> Graphs {
        let g = &r.active;
        let active = (0..g.w.len())
            .map(|m| {
                let arcs: Vec<(u32, u32, u16)> = (0..g.len()).filter(|&k| g.w[m][k] != INF).map(|k| (g.tail[k], g.head[k], g.w[m][k])).collect();
                let f = Csr::new(g.nodes, &arcs);
                let b = f.reversed();
                (f, b)
            })
            .collect();
        Graphs { car: CarStates::new(r), active }
    }

    /// The column of a mode from (`from`) or to (`to`) the sources with their start times:
    /// per ID, the least start ⊕ time over the sources in its expert.
    pub fn column(&self, r: &Reference, m: &Mode, src: &[(usize, u16)], from: bool) -> Vec<u16> {
        let n = r.n();
        let mut best = vec![u32::MAX; n];
        let mut by_expert: BTreeMap<u32, Vec<(usize, u32)>> = BTreeMap::new();
        for &(s, st) in src {
            if let Ok(e) = m.at[s] {
                by_expert.entry(e.expert).or_default().push((s, st as u32));
            }
        }
        for (x, srcs) in by_expert {
            self.expert_column(r, m, x, &srcs, from, &mut best);
        }
        best.into_iter().map(cap).collect()
    }

    fn expert_column(&self, r: &Reference, m: &Mode, x: u32, srcs: &[(usize, u32)], from: bool, best: &mut [u32]) {
        // The sources by piece, for the direct terms.
        let mut on_piece: Vec<(u32, usize, u32)> = srcs.iter().map(|&(s, st)| (m.at[s].unwrap().piece, s, st)).collect();
        on_piece.sort_unstable();
        for &(s, st) in srcs {
            best[s] = best[s].min(st);
        }
        if m.car {
            let (g, cs) = (&r.car, &self.car);
            let mut dist = vec![u32::MAX; cs.last.len()];
            if from {
                for &(s, st) in srcs {
                    for side in sides(r, m, &m.at[s].unwrap()).iter().flatten() {
                        let Some((_, after)) = split(side.w, side.g) else { continue };
                        for (k, c) in cs.fwd.arcs(side.tail) {
                            let t = st + after + c as u32;
                            dist[k as usize] = dist[k as usize].min(t);
                        }
                    }
                }
                run(&cs.fwd, &mut dist, |k, _| g.w[cs.last[k as usize] as usize] as u32);
                // Arrival at a segment's start, in any state.
                for &(a, k) in &cs.by_last {
                    dist[a as usize] = dist[a as usize].min(dist[k as usize]);
                }
            } else {
                for &(s, st) in srcs {
                    for side in sides(r, m, &m.at[s].unwrap()).iter().flatten() {
                        let Some((before, _)) = split(side.w, side.g) else { continue };
                        for k in cs.of(side.tail) {
                            dist[k as usize] = dist[k as usize].min(st + before);
                        }
                    }
                }
                // Backward, the arc k' → k also drives the segment of k.
                run(&cs.rev, &mut dist, |_, k| g.w[cs.last[k as usize] as usize] as u32);
            }
            self.read(r, m, x, &on_piece, from, best, |side| {
                if from {
                    let (before, _) = split(side.w, side.g)?;
                    Some(dist[side.tail as usize].saturating_add(before))
                } else {
                    let (_, after) = split(side.w, side.g)?;
                    cs.fwd.arcs(side.tail).map(|(k, c)| dist[k as usize].saturating_add(after + c as u32)).min()
                }
            });
        } else {
            let (f, b) = &self.active[m.metric];
            let mut dist = vec![u32::MAX; r.active.nodes];
            for &(s, st) in srcs {
                for side in sides(r, m, &m.at[s].unwrap()).iter().flatten() {
                    let Some((before, after)) = split(side.w, side.g) else { continue };
                    let (v, t) = if from { (side.head, st + after) } else { (side.tail, st + before) };
                    dist[v as usize] = dist[v as usize].min(t);
                }
            }
            run(if from { f } else { b }, &mut dist, |_, _| 0);
            self.read(r, m, x, &on_piece, from, best, |side| {
                let (before, after) = split(side.w, side.g)?;
                Some(if from { dist[side.tail as usize].saturating_add(before) } else { dist[side.head as usize].saturating_add(after) })
            });
        }
    }

    /// Per ID of expert x: the least over its sides of `at(side)`, and the direct terms with
    /// the sources on its piece.
    #[allow(clippy::too_many_arguments)]
    fn read(&self, r: &Reference, m: &Mode, x: u32, on_piece: &[(u32, usize, u32)], from: bool, best: &mut [u32], at: impl Fn(&Side) -> Option<u32>) {
        for (i, b) in best.iter_mut().enumerate() {
            let Ok(d) = m.at[i] else { continue };
            if d.expert != x {
                continue;
            }
            for s in sides(r, m, &d).iter().flatten() {
                if let Some(t) = at(s) {
                    *b = (*b).min(t);
                }
            }
            let lo = on_piece.partition_point(|p| p.0 < d.piece);
            for &(_, s, st) in on_piece[lo..].iter().take_while(|p| p.0 == d.piece) {
                let o = m.at[s].unwrap();
                let t = if from { direct(r, m, &o, &d) } else { direct(r, m, &d, &o) };
                *b = (*b).min(st.saturating_add(t));
            }
        }
    }
}
