//! Dijkstra on the original graphs, with IDs inserted at their offsets (FORMAT §8, last
//! paragraph). CAR runs on the turns before restrictions and tracks the segments driven:
//! a state is a segment with the longest suffix of the drive that begins some forbidden
//! sequence, and a turn completing a forbidden sequence is never taken. This is
//! independent of the via-way copies. Times are summed in u32 and capped at the end.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

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

struct Turns {
    forbidden: HashSet<Vec<u32>>,
    prefixes: HashSet<Vec<u32>>,
}

impl Turns {
    fn new(r: &Reference) -> Turns {
        let mut t = Turns { forbidden: HashSet::new(), prefixes: HashSet::new() };
        for s in &r.car.forbidden {
            for j in 1..s.len() {
                t.prefixes.insert(s[..j].to_vec());
            }
            t.forbidden.insert(s.clone());
        }
        t
    }

    /// The state after driving `d`: its longest suffix that begins a forbidden sequence,
    /// or its last segment.
    fn norm(&self, d: &[u32]) -> Vec<u32> {
        (0..d.len()).find(|&i| self.prefixes.contains(&d[i..])).map_or_else(|| vec![*d.last().unwrap()], |i| d[i..].to_vec())
    }

    fn completes(&self, d: &[u32]) -> bool {
        (0..d.len()).any(|i| self.forbidden.contains(&d[i..]))
    }
}

/// The times from the ID at `o` to every ID of the mode, in internal-index order.
pub fn row(r: &Reference, m: &Mode, oi: usize) -> Vec<u16> {
    let n = r.n();
    let mut out = vec![INF; n];
    let Ok(o) = m.at[oi] else { return out };
    let so = sides(r, m, &o);
    let mut best = vec![u32::MAX; n];
    best[oi] = 0;
    if m.car {
        let turns = Turns::new(r);
        let g = &r.car;
        let mut id: HashMap<(u32, Vec<u32>), usize> = HashMap::new();
        let mut states: Vec<(u32, Vec<u32>)> = Vec::new();
        let mut dist: Vec<u32> = Vec::new();
        let mut heap = BinaryHeap::new();
        let mut push = |s: (u32, Vec<u32>), t: u32, heap: &mut BinaryHeap<_>, dist: &mut Vec<u32>, states: &mut Vec<(u32, Vec<u32>)>| {
            let k = *id.entry(s.clone()).or_insert_with(|| {
                states.push(s);
                dist.push(u32::MAX);
                dist.len() - 1
            });
            if t < dist[k] {
                dist[k] = t;
                heap.push(Reverse((t, k)));
            }
        };
        for s in so.iter().flatten() {
            let Some((_, after)) = split(s.w, s.g) else { continue };
            let h = turns.norm(&[s.tail]);
            for &(b, c) in &g.base[s.tail as usize] {
                let mut d = h.clone();
                d.push(b);
                if !turns.completes(&d) {
                    push((b, turns.norm(&d)), after + c as u32, &mut heap, &mut dist, &mut states);
                }
            }
        }
        // Arrival at a segment's start, by segment.
        let mut at_seg: HashMap<u32, u32> = HashMap::new();
        while let Some(Reverse((t, k))) = heap.pop() {
            if t > dist[k] {
                continue;
            }
            let (a, h) = states[k].clone();
            let e = at_seg.entry(a).or_insert(u32::MAX);
            *e = (*e).min(t);
            let wa = g.w[a as usize] as u32;
            for &(b, c) in &g.base[a as usize] {
                let mut d = h.clone();
                d.push(b);
                if !turns.completes(&d) {
                    push((b, turns.norm(&d)), t + wa + c as u32, &mut heap, &mut dist, &mut states);
                }
            }
        }
        for (i, b) in best.iter_mut().enumerate() {
            let Ok(d) = m.at[i] else { continue };
            if d.expert != o.expert {
                continue;
            }
            for s in sides(r, m, &d).iter().flatten() {
                if let (Some(&t), Some((before, _))) = (at_seg.get(&s.tail), split(s.w, s.g)) {
                    *b = (*b).min(t.saturating_add(before));
                }
            }
            *b = (*b).min(direct(r, m, &o, &d));
        }
    } else {
        let (g, mi) = (&r.active, m.metric);
        let mut adj: Vec<Vec<(u32, u16)>> = vec![Vec::new(); g.nodes];
        for k in 0..g.len() {
            if g.w[mi][k] != INF {
                adj[g.tail[k] as usize].push((g.head[k], g.w[mi][k]));
            }
        }
        let mut dist = vec![u32::MAX; g.nodes];
        let mut heap = BinaryHeap::new();
        for s in so.iter().flatten() {
            if let Some((_, after)) = split(s.w, s.g)
                && after < dist[s.head as usize]
            {
                dist[s.head as usize] = after;
                heap.push(Reverse((after, s.head)));
            }
        }
        while let Some(Reverse((t, v))) = heap.pop() {
            if t > dist[v as usize] {
                continue;
            }
            for &(h, w) in &adj[v as usize] {
                let x = t + w as u32;
                if x < dist[h as usize] {
                    dist[h as usize] = x;
                    heap.push(Reverse((x, h)));
                }
            }
        }
        for (i, b) in best.iter_mut().enumerate() {
            let Ok(d) = m.at[i] else { continue };
            if d.expert != o.expert {
                continue;
            }
            for s in sides(r, m, &d).iter().flatten() {
                if let Some((before, _)) = split(s.w, s.g)
                    && dist[s.tail as usize] != u32::MAX
                {
                    *b = (*b).min(dist[s.tail as usize] + before);
                }
            }
            *b = (*b).min(direct(r, m, &o, &d));
        }
    }
    for i in 0..n {
        out[i] = cap(best[i]);
    }
    out
}

/// Every row of a mode (N Dijkstras; the tiny tier only).
pub fn matrix(r: &Reference, m: &Mode) -> Vec<Vec<u16>> {
    (0..r.n()).map(|i| row(r, m, i)).collect()
}
