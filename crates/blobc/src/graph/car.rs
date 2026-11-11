//! The CAR family's graph (SPEC §4, FORMAT §5.3): nodes are directed road segments, arcs
//! are allowed turns. Ported from the old `edge/edge.rs` (OSRM v6 turn costs, u-turns,
//! node-via restrictions) and `edge/restrict.rs` (via-way copies as Aho–Corasick states),
//! with `only_*` rewriting, two or more via ways, and copy rows added.

use std::collections::{HashMap, HashSet};

use blobmap_format::NONE;

use crate::graph::components::{Csr, Components, scc};
use crate::graph::round::{Member, round_all};
use crate::osm::geo::{Local, bearing_deg};
use crate::osm::{Pieces, Restriction, Via, cheap_ruler_m};
use crate::profile::CarTurns;

/// OSRM looks 10 m along a road for the turn angle (coordinate_extractor
/// LOOKAHEAD_DISTANCE_WITHOUT_LANES).
pub const LOOKAHEAD_M: f64 = 10.0;

/// A turn from segment a (arriving at the via node) onto segment b.
#[derive(Clone, Copy, Debug)]
pub struct Turn {
    /// Degrees in (-180, 180]: 0 straight, + right, - left.
    pub angle: f64,
    /// Roads (piece ends) at the via node.
    pub roads: usize,
    pub u_turn: bool,
    /// Traffic keeps left on the road arriving (OSRM takes the incoming edge's flag).
    pub left: bool,
}

/// OSRM v6.0.0 car.lua `process_turn`, the duration part: a sigmoid of the angle when the
/// node has more than two roads or the turn is a u-turn, plus the u-turn penalty. Where
/// traffic keeps left the bias is inverted, so right turns cost what left ones do elsewhere.
pub fn turn_seconds(c: &CarTurns, t: &Turn) -> f64 {
    if !(t.roads > 2 || t.u_turn) {
        return 0.0;
    }
    let (p, b) = (c.turn_penalty_s, if t.left { 1.0 / c.turn_bias } else { c.turn_bias });
    let mut s = if t.angle >= 0.0 {
        p / (1.0 + (-((13.0 / b) * t.angle / 180.0 - 6.5 * b)).exp())
    } else {
        p / (1.0 + (-((13.0 * b) * -t.angle / 180.0 - 6.5 / b)).exp())
    };
    if t.u_turn {
        s += c.u_turn_penalty_s;
    }
    s
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CarStats {
    pub segments: usize,
    /// Original segments on roads where traffic keeps left.
    pub left_segments: usize,
    pub copies: usize,
    pub turns: usize,
    pub u_turns_dropped: usize,
    /// Restrictions turned into forbidden sequences, by kind.
    pub node_via_applied: usize,
    pub way_via_applied: usize,
    /// Restrictions whose ways, nodes or chain are not in the graph, or whose sequence a
    /// one-way already makes impossible.
    pub unresolved: usize,
    pub forbidden_sequences: usize,
    /// Via-way copies with no turn out (SPEC §5): every exit completes a forbidden sequence.
    pub dead_end_copies: usize,
}

/// The turn graph. Nodes `0 .. originals` are the road segments in (way, seq, dir) order;
/// the rest are via-way copies, ordered by the prefix they stand for.
#[derive(Clone, Debug, Default)]
pub struct CarGraph {
    pub originals: usize,
    pub piece: Vec<u32>,
    pub way: Vec<i64>,
    pub dir: Vec<u8>,
    pub seq: Vec<u32>,
    /// Routing nodes at the segment's start and end.
    pub tail: Vec<u32>,
    pub head: Vec<u32>,
    pub time: Vec<f32>,
    pub w: Vec<u16>,
    /// For a copy, its original; `NONE` for an original.
    pub copy_of: Vec<u32>,
    /// Per piece, the original segments [along, against] its node order, or `NONE`.
    pub seg_of_piece: Vec<[u32; 2]>,
    /// Turns, sorted by (from, to).
    pub turn_from: Vec<u32>,
    pub turn_to: Vec<u32>,
    pub turn_cost: Vec<u16>,
    /// The turns before any restriction, by the original segment they leave, and the
    /// restrictions as forbidden sequences of original segments: what an independent
    /// turn-aware Dijkstra needs (SPEC §11).
    pub base: BaseTurns,
    pub forbidden: Vec<Vec<u32>>,
    /// Restrictions with a prefix copy that has no turn out, likely mis-tagged (data
    /// report): (relation id, the copy), sorted.
    pub dead_ends: Vec<(i64, u32)>,
    pub stats: CarStats,
}

/// Turns by the original segment they leave, CSR: the segment entered and the cost.
#[derive(Clone, Debug, Default)]
pub struct BaseTurns {
    first: Vec<u64>,
    to: Vec<u32>,
    cost: Vec<u16>,
}

impl BaseTurns {
    /// The segments they leave.
    pub fn len(&self) -> usize {
        self.first.len().saturating_sub(1)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The turns leaving segment a: (segment entered, cost).
    pub fn of(&self, a: usize) -> impl Iterator<Item = (u32, u16)> + '_ {
        let r = self.first[a] as usize..self.first[a + 1] as usize;
        self.to[r.clone()].iter().copied().zip(self.cost[r].iter().copied())
    }
}

/// The point at distance `d` along piece p.
fn point_at(p: &Pieces, piece: usize, cum: &[f64], d: f64) -> (f64, f64) {
    let r = p.points(piece);
    let d = d.clamp(0.0, *cum.last().unwrap());
    let k = cum.partition_point(|&c| c < d).clamp(1, cum.len() - 1);
    let (i, j) = (r.start + k - 1, r.start + k);
    let seg = cum[k] - cum[k - 1];
    let f = if seg > 0.0 { (d - cum[k - 1]) / seg } else { 0.0 };
    let l = Local::at(p.pt_lat[i], p.pt_lon[i]);
    let (x, y) = l.xy(p.pt_lat[j], p.pt_lon[j]);
    l.latlon(f * x, f * y)
}

/// (bearing leaving the tail, bearing arriving at the head) of each direction of a piece.
fn bearings(p: &Pieces, piece: usize) -> [(f64, f64); 2] {
    let r = p.points(piece);
    let mut cum = vec![0.0];
    for i in r.start + 1..r.end {
        cum.push(cum.last().unwrap() + cheap_ruler_m(p.pt_lat[i - 1], p.pt_lon[i - 1], p.pt_lat[i], p.pt_lon[i]));
    }
    let len = *cum.last().unwrap();
    let pt = |d: f64| point_at(p, piece, &cum, d);
    let b = |a: f64, c: f64| bearing_deg(pt(a), pt(c));
    let l = LOOKAHEAD_M;
    [(b(0.0, l.min(len)), b((len - l).max(0.0), len)), (b(len, (len - l).max(0.0)), b(l.min(len), 0.0))]
}

/// The automaton of FORMAT §5.3: forbidden sequences, and their proper prefixes of length
/// at least 2 as states.
struct Automaton {
    forbidden: HashSet<Vec<u32>>,
    prefixes: HashSet<Vec<u32>>,
    forbidden_pairs: HashSet<(u32, u32)>,
    prefix_pairs: HashSet<(u32, u32)>,
}

impl Automaton {
    fn new(seqs: &[Vec<u32>]) -> Automaton {
        let mut a = Automaton {
            forbidden: HashSet::new(),
            prefixes: HashSet::new(),
            forbidden_pairs: HashSet::new(),
            prefix_pairs: HashSet::new(),
        };
        for s in seqs {
            if s.len() == 2 {
                a.forbidden_pairs.insert((s[0], s[1]));
            }
            for j in 2..s.len() {
                if j == 2 {
                    a.prefix_pairs.insert((s[0], s[1]));
                }
                a.prefixes.insert(s[..j].to_vec());
            }
            a.forbidden.insert(s.clone());
        }
        a
    }

    /// The state reached from state `p` by a turn onto `b`: `Ok(None)` for the original b,
    /// `Ok(Some(prefix))` for a copy, `Err(())` if the turn completes a forbidden sequence.
    fn step(&self, p: &[u32], b: u32) -> Result<Option<Vec<u32>>, ()> {
        if p.len() == 1 {
            let pair = (p[0], b);
            if self.forbidden_pairs.contains(&pair) {
                return Err(());
            }
            return Ok(self.prefix_pairs.contains(&pair).then(|| vec![p[0], b]));
        }
        let mut q = p.to_vec();
        q.push(b);
        if (0..q.len() - 1).any(|i| self.forbidden.contains(&q[i..])) {
            return Err(());
        }
        Ok((0..q.len() - 1).find(|&i| self.prefixes.contains(&q[i..])).map(|i| q[i..].to_vec()))
    }
}

impl CarGraph {
    pub fn build(p: &Pieces, turns: &CarTurns, restrictions: &[Restriction]) -> CarGraph {
        let mut g = CarGraph { seg_of_piece: vec![[NONE; 2]; p.len()], ..CarGraph::default() };
        for i in 0..p.len() {
            for d in 0..2 {
                let t = p.time[0][i][d];
                if t.is_finite() {
                    g.seg_of_piece[i][d] = g.piece.len() as u32;
                    g.piece.push(i as u32);
                    g.way.push(p.way[i]);
                    g.dir.push(d as u8);
                    g.seq.push(p.seq[i]);
                    let (a, b) = if d == 0 { (p.u[i], p.v[i]) } else { (p.v[i], p.u[i]) };
                    g.tail.push(a);
                    g.head.push(b);
                    g.time.push(t);
                }
            }
        }
        let s = g.piece.len();
        g.originals = s;
        g.stats.segments = s;
        g.stats.left_segments = g.piece[..s].iter().filter(|&&q| p.left[q as usize]).count();
        let members: Vec<Member> = (0..s).map(|k| Member { way: g.way[k], dir: g.dir[k], seq: g.seq[k], time: g.time[k] }).collect();
        g.w = round_all(&members);
        g.copy_of = vec![NONE; s];

        // Roads at each node: piece ends, a closed piece counting twice.
        let mut roads = vec![0usize; p.node_osm.len()];
        for i in 0..p.len() {
            roads[p.u[i] as usize] += 1;
            roads[p.v[i] as usize] += 1;
        }
        let bear: Vec<[(f64, f64); 2]> = (0..p.len()).map(|i| bearings(p, i)).collect();
        let out = Csr::new(p.node_osm.len(), (0..s as u32).map(|k| (g.tail[k as usize], k)));
        // Base turns, by the segment they leave.
        let mut base = BaseTurns { first: Vec::with_capacity(s + 1), ..BaseTurns::default() };
        base.first.push(0);
        for a in 0..s {
            let x = g.head[a] as usize;
            for &b in out.out(x) {
                let b_ = b as usize;
                let u_turn = g.piece[a] == g.piece[b_] && g.dir[a] != g.dir[b_];
                if u_turn && roads[x] == 2 {
                    g.stats.u_turns_dropped += 1;
                    continue;
                }
                let mut angle = bear[g.piece[b_] as usize][g.dir[b_] as usize].0 - bear[g.piece[a] as usize][g.dir[a] as usize].1;
                while angle > 180.0 {
                    angle -= 360.0;
                }
                while angle <= -180.0 {
                    angle += 360.0;
                }
                if u_turn {
                    angle = 180.0;
                }
                let c = turn_seconds(turns, &Turn { angle, roads: roads[x], u_turn, left: p.left[g.piece[a] as usize] }).round();
                base.to.push(b);
                base.cost.push(c.min(65534.0) as u16);
            }
            base.first.push(base.to.len() as u64);
        }

        let (seqs, of_relation) = g.forbidden_sequences(p, &base, restrictions);
        g.stats.forbidden_sequences = seqs.len();
        let auto = Automaton::new(&seqs);

        // Breadth-first over the states reachable from the originals.
        let mut copies: HashMap<Vec<u32>, u32> = HashMap::new();
        let mut copy_keys: Vec<Vec<u32>> = Vec::new();
        let mut trans: Vec<(u32, u32, u16)> = Vec::new();
        for a in 0..s as u32 {
            for (b, c) in base.of(a as usize) {
                match auto.step(&[a], b) {
                    Err(()) => {}
                    Ok(None) => trans.push((a, b, c)),
                    Ok(Some(q)) => {
                        let next = (s + copy_keys.len()) as u32;
                        let id = *copies.entry(q.clone()).or_insert_with(|| {
                            copy_keys.push(q);
                            next
                        });
                        trans.push((a, id, c));
                    }
                }
            }
        }
        let mut k = 0;
        while k < copy_keys.len() {
            let pkey = copy_keys[k].clone();
            let from = (s + k) as u32;
            let x = *pkey.last().unwrap() as usize;
            for (b, c) in base.of(x) {
                match auto.step(&pkey, b) {
                    Err(()) => {}
                    Ok(None) => trans.push((from, b, c)),
                    Ok(Some(q)) => {
                        let next = (s + copy_keys.len()) as u32;
                        let id = *copies.entry(q.clone()).or_insert_with(|| {
                            copy_keys.push(q);
                            next
                        });
                        trans.push((from, id, c));
                    }
                }
            }
            k += 1;
        }
        // Copies renumbered by their prefix.
        let mut order: Vec<usize> = (0..copy_keys.len()).collect();
        order.sort_by(|&a, &b| copy_keys[a].cmp(&copy_keys[b]));
        let mut renum = vec![0u32; copy_keys.len()];
        for (new, &old) in order.iter().enumerate() {
            renum[old] = (s + new) as u32;
        }
        let map = |v: u32| if (v as usize) < s { v } else { renum[v as usize - s] };
        for &old in &order {
            let x = *copy_keys[old].last().unwrap() as usize;
            g.piece.push(g.piece[x]);
            g.way.push(g.way[x]);
            g.dir.push(g.dir[x]);
            g.seq.push(g.seq[x]);
            g.tail.push(g.tail[x]);
            g.head.push(g.head[x]);
            g.time.push(g.time[x]);
            g.w.push(g.w[x]);
            g.copy_of.push(x as u32);
        }
        g.stats.copies = copy_keys.len();
        let mut trans: Vec<(u32, u32, u16)> = trans.into_iter().map(|(a, b, c)| (map(a), map(b), c)).collect();
        trans.sort_unstable();
        // Copies with no turn out, and the restrictions whose sequences begin with their prefix.
        let mut exits = vec![false; s + copy_keys.len()];
        for &(a, _, _) in &trans {
            exits[a as usize] = true;
        }
        for (k, key) in copy_keys.iter().enumerate() {
            let c = renum[k];
            if exits[c as usize] {
                continue;
            }
            g.stats.dead_end_copies += 1;
            let lo = of_relation.partition_point(|x| x.0.as_slice() < key.as_slice());
            for (_, rel) in of_relation[lo..].iter().take_while(|x| x.0.starts_with(key)) {
                g.dead_ends.push((*rel, c));
            }
        }
        g.dead_ends.sort_unstable();
        g.dead_ends.dedup();
        g.stats.turns = trans.len();
        for (a, b, c) in trans {
            g.turn_from.push(a);
            g.turn_to.push(b);
            g.turn_cost.push(c);
        }
        g.base = base;
        g.forbidden = seqs;
        g
    }

    pub fn len(&self) -> usize {
        self.piece.len()
    }

    /// Drops what only the order, the hierarchy and the road part read, once the family's road
    /// units are written: the vocabulary and the data report read the rest.
    pub fn shed(&mut self) {
        (self.tail, self.head, self.time) = Default::default();
        (self.turn_from, self.turn_to, self.turn_cost) = Default::default();
        (self.base, self.forbidden) = Default::default();
    }

    /// The restrictions as forbidden sequences of original segments (FORMAT §5.3): length 2
    /// for a via node, at least 3 for via ways; `only_*` rewritten as every other exit.
    /// The forbidden sequences, sorted, and every (sequence, relation id), sorted.
    fn forbidden_sequences(&mut self, p: &Pieces, base: &BaseTurns, restrictions: &[Restriction]) -> (Vec<Vec<u32>>, Vec<(Vec<u32>, i64)>) {
        let mut by_way: HashMap<i64, Vec<usize>> = HashMap::new();
        for i in 0..p.len() {
            by_way.entry(p.way[i]).or_default().push(i);
        }
        let node = |osm: i64| p.osm_node(osm);
        let mut out: Vec<Vec<u32>> = Vec::new();
        let mut of_relation: Vec<(Vec<u32>, i64)> = Vec::new();
        for r in restrictions {
            let mut seqs: Vec<Vec<u32>> = Vec::new();
            // (prefix, allowed last segments) per from way: the sequences end with one turn.
            let mut prefixes: Vec<(Vec<u32>, Vec<u32>)> = Vec::new();
            match &r.via {
                Via::Node(n) => {
                    let Some(n) = node(*n) else {
                        self.stats.unresolved += 1;
                        continue;
                    };
                    let ins = |w: &i64| -> Vec<u32> {
                        self.segments_of(&by_way, *w).into_iter().filter(|&s| self.head[s as usize] == n).collect()
                    };
                    let outs = |w: &i64| -> Vec<u32> {
                        self.segments_of(&by_way, *w).into_iter().filter(|&s| self.tail[s as usize] == n).collect()
                    };
                    let to: Vec<u32> = r.to.iter().flat_map(outs).collect();
                    for a in r.from.iter().flat_map(ins) {
                        prefixes.push((vec![a], to.clone()));
                    }
                }
                Via::Ways(vias) => {
                    for f in &r.from {
                        for t in &r.to {
                            if let Some((pre, last)) = self.via_chain(p, &by_way, *f, vias, *t) {
                                prefixes.push((pre, vec![last]));
                            }
                        }
                    }
                }
            }
            for (pre, to) in prefixes {
                let x = *pre.last().unwrap() as usize;
                for (b, _) in base.of(x) {
                    if to.contains(&b) != r.only {
                        let mut s = pre.clone();
                        s.push(b);
                        seqs.push(s);
                    }
                }
            }
            if seqs.is_empty() {
                self.stats.unresolved += 1;
                continue;
            }
            match r.via {
                Via::Node(_) => self.stats.node_via_applied += 1,
                Via::Ways(_) => self.stats.way_via_applied += 1,
            }
            of_relation.extend(seqs.iter().map(|q| (q.clone(), r.id)));
            out.extend(seqs);
        }
        out.sort();
        out.dedup();
        of_relation.sort();
        of_relation.dedup();
        (out, of_relation)
    }

    fn segments_of(&self, by_way: &HashMap<i64, Vec<usize>>, way: i64) -> Vec<u32> {
        by_way.get(&way).into_iter().flatten().flat_map(|&i| self.seg_of_piece[i]).filter(|&s| s != NONE).collect()
    }

    /// The segments of `from` into the via chain, along the via ways, and the segment of
    /// `to` out of it: (from and via segments, to segment). None if the ways do not chain
    /// or a direction is not drivable.
    fn via_chain(&self, p: &Pieces, by_way: &HashMap<i64, Vec<usize>>, from: i64, vias: &[i64], to: i64) -> Option<(Vec<u32>, u32)> {
        let pieces = |w: i64| -> Option<&Vec<usize>> { by_way.get(&w).filter(|v| !v.is_empty()) };
        let ends = |w: i64| -> Option<(u32, u32)> {
            let v = pieces(w)?;
            Some((p.u[v[0]], p.v[*v.last().unwrap()]))
        };
        let (fa, fb) = ends(from)?;
        for start in [fb, fa] {
            // The from way's piece at `start`, arriving there.
            let fp = pieces(from)?;
            let first = if start == fb { self.seg_of_piece[*fp.last().unwrap()][0] } else { self.seg_of_piece[fp[0]][1] };
            if first == NONE {
                continue;
            }
            let mut seq = vec![first];
            let mut at = start;
            let mut left: Vec<i64> = vias.to_vec();
            let mut ok = true;
            while !left.is_empty() {
                let Some(k) = left.iter().position(|&w| ends(w).is_some_and(|(a, b)| a == at || b == at)) else {
                    ok = false;
                    break;
                };
                let w = left.remove(k);
                let (a, b) = ends(w)?;
                let ps = pieces(w)?;
                if a == at {
                    seq.extend(ps.iter().map(|&i| self.seg_of_piece[i][0]));
                    at = b;
                } else {
                    seq.extend(ps.iter().rev().map(|&i| self.seg_of_piece[i][1]));
                    at = a;
                }
            }
            if !ok || seq.contains(&NONE) {
                continue;
            }
            let (ta, tb) = ends(to)?;
            let tp = pieces(to)?;
            let last = if ta == at {
                self.seg_of_piece[tp[0]][0]
            } else if tb == at {
                self.seg_of_piece[*tp.last().unwrap()][1]
            } else {
                continue;
            };
            if last != NONE {
                return Some((seq, last));
            }
        }
        None
    }

    /// The components of the turn graph (SPEC §5), determined by the original segments only:
    /// the strongly connected components of the originals, a turn into or out of a copy
    /// counting as one into or out of its original; each copy is in its original's component,
    /// whatever its own connectivity. Ties go to the smallest (way, way_seq, way_dir), then
    /// originals before copies.
    pub fn components(&self) -> Components {
        let orig = |v: u32| if (v as usize) < self.originals { v } else { self.copy_of[v as usize] };
        let arcs = self.turn_from.iter().zip(&self.turn_to).map(|(&a, &b)| (orig(a), orig(b)));
        let g = Csr::new(self.originals, arcs);
        let mut comp = scc(&g);
        let of_copies: Vec<u32> = (self.originals..self.len()).map(|c| comp[self.copy_of[c] as usize]).collect();
        comp.extend(of_copies);
        let joined = self
            .turn_from
            .iter()
            .copied()
            .zip(self.turn_to.iter().copied())
            .chain(self.seg_of_piece.iter().filter(|s| s[0] != NONE && s[1] != NONE).map(|s| (s[0], s[1])))
            .chain((self.originals..self.len()).map(|c| (c as u32, self.copy_of[c])));
        Components::new(&comp, |v| (self.way[v], self.seq[v], self.dir[v], v), joined)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile::car::{CarProfile, WayAttrs};

    fn graph(name: &str) -> (Pieces, CarGraph) {
        let dir = std::env::temp_dir().join(format!("blobmap-graph-car-{}-{name}", std::process::id()));
        let w = crate::fixtures::by_name(name).unwrap().write(&dir).unwrap();
        let cfg = crate::profile::config::Config::load(&w.configs[0].1).unwrap();
        let osm = crate::osm::Osm::open(&w.pbf, &dir.join("nodes"), &cfg).unwrap();
        let net = crate::osm::Network::read(&osm, &osm.car_obstacles, &CarProfile::new(&cfg.car.config), &cfg.car.config.snap, None).unwrap();
        let p = net.pieces(&[|a: &WayAttrs| *a]).unwrap();
        let (r, _) = osm.restrictions().unwrap();
        let g = CarGraph::build(&p, &cfg.car.config.turns, &r);
        (p, g)
    }

    #[test]
    fn left_hand_turns_mirror() {
        let c = crate::profile::CarTurns { turn_penalty_s: 7.5, turn_bias: 1.075, u_turn_penalty_s: 20.0 };
        let t = |angle: f64, left: bool| turn_seconds(&c, &Turn { angle, roads: 3, u_turn: false, left });
        for a in [10.0, 45.0, 90.0, 135.0, 170.0] {
            assert!((t(a, false) - t(-a, true)).abs() < 1e-12 && (t(-a, false) - t(a, true)).abs() < 1e-12);
        }
        // Keeping right, a right turn is the cheaper; keeping left, a left one.
        assert!(t(90.0, false) < t(-90.0, false) && t(-90.0, true) < t(90.0, true));
    }

    #[test]
    fn viaway_copies() {
        let (_, g) = graph("viaway");
        assert_eq!(g.stats.way_via_applied, 4);
        assert_eq!(g.stats.unresolved, 0);
        assert!(g.stats.copies >= 4, "{:?}", g.stats);
        // Copies take their original's weight and key.
        for c in g.originals..g.len() {
            let o = g.copy_of[c] as usize;
            assert_eq!((g.w[c], g.way[c], g.seq[c], g.dir[c]), (g.w[o], g.way[o], g.seq[o], g.dir[o]));
        }
        let e = g.components();
        assert_eq!(e.nodes[0] as usize, g.len(), "one component: {:?}", e.nodes);
    }

    #[test]
    fn town_turns_and_components() {
        let (p, g) = graph("town");
        assert_eq!(g.stats.node_via_applied, 2);
        assert_eq!(g.stats.copies, 0);
        // Row 0 (way 1): 14.4 s pieces, rounded per way to 15, 15, 14, 14, 14.
        let row0: Vec<u16> = (0..g.originals).filter(|&k| g.way[k] == 1 && g.dir[k] == 0).map(|k| g.w[k]).collect();
        assert_eq!((row0.len(), row0.iter().sum::<u16>()), (5, 72));
        assert!(row0.iter().all(|&w| w == 14 || w == 15));
        let e = g.components();
        assert!(e.trapped.iter().filter(|&&t| t).count() >= 2);
        assert!(!e.trapped[0]);
        // The main grid is component 0, the island and the east bank are served islands.
        assert!(e.nodes[0] > 100);
        let _ = p;
    }
}
