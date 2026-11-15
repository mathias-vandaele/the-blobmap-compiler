//! Integer road weights (SPEC §7, FORMAT §6.1): each (way, direction, metric) group's total
//! rounded once, distributed by largest remainder, ties in travel order.

use blobmap_model::{INF, rnd};

/// The weights of one group, given its members' master times in travel order (all finite).
pub fn round_group(times: &[f32]) -> Vec<u16> {
    let t: Vec<f64> = times.iter().map(|&x| x as f64).collect();
    let mut s = 0.0f64;
    for &x in &t {
        s += x;
    }
    let total = rnd(s);
    let floor: Vec<u64> = t.iter().map(|x| x.floor() as u64).collect();
    let extra = total - floor.iter().sum::<u64>();
    let mut by: Vec<usize> = (0..t.len()).collect();
    // Remainder descending, ties in travel order (stable sort keeps it).
    by.sort_by(|&a, &b| (t[b] - floor[b] as f64).total_cmp(&(t[a] - floor[a] as f64)));
    let mut w: Vec<u64> = floor;
    for &i in by.iter().take(extra as usize) {
        w[i] += 1;
    }
    w.into_iter().map(|x| x.min(INF as u64) as u16).collect()
}

/// A road of a rounding group: its way, direction and position, and one metric's time.
#[derive(Clone, Copy, Debug)]
pub struct Member {
    pub way: i64,
    pub dir: u8,
    pub seq: u32,
    pub time: f32,
}

/// The weight of every road (`INF` where the time is +∞): groups are (way, dir) over the
/// finite members, in travel order (`seq` ascending for dir 0, descending for dir 1).
pub fn round_all(roads: &[Member]) -> Vec<u16> {
    let mut w = vec![INF; roads.len()];
    let mut idx: Vec<u64> = (0..roads.len() as u64).filter(|&i| roads[i as usize].time.is_finite()).collect();
    let key = |i: u64| {
        let r = &roads[i as usize];
        (r.way, r.dir, if r.dir == 0 { r.seq as i64 } else { -(r.seq as i64) })
    };
    idx.sort_unstable_by_key(|&i| key(i));
    for g in idx.chunk_by(|&a, &b| key(a).0 == key(b).0 && key(a).1 == key(b).1) {
        let times: Vec<f32> = g.iter().map(|&i| roads[i as usize].time).collect();
        for (&i, x) in g.iter().zip(round_group(&times)) {
            w[i as usize] = x;
        }
    }
    w
}

/// `round_all` over roads given as columns, without a member array: `way`, `dir`, `seq` per
/// road and `time(r)`. Where the roads come grouped by way (each way's roads consecutive, as
/// a family's pieces are), each way is rounded on its own, in parallel; otherwise the roads are
/// first sorted by way.
pub fn round_columns(way: &[i64], dir: &[u8], seq: &[u32], time: &(dyn Fn(usize) -> f32 + Sync)) -> Vec<u16> {
    use rayon::prelude::*;
    let n = way.len();
    // Each way's roads: consecutive, or listed by way.
    let grouped = way.par_windows(2).all(|w| w[0] <= w[1]);
    let order: Vec<u64> = if grouped {
        Vec::new()
    } else {
        let mut o: Vec<u64> = (0..n as u64).collect();
        o.par_sort_unstable_by_key(|&r| (way[r as usize], r));
        o
    };
    let at = |i: usize| if grouped { i } else { order[i] as usize };
    // Chunks of about `CHUNK` roads, cut between ways.
    const CHUNK: usize = 1 << 20;
    let mut cuts = vec![0usize];
    while *cuts.last().unwrap() < n {
        let mut e = (cuts.last().unwrap() + CHUNK).min(n);
        while e < n && way[at(e)] == way[at(e - 1)] {
            e += 1;
        }
        cuts.push(e);
    }
    let parts: Vec<Vec<(usize, u16)>> = cuts
        .par_windows(2)
        .map(|c| {
            let mut out = Vec::new();
            let mut i = c[0];
            let mut g: Vec<(u8, i64, usize)> = Vec::new();
            while i < c[1] {
                let mut j = i;
                g.clear();
                while j < c[1] && way[at(j)] == way[at(i)] {
                    let r = at(j);
                    if time(r).is_finite() {
                        g.push((dir[r], if dir[r] == 0 { seq[r] as i64 } else { -(seq[r] as i64) }, r));
                    }
                    j += 1;
                }
                g.sort_unstable();
                for d in g.chunk_by(|a, b| a.0 == b.0) {
                    let times: Vec<f32> = d.iter().map(|x| time(x.2)).collect();
                    out.extend(d.iter().map(|x| x.2).zip(round_group(&times)));
                }
                i = j;
            }
            out
        })
        .collect();
    let mut w = vec![INF; n];
    for p in parts {
        for (r, x) in p {
            w[r] = x;
        }
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn largest_remainder() {
        // 14.4 × 5 = 72: floors 70, the first two in travel order get the extra seconds.
        assert_eq!(round_group(&[14.4; 5]), [15, 15, 14, 14, 14]);
        assert_eq!(round_group(&[0.3, 0.3, 0.3]), [1, 0, 0]);
        assert_eq!(round_group(&[0.2, 0.7, 0.6]), [0, 1, 1]);
        assert_eq!(round_group(&[2.5]), [3]);
        assert_eq!(round_group(&[70000.0]), [INF]);
        let m = |seq, dir, time| Member { way: 7, dir, seq, time };
        let w = round_all(&[m(0, 0, 0.4), m(1, 0, 0.4), m(0, 1, 0.4), m(1, 1, 0.4), m(2, 0, f32::INFINITY)]);
        // Against the way, travel order starts at the last piece.
        assert_eq!(w, [1, 0, 0, 1, INF]);
    }

    #[test]
    fn columns_round_as_members() {
        // Ways out of order and in two runs, both directions, infinite times among them.
        let way = [9i64, 9, 9, 9, 3, 3, 9, 3];
        let dir = [0u8, 1, 0, 1, 0, 0, 0, 1];
        let seq = [0u32, 0, 1, 1, 0, 1, 2, 0];
        let time = [0.4f32, 0.4, 0.4, 0.4, 2.5, f32::INFINITY, 0.3, 0.7];
        let members: Vec<Member> = (0..8).map(|i| Member { way: way[i], dir: dir[i], seq: seq[i], time: time[i] }).collect();
        assert_eq!(round_columns(&way, &dir, &seq, &|r| time[r]), round_all(&members));
        let sorted: Vec<usize> = { let mut o: Vec<usize> = (0..8).collect(); o.sort_by_key(|&i| (way[i], i)); o };
        let (w2, d2, s2, t2): (Vec<i64>, Vec<u8>, Vec<u32>, Vec<f32>) = (
            sorted.iter().map(|&i| way[i]).collect(), sorted.iter().map(|&i| dir[i]).collect(),
            sorted.iter().map(|&i| seq[i]).collect(), sorted.iter().map(|&i| time[i]).collect(),
        );
        let m2: Vec<Member> = (0..8).map(|i| Member { way: w2[i], dir: d2[i], seq: s2[i], time: t2[i] }).collect();
        assert_eq!(round_columns(&w2, &d2, &s2, &|r| t2[r]), round_all(&m2));
    }
}
