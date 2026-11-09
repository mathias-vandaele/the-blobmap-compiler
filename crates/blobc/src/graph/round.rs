//! Integer road weights (SPEC §7, FORMAT §6.1): each (way, direction, metric) group's total
//! rounded once, distributed by largest remainder, ties in travel order.

use blobmap_format::{INF, rnd};

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
    let mut idx: Vec<u32> = (0..roads.len() as u32).filter(|&i| roads[i as usize].time.is_finite()).collect();
    let key = |i: u32| {
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
}
