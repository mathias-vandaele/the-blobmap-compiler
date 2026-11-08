//! Elimination (chordal completion) in rank space, ported from the old `contract`: nodes are
//! eliminated in rank order and each one's upper neighbours, but its parent, are merged into
//! its parent's (the elimination-tree lemma). Nodes with a rank of `k` or more are external:
//! never eliminated. When a node's upper neighbours are all external, the clique they form
//! is returned instead, for whoever eliminates them later (SPEC §5, cells).

/// Merges the sorted `src` into the sorted `dst`, without duplicates.
pub fn merge_into(dst: &mut Vec<u32>, src: &[u32]) {
    if src.is_empty() {
        return;
    }
    let mut out = Vec::with_capacity(dst.len() + src.len());
    let (mut i, mut j) = (0, 0);
    while i < dst.len() && j < src.len() {
        match dst[i].cmp(&src[j]) {
            std::cmp::Ordering::Less => {
                out.push(dst[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                out.push(src[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                out.push(dst[i]);
                i += 1;
                j += 1;
            }
        }
    }
    out.extend_from_slice(&dst[i..]);
    out.extend_from_slice(&src[j..]);
    *dst = out;
}

/// The result of eliminating ranks `0 .. k`.
pub struct Eliminated {
    /// Upper neighbours of each eliminated rank, ascending by rank.
    pub up: Vec<Vec<u32>>,
    /// Edges among external ranks created by the elimination, each (lower, upper) once,
    /// sorted.
    pub fill: Vec<(u32, u32)>,
}

/// `edges` are rank pairs in any direction and multiplicity; ranks ≥ `k` are external.
pub fn eliminate(k: usize, edges: impl Iterator<Item = (u32, u32)>) -> Eliminated {
    let mut up: Vec<Vec<u32>> = vec![Vec::new(); k];
    for (a, b) in edges {
        let (lo, hi) = (a.min(b), a.max(b));
        if lo != hi && (lo as usize) < k {
            up[lo as usize].push(hi);
        }
    }
    for l in &mut up {
        l.sort_unstable();
        l.dedup();
    }
    let mut fill = Vec::new();
    for v in 0..k {
        let list = std::mem::take(&mut up[v]);
        if let Some(&p) = list.first() {
            if (p as usize) < k {
                merge_into(&mut up[p as usize], &list[1..]);
            } else {
                for (i, &a) in list.iter().enumerate() {
                    for &b in &list[i + 1..] {
                        fill.push((a, b));
                    }
                }
            }
        }
        up[v] = list;
    }
    fill.sort_unstable();
    fill.dedup();
    Eliminated { up, fill }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_and_star() {
        // Path 0-1-2 eliminated in order: no fill.
        let e = eliminate(3, [(0, 1), (1, 2)].into_iter());
        assert_eq!(e.up, [vec![1], vec![2], vec![]]);
        // Star centre 0 first: its neighbours become a clique.
        let e = eliminate(4, [(0, 1), (0, 2), (0, 3)].into_iter());
        assert_eq!(e.up, [vec![1, 2, 3], vec![2, 3], vec![3], vec![]]);
        // The same with 1..4 external: the clique is returned as fill.
        let e = eliminate(1, [(0, 1), (0, 2), (0, 3)].into_iter());
        assert_eq!(e.fill, [(1, 2), (1, 3), (2, 3)]);
    }
}
