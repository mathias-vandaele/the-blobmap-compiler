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

/// The parent of each rank in the elimination tree, without the fill (Liu's algorithm with
/// path compression): O(n + m) memory, for a graph too large to eliminate at once.
/// `edges` are rank pairs in any direction and multiplicity.
pub fn etree(n: usize, edges: impl Iterator<Item = (u32, u32)>) -> Vec<Option<u32>> {
    const NONE: u32 = u32::MAX;
    // The lower neighbours of each rank, CSR.
    let pairs: Vec<(u32, u32)> = edges.filter(|(a, b)| a != b).map(|(a, b)| (a.max(b), a.min(b))).collect();
    let mut first = vec![0u32; n + 1];
    for &(hi, _) in &pairs {
        first[hi as usize + 1] += 1;
    }
    for i in 0..n {
        first[i + 1] += first[i];
    }
    let mut at = first.clone();
    let mut lower = vec![0u32; pairs.len()];
    for (hi, lo) in pairs {
        lower[at[hi as usize] as usize] = lo;
        at[hi as usize] += 1;
    }
    let (mut parent, mut anc) = (vec![NONE; n], vec![NONE; n]);
    for i in 0..n as u32 {
        for &k in &lower[first[i as usize] as usize..first[i as usize + 1] as usize] {
            // Up from k to the root of its current subtree, pointing the path at i.
            let mut r = k;
            while anc[r as usize] != NONE && anc[r as usize] != i {
                let t = anc[r as usize];
                anc[r as usize] = i;
                r = t;
            }
            if anc[r as usize] == NONE {
                anc[r as usize] = i;
                parent[r as usize] = i;
            }
        }
    }
    parent.into_iter().map(|p| (p != NONE).then_some(p)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn etree_matches_elimination() {
        // A 12 × 12 grid in a scrambled order: the parent is the lowest upper neighbour.
        let k = 12u32;
        let n = (k * k) as usize;
        let rank: Vec<u32> = (0..n as u32).map(|v| (v * 37) % n as u32).collect();
        let mut edges = Vec::new();
        for v in 0..k * k {
            if v % k + 1 < k {
                edges.push((rank[v as usize], rank[v as usize + 1]));
            }
            if v + k < k * k {
                edges.push((rank[v as usize], rank[(v + k) as usize]));
            }
        }
        let e = eliminate(n, edges.iter().copied());
        let want: Vec<Option<u32>> = e.up.iter().map(|u| u.first().copied()).collect();
        assert_eq!(etree(n, edges.iter().copied()), want);
    }

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
