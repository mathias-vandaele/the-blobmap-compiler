//! Components (SPEC §5, FORMAT §4): the strongly connected components of a family's graph,
//! numbered by node count descending, ties to the smallest road key; and which of them are
//! trapped: in each group of components joined by roads ignoring direction, the largest is
//! served and the others are trapped, ties to the smaller component index.

/// A directed graph as CSR; arc positions are u64, as a family's arcs may pass 2^32.
pub struct Csr {
    pub first: Vec<u64>,
    pub head: Vec<u32>,
}

impl Csr {
    pub fn new(n: usize, arcs: impl Iterator<Item = (u32, u32)> + Clone) -> Csr {
        let mut first = vec![0u64; n + 1];
        for (t, _) in arcs.clone() {
            first[t as usize + 1] += 1;
        }
        for i in 0..n {
            first[i + 1] += first[i];
        }
        let mut pos = first.clone();
        let mut head = vec![0u32; first[n] as usize];
        for (t, h) in arcs {
            head[pos[t as usize] as usize] = h;
            pos[t as usize] += 1;
        }
        Csr { first, head }
    }

    pub fn len(&self) -> usize {
        self.first.len() - 1
    }

    pub fn out(&self, v: usize) -> &[u32] {
        &self.head[self.first[v] as usize..self.first[v + 1] as usize]
    }
}

/// Strongly connected component per node (iterative Tarjan, ported from the old `graph`).
/// Component ids are in order of completion.
pub fn scc(g: &Csr) -> Vec<u32> {
    const NONE: u32 = u32::MAX;
    let n = g.len();
    let (mut index, mut low, mut on_stack, mut comp) = (vec![NONE; n], vec![0u32; n], vec![false; n], vec![NONE; n]);
    let (mut stack, mut call): (Vec<u32>, Vec<(u32, u64)>) = (Vec::new(), Vec::new());
    let (mut next_index, mut next_comp) = (0u32, 0u32);
    for root in 0..n as u32 {
        if index[root as usize] != NONE {
            continue;
        }
        call.push((root, g.first[root as usize]));
        index[root as usize] = next_index;
        low[root as usize] = next_index;
        next_index += 1;
        stack.push(root);
        on_stack[root as usize] = true;
        while let Some(&mut (v, ref mut pos)) = call.last_mut() {
            let vi = v as usize;
            if *pos < g.first[vi + 1] {
                let w = g.head[*pos as usize];
                *pos += 1;
                let wi = w as usize;
                if index[wi] == NONE {
                    index[wi] = next_index;
                    low[wi] = next_index;
                    next_index += 1;
                    stack.push(w);
                    on_stack[wi] = true;
                    call.push((w, g.first[wi]));
                } else if on_stack[wi] {
                    low[vi] = low[vi].min(index[wi]);
                }
            } else {
                call.pop();
                if let Some(&(p, _)) = call.last() {
                    low[p as usize] = low[p as usize].min(low[vi]);
                }
                if low[vi] == index[vi] {
                    loop {
                        let w = stack.pop().unwrap();
                        on_stack[w as usize] = false;
                        comp[w as usize] = next_comp;
                        if w == v {
                            break;
                        }
                    }
                    next_comp += 1;
                }
            }
        }
    }
    comp
}

/// Union-find with path halving.
pub struct Dsu(Vec<u32>);

impl Dsu {
    pub fn new(n: usize) -> Dsu {
        Dsu((0..n as u32).collect())
    }

    pub fn find(&mut self, mut x: u32) -> u32 {
        while self.0[x as usize] != x {
            let p = self.0[self.0[x as usize] as usize];
            self.0[x as usize] = p;
            x = p;
        }
        x
    }

    pub fn union(&mut self, a: u32, b: u32) {
        let (a, b) = (self.find(a), self.find(b));
        if a != b {
            let (lo, hi) = (a.min(b), a.max(b));
            self.0[hi as usize] = lo;
        }
    }
}

/// A family's components.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Components {
    /// Component index of each node.
    pub of: Vec<u32>,
    /// Node count of each component (descending).
    pub nodes: Vec<u32>,
    pub trapped: Vec<bool>,
    /// Weak group of each component, numbered by its smallest component.
    pub group: Vec<u32>,
}

impl Components {
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Numbers the components (`comp`, any ids) and finds the trapped ones. `key[v]` orders
    /// nodes for the tie-break (its smallest key over a component's nodes); `joined` lists
    /// node pairs that a road joins, ignoring direction.
    pub fn new<K: Ord + Copy>(comp: &[u32], key: impl Fn(usize) -> K, joined: impl Iterator<Item = (u32, u32)>) -> Components {
        let k = comp.iter().map(|&c| c as usize + 1).max().unwrap_or(0);
        let mut size = vec![0u32; k];
        let mut best: Vec<Option<K>> = vec![None; k];
        for (v, &c) in comp.iter().enumerate() {
            size[c as usize] += 1;
            let kv = key(v);
            let b = &mut best[c as usize];
            if b.is_none_or(|x| kv < x) {
                *b = Some(kv);
            }
        }
        let mut by: Vec<usize> = (0..k).filter(|&c| size[c] > 0).collect();
        by.sort_by(|&a, &b| size[b].cmp(&size[a]).then_with(|| best[a].cmp(&best[b])));
        let mut index = vec![u32::MAX; k];
        for (i, &c) in by.iter().enumerate() {
            index[c] = i as u32;
        }
        let of: Vec<u32> = comp.iter().map(|&c| index[c as usize]).collect();
        let nodes: Vec<u32> = by.iter().map(|&c| size[c]).collect();
        let mut dsu = Dsu::new(nodes.len());
        for (a, b) in joined {
            dsu.union(of[a as usize], of[b as usize]);
        }
        // The root of each group is its smallest component index, which is its largest
        // component (ties to the smaller index): it is served, every other one is trapped.
        let group: Vec<u32> = (0..nodes.len() as u32).map(|e| dsu.find(e)).collect();
        let trapped: Vec<bool> = group.iter().enumerate().map(|(e, &g)| g != e as u32).collect();
        Components { of, nodes, trapped, group }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn components_numbering_and_traps() {
        // 0 <-> 1 <-> 2, 2 -> 3, 3 <-> 4 (trapped pair), 5 <-> 6 (island of the same size).
        let arcs = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 4), (4, 3), (5, 6), (6, 5)];
        let g = Csr::new(7, arcs.iter().copied());
        let comp = scc(&g);
        let keys = [10, 11, 12, 13, 14, 1, 2];
        let e = Components::new(&comp, |v| keys[v], arcs.iter().copied());
        assert_eq!(e.nodes, [3, 2, 2]);
        // Ties by smallest key: the island (key 1) before the pocket (key 13).
        assert_eq!(e.of, [0, 0, 0, 2, 2, 1, 1]);
        assert_eq!(e.trapped, [false, false, true]);
        assert_eq!(e.group, [0, 1, 0]);
    }
}
