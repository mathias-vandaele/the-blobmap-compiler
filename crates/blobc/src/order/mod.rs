//! Node orders and cells (SPEC §5). An `Orderer` gives a contraction order; `plan` decides
//! whether a component is built in one piece or as cells, and orders each block.

pub mod cells;
pub mod elim;

pub use cells::{Block, Kind, plan};

/// A graph to order: arcs in any direction (the order treats them as undirected) and a
/// coordinate per node.
pub struct Graph<'a> {
    pub n: usize,
    pub tail: &'a [u32],
    pub head: &'a [u32],
    pub lat: &'a [f32],
    pub lon: &'a [f32],
}

/// A contraction order: `order[rank]` is the node of that rank, rank 0 contracted first.
pub trait Orderer: Sync {
    fn order(&self, g: &Graph) -> Vec<u32>;
}

/// Nested dissection with inertial-flow separators (RoutingKit, vendored: its recursion runs
/// as OpenMP tasks and gives the sequential order bit for bit, whatever OMP_NUM_THREADS).
pub struct InertialFlow;

impl Orderer for InertialFlow {
    fn order(&self, g: &Graph) -> Vec<u32> {
        if g.n <= 2 || g.tail.is_empty() {
            return (0..g.n as u32).collect();
        }
        routingkit_cch::compute_order_inertial(g.n as u32, g.tail, g.head, g.lat, g.lon)
    }
}

/// A hand-made order (the toy town): node ranks given in full.
pub struct Fixed(pub Vec<u32>);

impl Orderer for Fixed {
    fn order(&self, g: &Graph) -> Vec<u32> {
        assert_eq!(self.0.len(), g.n, "a fixed order names every node");
        self.0.clone()
    }
}

/// `rank[node]` from `order[rank]`.
pub fn ranks(order: &[u32]) -> Vec<u32> {
    let mut rank = vec![u32::MAX; order.len()];
    for (r, &v) in order.iter().enumerate() {
        rank[v as usize] = r as u32;
    }
    rank
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inertial_flow_is_a_permutation() {
        // A 10 × 10 grid plus an isolated pair.
        let (mut tail, mut head) = (Vec::new(), Vec::new());
        for i in 0..10u32 {
            for j in 0..10u32 {
                let v = i * 10 + j;
                if j < 9 {
                    tail.push(v);
                    head.push(v + 1);
                }
                if i < 9 {
                    tail.push(v);
                    head.push(v + 10);
                }
            }
        }
        tail.push(100);
        head.push(101);
        let lat: Vec<f32> = (0..102).map(|v| (v / 10) as f32 * 0.001).collect();
        let lon: Vec<f32> = (0..102).map(|v| (v % 10) as f32 * 0.001).collect();
        let g = Graph { n: 102, tail: &tail, head: &head, lat: &lat, lon: &lon };
        let o = InertialFlow.order(&g);
        let mut s = o.clone();
        s.sort();
        assert_eq!(s, (0..102).collect::<Vec<u32>>());
        assert_eq!(o, InertialFlow.order(&g));
    }
}
