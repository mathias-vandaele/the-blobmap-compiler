//! What the road part needs from a mode family (SPEC §4, FORMAT §5): its nodes, its arcs
//! (CAR turns, ACTIVE roads) with their link weights, and the family's own tensors. Both
//! families store their arcs by the row of their tail, so packs are written the same way
//! for both (`road.rs`); only the tensors named here differ.

use blobmap_format::{NONE, Tensor, Unit, sat};
use blobmap_graph::active::METRICS;
use blobmap_graph::{ActiveGraph, CarGraph, Experts};
use blobmap_osm::Pieces;

pub trait Family: Sync {
    /// `car` or `active`.
    fn name(&self) -> &'static str;
    /// FORMAT §3 `families`: `edge` or `node`.
    fn graph_kind(&self) -> &'static str;
    fn metrics(&self) -> Vec<String>;
    fn experts(&self) -> &Experts;
    fn nodes(&self) -> usize;
    /// A coordinate for the node order (SPEC §5); never stored.
    fn coord(&self, v: u32) -> (f32, f32);
    fn arc_count(&self) -> usize;
    /// Tail and head node of arc a.
    fn arc(&self, a: usize) -> (u32, u32);
    /// The weight arc a gives its link for metric m (FORMAT §6.2).
    fn arc_weight(&self, m: usize, a: usize) -> u16;
    /// Order of the arcs of one row after their head's layout id (FORMAT §5.3, §5.4).
    fn arc_key(&self, a: usize) -> (i64, u32, u8);
    /// `turn` or `road`: the arc tensors are `<word>_first`, `<word>_head`, `<word>_link`.
    fn arc_word(&self) -> &'static str;
    /// The family's per-row tensors of a pack, for its rows (nodes in row order).
    fn row_tensors(&self, rows: &[u32], layout_of: &[u32], unit: &mut Unit, prefix: &str);
    /// The family's per-arc tensors of a pack, for its arcs in stored order.
    fn arc_tensors(&self, arcs: &[u32], unit: &mut Unit, prefix: &str);
    /// Extra family tables (FORMAT §5.4 `road/active/cross/`).
    fn table_tensors(&self, unit: &mut Unit);
    /// The smallest road key (way, way_seq, way_dir) at node v (FORMAT §4).
    fn road_key(&self, v: u32) -> (i64, u32, u8);
}

pub struct Car<'a> {
    pub p: &'a Pieces,
    pub g: &'a CarGraph,
    pub e: Experts,
    /// Copies of each original, as nodes, CSR by node.
    copy_first: Vec<u32>,
    copies: Vec<u32>,
}

impl<'a> Car<'a> {
    pub fn new(p: &'a Pieces, g: &'a CarGraph) -> Car<'a> {
        let e = g.experts();
        let mut pairs: Vec<(u32, u32)> = (g.originals..g.len()).map(|c| (g.copy_of[c], c as u32)).collect();
        pairs.sort_unstable();
        let mut copy_first = vec![0u32; g.len() + 1];
        for &(o, _) in &pairs {
            copy_first[o as usize + 1] += 1;
        }
        for i in 0..g.len() {
            copy_first[i + 1] += copy_first[i];
        }
        Car { p, g, e, copy_first, copies: pairs.into_iter().map(|x| x.1).collect() }
    }
}

impl Family for Car<'_> {
    fn name(&self) -> &'static str {
        "car"
    }
    fn graph_kind(&self) -> &'static str {
        "edge"
    }
    fn metrics(&self) -> Vec<String> {
        vec!["car".into()]
    }
    fn experts(&self) -> &Experts {
        &self.e
    }
    fn nodes(&self) -> usize {
        self.g.len()
    }
    fn coord(&self, v: u32) -> (f32, f32) {
        let (a, b) = (self.g.tail[v as usize] as usize, self.g.head[v as usize] as usize);
        (((self.p.node_lat[a] + self.p.node_lat[b]) / 2.0) as f32, ((self.p.node_lon[a] + self.p.node_lon[b]) / 2.0) as f32)
    }
    fn arc_count(&self) -> usize {
        self.g.turn_from.len()
    }
    fn arc(&self, a: usize) -> (u32, u32) {
        (self.g.turn_from[a], self.g.turn_to[a])
    }
    fn arc_weight(&self, _: usize, a: usize) -> u16 {
        sat(self.g.w[self.g.turn_from[a] as usize], self.g.turn_cost[a])
    }
    fn arc_key(&self, _: usize) -> (i64, u32, u8) {
        (0, 0, 0)
    }
    fn arc_word(&self) -> &'static str {
        "turn"
    }
    fn row_tensors(&self, rows: &[u32], layout_of: &[u32], unit: &mut Unit, prefix: &str) {
        let g = self.g;
        let at = |f: &dyn Fn(usize) -> u32| rows.iter().map(|&v| f(v as usize)).collect::<Vec<u32>>();
        let same = |a: usize, b: usize| self.e.of[a] == self.e.of[b];
        unit.put_in(prefix, "car.time_s", Tensor::vec(&rows.iter().map(|&v| g.time[v as usize]).collect::<Vec<f32>>()));
        unit.put_in(prefix, "car.w", Tensor::vec(&rows.iter().map(|&v| g.w[v as usize]).collect::<Vec<u16>>()));
        unit.put_in(prefix, "way", Tensor::vec(&rows.iter().map(|&v| g.way[v as usize]).collect::<Vec<i64>>()));
        unit.put_in(prefix, "way_dir", Tensor::vec(&rows.iter().map(|&v| g.dir[v as usize]).collect::<Vec<u8>>()));
        unit.put_in(prefix, "way_seq", Tensor::vec(&at(&|v| g.seq[v])));
        unit.put_in(prefix, "kind", Tensor::vec(&rows.iter().map(|&v| if (v as usize) < g.originals { 0u8 } else { 2 }).collect::<Vec<u8>>()));
        unit.put_in(
            prefix,
            "copy_of",
            Tensor::vec(&at(&|v| {
                let o = g.copy_of[v];
                if o != NONE && same(v, o as usize) { layout_of[o as usize] } else { NONE }
            })),
        );
        let (mut first, mut copies) = (vec![0u32], Vec::new());
        for &v in rows {
            let v = v as usize;
            let mut c: Vec<u32> = self.copies[self.copy_first[v] as usize..self.copy_first[v + 1] as usize]
                .iter()
                .filter(|&&c| same(v, c as usize))
                .map(|&c| layout_of[c as usize])
                .collect();
            c.sort_unstable();
            copies.extend(c);
            first.push(copies.len() as u32);
        }
        unit.put_in(prefix, "copy_first", Tensor::vec(&first));
        unit.put_in(prefix, "copies", Tensor::vec(&copies));
    }
    fn arc_tensors(&self, arcs: &[u32], unit: &mut Unit, prefix: &str) {
        unit.put_in(prefix, "car.turn_cost", Tensor::vec(&arcs.iter().map(|&a| self.g.turn_cost[a as usize]).collect::<Vec<u16>>()));
    }
    fn table_tensors(&self, _: &mut Unit) {}
    fn road_key(&self, v: u32) -> (i64, u32, u8) {
        let v = v as usize;
        (self.g.way[v], self.g.seq[v], self.g.dir[v])
    }
}

pub struct Active<'a> {
    pub p: &'a Pieces,
    pub g: &'a ActiveGraph,
    pub e: Experts,
    key: Vec<(i64, u32, u8)>,
}

impl<'a> Active<'a> {
    pub fn new(p: &'a Pieces, g: &'a ActiveGraph) -> Active<'a> {
        let e = g.experts();
        let mut key = vec![(i64::MAX, u32::MAX, u8::MAX); g.nodes];
        for r in 0..g.len() {
            let k = (g.way[r], g.seq[r], g.dir[r]);
            for v in [g.tail[r], g.head[r]] {
                key[v as usize] = key[v as usize].min(k);
            }
        }
        Active { p, g, e, key }
    }

    /// Whether road r has its ends in different experts (FORMAT §5.4).
    pub fn cross(&self, r: usize) -> bool {
        self.e.of[self.g.tail[r] as usize] != self.e.of[self.g.head[r] as usize]
    }
}

impl Family for Active<'_> {
    fn name(&self) -> &'static str {
        "active"
    }
    fn graph_kind(&self) -> &'static str {
        "node"
    }
    fn metrics(&self) -> Vec<String> {
        METRICS.iter().map(|m| m.to_string()).collect()
    }
    fn experts(&self) -> &Experts {
        &self.e
    }
    fn nodes(&self) -> usize {
        self.g.nodes
    }
    fn coord(&self, v: u32) -> (f32, f32) {
        (self.p.node_lat[v as usize] as f32, self.p.node_lon[v as usize] as f32)
    }
    fn arc_count(&self) -> usize {
        self.g.len()
    }
    fn arc(&self, a: usize) -> (u32, u32) {
        (self.g.tail[a], self.g.head[a])
    }
    fn arc_weight(&self, m: usize, a: usize) -> u16 {
        self.g.w[m][a]
    }
    fn arc_key(&self, a: usize) -> (i64, u32, u8) {
        (self.g.way[a], self.g.seq[a], self.g.dir[a])
    }
    fn arc_word(&self) -> &'static str {
        "road"
    }
    fn row_tensors(&self, _: &[u32], _: &[u32], _: &mut Unit, _: &str) {}
    fn arc_tensors(&self, arcs: &[u32], unit: &mut Unit, prefix: &str) {
        road_tensors(self.g, arcs, unit, prefix);
    }
    fn table_tensors(&self, unit: &mut Unit) {
        let g = self.g;
        let mut cross: Vec<u32> = (0..g.len()).filter(|&r| self.cross(r)).map(|r| r as u32).collect();
        cross.sort_by_key(|&r| (g.way[r as usize], g.dir[r as usize], g.seq[r as usize]));
        road_tensors(g, &cross, unit, "road/active/cross/");
    }
    fn road_key(&self, v: u32) -> (i64, u32, u8) {
        self.key[v as usize]
    }
}

fn road_tensors(g: &ActiveGraph, roads: &[u32], unit: &mut Unit, prefix: &str) {
    let at = |r: &u32| *r as usize;
    unit.put_in(prefix, "way", Tensor::vec(&roads.iter().map(|r| g.way[at(r)]).collect::<Vec<i64>>()));
    unit.put_in(prefix, "way_dir", Tensor::vec(&roads.iter().map(|r| g.dir[at(r)]).collect::<Vec<u8>>()));
    unit.put_in(prefix, "way_seq", Tensor::vec(&roads.iter().map(|r| g.seq[at(r)]).collect::<Vec<u32>>()));
    unit.put_in(prefix, "kind", Tensor::vec(&vec![0u8; roads.len()]));
    for (m, name) in METRICS.iter().enumerate() {
        unit.put_in(prefix, &format!("{name}.time_s"), Tensor::vec(&roads.iter().map(|r| g.time[m][at(r)]).collect::<Vec<f32>>()));
        unit.put_in(prefix, &format!("{name}.w"), Tensor::vec(&roads.iter().map(|r| g.w[m][at(r)]).collect::<Vec<u16>>()));
    }
}
