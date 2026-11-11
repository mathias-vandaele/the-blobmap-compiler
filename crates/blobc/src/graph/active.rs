//! The ACTIVE family's graph (SPEC §4, FORMAT §5.4): one node-based union graph of the ways
//! usable by bike or on foot, nodes at intersections, one road per direction of a piece that
//! at least one metric may use, with one weight per metric (`INF` where it may not).

use std::path::Path;

use anyhow::Result;

use crate::graph::components::{Csr, Components, scc};
use crate::graph::round::{Member, round_all};
use crate::osm::column::Column;
use crate::osm::Pieces;

/// The ACTIVE metrics, in the manifest's order (FORMAT §3 `families`).
pub const METRICS: [&str; 2] = ["bike", "walk"];

#[derive(Clone, Debug, Default)]
pub struct ActiveGraph {
    pub nodes: usize,
    /// Per road, in (piece, dir) order.
    pub piece: Column<u32>,
    pub way: Column<i64>,
    pub dir: Column<u8>,
    pub seq: Column<u32>,
    pub tail: Column<u32>,
    pub head: Column<u32>,
    /// Per metric, per road.
    pub time: Vec<Column<f32>>,
    pub w: Vec<Column<u16>>,
    /// Per piece, its roads [along, against] or `NONE`.
    pub road_of_piece: Column<[u32; 2]>,
}

impl ActiveGraph {
    pub fn build(p: &Pieces) -> ActiveGraph {
        let m = p.time.len();
        let mut g = ActiveGraph {
            nodes: p.node_osm.len(),
            time: vec![Column::default(); m],
            ..ActiveGraph::default()
        };
        let mut road_of_piece = vec![[blobmap_format::NONE; 2]; p.len()];
        for i in 0..p.len() {
            for d in 0..2 {
                if (0..m).all(|k| p.time[k][i][d].is_infinite()) {
                    continue;
                }
                road_of_piece[i][d] = g.piece.len() as u32;
                g.piece.push(i as u32);
                g.way.push(p.way[i]);
                g.dir.push(d as u8);
                g.seq.push(p.seq[i]);
                let (a, b) = if d == 0 { (p.u[i], p.v[i]) } else { (p.v[i], p.u[i]) };
                g.tail.push(a);
                g.head.push(b);
                for k in 0..m {
                    g.time[k].push(p.time[k][i][d]);
                }
            }
        }
        g.road_of_piece = road_of_piece.into();
        g.w = (0..m)
            .map(|k| {
                let members: Vec<Member> =
                    (0..g.len()).map(|r| Member { way: g.way[r], dir: g.dir[r], seq: g.seq[r], time: g.time[k][r] }).collect();
                round_all(&members).into()
            })
            .collect();
        g
    }

    pub fn len(&self) -> usize {
        self.piece.len()
    }

    /// Drops what only the order, the hierarchy and the road part read, once the family's road
    /// units are written: the vocabulary and the data report read the rest.
    /// Moves the graph's arrays to files of `dir`, mapped (SPEC §10, memory).
    pub fn spill(&mut self, dir: &Path) -> Result<()> {
        let d = Some(dir);
        self.piece.spill(d, "piece")?;
        self.way.spill(d, "way")?;
        self.dir.spill(d, "dir")?;
        self.seq.spill(d, "seq")?;
        self.tail.spill(d, "tail")?;
        self.head.spill(d, "head")?;
        for (k, c) in self.time.iter_mut().enumerate() {
            c.spill(d, &format!("time{k}"))?;
        }
        for (k, c) in self.w.iter_mut().enumerate() {
            c.spill(d, &format!("w{k}"))?;
        }
        self.road_of_piece.spill(d, "road_of_piece")?;
        Ok(())
    }

    pub fn shed(&mut self) {
        (self.tail, self.head) = Default::default();
        (self.way, self.seq, self.dir) = Default::default();
    }

    /// The components of the union graph (SPEC §5). Ties go to the component with the smallest
    /// (way, way_seq, way_dir, end) over the roads at its nodes, `end` 0 for the road's tail
    /// and 1 for its head.
    pub fn components(&self) -> Components {
        let g = Csr::new(self.nodes, self.tail.iter().copied().zip(self.head.iter().copied()));
        let comp = scc(&g);
        let mut key = vec![(i64::MAX, u32::MAX, u8::MAX, u8::MAX); self.nodes];
        for r in 0..self.len() {
            for (end, v) in [(0u8, self.tail[r]), (1, self.head[r])] {
                let k = (self.way[r], self.seq[r], self.dir[r], end);
                if k < key[v as usize] {
                    key[v as usize] = k;
                }
            }
        }
        Components::new(&comp, |v| key[v], self.tail.iter().copied().zip(self.head.iter().copied()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile::active::{ActiveMode, ActiveProfile, Mode};

    #[test]
    fn town_union_graph() {
        let dir = std::env::temp_dir().join(format!("blobmap-graph-active-{}", std::process::id()));
        let w = crate::fixtures::by_name("town").unwrap().write(&dir).unwrap();
        let cfg = crate::profile::config::Config::load(&w.configs[0].1).unwrap();
        let prof = ActiveProfile::new(&cfg.active.config);
        let osm = crate::osm::Osm::open(&w.pbf, &dir.join("nodes"), &cfg).unwrap();
        let net = crate::osm::Network::read(&osm, &osm.active_obstacles, &prof, &cfg.active.config.snap, None).unwrap();
        let modes: Vec<ActiveMode> = [Mode::Bike, Mode::Walk].map(|mode| ActiveMode { mode, cfg: prof.cfg }).to_vec();
        let g = ActiveGraph::build(&net.pieces(&modes).unwrap());
        // Walk 100 m at 5 km/h: 72 s on every piece of row 0 (way 1).
        let row0: Vec<u16> = (0..g.len()).filter(|&r| g.way[r] == 1).map(|r| g.w[1][r]).collect();
        assert_eq!(row0, [72; 10]);
        // Roads that walk may use and bike may not: the footbridge and the walk trap.
        assert!((0..g.len()).any(|r| g.w[0][r] == blobmap_format::INF && g.w[1][r] != blobmap_format::INF));
        let e = g.components();
        assert!(!e.trapped[0] && e.trapped.iter().any(|&t| t));
    }
}
