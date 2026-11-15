//! The ACTIVE family's graph (SPEC §4, FORMAT §5.4): one node-based union graph of the ways
//! usable by bike or on foot, nodes at intersections, one road per direction of a piece that
//! at least one metric may use, with one weight per metric (`INF` where it may not).

use std::path::Path;

use anyhow::Result;

use crate::graph::components::{Components, Csr, scc};
use crate::graph::round::round_columns;
use crate::osm::column::{Column, ColumnMut, ColumnWriter};
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
    /// Per piece, its roads [along, against] or `NONE64`: road indices of the whole family.
    pub road_of_piece: Column<[u64; 2]>,
    /// Each node's smallest road key, which `shed` keeps.
    pub key: NodeKeys,
}

impl ActiveGraph {
    /// The union graph of the pieces `p`. With `dir`, its columns are written to files there as
    /// they are made, then mapped (SPEC §10, memory).
    pub fn build(p: &Pieces, dir: Option<&Path>) -> Result<ActiveGraph> {
        let m = p.time.len();
        let mut g = ActiveGraph { nodes: p.node_osm.len(), ..ActiveGraph::default() };
        let mut piece = ColumnWriter::<u32>::new(dir, "piece")?;
        let mut way = ColumnWriter::<i64>::new(dir, "way")?;
        let mut rdir = ColumnWriter::<u8>::new(dir, "dir")?;
        let mut seq = ColumnWriter::<u32>::new(dir, "seq")?;
        let mut tail = ColumnWriter::<u32>::new(dir, "tail")?;
        let mut head = ColumnWriter::<u32>::new(dir, "head")?;
        let mut time: Vec<ColumnWriter<f32>> = (0..m).map(|k| ColumnWriter::new(dir, &format!("time{k}"))).collect::<Result<_>>()?;
        let mut road_of_piece = ColumnWriter::<[u64; 2]>::new(dir, "road_of_piece")?;
        for i in 0..p.len() {
            let mut rp = [blobmap_model::NONE64; 2];
            for d in 0..2 {
                if (0..m).all(|k| p.time[k][i][d].is_infinite()) {
                    continue;
                }
                rp[d] = piece.len() as u64;
                piece.push(i as u32)?;
                way.push(p.way[i])?;
                rdir.push(d as u8)?;
                seq.push(p.seq[i])?;
                let (a, b) = if d == 0 { (p.u[i], p.v[i]) } else { (p.v[i], p.u[i]) };
                tail.push(a)?;
                head.push(b)?;
                for (k, t) in time.iter_mut().enumerate() {
                    t.push(p.time[k][i][d])?;
                }
            }
            road_of_piece.push(rp)?;
        }
        (g.piece, g.way, g.dir, g.seq) = (piece.finish()?, way.finish()?, rdir.finish()?, seq.finish()?);
        (g.tail, g.head, g.road_of_piece) = (tail.finish()?, head.finish()?, road_of_piece.finish()?);
        g.time = time.into_iter().map(ColumnWriter::finish).collect::<Result<_>>()?;
        for k in 0..m {
            let mut w = Column::from(round_columns(&g.way, &g.dir, &g.seq, &|r| g.time[k][r]));
            w.spill(dir, &format!("w{k}"))?;
            g.w.push(w);
        }
        // Each node's smallest road key (FORMAT §4).
        let n = g.nodes;
        let (mut kw, mut ks, mut kd) = (
            ColumnMut::filled(dir, "key_way", n, i64::MAX)?,
            ColumnMut::filled(dir, "key_seq", n, u32::MAX)?,
            ColumnMut::filled(dir, "key_dir", n, u8::MAX)?,
        );
        for r in 0..g.len() {
            let k = (g.way[r], g.seq[r], g.dir[r]);
            for v in [g.tail[r] as usize, g.head[r] as usize] {
                if k < (kw[v], ks[v], kd[v]) {
                    (kw[v], ks[v], kd[v]) = k;
                }
            }
        }
        g.key = NodeKeys { way: kw.freeze()?, seq: ks.freeze()?, dir: kd.freeze()? };
        Ok(g)
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
    pub fn components(&self, dir: Option<&Path>) -> Result<Components> {
        let arcs = self.tail.iter().copied().zip(self.head.iter().copied());
        let g = Csr::build(self.nodes, arcs.clone(), dir)?;
        let comp = scc(&g);
        drop(g);
        // Each component's smallest key over the roads at its nodes; (MAX, ..) for a node
        // without roads.
        let k = comp.iter().map(|&c| c as usize + 1).max().unwrap_or(0);
        let mut best = vec![Some((i64::MAX, u32::MAX, u8::MAX, u8::MAX)); k];
        for r in 0..self.len() {
            for (end, v) in [(0u8, self.tail[r]), (1, self.head[r])] {
                let key = (self.way[r], self.seq[r], self.dir[r], end);
                let b = &mut best[comp[v as usize] as usize];
                if Some(key) < *b {
                    *b = Some(key);
                }
            }
        }
        Ok(Components::with_best(&comp, &best, arcs))
    }
}

/// Each node's smallest (way, way_seq, way_dir) over its roads, `(MAX, MAX, MAX)` for none.
#[derive(Clone, Debug, Default)]
pub struct NodeKeys {
    pub way: Column<i64>,
    pub seq: Column<u32>,
    pub dir: Column<u8>,
}

impl NodeKeys {
    pub fn of(&self, v: usize) -> (i64, u32, u8) {
        (self.way[v], self.seq[v], self.dir[v])
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
        let g = ActiveGraph::build(&net.pieces(&modes).unwrap(), None).unwrap();
        // Walk 100 m at 5 km/h: 72 s on every piece of row 0 (way 1).
        let row0: Vec<u16> = (0..g.len()).filter(|&r| g.way[r] == 1).map(|r| g.w[1][r]).collect();
        assert_eq!(row0, [72; 10]);
        // Roads that walk may use and bike may not: the footbridge and the walk trap.
        assert!((0..g.len()).any(|r| g.w[0][r] == blobmap_model::INF && g.w[1][r] != blobmap_model::INF));
        let e = g.components(None).unwrap();
        assert!(!e.trapped[0] && e.trapped.iter().any(|&t| t));
    }
}
