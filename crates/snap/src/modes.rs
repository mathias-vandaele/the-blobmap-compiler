//! The three sets of SPEC §6 for each mode, from a family's pieces, graph and experts.
//!
//! - **graph:** the family's graph (NoRoad when none is in the radius): for CAR, the pieces
//!   with a segment; for ACTIVE, the pieces with a road.
//! - **eligible:** pieces of the mode's graph that pass the profile's tag rules. A mode's
//!   graph is the pieces with a road it may use (SPEC §6): a finite master time, which is
//!   a finite weight since long pieces are cut (SPEC §7).
//! - **servable:** eligible pieces in an expert that is not trapped. An ACTIVE piece whose
//!   two ends lie in different experts never lies on a route (FORMAT §5.4), so it is skipped
//!   like a trapped one.

use blobmap_format::NONE;
use blobmap_graph::{ActiveGraph, CarGraph, Experts};
use blobmap_osm::Pieces;

use crate::Sets;

pub struct Membership {
    graph: Vec<bool>,
    eligible: Vec<bool>,
    servable: Vec<bool>,
}

impl Sets for Membership {
    fn in_graph(&self, piece: u32) -> bool {
        self.graph[piece as usize]
    }
    fn eligible(&self, piece: u32) -> bool {
        self.eligible[piece as usize]
    }
    fn servable(&self, piece: u32) -> bool {
        self.servable[piece as usize]
    }
}

/// The CAR expert an ID on `piece` is served in: the smallest index among the experts of
/// the piece's segments that are not trapped (FORMAT §7.2), or None.
pub fn car_expert(g: &CarGraph, e: &Experts, piece: usize) -> Option<u32> {
    g.seg_of_piece[piece].iter().filter(|&&s| s != NONE).map(|&s| e.of[s as usize]).filter(|&x| !e.trapped[x as usize]).min()
}

/// The ACTIVE expert of `piece`, if both its ends are in the same expert and it is not
/// trapped.
pub fn active_expert(p: &Pieces, e: &Experts, piece: usize) -> Option<u32> {
    let (a, b) = (e.of[p.u[piece] as usize], e.of[p.v[piece] as usize]);
    (a == b && !e.trapped[a as usize]).then_some(a)
}

impl Membership {
    pub fn car(p: &Pieces, g: &CarGraph, e: &Experts) -> Membership {
        let graph: Vec<bool> = g.seg_of_piece.iter().map(|s| s.iter().any(|&x| x != NONE)).collect();
        let eligible: Vec<bool> = (0..p.len()).map(|k| graph[k] && p.eligible[k]).collect();
        let servable = (0..p.len()).map(|k| eligible[k] && car_expert(g, e, k).is_some()).collect();
        Membership { graph, eligible, servable }
    }

    /// For ACTIVE metric `m` (an index into `blobmap_graph::active::METRICS`).
    pub fn active(p: &Pieces, g: &ActiveGraph, e: &Experts, m: usize) -> Membership {
        let graph: Vec<bool> = g.road_of_piece.iter().map(|r| r.iter().any(|&x| x != NONE)).collect();
        let usable = |k: usize| g.road_of_piece[k].iter().any(|&r| r != NONE && g.time[m][r as usize].is_finite());
        let eligible: Vec<bool> = (0..p.len()).map(|k| graph[k] && p.eligible[k] && usable(k)).collect();
        let servable = (0..p.len()).map(|k| eligible[k] && active_expert(p, e, k).is_some()).collect();
        Membership { graph, eligible, servable }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Grid, NO_ELIGIBLE_ROAD, NO_ROAD, TRAPPED_COMPONENT};
    use blobmap_fixtures::Check;
    use blobmap_profile::active::{ActiveMode, ActiveProfile, Mode};
    use blobmap_profile::profile::{CarProfile, WayAttrs};

    #[test]
    fn town_reasons() {
        let s = blobmap_fixtures::by_name("town").unwrap();
        let dir = std::env::temp_dir().join(format!("blobmap-snap-{}", std::process::id()));
        let w = s.write(&dir).unwrap();
        let cfg = blobmap_profile::config::Config::load(&w.configs[0].1).unwrap();
        let car_net = blobmap_osm::Network::read(&w.pbf, &CarProfile::new(&cfg.car.config), &cfg.car.config.snap).unwrap();
        let cp = car_net.pieces(&[|a: &WayAttrs| *a]);
        let (r, _) = blobmap_osm::restrictions::read(&w.pbf).unwrap();
        let cg = CarGraph::build(&cp, &cfg.car.config.turns, &r);
        let ce = cg.experts();
        let prof = ActiveProfile::new(&cfg.active.config);
        let act_net = blobmap_osm::Network::read(&w.pbf, &prof, &cfg.active.config.snap).unwrap();
        let modes: Vec<ActiveMode> = [Mode::Bike, Mode::Walk].map(|mode| ActiveMode { mode, cfg: prof.cfg }).to_vec();
        let ap = act_net.pieces(&modes);
        let ag = ActiveGraph::build(&ap);
        let ae = ag.experts();
        let (cgrid, agrid) = (Grid::new(&cp), Grid::new(&ap));
        let car = Membership::car(&cp, &cg, &ce);
        let bike = Membership::active(&ap, &ag, &ae, 0);
        let walk = Membership::active(&ap, &ag, &ae, 1);
        let at = |id: &str| s.release.addresses.iter().find(|a| a.id == id).map(|a| (a.lat, a.lon));
        let mut seen = 0;
        for c in &s.checks {
            let (id, mode, want) = match c {
                Check::Reason { id, mode, reason } => (id, *mode, Some(*reason)),
                Check::Served { id, mode } => (id, *mode, None),
                _ => continue,
            };
            let Some((lat, lon)) = at(id) else { continue };
            let f = match mode {
                "car" => cgrid.find(&cp, lat, lon, cfg.car.config.snap.radius_m as f64, &car),
                "bike" => agrid.find(&ap, lat, lon, cfg.active.config.snap.radius_m as f64, &bike),
                _ => agrid.find(&ap, lat, lon, cfg.active.config.snap.radius_m as f64, &walk),
            };
            let got = f.outcome().err().map(|(code, _)| match code {
                NO_ELIGIBLE_ROAD => "NoEligibleRoad",
                NO_ROAD => "NoRoad",
                TRAPPED_COMPONENT => "TrappedComponent",
                _ => "?",
            });
            assert_eq!(got, want, "{id} {mode}");
            seen += 1;
        }
        assert!(seen >= 8, "{seen}");
    }
}
