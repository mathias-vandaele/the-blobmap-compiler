//! The reference from the inputs: the original family graphs, their experts, the catalog,
//! and every ID snapped per mode (SPEC §6), computed with the compiler's input crates but
//! never from the model.

use std::path::PathBuf;

use anyhow::Result;
use blobmap_build::{ActiveNetwork, CarNetwork, ReadCatalog, Stage};
use blobmap_graph::{ActiveGraph, CarGraph, Experts, active::METRICS};
use blobmap_osm::Pieces;
use blobmap_overture::{Catalog, Id, Polygon, Release};
use blobmap_profile::config::Config;
use blobmap_snap::modes::{active_expert, car_expert};
use blobmap_format::NONE;
use blobmap_snap::{Grid, Membership, Snapped};
use rayon::prelude::*;

/// The inputs of a build.
#[derive(Clone, Debug)]
pub struct Inputs {
    pub pbf: PathBuf,
    pub overture: PathBuf,
    pub config: PathBuf,
    pub extent: Option<PathBuf>,
}

/// Where an ID stands for a mode.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Entry {
    pub expert: u32,
    pub piece: u32,
    pub frac: f64,
}

pub struct Mode {
    pub name: &'static str,
    pub car: bool,
    /// The metric's index in its family.
    pub metric: usize,
    /// Per ID (internal-index order): the entry, or (code, metres); code 4 for a filtered
    /// place.
    pub at: Vec<Result<Entry, (u8, f32)>>,
}

pub struct Reference {
    pub car_pieces: Pieces,
    pub car: CarGraph,
    pub car_experts: Experts,
    pub active_pieces: Pieces,
    pub active: ActiveGraph,
    pub active_experts: Experts,
    pub catalog: Catalog,
    /// Every ID of the model in internal-index order: addresses, then places (catalog
    /// index), each with its point.
    pub ids: Vec<Id>,
    pub place_of: Vec<Option<usize>>,
    /// (alias, internal index).
    pub aliases: Vec<(Id, u32)>,
    pub modes: Vec<Mode>,
    pub config: Config,
    /// Per ID (internal-index order): its point. Per mode: the snapping sets and radius.
    pub points: Vec<(f64, f64)>,
    pub sets: [Membership; 3],
    pub radius: [f64; 3],
    /// The IDs and the aliases, each sorted, with their internal index.
    sorted_ids: Vec<(Id, u32)>,
    sorted_aliases: Vec<(Id, u32)>,
}

impl Reference {
    pub fn build(i: &Inputs) -> Result<Reference> {
        let config = Config::load(&i.config)?;
        let (car_pieces, car) = CarNetwork.run((i.pbf.as_path(), &config))?;
        let (active_pieces, active) = ActiveNetwork.run((i.pbf.as_path(), &config))?;
        let (car_experts, active_experts) = (car.experts(), active.experts());
        let release = Release::open(&i.overture)?;
        let polygon = match &i.extent {
            Some(p) => Some(Polygon::parse_poly(&std::fs::read_to_string(p)?)?),
            None => None,
        };
        let catalog = ReadCatalog.run((&release, &config, polygon.as_ref()))?;
        let points: Vec<(f64, f64)> =
            catalog.addresses.iter().map(|a| (a.lat, a.lon)).chain(catalog.places.iter().map(|p| (p.lat, p.lon))).collect();
        let (cg, ag) = (Grid::new(&car_pieces), Grid::new(&active_pieces));
        let sets = [
            Membership::car(&car_pieces, &car, &car_experts),
            Membership::active(&active_pieces, &active, &active_experts, 0),
            Membership::active(&active_pieces, &active, &active_experts, 1),
        ];
        let radius = [config.car.config.snap.radius_m as f64, config.active.config.snap.radius_m as f64, config.active.config.snap.radius_m as f64];
        let found: Vec<Vec<Snapped>> = (0..3)
            .map(|m| {
                let (g, p) = if m == 0 { (&cg, &car_pieces) } else { (&ag, &active_pieces) };
                points.par_iter().map(|&(lat, lon)| Snapped::new(&g.find(p, lat, lon, radius[m], &sets[m]), NONE)).collect()
            })
            .collect();
        let na = catalog.addresses.len();
        let inside = |k: usize| polygon.is_some() || found.iter().any(|f| f[k].any);
        let mut pts: Vec<usize> = (0..na).filter(|&k| inside(k)).collect();
        let n_addr = pts.len();
        pts.extend((0..catalog.places.len()).filter(|&p| inside(na + p)).map(|p| na + p));
        let ids: Vec<Id> = pts.iter().map(|&k| if k < na { catalog.addresses[k].id } else { catalog.places[k - na].id }).collect();
        let place_of: Vec<Option<usize>> = pts.iter().map(|&k| (k >= na).then(|| k - na)).collect();
        let index_of_place = |p: u32| pts[n_addr..].binary_search(&(na + p as usize)).ok().map(|i| (n_addr + i) as u32);
        let aliases: Vec<(Id, u32)> = catalog.aliases.iter().filter_map(|&(a, p)| index_of_place(p).map(|i| (a, i))).collect();
        let names = ["car", "bike", "walk"];
        let modes = (0..3)
            .map(|m| {
                let at = pts
                    .iter()
                    .map(|&k| {
                        if k >= na && catalog.places[k - na].status != 0 {
                            return Err((4, 0.0));
                        }
                        let (piece, frac) = found[m][k].outcome()?;
                        let expert = if m == 0 {
                            car_expert(&car, &car_experts, piece as usize).expect("servable")
                        } else {
                            active_expert(&active_pieces, &active_experts, piece as usize).expect("servable")
                        };
                        Ok(Entry { expert, piece, frac })
                    })
                    .collect();
                Mode { name: names[m], car: m == 0, metric: if m == 0 { 0 } else { m - 1 }, at }
            })
            .collect();
        debug_assert_eq!(METRICS, ["bike", "walk"]);
        let points = pts.iter().map(|&k| points[k]).collect();
        let sorted = |v: Vec<(Id, u32)>| {
            let mut v = v;
            v.sort_unstable();
            v
        };
        let sorted_ids = sorted(ids.iter().enumerate().map(|(i, id)| (*id, i as u32)).collect());
        let sorted_aliases = sorted(aliases.to_vec());
        Ok(Reference {
            car_pieces,
            car,
            car_experts,
            active_pieces,
            active,
            active_experts,
            catalog,
            ids,
            place_of,
            aliases,
            modes,
            config,
            points,
            sets,
            radius,
            sorted_ids,
            sorted_aliases,
        })
    }

    pub fn n(&self) -> usize {
        self.ids.len()
    }

    pub fn mode(&self, name: &str) -> &Mode {
        self.modes.iter().find(|m| m.name == name).expect("mode")
    }

    /// FORMAT §7.1 lookup on the reference IDs.
    pub fn lookup(&self, id: &Id) -> Option<u32> {
        let find = |v: &[(Id, u32)]| v.binary_search_by(|x| x.0.cmp(id)).ok().map(|k| v[k].1);
        find(&self.sorted_ids).or_else(|| find(&self.sorted_aliases))
    }
}
