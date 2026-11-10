//! The reference from the inputs: the original family graphs, their components, the catalog,
//! and every ID snapped per mode (SPEC §6), computed with the compiler's input modules but
//! never from the model.
//!
//! A reference holds the families asked for, so a large input is checked one family at a
//! time; the other family's fields stay empty and its modes absent.

use std::path::PathBuf;

use anyhow::Result;
use blobmap_format::NONE;
use blobmap_format::runtime::Fam;
use rayon::prelude::*;

use crate::build::{ActiveNetwork, CarNetwork, ReadCatalog, Stage};
use crate::graph::{ActiveGraph, CarGraph, Components, active::METRICS};
use crate::osm::Pieces;
use crate::overture::{Catalog, Id, Polygon, Release};
use crate::profile::config::Config;
use crate::snap::modes::{active_component, car_component};
use crate::snap::{FILTERED_PLACE, Grid, Membership, Snapped};
use crate::verify::phase;

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
    pub component: u32,
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
    /// The snapping sets, the radius and the small-component size (SPEC §6).
    pub sets: Membership,
    pub radius: f64,
    pub small_max: u32,
}

pub struct Reference {
    /// The families held, in order.
    pub fams: Vec<Fam>,
    pub car_pieces: Pieces,
    pub car: CarGraph,
    pub car_components: Components,
    pub active_pieces: Pieces,
    pub active: ActiveGraph,
    pub active_components: Components,
    /// The catalog without its address rows, which give only the IDs and points below.
    pub catalog: Catalog,
    /// Every ID of the model in internal-index order: addresses, then places (catalog
    /// index), each with its point.
    pub ids: Vec<Id>,
    pub place_of: Vec<Option<usize>>,
    /// (alias, internal index).
    pub aliases: Vec<(Id, u32)>,
    /// The modes of the families held, in manifest order.
    pub modes: Vec<Mode>,
    /// Per ID (internal-index order): its point.
    pub points: Vec<(f64, f64)>,
    /// The IDs and the aliases, each sorted, with their internal index.
    sorted_ids: Vec<(Id, u32)>,
    sorted_aliases: Vec<(Id, u32)>,
}

/// One family's network, components, and per mode its sets, radius and small-component size,
/// with every point snapped.
struct Snapping {
    pieces: Pieces,
    car: CarGraph,
    active: ActiveGraph,
    components: Components,
    modes: Vec<(&'static str, usize, Membership, f64, u32)>,
    found: Vec<Vec<Snapped>>,
}

fn snapping(i: &Inputs, config: &Config, fam: Fam, points: &[(f64, f64)]) -> Result<Snapping> {
    let car = fam == Fam::Car;
    let (pieces, cg, ag) = if car {
        let (p, g) = phase("  car network", || CarNetwork.run((i.pbf.as_path(), config)))?;
        (p, g, ActiveGraph::default())
    } else {
        let (p, g) = phase("  active network", || ActiveNetwork.run((i.pbf.as_path(), config)))?;
        (p, CarGraph::default(), g)
    };
    let components = if car { cg.components() } else { ag.components() };
    let modes: Vec<(&'static str, usize, Membership, f64, u32)> = if car {
        let s = &config.car.config.snap;
        vec![("car", 0, Membership::car(&pieces, &cg, &components), s.radius_m as f64, s.small_component_max_nodes)]
    } else {
        let s = &config.active.config.snap;
        debug_assert_eq!(METRICS, ["bike", "walk"]);
        (0..2).map(|m| (METRICS[m], m, Membership::active(&pieces, &ag, &components, m), s.radius_m as f64, s.small_component_max_nodes)).collect()
    };
    let grid = Grid::new(&pieces);
    let found = phase(if car { "  car snapping" } else { "  active snapping" }, || {
        modes.iter().map(|(_, _, sets, radius, small)| points.par_iter().map(|&(lat, lon)| Snapped::new(&grid.find(&pieces, lat, lon, *radius, *small, sets), NONE)).collect()).collect()
    });
    Ok(Snapping { pieces, car: cg, active: ag, components, modes, found })
}

impl Reference {
    /// The reference of the families asked for (both, in the tools and the tiny tier).
    pub fn build(i: &Inputs, fams: &[Fam]) -> Result<Reference> {
        let config = Config::load(&i.config)?;
        // The catalog first: its reading peaks well above what it keeps, and so do the
        // networks', so neither peak sits on the other's result.
        let release = Release::open(&i.overture)?;
        let polygon = match &i.extent {
            Some(p) => Some(Polygon::parse_poly(&std::fs::read_to_string(p)?)?),
            None => None,
        };
        let mut catalog = phase("  catalog", || ReadCatalog.run((&release, &config, polygon.as_ref())))?;
        let points: Vec<(f64, f64)> =
            catalog.addresses.iter().map(|a| (a.lat, a.lon)).chain(catalog.places.iter().map(|p| (p.lat, p.lon))).collect();
        // Without an extent, an ID is in the model when some mode has a road within its
        // radius (SPEC §3): the families not held are snapped for that alone, one at a time.
        let mut near = vec![polygon.is_some(); points.len()];
        if polygon.is_none() {
            for fam in [Fam::Car, Fam::Active].into_iter().filter(|f| !fams.contains(f)) {
                let s = snapping(i, &config, fam, &points)?;
                s.found.iter().for_each(|f| near.iter_mut().zip(f).for_each(|(n, x)| *n |= x.any));
            }
        }
        let held: Vec<Snapping> = [Fam::Car, Fam::Active].into_iter().filter(|f| fams.contains(f)).map(|f| snapping(i, &config, f, &points)).collect::<Result<_>>()?;
        for s in &held {
            s.found.iter().for_each(|f| near.iter_mut().zip(f).for_each(|(n, x)| *n |= x.any));
        }
        let na = catalog.addresses.len();
        let mut pts: Vec<usize> = (0..na).filter(|&k| near[k]).collect();
        let n_addr = pts.len();
        pts.extend((0..catalog.places.len()).filter(|&p| near[na + p]).map(|p| na + p));
        drop(near);
        let ids: Vec<Id> = pts.iter().map(|&k| if k < na { catalog.addresses[k].id } else { catalog.places[k - na].id }).collect();
        let place_of: Vec<Option<usize>> = pts.iter().map(|&k| (k >= na).then(|| k - na)).collect();
        let index_of_place = |p: u32| pts[n_addr..].binary_search(&(na + p as usize)).ok().map(|i| (n_addr + i) as u32);
        let aliases: Vec<(Id, u32)> = catalog.aliases.iter().filter_map(|&(a, p)| index_of_place(p).map(|i| (a, i))).collect();
        let mut r = Reference {
            fams: [Fam::Car, Fam::Active].into_iter().filter(|f| fams.contains(f)).collect(),
            car_pieces: Pieces::default(),
            car: CarGraph::default(),
            car_components: Components::default(),
            active_pieces: Pieces::default(),
            active: ActiveGraph::default(),
            active_components: Components::default(),
            catalog: Catalog::default(),
            ids,
            place_of,
            aliases,
            modes: Vec::new(),
            points: pts.iter().map(|&k| points[k]).collect(),
            sorted_ids: Vec::new(),
            sorted_aliases: Vec::new(),
        };
        for s in held {
            let car = !s.modes.is_empty() && s.modes[0].0 == "car";
            for ((name, metric, sets, radius, small_max), found) in s.modes.into_iter().zip(&s.found) {
                let at = pts
                    .iter()
                    .map(|&k| {
                        if k >= na && catalog.places[k - na].status != 0 {
                            return Err((FILTERED_PLACE, 0.0));
                        }
                        let (piece, frac) = found[k].outcome()?;
                        let component = if car {
                            car_component(&s.car, &s.components, piece as usize).expect("servable")
                        } else {
                            active_component(&s.pieces, &s.components, piece as usize).expect("servable")
                        };
                        Ok(Entry { component, piece, frac })
                    })
                    .collect();
                r.modes.push(Mode { name, car, metric, at, sets, radius, small_max });
            }
            if car {
                (r.car_pieces, r.car, r.car_components) = (s.pieces, s.car, s.components);
            } else {
                (r.active_pieces, r.active, r.active_components) = (s.pieces, s.active, s.components);
            }
        }
        catalog.addresses = Vec::new();
        r.catalog = catalog;
        crate::build::mem::trim();
        let sorted = |v: Vec<(Id, u32)>| {
            let mut v = v;
            v.sort_unstable();
            v
        };
        r.sorted_ids = sorted(r.ids.iter().enumerate().map(|(i, id)| (*id, i as u32)).collect());
        r.sorted_aliases = sorted(r.aliases.to_vec());
        Ok(r)
    }

    pub fn n(&self) -> usize {
        self.ids.len()
    }

    pub fn has(&self, fam: Fam) -> bool {
        self.fams.contains(&fam)
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
