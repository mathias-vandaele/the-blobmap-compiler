//! The reference from the inputs: the original family graphs, their components, the catalog,
//! and every ID snapped per mode (SPEC §6), computed with the compiler's input modules but
//! never from the model.
//!
//! A reference holds the families asked for, so a large input is checked one family at a
//! time; the other family's fields stay empty and its modes absent.

use std::path::{Path, PathBuf};

use anyhow::Result;
use blobmap_format::NONE;
use blobmap_format::runtime::Fam;
use rayon::prelude::*;

use crate::build::{ActiveNetwork, CarNetwork, ReadCatalog, ReadNodes, Stage};
use crate::graph::{ActiveGraph, CarGraph, Components, active::METRICS};
use crate::osm::column::{Column, ColumnWriter, Plain};
use crate::osm::{Osm, Pieces};
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
    /// Scratch space for the node locations and the reference's columns (on disk, not in
    /// memory).
    pub work: PathBuf,
}

/// Where an ID stands for a mode.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Entry {
    pub component: u32,
    pub piece: u32,
    pub frac: f64,
}

/// An ID's `Entry`, or why it is not served, as a column row of 16 bytes: not served is
/// `piece == NONE`, with the code in `component` and the metres in `frac`.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct At {
    frac: f64,
    component: u32,
    piece: u32,
}

// SAFETY: `repr(C)` of 8 + 4 + 4 bytes, no padding.
unsafe impl Plain for At {}

impl At {
    fn new(x: Result<Entry, (u8, f32)>) -> At {
        match x {
            Ok(e) => At { frac: e.frac, component: e.component, piece: e.piece },
            Err((code, m)) => At { frac: m as f64, component: code as u32, piece: NONE },
        }
    }

    fn get(self) -> Result<Entry, (u8, f32)> {
        if self.piece == NONE { Err((self.component as u8, self.frac as f32)) } else { Ok(Entry { component: self.component, piece: self.piece, frac: self.frac }) }
    }
}

pub struct Mode {
    pub name: &'static str,
    pub car: bool,
    /// The metric's index in its family.
    pub metric: usize,
    /// Per ID (internal-index order), on disk.
    rows: Column<At>,
    /// The reference split per component, on disk: the served IDs of component x, ascending,
    /// are `by_component[first[x] .. first[x + 1]]`.
    by_component: Column<u32>,
    first: Vec<u64>,
    /// The snapping sets, the radius and the small-component size (SPEC §6).
    pub sets: Membership,
    pub radius: f64,
    pub small_max: u32,
}

impl Mode {
    /// ID i's entry, or (code, metres); code 4 for a filtered place.
    pub fn at(&self, i: usize) -> Result<Entry, (u8, f32)> {
        self.rows[i].get()
    }

    /// Every ID's `at`, in internal-index order.
    pub fn iter(&self) -> impl Iterator<Item = Result<Entry, (u8, f32)>> + '_ {
        self.rows.iter().map(|x| x.get())
    }

    /// The served IDs of component x, ascending.
    pub fn in_component(&self, x: u32) -> &[u32] {
        &self.by_component[self.first[x as usize] as usize..self.first[x as usize + 1] as usize]
    }

    /// The family's components.
    pub fn components(&self) -> u32 {
        self.first.len() as u32 - 1
    }

    /// The served IDs.
    pub fn served(&self) -> usize {
        self.by_component.len()
    }

    /// Its served IDs by component (a counting sort), written to `dir`.
    fn split(rows: &[At], components: usize, dir: &Path, name: &str) -> Result<(Column<u32>, Vec<u64>)> {
        let mut first = vec![0u64; components + 1];
        for x in rows.iter().filter(|x| x.piece != NONE) {
            first[x.component as usize + 1] += 1;
        }
        for k in 0..components {
            first[k + 1] += first[k];
        }
        let mut at = first.clone();
        let mut by = vec![0u32; first[components] as usize];
        for (i, x) in rows.iter().enumerate().filter(|(_, x)| x.piece != NONE) {
            by[at[x.component as usize] as usize] = i as u32;
            at[x.component as usize] += 1;
        }
        let mut w = ColumnWriter::new(Some(dir), name)?;
        by.into_iter().try_for_each(|i| w.push(i))?;
        Ok((w.finish()?, first))
    }
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
    /// Every ID of the model in internal-index order: addresses, then places, on disk.
    pub ids: Column<Id>,
    /// The addresses among them, which come first.
    pub n_addr: usize,
    /// Each place's catalog index, in internal-index order.
    places: Vec<u32>,
    /// (alias, internal index).
    pub aliases: Vec<(Id, u32)>,
    /// The modes of the families held, in manifest order.
    pub modes: Vec<Mode>,
    /// Per ID (internal-index order): its point (lat, lon), on disk.
    pub points: Column<[f64; 2]>,
    /// The antimeridian's twin groups (SPEC §4), found in the `.pbf` apart from the input
    /// modules: per latitude with nodes at both +180° and -180°, their ids ascending.
    pub twins: Vec<Vec<i64>>,
    /// The internal indices in the order of their IDs, and the aliases sorted.
    sorted_ids: Vec<u32>,
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
    found: Vec<Column<Snapped>>,
}

/// The `.pbf`'s nodes on the antimeridian, grouped by latitude where both signs of longitude
/// occur.
fn twin_groups(pbf: &std::path::Path) -> Result<Vec<Vec<i64>>> {
    let at = |id: i64, lat: i32, lon: i32| if lon.abs() == 1_800_000_000 { vec![(lat, lon > 0, id)] } else { Vec::new() };
    let mut nodes = osmpbf::ElementReader::from_path(pbf)?.par_map_reduce(
        |e| match e {
            osmpbf::Element::DenseNode(n) => at(n.id(), n.decimicro_lat(), n.decimicro_lon()),
            osmpbf::Element::Node(n) => at(n.id(), n.decimicro_lat(), n.decimicro_lon()),
            _ => Vec::new(),
        },
        Vec::new,
        |mut a, b| {
            a.extend(b);
            a
        },
    )?;
    nodes.sort_unstable();
    Ok(nodes
        .chunk_by(|a, b| a.0 == b.0)
        .filter(|g| g.iter().any(|x| x.1) && g.iter().any(|x| !x.1))
        .map(|g| {
            let mut ids: Vec<i64> = g.iter().map(|x| x.2).collect();
            ids.sort_unstable();
            ids
        })
        .collect())
}

/// Points snapped at once, before their outcomes go to the column.
const SNAP_BATCH: usize = 1 << 22;

fn snapping(osm: &Osm, config: &Config, fam: Fam, points: &[[f64; 2]], dir: &Path) -> Result<Snapping> {
    let car = fam == Fam::Car;
    let (pieces, cg, ag) = if car {
        let (p, g) = phase("  car network", || CarNetwork.run((osm, config)))?;
        (p, g, ActiveGraph::default())
    } else {
        let (p, g) = phase("  active network", || ActiveNetwork.run((osm, config)))?;
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
        modes
            .iter()
            .map(|(name, _, sets, radius, small)| {
                let mut w = ColumnWriter::new(Some(dir), &format!("found-{name}"))?;
                for batch in points.chunks(SNAP_BATCH) {
                    let done: Vec<Snapped> = batch.par_iter().map(|&[lat, lon]| Snapped::new(&grid.find(&pieces, lat, lon, *radius, *small, sets), NONE)).collect();
                    for x in done {
                        w.push(x)?;
                    }
                }
                w.finish()
            })
            .collect::<Result<Vec<_>>>()
    })?;
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
        let osm = phase("  nodes", || ReadNodes.run((i.pbf.as_path(), i.work.join("nodes").as_path(), &config)))?;
        // The reference's columns, apart for each set of families (a verify builds several).
        let dir = i.work.join(format!("ref-{}", fams.iter().map(|f| f.name()).collect::<Vec<_>>().join("+")));
        let _ = std::fs::remove_dir_all(&dir);
        let column = |name: &str, xs: &mut dyn Iterator<Item = [f64; 2]>| -> Result<Column<[f64; 2]>> {
            let mut w = ColumnWriter::new(Some(&dir), name)?;
            for x in xs {
                w.push(x)?;
            }
            w.finish()
        };
        let points = column("points-all", &mut catalog.addresses.iter().map(|a| [a.lat, a.lon]).chain(catalog.places.iter().map(|p| [p.lat, p.lon])))?;
        // Without an extent, an ID is in the model when some mode has a road within its
        // radius (SPEC §3): the families not held are snapped for that alone, one at a time.
        let mut near = vec![polygon.is_some(); points.len()];
        if polygon.is_none() {
            for fam in [Fam::Car, Fam::Active].into_iter().filter(|f| !fams.contains(f)) {
                let s = snapping(&osm, &config, fam, &points, &dir)?;
                s.found.iter().for_each(|f| near.iter_mut().zip(f).for_each(|(n, x)| *n |= x.any));
            }
        }
        let held: Vec<Snapping> =
            [Fam::Car, Fam::Active].into_iter().filter(|f| fams.contains(f)).map(|f| snapping(&osm, &config, f, &points, &dir)).collect::<Result<_>>()?;
        for s in &held {
            s.found.iter().for_each(|f| near.iter_mut().zip(f).for_each(|(n, x)| *n |= x.any));
        }
        let na = catalog.addresses.len();
        let mut pts: Vec<usize> = (0..na).filter(|&k| near[k]).collect();
        let n_addr = pts.len();
        pts.extend((0..catalog.places.len()).filter(|&p| near[na + p]).map(|p| na + p));
        drop(near);
        let mut w = ColumnWriter::new(Some(&dir), "ids")?;
        pts.iter().try_for_each(|&k| w.push(if k < na { catalog.addresses[k].id } else { catalog.places[k - na].id }))?;
        let ids = w.finish()?;
        let places: Vec<u32> = pts[n_addr..].iter().map(|&k| (k - na) as u32).collect();
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
            n_addr,
            places,
            aliases,
            modes: Vec::new(),
            points: column("points", &mut pts.iter().map(|&k| points[k]))?,
            twins: phase("  twins", || twin_groups(&i.pbf))?,
            sorted_ids: Vec::new(),
            sorted_aliases: Vec::new(),
        };
        for s in held {
            let car = !s.modes.is_empty() && s.modes[0].0 == "car";
            for ((name, metric, sets, radius, small_max), found) in s.modes.into_iter().zip(&s.found) {
                let at = |k: usize| {
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
                };
                let mut w = ColumnWriter::new(Some(&dir), &format!("at-{name}"))?;
                pts.iter().try_for_each(|&k| w.push(At::new(at(k))))?;
                let rows = w.finish()?;
                let (by_component, first) = Mode::split(&rows, s.components.len(), &dir, &format!("by-component-{name}"))?;
                r.modes.push(Mode { name, car, metric, rows, by_component, first, sets, radius, small_max });
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
        let mut order: Vec<u32> = (0..r.ids.len() as u32).collect();
        order.par_sort_unstable_by_key(|&i| r.ids[i as usize]);
        r.sorted_ids = order;
        r.sorted_aliases = r.aliases.to_vec();
        r.sorted_aliases.sort_unstable();
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

    /// The catalog index of ID i if it is a place.
    pub fn place_of(&self, i: usize) -> Option<usize> {
        i.checked_sub(self.n_addr).map(|p| self.places[p] as usize)
    }

    /// FORMAT §7.1 lookup on the reference IDs.
    pub fn lookup(&self, id: &Id) -> Option<u32> {
        let s = &self.sorted_ids;
        let own = s.binary_search_by(|&i| self.ids[i as usize].cmp(id)).ok().map(|k| s[k]);
        let a = &self.sorted_aliases;
        own.or_else(|| a.binary_search_by(|x| x.0.cmp(id)).ok().map(|k| a[k].1))
    }
}
