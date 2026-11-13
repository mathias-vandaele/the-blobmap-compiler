//! `blobc inputs` (PLAN, milestone 4): the build's input stages alone, on any `.pbf` the
//! planet included. The node locations, each family's ways and pieces, the turn
//! restrictions, the Overture catalog and the island polygons are read as a build reads
//! them, each dropped when its stage ends, with each stage's wall time and peak RSS as the
//! build report gives them, and what each stage read.

use std::path::Path;

use anyhow::{Result, bail};
use serde_json::{Value, json};

use crate::build::{ReadCatalog, ReadIslands, ReadNodes, Runner, Stage};
use crate::osm::{Network, Pieces};
use crate::overture::{Polygon, Release};
use crate::profile::active::{ActiveMode, ActiveProfile, Mode};
use crate::profile::car::{CarProfile, WayAttrs};
use crate::profile::config::Config;

/// A stage made of a closure, for the steps a build runs inside its own stages.
struct Step<F>(&'static str, F);

impl<T, F: Fn() -> Result<T>> Stage<()> for Step<F> {
    type Output = T;
    fn name(&self) -> &'static str {
        self.0
    }
    fn run(&self, _: ()) -> Result<T> {
        (self.1)()
    }
}

fn sizes<A>(net: &Network<A>, p: &Pieces) -> Value {
    json!({
        "ways": net.ways.len(),
        "way_nodes": net.way_nodes(),
        "nodes": net.nodes(),
        "pieces": p.len(),
        "points": p.pt_lat.len(),
        "routing_nodes": p.node_osm.len(),
        "missing_coords": p.missing_coords,
    })
}

/// Keeps what a stage read, and logs it at once, so that a run cut short still shows it.
fn note(read: &mut serde_json::Map<String, Value>, stage: &str, v: Value) {
    eprintln!("{stage}: {v}");
    read.insert(stage.into(), v);
}

pub fn main(a: Vec<String>) -> Result<()> {
    let (pbf, overture, config, work, extent) = match &a[..] {
        [p, o, c, w] => (p, o, c, w, None),
        [p, o, c, w, e] => (p, o, c, w, Some(e)),
        _ => bail!("usage: blobc inputs <pbf> <overture release dir> <config dir> <work dir> [<extent .poly>]"),
    };
    let (pbf, work) = (Path::new(pbf), Path::new(work));
    let config = Config::load(Path::new(config))?;
    let release = Release::open(Path::new(overture))?;
    let polygon = match extent {
        Some(e) => Some(Polygon::parse_poly(&std::fs::read_to_string(e)?)?),
        None => None,
    };
    let mut run = Runner::default();
    let mut read = serde_json::Map::new();
    let osm = run.run(&ReadNodes, (pbf, work, &config))?;
    // The antimeridian's twins (SPEC §4): each node others are joined to, with them.
    let mut by_node: Vec<(i64, i64)> = osm.twins.iter().map(|&(from, to)| (to, from)).collect();
    by_node.sort_unstable();
    let mut twins: Vec<Value> = Vec::new();
    for g in by_node.chunk_by(|a, b| a.0 == b.0) {
        let to = g[0].0;
        let (lat, lon) = crate::osm::nodes::unpack(osm.coord(to));
        twins.push(json!({"node": to, "lat": lat, "lon": lon, "joined": g.iter().map(|t| t.1).collect::<Vec<_>>()}));
    }
    note(&mut read, "nodes", json!({"nodes": osm.len(), "car_obstacles": osm.car_obstacles.len(), "active_obstacles": osm.active_obstacles.len(), "twins": twins}));
    let car = run.run(
        &Step("car-pieces", || {
            let net = Network::read(&osm, &osm.car_obstacles, &CarProfile::new(&config.car.config), &config.car.config.snap, Some(&osm.dir().join("car")))?;
            let p = net.pieces(&[|a: &WayAttrs| *a])?;
            Ok(sizes(&net, &p))
        }),
        (),
    )?;
    note(&mut read, "car", car);
    let restrictions = run.run(
        &Step("restrictions", || {
            let (r, _) = osm.restrictions()?;
            Ok(json!({"restrictions": r.len()}))
        }),
        (),
    )?;
    note(&mut read, "restrictions", restrictions);
    let active = run.run(
        &Step("active-pieces", || {
            let prof = ActiveProfile::new(&config.active.config);
            let net = Network::read(&osm, &osm.active_obstacles, &prof, &config.active.config.snap, Some(&osm.dir().join("active")))?;
            let modes: Vec<ActiveMode> = [Mode::Bike, Mode::Walk].map(|mode| ActiveMode { mode, cfg: prof.cfg }).to_vec();
            let p = net.pieces(&modes)?;
            Ok(sizes(&net, &p))
        }),
        (),
    )?;
    note(&mut read, "active", active);
    let islands = run.run(&ReadIslands, (&osm, Some(work.join("islands").as_path())))?;
    note(&mut read, "islands", json!({"polygons": islands.all.len()}));
    drop((islands, osm));
    let catalog = run.run(&ReadCatalog, (&release, &config, polygon.as_ref(), Some(work.join("catalog").as_path())))?;
    note(&mut read, "catalog", json!({"addresses": catalog.addresses.len(), "places": catalog.places.len(), "aliases": catalog.aliases.len()}));
    drop(catalog);
    let report = json!({
        "pbf": pbf.display().to_string(),
        "stages": run.stages.iter().map(|s| json!({"name": s.name, "wall_s": s.wall_s, "peak_rss_bytes": s.peak_rss_bytes})).collect::<Vec<_>>(),
        "peak_rss_bytes": run.stages.iter().filter_map(|s| s.peak_rss_bytes).max(),
        "read": read,
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
