//! What the small-component rule moves (SPEC §6): `blobc fragments <tier file> <inputs name> <data dir>`.
//!
//! Every ID of the inputs is snapped per mode with the rule off (the nearest servable piece)
//! and on (the model's choice). Per mode: the IDs moved and how much farther they stand; the
//! served IDs on components other than 0, before and after, on islands and on the mainland;
//! and every component the rule empties that served more than `LISTED` IDs.
//!
//! Real islands are those of `snap::islands`; IDs moved off one go to a larger component
//! within the radius (SPEC §6), and the data report lists every island so emptied.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use crate::profile::config::Config;

use anyhow::{Context, Result, bail, ensure};
use rayon::prelude::*;

use crate::build::sha256_file;
use crate::snap::islands::{Islands, first_point};
use crate::snap::modes::{active_component, car_component};
use crate::snap::{Grid, Sets};
use crate::verify::reference::{Inputs, Reference};
use blobmap_format::runtime::Fam;
use crate::verify::tier::TierFile;

/// Emptied components serving more IDs than this before are listed.
const LISTED: u64 = 20;
fn quantiles(v: &mut [f64]) -> String {
    if v.is_empty() {
        return "–".into();
    }
    v.sort_by(f64::total_cmp);
    let q = |p: f64| v[((p * v.len() as f64).ceil() as usize).clamp(1, v.len()) - 1];
    format!("p50 {:.1}  p90 {:.1}  p99 {:.1}  max {:.1}", q(0.5), q(0.9), q(0.99), v[v.len() - 1])
}


pub fn main(a: Vec<String>) -> Result<()> {
    let [tier_file, name, data] = &a[..] else {
        bail!("usage: blobc fragments <tier file> <inputs name> <data dir>");
    };
    let data = Path::new(data);
    let tf = TierFile::read(Path::new(tier_file))?;
    let set = tf.inputs.iter().find(|s| &s.name == name).with_context(|| format!("the tier lists no inputs named {name}"))?;
    for (p, want) in set.files.iter().filter(|(p, _)| p.starts_with(&format!("{}/", set.config))) {
        ensure!(&sha256_file(&data.join(p))?.0 == want, "{p}: not the file the tier lists");
    }
    let inputs = Inputs {
        pbf: data.join(&set.pbf),
        overture: data.join(&set.overture),
        config: data.join(&set.config),
        extent: set.extent.as_ref().map(|e| data.join(e)),
        work: data.join(format!("{name}.fragments-work")),
    };
    let islands = Islands::read(&crate::osm::Osm::open(&inputs.pbf, &inputs.work.join("islands"), &Config::load(&inputs.config)?)?)?;
    println!("{} island polygons", islands.all.len());
    let r = Reference::build(&inputs, &[Fam::Car, Fam::Active])?;
    let (cgrid, agrid) = (Grid::new(&r.car_pieces), Grid::new(&r.active_pieces));
    let mut real_emptied = 0;
    for mode in &r.modes {
        let (p, e, grid) = if mode.car { (&r.car_pieces, &r.car_components, &cgrid) } else { (&r.active_pieces, &r.active_components, &agrid) };
        let sets: &dyn Sets = &mode.sets;
        let component = |k: u32| if mode.car { car_component(&r.car, e, k as usize) } else { active_component(p, e, k as usize) }.expect("servable");
        // Per served ID: (piece before, piece after, extra metres).
        let moves: Vec<Option<(u32, u32, f64)>> = (0..r.n())
            .into_par_iter()
            .map(|i| {
                let en = mode.at[i].ok()?;
                let (lat, lon) = r.points[i];
                let f = grid.find(p, lat, lon, mode.radius, mode.small_max, sets);
                let (b, a) = (f.nearest.expect("served"), f.servable.expect("served"));
                assert_eq!(a.piece, en.piece, "the re-snap differs from the reference");
                Some((b.piece, a.piece, a.dist_m - b.dist_m))
            })
            .collect();
        // Per component: IDs before and after, the least piece of each (a point on it), and
        // where the IDs moved away from it went.
        let mut before: BTreeMap<u32, (u64, u32)> = BTreeMap::new();
        let mut after: BTreeMap<u32, (u64, u32)> = BTreeMap::new();
        let mut went: BTreeMap<u32, (BTreeMap<u32, u64>, Vec<f64>)> = BTreeMap::new();
        let mut extra = Vec::new();
        for &(b, a, d) in moves.iter().flatten() {
            let (xb, xa) = (component(b), component(a));
            for (map, x, k) in [(&mut before, xb, b), (&mut after, xa, a)] {
                let s = map.entry(x).or_insert((0, k));
                s.0 += 1;
                s.1 = s.1.min(k);
            }
            if b != a {
                extra.push(d);
                let w = went.entry(xb).or_default();
                *w.0.entry(xa).or_default() += 1;
                w.1.push(d);
            }
        }
        // The real island each component serving IDs stands on, if any.
        let asked: BTreeSet<u32> = before.keys().chain(after.keys()).copied().collect();
        let island_of = islands.real(p, sets, &e.nodes, &asked, &component);
        let on_island = |x: u32| island_of[&x];
        let label = |x: u32| island_of[&x].map_or("–".to_string(), |j| islands.all[j as usize].label.clone());
        let sum = |map: &BTreeMap<u32, (u64, u32)>, island: bool| -> u64 {
            map.iter().filter(|(x, _)| **x != 0 && on_island(**x).is_some() == island).map(|(_, s)| s.0).sum()
        };
        println!("{} ({}): {} IDs served, {} moved; extra metres {}", mode.name, if mode.car { "car" } else { "active" }, moves.iter().flatten().count(), extra.len(), quantiles(&mut extra));
        println!("  served on components other than 0, mainland: {} before, {} after", sum(&before, false), sum(&after, false));
        println!("  served on components other than 0, islands:  {} before, {} after", sum(&before, true), sum(&after, true));
        let emptied: Vec<(u32, u64)> = before.iter().filter(|(x, s)| s.0 > LISTED && !after.contains_key(x)).map(|(&x, s)| (x, s.0)).collect();
        let mut rows: Vec<(u32, u64)> = emptied.clone();
        rows.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let real: Vec<u32> = rows.iter().map(|r| r.0).filter(|&x| on_island(x).is_some()).collect();
        real_emptied += real.len();
        println!("  {} components emptied that served more than {LISTED} IDs, {} of them on a real island:", rows.len(), real.len());
        println!("    {:>7} {:>5} {:>6} {:>8} {:>9}  {:>20}  {:>11}  {:<7} island", "component", "nodes", "IDs", "to", "median m", "point", "OSM way", "verdict");
        for (x, n) in rows {
            let (to, mut d) = went.remove(&x).unwrap_or_default();
            let to = to.iter().max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0))).map_or(u32::MAX, |t| *t.0);
            let med = { d.sort_by(f64::total_cmp); d.get(d.len() / 2).copied().unwrap_or(f64::NAN) };
            let k = before[&x].1;
            let pt = first_point(p, k);
            let verdict = if on_island(x).is_some() { "ISLAND" } else { "fragment" };
            println!("    {x:>7} {:>5} {n:>6} {to:>8} {med:>9.1}  {:>9.5}, {:>9.5}  {:>11}  {verdict:<8} {}", e.nodes[x as usize], pt.0, pt.1, p.way[k as usize], label(x));
        }
        // The real islands still served, for checking the island test against known ones.
        let mut isl: Vec<(u32, u64)> = after.iter().filter(|(x, _)| **x != 0 && on_island(**x).is_some()).map(|(&x, s)| (x, s.0)).collect();
        isl.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        println!("  the 15 islands serving the most IDs after:");
        for (x, n) in isl.into_iter().take(15) {
            println!("    {x:>7} {:>8} nodes {n:>7} IDs  {}", e.nodes[x as usize], label(x));
        }
    }
    println!("{real_emptied} components on real islands emptied");
    Ok(())
}
