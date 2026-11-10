//! Every component served (PLAN, milestone 3): `blobc components <model> <tier file> <data dir>`.
//!
//! From the model's inputs (found in the tier file, as `verify` does), per family: the IDs
//! per reason code and mode; the largest served (not trapped) components, each with its nodes,
//! its IDs per mode and the centroid of those IDs; the largest served components that no ID
//! uses, and their count by size; every ID unserved because of its component (TrappedComponent)
//! with the trapped component it would snap to. `--all` lists every served component. The
//! model's component sizes must equal the reference's.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use blobmap_format::NONE;
use blobmap_format::runtime::{Fam, Runtime};
use rayon::prelude::*;

use crate::snap::{TRAPPED_COMPONENT, find_brute};
use crate::verify::expected::{reason_name, uuid};
use crate::verify::reference::Reference;
use crate::verify::tier::TierFile;

/// Components listed in each table.
const TOP: usize = 25;

pub fn main(a: Vec<String>) -> Result<()> {
    let all = a.iter().any(|x| x == "--all");
    let a: Vec<&String> = a.iter().filter(|x| *x != "--all").collect();
    let [model, tier_file, data] = a[..] else {
        bail!("usage: blobc components <model> <tier file> <data dir> [--all]");
    };
    let rt = Runtime::open(Path::new(model))?;
    let tf = TierFile::read(Path::new(tier_file))?;
    let work = crate::verify::work_dir(Path::new(model), "components");
    let inputs = tf.find(&rt.manifest)?.check(Path::new(data), &rt.manifest, &work)?;
    let r = Reference::build(&inputs, &[Fam::Car, Fam::Active]);
    let _ = std::fs::remove_dir_all(&work);
    let r = r?;
    for (fam, e) in [(Fam::Car, &r.car_components), (Fam::Active, &r.active_components)] {
        let t = &rt.tables[&fam];
        ensure!((0..t.components() as u32).map(|x| t.nodes(x)).eq(e.nodes.iter().copied()), "{}: the model's components differ from the reference's", fam.name());
        let modes: Vec<usize> = (0..r.modes.len()).filter(|&m| r.modes[m].car == (fam == Fam::Car)).collect();
        // Per component: IDs per mode, and the sums of their latitudes and of their longitudes
        // as unit vectors (so that a centroid across the antimeridian stays on it).
        let mut ids = vec![vec![0u64; modes.len()]; e.len()];
        let mut sum = vec![(0.0f64, 0.0f64, 0.0f64, 0u64); e.len()];
        for i in 0..r.n() {
            let mut seen = None;
            for (k, &m) in modes.iter().enumerate() {
                if let Ok(en) = r.modes[m].at[i] {
                    ids[en.component as usize][k] += 1;
                    seen = Some(en.component);
                }
            }
            if let Some(x) = seen {
                let s = &mut sum[x as usize];
                let (sin, cos) = r.points[i].1.to_radians().sin_cos();
                s.0 += r.points[i].0;
                s.1 += sin;
                s.2 += cos;
                s.3 += 1;
            }
        }
        let served: Vec<usize> = (0..e.len()).filter(|&x| !e.trapped[x]).collect();
        let unused: Vec<usize> = served.iter().copied().filter(|&x| ids[x].iter().all(|&c| c == 0)).collect();
        let names: Vec<&str> = modes.iter().map(|&m| r.modes[m].name).collect();
        println!(
            "{}: {} components, {} trapped, {} served, {} served with no ID",
            fam.name(),
            e.len(),
            e.len() - served.len(),
            served.len(),
            unused.len()
        );
        for &m in &modes {
            let mut reasons: BTreeMap<&str, u64> = BTreeMap::new();
            for at in &r.modes[m].at {
                *reasons.entry(match at {
                    Ok(_) => "served",
                    Err((c, _)) => reason_name(*c),
                }).or_default() += 1;
            }
            println!("  {}: {reasons:?}", r.modes[m].name);
        }
        for &m in &modes {
            let mode = &r.modes[m];
            let at: Vec<usize> = (0..r.n()).filter(|&i| matches!(mode.at[i], Err((TRAPPED_COMPONENT, _)))).collect();
            if at.is_empty() {
                continue;
            }
            // The nearest eligible piece, every piece searched: its component is the trapped one.
            let p = if mode.car { &r.car_pieces } else { &r.active_pieces };
            let hits: Vec<_> = at.par_iter().map(|&i| find_brute(p, r.points[i].0, r.points[i].1, r.modes[m].radius, r.modes[m].small_max, &r.modes[m].sets).eligible).collect();
            println!("  {}: {} IDs unserved because of their component (TrappedComponent)", mode.name, at.len());
            println!("  {:>36} {:>21} {:>7} {:>7} {:>6} {:>11}", "ID", "point", "metres", "component", "nodes", "OSM way");
            for (&i, h) in at.iter().zip(hits) {
                let h = h.context("a trapped ID has an eligible piece")?;
                let k = h.piece as usize;
                let (x, way) = if mode.car {
                    let s = r.car.seg_of_piece[k].into_iter().find(|&s| s != NONE && e.trapped[e.of[s as usize] as usize]).context("a trapped segment")?;
                    (e.of[s as usize], r.car.way[s as usize])
                } else {
                    let road = r.active.road_of_piece[k].into_iter().find(|&d| d != NONE).context("a road")?;
                    (e.of[r.active.tail[road as usize] as usize], r.active.way[road as usize])
                };
                let (lat, lon) = r.points[i];
                println!("  {:>36} {lat:>10.5},{lon:>10.5} {:>7.1} {x:>7} {:>6} {way:>11}", uuid(&r.ids[i]), h.dist_m, e.nodes[x as usize]);
            }
        }
        let row = |x: usize| {
            let s = sum[x];
            let at = if s.3 > 0 { format!("{:.4}, {:.4}", s.0 / s.3 as f64, s.1.atan2(s.2).to_degrees()) } else { "–".into() };
            let c: Vec<String> = ids[x].iter().map(u64::to_string).collect();
            println!("  {x:>6} {:>10} {:>28}  {at}", e.nodes[x], c.join(", "));
        };
        let head = format!("  {:>6} {:>10} {:>28}  centroid of its IDs", "component", "nodes", format!("IDs ({})", names.join(", ")));
        println!("  the {} largest served components:\n{head}", TOP.min(served.len()));
        served.iter().take(TOP).for_each(|&x| row(x));
        println!("  the {} largest served components with no ID:\n{head}", TOP.min(unused.len()));
        unused.iter().take(TOP).for_each(|&x| row(x));
        let sizes: BTreeMap<u32, u64> = unused.iter().fold(BTreeMap::new(), |mut m, &x| {
            *m.entry(e.nodes[x]).or_default() += 1;
            m
        });
        println!("  served components with no ID, by node count: {sizes:?}");
        if !all {
            continue;
        }
        println!("  every served component:\n{head}");
        for &x in &served {
            row(x);
        }
    }
    Ok(())
}
