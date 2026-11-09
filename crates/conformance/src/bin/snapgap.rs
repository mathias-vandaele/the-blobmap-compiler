//! How far beyond its piece a larger component lies (SPEC §6, choosing Δ):
//! `snapgap <model> <tier file> <data dir> <car islands> <active islands>`, each islands
//! argument a comma-separated list of the family's expert indices that are real islands.
//!
//! For every ID served on an expert other than 0, per mode: d0, the distance to its piece;
//! d1, the distance to the nearest servable piece of an expert with more nodes (searched up
//! to `REACH_M`); and the extra distance d1 − d0. The IDs on the listed islands and the
//! others (mainland fragments) are reported apart: their distributions, the fragments whose
//! IDs lie farthest from a larger expert, and the islands nearest to one.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use blobmap_conformance::reference::Reference;
use blobmap_conformance::tier::TierFile;
use blobmap_runtime::{Fam, Runtime};
use blobmap_snap::modes::{active_expert, car_expert};
use blobmap_snap::{Grid, Sets};
use rayon::prelude::*;

/// How far a larger component is searched for.
const REACH_M: f64 = 5000.0;
/// Upper bounds of the histogram buckets of the extra distance, in metres.
const BUCKETS: [f64; 13] = [1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 50.0, 100.0, 200.0, 300.0, 500.0, 1000.0, REACH_M];
/// Rows in the tables of farthest fragments and nearest islands.
const TOP: usize = 25;

/// The servable pieces of experts with more than `min` nodes.
struct Larger<'a> {
    base: &'a dyn Sets,
    expert: &'a [u32],
    nodes: &'a [u32],
    min: u32,
}

impl Sets for Larger<'_> {
    fn in_graph(&self, k: u32) -> bool {
        self.base.in_graph(k)
    }
    fn eligible(&self, k: u32) -> bool {
        self.base.eligible(k)
    }
    fn servable(&self, k: u32) -> bool {
        self.base.servable(k) && self.nodes[self.expert[k as usize] as usize] > self.min
    }
}

fn quantiles(v: &mut [f64]) -> String {
    if v.is_empty() {
        return "–".into();
    }
    v.sort_by(f64::total_cmp);
    let q = |p: f64| v[((p * v.len() as f64).ceil() as usize).clamp(1, v.len()) - 1];
    let f = |x: f64| if x.is_finite() { format!("{x:.1}") } else { format!(">{REACH_M}") };
    format!("p50 {}  p90 {}  p99 {}  p99.9 {}  min {}  max {}", f(q(0.5)), f(q(0.9)), f(q(0.99)), f(q(0.999)), f(v[0]), f(v[v.len() - 1]))
}

fn histogram(v: &[f64]) -> String {
    let mut lo = 0.0;
    let mut out = Vec::new();
    for hi in BUCKETS {
        out.push(format!("≤{hi}: {}", v.iter().filter(|&&x| x > lo && x <= hi || (lo == 0.0 && x <= hi)).count()));
        lo = hi;
    }
    out.push(format!("beyond: {}", v.iter().filter(|x| !x.is_finite()).count()));
    out.join("  ")
}

fn list(s: &str) -> Result<Vec<u32>> {
    s.split(',').filter(|x| !x.is_empty()).map(|x| Ok(x.trim().parse()?)).collect()
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let [model, tier_file, data, car_islands, active_islands] = &a[..] else {
        bail!("usage: snapgap <model> <tier file> <data dir> <car islands> <active islands>");
    };
    let rt = Runtime::open(Path::new(model))?;
    let tf = TierFile::read(Path::new(tier_file))?;
    let inputs = tf.find(&rt.manifest)?.check(Path::new(data), &rt.manifest)?;
    let r = Reference::build(&inputs)?;
    let islands = [list(car_islands)?, list(active_islands)?];
    for (f, (fam, e)) in [(Fam::Car, &r.car_experts), (Fam::Active, &r.active_experts)].into_iter().enumerate() {
        ensure!(rt.tables[&fam].nodes == e.nodes, "{}: the model's experts differ from the reference's", fam.name());
        let p = if fam == Fam::Car { &r.car_pieces } else { &r.active_pieces };
        let grid = Grid::new(p);
        // Each piece's expert, `u32::MAX` where it is not servable.
        let expert: Vec<u32> = (0..p.len())
            .map(|k| if fam == Fam::Car { car_expert(&r.car, e, k) } else { active_expert(p, e, k) }.unwrap_or(u32::MAX))
            .collect();
        let mut nodes = e.nodes.clone();
        nodes.push(0);
        for (m, mode) in r.modes.iter().enumerate().filter(|(_, md)| md.car == (fam == Fam::Car)) {
            let ids: Vec<(usize, u32)> = (0..r.n()).filter_map(|i| mode.at[i].ok().map(|en| (i, en.expert))).filter(|&(_, x)| x != 0).collect();
            // Per ID: its expert, d0, d1.
            let found: Vec<Result<(usize, u32, f64, f64)>> = ids
                .par_iter()
                .map(|&(i, x)| {
                    let (lat, lon) = r.points[i];
                    let d0 = grid.find(p, lat, lon, r.radius[m], 0.0, &r.sets[m]).nearest.context("a served ID has a piece")?.dist_m;
                    let larger = Larger { base: &r.sets[m], expert: &expert, nodes: &nodes, min: nodes[x as usize] };
                    let d1 = grid.find(p, lat, lon, REACH_M, 0.0, &larger).nearest.map_or(f64::INFINITY, |h| h.dist_m);
                    Ok((i, x, d0, d1))
                })
                .collect();
            let found: Vec<(usize, u32, f64, f64)> = found.into_iter().collect::<Result<_>>()?;
            let island = |x: u32| islands[f].contains(&x);
            println!("{} ({}): {} IDs served on experts other than 0", mode.name, fam.name(), found.len());
            for (name, on) in [("mainland fragments", false), ("listed islands", true)] {
                let mut extra: Vec<f64> = found.iter().filter(|t| island(t.1) == on).map(|t| t.3 - t.2).collect();
                let mut d1: Vec<f64> = found.iter().filter(|t| island(t.1) == on).map(|t| t.3).collect();
                println!("  {name}: {} IDs", extra.len());
                println!("    extra d1 − d0 (m): {}", quantiles(&mut extra));
                println!("    histogram:         {}", histogram(&extra));
                println!("    d1 (m):            {}", quantiles(&mut d1));
            }
            // Per expert: IDs, the least and greatest extra distance, and a point.
            let mut by: BTreeMap<u32, (u64, f64, f64, (f64, f64))> = BTreeMap::new();
            for &(i, x, d0, d1) in &found {
                let b = by.entry(x).or_insert((0, f64::INFINITY, 0.0, r.points[i]));
                b.0 += 1;
                b.1 = b.1.min(d1 - d0);
                b.2 = b.2.max(d1 - d0);
            }
            let row = |x: u32, b: &(u64, f64, f64, (f64, f64))| {
                println!("    {x:>7} {:>7} {:>6} {:>9.1} {:>9.1}  {:.5}, {:.5}", e.nodes[x as usize], b.0, b.1, b.2, b.3.0, b.3.1);
            };
            let head = format!("    {:>7} {:>7} {:>6} {:>9} {:>9}  an ID's point", "expert", "nodes", "IDs", "least m", "most m");
            let mut frag: Vec<(&u32, &(u64, f64, f64, (f64, f64)))> = by.iter().filter(|(x, _)| !island(**x)).collect();
            frag.sort_by(|a, b| b.1.2.total_cmp(&a.1.2).then(a.0.cmp(b.0)));
            println!("  the {TOP} fragments whose IDs lie farthest from a larger expert:\n{head}");
            frag.iter().take(TOP).for_each(|(x, b)| row(**x, b));
            let mut isl: Vec<(&u32, &(u64, f64, f64, (f64, f64)))> = by.iter().filter(|(x, _)| island(**x)).collect();
            isl.sort_by(|a, b| a.1.1.total_cmp(&b.1.1).then(a.0.cmp(b.0)));
            println!("  the listed islands, nearest to a larger expert first:\n{head}");
            isl.iter().for_each(|(x, b)| row(**x, b));
        }
    }
    Ok(())
}
