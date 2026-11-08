//! Every component served (PLAN, milestone 3): `experts <model> <tier file> <data dir>`.
//!
//! From the model's inputs (found in the tier file, as `verify` does), per family: the
//! experts that are served (not trapped), each with its nodes, its IDs per mode and the
//! centroid of those IDs; and the served experts that no ID uses. The model's expert sizes
//! must equal the reference's.

use std::path::Path;

use anyhow::{Result, bail, ensure};
use blobmap_conformance::reference::Reference;
use blobmap_conformance::tier::TierFile;
use blobmap_runtime::{Fam, Runtime};

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let [model, tier_file, data] = &a[..] else {
        bail!("usage: experts <model> <tier file> <data dir>");
    };
    let rt = Runtime::open(Path::new(model))?;
    let tf = TierFile::read(Path::new(tier_file))?;
    let inputs = tf.find(&rt.manifest)?.check(Path::new(data), &rt.manifest)?;
    let r = Reference::build(&inputs)?;
    for (fam, e) in [(Fam::Car, &r.car_experts), (Fam::Active, &r.active_experts)] {
        ensure!(rt.tables[&fam].nodes == e.nodes, "{}: the model's experts differ from the reference's", fam.name());
        let modes: Vec<usize> = (0..r.modes.len()).filter(|&m| r.modes[m].car == (fam == Fam::Car)).collect();
        // Per expert: IDs per mode, and the sums of their points.
        let mut ids = vec![vec![0u64; modes.len()]; e.len()];
        let mut sum = vec![(0.0f64, 0.0f64, 0u64); e.len()];
        for i in 0..r.n() {
            let mut seen = None;
            for (k, &m) in modes.iter().enumerate() {
                if let Ok(en) = r.modes[m].at[i] {
                    ids[en.expert as usize][k] += 1;
                    seen = Some(en.expert);
                }
            }
            if let Some(x) = seen {
                let s = &mut sum[x as usize];
                s.0 += r.points[i].0;
                s.1 += r.points[i].1;
                s.2 += 1;
            }
        }
        let served: Vec<usize> = (0..e.len()).filter(|&x| !e.trapped[x]).collect();
        let unused: Vec<usize> = served.iter().copied().filter(|&x| ids[x].iter().all(|&c| c == 0)).collect();
        let names: Vec<&str> = modes.iter().map(|&m| r.modes[m].name).collect();
        println!(
            "{}: {} experts, {} trapped, {} served, {} served with no ID",
            fam.name(),
            e.len(),
            e.len() - served.len(),
            served.len(),
            unused.len()
        );
        println!("  {:>6} {:>10} {:>28}  centroid of its IDs", "expert", "nodes", format!("IDs ({})", names.join(", ")));
        for &x in &served {
            let s = sum[x];
            let at = if s.2 > 0 { format!("{:.4}, {:.4}", s.0 / s.2 as f64, s.1 / s.2 as f64) } else { "–".into() };
            let c: Vec<String> = ids[x].iter().map(u64::to_string).collect();
            println!("  {x:>6} {:>10} {:>28}  {at}", e.nodes[x], c.join(", "));
        }
    }
    Ok(())
}
