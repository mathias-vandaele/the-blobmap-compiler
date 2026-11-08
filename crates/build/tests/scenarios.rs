//! Every fixture scenario builds, and reads back through the validating reader.

use blobmap_build::{Args, build};

#[test]
fn every_scenario_builds() {
    let dir = std::env::temp_dir().join(format!("blobmap-build-{}", std::process::id()));
    for s in blobmap_fixtures::scenarios::all() {
        let w = s.write(&dir).unwrap();
        for (variant, config) in &w.configs {
            let out = dir.join(format!("{}-{variant}.model", s.name));
            let a = Args {
                pbf: w.pbf.clone(),
                overture: w.overture.clone(),
                config: config.clone(),
                out: out.clone(),
                extent: w.extent.clone(),
                previous_report: None,
            };
            let r = build(&a).unwrap_or_else(|e| panic!("{} {variant}: {e:#}", s.name));
            let m = blobmap_format::Model::open(&out).unwrap();
            m.verify_shard_hashes().unwrap();
            assert!(a.report_path("data").exists() && a.report_path("build").exists());
            let names: Vec<&str> = m.names().collect();
            assert!(names.iter().any(|n| n.starts_with("vocab/")), "{}", s.name);
            eprintln!("{} {variant}: {} entries, model {}", s.name, r.manifest.experts.len(), r.manifest.model_id);
        }
    }
}
