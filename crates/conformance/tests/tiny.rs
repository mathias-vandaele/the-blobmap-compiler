//! The toy town and the via-way grid, built and verified end to end: zero mismatches, a
//! golden file written then replayed, and every applicable model mutant caught.

use blobmap_build::{Args, build};
use blobmap_conformance::{Options, verify};

#[test]
fn toy_and_viaway_verify() {
    let dir = std::env::temp_dir().join(format!("blobmap-conformance-{}", std::process::id()));
    for name in ["toy", "viaway"] {
        let s = blobmap_fixtures::by_name(name).unwrap();
        let w = s.write(&dir.join("in")).unwrap();
        let out = dir.join(format!("{name}.model"));
        let a = Args { pbf: w.pbf, overture: w.overture, config: w.configs[0].1.clone(), out: out.clone(), extent: w.extent, previous_report: None };
        build(&a).unwrap();
        let o = Options { tier: "tiny".into(), golden: Some(dir.join("golden")), mutants: true, work: dir.join("work") };
        let v = verify(&out, &o).unwrap();
        assert_eq!(v.report.mismatches(), 0, "{name}:\n{}", v.report);
        assert!(v.golden_written.is_some());
        for (m, caught) in &v.mutants {
            assert!(m.starts_with("runtime") || *caught != Some(false), "{name}: mutant {m} not caught");
        }
        if name == "viaway" {
            assert!(v.mutants.iter().all(|(_, c)| *c == Some(true)), "{:?}", v.mutants);
        }
        // Replaying the golden file just written.
        let v = verify(&out, &Options { mutants: false, ..o }).unwrap();
        assert!(v.golden_written.is_none() && v.report.mismatches() == 0);
    }
}
