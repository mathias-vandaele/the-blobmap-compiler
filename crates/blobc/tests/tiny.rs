//! The tiny tier file is current, and the toy town and the via-way grid build and verify
//! end to end from it: zero mismatches, a golden file written then replayed, and every
//! applicable model mutant caught.

use std::path::Path;

use blobc::build::{Args, build};
use blobc::verify::tier::{TierFile, tiny};
use blobc::verify::{Options, verify};

#[test]
fn toy_and_viaway_verify() {
    let dir = std::env::temp_dir().join(format!("blobmap-conformance-{}", std::process::id()));
    let data = dir.join("in");
    let tf = tiny(&data).unwrap();
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tiers/tiny.toml");
    assert!(TierFile::read(&repo).unwrap() == tf, "tiers/tiny.toml is stale: run `tier tiny <dir> tiers/tiny.toml`");
    for name in ["toy", "viaway"] {
        let s = tf.inputs.iter().find(|s| s.name == name).unwrap();
        let out = dir.join(format!("{name}.model"));
        let a = Args {
            pbf: data.join(&s.pbf),
            overture: data.join(&s.overture),
            config: data.join(&s.config),
            out: out.clone(),
            extent: s.extent.as_ref().map(|e| data.join(e)),
            previous_report: None,
        };
        build(&a).unwrap();
        let o = Options {
            tier: "tiny".into(),
            tier_file: repo.clone(),
            data: data.clone(),
            golden: Some(dir.join("golden")),
            mutants: true,
            golden_only: false,
            work: dir.join("work"),
        };
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
