//! The conformance suite (SPEC §11). A model is checked against a reference computed from
//! its inputs alone: Dijkstra on the original graphs with the IDs inserted, re-snapping,
//! and the scenario's facts; plus the format's invariants and hashes, re-customization,
//! golden files, and broken models and runtimes that must fail.

pub mod checks;
pub mod dijkstra;
pub mod expected;
pub mod golden;
pub mod invariants;
pub mod mutants;
pub mod reference;
pub mod report;
pub mod tier;

use std::path::{Path, PathBuf};

use anyhow::Result;
use blobmap_fixtures::Check;
use blobmap_runtime::Runtime;

use crate::expected::Expected;
use crate::reference::Reference;
pub use crate::report::Report;

#[derive(Clone, Debug)]
pub struct Options {
    pub tier: String,
    /// The tier file (SPEC §10, verify).
    pub tier_file: PathBuf,
    /// The directory holding the tier's inputs.
    pub data: PathBuf,
    /// The golden root: files live in `<root>/<tier>/<inputs>/`. Written when absent.
    pub golden: Option<PathBuf>,
    pub mutants: bool,
    /// Scratch space for mutant models.
    pub work: PathBuf,
}

pub struct Verified {
    pub scenario: String,
    pub report: Report,
    /// Each mutant, and whether some check caught it (`None`: not applicable here).
    pub mutants: Vec<(String, Option<bool>)>,
    pub golden_written: Option<PathBuf>,
}

/// IDs a scenario's facts name.
fn named(checks: &[Check]) -> Vec<String> {
    let mut out = Vec::new();
    for c in checks {
        let s = format!("{c:?}");
        let mut rest = s.as_str();
        while let Some(k) = rest.find('"') {
            let tail = &rest[k + 1..];
            let Some(e) = tail.find('"') else { break };
            let w = &tail[..e];
            if w.len() == 36 && w.as_bytes()[8] == b'-' && !out.iter().any(|x: &String| x == w) {
                out.push(w.to_string());
            }
            rest = &tail[e + 1..];
        }
    }
    out
}

/// How much of a tier the sampled checks and golden queries cover: the tiny tier is
/// checked whole, larger tiers on even samples.
#[derive(Clone, Debug)]
pub struct Samples {
    /// Sources of single-source columns, per mode.
    pub columns: usize,
    /// IDs whose every pair is asked for `travel_time`, per mode.
    pub pairs: usize,
    /// IDs re-snapped without the grid, per mode.
    pub brute: usize,
    /// Aliases and category keys in the golden queries.
    pub aliases: usize,
    pub categories: usize,
}

impl Samples {
    pub fn for_tier(tier: &str) -> Samples {
        if tier == "tiny" {
            Samples { columns: 48, pairs: 20, brute: 5000, aliases: usize::MAX, categories: usize::MAX }
        } else {
            Samples { columns: 12, pairs: 12, brute: 200, aliases: 16, categories: 8 }
        }
    }
}

/// A tiny-tier scenario's facts, and whether this model is its cells variant.
struct Facts {
    checks: Vec<Check>,
    cells: bool,
}

/// Every check except golden replay and mutants.
fn run(rt: &Runtime, ex: &Expected, s: &Facts, n: &Samples, hashes: bool, rep: &mut Report) -> Result<()> {
    invariants::structure(rt, rep)?;
    if hashes {
        invariants::hashes(rt, rep)?;
    }
    checks::experts(rt, ex.r, rep);
    checks::snapping(rt, ex.r, rep)?;
    checks::snapping_brute(ex.r, n.brute, rep);
    checks::recustomization(rt, rep)?;
    checks::distances(rt, ex, n, rep)?;
    checks::via_copies(rt, ex, rep)?;
    checks::via_segments(rt, ex, rep)?;
    checks::scenario(rt, ex, &s.checks, s.cells, rep)?;
    Ok(())
}

/// Runs a check suite, turning an error into a mismatch of its own.
fn guarded(rep: &mut Report, name: &str, f: impl FnOnce(&mut Report) -> Result<()>) {
    if let Err(e) = f(rep) {
        rep.check(name).fail(format!("{e:#}"));
    }
}

pub fn verify(model: &Path, o: &Options) -> Result<Verified> {
    let rt = Runtime::open(model)?;
    let tf = tier::TierFile::read(&o.tier_file)?;
    anyhow::ensure!(tf.tier == o.tier, "{}: a tier file for {}, not {}", o.tier_file.display(), tf.tier, o.tier);
    let set = tf.find(&rt.manifest)?;
    let inputs = set.check(&o.data, &rt.manifest)?;
    // The tiny tier's sets are fixture scenarios, with facts of their own.
    let scenario = if o.tier == "tiny" { blobmap_fixtures::by_name(&set.name) } else { None };
    let budget = format!("memory_budget_bytes = {}\n", rt.manifest.build.memory_budget_bytes);
    let scenario = Facts {
        cells: scenario.as_ref().is_some_and(|s| s.variants.iter().any(|v| v.name == "cells" && v.build_toml.contains(&budget))),
        checks: scenario.map(|s| s.checks).unwrap_or_default(),
    };
    let r = Reference::build(&inputs)?;
    let ex = Expected::new(&r);
    let n = Samples::for_tier(&o.tier);
    let mut report = Report::default();
    guarded(&mut report, "checks", |rep| run(&rt, &ex, &scenario, &n, true, rep));
    let mut golden_written = None;
    let mut golden_dir = None;
    if let Some(root) = &o.golden {
        let dir = root.join(&o.tier).join(&set.name);
        if !dir.join(golden::HEADER).exists() {
            golden::write(&dir, &o.tier, &rt.manifest.inputs_hash, &golden::queries(&ex, &named(&scenario.checks), &n))?;
            golden_written = Some(dir.clone());
        }
        guarded(&mut report, "golden replay", |rep| golden::replay(&rt, &dir, rep));
        golden_dir = Some(dir);
    }
    let mut mutants = Vec::new();
    if o.mutants {
        for (name, faults) in mutants::RUNTIME_FAULTS {
            let mut broken = Runtime::open(model)?;
            broken.faults = *faults;
            let mut rep = Report::default();
            guarded(&mut rep, "checks", |rep| {
                checks::distances(&broken, &ex, &n, rep)?;
                checks::scenario(&broken, &ex, &scenario.checks, false, rep)
            });
            if let Some(d) = &golden_dir {
                guarded(&mut rep, "golden replay", |rep| golden::replay(&broken, d, rep));
            }
            mutants.push((name.to_string(), Some(rep.mismatches() > 0)));
        }
        for mu in mutants::MODEL_MUTANTS {
            let Some(dir) = mutants::apply(model, &o.work.join("mutants"), mu)? else {
                mutants.push((mu.name.to_string(), None));
                continue;
            };
            let caught = match Runtime::open(&dir) {
                Err(_) => true,
                Ok(b) => {
                    let mut rep = Report::default();
                    guarded(&mut rep, "checks", |rep| run(&b, &ex, &scenario, &n, false, rep));
                    if let Some(d) = &golden_dir {
                        guarded(&mut rep, "golden replay", |rep| golden::replay(&b, d, rep));
                    }
                    rep.mismatches() > 0
                }
            };
            mutants.push((mu.name.to_string(), Some(caught)));
        }
    }
    Ok(Verified { scenario: set.name.clone(), report, mutants, golden_written })
}
