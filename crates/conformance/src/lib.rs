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

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use blobmap_fixtures::{Check, Scenario};
use blobmap_runtime::Runtime;

use crate::expected::Expected;
use crate::reference::Reference;
pub use crate::report::Report;

#[derive(Clone, Debug)]
pub struct Options {
    pub tier: String,
    /// The golden root: files live in `<root>/<tier>/<scenario>/`. Written when absent.
    pub golden: Option<PathBuf>,
    pub mutants: bool,
    /// Scratch space for regenerated inputs and mutant models.
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

/// Every check except golden replay and mutants.
fn run(rt: &Runtime, ex: &Expected, s: &Scenario, hashes: bool, rep: &mut Report) -> Result<()> {
    invariants::structure(rt, rep)?;
    if hashes {
        invariants::hashes(rt, rep)?;
    }
    checks::experts(rt, ex.r, rep);
    checks::snapping(rt, ex.r, rep)?;
    checks::snapping_brute(ex.r, 5000, rep);
    checks::recustomization(rt, rep)?;
    checks::distances(rt, ex, rep)?;
    let budget = format!("memory_budget_bytes = {}\n", rt.manifest.build.memory_budget_bytes);
    let cells = s.variants.iter().any(|v| v.name == "cells" && v.build_toml.contains(&budget));
    checks::scenario(rt, ex, &s.checks, cells, rep)?;
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
    if o.tier != "tiny" {
        bail!("tier {}: its inputs are not part of milestone 2 (only the tiny tier regenerates its inputs)", o.tier);
    }
    let sha = &rt.manifest.inputs.pbf.sha256;
    let inputs_dir = o.work.join("inputs");
    let (scenario, inputs) = reference::tiny_inputs(sha, &inputs_dir)?.with_context(|| format!("no tiny scenario has a .pbf with SHA-256 {sha}"))?;
    let r = Reference::build(&inputs)?;
    let m = r.modes.iter().map(|mode| dijkstra::matrix(&r, mode)).collect();
    let ex = Expected { r: &r, m };
    let mut report = Report::default();
    guarded(&mut report, "checks", |rep| run(&rt, &ex, &scenario, true, rep));
    let mut golden_written = None;
    let mut golden_dir = None;
    if let Some(root) = &o.golden {
        let dir = root.join(&o.tier).join(scenario.name);
        if !dir.join(golden::HEADER).exists() {
            golden::write(&dir, &o.tier, &rt.manifest.inputs_hash, &golden::queries(&ex, &named(&scenario.checks)))?;
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
                checks::distances(&broken, &ex, rep)?;
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
                    guarded(&mut rep, "checks", |rep| run(&b, &ex, &scenario, false, rep));
                    if let Some(d) = &golden_dir {
                        guarded(&mut rep, "golden replay", |rep| golden::replay(&b, d, rep));
                    }
                    rep.mismatches() > 0
                }
            };
            mutants.push((mu.name.to_string(), Some(caught)));
        }
    }
    Ok(Verified { scenario: scenario.name.to_string(), report, mutants, golden_written })
}
