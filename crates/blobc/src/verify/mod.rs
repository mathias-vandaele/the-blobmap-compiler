//! The conformance suite (SPEC §11). A model is checked against a reference computed from
//! its inputs alone: Dijkstra on the original graphs with the IDs inserted, re-snapping,
//! and the scenario's facts; plus the format's invariants and hashes, re-customization,
//! golden files, and broken models and runtimes that must fail.

pub mod accuracy;
pub mod checks;
pub mod components;
pub mod dijkstra;
pub mod expected;
pub mod fragments;
pub mod golden;
pub mod invariants;
pub mod mutants;
pub mod reference;
pub mod report;
pub mod tier;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use anyhow::Result;
use blobmap_format::runtime::{Fam, Runtime};

use crate::build::mem;
use crate::fixtures::Check;
use crate::verify::expected::Expected;
use crate::verify::reference::Reference;
pub use crate::verify::report::Report;

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

/// The highest peak of the phases ended inside the running one: a phase resets the
/// high-water mark, so an enclosing phase takes its nested phases' peaks from here.
static NESTED_PEAK: AtomicU64 = AtomicU64::new(0);

/// Runs a phase of verify and, as it ends, prints its time and memory, as a build does
/// per stage, so a long verify shows where it is. Phases may nest.
pub(crate) fn phase<T>(name: &str, f: impl FnOnce() -> T) -> T {
    let outer = NESTED_PEAK.swap(0, Ordering::Relaxed);
    mem::reset_peak();
    let t = Instant::now();
    let out = f();
    mem::trim();
    let peak = mem::peak_rss().unwrap_or(0).max(NESTED_PEAK.load(Ordering::Relaxed));
    NESTED_PEAK.store(outer.max(peak), Ordering::Relaxed);
    eprintln!("{name:>18} {:>9.3} s  peak {:>6} MiB  after {:>6} MiB", t.elapsed().as_secs_f64(), peak >> 20, mem::rss().unwrap_or(0) >> 20);
    out
}

/// Every check except golden replay and mutants.
/// Runs a phase, then releases the model's components it loaded; the phases over every
/// component hold one at a time.
fn released(rt: &Runtime, name: &str, rep: &mut Report, f: impl FnOnce(&mut Report) -> Result<()>) -> Result<()> {
    let out = phase(name, || f(rep));
    rt.release();
    out
}

/// The checks on the model's bytes alone, before any reference is built.
fn model_checks(rt: &Runtime, hashes: bool, rep: &mut Report) -> Result<()> {
    released(rt, "structure", rep, |rep| invariants::structure(rt, rep))?;
    if hashes {
        released(rt, "hashes", rep, |rep| invariants::hashes(rt, rep))?;
    }
    released(rt, "recustomization", rep, |rep| checks::recustomization(rt, rep))
}

/// The checks against the reference, for the families it holds; the scenario's facts with
/// the last family.
fn reference_checks(rt: &Runtime, ex: &Expected, s: &Facts, n: &Samples, rep: &mut Report) -> Result<()> {
    let p = |name: &str| format!("{} {name}", ex.r.fams.iter().map(|f| f.name()).collect::<Vec<_>>().join("+"));
    released(rt, &p("components"), rep, |rep| {
        checks::components(rt, ex.r, rep);
        Ok(())
    })?;
    released(rt, &p("snapping"), rep, |rep| checks::snapping(rt, ex.r, rep))?;
    released(rt, &p("snapping brute"), rep, |rep| {
        checks::snapping_brute(ex.r, n.brute, rep);
        Ok(())
    })?;
    released(rt, &p("distances"), rep, |rep| checks::distances(rt, ex, n, rep))?;
    released(rt, &p("via copies"), rep, |rep| checks::via_copies(rt, ex, rep))?;
    released(rt, &p("via segments"), rep, |rep| checks::via_segments(rt, ex, rep))?;
    if ex.r.has(Fam::Active) {
        released(rt, &p("scenario"), rep, |rep| checks::scenario(rt, ex, &s.checks, s.cells, rep))?;
    }
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
    let scenario = if o.tier == "tiny" { crate::fixtures::by_name(&set.name) } else { None };
    let budget = format!("memory_budget_bytes = {}\n", rt.manifest.build.memory_budget_bytes);
    let scenario = Facts {
        cells: scenario.as_ref().is_some_and(|s| s.variants.iter().any(|v| v.name == "cells" && v.build_toml.contains(&budget))),
        checks: scenario.map(|s| s.checks).unwrap_or_default(),
    };
    let n = Samples::for_tier(&o.tier);
    let mut report = Report::default();
    guarded(&mut report, "checks", |rep| model_checks(&rt, true, rep));
    // The tiny tier is checked with both families at once, as its facts and its mutants span
    // them; a larger tier one family at a time, each reference dropped before the next. The
    // golden queries span the families too (chained trips), so writing them takes both.
    let golden_dir = o.golden.as_ref().map(|root| root.join(&o.tier).join(&set.name));
    let write_golden = golden_dir.as_ref().is_some_and(|d| !d.join(golden::HEADER).exists());
    const BOTH: &[Fam] = &[Fam::Car, Fam::Active];
    // Each pass: its families, and whether it runs the checks (else it only writes the golden
    // queries and runs the mutants).
    let mut passes: Vec<(&[Fam], bool)> = if o.tier == "tiny" { vec![(BOTH, true)] } else { vec![(&[Fam::Car], true), (&[Fam::Active], true)] };
    if (write_golden || o.mutants) && passes.len() > 1 {
        passes.push((BOTH, false));
    }
    let mut golden_written = None;
    let mut mutants = Vec::new();
    for (fams, check) in passes {
        let r = phase(&format!("{} reference", fams.iter().map(|f| f.name()).collect::<Vec<_>>().join("+")), || Reference::build(&inputs, fams))?;
        let ex = phase("dijkstra graphs", || Expected::new(&r));
        if check {
            guarded(&mut report, "checks", |rep| reference_checks(&rt, &ex, &scenario, &n, rep));
        }
        if fams != BOTH {
            continue;
        }
        if let (Some(dir), true) = (&golden_dir, write_golden) {
            let q = phase("golden queries", || golden::queries(&ex, &named(&scenario.checks), &n));
            golden::write(dir, &o.tier, &rt.manifest.inputs_hash, &q)?;
            golden_written = Some(dir.clone());
        }
        if o.mutants {
            mutants = run_mutants(model, o, &ex, &scenario, &n, golden_dir.as_deref())?;
        }
    }
    mem::trim();
    // The replay needs the model and the golden files only.
    if let Some(dir) = &golden_dir {
        phase("golden replay", || guarded(&mut report, "golden replay", |rep| golden::replay(&rt, dir, rep)));
    }
    Ok(Verified { scenario: set.name.clone(), report, mutants, golden_written })
}

/// Broken runtimes and models, each of which some check must catch.
fn run_mutants(model: &Path, o: &Options, ex: &Expected, scenario: &Facts, n: &Samples, golden_dir: Option<&Path>) -> Result<Vec<(String, Option<bool>)>> {
    let mut mutants = Vec::new();
    for (name, faults) in mutants::RUNTIME_FAULTS {
        let mut broken = Runtime::open(model)?;
        broken.faults = *faults;
        let mut rep = Report::default();
        guarded(&mut rep, "checks", |rep| {
            checks::distances(&broken, ex, n, rep)?;
            checks::scenario(&broken, ex, &scenario.checks, false, rep)
        });
        if let Some(d) = golden_dir {
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
                guarded(&mut rep, "checks", |rep| {
                    model_checks(&b, false, rep)?;
                    reference_checks(&b, ex, scenario, n, rep)
                });
                if let Some(d) = golden_dir {
                    guarded(&mut rep, "golden replay", |rep| golden::replay(&b, d, rep));
                }
                rep.mismatches() > 0
            }
        };
        mutants.push((mu.name.to_string(), Some(caught)));
    }
    Ok(mutants)
}
