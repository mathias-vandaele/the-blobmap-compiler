//! `blobc build | verify | inspect` (SPEC §10), thin over the `build`, `conformance` and
//! `format` crates.

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "blobc", version, about = "The BlobMap world-model compiler")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Builds a model from a .pbf and an Overture release.
    Build {
        #[arg(long)]
        pbf: PathBuf,
        #[arg(long)]
        overture: PathBuf,
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        extent: Option<PathBuf>,
        #[arg(long)]
        previous_report: Option<PathBuf>,
    },
    /// Checks a model with the conformance suite (SPEC §11).
    Verify {
        model: PathBuf,
        #[arg(long)]
        tier: String,
        /// The directory holding the tier's inputs, as its tier file lists them.
        #[arg(long)]
        data: PathBuf,
        /// The directory of tier files, `<tier>.toml` each.
        #[arg(long, default_value = "tiers")]
        tiers: PathBuf,
        /// Golden root (`<root>/<tier>/<inputs>/`): replayed, or written when absent.
        #[arg(long)]
        golden: Option<PathBuf>,
        /// Also run the broken models and runtimes, which must fail.
        #[arg(long)]
        mutants: bool,
    },
    /// Prints a model's manifest summary.
    Inspect { model: PathBuf },
}

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<bool> {
    match Cli::parse().cmd {
        Cmd::Build { pbf, overture, config, out, extent, previous_report } => {
            let a = blobmap_build::Args { pbf, overture, config, out, extent, previous_report };
            let r = blobmap_build::build(&a)?;
            println!("model_id {}", r.manifest.model_id);
            println!("inputs_hash {}", r.manifest.inputs_hash);
            Ok(true)
        }
        Cmd::Verify { model, tier, data, tiers, golden, mutants } => {
            let work = std::env::temp_dir().join(format!("blobc-verify-{}", std::process::id()));
            let tier_file = tiers.join(format!("{tier}.toml"));
            let o = blobmap_conformance::Options { tier, tier_file, data, golden, mutants, work: work.clone() };
            let v = blobmap_conformance::verify(&model, &o);
            let _ = std::fs::remove_dir_all(&work);
            let v = v?;
            println!("inputs {}", v.scenario);
            print!("{}", v.report);
            if let Some(d) = &v.golden_written {
                println!("golden written to {}", d.display());
            }
            let mut ok = v.report.mismatches() == 0;
            for (name, caught) in &v.mutants {
                let s = match caught {
                    Some(true) => "caught",
                    Some(false) => "NOT CAUGHT",
                    None => "not applicable",
                };
                println!("mutant {s:>14}  {name}");
            }
            // A model mutant that applies must be caught; runtime faults are judged over the
            // whole tier, since a scenario may not exercise every one.
            ok &= v.mutants.iter().filter(|(n, _)| !n.starts_with("runtime")).all(|(_, c)| *c != Some(false));
            println!("{} mismatches", v.report.mismatches());
            Ok(ok)
        }
        Cmd::Inspect { model } => {
            let m = blobmap_format::Model::open(&model)?;
            let mf = blobmap_format::manifest::Manifest::from_value(&m.manifest)?;
            println!("model_id     {}", mf.model_id);
            println!("inputs_hash  {}", mf.inputs_hash);
            println!("spec         {}  format {}", mf.spec_version, mf.format_version);
            println!("compiler     {} {} {}{}", mf.compiler.name, mf.compiler.version, mf.compiler.commit, if mf.compiler.dirty { " (dirty)" } else { "" });
            println!("pbf          {} ({} bytes)", mf.inputs.pbf.name, mf.inputs.pbf.bytes);
            println!("overture     {}", mf.inputs.overture.release);
            for e in &mf.experts {
                println!(
                    "{:>6} {:>6} experts {:>6}+{:<4} nodes {:>9} links {:>10} levels {:>4} cells {:>3} shards {}",
                    e.family,
                    e.kind,
                    e.experts[0],
                    e.experts[1],
                    e.nodes,
                    e.links,
                    e.levels,
                    e.cells,
                    e.shards.len()
                );
            }
            Ok(true)
        }
    }
}
