//! `blobc build | verify | inspect` (SPEC §10), thin over the `build` and `verify` modules
//! and `blobmap_format`; and the tools of the tiers and of milestones 3 and 4, each with its own
//! arguments.

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
    /// Writes every tiny-tier scenario's inputs: `fixtures <out dir> [scenario …]`.
    Fixtures(Tool),
    /// Writes a city extract of a .pbf (PLAN, milestone 3).
    Extract(Tool),
    /// Writes tier files: `tier tiny …` or `tier add …`.
    Tier(Tool),
    /// Accuracy against cached OSRM times (PLAN, milestone 3).
    Accuracy(Tool),
    /// The components that serve IDs, per mode.
    Components(Tool),
    /// What the small-component rule moves, and whether off an island (SPEC §6).
    Fragments(Tool),
    /// The build's input stages alone, with their peak RSS (PLAN, milestone 4).
    Inputs(Tool),
}

/// A tool's own arguments, passed through as given.
#[derive(clap::Args)]
struct Tool {
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    args: Vec<String>,
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
            let a = blobc::build::Args { pbf, overture, config, out, extent, previous_report };
            let r = blobc::build::build(&a)?;
            println!("model_id {}", r.manifest.model_id);
            println!("inputs_hash {}", r.manifest.inputs_hash);
            Ok(true)
        }
        Cmd::Verify { model, tier, data, tiers, golden, mutants } => {
            let work = blobc::verify::work_dir(&model, "verify");
            let tier_file = tiers.join(format!("{tier}.toml"));
            let o = blobc::verify::Options { tier, tier_file, data, golden, mutants, work: work.clone() };
            let v = blobc::verify::verify(&model, &o);
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
        Cmd::Fixtures(t) => blobc::fixtures::main(t.args).map(|()| true),
        Cmd::Extract(t) => blobc::fixtures::extract::main(t.args).map(|()| true),
        Cmd::Tier(t) => blobc::verify::tier::main(t.args).map(|()| true),
        Cmd::Accuracy(t) => blobc::verify::accuracy::main(t.args).map(|()| true),
        Cmd::Components(t) => blobc::verify::components::main(t.args).map(|()| true),
        Cmd::Fragments(t) => blobc::verify::fragments::main(t.args).map(|()| true),
        Cmd::Inputs(t) => blobc::verify::inputs::main(t.args).map(|()| true),
        Cmd::Inspect { model } => {
            let m = blobmap_format::Model::open(&model)?;
            let mf = blobmap_format::manifest::Manifest::from_value(&m.manifest)?;
            println!("model_id     {}", mf.model_id);
            println!("inputs_hash  {}", mf.inputs_hash);
            println!("spec         {}  format {}", mf.spec_version, mf.format_version);
            println!("compiler     {} {} {}{}", mf.compiler.name, mf.compiler.version, mf.compiler.commit, if mf.compiler.dirty { " (dirty)" } else { "" });
            println!("pbf          {} ({} bytes)", mf.inputs.pbf.name, mf.inputs.pbf.bytes);
            println!("overture     {}", mf.inputs.overture.release);
            for (f, x) in &mf.families {
                println!("{f:<12} {} nodes, {} links, {} components, {} packs", x.nodes, x.links, x.components, x.packs);
            }
            // Components built as cells: the top block's levels and the deepest cell's (SPEC §5).
            let rt = blobmap_format::runtime::Runtime::open(&model)?;
            for (fam, t) in &rt.tables {
                for x in 0..t.components() {
                    let bs = t.block_first[x] as usize..t.block_first[x + 1] as usize;
                    if bs.len() < 2 {
                        continue;
                    }
                    let top: Vec<usize> = bs.clone().filter(|&b| t.b_kind[b] == 1).collect();
                    let cells: Vec<usize> = bs.filter(|&b| t.b_kind[b] == 2).collect();
                    let deepest = cells.iter().copied().max_by_key(|&b| (t.b_levels[b], std::cmp::Reverse(b))).unwrap();
                    println!(
                        "{:>6} component {x}: levels {} = top {} ({} nodes) + deepest cell {} ({} nodes); {} cells of {}..{} nodes",
                        fam.name(),
                        t.levels(x as u32),
                        top.iter().map(|&b| t.b_levels[b]).sum::<u32>(),
                        top.iter().map(|&b| t.b_nodes[b]).sum::<u32>(),
                        t.b_levels[deepest],
                        t.b_nodes[deepest],
                        cells.len(),
                        cells.iter().map(|&b| t.b_nodes[b]).min().unwrap(),
                        cells.iter().map(|&b| t.b_nodes[b]).max().unwrap(),
                    );
                }
            }
            Ok(true)
        }
    }
}
