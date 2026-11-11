//! `blobc build` (SPEC §10): the stages from the inputs to a model directory, behind a
//! `Stage` trait so each stage's wall time and peak RSS land in the build report.

pub mod family;
pub mod hier;
pub mod layout;
pub mod mem;
pub mod report;
pub mod road;
pub mod vocab;

use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use blobmap_format::hash::{Hasher, ROAD_TAG, VOCAB_TAG, inputs_hash, model_id, of_hashes};
use blobmap_format::manifest::{self as mf, Manifest};
use blobmap_format::{Hash, ModelWriter, NONE, ShardLimits};
use serde_json::json;
use sha2::Digest;

use crate::build::family::{Active, Car, Family};
use crate::build::hier::{Built, FixedRanks};
use crate::build::layout::Layout;
use crate::build::vocab::{Families, Ids, ModeSnap};
use crate::graph::{ActiveGraph, CarGraph};
use crate::osm::{Network, Osm, Pieces};
use crate::overture::{Catalog, Polygon, Release};
use crate::profile::active::{ActiveMode, ActiveProfile, Mode};
use crate::profile::car::{CarProfile, WayAttrs};
use crate::profile::config::Config;
use crate::snap::islands::Islands;

/// The SPEC version this compiler builds to (FORMAT §3 `spec_version`).
pub fn spec_version() -> &'static str {
    const SPEC: &str = include_str!("../../../../SPEC.md");
    SPEC.lines().find_map(|l| l.strip_prefix("SPEC version ")).map(str::trim).expect("SPEC.md has a version line")
}

pub fn compiler() -> mf::Compiler {
    mf::Compiler {
        name: "blobc".into(),
        version: env!("CARGO_PKG_VERSION").into(),
        commit: env!("BLOBC_COMMIT").into(),
        dirty: env!("BLOBC_DIRTY") == "true",
    }
}

/// `blobc build`'s arguments.
#[derive(Clone, Debug)]
pub struct Args {
    pub pbf: PathBuf,
    pub overture: PathBuf,
    pub config: PathBuf,
    pub out: PathBuf,
    pub extent: Option<PathBuf>,
    pub previous_report: Option<PathBuf>,
}

impl Args {
    /// The reports are written next to the model directory, never inside it (SPEC §8).
    pub fn report_path(&self, kind: &str) -> PathBuf {
        let name = self.out.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "model".into());
        self.out.with_file_name(format!("{name}.{kind}-report.json"))
    }

    /// The build's scratch files (the node locations), next to the model directory and
    /// removed when the build ends.
    pub fn work_dir(&self) -> PathBuf {
        let name = self.out.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "model".into());
        self.out.with_file_name(format!("{name}.work"))
    }
}

/// One step of the build.
pub trait Stage<I> {
    type Output;
    fn name(&self) -> &'static str;
    fn run(&self, input: I) -> Result<Self::Output>;
}

#[derive(Clone, Debug)]
pub struct StageReport {
    pub name: &'static str,
    pub wall_s: f64,
    pub peak_rss_bytes: Option<u64>,
}

/// Runs stages and records their wall time and peak RSS.
#[derive(Default)]
pub struct Runner {
    pub stages: Vec<StageReport>,
}

impl Runner {
    pub fn run<I, S: Stage<I>>(&mut self, s: &S, input: I) -> Result<S::Output> {
        mem::reset_peak();
        let t = Instant::now();
        let out = s.run(input).with_context(|| format!("stage {}", s.name()))?;
        mem::trim();
        let r = StageReport { name: s.name(), wall_s: t.elapsed().as_secs_f64(), peak_rss_bytes: mem::peak_rss() };
        // As each stage ends, so a long build shows where it is.
        let (rss, anon) = (mem::rss().unwrap_or(0) >> 20, mem::anon().unwrap_or(0) >> 20);
        eprintln!("{:>18} {:>9.3} s  peak {:>6} MiB  after {rss:>6} MiB, anon {anon:>6} MiB", r.name, r.wall_s, r.peak_rss_bytes.unwrap_or(0) >> 20);
        self.stages.push(r);
        Ok(out)
    }
}

/// A file's SHA-256 in hex, and its size in bytes.
pub fn sha256_file(path: &Path) -> Result<(String, u64)> {
    let mut f = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut h = sha2::Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut n = 0u64;
    loop {
        let k = f.read(&mut buf)?;
        if k == 0 {
            break;
        }
        h.update(&buf[..k]);
        n += k as u64;
    }
    Ok((hex::encode(h.finalize()), n))
}

fn file_name(p: &Path) -> String {
    p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
}

/// ISO 8601 UTC of Unix seconds.
pub fn iso8601(secs: i64) -> String {
    let (days, rem) = (secs.div_euclid(86400), secs.rem_euclid(86400));
    // Civil from days (H. Hinnant).
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + (m <= 2) as i64;
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z", rem / 3600, rem / 60 % 60, rem % 60)
}

/// Everything read before any graph is built.
pub struct InputsRead {
    pub config: Config,
    pub release: Release,
    pub polygon: Option<Polygon>,
    pub manifest: mf::Inputs,
    pub config_files: mf::Config,
}

pub struct ReadInputs;

impl Stage<&Args> for ReadInputs {
    type Output = InputsRead;
    fn name(&self) -> &'static str {
        "inputs"
    }
    fn run(&self, a: &Args) -> Result<InputsRead> {
        let config = Config::load(&a.config)?;
        let release = Release::open(&a.overture)?;
        let (sha, bytes) = sha256_file(&a.pbf)?;
        let timestamp = crate::osm::header_timestamp(&a.pbf)?.map(iso8601);
        let rel = |p: &Path| -> Result<mf::NamedFile> {
            let name = p.strip_prefix(&release.dir).unwrap_or(p).to_string_lossy().replace('\\', "/");
            Ok(mf::NamedFile { name, sha256: sha256_file(p)?.0 })
        };
        let overture = mf::Overture { release: release.name.clone(), files: release.files().iter().map(|p| rel(p)).collect::<Result<_>>()? };
        let mut tax_files = Vec::new();
        for f in ["taxonomy.csv", "basic_categories.csv", "mapping_sheet.csv"] {
            let p = release.taxonomy_dir().join(f);
            if p.exists() {
                tax_files.push(rel(&p)?);
            }
        }
        let (polygon, extent) = match &a.extent {
            Some(p) => {
                let text = std::fs::read_to_string(p).with_context(|| format!("reading {}", p.display()))?;
                (Some(Polygon::parse_poly(&text)?), Some(mf::NamedFile { name: file_name(p), sha256: sha256_file(p)?.0 }))
            }
            None => (None, None),
        };
        let manifest = mf::Inputs {
            pbf: mf::Pbf { name: file_name(&a.pbf), bytes, sha256: sha, timestamp },
            overture,
            taxonomy: mf::Taxonomy { version: release.name.clone(), files: tax_files },
            extent,
        };
        let named = |path: &str, sha: &str| mf::NamedFile { name: file_name(Path::new(path)), sha256: sha.to_string() };
        let config_files = mf::Config {
            profiles: mf::Profiles { car: named(&config.car.source, &config.car.sha256), active: named(&config.active.source, &config.active.sha256) },
            place_rules: named(&config.places.source, &config.places.sha256),
        };
        Ok(InputsRead { config, release, polygon, manifest, config_files })
    }
}

/// The `.pbf`'s node locations, on disk in a work directory (SPEC §10, memory).
pub struct ReadNodes;

impl Stage<(&Path, &Path, &Config)> for ReadNodes {
    type Output = Osm;
    fn name(&self) -> &'static str {
        "nodes"
    }
    fn run(&self, (pbf, work, c): (&Path, &Path, &Config)) -> Result<Osm> {
        Osm::open(pbf, work, c)
    }
}

pub struct CarNetwork;

impl Stage<(&Osm, &Config)> for CarNetwork {
    type Output = (Pieces, CarGraph);
    fn name(&self) -> &'static str {
        "car-network"
    }
    fn run(&self, (osm, c): (&Osm, &Config)) -> Result<(Pieces, CarGraph)> {
        let net = Network::read(osm, &osm.car_obstacles, &CarProfile::new(&c.car.config), &c.car.config.snap, Some(&osm.dir().join("car")))?;
        let p = net.pieces(&[|a: &WayAttrs| *a])?;
        drop(net);
        let (r, _) = osm.restrictions()?;
        let mut g = CarGraph::build(&p, &c.car.config.turns, &r);
        // The graph's arrays on disk, like the pieces' (SPEC §10, memory).
        g.spill(&osm.dir().join("car-graph"))?;
        Ok((p, g))
    }
}

pub struct ActiveNetwork;

impl Stage<(&Osm, &Config)> for ActiveNetwork {
    type Output = (Pieces, ActiveGraph);
    fn name(&self) -> &'static str {
        "active-network"
    }
    fn run(&self, (osm, c): (&Osm, &Config)) -> Result<(Pieces, ActiveGraph)> {
        let prof = ActiveProfile::new(&c.active.config);
        let net = Network::read(osm, &osm.active_obstacles, &prof, &c.active.config.snap, Some(&osm.dir().join("active")))?;
        let modes: Vec<ActiveMode> = [Mode::Bike, Mode::Walk].map(|mode| ActiveMode { mode, cfg: prof.cfg }).to_vec();
        let p = net.pieces(&modes)?;
        drop(net);
        let mut g = ActiveGraph::build(&p);
        g.spill(&osm.dir().join("active-graph"))?;
        Ok((p, g))
    }
}

/// Every component's order and blocks (SPEC §5), a stage of its own so that the build report
/// shows the memory of nested dissection.
pub struct Orders(pub &'static str);

impl Stage<(&dyn Family, &Config, Option<&FixedRanks>)> for Orders {
    type Output = hier::Ordered;
    fn name(&self) -> &'static str {
        self.0
    }
    fn run(&self, (f, c, fixed): (&dyn Family, &Config, Option<&FixedRanks>)) -> Result<hier::Ordered> {
        let budget = c.build.build.memory_budget_bytes;
        let max_cell = (budget / hier::BYTES_PER_NODE).max(1) as usize;
        let nd_max = (c.build.build.order_memory_bytes / hier::ND_BYTES_PER_NODE) as usize;
        hier::order(f, max_cell, nd_max, fixed)
    }
}

/// The family's hierarchy, one component at a time, then its packs.
pub struct Hierarchies(pub &'static str);

impl Stage<(&dyn Family, &Config, hier::Ordered)> for Hierarchies {
    type Output = (Built, Layout);
    fn name(&self) -> &'static str {
        self.0
    }
    fn run(&self, (f, c, o): (&dyn Family, &Config, hier::Ordered)) -> Result<(Built, Layout)> {
        let b = hier::build(f, o)?;
        let l = layout::plan(f, &b, c.build.build.pack_max_nodes)?;
        Ok((b, l))
    }
}

pub struct ReadCatalog;

impl Stage<(&Release, &Config, Option<&Polygon>)> for ReadCatalog {
    type Output = Catalog;
    fn name(&self) -> &'static str {
        "catalog"
    }
    fn run(&self, (r, c, poly): (&Release, &Config, Option<&Polygon>)) -> Result<Catalog> {
        match poly {
            Some(p) => Catalog::read(r, &c.places.config, &|lat, lon| p.contains(lat, lon)),
            None => Catalog::read(r, &c.places.config, &|_, _| true),
        }
    }
}

pub struct Snap;

impl Stage<(&Families<'_>, &Catalog, bool, &Path)> for Snap {
    type Output = (Vec<ModeSnap>, Ids);
    fn name(&self) -> &'static str {
        "snap"
    }
    /// Each mode's outcomes go to a column of `dir` (SPEC §10, memory).
    fn run(&self, (f, c, polygon, dir): (&Families, &Catalog, bool, &Path)) -> Result<(Vec<ModeSnap>, Ids)> {
        let na = c.addresses.len();
        let point = |i: usize| match c.addresses.get(i) {
            Some(a) => (a.lat, a.lon),
            None => (c.places[i - na].lat, c.places[i - na].lon),
        };
        let snaps = vocab::snap(f, na + c.places.len(), &point, Some(dir))?;
        let ids = vocab::select(c, &snaps, polygon);
        Ok((snaps, ids))
    }
}

/// The island polygons of the `.pbf`, for the data report (SPEC §3).
pub struct ReadIslands;

impl Stage<&Osm> for ReadIslands {
    type Output = Islands;
    fn name(&self) -> &'static str {
        "islands"
    }
    fn run(&self, osm: &Osm) -> Result<Islands> {
        Islands::read(osm)
    }
}

/// The largest pack of a family (FORMAT §4 limits), for the build report.
#[derive(Clone, Debug, Default)]
pub struct PackSize {
    pub pack: u32,
    pub rows: u64,
    pub links: u64,
    pub arcs: u64,
}

/// A family's road units, written (FORMAT §2.4): what the vocabulary, the manifest and the
/// reports need of its hierarchy, which is then dropped.
pub struct RoadOut {
    /// Each node's layout id.
    pub layout_of: Vec<u32>,
    /// Each arc's index among its row's arcs (FORMAT §5.4), if the vocabulary needs it.
    pub arc_local: Vec<u32>,
    /// Each road tensor's hash, by full name (FORMAT §9).
    pub hashes: BTreeMap<String, Hash>,
    pub largest: PackSize,
    pub links: u64,
    pub components: u64,
    pub packs: u64,
    /// The bytes of its shard files, tables and packs.
    pub bytes: u64,
    /// The largest component and every component built as cells: its links, its sweep layers
    /// (FORMAT's level count) and its elimination tree's height (SPEC §13, milestone 4).
    pub hierarchies: Vec<serde_json::Value>,
}

/// A family's road units, written as soon as its hierarchy is built so that the hierarchy is
/// dropped before the next family's: its tables in slot `slots.0`, its packs in `slots.1`
/// (FORMAT §2.4 puts every family's tables before the packs).
pub struct WriteRoad {
    pub name: &'static str,
    pub slots: (u32, u32),
    /// Whether the vocabulary needs the arcs' local indices (ACTIVE, FORMAT §7.2).
    pub arc_local: bool,
}

impl Stage<(&dyn Family, Built, Layout, &mut ModelWriter)> for WriteRoad {
    type Output = RoadOut;
    fn name(&self) -> &'static str {
        self.name
    }
    fn run(&self, (fam, b, l, writer): (&dyn Family, Built, Layout, &mut ModelWriter)) -> Result<RoadOut> {
        let mut hashes = BTreeMap::new();
        writer.set_slot(self.slots.0)?;
        let t = writer.write_unit(road::tables(fam, &b, &l))?;
        let mut bytes = t.bytes;
        hashes.extend(t.hashes);
        writer.set_slot(self.slots.1)?;
        let mut largest = PackSize::default();
        for k in 0..l.packs.len() {
            let pk = &l.packs[k];
            let size = PackSize { pack: k as u32, rows: pk.rows.len() as u64, links: pk.links, arcs: pk.arcs.len() as u64 };
            if (size.rows, size.links) > (largest.rows, largest.links) {
                largest = size;
            }
            let w = writer.write_unit(road::pack(fam, &b, &l, k))?;
            bytes += w.bytes;
            hashes.extend(w.hashes);
        }
        let hierarchies = b
            .components
            .iter()
            .enumerate()
            .filter(|(x, e)| *x == 0 || e.h.cells() > 0)
            .map(|(x, e)| {
                json!({
                    "family": fam.name(), "component": x, "nodes": e.h.len(), "links": e.h.links(), "cells": e.h.cells(),
                    "sweep_layers": e.h.levels(), "tree_levels": e.h.tree_levels(),
                })
            })
            .collect();
        let (links, components, packs) = (b.components.iter().map(|x| x.h.links()).sum(), b.components.len() as u64, l.packs.len() as u64);
        let arc_local = if self.arc_local { l.arc_local } else { Vec::new() };
        Ok(RoadOut { layout_of: b.layout_of, arc_local, hashes, largest, links, components, packs, bytes, hierarchies })
    }
}

pub struct WriteInput<'a> {
    pub args: &'a Args,
    pub inputs: &'a InputsRead,
    pub families: &'a Families<'a>,
    pub catalog: &'a Catalog,
    pub snaps: &'a [ModeSnap],
    pub ids: &'a Ids,
    pub writer: ModelWriter,
}

pub struct WriteOutput {
    pub manifest: Manifest,
    pub served: vocab::Served,
    /// The bytes of the vocabulary's shard files.
    pub vocab_bytes: u64,
    pub shard_bytes: u64,
    pub shards: usize,
}

/// The vocabulary part, last in the model (FORMAT §2.4), then the index and the manifest.
pub struct WriteModel;

impl Stage<WriteInput<'_>> for WriteModel {
    type Output = WriteOutput;
    fn name(&self) -> &'static str {
        "write"
    }
    fn run(&self, w: WriteInput) -> Result<WriteOutput> {
        let out = &w.args.out;
        let o = w.inputs.config.build.build;
        let mut writer = w.writer;
        let f = w.families;
        let fams: [(&dyn Family, &RoadOut); 2] = [(f.car, f.car_road), (f.active, f.active_road)];
        let road_hash = of_hashes(ROAD_TAG, f.car_road.hashes.iter().chain(&f.active_road.hashes).collect::<BTreeMap<_, _>>().into_values());
        writer.set_slot(4)?;
        let (vw, served) = vocab::write(f, w.catalog, w.snaps, w.ids, &road_hash.0, &mut writer)?;
        let mut vh = Hasher::tagged(VOCAB_TAG);
        for (name, h) in &vw.hashes {
            if name != "vocab/road_hash" {
                vh.hash(h);
            }
        }
        vh.hash(&road_hash);
        let vocab_hash = vh.finish();
        let written = writer.finish()?;
        let mut shard_bytes = 0;
        for s in &written.shard_files {
            shard_bytes += std::fs::metadata(out.join(s))?.len();
        }
        let i = &w.inputs;
        let spec = spec_version();
        let hash_of = |s: &str| Hash::from_hex(s).expect("sha256 hex");
        let ih = inputs_hash(&blobmap_format::hash::Inputs {
            pbf_sha256: hash_of(&i.manifest.pbf.sha256),
            overture_release: &i.manifest.overture.release,
            car_profile: hash_of(&i.config_files.profiles.car.sha256),
            active_profile: hash_of(&i.config_files.profiles.active.sha256),
            place_rules: hash_of(&i.config_files.place_rules.sha256),
            extent: i.manifest.extent.as_ref().map(|e| hash_of(&e.sha256)),
            spec_version: spec,
        });
        let (licenses, attribution) = licenses(i, w.catalog, w.ids);
        std::fs::write(out.join(blobmap_format::LICENSES_FILE), &licenses)?;
        let c = &i.config;
        let manifest = Manifest {
            format: blobmap_format::FORMAT.into(),
            format_version: blobmap_format::FORMAT_VERSION.into(),
            model_id: model_id(&road_hash, &vocab_hash).hex(),
            inputs_hash: ih.hex(),
            spec_version: spec.into(),
            compiler: compiler(),
            inputs: i.manifest.clone(),
            config: i.config_files.clone(),
            build: mf::Build {
                shard_target_bytes: o.shard_target_bytes,
                shard_max_bytes: o.shard_max_bytes,
                memory_budget_bytes: o.memory_budget_bytes,
                order_memory_bytes: o.order_memory_bytes,
                pack_max_nodes: o.pack_max_nodes,
            },
            families: fams
                .iter()
                .map(|&(fam, r)| {
                    let family = mf::Family {
                        graph: fam.graph_kind().into(),
                        metrics: fam.metrics(),
                        nodes: fam.nodes() as u64,
                        links: r.links,
                        components: r.components,
                        packs: r.packs,
                    };
                    (fam.name().to_string(), family)
                })
                .collect(),
            modes: vocab::MODES
                .iter()
                .map(|&(m, fam)| {
                    let r = if fam == "car" { c.car.config.snap.radius_m } else { c.active.config.snap.radius_m };
                    (m.to_string(), mf::Mode { family: fam.into(), metric: m.into(), snap_radius_m: r })
                })
                .collect(),
            parts: mf::Parts {
                road: mf::RoadPart { hash: road_hash.hex() },
                vocabulary: mf::VocabPart { hash: vocab_hash.hex(), road_hash: road_hash.hex() },
            },
            attribution,
            licenses_sha256: hex::encode(sha2::Sha256::digest(licenses.as_bytes())),
        };
        manifest.write(&out.join(blobmap_format::MANIFEST_FILE))?;
        Ok(WriteOutput { manifest, served, vocab_bytes: vw.bytes, shard_bytes, shards: written.shard_files.len() })
    }
}

/// An existing model directory is replaced; any other non-empty directory is refused.
fn prepare_out(out: &Path) -> Result<()> {
    if out.exists() {
        let empty = std::fs::read_dir(out)?.next().is_none();
        if !empty {
            ensure!(out.join(blobmap_format::MANIFEST_FILE).exists(), "{} exists and is not a model directory", out.display());
            std::fs::remove_dir_all(out)?;
        }
    }
    Ok(())
}

/// The `LICENSES` file and the manifest's attribution strings (SPEC §12).
fn licenses(i: &InputsRead, c: &Catalog, ids: &vocab::Ids) -> (String, Vec<String>) {
    let counts = c.license_counts(&ids.addresses, &ids.places, &ids.alias_rows);
    let mut t = String::new();
    t.push_str("BlobMap model: licences of the inputs (SPEC §12)\n\n");
    t.push_str("Roads\n");
    t.push_str(&format!("  Source: OpenStreetMap, {} (sha256 {})\n", i.manifest.pbf.name, i.manifest.pbf.sha256));
    t.push_str("  Licence: Open Database License (ODbL) 1.0, https://opendatacommons.org/licenses/odbl/1-0/\n");
    t.push_str("  Attribution: © OpenStreetMap contributors\n");
    t.push_str("  This model is a derived database of OpenStreetMap under the ODbL. Distributing its files\n");
    t.push_str("  means offering them under the ODbL with attribution; answers computed from it are\n");
    t.push_str("  produced works and need the attribution.\n\n");
    t.push_str("Addresses and places\n");
    t.push_str(&format!("  Source: Overture Maps Foundation, release {}\n", c.release));
    t.push_str("  Licences of the source records in this model, with their counts:\n");
    for (l, n) in &counts {
        t.push_str(&format!("    {l}: {n}\n"));
    }
    let mut a = vec!["© OpenStreetMap contributors (ODbL 1.0)".to_string(), format!("Overture Maps Foundation, release {}", c.release)];
    a.extend(counts.keys().map(|l| format!("Overture source data under {l}")));
    (t, a)
}

/// The fixed ACTIVE order of `build.toml`, as ranks by node.
fn fixed_ranks(p: &Pieces, osm: &[i64]) -> Result<Option<FixedRanks>> {
    if osm.is_empty() {
        return Ok(None);
    }
    let mut rank = vec![NONE; p.node_osm.len()];
    for (r, id) in osm.iter().enumerate() {
        match p.osm_node(*id) {
            Some(v) => rank[v as usize] = r as u32,
            None => bail!("build.toml [order]: OSM node {id} is not a routing node"),
        }
    }
    Ok(Some(rank))
}

/// What a build returns besides the files it writes.
pub struct BuildResult {
    pub manifest: Manifest,
}

pub fn build(a: &Args) -> Result<BuildResult> {
    let t0 = Instant::now();
    let mut run = Runner::default();
    let inputs = run.run(&ReadInputs, a)?;
    let c = &inputs.config;
    let osm = run.run(&ReadNodes, (a.pbf.as_path(), a.work_dir().as_path(), c))?;
    prepare_out(&a.out)?;
    let o = c.build.build;
    let mut writer = ModelWriter::create(&a.out, ShardLimits { target: o.shard_target_bytes, max: o.shard_max_bytes })?;
    // Each family is written as soon as its hierarchy is built, then keeps only what the
    // vocabulary and the reports read (SPEC §10, memory). Slots: FORMAT §2.4's unit order.
    let (cp, mut cg) = run.run(&CarNetwork, (&osm, c))?;
    let car = Car::new(&cp, &cg);
    let co = run.run(&Orders("car-order"), (&car as &dyn Family, c, None))?;
    let (cb, cl) = run.run(&Hierarchies("car-hierarchy"), (&car as &dyn Family, c, co))?;
    let car_road = run.run(&WriteRoad { name: "car-write", slots: (0, 2), arc_local: false }, (&car as &dyn Family, cb, cl, &mut writer))?;
    let ce = car.into_components();
    cg.shed();
    let car = Car::with(&cp, &cg, ce);
    let (ap, mut ag) = run.run(&ActiveNetwork, (&osm, c))?;
    let act = Active::new(&ap, &ag);
    let fixed = fixed_ranks(&ap, &c.build.order.active)?;
    let ao = run.run(&Orders("active-order"), (&act as &dyn Family, c, fixed.as_ref()))?;
    let (ab, al) = run.run(&Hierarchies("active-hierarchy"), (&act as &dyn Family, c, ao))?;
    let active_road = run.run(&WriteRoad { name: "active-write", slots: (1, 3), arc_local: true }, (&act as &dyn Family, ab, al, &mut writer))?;
    drop(fixed);
    let parts = act.into_parts();
    ag.shed();
    let act = Active::with(&ap, &ag, parts);
    let catalog = run.run(&ReadCatalog, (&inputs.release, c, inputs.polygon.as_ref()))?;
    let families = Families {
        car: &car,
        car_road: &car_road,
        active: &act,
        active_road: &active_road,
        car_radius_m: c.car.config.snap.radius_m as f64,
        active_radius_m: c.active.config.snap.radius_m as f64,
        car_small_max: c.car.config.snap.small_component_max_nodes,
        active_small_max: c.active.config.snap.small_component_max_nodes,
    };
    let (snaps, ids) = run.run(&Snap, (&families, &catalog, inputs.polygon.is_some(), osm.dir().join("snap").as_path()))?;
    let wo = run.run(&WriteModel, WriteInput { args: a, inputs: &inputs, families: &families, catalog: &catalog, snaps: &snaps, ids: &ids, writer })?;
    let islands = run.run(&ReadIslands, &osm)?;
    let data = report::data(a, &families, &catalog, &snaps, &ids, &wo.served, &islands)?;
    blobmap_format::json::write_file(&a.report_path("data"), &data)?;
    let hierarchies: Vec<&serde_json::Value> = car_road.hierarchies.iter().chain(&active_road.hierarchies).collect();
    // What serving costs (SPEC §13, milestone 5): the road part per link and per family, a
    // column's distance state (u16 per node of the family's largest component), and the
    // vocabulary per ID.
    let (ids_addresses, ids) = (ids.addresses.len() as u64, (ids.addresses.len() + ids.places.len()) as u64);
    let family = |r: &RoadOut, f: &dyn Family| {
        let largest = f.components().nodes.first().copied().unwrap_or(0) as u64;
        json!({
            "bytes": r.bytes, "links": r.links, "bytes_per_link": r.bytes as f64 / r.links.max(1) as f64,
            "nodes": f.nodes(), "largest_component_nodes": largest, "column_state_bytes": 2 * largest,
        })
    };
    let road_bytes = car_road.bytes + active_road.bytes;
    let sizes = json!({
        "car": family(&car_road, &car), "active": family(&active_road, &act),
        "road_bytes": road_bytes, "bytes_per_link": road_bytes as f64 / (car_road.links + active_road.links).max(1) as f64,
        "vocab_bytes": wo.vocab_bytes, "ids": ids, "addresses": ids_addresses, "bytes_per_id": wo.vocab_bytes as f64 / ids.max(1) as f64,
        "column_result_bytes": 2 * ids,
    });
    let largest: serde_json::Map<String, serde_json::Value> = [("car", &car_road, "turns"), ("active", &active_road, "roads")]
        .iter()
        .map(|&(f, r, arcs)| {
            let p = &r.largest;
            (f.to_string(), json!({"pack": p.pack, "rows": p.rows, "links": p.links, arcs: p.arcs}))
        })
        .collect();
    let build = json!({
        "model_id": wo.manifest.model_id,
        "stages": run.stages.iter().map(|s| json!({"name": s.name, "wall_s": s.wall_s, "peak_rss_bytes": s.peak_rss_bytes})).collect::<Vec<_>>(),
        "wall_s": t0.elapsed().as_secs_f64(),
        "peak_rss_bytes": run.stages.iter().filter_map(|s| s.peak_rss_bytes).max(),
        "largest_pack": largest,
        "hierarchies": hierarchies,
        "sizes": sizes,
        "shard_bytes": wo.shard_bytes,
        "shards": wo.shards,
        // Layout ids per family against their U32 bound (FORMAT §4).
        "layout_ids": wo.manifest.families.iter().map(|(f, x)| (f.clone(), json!({"nodes": x.nodes, "bound": NONE as u64}))).collect::<serde_json::Map<_, _>>(),
    });
    blobmap_format::json::write_file(&a.report_path("build"), &build)?;
    // The pieces' files are unlinked under their maps, which end with this function.
    let _ = std::fs::remove_dir_all(a.work_dir());
    Ok(BuildResult { manifest: wo.manifest })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iso_dates() {
        assert_eq!(iso8601(0), "1970-01-01T00:00:00Z");
        assert_eq!(iso8601(1_794_000_000), "2026-11-06T21:20:00Z");
    }

    #[test]
    fn spec_version_line() {
        assert_eq!(spec_version(), "2.0.0");
    }
}
