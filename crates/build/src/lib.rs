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
use blobmap_format::hash::{ENTRY_TAG, Hasher, ROAD_TAG, VOCAB_TAG, inputs_hash, model_id, of_hashes};
use blobmap_format::manifest::{self as mf, Manifest};
use blobmap_format::{Hash, ModelWriter, NONE, ShardLimits};
use blobmap_graph::{ActiveGraph, CarGraph};
use blobmap_osm::{Network, Pieces};
use blobmap_overture::{Catalog, Polygon, Release};
use blobmap_profile::active::{ActiveMode, ActiveProfile, Mode};
use blobmap_profile::config::Config;
use blobmap_profile::profile::{CarProfile, WayAttrs};
use serde_json::json;
use sha2::Digest;

use crate::family::{Active, Car, Family};
use crate::hier::{Built, FixedRanks};
use crate::layout::Layout;
use crate::vocab::{Families, Ids, ModeSnap};

/// The SPEC version this compiler builds to (FORMAT §3 `spec_version`).
pub fn spec_version() -> &'static str {
    const SPEC: &str = include_str!("../../../SPEC.md");
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
        self.stages.push(StageReport { name: s.name(), wall_s: t.elapsed().as_secs_f64(), peak_rss_bytes: mem::peak_rss() });
        Ok(out)
    }
}

fn sha256_file(path: &Path) -> Result<(String, u64)> {
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
        let timestamp = blobmap_osm::header_timestamp(&a.pbf)?.map(iso8601);
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

pub struct CarNetwork;

impl Stage<(&Path, &Config)> for CarNetwork {
    type Output = (Pieces, CarGraph);
    fn name(&self) -> &'static str {
        "car-network"
    }
    fn run(&self, (pbf, c): (&Path, &Config)) -> Result<(Pieces, CarGraph)> {
        let net = Network::read(pbf, &CarProfile::new(&c.car.config), &c.car.config.snap)?;
        let p = net.pieces(&[|a: &WayAttrs| *a]);
        let (r, _) = blobmap_osm::restrictions::read(pbf)?;
        let g = CarGraph::build(&p, &c.car.config.turns, &r);
        Ok((p, g))
    }
}

pub struct ActiveNetwork;

impl Stage<(&Path, &Config)> for ActiveNetwork {
    type Output = (Pieces, ActiveGraph);
    fn name(&self) -> &'static str {
        "active-network"
    }
    fn run(&self, (pbf, c): (&Path, &Config)) -> Result<(Pieces, ActiveGraph)> {
        let prof = ActiveProfile::new(&c.active.config);
        let net = Network::read(pbf, &prof, &c.active.config.snap)?;
        let modes: Vec<ActiveMode> = [Mode::Bike, Mode::Walk].map(|mode| ActiveMode { mode, cfg: prof.cfg }).to_vec();
        let p = net.pieces(&modes);
        let g = ActiveGraph::build(&p);
        Ok((p, g))
    }
}

/// Every expert's hierarchy, then the entries and packs.
pub struct Hierarchies(pub &'static str);

impl Stage<(&dyn Family, &Config, Option<&FixedRanks>)> for Hierarchies {
    type Output = (Built, Layout);
    fn name(&self) -> &'static str {
        self.0
    }
    fn run(&self, (f, c, fixed): (&dyn Family, &Config, Option<&FixedRanks>)) -> Result<(Built, Layout)> {
        let o = &c.build.build;
        let max_cell = (o.memory_budget_bytes / hier::BYTES_PER_NODE).max(1) as usize;
        let b = hier::build(f, max_cell, fixed)?;
        let l = layout::plan(f, &b, o.group_max_nodes)?;
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

impl Stage<(&Families<'_>, &Catalog, bool)> for Snap {
    type Output = (Vec<ModeSnap>, Ids);
    fn name(&self) -> &'static str {
        "snap"
    }
    fn run(&self, (f, c, polygon): (&Families, &Catalog, bool)) -> Result<(Vec<ModeSnap>, Ids)> {
        let points: Vec<(f64, f64)> = c.addresses.iter().map(|a| (a.lat, a.lon)).chain(c.places.iter().map(|p| (p.lat, p.lon))).collect();
        let snaps = vocab::snap(f, &points);
        let ids = vocab::select(c, &snaps, polygon);
        Ok((snaps, ids))
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

pub struct WriteInput<'a> {
    pub args: &'a Args,
    pub inputs: &'a InputsRead,
    pub families: &'a Families<'a>,
    pub catalog: &'a Catalog,
    pub snaps: &'a [ModeSnap],
    pub ids: &'a Ids,
}

pub struct WriteOutput {
    pub manifest: Manifest,
    pub served: vocab::Served,
    pub largest: BTreeMap<String, PackSize>,
    pub shard_bytes: u64,
}

pub struct WriteModel;

impl Stage<WriteInput<'_>> for WriteModel {
    type Output = WriteOutput;
    fn name(&self) -> &'static str {
        "write"
    }
    fn run(&self, w: WriteInput) -> Result<WriteOutput> {
        let out = &w.args.out;
        prepare_out(out)?;
        let o = w.inputs.config.build.build;
        let mut writer = ModelWriter::create(out, ShardLimits { target: o.shard_target_bytes, max: o.shard_max_bytes })?;
        let f = w.families;
        let fams: [(&dyn Family, &Built, &Layout); 2] =
            [(f.car, f.car_built, f.car_layout), (f.active, f.active_built, f.active_layout)];
        let mut tables: BTreeMap<String, Hash> = BTreeMap::new();
        let mut entry_base = 0;
        for (fam, b, l) in fams {
            tables.extend(writer.write_unit(&road::tables(fam, b, l, entry_base))?.hashes);
            entry_base += l.entries.len() as u32;
        }
        // Packs, then entries in manifest order.
        let mut entries: Vec<(mf::Entry, Vec<usize>)> = Vec::new();
        let mut largest = BTreeMap::new();
        for (fam, b, l) in fams {
            let mut pack_out = Vec::new();
            let mut big = PackSize::default();
            for k in 0..l.packs.len() {
                let pk = &l.packs[k];
                let size = PackSize { pack: k as u32, rows: pk.rows.len() as u64, links: pk.links, arcs: pk.arcs.len() as u64 };
                if (size.rows, size.links) > (big.rows, big.links) {
                    big = size;
                }
                pack_out.push(writer.write_unit(&road::pack(fam, b, l, k))?);
            }
            largest.insert(fam.name().to_string(), big);
            for en in &l.entries {
                let xs = &b.experts[en.first as usize..(en.first + en.count) as usize];
                let mut hashes: BTreeMap<&String, &Hash> = BTreeMap::new();
                let mut shards = Vec::new();
                for p in &pack_out[en.pack_first as usize..(en.pack_first + en.packs) as usize] {
                    hashes.extend(p.hashes.iter());
                    shards.extend(p.shards.iter().copied());
                }
                let cells = if en.group { 0 } else { xs[0].h.cells() as u64 };
                let e = mf::Entry {
                    family: fam.name().into(),
                    kind: if en.group { "group" } else { "expert" }.into(),
                    experts: [en.first as u64, en.count as u64],
                    components: en.count as u64,
                    nodes: xs.iter().map(|x| x.nodes.len() as u64).sum(),
                    links: xs.iter().map(|x| x.h.links()).sum(),
                    levels: xs.iter().map(|x| x.h.levels() as u64).max().unwrap_or(0),
                    cells,
                    packs: [en.pack_first as u64, en.packs as u64],
                    shards: Vec::new(),
                    hash: of_hashes(ENTRY_TAG, hashes.into_values()).hex(),
                };
                entries.push((e, shards));
            }
        }
        let mut rh = Hasher::tagged(ROAD_TAG);
        for h in tables.values() {
            rh.hash(h);
        }
        for (e, _) in &entries {
            rh.hash(&Hash::from_hex(&e.hash).expect("hex"));
        }
        let road_hash = rh.finish();
        let (unit, served) = vocab::unit(f, w.catalog, w.snaps, w.ids, &road_hash.0);
        let vw = writer.write_unit(&unit)?;
        drop(unit);
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
        let experts: Vec<mf::Entry> = entries
            .into_iter()
            .map(|(mut e, s)| {
                e.shards = s.into_iter().map(|k| written.shard_files[k].clone()).collect();
                e
            })
            .collect();
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
        let (licenses, attribution) = licenses(i, w.catalog);
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
                group_max_nodes: o.group_max_nodes,
            },
            families: [(f.car as &dyn Family), f.active]
                .iter()
                .map(|fam| (fam.name().to_string(), mf::Family { graph: fam.graph_kind().into(), metrics: fam.metrics() }))
                .collect(),
            modes: vocab::MODES
                .iter()
                .map(|&(m, fam)| {
                    let r = if fam == "car" { c.car.config.snap.radius_m } else { c.active.config.snap.radius_m };
                    (m.to_string(), mf::Mode { family: fam.into(), metric: m.into(), snap_radius_m: r })
                })
                .collect(),
            experts,
            parts: mf::Parts {
                road: mf::RoadPart { hash: road_hash.hex() },
                vocabulary: mf::VocabPart { hash: vocab_hash.hex(), road_hash: road_hash.hex() },
            },
            attribution,
            licenses_sha256: hex::encode(sha2::Sha256::digest(licenses.as_bytes())),
        };
        manifest.write(&out.join(blobmap_format::MANIFEST_FILE))?;
        Ok(WriteOutput { manifest, served, largest, shard_bytes })
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
fn licenses(i: &InputsRead, c: &Catalog) -> (String, Vec<String>) {
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
    t.push_str("  Licences of the source records read, with their counts:\n");
    for (l, n) in &c.licenses {
        t.push_str(&format!("    {l}: {n}\n"));
    }
    let mut a = vec!["© OpenStreetMap contributors (ODbL 1.0)".to_string(), format!("Overture Maps Foundation, release {}", c.release)];
    a.extend(c.licenses.keys().map(|l| format!("Overture source data under {l}")));
    (t, a)
}

/// The fixed ACTIVE order of `build.toml`, as ranks by node.
fn fixed_ranks(p: &Pieces, osm: &[i64]) -> Result<Option<FixedRanks>> {
    if osm.is_empty() {
        return Ok(None);
    }
    let mut rank = vec![NONE; p.node_osm.len()];
    for (r, id) in osm.iter().enumerate() {
        match p.node_osm.binary_search(id) {
            Ok(v) => rank[v] = r as u32,
            Err(_) => bail!("build.toml [order]: OSM node {id} is not a routing node"),
        }
    }
    Ok(Some(rank))
}

/// What a build returns besides the files it writes.
pub struct BuildResult {
    pub manifest: Manifest,
    pub stages: Vec<StageReport>,
}

pub fn build(a: &Args) -> Result<BuildResult> {
    let t0 = Instant::now();
    let mut run = Runner::default();
    let inputs = run.run(&ReadInputs, a)?;
    let c = &inputs.config;
    let (cp, cg) = run.run(&CarNetwork, (a.pbf.as_path(), c))?;
    let car = Car::new(&cp, &cg);
    let (cb, cl) = run.run(&Hierarchies("car-hierarchy"), (&car as &dyn Family, c, None))?;
    let (ap, ag) = run.run(&ActiveNetwork, (a.pbf.as_path(), c))?;
    let act = Active::new(&ap, &ag);
    let fixed = fixed_ranks(&ap, &c.build.order.active)?;
    let (ab, al) = run.run(&Hierarchies("active-hierarchy"), (&act as &dyn Family, c, fixed.as_ref()))?;
    let catalog = run.run(&ReadCatalog, (&inputs.release, c, inputs.polygon.as_ref()))?;
    let families = Families {
        car: &car,
        car_built: &cb,
        car_layout: &cl,
        active: &act,
        active_built: &ab,
        active_layout: &al,
        car_radius_m: c.car.config.snap.radius_m as f64,
        active_radius_m: c.active.config.snap.radius_m as f64,
    };
    let (snaps, ids) = run.run(&Snap, (&families, &catalog, inputs.polygon.is_some()))?;
    let wo = run.run(&WriteModel, WriteInput { args: a, inputs: &inputs, families: &families, catalog: &catalog, snaps: &snaps, ids: &ids })?;
    let data = report::data(a, &families, &catalog, &snaps, &ids, &wo.served)?;
    blobmap_format::json::write_file(&a.report_path("data"), &data)?;
    let build = json!({
        "model_id": wo.manifest.model_id,
        "stages": run.stages.iter().map(|s| json!({"name": s.name, "wall_s": s.wall_s, "peak_rss_bytes": s.peak_rss_bytes})).collect::<Vec<_>>(),
        "wall_s": t0.elapsed().as_secs_f64(),
        "peak_rss_bytes": run.stages.iter().filter_map(|s| s.peak_rss_bytes).max(),
        "largest_pack": wo.largest.iter().map(|(f, p)| {
            let arcs = if f == "car" { "turns" } else { "roads" };
            (f.clone(), json!({"pack": p.pack, "rows": p.rows, "links": p.links, arcs: p.arcs}))
        }).collect::<serde_json::Map<_, _>>(),
        "shard_bytes": wo.shard_bytes,
        "shards": wo.manifest.experts.iter().flat_map(|e| e.shards.iter()).collect::<std::collections::BTreeSet<_>>().len(),
    });
    blobmap_format::json::write_file(&a.report_path("build"), &build)?;
    Ok(BuildResult { manifest: wo.manifest, stages: run.stages })
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
