//! Opening a model: the manifest, the family tables, the vocabulary, and components loaded on
//! demand from their packs (FORMAT §5, §7). A component is a range of the family's layout
//! ids; loaded, its nodes are numbered from 0, and so are the vocabulary's nodes.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};

use anyhow::{Context, Result, bail, ensure};

use crate::manifest::Manifest;
use crate::runtime::arr::{Arr, ArrWriter};
use crate::{Model, NONE, NONE64};

pub type Id = [u8; 16];

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Fam {
    Car,
    Active,
}

impl Fam {
    pub fn name(self) -> &'static str {
        match self {
            Fam::Car => "car",
            Fam::Active => "active",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dir {
    /// From the sources to each ID.
    From,
    /// From each ID to the sources (the reversed graph).
    To,
}

impl Dir {
    pub fn parse(s: &str) -> Result<Dir> {
        match s {
            "from" => Ok(Dir::From),
            "to" => Ok(Dir::To),
            _ => bail!("dir must be \"from\" or \"to\", not {s:?}"),
        }
    }

    pub fn flip(self) -> Dir {
        match self {
            Dir::From => Dir::To,
            Dir::To => Dir::From,
        }
    }
}

/// Answers that are not a time (SPEC §6, §9; FORMAT §11).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reason {
    UnknownId,
    DifferentComponent,
    NoEligibleRoad,
    NoRoad,
    TrappedComponent,
    FilteredPlace,
}

impl Reason {
    pub fn name(self) -> &'static str {
        match self {
            Reason::UnknownId => "UnknownId",
            Reason::DifferentComponent => "DifferentComponent",
            Reason::NoEligibleRoad => "NoEligibleRoad",
            Reason::NoRoad => "NoRoad",
            Reason::TrappedComponent => "TrappedComponent",
            Reason::FilteredPlace => "FilteredPlace",
        }
    }

    pub fn from_code(c: u8) -> Result<Reason> {
        Ok(match c {
            1 => Reason::NoEligibleRoad,
            2 => Reason::NoRoad,
            3 => Reason::TrappedComponent,
            4 => Reason::FilteredPlace,
            _ => bail!("reason code {c} is not used in 2.0"),
        })
    }
}

/// Deliberate defects, for the conformance suite's broken runtimes (SPEC §11).
#[derive(Clone, Copy, Debug, Default)]
pub struct Faults {
    /// Arrivals ignore via-way copies (FORMAT §8).
    pub no_copies: bool,
    /// No direct term on a shared piece (SPEC §9).
    pub no_direct: bool,
    /// Additions wrap at 2^16 instead of saturating (SPEC §7).
    pub wrapping: bool,
    /// Ties go to the largest ID (SPEC §7).
    pub largest_id: bool,
    /// Start times are ignored.
    pub no_start: bool,
}

/// One mode: its family and its metric's index in the family.
#[derive(Clone, Debug)]
pub struct ModeInfo {
    pub name: String,
    pub fam: Fam,
    pub metric: usize,
}

pub struct Tables {
    pub metrics: Vec<String>,
    /// The component of each layout id (FORMAT §5.1).
    pub component: Vec<u32>,
    /// Derived: per component, its first layout id, and the family's node count at the end.
    pub first: Vec<u32>,
    /// Derived: per component, its first block, and the block count at the end.
    pub block_first: Vec<u32>,
    pub b_kind: Vec<u8>,
    pub b_id_first: Vec<u32>,
    pub b_nodes: Vec<u32>,
    pub b_pack: Vec<u32>,
    pub b_row_first: Vec<u32>,
    pub b_levels: Vec<u32>,
    pub b_level_base: Vec<u32>,
}

impl Tables {
    pub fn components(&self) -> usize {
        self.first.len() - 1
    }

    /// The node count of component x.
    pub fn nodes(&self, x: u32) -> u32 {
        self.first[x as usize + 1] - self.first[x as usize]
    }

    /// The longest top-down chain of levels in component x: its whole block's levels, or the
    /// top block's plus the deepest cell's (FORMAT §5.2).
    pub fn levels(&self, x: u32) -> u32 {
        let bs = self.block_first[x as usize] as usize..self.block_first[x as usize + 1] as usize;
        let of = |cell: bool| bs.clone().filter(move |&b| (self.b_kind[b] == 2) == cell).map(|b| self.b_levels[b]);
        of(false).sum::<u32>() + of(true).max().unwrap_or(0)
    }

    /// Each component's layout ids and blocks, from `component` and the block table: the
    /// components must be consecutive ranges in index order (FORMAT §4).
    fn derive(&mut self, f: &str) -> Result<()> {
        let c = &self.component;
        ensure!(c.first().is_none_or(|&x| x == 0) && c.windows(2).all(|w| w[1] == w[0] || w[1] == w[0] + 1), "road/{f}/nodes/component: components are not consecutive ranges from 0");
        let n = c.last().map_or(0, |&x| x as usize + 1);
        let mut first = vec![0u32; n + 1];
        for &x in c {
            first[x as usize + 1] += 1;
        }
        for i in 0..n {
            first[i + 1] += first[i];
        }
        ensure!(self.b_id_first.windows(2).all(|w| w[0] < w[1]), "road/{f}/blocks: id_first is not ascending");
        self.block_first = first[..n].iter().map(|&v| self.b_id_first.partition_point(|&b| b < v) as u32).collect();
        self.block_first.push(self.b_id_first.len() as u32);
        self.first = first;
        Ok(())
    }
}

/// One mode's entries (FORMAT §7.2) in a common shape. CAR: `node` = `seg`, `road` = NONE,
/// `off` = [before₀, after₀, before₁, after₁]. ACTIVE: `node` = [U, V], `off` = [U → ID,
/// ID → U, V → ID, ID → V]. `component` is derived from the nodes, `NONE` when unserved,
/// and `node` is numbered within that component.
pub struct ModeVocab {
    pub info: ModeInfo,
    pub component: Arr<u32>,
    pub node: Arr<[u32; 2]>,
    pub road: Arr<u32>,
    pub off: Arr<[u16; 4]>,
    /// (internal index, code, metres), sorted by index.
    pub reasons: Vec<(u32, u8, f32)>,
}

pub struct Vocab {
    pub address: Arr<Id>,
    pub place: Arr<Id>,
    pub alias: Vec<Id>,
    pub alias_to: Vec<u32>,
    pub road_hash: Vec<u8>,
    pub modes: Vec<ModeVocab>,
    pub status: Vec<u8>,
    pub primary: Vec<u16>,
    pub basic: Vec<u16>,
    pub alt_first: Vec<u32>,
    pub alt: Vec<u16>,
    pub tax_name: Vec<String>,
    pub tax_parent: Vec<u16>,
    pub tax_basic: Vec<u16>,
    pub tax_is_basic: Vec<u8>,
    pub redirect: Vec<String>,
    pub redirect_to: Vec<u16>,
}

impl Vocab {
    pub fn n(&self) -> usize {
        self.address.len() + self.place.len()
    }

    pub fn id(&self, i: u32) -> Id {
        let i = i as usize;
        if i < self.address.len() { self.address[i] } else { self.place[i - self.address.len()] }
    }

    /// FORMAT §7.1 lookup: addresses, places, then aliases.
    pub fn lookup(&self, id: &Id) -> Option<u32> {
        if let Ok(i) = self.address.binary_search(id) {
            return Some(i as u32);
        }
        if let Ok(i) = self.place.binary_search(id) {
            return Some((self.address.len() + i) as u32);
        }
        self.alias.binary_search(id).ok().map(|k| self.alias_to[k])
    }

    pub fn is_place(&self, i: u32) -> bool {
        i as usize >= self.address.len()
    }

    /// Status of internal index i: 0, or 4 for a filtered place.
    pub fn status_of(&self, i: u32) -> u8 {
        if self.is_place(i) { self.status[i as usize - self.address.len()] } else { 0 }
    }
}

/// A block of a component and where it is stored.
#[derive(Clone, Copy, Debug)]
pub struct BlockAt {
    pub kind: u8,
    pub id_first: u32,
    /// Pack-local link index of the block's first row's first link.
    pub pack_link_first: u64,
}

/// One component, with its rows concatenated in layout-id order. Link indices here are
/// component-local: positions in `up_head`.
pub struct Component {
    pub fam: Fam,
    pub n: usize,
    pub blocks: Vec<BlockAt>,
    /// Per block, its level bounds as component-local ids (FORMAT §5.2): level l is
    /// `levels[b][l] .. levels[b][l + 1]`, level 0 the top.
    pub levels: Vec<Vec<u32>>,
    pub up_first: Arr<u64>,
    pub up_head: Arr<u32>,
    /// Per metric.
    pub up: Vec<Arr<u16>>,
    pub down: Vec<Arr<u16>>,
    /// Turns (CAR) or roads (ACTIVE), CSR by tail.
    pub arc_first: Arr<u64>,
    pub arc_head: Arr<u32>,
    /// Component-local link of each arc, `NONE` when tail = head.
    pub arc_link: Arr<u64>,
    /// The pack-local index of each arc (ACTIVE `road`).
    pub arc_local: Arr<u32>,
    pub turn_cost: Arr<u16>,
    /// The roads' tensors: per node for CAR, per arc for ACTIVE.
    pub way: Arr<i64>,
    pub way_dir: Arr<u8>,
    pub way_seq: Arr<u32>,
    pub kind: Arr<u8>,
    pub time_s: Vec<Arr<f32>>,
    pub w: Vec<Arr<u16>>,
    pub copy_of: Arr<u32>,
    pub copy_first: Arr<u32>,
    pub copies: Arr<u32>,
}

impl Component {
    pub fn row(&self, v: u32) -> std::ops::Range<usize> {
        self.up_first[v as usize] as usize..self.up_first[v as usize + 1] as usize
    }

    pub fn arcs(&self, v: u32) -> std::ops::Range<usize> {
        self.arc_first[v as usize] as usize..self.arc_first[v as usize + 1] as usize
    }

    /// The component-local link between a and b.
    pub fn link(&self, a: u32, b: u32) -> Option<usize> {
        let (lo, hi) = (a.max(b), a.min(b));
        let r = self.row(lo);
        self.up_head[r.clone()].binary_search(&hi).ok().map(|i| r.start + i)
    }

    pub fn block_of(&self, v: u32) -> usize {
        self.blocks.partition_point(|b| b.id_first <= v) - 1
    }

    /// A CAR segment and its copies in this component (FORMAT §8 `arr`).
    pub fn arrivals(&self, s: u32, faults: &Faults) -> impl Iterator<Item = u32> + '_ {
        let copies: &[u32] = if self.fam == Fam::Car && !faults.no_copies {
            &self.copies[self.copy_first[s as usize] as usize..self.copy_first[s as usize + 1] as usize]
        } else {
            &[]
        };
        std::iter::once(s).chain(copies.iter().copied())
    }
}

pub struct Runtime {
    pub model: Model,
    pub manifest: Manifest,
    pub tables: BTreeMap<Fam, Tables>,
    pub vocab: Vocab,
    pub faults: Faults,
    /// The components loaded, kept until `release`.
    cache: Mutex<BTreeMap<(Fam, u32), Arc<Component>>>,
    /// The components being loaded, so that concurrent queries load each once while
    /// different components load at the same time; `loaded` wakes the queries waiting on one.
    loading: Mutex<BTreeSet<(Fam, u32)>>,
    loaded: Condvar,
    /// Where the vocabulary and the components go to mapped files (`Arr`), if anywhere.
    scratch: Option<PathBuf>,
    /// Components load only what queries read (`set_queries_only`).
    queries_only: std::sync::atomic::AtomicBool,
}

fn strings(m: &Model, name: &str) -> Result<Vec<String>> {
    let bytes = m.get::<u8>(&format!("{name}.bytes"))?;
    let first = m.get::<u32>(&format!("{name}.first"))?;
    ensure!(!first.is_empty() && *first.last().unwrap() as usize == bytes.len(), "{name}: bad offsets");
    first.windows(2).map(|w| Ok(String::from_utf8(bytes[w[0] as usize..w[1] as usize].to_vec())?)).collect()
}

fn ids(m: &Model, name: &str) -> Result<Vec<Id>> {
    let (v, n) = m.get_rows::<u8>(name, &[16])?;
    Ok((0..n).map(|i| v[16 * i..16 * i + 16].try_into().unwrap()).collect())
}

fn ids_in(m: &Model, name: &str, scratch: Option<&Path>) -> Result<Arr<Id>> {
    let (v, n) = m.get_rows::<u8>(name, &[16])?;
    let mut w = ArrWriter::new(scratch)?;
    w.extend_from_slice(bytemuck::cast_slice(&v[..16 * n]))?;
    w.finish()
}

impl Runtime {
    pub fn open(dir: &Path) -> Result<Runtime> {
        Runtime::open_in(dir, None)
    }

    /// The runtime of the model in `dir`, its vocabulary and components in mapped files under
    /// `scratch` when given (SPEC §10).
    pub fn open_in(dir: &Path, scratch: Option<&Path>) -> Result<Runtime> {
        let model = Model::open(dir)?;
        let manifest = Manifest::from_value(&model.manifest).context("manifest")?;
        let road_hash = &manifest.parts.road.hash;
        ensure!(&manifest.parts.vocabulary.road_hash == road_hash, "the vocabulary names another road part (FORMAT §9)");
        let mut tables = BTreeMap::new();
        for fam in [Fam::Car, Fam::Active] {
            let f = fam.name();
            let metrics = manifest.families.get(f).with_context(|| format!("no family {f}"))?.metrics.clone();
            let g = |t: &str, n: &str| -> Result<Vec<u32>> { Ok(model.get::<u32>(&format!("road/{f}/{t}/{n}"))?.into_owned()) };
            let mut t = Tables {
                metrics,
                component: g("nodes", "component")?,
                first: Vec::new(),
                block_first: Vec::new(),
                b_kind: model.get::<u8>(&format!("road/{f}/blocks/kind"))?.into_owned(),
                b_id_first: g("blocks", "id_first")?,
                b_nodes: g("blocks", "nodes")?,
                b_pack: g("blocks", "pack")?,
                b_row_first: g("blocks", "row_first")?,
                b_levels: g("blocks", "levels")?,
                b_level_base: g("blocks", "level_base")?,
            };
            t.derive(f)?;
            tables.insert(fam, t);
        }
        let address = ids_in(&model, "vocab/ids/address", scratch)?;
        let place = ids_in(&model, "vocab/ids/place", scratch)?;
        let n = address.len() + place.len();
        let mut modes = Vec::new();
        for (name, mode) in &manifest.modes {
            let fam = if mode.family == "car" { Fam::Car } else { Fam::Active };
            let metric = tables[&fam].metrics.iter().position(|m| m == &mode.metric).context("mode metric")?;
            let p = format!("vocab/{}/{name}.", mode.family);
            let (mut node, road, off): (Vec<[u32; 2]>, Vec<u32>, Vec<[u16; 4]>) = if fam == Fam::Car {
                let (s, k) = model.get_rows::<u32>(&format!("{p}seg"), &[2])?;
                let (o, k2) = model.get_rows::<u16>(&format!("{p}off"), &[2, 2])?;
                ensure!(k == n && k2 == n, "{p}: entries do not cover every ID");
                (
                    (0..n).map(|i| [s[2 * i], s[2 * i + 1]]).collect(),
                    vec![NONE; n],
                    (0..n).map(|i| o[4 * i..4 * i + 4].try_into().unwrap()).collect(),
                )
            } else {
                let (s, k) = model.get_rows::<u32>(&format!("{p}node"), &[2])?;
                let (o, k2) = model.get_rows::<u16>(&format!("{p}off"), &[4])?;
                ensure!(k == n && k2 == n, "{p}: entries do not cover every ID");
                let road = model.get_len::<u32>(&format!("{p}road"), n)?.into_owned();
                (
                    (0..n).map(|i| [s[2 * i], s[2 * i + 1]]).collect(),
                    road,
                    (0..n).map(|i| o[4 * i..4 * i + 4].try_into().unwrap()).collect(),
                )
            };
            // Each served ID's component, from its first node; its nodes made local to it.
            let t = &tables[&fam];
            let mut component = vec![NONE; n];
            for (i, nd) in node.iter_mut().enumerate() {
                let Some(&v) = nd.iter().find(|&&v| v != NONE) else { continue };
                let x = *t.component.get(v as usize).with_context(|| format!("{p}: ID {i} names layout id {v}, out of range"))?;
                component[i] = x;
                for v in nd.iter_mut().filter(|v| **v != NONE) {
                    ensure!(t.component.get(*v as usize) == Some(&x), "{p}: ID {i} has nodes in two components");
                    *v -= t.first[x as usize];
                }
            }
            let rid = model.get::<u32>(&format!("{p}reason_id"))?;
            let rcode = model.get_len::<u8>(&format!("{p}reason_code"), rid.len())?;
            let rm = model.get_len::<f32>(&format!("{p}reason_m"), rid.len())?;
            let reasons = (0..rid.len()).map(|k| (rid[k], rcode[k], rm[k])).collect();
            let (component, node, road, off) = (Arr::new(component, scratch)?, Arr::new(node, scratch)?, Arr::new(road, scratch)?, Arr::new(off, scratch)?);
            modes.push(ModeVocab { info: ModeInfo { name: name.clone(), fam, metric }, component, node, road, off, reasons });
        }
        let vocab = Vocab {
            alias: ids(&model, "vocab/ids/alias")?,
            alias_to: model.get::<u32>("vocab/ids/alias_to")?.into_owned(),
            road_hash: model.get_len::<u8>("vocab/road_hash", 32)?.into_owned(),
            modes,
            status: model.get_len::<u8>("vocab/places/status", place.len())?.into_owned(),
            primary: model.get_len::<u16>("vocab/places/primary", place.len())?.into_owned(),
            basic: model.get_len::<u16>("vocab/places/basic", place.len())?.into_owned(),
            alt_first: model.get_len::<u32>("vocab/places/alt_first", place.len() + 1)?.into_owned(),
            alt: model.get::<u16>("vocab/places/alt")?.into_owned(),
            tax_name: strings(&model, "vocab/taxonomy/name")?,
            tax_parent: model.get::<u16>("vocab/taxonomy/parent")?.into_owned(),
            tax_basic: model.get::<u16>("vocab/taxonomy/basic")?.into_owned(),
            tax_is_basic: model.get::<u8>("vocab/taxonomy/is_basic")?.into_owned(),
            redirect: strings(&model, "vocab/taxonomy/redirect")?,
            redirect_to: model.get::<u16>("vocab/taxonomy/redirect_to")?.into_owned(),
            address,
            place,
        };
        ensure!(hex::encode(&vocab.road_hash) == *road_hash, "vocab/road_hash differs from the manifest (FORMAT §9)");
        Ok(Runtime {
            model,
            manifest,
            tables,
            vocab,
            faults: Faults::default(),
            cache: Mutex::new(BTreeMap::new()),
            loading: Mutex::new(BTreeSet::new()),
            loaded: Condvar::new(),
            scratch: scratch.map(Path::to_path_buf),
            queries_only: std::sync::atomic::AtomicBool::new(false),
        })
    }

    pub fn mode(&self, name: &str) -> Result<usize> {
        self.vocab.modes.iter().position(|m| m.info.name == name).with_context(|| format!("unknown mode {name:?}"))
    }

    pub fn components(&self, fam: Fam) -> usize {
        self.tables[&fam].components()
    }

    /// Component x of a family, loaded from its packs.
    pub fn component(&self, fam: Fam, x: u32) -> Result<Arc<Component>> {
        if let Some(e) = self.cache.lock().unwrap().get(&(fam, x)) {
            return Ok(e.clone());
        }
        let mut busy = self.loading.lock().unwrap();
        loop {
            if let Some(e) = self.cache.lock().unwrap().get(&(fam, x)) {
                return Ok(e.clone());
            }
            if busy.insert((fam, x)) {
                break;
            }
            busy = self.loaded.wait(busy).unwrap();
        }
        drop(busy);
        let e = self.load(fam, x).map(Arc::new);
        if let Ok(e) = &e {
            self.cache.lock().unwrap().insert((fam, x), e.clone());
        }
        self.loading.lock().unwrap().remove(&(fam, x));
        self.loaded.notify_all();
        e
    }

    /// Drops every component loaded so far; a query still holding one keeps it until it ends.
    pub fn release(&self) {
        self.cache.lock().unwrap().clear();
    }

    /// From now on, components load only the arrays that queries read (`answer`, columns,
    /// pairs), not those of the roads, the links of arcs and the copies' originals, which only
    /// the checks read: on ACTIVE over half of a component's scratch. Drops the components
    /// loaded so far, so that none is held twice.
    pub fn set_queries_only(&self, on: bool) {
        self.release();
        self.queries_only.store(on, std::sync::atomic::Ordering::Relaxed);
    }

    fn load(&self, fam: Fam, x: u32) -> Result<Component> {
        let t = &self.tables[&fam];
        let f = fam.name();
        let m = &self.model;
        ensure!((x as usize) < t.components(), "{f} component {x} does not exist");
        let full = !self.queries_only.load(std::sync::atomic::Ordering::Relaxed);
        let metrics = &t.metrics;
        let word = if fam == Fam::Car { "turn" } else { "road" };
        let n = t.nodes(x) as usize;
        let base = t.first[x as usize];
        // A layout id of this component, numbered from 0.
        let local = |v: u32, what: &str| -> Result<u32> {
            ensure!(v >= base && v - base < n as u32, "{f} component {x}: {what} {v} is outside its layout ids");
            Ok(v - base)
        };
        // Each array is written as it is read, to a mapped file when there is a scratch
        // directory, so that loading holds no second copy (SPEC §10).
        let sc = self.scratch.as_deref();
        let per_metric = || -> Result<Vec<ArrWriter<u16>>> { metrics.iter().map(|_| ArrWriter::new(sc)).collect() };
        let (mut up_first_w, mut up_head_w, mut up_w, mut down_w) = (ArrWriter::new(sc)?, ArrWriter::new(sc)?, per_metric()?, per_metric()?);
        let (mut arc_first_w, mut arc_head_w, mut raw_links, mut arc_local_w) = (ArrWriter::new(sc)?, ArrWriter::new(sc)?, ArrWriter::new(sc)?, ArrWriter::new(sc)?);
        let (mut turn_cost_w, mut way_w, mut way_dir_w, mut way_seq_w, mut kind_w) = (ArrWriter::new(sc)?, ArrWriter::new(sc)?, ArrWriter::new(sc)?, ArrWriter::new(sc)?, ArrWriter::new(sc)?);
        let mut time_s_w: Vec<ArrWriter<f32>> = metrics.iter().map(|_| ArrWriter::new(sc)).collect::<Result<_>>()?;
        let mut w_w = per_metric()?;
        let (mut copy_of_w, mut copy_first_w, mut copies_w) = (ArrWriter::new(sc)?, ArrWriter::new(sc)?, ArrWriter::new(sc)?);
        let mut blocks = Vec::new();
        let mut levels = Vec::new();
        let (mut up_last, mut arc_last) = (0u64, 0u64);
        up_first_w.push(0u64)?;
        arc_first_w.push(0u64)?;
        copy_first_w.push(0u32)?;
        for b in t.block_first[x as usize] as usize..t.block_first[x as usize + 1] as usize {
            let (pack, row_first, id_first, nodes) = (t.b_pack[b], t.b_row_first[b], t.b_id_first[b], t.b_nodes[b]);
            ensure!(id_first == base + up_first_w.len() as u32 - 1, "{f} component {x}: blocks do not follow layout ids");
            let id_first = id_first - base;
            let p = format!("road/{f}/p{pack}/");
            let u32s = |name: &str| m.get::<u32>(&format!("{p}{name}"));
            let up_first = u32s("up_first")?;
            let up_head = u32s("up_head")?;
            let arc_first = u32s(&format!("{word}_first"))?;
            let arc_head = u32s(&format!("{word}_head"))?;
            let arc_link = u32s(&format!("{word}_link"))?;
            let rows = row_first as usize..(row_first + nodes) as usize;
            ensure!(rows.end < up_first.len() && rows.end < arc_first.len(), "{p}: rows out of range");
            blocks.push(BlockAt { kind: t.b_kind[b], id_first, pack_link_first: up_first[rows.start] as u64 });
            let (lb, nl) = (t.b_level_base[b] as usize, t.b_levels[b] as usize);
            let lf = u32s("level_first")?;
            ensure!(lb + nl < lf.len(), "{p}: block {b}'s level bounds out of range");
            let bounds: Vec<u32> = lf[lb..=lb + nl].iter().map(|&v| v.wrapping_sub(base)).collect();
            ensure!(
                bounds[0] == id_first && bounds[nl] == id_first + nodes && bounds.windows(2).all(|w| w[0] < w[1]),
                "{f} component {x} block {b}: level bounds do not cover its ids in order"
            );
            levels.push(bounds);
            let links = up_first[rows.start] as usize..up_first[rows.end] as usize;
            let arcs = arc_first[rows.start] as usize..arc_first[rows.end] as usize;
            for r in rows.clone() {
                up_last += (up_first[r + 1] - up_first[r]) as u64;
                arc_last += (arc_first[r + 1] - arc_first[r]) as u64;
                up_first_w.push(up_last)?;
                arc_first_w.push(arc_last)?;
            }
            for &h in &up_head[links.clone()] {
                up_head_w.push(local(h, "upper neighbour")?)?;
            }
            for (k, name) in metrics.iter().enumerate() {
                up_w[k].extend_from_slice(&m.get::<u16>(&format!("{p}{name}.up"))?[links.clone()])?;
                down_w[k].extend_from_slice(&m.get::<u16>(&format!("{p}{name}.down"))?[links.clone()])?;
            }
            for &h in &arc_head[arcs.clone()] {
                arc_head_w.push(local(h, "head")?)?;
            }
            if full {
                raw_links.extend_from_slice(&arc_link[arcs.clone()])?;
                for a in arcs.clone() {
                    arc_local_w.push(a as u32)?;
                }
                let range = if fam == Fam::Car { rows.clone() } else { arcs.clone() };
                way_w.extend_from_slice(&m.get::<i64>(&format!("{p}way"))?[range.clone()])?;
                way_dir_w.extend_from_slice(&m.get::<u8>(&format!("{p}way_dir"))?[range.clone()])?;
                way_seq_w.extend_from_slice(&u32s("way_seq")?[range.clone()])?;
                kind_w.extend_from_slice(&m.get::<u8>(&format!("{p}kind"))?[range.clone()])?;
                for (k, name) in metrics.iter().enumerate() {
                    time_s_w[k].extend_from_slice(&m.get::<f32>(&format!("{p}{name}.time_s"))?[range.clone()])?;
                    w_w[k].extend_from_slice(&m.get::<u16>(&format!("{p}{name}.w"))?[range.clone()])?;
                }
            }
            if fam == Fam::Car {
                turn_cost_w.extend_from_slice(&m.get::<u16>(&format!("{p}car.turn_cost"))?[arcs.clone()])?;
                if full {
                    for &o in &u32s("copy_of")?[rows.clone()] {
                        copy_of_w.push(if o == NONE { NONE } else { local(o, "copy_of")? })?;
                    }
                }
                let cf = u32s("copy_first")?;
                let cs = u32s("copies")?;
                for r in rows.clone() {
                    for &c in &cs[cf[r] as usize..cf[r + 1] as usize] {
                        copies_w.push(local(c, "copy")?)?;
                    }
                    copy_first_w.push(copies_w.len() as u32)?;
                }
            }
        }
        ensure!(up_first_w.len() == n + 1, "{f} component {x}: blocks cover {} of {n} nodes", up_first_w.len() - 1);
        let all = |ws: Vec<ArrWriter<u16>>| -> Result<Vec<Arr<u16>>> { ws.into_iter().map(ArrWriter::finish).collect() };
        let mut e = Component {
            fam,
            n,
            blocks,
            levels,
            up_first: up_first_w.finish()?,
            up_head: up_head_w.finish()?,
            up: all(up_w)?,
            down: all(down_w)?,
            arc_first: arc_first_w.finish()?,
            arc_head: arc_head_w.finish()?,
            arc_link: Arr::default(),
            arc_local: arc_local_w.finish()?,
            turn_cost: turn_cost_w.finish()?,
            way: way_w.finish()?,
            way_dir: way_dir_w.finish()?,
            way_seq: way_seq_w.finish()?,
            kind: kind_w.finish()?,
            time_s: time_s_w.into_iter().map(ArrWriter::finish).collect::<Result<_>>()?,
            w: all(w_w)?,
            copy_of: copy_of_w.finish()?,
            copy_first: copy_first_w.finish()?,
            copies: copies_w.finish()?,
        };
        if !full {
            return Ok(e);
        }
        let raw_links = raw_links.finish()?;
        // Arc links: pack-local in the pack of the lower node → component-local.
        let mut arc_link = ArrWriter::new(sc)?;
        let mut tail = 0u32;
        for a in 0..e.arc_head.len() {
            while e.arc_first[tail as usize + 1] as usize <= a {
                tail += 1;
            }
            let raw = raw_links[a];
            let h = e.arc_head[a];
            if raw == NONE {
                ensure!(h == tail, "{f} component {x}: a NONE link on an arc {tail} -> {h}");
                arc_link.push(NONE64)?;
                continue;
            }
            let lo = tail.max(h);
            let bl = e.blocks[e.block_of(lo)];
            let local = e.up_first[bl.id_first as usize] + (raw as u64 - bl.pack_link_first);
            let expect = e.link(tail, h).with_context(|| format!("{f} component {x}: arc {tail} -> {h} has no link"))?;
            ensure!(local as usize == expect, "{f} component {x}: arc {tail} -> {h} names link {local}, not {expect}");
            arc_link.push(local)?;
        }
        e.arc_link = arc_link.finish()?;
        Ok(e)
    }
}
