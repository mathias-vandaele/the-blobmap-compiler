//! Opening a model: the manifest, the family tables, the vocabulary, and components loaded on
//! demand from their packs (FORMAT §5, §7). A component is a range of the family's layout
//! ids; loaded, its nodes are numbered from 0, and so are the vocabulary's nodes.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, bail, ensure};

use crate::manifest::Manifest;
use crate::{Model, NONE};

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
    pub component: Vec<u32>,
    pub node: Vec<[u32; 2]>,
    pub road: Vec<u32>,
    pub off: Vec<[u16; 4]>,
    /// (internal index, code, metres), sorted by index.
    pub reasons: Vec<(u32, u8, f32)>,
}

pub struct Vocab {
    pub address: Vec<Id>,
    pub place: Vec<Id>,
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
    pub up_first: Vec<u64>,
    pub up_head: Vec<u32>,
    /// Per metric.
    pub up: Vec<Vec<u16>>,
    pub down: Vec<Vec<u16>>,
    /// Turns (CAR) or roads (ACTIVE), CSR by tail.
    pub arc_first: Vec<u64>,
    pub arc_head: Vec<u32>,
    /// Component-local link of each arc, `NONE` when tail = head.
    pub arc_link: Vec<u32>,
    /// The pack-local index of each arc (ACTIVE `road`).
    pub arc_local: Vec<u32>,
    pub turn_cost: Vec<u16>,
    /// The roads' tensors: per node for CAR, per arc for ACTIVE.
    pub way: Vec<i64>,
    pub way_dir: Vec<u8>,
    pub way_seq: Vec<u32>,
    pub kind: Vec<u8>,
    pub time_s: Vec<Vec<f32>>,
    pub w: Vec<Vec<u16>>,
    pub copy_of: Vec<u32>,
    pub copy_first: Vec<u32>,
    pub copies: Vec<u32>,
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
    cache: Mutex<BTreeMap<(Fam, u32), Arc<Component>>>,
    /// Held while a component loads, so that concurrent queries load it once.
    loading: Mutex<()>,
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

impl Runtime {
    pub fn open(dir: &Path) -> Result<Runtime> {
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
        let address = ids(&model, "vocab/ids/address")?;
        let place = ids(&model, "vocab/ids/place")?;
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
            loading: Mutex::new(()),
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
        let _one = self.loading.lock().unwrap();
        if let Some(e) = self.cache.lock().unwrap().get(&(fam, x)) {
            return Ok(e.clone());
        }
        let e = Arc::new(self.load(fam, x)?);
        self.cache.lock().unwrap().insert((fam, x), e.clone());
        Ok(e)
    }

    fn load(&self, fam: Fam, x: u32) -> Result<Component> {
        let t = &self.tables[&fam];
        let f = fam.name();
        let m = &self.model;
        ensure!((x as usize) < t.components(), "{f} component {x} does not exist");
        let metrics = &t.metrics;
        let word = if fam == Fam::Car { "turn" } else { "road" };
        let n = t.nodes(x) as usize;
        let base = t.first[x as usize];
        // A layout id of this component, numbered from 0.
        let local = |v: u32, what: &str| -> Result<u32> {
            ensure!(v >= base && v - base < n as u32, "{f} component {x}: {what} {v} is outside its layout ids");
            Ok(v - base)
        };
        let mut e = Component {
            fam,
            n,
            blocks: Vec::new(),
            up_first: vec![0],
            up_head: Vec::new(),
            up: vec![Vec::new(); metrics.len()],
            down: vec![Vec::new(); metrics.len()],
            arc_first: vec![0],
            arc_head: Vec::new(),
            arc_link: Vec::new(),
            arc_local: Vec::new(),
            turn_cost: Vec::new(),
            way: Vec::new(),
            way_dir: Vec::new(),
            way_seq: Vec::new(),
            kind: Vec::new(),
            time_s: vec![Vec::new(); metrics.len()],
            w: vec![Vec::new(); metrics.len()],
            copy_of: Vec::new(),
            copy_first: vec![0],
            copies: Vec::new(),
        };
        let mut raw_links: Vec<u32> = Vec::new();
        for b in t.block_first[x as usize] as usize..t.block_first[x as usize + 1] as usize {
            let (pack, row_first, id_first, nodes) = (t.b_pack[b], t.b_row_first[b], t.b_id_first[b], t.b_nodes[b]);
            ensure!(id_first == base + e.up_first.len() as u32 - 1, "{f} component {x}: blocks do not follow layout ids");
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
            e.blocks.push(BlockAt { kind: t.b_kind[b], id_first, pack_link_first: up_first[rows.start] as u64 });
            let links = up_first[rows.start] as usize..up_first[rows.end] as usize;
            let arcs = arc_first[rows.start] as usize..arc_first[rows.end] as usize;
            for r in rows.clone() {
                e.up_first.push(e.up_first.last().unwrap() + (up_first[r + 1] - up_first[r]) as u64);
                e.arc_first.push(e.arc_first.last().unwrap() + (arc_first[r + 1] - arc_first[r]) as u64);
            }
            for &h in &up_head[links.clone()] {
                e.up_head.push(local(h, "upper neighbour")?);
            }
            for (k, name) in metrics.iter().enumerate() {
                e.up[k].extend_from_slice(&m.get::<u16>(&format!("{p}{name}.up"))?[links.clone()]);
                e.down[k].extend_from_slice(&m.get::<u16>(&format!("{p}{name}.down"))?[links.clone()]);
            }
            for &h in &arc_head[arcs.clone()] {
                e.arc_head.push(local(h, "head")?);
            }
            raw_links.extend_from_slice(&arc_link[arcs.clone()]);
            e.arc_local.extend(arcs.clone().map(|a| a as u32));
            let range = if fam == Fam::Car { rows.clone() } else { arcs.clone() };
            e.way.extend_from_slice(&m.get::<i64>(&format!("{p}way"))?[range.clone()]);
            e.way_dir.extend_from_slice(&m.get::<u8>(&format!("{p}way_dir"))?[range.clone()]);
            e.way_seq.extend_from_slice(&u32s("way_seq")?[range.clone()]);
            e.kind.extend_from_slice(&m.get::<u8>(&format!("{p}kind"))?[range.clone()]);
            for (k, name) in metrics.iter().enumerate() {
                e.time_s[k].extend_from_slice(&m.get::<f32>(&format!("{p}{name}.time_s"))?[range.clone()]);
                e.w[k].extend_from_slice(&m.get::<u16>(&format!("{p}{name}.w"))?[range.clone()]);
            }
            if fam == Fam::Car {
                e.turn_cost.extend_from_slice(&m.get::<u16>(&format!("{p}car.turn_cost"))?[arcs.clone()]);
                for &o in &u32s("copy_of")?[rows.clone()] {
                    e.copy_of.push(if o == NONE { NONE } else { local(o, "copy_of")? });
                }
                let cf = u32s("copy_first")?;
                let cs = u32s("copies")?;
                for r in rows.clone() {
                    for &c in &cs[cf[r] as usize..cf[r + 1] as usize] {
                        e.copies.push(local(c, "copy")?);
                    }
                    e.copy_first.push(e.copies.len() as u32);
                }
            }
        }
        ensure!(e.up_first.len() == n + 1, "{f} component {x}: blocks cover {} of {n} nodes", e.up_first.len() - 1);
        // Arc links: pack-local in the pack of the lower node → component-local.
        let mut tail = 0u32;
        for a in 0..e.arc_head.len() {
            while e.arc_first[tail as usize + 1] as usize <= a {
                tail += 1;
            }
            let raw = raw_links[a];
            let h = e.arc_head[a];
            if raw == NONE {
                ensure!(h == tail, "{f} component {x}: a NONE link on an arc {tail} -> {h}");
                e.arc_link.push(NONE);
                continue;
            }
            let lo = tail.max(h);
            let bl = e.blocks[e.block_of(lo)];
            let local = e.up_first[bl.id_first as usize] + (raw as u64 - bl.pack_link_first);
            let expect = e.link(tail, h).with_context(|| format!("{f} component {x}: arc {tail} -> {h} has no link"))?;
            ensure!(local as usize == expect, "{f} component {x}: arc {tail} -> {h} names link {local}, not {expect}");
            e.arc_link.push(local as u32);
        }
        Ok(e)
    }
}
