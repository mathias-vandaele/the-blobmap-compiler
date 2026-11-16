//! What the runtime reads from the model besides the hierarchy: each family's tables
//! (FORMAT §5.1) and the vocabulary (FORMAT §7), held in host memory (warm).

use anyhow::{Context, Result, bail, ensure};
use blobmap_model::manifest::Manifest;
use blobmap_model::{Model, NONE, NONE16};
use serde_json::Value;

pub use blobmap_model::reference::model::Id;
pub use blobmap_model::reference::{Dir, Fam, Reason};

/// A family's component and block tables (FORMAT §5.1).
pub struct Family {
    pub fam: Fam,
    pub metrics: Vec<String>,
    /// Per component, its first layout id; the node count at the end.
    pub first: Vec<u32>,
    /// Per component, its first block; the block count at the end.
    pub block_first: Vec<u32>,
    pub b_kind: Vec<u8>,
    pub b_id_first: Vec<u32>,
    pub b_nodes: Vec<u32>,
    pub b_pack: Vec<u32>,
    pub b_row_first: Vec<u32>,
    pub b_levels: Vec<u32>,
    pub b_level_base: Vec<u32>,
}

impl Family {
    fn load(m: &Model, man: &Manifest, fam: Fam) -> Result<Family> {
        let f = fam.name();
        let metrics = man.families.get(f).with_context(|| format!("no family {f}"))?.metrics.clone();
        let g = |t: &str| -> Result<Vec<u32>> { Ok(m.get::<u32>(&format!("road/{f}/{t}"))?.into_owned()) };
        let component = g("nodes/component")?;
        ensure!(component.first().is_none_or(|&x| x == 0) && component.windows(2).all(|w| w[1] == w[0] || w[1] == w[0] + 1), "road/{f}/nodes/component: components are not consecutive ranges from 0");
        let k = component.last().map_or(0, |&x| x as usize + 1);
        let mut first = vec![0u32; k + 1];
        for &x in &component {
            first[x as usize + 1] += 1;
        }
        for i in 0..k {
            first[i + 1] += first[i];
        }
        let b_id_first = g("blocks/id_first")?;
        ensure!(b_id_first.windows(2).all(|w| w[0] < w[1]), "road/{f}/blocks: id_first is not ascending");
        let mut block_first: Vec<u32> = first[..k].iter().map(|&v| b_id_first.partition_point(|&b| b < v) as u32).collect();
        block_first.push(b_id_first.len() as u32);
        Ok(Family {
            fam,
            metrics,
            first,
            block_first,
            b_kind: m.get::<u8>(&format!("road/{f}/blocks/kind"))?.into_owned(),
            b_id_first,
            b_nodes: g("blocks/nodes")?,
            b_pack: g("blocks/pack")?,
            b_row_first: g("blocks/row_first")?,
            b_levels: g("blocks/levels")?,
            b_level_base: g("blocks/level_base")?,
        })
    }

    pub fn components(&self) -> usize {
        self.first.len() - 1
    }

    pub fn nodes(&self, x: u32) -> usize {
        (self.first[x as usize + 1] - self.first[x as usize]) as usize
    }
}

/// One mode's entries (FORMAT §7.2) in a common shape. CAR: `node` = `seg`, `road` = NONE,
/// `off` = [before₀, after₀, before₁, after₁]. ACTIVE: `node` = [U, V], `off` = [U → ID,
/// ID → U, V → ID, ID → V]. Nodes are numbered within the ID's component (`component`,
/// `NONE` when the ID is not served).
pub struct Mode {
    pub name: String,
    pub fam: Fam,
    pub metric: usize,
    pub component: Vec<u32>,
    pub node: Vec<[u32; 2]>,
    pub road: Vec<u32>,
    pub off: Vec<[u16; 4]>,
    /// (internal index, code), sorted by index.
    pub reasons: Vec<(u32, u8)>,
}

pub struct Vocab {
    pub address: Vec<Id>,
    pub place: Vec<Id>,
    pub alias: Vec<Id>,
    pub alias_to: Vec<u32>,
    pub modes: Vec<Mode>,
    pub status: Vec<u8>,
    pub primary: Vec<u16>,
    pub basic: Vec<u16>,
    pub alt_first: Vec<u32>,
    pub alt: Vec<u16>,
    pub tax_name: Vec<String>,
    pub tax_parent: Vec<u16>,
    pub tax_is_basic: Vec<u8>,
    pub redirect: Vec<String>,
    pub redirect_to: Vec<u16>,
}

/// A category of a query (SPEC §6): a taxonomy node or a basic category.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Category {
    Taxonomy(u16),
    Basic(u16),
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

pub fn load(m: &Model, man: &Manifest) -> Result<(Vec<Family>, Vocab)> {
    let fams = vec![Family::load(m, man, Fam::Car)?, Family::load(m, man, Fam::Active)?];
    let address = ids(m, "vocab/ids/address")?;
    let place = ids(m, "vocab/ids/place")?;
    let n = address.len() + place.len();
    let mut modes = Vec::new();
    for (name, mode) in &man.modes {
        let fam = if mode.family == "car" { Fam::Car } else { Fam::Active };
        let t = &fams[fam as usize];
        let metric = t.metrics.iter().position(|x| x == &mode.metric).context("mode metric")?;
        let p = format!("vocab/{}/{name}.", mode.family);
        let (nodes, k) = m.get_rows::<u32>(&format!("{p}{}", if fam == Fam::Car { "seg" } else { "node" }), &[2])?;
        let (o, k2) = m.get_rows::<u16>(&format!("{p}off"), if fam == Fam::Car { &[2, 2] } else { &[4] })?;
        ensure!(k == n && k2 == n, "{p}: entries do not cover every ID");
        let road = if fam == Fam::Car { vec![NONE; n] } else { m.get_len::<u32>(&format!("{p}road"), n)?.into_owned() };
        let mut node: Vec<[u32; 2]> = (0..n).map(|i| [nodes[2 * i], nodes[2 * i + 1]]).collect();
        let off: Vec<[u16; 4]> = (0..n).map(|i| o[4 * i..4 * i + 4].try_into().unwrap()).collect();
        let mut component = vec![NONE; n];
        for (i, nd) in node.iter_mut().enumerate() {
            let Some(&v) = nd.iter().find(|&&v| v != NONE) else { continue };
            ensure!((v as usize) < *t.first.last().unwrap() as usize, "{p}: ID {i} names layout id {v}, out of range");
            let x = (t.first.partition_point(|&f| f <= v) - 1) as u32;
            component[i] = x;
            for v in nd.iter_mut().filter(|v| **v != NONE) {
                ensure!(*v >= t.first[x as usize] && *v < t.first[x as usize + 1], "{p}: ID {i} has nodes in two components");
                *v -= t.first[x as usize];
            }
        }
        let rid = m.get::<u32>(&format!("{p}reason_id"))?;
        let rcode = m.get_len::<u8>(&format!("{p}reason_code"), rid.len())?;
        let reasons = (0..rid.len()).map(|k| (rid[k], rcode[k])).collect();
        modes.push(Mode { name: name.clone(), fam, metric, component, node, road, off, reasons });
    }
    let np = place.len();
    let vocab = Vocab {
        alias: ids(m, "vocab/ids/alias")?,
        alias_to: m.get::<u32>("vocab/ids/alias_to")?.into_owned(),
        modes,
        status: m.get_len::<u8>("vocab/places/status", np)?.into_owned(),
        primary: m.get_len::<u16>("vocab/places/primary", np)?.into_owned(),
        basic: m.get_len::<u16>("vocab/places/basic", np)?.into_owned(),
        alt_first: m.get_len::<u32>("vocab/places/alt_first", np + 1)?.into_owned(),
        alt: m.get::<u16>("vocab/places/alt")?.into_owned(),
        tax_name: strings(m, "vocab/taxonomy/name")?,
        tax_parent: m.get::<u16>("vocab/taxonomy/parent")?.into_owned(),
        tax_is_basic: m.get::<u8>("vocab/taxonomy/is_basic")?.into_owned(),
        redirect: strings(m, "vocab/taxonomy/redirect")?,
        redirect_to: m.get::<u16>("vocab/taxonomy/redirect_to")?.into_owned(),
        address,
        place,
    };
    Ok((fams, vocab))
}

pub fn reason_from_code(c: u8) -> Reason {
    match c {
        1 => Reason::NoEligibleRoad,
        3 => Reason::TrappedComponent,
        4 => Reason::FilteredPlace,
        _ => Reason::NoRoad,
    }
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

    pub fn status_of(&self, i: u32) -> u8 {
        if i as usize >= self.address.len() { self.status[i as usize - self.address.len()] } else { 0 }
    }

    /// Whether internal index i is served for the mode (its component), or why not.
    pub fn served(&self, i: u32, mode: usize) -> Result<u32, Reason> {
        if self.status_of(i) == 4 {
            return Err(Reason::FilteredPlace);
        }
        let mv = &self.modes[mode];
        match mv.component[i as usize] {
            NONE => {
                let k = mv.reasons.binary_search_by_key(&i, |r| r.0).map_err(|_| Reason::NoRoad)?;
                Err(reason_from_code(mv.reasons[k].1))
            }
            x => Ok(x),
        }
    }

    /// A category key after Overture's redirects (FORMAT §7.4).
    pub fn category_key(&self, key: &str, basic: bool) -> Result<Category> {
        let key = match self.redirect.binary_search_by(|x| x.as_str().cmp(key)) {
            Ok(k) => self.tax_name[self.redirect_to[k] as usize].as_str(),
            Err(_) => key,
        };
        let node = self.tax_name.binary_search_by(|x| x.as_str().cmp(key)).ok().with_context(|| format!("no category {key:?}"))? as u16;
        if basic {
            ensure!(self.tax_is_basic[node as usize] == 1, "{key:?} is not a basic category");
            Ok(Category::Basic(node))
        } else {
            Ok(Category::Taxonomy(node))
        }
    }

    /// `{"category": key}` or `{"basic": key}`.
    pub fn category(&self, v: &Value) -> Result<Category> {
        match (v.get("category"), v.get("basic")) {
            (Some(k), None) => self.category_key(k.as_str().context("category key")?, false),
            (None, Some(k)) => self.category_key(k.as_str().context("basic key")?, true),
            _ => bail!("a category is {{\"category\": key}} or {{\"basic\": key}}"),
        }
    }

    /// Whether place p (index into places) matches the category (FORMAT §7.4).
    pub fn matches(&self, p: usize, c: Category) -> bool {
        if self.status[p] != 0 || self.primary[p] == NONE16 {
            return false;
        }
        match c {
            Category::Basic(b) => self.basic[p] == b,
            Category::Taxonomy(x) => {
                let reaches = |mut n: u16| {
                    while n != NONE16 {
                        if n == x {
                            return true;
                        }
                        n = self.tax_parent[n as usize];
                    }
                    false
                };
                reaches(self.primary[p]) || self.alt[self.alt_first[p] as usize..self.alt_first[p + 1] as usize].iter().any(|&a| reaches(a))
            }
        }
    }
}

pub fn parse_id(s: &str) -> Option<Id> {
    let b = s.as_bytes();
    let ok = b.len() == 36 && b.iter().enumerate().all(|(i, &c)| if matches!(i, 8 | 13 | 18 | 23) { c == b'-' } else { c.is_ascii_digit() || (b'a'..=b'f').contains(&c) });
    if !ok {
        return None;
    }
    let hex: String = s.chars().filter(|&c| c != '-').collect();
    let mut id = [0u8; 16];
    for (k, x) in id.iter_mut().enumerate() {
        *x = u8::from_str_radix(&hex[2 * k..2 * k + 2], 16).ok()?;
    }
    Some(id)
}

pub fn id_string(id: &Id) -> String {
    let h = hex::encode(id);
    format!("{}-{}-{}-{}-{}", &h[0..8], &h[8..12], &h[12..16], &h[16..20], &h[20..32])
}
