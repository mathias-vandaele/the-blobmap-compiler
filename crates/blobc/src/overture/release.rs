//! Reading an Overture release directory (ported from the old `overture/release.rs`):
//! `raw/address.parquet` and `raw/place.parquet`, row groups decoded in parallel and put
//! back in file order.

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{Context, Result, bail, ensure};
use arrow_array::cast::AsArray;
use arrow_array::types::Float64Type;
use arrow_array::{Array, ArrayRef, RecordBatch, StructArray};
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rayon::prelude::*;

use crate::overture::{Address, Id, parse_uuid};

pub struct Release {
    pub dir: PathBuf,
    /// The directory's name, e.g. "2026-10-21.0".
    pub name: String,
}

/// No string: a NULL, or an empty category key or country.
pub const NO_STR: u32 = u32::MAX;

/// A place row as read, compact: its ID parsed, the strings a release repeats (category
/// keys, statuses, countries) as numbers in `RawPlaces::strings`, and its name and
/// alternate categories in pools, so that the planet's places are held in a few bytes each.
#[derive(Clone, Debug, PartialEq)]
pub struct RawPlace {
    pub id: Id,
    pub lat: f64,
    pub lon: f64,
    /// NULL is 0.
    pub confidence: f64,
    pub primary: u32,
    pub basic: u32,
    pub status: u32,
    pub country: u32,
    /// The licence set's number in the reading's `Licenses`.
    pub license: u32,
}

/// The place rows read, with the strings and pools their numbers point into.
#[derive(Debug)]
pub struct RawPlaces {
    pub rows: Vec<RawPlace>,
    /// Numbered as first met in file order.
    pub strings: Vec<String>,
    /// Row i's alternate categories are `alts[alt_first[i]..alt_first[i + 1]]`.
    alt_first: Vec<u64>,
    alts: Vec<u32>,
    /// Row i's primary name is `names[name_first[i]..name_first[i + 1]]`.
    name_first: Vec<u64>,
    names: Vec<u8>,
}

impl RawPlaces {
    fn new() -> RawPlaces {
        RawPlaces { rows: Vec::new(), strings: Vec::new(), alt_first: vec![0], alts: Vec::new(), name_first: vec![0], names: Vec::new() }
    }

    pub fn str(&self, k: u32) -> Option<&str> {
        (k != NO_STR).then(|| self.strings[k as usize].as_str())
    }

    pub fn alternates(&self, i: usize) -> &[u32] {
        &self.alts[self.alt_first[i] as usize..self.alt_first[i + 1] as usize]
    }

    pub fn name(&self, i: usize) -> &str {
        // SAFETY: the pool holds whole UTF-8 strings, cut at their ends.
        unsafe { std::str::from_utf8_unchecked(&self.names[self.name_first[i] as usize..self.name_first[i + 1] as usize]) }
    }

    /// Appends a batch, its local string numbers turned into this table's.
    fn append(&mut self, b: RawPlaces, index: &mut HashMap<String, u32>) {
        let global: Vec<u32> = b
            .strings
            .into_iter()
            .map(|s| {
                let next = self.strings.len() as u32;
                *index.entry(s).or_insert_with_key(|s| {
                    self.strings.push(s.clone());
                    next
                })
            })
            .collect();
        let g = |k: u32| if k == NO_STR { NO_STR } else { global[k as usize] };
        self.rows.extend(b.rows.into_iter().map(|r| RawPlace { primary: g(r.primary), basic: g(r.basic), status: g(r.status), country: g(r.country), ..r }));
        let (a0, n0) = (self.alts.len() as u64, self.names.len() as u64);
        self.alts.extend(b.alts.into_iter().map(g));
        self.alt_first.extend(b.alt_first[1..].iter().map(|&x| a0 + x));
        self.names.extend_from_slice(&b.names);
        self.name_first.extend(b.name_first[1..].iter().map(|&x| n0 + x));
    }
}

/// An empty category key or country is none.
fn key(x: Option<&str>) -> Option<&str> {
    x.filter(|x| !x.is_empty())
}

/// A batch's own string numbers, as first met.
#[derive(Default)]
struct Local(HashMap<String, u32>, Vec<String>);

impl Local {
    fn number(&mut self, s: &str) -> u32 {
        if let Some(&k) = self.0.get(s) {
            return k;
        }
        let k = self.1.len() as u32;
        self.0.insert(s.to_string(), k);
        self.1.push(s.to_string());
        k
    }

    fn of(&mut self, s: Option<&str>) -> u32 {
        s.map_or(NO_STR, |s| self.number(s))
    }
}

/// The distinct licence sets met while reading, numbered as first met: the numbers depend
/// on the threads, so `sorted` gives the deterministic order and each number's place in it.
#[derive(Default)]
pub struct Licenses {
    sets: Mutex<BTreeMap<Vec<String>, u32>>,
}

impl Licenses {
    fn number(&self, set: Vec<String>) -> u32 {
        let mut sets = self.sets.lock().unwrap();
        let next = sets.len() as u32;
        *sets.entry(set).or_insert(next)
    }

    /// The sets in ascending order, and per number its index in that order.
    pub fn sorted(self) -> (Vec<Vec<String>>, Vec<u32>) {
        let sets = self.sets.into_inner().unwrap();
        let mut at = vec![0u32; sets.len()];
        for (k, n) in sets.values().enumerate() {
            at[*n as usize] = k as u32;
        }
        (sets.into_keys().collect(), at)
    }
}

impl Release {
    pub fn open(dir: &Path) -> Result<Release> {
        let name = dir.file_name().and_then(|n| n.to_str()).with_context(|| format!("{}: no release name", dir.display()))?.to_string();
        let r = Release { dir: dir.to_path_buf(), name };
        for p in [r.addresses(), r.places(), r.taxonomy_dir().join("taxonomy.csv")] {
            ensure!(p.exists(), "Overture release {}: no {}", dir.display(), p.display());
        }
        Ok(r)
    }

    pub fn addresses(&self) -> PathBuf {
        self.dir.join("raw/address.parquet")
    }

    pub fn places(&self) -> PathBuf {
        self.dir.join("raw/place.parquet")
    }

    pub fn taxonomy_dir(&self) -> PathBuf {
        self.dir.join("taxonomy")
    }

    /// The files the manifest lists (FORMAT §3 `inputs.overture.files`).
    pub fn files(&self) -> [PathBuf; 2] {
        [self.addresses(), self.places()]
    }

    /// The address rows `keep` accepts, in file order, each licence set as its number in
    /// `lic`.
    pub fn read_addresses(&self, keep: &(dyn Fn(f64, f64) -> bool + Sync), lic: &Licenses) -> Result<Vec<Address>> {
        let batches = read_groups(&self.addresses(), &["id", "geometry", "sources"], |b, first| {
            let id = b.column(0).as_string::<i32>();
            let geom = b.column(1).as_binary::<i32>();
            let sources = b.column(2).as_list::<i32>();
            let mut out = Vec::new();
            for i in 0..b.num_rows() {
                let (lat, lon) = wkb_point(geom.value(i)).with_context(|| format!("address row {}", first + i))?;
                if keep(lat, lon) {
                    out.push(Address { id: parse_uuid(id.value(i))?, lat, lon, license: lic.number(licenses(sources, i)?) });
                }
            }
            Ok(out)
        })?;
        let mut out = Vec::with_capacity(batches.iter().map(Vec::len).sum());
        for b in batches {
            out.extend(b);
        }
        Ok(out)
    }

    /// The place rows `keep` accepts, in file order, each licence set as its number in `lic`.
    pub fn read_places(&self, keep: &(dyn Fn(f64, f64) -> bool + Sync), lic: &Licenses) -> Result<RawPlaces> {
        let cols = ["id", "geometry", "taxonomy", "basic_category", "confidence", "operating_status", "names", "sources", "addresses"];
        let batches = read_groups(&self.places(), &cols, |b, first| {
            let id = b.column(0).as_string::<i32>();
            let geom = b.column(1).as_binary::<i32>();
            let tax = b.column(2).as_struct();
            let primary = field(tax, "primary")?;
            let primary = primary.as_string::<i32>();
            let alternates = field(tax, "alternates")?;
            let alternates = alternates.as_list::<i32>();
            let alt_values = alternates.values().as_string::<i32>();
            let basic = b.column(3).as_string::<i32>();
            let conf = b.column(4).as_primitive::<Float64Type>();
            let status = b.column(5).as_string::<i32>();
            let names = b.column(6).as_struct();
            let name = field(names, "primary")?;
            let name = name.as_string::<i32>();
            let sources = b.column(7).as_list::<i32>();
            let addrs = b.column(8).as_list::<i32>();
            let addr_country = field(addrs.values().as_struct(), "country")?;
            let addr_country = addr_country.as_string::<i32>();
            let mut out = RawPlaces::new();
            let mut local = Local::default();
            for i in 0..b.num_rows() {
                let (lat, lon) = wkb_point(geom.value(i)).with_context(|| format!("place row {}", first + i))?;
                if !keep(lat, lon) {
                    continue;
                }
                let tax_ok = tax.is_valid(i);
                if tax_ok && alternates.is_valid(i) {
                    let o = alternates.value_offsets();
                    for k in o[i] as usize..o[i + 1] as usize {
                        if let Some(a) = s(alt_values, k) {
                            out.alts.push(local.number(a));
                        }
                    }
                }
                out.alt_first.push(out.alts.len() as u64);
                let country = if addrs.is_valid(i) && addrs.value_length(i) > 0 { s(addr_country, addrs.value_offsets()[i] as usize) } else { None };
                if names.is_valid(i) {
                    out.names.extend_from_slice(s(name, i).unwrap_or("").as_bytes());
                }
                out.name_first.push(out.names.len() as u64);
                out.rows.push(RawPlace {
                    id: parse_uuid(id.value(i))?,
                    lat,
                    lon,
                    confidence: if conf.is_valid(i) { conf.value(i) } else { 0.0 },
                    primary: local.of(if tax_ok { key(s(primary, i)) } else { None }),
                    basic: local.of(key(s(basic, i))),
                    status: local.of(s(status, i)),
                    country: local.of(key(country)),
                    license: lic.number(licenses(sources, i)?),
                });
            }
            out.strings = local.1;
            Ok(out)
        })?;
        let mut out = RawPlaces::new();
        let mut index = HashMap::new();
        for b in batches {
            out.append(b, &mut index);
        }
        Ok(out)
    }
}

/// The distinct licences of row i's sources.
fn licenses(sources: &arrow_array::ListArray, i: usize) -> Result<Vec<String>> {
    if !sources.is_valid(i) {
        return Ok(Vec::new());
    }
    let lic = field(sources.values().as_struct(), "license")?;
    let lic = lic.as_string::<i32>();
    let o = sources.value_offsets();
    let mut v: Vec<String> = (o[i] as usize..o[i + 1] as usize).filter_map(|k| s(lic, k).map(String::from)).collect();
    v.sort();
    v.dedup();
    Ok(v)
}

fn s(a: &arrow_array::StringArray, i: usize) -> Option<&str> {
    a.is_valid(i).then(|| a.value(i))
}

fn field(s: &StructArray, name: &str) -> Result<ArrayRef> {
    s.column_by_name(name).cloned().with_context(|| format!("no struct field {name}"))
}

/// (lat, lon) of a WKB point (2D, either byte order; ISO or EWKB type codes).
pub fn wkb_point(b: &[u8]) -> Result<(f64, f64)> {
    ensure!(b.len() >= 21, "WKB too short for a point");
    let le = match b[0] {
        1 => true,
        0 => false,
        x => bail!("WKB byte order {x}"),
    };
    let u32_at = |o: usize| {
        let a = [b[o], b[o + 1], b[o + 2], b[o + 3]];
        if le { u32::from_le_bytes(a) } else { u32::from_be_bytes(a) }
    };
    let f64_at = |o: usize| {
        let mut a = [0u8; 8];
        a.copy_from_slice(&b[o..o + 8]);
        if le { f64::from_le_bytes(a) } else { f64::from_be_bytes(a) }
    };
    let t = u32_at(1);
    ensure!(t & 0xffff == 1 && t & 0x2000_0000 == 0, "WKB geometry type {t:#x} is not a 2D point");
    let (x, y) = (f64_at(5), f64_at(13));
    Ok((y, x))
}

/// Decodes the named root columns of every row group in parallel: `f(batch, first row)`
/// per batch, the results in file order.
fn read_groups<T: Send>(path: &Path, cols: &[&str], f: impl Fn(&RecordBatch, usize) -> Result<T> + Sync) -> Result<Vec<T>> {
    let open = || File::open(path).with_context(|| path.display().to_string());
    let meta = ParquetRecordBatchReaderBuilder::try_new(open()?)?;
    let schema = meta.parquet_schema().clone();
    let roots: Vec<usize> = cols
        .iter()
        .map(|c| {
            (0..schema.root_schema().get_fields().len())
                .find(|&i| schema.root_schema().get_fields()[i].name() == *c)
                .with_context(|| format!("{}: no column {c}", path.display()))
        })
        .collect::<Result<_>>()?;
    let mut starts = vec![0usize];
    for rg in meta.metadata().row_groups() {
        starts.push(starts.last().unwrap() + rg.num_rows() as usize);
    }
    drop(meta);
    let per: Vec<Vec<T>> = (0..starts.len() - 1)
        .into_par_iter()
        .map(|g| -> Result<Vec<T>> {
            let b = ParquetRecordBatchReaderBuilder::try_new(open()?)?;
            let mask = ProjectionMask::roots(b.parquet_schema(), roots.iter().copied());
            let reader = b.with_row_groups(vec![g]).with_projection(mask).with_batch_size(65536).build()?;
            let mut first = starts[g];
            let mut out = Vec::new();
            for batch in reader {
                let batch = batch?;
                let names: Vec<String> = batch.schema().fields().iter().map(|x| x.name().clone()).collect();
                let order: Vec<usize> = cols.iter().map(|c| names.iter().position(|n| n == c).unwrap()).collect();
                let batch = batch.project(&order)?;
                out.push(f(&batch, first)?);
                first += batch.num_rows();
            }
            Ok(out)
        })
        .collect::<Result<_>>()?;
    Ok(per.into_iter().flatten().collect())
}
