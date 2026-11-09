//! Reading an Overture release directory (ported from the old `overture/release.rs`):
//! `raw/address.parquet` and `raw/place.parquet`, row groups decoded in parallel and put
//! back in file order.

use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use arrow_array::cast::AsArray;
use arrow_array::types::Float64Type;
use arrow_array::{Array, ArrayRef, RecordBatch, StructArray};
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rayon::prelude::*;

pub struct Release {
    pub dir: PathBuf,
    /// The directory's name, e.g. "2026-10-21.0".
    pub name: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RawAddress {
    pub id: String,
    pub lat: f64,
    pub lon: f64,
    pub country: String,
    pub licenses: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RawPlace {
    pub id: String,
    pub lat: f64,
    pub lon: f64,
    pub primary: Option<String>,
    pub alternates: Vec<String>,
    pub basic: Option<String>,
    /// NULL is 0.
    pub confidence: f64,
    pub status: Option<String>,
    pub name: String,
    pub country: String,
    pub licenses: Vec<String>,
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

    pub fn read_addresses(&self, keep: &(dyn Fn(f64, f64) -> bool + Sync)) -> Result<Vec<RawAddress>> {
        let groups = read_groups(&self.addresses(), &["id", "geometry", "country", "sources"], |b, first| {
            let id = b.column(0).as_string::<i32>();
            let geom = b.column(1).as_binary::<i32>();
            let country = b.column(2).as_string::<i32>();
            let sources = b.column(3).as_list::<i32>();
            let mut out = Vec::new();
            for i in 0..b.num_rows() {
                let (lat, lon) = wkb_point(geom.value(i)).with_context(|| format!("address row {}", first + i))?;
                if keep(lat, lon) {
                    out.push(RawAddress {
                        id: id.value(i).to_string(),
                        lat,
                        lon,
                        country: s(country, i).unwrap_or("").to_string(),
                        licenses: licenses(sources, i)?,
                    });
                }
            }
            Ok(out)
        })?;
        Ok(groups.into_iter().flatten().collect())
    }

    pub fn read_places(&self, keep: &(dyn Fn(f64, f64) -> bool + Sync)) -> Result<Vec<RawPlace>> {
        let cols = ["id", "geometry", "taxonomy", "basic_category", "confidence", "operating_status", "names", "sources", "addresses"];
        let groups = read_groups(&self.places(), &cols, |b, first| {
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
            let mut out = Vec::new();
            for i in 0..b.num_rows() {
                let (lat, lon) = wkb_point(geom.value(i)).with_context(|| format!("place row {}", first + i))?;
                if !keep(lat, lon) {
                    continue;
                }
                let tax_ok = tax.is_valid(i);
                let mut alts = Vec::new();
                if tax_ok && alternates.is_valid(i) {
                    let o = alternates.value_offsets();
                    for k in o[i] as usize..o[i + 1] as usize {
                        if let Some(a) = s(alt_values, k) {
                            alts.push(a.to_string());
                        }
                    }
                }
                let country = if addrs.is_valid(i) && addrs.value_length(i) > 0 {
                    s(addr_country, addrs.value_offsets()[i] as usize).unwrap_or("").to_string()
                } else {
                    String::new()
                };
                out.push(RawPlace {
                    id: id.value(i).to_string(),
                    lat,
                    lon,
                    primary: if tax_ok { s(primary, i).filter(|x| !x.is_empty()).map(String::from) } else { None },
                    alternates: alts,
                    basic: s(basic, i).filter(|x| !x.is_empty()).map(String::from),
                    confidence: if conf.is_valid(i) { conf.value(i) } else { 0.0 },
                    status: s(status, i).map(String::from),
                    name: if names.is_valid(i) { s(name, i).unwrap_or("").to_string() } else { String::new() },
                    country,
                    licenses: licenses(sources, i)?,
                });
            }
            Ok(out)
        })?;
        Ok(groups.into_iter().flatten().collect())
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
fn read_groups<T: Send>(path: &Path, cols: &[&str], f: impl Fn(&RecordBatch, usize) -> Result<Vec<T>> + Sync) -> Result<Vec<Vec<T>>> {
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
    let per: Vec<Vec<Vec<T>>> = (0..starts.len() - 1)
        .into_par_iter()
        .map(|g| -> Result<Vec<Vec<T>>> {
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
