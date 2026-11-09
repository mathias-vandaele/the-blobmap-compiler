//! The validating reader (FORMAT §10): it maps every shard, checks each header against the
//! index, and serves logical tensors. A tensor that is one physical tensor is served in
//! place from the map (shards keep every tensor aligned, FORMAT §2.1); a chunked one is
//! concatenated on request.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fs::File;
use std::ops::Range;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use memmap2::Mmap;
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::dtype::{Dtype, Element};
use crate::hash::{Hash, tensor_hash};
use crate::{FORMAT, FORMAT_MAJOR, INDEX_FILE, MANIFEST_FILE, json, major};

struct Physical {
    shard: usize,
    dtype: Dtype,
    shape: Vec<u64>,
    range: Range<usize>,
}

struct Logical {
    dtype: Dtype,
    shape: Vec<u64>,
    /// Physical tensors, in chunk order (one for an unchunked tensor).
    parts: Vec<String>,
}

pub struct ShardInfo {
    pub file: String,
    pub bytes: u64,
    pub sha256: String,
}

/// A model directory, open for reading.
pub struct Model {
    pub root: PathBuf,
    pub manifest: Value,
    pub shards: Vec<ShardInfo>,
    maps: Vec<Mmap>,
    physical: BTreeMap<String, Physical>,
    logical: BTreeMap<String, Logical>,
}

fn str_field<'a>(v: &'a Value, key: &str) -> Result<&'a str> {
    v.get(key).and_then(Value::as_str).with_context(|| format!("missing string {key}"))
}

/// Parses one shard's header (FORMAT §2.1): tensors with contiguous, in-bounds data.
fn parse_shard(bytes: &[u8], file: &str) -> Result<Vec<(String, Dtype, Vec<u64>, Range<usize>)>> {
    ensure!(bytes.len() >= 8, "{file}: too short");
    let hl = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    ensure!(hl.is_multiple_of(8) && 8 + hl <= bytes.len(), "{file}: bad header length");
    let header: Value = serde_json::from_slice(&bytes[8..8 + hl]).with_context(|| format!("{file}: header"))?;
    let obj = header.as_object().with_context(|| format!("{file}: header is not an object"))?;
    let meta = obj.get("__metadata__").with_context(|| format!("{file}: no __metadata__"))?;
    ensure!(str_field(meta, "format")? == "blobmap-shard", "{file}: not a blobmap shard");
    ensure!(major(str_field(meta, "format_version")?) == Some(FORMAT_MAJOR), "{file}: another major version");
    let base = 8 + hl;
    let mut out = Vec::new();
    for (name, t) in obj {
        if name == "__metadata__" {
            continue;
        }
        let dtype = Dtype::parse(str_field(t, "dtype")?)?;
        let shape: Vec<u64> = serde_json::from_value(t["shape"].clone()).with_context(|| format!("{file}: {name} shape"))?;
        let off: [u64; 2] = serde_json::from_value(t["data_offsets"].clone()).with_context(|| format!("{file}: {name} offsets"))?;
        let n: u64 = shape.iter().product();
        ensure!(off[1] >= off[0] && off[1] - off[0] == n * dtype.size() as u64, "{file}: {name}: offsets do not match its shape");
        let range = base + off[0] as usize..base + off[1] as usize;
        ensure!(range.end <= bytes.len(), "{file}: {name}: data out of bounds");
        out.push((name.clone(), dtype, shape, range));
    }
    // Contiguous data without holes, as safetensors requires.
    let mut ranges: Vec<Range<usize>> = out.iter().map(|x| x.3.clone()).collect();
    ranges.sort_by_key(|r| (r.start, r.end));
    let mut at = base;
    for r in &ranges {
        ensure!(r.start == at, "{file}: tensor data is not contiguous");
        at = r.end;
    }
    ensure!(at == bytes.len(), "{file}: trailing bytes after the tensor data");
    Ok(out)
}

impl Model {
    pub fn open(root: &Path) -> Result<Model> {
        let manifest = json::read_file(&root.join(MANIFEST_FILE))?;
        ensure!(str_field(&manifest, "format")? == FORMAT, "not a {FORMAT} model");
        let v = str_field(&manifest, "format_version")?;
        ensure!(major(v) == Some(FORMAT_MAJOR), "format version {v}: this reader reads {FORMAT_MAJOR}.x only");
        let index = json::read_file(&root.join(INDEX_FILE))?;
        ensure!(str_field(&index["metadata"], "format")? == "blobmap-index", "not a blobmap index");
        let weight_map: BTreeMap<String, String> = serde_json::from_value(index["weight_map"].clone()).context("index weight_map")?;
        let chunks: BTreeMap<String, Vec<String>> = serde_json::from_value(index["chunks"].clone()).context("index chunks")?;
        let mut shards = Vec::new();
        for s in index["shards"].as_array().context("index shards")? {
            shards.push(ShardInfo {
                file: str_field(s, "file")?.to_string(),
                bytes: s["bytes"].as_u64().context("shard bytes")?,
                sha256: str_field(s, "sha256")?.to_string(),
            });
        }
        let mut maps = Vec::new();
        let mut physical = BTreeMap::new();
        for (k, s) in shards.iter().enumerate() {
            let path = root.join(&s.file);
            let f = File::open(&path).with_context(|| format!("opening {}", path.display()))?;
            // SAFETY: model files are written once and not modified while mapped.
            let m = unsafe { Mmap::map(&f) }.with_context(|| format!("mapping {}", path.display()))?;
            ensure!(m.len() as u64 == s.bytes, "{}: {} bytes, the index says {}", s.file, m.len(), s.bytes);
            for (name, dtype, shape, range) in parse_shard(&m, &s.file)? {
                ensure!(weight_map.get(&name) == Some(&s.file), "{}: holds {name}, which the index puts elsewhere", s.file);
                physical.insert(name, Physical { shard: k, dtype, shape, range });
            }
            maps.push(m);
        }
        ensure!(physical.len() == weight_map.len(), "the index lists {} tensors, the shards hold {}", weight_map.len(), physical.len());
        let mut logical = BTreeMap::new();
        for (name, parts) in &chunks {
            ensure!(!parts.is_empty(), "{name}: no chunks");
            let first = physical.get(&parts[0]).with_context(|| format!("{name}: chunk {} missing", parts[0]))?;
            let (dtype, mut shape) = (first.dtype, first.shape.clone());
            shape[0] = 0;
            for (i, p) in parts.iter().enumerate() {
                ensure!(*p == format!("{name}@{i}"), "{name}: chunk {i} is named {p}");
                let c = physical.get(p).with_context(|| format!("{name}: chunk {p} missing"))?;
                ensure!(c.dtype == dtype && c.shape[1..] == shape[1..], "{name}: chunks do not concatenate to one shape");
                shape[0] += c.shape[0];
            }
            logical.insert(name.clone(), Logical { dtype, shape, parts: parts.clone() });
        }
        for (name, p) in &physical {
            if let Some((base, _)) = name.rsplit_once('@') {
                ensure!(chunks.contains_key(base), "{name}: a chunk of an unlisted tensor");
                continue;
            }
            logical.insert(name.clone(), Logical { dtype: p.dtype, shape: p.shape.clone(), parts: vec![name.clone()] });
        }
        Ok(Model { root: root.to_path_buf(), manifest, shards, maps, physical, logical })
    }

    pub fn has(&self, name: &str) -> bool {
        self.logical.contains_key(name)
    }

    /// Logical tensor names, sorted.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.logical.keys().map(String::as_str)
    }

    /// Logical names under `<prefix>/`, sorted.
    pub fn names_in<'a>(&'a self, prefix: &'a str) -> impl Iterator<Item = &'a str> + 'a {
        let p = format!("{prefix}/");
        self.logical.range(p.clone()..).map(|(k, _)| k.as_str()).take_while(move |k| k.starts_with(&p))
    }

    pub fn info(&self, name: &str) -> Result<(Dtype, &[u64])> {
        let l = self.logical.get(name).with_context(|| format!("missing tensor {name}"))?;
        Ok((l.dtype, &l.shape))
    }

    fn part(&self, p: &str) -> &[u8] {
        let x = &self.physical[p];
        &self.maps[x.shard][x.range.clone()]
    }

    /// The tensor's data, borrowed when it is one physical tensor.
    pub fn bytes(&self, name: &str) -> Result<Cow<'_, [u8]>> {
        let l = self.logical.get(name).with_context(|| format!("missing tensor {name}"))?;
        if l.parts.len() == 1 {
            return Ok(Cow::Borrowed(self.part(&l.parts[0])));
        }
        let mut v = Vec::new();
        for p in &l.parts {
            v.extend_from_slice(self.part(p));
        }
        Ok(Cow::Owned(v))
    }

    /// A tensor of element type T and the given rank, with its shape.
    pub fn get_nd<T: Element>(&self, name: &str, rank: usize) -> Result<(Cow<'_, [T]>, Vec<u64>)> {
        let (dtype, shape) = self.info(name)?;
        if dtype != T::DTYPE {
            bail!("tensor {name}: dtype {}, expected {}", dtype.name(), T::DTYPE.name());
        }
        ensure!(shape.len() == rank, "tensor {name}: rank {}, expected {rank}", shape.len());
        let shape = shape.to_vec();
        let data = match self.bytes(name)? {
            Cow::Borrowed(b) => match bytemuck::try_cast_slice(b) {
                Ok(s) => Cow::Borrowed(s),
                Err(_) => Cow::Owned(bytemuck::pod_collect_to_vec(b)),
            },
            Cow::Owned(v) => Cow::Owned(bytemuck::pod_collect_to_vec(&v)),
        };
        Ok((data, shape))
    }

    /// A 1-D tensor.
    pub fn get<T: Element>(&self, name: &str) -> Result<Cow<'_, [T]>> {
        Ok(self.get_nd(name, 1)?.0)
    }

    /// A 1-D tensor of exactly `len` elements.
    pub fn get_len<T: Element>(&self, name: &str, len: usize) -> Result<Cow<'_, [T]>> {
        let v = self.get::<T>(name)?;
        ensure!(v.len() == len, "tensor {name}: {} elements, expected {len}", v.len());
        Ok(v)
    }

    /// A tensor of shape [n, dims...] with the given trailing dimensions; returns n.
    pub fn get_rows<T: Element>(&self, name: &str, dims: &[u64]) -> Result<(Cow<'_, [T]>, usize)> {
        let (v, shape) = self.get_nd::<T>(name, dims.len() + 1)?;
        ensure!(shape[1..] == *dims, "tensor {name}: shape {shape:?}, expected [n, {dims:?}]");
        Ok((v, shape[0] as usize))
    }

    /// The tensor hash of a logical tensor (FORMAT §9).
    pub fn tensor_hash(&self, name: &str) -> Result<Hash> {
        let l = self.logical.get(name).with_context(|| format!("missing tensor {name}"))?;
        Ok(tensor_hash(name, l.dtype, &l.shape, l.parts.iter().map(|p| self.part(p))))
    }

    /// Checks every shard file against its SHA-256 in the index (transport check).
    pub fn verify_shard_hashes(&self) -> Result<()> {
        for (s, m) in self.shards.iter().zip(&self.maps) {
            ensure!(hex::encode(Sha256::digest(&m[..])) == s.sha256, "{}: SHA-256 differs from the index", s.file);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Tensor, Unit};
    use crate::writer::{ModelWriter, ShardLimits};

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("blobmap-format-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn round_trip_with_chunks_and_alignment() {
        let dir = tmp("rt");
        let mut w = ModelWriter::create(&dir, ShardLimits { target: 64, max: 128 }).unwrap();
        let mut u = Unit::default();
        let big: Vec<u32> = (0..40).collect(); // 160 bytes: 4 chunks of 16 rows
        u.put("a/big", Tensor::vec(&big));
        u.put("a/small", Tensor::vec(&[7u8, 8, 9]));
        u.put("a/wide", Tensor::rows(2, &[1u16, 2, 3, 4]));
        u.put("a/f", Tensor::vec(&[1.5f32]));
        let uw = w.write_unit(&u).unwrap();
        let mut v = Unit::default();
        v.put("b/x", Tensor::vec(&[1u64, 2]));
        let vw = w.write_unit(&v).unwrap();
        assert!(uw.shards.iter().all(|s| !vw.shards.contains(s)), "units never share a shard");
        w.finish().unwrap();
        std::fs::write(dir.join(crate::MANIFEST_FILE), "{\"format\":\"blobmap-model\",\"format_version\":\"3.0.0\",\"unknown\":1}")
            .unwrap();
        let m = Model::open(&dir).unwrap();
        assert_eq!(&*m.get::<u32>("a/big").unwrap(), &big[..]);
        assert_eq!(&*m.get::<u8>("a/small").unwrap(), &[7, 8, 9]);
        let (wide, n) = m.get_rows::<u16>("a/wide", &[2]).unwrap();
        assert_eq!((n, &*wide), (2, &[1u16, 2, 3, 4][..]));
        assert_eq!(m.get::<u64>("b/x").unwrap()[1], 2);
        assert!(m.get::<u32>("a/small").is_err());
        assert_eq!(m.tensor_hash("a/big").unwrap(), uw.hashes["a/big"]);
        m.verify_shard_hashes().unwrap();
        // Every shard is a valid safetensors file, with each tensor aligned.
        for s in &m.shards {
            let bytes = std::fs::read(dir.join(&s.file)).unwrap();
            let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
            for (_, t) in st.tensors() {
                let off = t.data().as_ptr() as usize - bytes.as_ptr() as usize;
                assert_eq!(off % (t.dtype().bitsize() / 8), 0);
            }
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn same_input_same_bytes() {
        let write = |name: &str| {
            let dir = tmp(name);
            let mut w = ModelWriter::create(&dir, ShardLimits { target: 32, max: 64 }).unwrap();
            let mut u = Unit::default();
            for i in 0..5u32 {
                u.put(format!("t/{i}"), Tensor::vec(&[i; 5]));
            }
            w.write_unit(&u).unwrap();
            let names = w.finish().unwrap().shard_files;
            let mut all = Vec::new();
            for f in names.iter().chain(std::iter::once(&INDEX_FILE.to_string())) {
                all.push(std::fs::read(dir.join(f)).unwrap());
            }
            std::fs::remove_dir_all(&dir).unwrap();
            all
        };
        assert_eq!(write("d1"), write("d2"));
    }
}
