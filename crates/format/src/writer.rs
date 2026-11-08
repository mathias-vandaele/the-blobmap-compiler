//! The deterministic shard writer (FORMAT §2): units fill shards in order, a tensor larger
//! than the shard target is chunked along dimension 0, and inside a shard the data is
//! ordered by element size so that every tensor is aligned to it.
//!
//! Units are written one at a time and dropped by the caller afterwards, so the writer
//! holds only one unit in memory. Shards get temporary names until `finish`, which knows
//! their count and renames them `model-0000k-of-0000n.safetensors`.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::hash::{Hash, tensor_hash};
use crate::tensor::{Tensor, Unit};
use crate::{FORMAT_VERSION, INDEX_FILE, json as cjson};

#[derive(Clone, Copy, Debug)]
pub struct ShardLimits {
    /// `build.shard_target_bytes`.
    pub target: u64,
    /// `build.shard_max_bytes`.
    pub max: u64,
}

impl Default for ShardLimits {
    fn default() -> Self {
        ShardLimits { target: 2 << 30, max: 4 << 30 }
    }
}

struct ShardOut {
    temp: PathBuf,
    bytes: u64,
    sha256: String,
}

/// One unit, written: its shards (indices into the model's shard list) and the hash of each
/// of its logical tensors, by full name.
#[derive(Debug, Clone)]
pub struct UnitWritten {
    pub shards: Vec<usize>,
    pub hashes: BTreeMap<String, Hash>,
}

pub struct ModelWriter {
    dir: PathBuf,
    limits: ShardLimits,
    shards: Vec<ShardOut>,
    /// Physical tensor -> shard index.
    weight_map: BTreeMap<String, usize>,
    /// Logical tensor -> its chunks, when chunked.
    chunks: BTreeMap<String, Vec<String>>,
    total_size: u64,
}

/// The model directory's shard files after `finish`.
pub struct Written {
    pub shard_files: Vec<String>,
}

/// A piece of a logical tensor that goes into a shard as one physical tensor.
struct Piece<'a> {
    name: String,
    t: &'a Tensor,
    rows: std::ops::Range<u64>,
}

impl Piece<'_> {
    fn data(&self) -> &[u8] {
        let rb = self.t.row_bytes();
        &self.t.data[(self.rows.start * rb) as usize..(self.rows.end * rb) as usize]
    }

    fn shape(&self) -> Vec<u64> {
        let mut s = self.t.shape.clone();
        s[0] = self.rows.end - self.rows.start;
        s
    }
}

impl ModelWriter {
    pub fn create(dir: &Path, limits: ShardLimits) -> Result<ModelWriter> {
        ensure!(limits.target > 0 && limits.target <= limits.max, "shard target must be in 1..=max");
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        Ok(ModelWriter {
            dir: dir.to_path_buf(),
            limits,
            shards: Vec::new(),
            weight_map: BTreeMap::new(),
            chunks: BTreeMap::new(),
            total_size: 0,
        })
    }

    /// Writes one unit: its tensors in full-name order fill new shards (FORMAT §2.4).
    pub fn write_unit(&mut self, unit: &Unit) -> Result<UnitWritten> {
        let mut hashes = BTreeMap::new();
        let mut pieces: Vec<Piece> = Vec::new();
        for (name, t) in &unit.tensors {
            t.check().with_context(|| format!("tensor {name}"))?;
            ensure!(!self.weight_map.contains_key(name) && !self.chunks.contains_key(name), "tensor {name} written twice");
            hashes.insert(name.clone(), tensor_hash(name, t.dtype, &t.shape, [&t.data[..]]));
            let rows = t.rows_count();
            if t.data.len() as u64 <= self.limits.target {
                pieces.push(Piece { name: name.clone(), t, rows: 0..rows });
                continue;
            }
            let rb = t.row_bytes();
            if rb > self.limits.target {
                bail!("tensor {name}: one row ({rb} bytes) exceeds the shard target");
            }
            let per = self.limits.target / rb;
            let mut names = Vec::new();
            let mut r = 0;
            while r < rows {
                let n = format!("{name}@{}", names.len());
                names.push(n.clone());
                pieces.push(Piece { name: n, t, rows: r..(r + per).min(rows) });
                r += per;
            }
            self.chunks.insert(name.clone(), names);
        }
        let mut shards = Vec::new();
        let mut cur: Vec<Piece> = Vec::new();
        let mut cur_bytes = 0u64;
        for p in pieces {
            let b = p.data().len() as u64;
            if !cur.is_empty() && cur_bytes + b > self.limits.target {
                shards.push(self.flush(std::mem::take(&mut cur))?);
                cur_bytes = 0;
            }
            cur_bytes += b;
            cur.push(p);
        }
        if !cur.is_empty() {
            shards.push(self.flush(cur)?);
        }
        Ok(UnitWritten { shards, hashes })
    }

    /// Writes one shard with the given pieces.
    fn flush(&mut self, mut pieces: Vec<Piece>) -> Result<usize> {
        // Data order: element size descending, then full name (FORMAT §2.1).
        pieces.sort_by(|a, b| b.t.dtype.size().cmp(&a.t.dtype.size()).then_with(|| a.name.cmp(&b.name)));
        let k = self.shards.len();
        let mut header = String::from("{\"__metadata__\":");
        header.push_str(&json!({"format": "blobmap-shard", "format_version": FORMAT_VERSION}).to_string());
        let mut off = 0u64;
        for p in &pieces {
            let len = p.data().len() as u64;
            header.push(',');
            header.push_str(&Value::String(p.name.clone()).to_string());
            header.push_str(&format!(
                ":{{\"dtype\":\"{}\",\"shape\":{},\"data_offsets\":[{off},{}]}}",
                p.t.dtype.name(),
                json!(p.shape()),
                off + len
            ));
            off += len;
        }
        header.push('}');
        while !header.len().is_multiple_of(8) {
            header.push(' ');
        }
        let total = 8 + header.len() as u64 + off;
        ensure!(off <= self.limits.max, "shard {k}: {off} bytes of data exceed the shard maximum");
        let temp = self.dir.join(format!(".shard-{k:05}.tmp"));
        let mut sha = Sha256::new();
        let mut f = BufWriter::new(File::create(&temp).with_context(|| format!("creating {}", temp.display()))?);
        let hl = (header.len() as u64).to_le_bytes();
        for chunk in [&hl[..], header.as_bytes()] {
            f.write_all(chunk)?;
            sha.update(chunk);
        }
        for p in &pieces {
            f.write_all(p.data())?;
            sha.update(p.data());
            self.weight_map.insert(p.name.clone(), k);
        }
        f.flush()?;
        self.total_size += off;
        self.shards.push(ShardOut { temp, bytes: total, sha256: hex::encode(sha.finalize()) });
        Ok(k)
    }

    /// Renames the shards to their final names and writes the index.
    pub fn finish(self) -> Result<Written> {
        let n = self.shards.len();
        let names: Vec<String> = (0..n).map(|k| format!("model-{:05}-of-{n:05}.safetensors", k + 1)).collect();
        for (s, name) in self.shards.iter().zip(&names) {
            std::fs::rename(&s.temp, self.dir.join(name)).with_context(|| format!("renaming {}", s.temp.display()))?;
        }
        let index = json!({
            "chunks": self.chunks,
            "metadata": {"format": "blobmap-index", "format_version": FORMAT_VERSION, "total_size": self.total_size},
            "shards": self.shards.iter().zip(&names).map(|(s, f)| json!({"bytes": s.bytes, "file": f, "sha256": s.sha256})).collect::<Vec<_>>(),
            "weight_map": self.weight_map.iter().map(|(t, &k)| (t.clone(), Value::String(names[k].clone()))).collect::<serde_json::Map<_, _>>(),
        });
        cjson::write_file(&self.dir.join(INDEX_FILE), &index)?;
        Ok(Written { shard_files: names })
    }
}
