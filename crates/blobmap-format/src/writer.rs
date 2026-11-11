//! The deterministic shard writer (FORMAT §2): units fill shards in order, a tensor larger
//! than the shard target is chunked along dimension 0, and inside a shard the data is
//! ordered by element size so that every tensor is aligned to it.
//!
//! Units are written one at a time, each whole or one tensor at a time, so the writer holds
//! at most one shard's tensors. A unit may be written before one that precedes it in the
//! model: shards get temporary names until `finish`, which puts them in unit order, knows
//! their count and renames them `model-0000k-of-0000n.safetensors`.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

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
    /// The slot of its unit (`set_slot`).
    slot: u32,
    temp: PathBuf,
    bytes: u64,
    sha256: String,
}

/// One unit, written: its shards (in write order) and the hash of each of its logical
/// tensors, by full name.
#[derive(Debug, Clone, Default)]
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
    slot: u32,
    open: Option<Open>,
}

/// The model directory's shard files after `finish`.
pub struct Written {
    pub shard_files: Vec<String>,
}

/// A piece of a logical tensor that goes into a shard as one physical tensor.
struct Piece {
    name: String,
    t: Arc<Tensor>,
    rows: std::ops::Range<u64>,
}

impl Piece {
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

/// The unit being written: its open shard's pieces, its last tensor, and what it wrote.
#[derive(Default)]
struct Open {
    cur: Vec<Piece>,
    cur_bytes: u64,
    last: Option<String>,
    written: UnitWritten,
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
            slot: 0,
            open: None,
        })
    }

    /// Units written from here on take this slot: `finish` orders the shards by slot, and in
    /// write order within one, so that units may be written out of their order (FORMAT §2.4)
    /// while a shard never holds two units.
    pub fn set_slot(&mut self, slot: u32) -> Result<()> {
        ensure!(self.open.is_none(), "a unit is open");
        self.slot = slot;
        Ok(())
    }

    /// Writes one unit: its tensors in full-name order fill new shards (FORMAT §2.4).
    pub fn write_unit(&mut self, unit: Unit) -> Result<UnitWritten> {
        self.begin_unit()?;
        for (name, t) in unit.tensors {
            self.put(name, t)?;
        }
        self.end_unit()
    }

    /// Opens a unit whose tensors then come one at a time with `put`, in full-name order.
    pub fn begin_unit(&mut self) -> Result<()> {
        ensure!(self.open.is_none(), "a unit is already open");
        self.open = Some(Open::default());
        Ok(())
    }

    /// The open unit's next tensor; the shards it fills are written as they fill.
    pub fn put(&mut self, name: impl Into<String>, t: Tensor) -> Result<()> {
        let name = name.into();
        let last = &self.open.as_ref().context("no unit is open")?.last;
        t.check().with_context(|| format!("tensor {name}"))?;
        ensure!(!name.contains('@'), "{name}: '@' is reserved for chunks");
        ensure!(last.as_ref().is_none_or(|l| *l < name), "tensor {name}: a unit's tensors come in full-name order");
        ensure!(!self.weight_map.contains_key(&name) && !self.chunks.contains_key(&name), "tensor {name} written twice");
        let mut o = self.open.take().expect("open");
        o.written.hashes.insert(name.clone(), tensor_hash(&name, t.dtype, &t.shape, [&t.data[..]]));
        let rows = t.rows_count();
        let t = Arc::new(t);
        let mut pieces = Vec::new();
        if t.data.len() as u64 <= self.limits.target {
            pieces.push(Piece { name: name.clone(), t, rows: 0..rows });
        } else {
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
                pieces.push(Piece { name: n, t: t.clone(), rows: r..(r + per).min(rows) });
                r += per;
            }
            self.chunks.insert(name.clone(), names);
        }
        for p in pieces {
            let b = p.data().len() as u64;
            if !o.cur.is_empty() && o.cur_bytes + b > self.limits.target {
                let k = self.flush(std::mem::take(&mut o.cur))?;
                o.written.shards.push(k);
                o.cur_bytes = 0;
            }
            o.cur_bytes += b;
            o.cur.push(p);
        }
        o.last = Some(name);
        self.open = Some(o);
        Ok(())
    }

    /// Closes the open unit: its last shard is written.
    pub fn end_unit(&mut self) -> Result<UnitWritten> {
        let mut o = self.open.take().context("no unit is open")?;
        if !o.cur.is_empty() {
            let k = self.flush(o.cur)?;
            o.written.shards.push(k);
        }
        Ok(o.written)
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
        self.shards.push(ShardOut { slot: self.slot, temp, bytes: total, sha256: hex::encode(sha.finalize()) });
        Ok(k)
    }

    /// Puts the shards in slot order, renames them to their final names and writes the index.
    pub fn finish(self) -> Result<Written> {
        ensure!(self.open.is_none(), "a unit is still open");
        let n = self.shards.len();
        let names: Vec<String> = (0..n).map(|k| format!("model-{:05}-of-{n:05}.safetensors", k + 1)).collect();
        // `order[i]` is the shard written `i`-th in the model; `name_of[k]`, shard k's file.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&k| self.shards[k].slot);
        let mut name_of = vec![String::new(); n];
        for (i, &k) in order.iter().enumerate() {
            name_of[k] = names[i].clone();
            let s = &self.shards[k];
            std::fs::rename(&s.temp, self.dir.join(&names[i])).with_context(|| format!("renaming {}", s.temp.display()))?;
        }
        let shard = |k: usize| json!({"bytes": self.shards[k].bytes, "file": name_of[k], "sha256": self.shards[k].sha256});
        let index = json!({
            "chunks": self.chunks,
            "metadata": {"format": "blobmap-index", "format_version": FORMAT_VERSION, "total_size": self.total_size},
            "shards": order.iter().map(|&k| shard(k)).collect::<Vec<_>>(),
            "weight_map": self.weight_map.iter().map(|(t, &k)| (t.clone(), Value::String(name_of[k].clone()))).collect::<serde_json::Map<_, _>>(),
        });
        cjson::write_file(&self.dir.join(INDEX_FILE), &index)?;
        Ok(Written { shard_files: names })
    }
}
