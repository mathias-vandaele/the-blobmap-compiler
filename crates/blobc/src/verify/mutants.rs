//! Deliberately broken models and runtimes (SPEC §11): each must fail some check.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use blobmap_format::INF;
use blobmap_format::runtime::Faults;
use serde_json::Value;
use sha2::{Digest, Sha256};

/// A model mutant: a tensor and how to break one of its elements.
pub struct Mutant {
    pub name: &'static str,
    pub tensor: &'static str,
    pub size: usize,
    pub edit: fn(&mut [u8]) -> bool,
}

fn u16s(b: &mut [u8], f: impl Fn(u16) -> Option<u16>) -> bool {
    for k in 0..b.len() / 2 {
        let x = u16::from_le_bytes([b[2 * k], b[2 * k + 1]]);
        if let Some(y) = f(x) {
            b[2 * k..2 * k + 2].copy_from_slice(&y.to_le_bytes());
            return true;
        }
    }
    false
}

fn u32s(b: &mut [u8], skip: usize, f: impl Fn(u32) -> Option<u32>) -> bool {
    for k in skip..b.len() / 4 {
        let x = u32::from_le_bytes(b[4 * k..4 * k + 4].try_into().unwrap());
        if let Some(y) = f(x) {
            b[4 * k..4 * k + 4].copy_from_slice(&y.to_le_bytes());
            return true;
        }
    }
    false
}

fn lower(x: u16) -> Option<u16> {
    (x != INF && x > 0).then(|| x - 1)
}

pub const MODEL_MUTANTS: &[Mutant] = &[
    Mutant { name: "car link weight lowered", tensor: "road/car/p0/car.up", size: 2, edit: |b| u16s(b, lower) },
    Mutant { name: "walk link weight lowered", tensor: "road/active/p0/walk.down", size: 2, edit: |b| u16s(b, lower) },
    Mutant { name: "turn cost raised", tensor: "road/car/p0/car.turn_cost", size: 2, edit: |b| u16s(b, |x| Some(x + 7)) },
    Mutant { name: "walk road weight raised", tensor: "road/active/p0/walk.w", size: 2, edit: |b| u16s(b, |x| (x != INF).then(|| x + 1)) },
    Mutant { name: "walk offset raised", tensor: "vocab/active/walk.off", size: 2, edit: |b| u16s(b, |x| (x != INF && x > 0).then(|| x + 1)) },
    Mutant { name: "car segment of an ID", tensor: "vocab/car/car.seg", size: 4, edit: |b| u32s(b, 0, |x| (x != u32::MAX).then_some(u32::MAX)) },
    Mutant { name: "car component of a node", tensor: "road/car/nodes/component", size: 4, edit: |b| {
        let Some(k) = (b.len() / 4).checked_sub(1) else { return false };
        let x = u32::from_le_bytes(b[4 * k..4 * k + 4].try_into().unwrap());
        b[4 * k..4 * k + 4].copy_from_slice(&(x + 1).to_le_bytes());
        true
    } },
    Mutant { name: "address IDs out of order", tensor: "vocab/ids/address", size: 16, edit: |b| {
        if b.len() < 32 {
            return false;
        }
        let (x, y) = b.split_at_mut(16);
        x.swap_with_slice(&mut y[..16]);
        true
    } },
    Mutant { name: "copy list", tensor: "road/car/p0/copies", size: 4, edit: |b| u32s(b, 0, |x| Some(x ^ 1)) },
    Mutant { name: "level bound", tensor: "road/active/p0/level_first", size: 4, edit: |b| u32s(b, 1, |x| Some(x + 1)) },
    Mutant { name: "upper neighbour", tensor: "road/active/p0/up_head", size: 4, edit: |b| u32s(b, 0, |x| Some(x + 1)) },
];

pub const RUNTIME_FAULTS: &[(&str, Faults)] = &[
    ("runtime without copies", Faults { no_copies: true, no_direct: false, wrapping: false, largest_id: false, no_start: false }),
    ("runtime without the direct term", Faults { no_copies: false, no_direct: true, wrapping: false, largest_id: false, no_start: false }),
    ("runtime with wrapping additions", Faults { no_copies: false, no_direct: false, wrapping: true, largest_id: false, no_start: false }),
    ("runtime with largest-ID ties", Faults { no_copies: false, no_direct: false, wrapping: false, largest_id: true, no_start: false }),
    ("runtime ignoring start times", Faults { no_copies: false, no_direct: false, wrapping: false, largest_id: false, no_start: true }),
];

fn copy_dir(from: &Path, to: &Path) -> Result<()> {
    if to.exists() {
        std::fs::remove_dir_all(to)?;
    }
    std::fs::create_dir_all(to)?;
    for e in std::fs::read_dir(from)? {
        let e = e?;
        std::fs::copy(e.path(), to.join(e.file_name()))?;
    }
    Ok(())
}

/// Copies the model to `work` and breaks one element of a tensor, keeping the shard
/// transport hashes valid so that only the semantic checks can catch it. None when the
/// tensor is absent or has nothing to break.
pub fn apply(model: &Path, work: &Path, m: &Mutant) -> Result<Option<PathBuf>> {
    let index: Value = blobmap_format::json::read_file(&model.join(blobmap_format::INDEX_FILE))?;
    let Some(file) = index["weight_map"][m.tensor].as_str().map(str::to_string) else { return Ok(None) };
    let dir = work.join(m.name.replace(' ', "-"));
    copy_dir(model, &dir)?;
    let path = dir.join(&file);
    let mut bytes = std::fs::read(&path)?;
    let hl = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: Value = serde_json::from_slice(&bytes[8..8 + hl])?;
    let off = &header[m.tensor]["data_offsets"];
    let (a, b) = (off[0].as_u64().context("offsets")? as usize + 8 + hl, off[1].as_u64().context("offsets")? as usize + 8 + hl);
    ensure!((b - a) % m.size == 0, "{}: element size", m.tensor);
    if !(m.edit)(&mut bytes[a..b]) {
        return Ok(None);
    }
    std::fs::write(&path, &bytes)?;
    let mut index = index;
    for s in index["shards"].as_array_mut().context("shards")? {
        if s["file"] == file.as_str() {
            s["sha256"] = Value::String(hex::encode(Sha256::digest(&bytes)));
        }
    }
    blobmap_format::json::write_file(&dir.join(blobmap_format::INDEX_FILE), &index)?;
    Ok(Some(dir))
}
