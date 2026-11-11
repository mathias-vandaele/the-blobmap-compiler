//! Cuts a polygon out of a `.osm.pbf` of any size, up to the planet (PLAN, milestone 5):
//! `blobc cut <in.osm.pbf> <out.osm.pbf> <poly>`.
//!
//! Complete ways, as osmium's `complete_ways` strategy: every way with a node inside the
//! polygon, with all its nodes; every relation with a node or way member kept, all its
//! members listed. Three passes over the input, its blobs decoded in parallel and written back
//! in file order, with bitmaps over the node and way ids, so memory does not grow with the
//! output. The header's replication timestamp is kept.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result, bail, ensure};
use osmpbf::{Blob, BlobDecode, BlobReader, PrimitiveBlock, RelMemberType};
use rayon::prelude::*;

use crate::fixtures::pbf::{Member, MemberType, Node, Relation, Way, data_block, header_block};
use crate::overture::Polygon;

/// Blobs decoded at once, in file order.
const BATCH: usize = 1024;

/// Ids up to this bound (the planet's nodes stay well below 2³⁴).
const MAX_NODE_ID: u64 = 1 << 34;
const MAX_WAY_ID: u64 = 1 << 32;

struct Bits(Vec<AtomicU64>);

impl Bits {
    fn new(n: u64) -> Bits {
        Bits((0..n.div_ceil(64)).map(|_| AtomicU64::new(0)).collect())
    }

    fn set(&self, id: i64) -> Result<()> {
        ensure!(id >= 0 && (id as u64) < self.0.len() as u64 * 64, "id {id} beyond the bitmap");
        self.0[id as usize / 64].fetch_or(1 << (id % 64), Ordering::Relaxed);
        Ok(())
    }

    fn get(&self, id: i64) -> bool {
        id >= 0 && (id as u64) < self.0.len() as u64 * 64 && self.0[id as usize / 64].load(Ordering::Relaxed) >> (id % 64) & 1 == 1
    }

    fn count(&self) -> u64 {
        self.0.iter().map(|w| w.load(Ordering::Relaxed).count_ones() as u64).sum()
    }
}

fn tags<'a>(t: impl Iterator<Item = (&'a str, &'a str)>) -> Vec<(String, String)> {
    t.map(|(k, v)| (k.to_string(), v.to_string())).collect()
}

/// Runs `f` on every data block of the file, a batch of blobs at a time in parallel, and hands
/// the results to `sink` in file order.
fn each_block<T: Send>(path: &Path, f: impl Fn(&PrimitiveBlock) -> Result<T> + Sync, mut sink: impl FnMut(T) -> Result<()>) -> Result<()> {
    let mut blobs = BlobReader::from_path(path).with_context(|| format!("opening {}", path.display()))?;
    loop {
        let batch: Vec<Blob> = blobs.by_ref().take(BATCH).collect::<std::result::Result<_, _>>()?;
        if batch.is_empty() {
            return Ok(());
        }
        let out: Vec<Option<T>> = batch
            .into_par_iter()
            .map(|blob| match blob.decode()? {
                BlobDecode::OsmData(b) => f(&b).map(Some),
                _ => Ok(None),
            })
            .collect::<Result<_>>()?;
        for t in out.into_iter().flatten() {
            sink(t)?;
        }
    }
}

pub fn main(a: Vec<String>) -> Result<()> {
    let [input, output, poly] = &a[..] else {
        bail!("usage: blobc cut <in.osm.pbf> <out.osm.pbf> <poly>");
    };
    let input = Path::new(input);
    let poly = Polygon::parse_poly(&std::fs::read_to_string(poly).with_context(|| format!("reading {poly}"))?)?;
    let (mut lo, mut hi) = ((f64::MAX, f64::MAX), (f64::MIN, f64::MIN));
    for &(la, lon) in poly.rings.iter().flatten() {
        lo = (lo.0.min(la), lo.1.min(lon));
        hi = (hi.0.max(la), hi.1.max(lon));
    }
    let in_box = |la: f64, lon: f64| la >= lo.0 && la <= hi.0 && ((lon >= lo.1 && lon <= hi.1) || hi.1 > 180.0 || lo.1 < -180.0);
    let inside = |la: f64, lon: f64| in_box(la, lon) && poly.contains(la, lon);
    // Pass 1: the nodes inside.
    let (inner, needed, kept_ways) = (Bits::new(MAX_NODE_ID), Bits::new(MAX_NODE_ID), Bits::new(MAX_WAY_ID));
    each_block(
        input,
        |b| {
            for g in b.groups() {
                for x in g.dense_nodes() {
                    if inside(x.lat(), x.lon()) {
                        inner.set(x.id())?;
                    }
                }
                for x in g.nodes() {
                    if inside(x.lat(), x.lon()) {
                        inner.set(x.id())?;
                    }
                }
            }
            Ok(())
        },
        |()| Ok(()),
    )?;
    eprintln!("pass 1: {} nodes inside", inner.count());
    // Pass 2: the ways with a node inside, and all their nodes.
    each_block(
        input,
        |b| {
            for g in b.groups() {
                for w in g.ways() {
                    if w.refs().any(|r| inner.get(r)) {
                        kept_ways.set(w.id())?;
                        for r in w.refs() {
                            needed.set(r)?;
                        }
                    }
                }
            }
            Ok(())
        },
        |()| Ok(()),
    )?;
    for (n, i) in needed.0.iter().zip(&inner.0) {
        n.fetch_or(i.load(Ordering::Relaxed), Ordering::Relaxed);
    }
    drop(inner);
    eprintln!("pass 2: {} ways, {} nodes", kept_ways.count(), needed.count());
    // Pass 3: write what is kept, block by block.
    let timestamp = BlobReader::from_path(input)?.next().context("empty file")??.to_headerblock()?.osmosis_replication_timestamp();
    let mut out = BufWriter::with_capacity(1 << 24, File::create(output).with_context(|| format!("creating {output}"))?);
    let mut head = Vec::new();
    header_block(Some((lo, hi)), timestamp, &mut head)?;
    out.write_all(&head)?;
    let (mut nodes, mut ways, mut relations) = (0u64, 0u64, 0u64);
    each_block(
        input,
        |b| {
            let (mut ns, mut ws, mut rs) = (Vec::new(), Vec::new(), Vec::new());
            for g in b.groups() {
                for x in g.dense_nodes() {
                    if needed.get(x.id()) {
                        ns.push(Node { id: x.id(), lat: x.lat(), lon: x.lon(), tags: tags(x.tags()) });
                    }
                }
                for x in g.nodes() {
                    if needed.get(x.id()) {
                        ns.push(Node { id: x.id(), lat: x.lat(), lon: x.lon(), tags: tags(x.tags()) });
                    }
                }
                for w in g.ways() {
                    if kept_ways.get(w.id()) {
                        ws.push(Way { id: w.id(), refs: w.refs().collect(), tags: tags(w.tags()) });
                    }
                }
                for r in g.relations() {
                    let kept = r.members().any(|m| match m.member_type {
                        RelMemberType::Node => needed.get(m.member_id),
                        RelMemberType::Way => kept_ways.get(m.member_id),
                        RelMemberType::Relation => false,
                    });
                    if kept {
                        let members = r
                            .members()
                            .map(|m| Member {
                                kind: match m.member_type {
                                    RelMemberType::Node => MemberType::Node,
                                    RelMemberType::Way => MemberType::Way,
                                    RelMemberType::Relation => MemberType::Relation,
                                },
                                id: m.member_id,
                                role: m.role().unwrap_or("").to_string(),
                            })
                            .collect();
                        rs.push(Relation { id: r.id(), members, tags: tags(r.tags()) });
                    }
                }
            }
            let counts = (ns.len() as u64, ws.len() as u64, rs.len() as u64);
            let mut bytes = Vec::new();
            if counts != (0, 0, 0) {
                data_block(&ns, &ws, &rs, &mut bytes)?;
            }
            Ok((counts, bytes))
        },
        |((n, w, r), bytes)| {
            (nodes, ways, relations) = (nodes + n, ways + w, relations + r);
            out.write_all(&bytes)?;
            Ok(())
        },
    )?;
    out.flush()?;
    eprintln!("{nodes} nodes, {ways} ways, {relations} relations");
    Ok(())
}
