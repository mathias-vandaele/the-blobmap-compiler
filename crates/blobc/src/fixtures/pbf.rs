//! A minimal `.osm.pbf` writer: protobuf encoded by hand, blobs zlib-compressed. It writes
//! a header block, then data blocks of at most `BLOCK` entities: nodes, then ways, then
//! relations, each block with one group per kind it holds (a fixture fits in one block).
//! Coordinates use the default granularity (100 nanodegrees).

use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result};
use flate2::Compression;
use flate2::write::ZlibEncoder;

#[derive(Clone, Debug)]
pub struct Node {
    pub id: i64,
    pub lat: f64,
    pub lon: f64,
    pub tags: Vec<(String, String)>,
}

#[derive(Clone, Debug)]
pub struct Way {
    pub id: i64,
    pub refs: Vec<i64>,
    pub tags: Vec<(String, String)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemberType {
    Node = 0,
    Way = 1,
    Relation = 2,
}

#[derive(Clone, Debug)]
pub struct Member {
    pub kind: MemberType,
    pub id: i64,
    pub role: String,
}

#[derive(Clone, Debug)]
pub struct Relation {
    pub id: i64,
    pub members: Vec<Member>,
    pub tags: Vec<(String, String)>,
}

#[derive(Clone, Debug, Default)]
pub struct OsmData {
    pub nodes: Vec<Node>,
    pub ways: Vec<Way>,
    pub relations: Vec<Relation>,
    /// `osmosis_replication_timestamp`, seconds since the epoch.
    pub timestamp: Option<i64>,
}

/// Protobuf message builder.
#[derive(Default)]
struct Msg(Vec<u8>);

fn varint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

fn zigzag(v: i64) -> u64 {
    ((v << 1) ^ (v >> 63)) as u64
}

impl Msg {
    fn key(&mut self, field: u32, wire: u32) {
        varint(&mut self.0, ((field << 3) | wire) as u64);
    }
    fn uint(&mut self, field: u32, v: u64) -> &mut Self {
        self.key(field, 0);
        varint(&mut self.0, v);
        self
    }
    fn sint(&mut self, field: u32, v: i64) -> &mut Self {
        self.uint(field, zigzag(v))
    }
    fn bytes(&mut self, field: u32, b: &[u8]) -> &mut Self {
        self.key(field, 2);
        varint(&mut self.0, b.len() as u64);
        self.0.extend_from_slice(b);
        self
    }
    fn packed_uint(&mut self, field: u32, vs: impl IntoIterator<Item = u64>) -> &mut Self {
        let mut p = Vec::new();
        for v in vs {
            varint(&mut p, v);
        }
        self.bytes(field, &p)
    }
    fn packed_sint(&mut self, field: u32, vs: impl IntoIterator<Item = i64>) -> &mut Self {
        self.packed_uint(field, vs.into_iter().map(zigzag))
    }
}

fn delta(vs: impl IntoIterator<Item = i64>) -> Vec<i64> {
    let mut last = 0;
    vs.into_iter()
        .map(|v| {
            let d = v - last;
            last = v;
            d
        })
        .collect()
}

/// The string table: index 0 is the empty string.
#[derive(Default)]
struct Strings {
    list: Vec<String>,
    index: std::collections::BTreeMap<String, u64>,
}

impl Strings {
    fn id(&mut self, s: &str) -> u64 {
        if self.list.is_empty() {
            self.list.push(String::new());
            self.index.insert(String::new(), 0);
        }
        if let Some(&i) = self.index.get(s) {
            return i;
        }
        let i = self.list.len() as u64;
        self.list.push(s.to_string());
        self.index.insert(s.to_string(), i);
        i
    }
}

fn units(deg: f64) -> i64 {
    (deg * 1e7).round() as i64
}

fn blob(kind: &str, payload: &[u8], out: &mut Vec<u8>) -> Result<()> {
    let mut z = ZlibEncoder::new(Vec::new(), Compression::default());
    z.write_all(payload)?;
    let mut b = Msg::default();
    b.uint(2, payload.len() as u64).bytes(3, &z.finish()?);
    let mut h = Msg::default();
    h.bytes(1, kind.as_bytes()).uint(3, b.0.len() as u64);
    out.extend_from_slice(&(h.0.len() as u32).to_be_bytes());
    out.extend_from_slice(&h.0);
    out.extend_from_slice(&b.0);
    Ok(())
}

/// Entities per data block, as osmium writes them.
pub const BLOCK: usize = 8000;

impl OsmData {
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        let mut header = Msg::default();
        if !self.nodes.is_empty() {
            let (mut lo, mut hi) = ((f64::MAX, f64::MAX), (f64::MIN, f64::MIN));
            for n in &self.nodes {
                lo = (lo.0.min(n.lat), lo.1.min(n.lon));
                hi = (hi.0.max(n.lat), hi.1.max(n.lon));
            }
            let mut bbox = Msg::default();
            let nano = |d: f64| (d * 1e9).round() as i64;
            bbox.sint(1, nano(lo.1)).sint(2, nano(hi.1)).sint(3, nano(hi.0)).sint(4, nano(lo.0));
            header.bytes(1, &bbox.0);
        }
        header.bytes(4, b"OsmSchema-V0.6").bytes(4, b"DenseNodes").bytes(16, b"blobmap-fixtures");
        if let Some(t) = self.timestamp {
            header.uint(32, t as u64);
        }
        blob("OSMHeader", &header.0, &mut out)?;
        let (nn, nw, nr) = (self.nodes.len(), self.ways.len(), self.relations.len());
        let mut at = 0;
        while at < nn + nw + nr {
            let end = (at + BLOCK).min(nn + nw + nr);
            let part = |len: usize, start: usize| at.max(start).min(start + len)..end.max(start).min(start + len);
            let (n, w, r) = (part(nn, 0), part(nw, nn), part(nr, nn + nw));
            let shift = |x: std::ops::Range<usize>, by: usize| x.start - by..x.end - by;
            self.block(&self.nodes[n], &self.ways[shift(w, nn)], &self.relations[shift(r, nn + nw)], &mut out)?;
            at = end;
        }
        Ok(out)
    }

    fn block(&self, nodes: &[Node], ways: &[Way], relations: &[Relation], out: &mut Vec<u8>) -> Result<()> {
        let mut st = Strings::default();
        st.id("");
        let mut groups: Vec<Vec<u8>> = Vec::new();
        if !nodes.is_empty() {
            let mut kv = Vec::new();
            for n in nodes {
                for (k, v) in &n.tags {
                    kv.push(st.id(k));
                    kv.push(st.id(v));
                }
                kv.push(0);
            }
            let mut dense = Msg::default();
            dense
                .packed_sint(1, delta(nodes.iter().map(|n| n.id)))
                .packed_sint(8, delta(nodes.iter().map(|n| units(n.lat))))
                .packed_sint(9, delta(nodes.iter().map(|n| units(n.lon))))
                .packed_uint(10, kv);
            let mut g = Msg::default();
            g.bytes(2, &dense.0);
            groups.push(g.0);
        }
        if !ways.is_empty() {
            let mut g = Msg::default();
            for w in ways {
                let mut m = Msg::default();
                m.uint(1, w.id as u64);
                let keys: Vec<u64> = w.tags.iter().map(|(k, _)| st.id(k)).collect();
                let vals: Vec<u64> = w.tags.iter().map(|(_, v)| st.id(v)).collect();
                m.packed_uint(2, keys).packed_uint(3, vals).packed_sint(8, delta(w.refs.iter().copied()));
                g.bytes(3, &m.0);
            }
            groups.push(g.0);
        }
        if !relations.is_empty() {
            let mut g = Msg::default();
            for r in relations {
                let mut m = Msg::default();
                m.uint(1, r.id as u64);
                let keys: Vec<u64> = r.tags.iter().map(|(k, _)| st.id(k)).collect();
                let vals: Vec<u64> = r.tags.iter().map(|(_, v)| st.id(v)).collect();
                let roles: Vec<u64> = r.members.iter().map(|x| st.id(&x.role)).collect();
                m.packed_uint(2, keys)
                    .packed_uint(3, vals)
                    .packed_uint(8, roles)
                    .packed_sint(9, delta(r.members.iter().map(|x| x.id)))
                    .packed_uint(10, r.members.iter().map(|x| x.kind as u64));
                g.bytes(4, &m.0);
            }
            groups.push(g.0);
        }
        let mut table = Msg::default();
        for s in &st.list {
            table.bytes(1, s.as_bytes());
        }
        let mut block = Msg::default();
        block.bytes(1, &table.0);
        for g in &groups {
            block.bytes(2, g);
        }
        blob("OSMData", &block.0, out)
    }

    pub fn write(&self, path: &Path) -> Result<()> {
        std::fs::write(path, self.to_bytes()?).with_context(|| format!("writing {}", path.display()))
    }
}
