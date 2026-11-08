//! A scenario builder: nodes in a local metric frame, ways, restrictions, and Overture rows,
//! with deterministic OSM ids and UUIDs.

use crate::overture::{Address, Place, Release};
use crate::pbf::{Member, MemberType, Node, OsmData, Relation, Way};

/// A local frame: x metres east and y metres north of (lat0, lon0), converted with the
/// cheap-ruler factors at lat0 (the compiler measures lengths the same way).
#[derive(Clone, Copy, Debug)]
pub struct Frame {
    pub lat0: f64,
    pub lon0: f64,
    kx: f64,
    ky: f64,
}

impl Frame {
    pub fn new(lat0: f64, lon0: f64) -> Frame {
        const RE: f64 = 6_378_137.0;
        const FE: f64 = 1.0 / 298.257223563;
        const E2: f64 = FE * (2.0 - FE);
        let m = RE * std::f64::consts::PI / 180.0;
        let cl = lat0.to_radians().cos();
        let w2 = 1.0 / (1.0 - E2 * (1.0 - cl * cl));
        let w = w2.sqrt();
        Frame { lat0, lon0, kx: m * w * cl, ky: m * w * w2 * (1.0 - E2) }
    }

    /// (lat, lon) of the point (x, y), on the PBF's grid of 1e-7 degrees, so that an
    /// Overture point placed on a node is exactly on it.
    pub fn ll(&self, x: f64, y: f64) -> (f64, f64) {
        let q = |d: f64| (d * 1e7).round() / 1e7;
        (q(self.lat0 + y / self.ky), q(self.lon0 + x / self.kx))
    }
}

pub type Tags = Vec<(String, String)>;

pub fn tags(t: &[(&str, &str)]) -> Tags {
    t.iter().map(|&(k, v)| (k.to_string(), v.to_string())).collect()
}

/// The restriction's via: a node, or a chain of ways.
pub enum Via {
    Node(i64),
    Ways(Vec<i64>),
}

pub struct Builder {
    pub frame: Frame,
    pub osm: OsmData,
    pub release: Release,
    next_node: i64,
    next_way: i64,
    next_rel: i64,
    next_uuid: u64,
    uuid_prefix: u32,
}

impl Builder {
    /// `uuid_prefix` keeps the scenarios' IDs apart.
    pub fn new(frame: Frame, uuid_prefix: u32) -> Builder {
        Builder {
            frame,
            osm: OsmData { timestamp: Some(1_793_836_800), ..OsmData::default() },
            release: Release::default(),
            next_node: 1,
            next_way: 1,
            next_rel: 1,
            next_uuid: 1,
            uuid_prefix,
        }
    }

    pub fn node(&mut self, x: f64, y: f64) -> i64 {
        self.node_tags(x, y, &[])
    }

    pub fn node_tags(&mut self, x: f64, y: f64, t: &[(&str, &str)]) -> i64 {
        let (lat, lon) = self.frame.ll(x, y);
        let id = self.next_node;
        self.next_node += 1;
        self.osm.nodes.push(Node { id, lat, lon, tags: tags(t) });
        id
    }

    /// Adds tags to an existing node.
    pub fn tag_node(&mut self, id: i64, t: &[(&str, &str)]) {
        let n = self.osm.nodes.iter_mut().find(|n| n.id == id).expect("node");
        n.tags.extend(tags(t));
    }

    pub fn way(&mut self, refs: &[i64], t: &[(&str, &str)]) -> i64 {
        let id = self.next_way;
        self.next_way += 1;
        self.osm.ways.push(Way { id, refs: refs.to_vec(), tags: tags(t) });
        id
    }

    pub fn restriction(&mut self, kind: &str, from: i64, via: Via, to: i64) -> i64 {
        let id = self.next_rel;
        self.next_rel += 1;
        let mut members = vec![Member { kind: MemberType::Way, id: from, role: "from".into() }];
        match via {
            Via::Node(n) => members.push(Member { kind: MemberType::Node, id: n, role: "via".into() }),
            Via::Ways(ws) => members.extend(ws.into_iter().map(|w| Member { kind: MemberType::Way, id: w, role: "via".into() })),
        }
        members.push(Member { kind: MemberType::Way, id: to, role: "to".into() });
        self.osm.relations.push(Relation { id, members, tags: tags(&[("type", "restriction"), ("restriction", kind)]) });
        id
    }

    pub fn uuid(&mut self) -> String {
        let n = self.next_uuid;
        self.next_uuid += 1;
        format!("{:08x}-0000-4000-8000-{n:012x}", self.uuid_prefix)
    }

    pub fn address(&mut self, x: f64, y: f64) -> String {
        let id = self.uuid();
        let (lat, lon) = self.frame.ll(x, y);
        self.release.addresses.push(Address {
            id: id.clone(),
            lat,
            lon,
            country: "FR".into(),
            dataset: "fixture".into(),
            license: "ODbL-1.0".into(),
        });
        id
    }

    /// A kept-looking place: confidence 0.9, open, with this primary category and basic category.
    pub fn place(&mut self, x: f64, y: f64, primary: Option<&str>, basic: Option<&str>, name: &str) -> String {
        let id = self.uuid();
        let (lat, lon) = self.frame.ll(x, y);
        self.release.places.push(Place {
            id: id.clone(),
            lat,
            lon,
            name: Some(name.to_string()),
            confidence: Some(0.9),
            operating_status: Some("open".into()),
            basic_category: basic.map(str::to_string),
            primary: primary.map(str::to_string),
            alternates: Vec::new(),
            country: Some("FR".into()),
            dataset: "fixture".into(),
            license: "CDLA-Permissive-2.0".into(),
        });
        id
    }

    /// The last place added, to adjust it.
    pub fn last_place(&mut self) -> &mut Place {
        self.release.places.last_mut().unwrap()
    }
}
