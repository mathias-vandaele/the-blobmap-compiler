//! Car turn restrictions (ported from the old `edge/edge.rs`): OSM `type=restriction`
//! relations with a `no_*` or `only_*` value for car (`restriction`, or `restriction:` with
//! motorcar, motor_vehicle or vehicle), `except` honoured. Via a node, or via one or more
//! ways; the old compiler skipped the latter, this one keeps them (SPEC §4).

use std::path::Path;
use std::sync::Mutex;

use anyhow::Result;
use osmpbf::{BlobDecode, BlobReader, RelMemberType};
use rayon::prelude::*;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Via {
    Node(i64),
    /// The via ways as the relation lists them; their order along the route is found from
    /// the graph.
    Ways(Vec<i64>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Restriction {
    pub id: i64,
    pub only: bool,
    pub from: Vec<i64>,
    pub via: Via,
    pub to: Vec<i64>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RestrictionStats {
    pub relations: usize,
    pub node_via: usize,
    pub way_via: usize,
    pub except_car: usize,
    pub not_no_or_only: usize,
    pub malformed: usize,
}

impl RestrictionStats {
    fn add(&mut self, o: &RestrictionStats) {
        self.relations += o.relations;
        self.node_via += o.node_via;
        self.way_via += o.way_via;
        self.except_car += o.except_car;
        self.not_no_or_only += o.not_no_or_only;
        self.malformed += o.malformed;
    }
}

const MODES: [&str; 3] = ["motorcar", "motor_vehicle", "vehicle"];

/// The car restrictions of the file, by relation id.
pub fn read(path: &Path) -> Result<(Vec<Restriction>, RestrictionStats)> {
    let out = Mutex::new((Vec::new(), RestrictionStats::default()));
    BlobReader::from_path(path)?.par_bridge().try_for_each(|blob| -> Result<()> {
        if let BlobDecode::OsmData(block) = blob?.decode()? {
            let (mut local, mut st) = (Vec::new(), RestrictionStats::default());
            for group in block.groups() {
                for r in group.relations() {
                    let (mut is_restriction, mut value, mut except) = (false, None, None);
                    for (k, v) in r.tags() {
                        match k {
                            "type" => is_restriction = v == "restriction",
                            "restriction" => value = value.or(Some(v)),
                            "except" => except = Some(v),
                            _ => {
                                if let Some(m) = k.strip_prefix("restriction:")
                                    && MODES.contains(&m)
                                {
                                    value = Some(v);
                                }
                            }
                        }
                    }
                    if !is_restriction {
                        continue;
                    }
                    st.relations += 1;
                    if except.is_some_and(|e| e.split(';').any(|x| MODES.contains(&x.trim()))) {
                        st.except_car += 1;
                        continue;
                    }
                    let only = match value {
                        Some(v) if v.starts_with("only_") => true,
                        Some(v) if v.starts_with("no_") => false,
                        _ => {
                            st.not_no_or_only += 1;
                            continue;
                        }
                    };
                    let (mut from, mut to, mut via_node, mut via_ways) = (Vec::new(), Vec::new(), None, Vec::new());
                    let mut ok = true;
                    for m in r.members() {
                        match (m.role().unwrap_or(""), m.member_type) {
                            ("from", RelMemberType::Way) => from.push(m.member_id),
                            ("to", RelMemberType::Way) => to.push(m.member_id),
                            ("via", RelMemberType::Node) => ok &= via_node.replace(m.member_id).is_none(),
                            ("via", RelMemberType::Way) => via_ways.push(m.member_id),
                            _ => {}
                        }
                    }
                    ok &= !from.is_empty() && !to.is_empty() && (via_node.is_some() != !via_ways.is_empty());
                    ok &= !(only && to.len() > 1);
                    if !ok {
                        st.malformed += 1;
                        continue;
                    }
                    let via = match via_node {
                        Some(n) => {
                            st.node_via += 1;
                            Via::Node(n)
                        }
                        None => {
                            st.way_via += 1;
                            Via::Ways(via_ways)
                        }
                    };
                    local.push(Restriction { id: r.id(), only, from, via, to });
                }
            }
            let mut g = out.lock().unwrap();
            g.0.extend(local);
            g.1.add(&st);
        }
        Ok(())
    })?;
    let (mut v, st) = out.into_inner().unwrap();
    v.sort_by_key(|r| r.id);
    Ok((v, st))
}

#[cfg(test)]
mod tests {
    #[test]
    fn viaway_restrictions_are_kept() {
        let dir = std::env::temp_dir().join(format!("blobmap-osm-r-{}", std::process::id()));
        let w = blobmap_fixtures::by_name("viaway").unwrap().write(&dir).unwrap();
        let (r, st) = super::read(&w.pbf).unwrap();
        assert_eq!((st.relations, st.way_via, st.node_via, st.malformed), (4, 4, 0, 0));
        assert_eq!(r.iter().filter(|x| x.only).count(), 2);
        assert!(matches!(&r[2].via, super::Via::Ways(v) if v.len() == 2));
    }
}
