//! Cuts a bounding box out of a `.osm.pbf`, for the small tier (PLAN, milestone 3):
//! `extract <in.osm.pbf> <out.osm.pbf> <min_lat,min_lon,max_lat,max_lon> [<out.poly>]`.
//!
//! Complete ways: every way with a node inside the box, with all its nodes; the relations
//! whose way and node members are all kept. The header's replication timestamp is kept. The
//! optional `.poly` is the box, for `--extent`.

use std::collections::HashSet;
use std::path::Path;

use anyhow::{Context, Result, bail};
use blobmap_fixtures::pbf::{Member, MemberType, Node, OsmData, Relation, Way};
use osmpbf::{Element, ElementReader, RelMemberType};

fn tags<'a>(t: impl Iterator<Item = (&'a str, &'a str)>) -> Vec<(String, String)> {
    t.map(|(k, v)| (k.to_string(), v.to_string())).collect()
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() < 3 || a.len() > 4 {
        bail!("usage: extract <in.osm.pbf> <out.osm.pbf> <min_lat,min_lon,max_lat,max_lon> [<out.poly>]");
    }
    let b: Vec<f64> = a[2].split(',').map(str::parse).collect::<Result<_, _>>().context("bbox")?;
    let [s, w, n, e] = b[..] else { bail!("bbox: four numbers") };
    let inside = |lat: f64, lon: f64| (s..=n).contains(&lat) && (w..=e).contains(&lon);
    let input = Path::new(&a[0]);
    // Pass 1: nodes in the box.
    let mut in_box: Vec<i64> = Vec::new();
    ElementReader::from_path(input)?.for_each(|el| match el {
        Element::DenseNode(x) if inside(x.lat(), x.lon()) => in_box.push(x.id()),
        Element::Node(x) if inside(x.lat(), x.lon()) => in_box.push(x.id()),
        _ => {}
    })?;
    in_box.sort_unstable();
    // Pass 2: the ways touching the box, and every relation.
    let (mut ways, mut relations) = (Vec::new(), Vec::new());
    ElementReader::from_path(input)?.for_each(|el| match el {
        Element::Way(x) => {
            let refs: Vec<i64> = x.refs().collect();
            if refs.iter().any(|r| in_box.binary_search(r).is_ok()) {
                ways.push(Way { id: x.id(), refs, tags: tags(x.tags()) });
            }
        }
        Element::Relation(x) => {
            let members = x
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
            relations.push(Relation { id: x.id(), members, tags: tags(x.tags()) });
        }
        _ => {}
    })?;
    let mut need: Vec<i64> = ways.iter().flat_map(|w: &Way| w.refs.iter().copied()).collect();
    need.sort_unstable();
    need.dedup();
    let kept_ways: HashSet<i64> = ways.iter().map(|w| w.id).collect();
    relations.retain(|r| {
        r.members.iter().all(|m| match m.kind {
            MemberType::Way => kept_ways.contains(&m.id),
            MemberType::Node => need.binary_search(&m.id).is_ok(),
            MemberType::Relation => false,
        })
    });
    // Pass 3: the nodes of the kept ways.
    let mut nodes = Vec::with_capacity(need.len());
    ElementReader::from_path(input)?.for_each(|el| {
        let (id, lat, lon, t) = match el {
            Element::DenseNode(x) => (x.id(), x.lat(), x.lon(), tags(x.tags())),
            Element::Node(x) => (x.id(), x.lat(), x.lon(), tags(x.tags())),
            _ => return,
        };
        if need.binary_search(&id).is_ok() {
            nodes.push(Node { id, lat, lon, tags: t });
        }
    })?;
    nodes.sort_by_key(|x| x.id);
    ways.sort_by_key(|x| x.id);
    relations.sort_by_key(|x| x.id);
    let timestamp = {
        let mut r = osmpbf::BlobReader::from_path(input)?;
        let blob = r.next().context("empty file")??;
        blob.to_headerblock()?.osmosis_replication_timestamp()
    };
    println!("{} nodes, {} ways, {} relations", nodes.len(), ways.len(), relations.len());
    OsmData { nodes, ways, relations, timestamp }.write(Path::new(&a[1]))?;
    if let Some(p) = a.get(3) {
        let ring = [(w, s), (e, s), (e, n), (w, n), (w, s)];
        let mut t = String::from("bbox\n1\n");
        for (lon, lat) in ring {
            t.push_str(&format!("   {lon:.7E}   {lat:.7E}\n"));
        }
        t.push_str("END\nEND\n");
        std::fs::write(p, t)?;
    }
    Ok(())
}
