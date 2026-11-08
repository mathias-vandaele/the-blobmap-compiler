//! The invariants of FORMAT §3–§9 on a model's bytes: tables, blocks, levels, links,
//! chordality, turns, roads, copies, entries, vocabulary, and every hash.

use std::collections::BTreeMap;

use anyhow::Result;
use blobmap_format::hash::{ENTRY_TAG, Hasher, ROAD_TAG, VOCAB_TAG, inputs_hash, model_id, of_hashes};
use blobmap_format::{Hash, INF, NONE, NONE16};
use blobmap_runtime::{Fam, Runtime};

use crate::report::Report;

pub fn structure(rt: &Runtime, rep: &mut Report) -> Result<()> {
    for fam in [Fam::Car, Fam::Active] {
        let t = &rt.tables[&fam];
        let f = fam.name();
        let o = rep.check(&format!("{f}: tables, blocks, levels, links"));
        let ne = t.nodes.len();
        o.expect(t.block_first.len() == ne + 1 && t.block_first[0] == 0, || "block_first is not a CSR over the experts".into());
        o.expect(t.nodes.windows(2).all(|w| w[0] >= w[1]), || "experts are not numbered by node count descending".into());
        for x in 0..ne as u32 {
            let e = match rt.expert(fam, x) {
                Ok(e) => e,
                Err(err) => {
                    o.fail(format!("expert {x}: {err:#}"));
                    continue;
                }
            };
            o.count();
            let bs = t.block_first[x as usize] as usize..t.block_first[x as usize + 1] as usize;
            let kinds: Vec<u8> = t.b_kind[bs.clone()].to_vec();
            o.expect(kinds == [0] || (kinds.len() >= 3 && kinds[0] == 1 && kinds[1..].iter().all(|&k| k == 2)), || format!("expert {x}: block kinds {kinds:?}"));
            o.expect(t.b_id_first[bs.start] == 0, || format!("expert {x}: the first block does not start at 0"));
            // Levels of each block cover its ids, top first.
            let mut level_of = vec![u32::MAX; e.n];
            for b in bs.clone() {
                let (pack, base, l) = (t.b_pack[b], t.b_level_base[b] as usize, t.b_levels[b] as usize);
                let lf = rt.model.get::<u32>(&format!("road/{f}/p{pack}/level_first")).map(|v| v.into_owned()).unwrap_or_default();
                if base + l >= lf.len() {
                    o.fail(format!("expert {x} block {b}: level bounds out of range"));
                    continue;
                }
                let bounds = &lf[base..=base + l];
                let (a, z) = (t.b_id_first[b], t.b_id_first[b] + t.b_nodes[b]);
                o.expect(bounds[0] == a && bounds[l] == z && bounds.windows(2).all(|w| w[0] < w[1]), || format!("expert {x} block {b}: level bounds {bounds:?}"));
                for lv in 0..l {
                    for v in bounds[lv]..bounds[lv + 1] {
                        if (v as usize) < e.n {
                            level_of[v as usize] = lv as u32;
                        }
                    }
                }
            }
            for v in 0..e.n as u32 {
                let r = e.row(v);
                let row = &e.up_head[r.clone()];
                o.expect(row.windows(2).all(|w| w[0] < w[1]) && row.last().is_none_or(|&u| u < v), || format!("expert {x} node {v}: row {row:?}"));
                let bv = e.block_of(v);
                for &u in row {
                    let bu = e.block_of(u);
                    if bu == bv {
                        o.expect(level_of[u as usize] < level_of[v as usize], || format!("expert {x}: link {v}-{u} not upward in levels"));
                    } else {
                        o.expect(e.blocks[bu].kind == 1, || format!("expert {x}: link {v}-{u} joins two cells"));
                    }
                }
                for i in 0..row.len() {
                    for j in 0..i {
                        o.expect(e.link(row[i], row[j]).is_some(), || format!("expert {x}: not chordal at {v}"));
                    }
                }
                // Arcs: sorted by head (CAR: strictly), heads in range.
                let ar = e.arcs(v);
                let heads = &e.arc_head[ar.clone()];
                o.expect(heads.iter().all(|&h| (h as usize) < e.n), || format!("expert {x} node {v}: arc head out of range"));
                if fam == Fam::Car {
                    o.expect(heads.windows(2).all(|w| w[0] < w[1]), || format!("expert {x} node {v}: turns not ascending"));
                    o.expect(e.turn_cost[ar.clone()].iter().all(|&c| c != INF), || format!("expert {x} node {v}: an INF turn cost"));
                } else {
                    let keys: Vec<(u32, i64, u32)> = ar.clone().map(|a| (e.arc_head[a], e.way[a], e.way_seq[a])).collect();
                    o.expect(keys.windows(2).all(|w| w[0] <= w[1]), || format!("expert {x} node {v}: roads not ordered"));
                }
            }
            if fam == Fam::Car {
                for v in 0..e.n {
                    let cs = &e.copies[e.copy_first[v] as usize..e.copy_first[v + 1] as usize];
                    o.expect(cs.windows(2).all(|w| w[0] < w[1]), || format!("expert {x} node {v}: copies not ascending"));
                    for &c in cs {
                        let c = c as usize;
                        o.expect(
                            e.kind[c] == 2 && e.copy_of[c] == v as u32 && e.kind[v] != 2,
                            || format!("expert {x}: copy {c} of {v} is not marked as one"),
                        );
                        o.expect(
                            (e.way[c], e.way_dir[c], e.way_seq[c], e.w[0][c], e.time_s[0][c].to_bits())
                                == (e.way[v], e.way_dir[v], e.way_seq[v], e.w[0][v], e.time_s[0][v].to_bits()),
                            || format!("expert {x}: copy {c} differs from its original {v}"),
                        );
                    }
                    o.expect(e.kind[v] == 2 || e.copy_of[v] == NONE, || format!("expert {x}: original {v} has copy_of"));
                }
            }
            // Road weights (FORMAT §1, §6.1): a finite master time is at most 65534 and its
            // weight below INF; an infinite one has weight INF.
            for (m, (ts, ws)) in e.time_s.iter().zip(&e.w).enumerate() {
                for (r, (&ti, &wi)) in ts.iter().zip(ws).enumerate() {
                    let ok = if ti.is_finite() { ti <= blobmap_osm::MAX_PIECE_S && wi < INF } else { wi == INF };
                    o.expect(ok, || format!("expert {x} road {r} metric {m}: time {ti}, weight {wi}"));
                }
            }
            o.expect(t.levels[x as usize] == expert_levels(&t.b_kind[bs.clone()], &t.b_levels[bs.clone()]), || format!("expert {x}: levels"));
        }
        // Manifest entries.
        let (mut next, mut pack) = (0u64, 0u64);
        for (gi, en) in rt.manifest.experts.iter().enumerate().filter(|(_, e)| e.family == f) {
            o.expect(en.experts[0] == next && en.packs[0] == pack, || format!("entry {gi}: experts or packs not consecutive"));
            for xx in en.experts[0]..en.experts[0] + en.experts[1] {
                o.expect(t.entry.get(xx as usize) == Some(&(gi as u32)), || format!("expert {xx}: entry index"));
            }
            next += en.experts[1];
            pack += en.packs[1];
        }
        o.expect(next == ne as u64, || "entries do not cover the experts".into());
    }
    vocab(rt, rep);
    Ok(())
}

fn expert_levels(kinds: &[u8], levels: &[u32]) -> u32 {
    let top: u32 = kinds.iter().zip(levels).filter(|(k, _)| **k != 2).map(|(_, l)| l).sum();
    top + kinds.iter().zip(levels).filter(|(k, _)| **k == 2).map(|(_, l)| *l).max().unwrap_or(0)
}

fn vocab(rt: &Runtime, rep: &mut Report) {
    let v = &rt.vocab;
    let o = rep.check("vocabulary: IDs, entries, reasons, places");
    let asc = |x: &[[u8; 16]]| x.windows(2).all(|w| w[0] < w[1]);
    o.expect(asc(&v.address) && asc(&v.place) && asc(&v.alias), || "ID tables not strictly ascending".into());
    let mut all: Vec<&[u8; 16]> = v.address.iter().chain(&v.place).chain(&v.alias).collect();
    all.sort();
    o.expect(all.windows(2).all(|w| w[0] != w[1]), || "an ID appears twice".into());
    let na = v.address.len();
    o.expect(v.alias_to.iter().all(|&i| (i as usize) >= na && (i as usize) < v.n() && v.status[i as usize - na] == 0), || "alias_to names no kept place".into());
    for mv in &v.modes {
        let mode = &mv.info.name;
        o.expect(mv.reasons.windows(2).all(|w| w[0].0 < w[1].0), || format!("{mode}: reasons not sorted"));
        let mut reasons = mv.reasons.iter().peekable();
        for i in 0..v.n() {
            o.count();
            let has_reason = reasons.peek().is_some_and(|r| r.0 as usize == i);
            let reason = if has_reason { reasons.next() } else { None };
            let filtered = v.status_of(i as u32) == 4;
            let x = mv.expert[i];
            if x == NONE {
                o.expect(mv.node[i] == [NONE, NONE] && mv.off[i] == [INF; 4], || format!("{mode} id {i}: unserved with entries"));
                o.expect(filtered != reason.is_some(), || format!("{mode} id {i}: unserved needs exactly one reason unless filtered"));
                if let Some(r) = reason {
                    o.expect((1..=3).contains(&r.1) && (r.1 == 2) == r.2.is_infinite(), || format!("{mode} id {i}: reason {r:?}"));
                }
                continue;
            }
            o.expect(!filtered && reason.is_none(), || format!("{mode} id {i}: served and filtered or with a reason"));
            let t = &rt.tables[&mv.info.fam];
            o.expect(t.trapped.get(x as usize) == Some(&0), || format!("{mode} id {i}: served in a trapped expert"));
            let Ok(e) = rt.expert(mv.info.fam, x) else {
                o.fail(format!("{mode} id {i}: expert {x} does not load"));
                continue;
            };
            let (nd, off) = (mv.node[i], mv.off[i]);
            let ok_node = |n: u32| (n as usize) < e.n;
            match mv.info.fam {
                Fam::Car => {
                    o.expect(nd.iter().any(|&s| s != NONE), || format!("car id {i}: no segment"));
                    for k in 0..2 {
                        if nd[k] == NONE {
                            o.expect(off[2 * k] == INF && off[2 * k + 1] == INF, || format!("car id {i}: offsets on a NONE slot"));
                            continue;
                        }
                        let s = nd[k] as usize;
                        o.expect(ok_node(nd[k]) && e.kind[s] != 2, || format!("car id {i}: slot {k} is not an original segment"));
                        if ok_node(nd[k]) {
                            let w = e.w[0][s];
                            let sum = if w == INF { off[2 * k] == INF && off[2 * k + 1] == INF } else { off[2 * k] as u32 + off[2 * k + 1] as u32 == w as u32 };
                            o.expect(sum, || format!("car id {i}: slot {k} parts do not sum to the weight"));
                        }
                    }
                    if nd[0] != NONE && nd[1] != NONE && ok_node(nd[0]) && ok_node(nd[1]) {
                        let (a, b) = (nd[0] as usize, nd[1] as usize);
                        o.expect(e.way[a] == e.way[b] && e.way_seq[a] == e.way_seq[b] && e.way_dir[a] != e.way_dir[b] && e.way_dir[a] == 0, || format!("car id {i}: slots are not the piece's [A→B, B→A]"));
                    }
                }
                Fam::Active => {
                    let m = mv.info.metric;
                    o.expect(ok_node(nd[0]) && ok_node(nd[1]), || format!("{mode} id {i}: nodes out of range"));
                    if !(ok_node(nd[0]) && ok_node(nd[1])) {
                        continue;
                    }
                    let uv = e.arcs(nd[0]).find(|&a| e.arc_local[a] == mv.road[i] && e.arc_head[a] == nd[1]);
                    o.expect(uv.is_some(), || format!("{mode} id {i}: road U → V does not exist"));
                    let Some(uv) = uv else { continue };
                    let w = e.w[m][uv];
                    let ok = if w == INF { off[0] == INF && off[3] == INF } else { off[0] as u32 + off[3] as u32 == w as u32 };
                    o.expect(ok, || format!("{mode} id {i}: U → V parts do not sum"));
                    let vu = e.arcs(nd[1]).find(|&a| e.arc_head[a] == nd[0] && e.way[a] == e.way[uv] && e.way_seq[a] == e.way_seq[uv] && e.way_dir[a] != e.way_dir[uv]);
                    match vu {
                        Some(vu) => {
                            let w = e.w[m][vu];
                            let ok = if w == INF { off[2] == INF && off[1] == INF } else { off[2] as u32 + off[1] as u32 == w as u32 };
                            o.expect(ok, || format!("{mode} id {i}: V → U parts do not sum"));
                        }
                        None => o.expect(off[2] == INF && off[1] == INF, || format!("{mode} id {i}: offsets on an absent road")),
                    }
                    let usable = e.time_s[m][uv].is_finite() || vu.is_some_and(|vu| e.time_s[m][vu].is_finite());
                    o.expect(usable, || format!("{mode} id {i}: no direction the mode may use"));
                }
            }
        }
    }
    for p in 0..v.place.len() {
        let alt = &v.alt[v.alt_first[p] as usize..v.alt_first[p + 1] as usize];
        o.expect(alt.windows(2).all(|w| w[0] < w[1]) && !alt.contains(&v.primary[p]), || format!("place {p}: alternates"));
        o.expect(v.status[p] == 0 || v.status[p] == 4, || format!("place {p}: status {}", v.status[p]));
    }
    let t = v.tax_name.len();
    o.expect(v.tax_name.windows(2).all(|w| w[0] < w[1]) && v.redirect.windows(2).all(|w| w[0] < w[1]), || "taxonomy keys not sorted".into());
    o.expect(v.tax_basic.iter().all(|&b| b == NONE16 || (b as usize) < t && v.tax_is_basic[b as usize] == 1), || "a basic category that is not basic".into());
    o.expect(v.tax_parent.iter().all(|&b| b == NONE16 || (b as usize) < t), || "a parent out of range".into());
}

/// Every hash of FORMAT §9, recomputed from the bytes.
pub fn hashes(rt: &Runtime, rep: &mut Report) -> Result<()> {
    let m = &rt.model;
    let mf = &rt.manifest;
    let o = rep.check("hashes (FORMAT §9)");
    let mut tables: BTreeMap<String, Hash> = BTreeMap::new();
    for name in m.names() {
        if name.starts_with("road/car/experts/") || name.starts_with("road/car/blocks/") || name.starts_with("road/active/experts/")
            || name.starts_with("road/active/blocks/") || name.starts_with("road/active/cross/")
        {
            tables.insert(name.to_string(), m.tensor_hash(name)?);
        }
    }
    let mut rh = Hasher::tagged(ROAD_TAG);
    for h in tables.values() {
        rh.hash(h);
    }
    for en in &mf.experts {
        let mut hs: BTreeMap<String, Hash> = BTreeMap::new();
        for k in en.packs[0]..en.packs[0] + en.packs[1] {
            for name in m.names_in(&format!("road/{}/p{k}", en.family)) {
                hs.insert(name.to_string(), m.tensor_hash(name)?);
            }
        }
        let h = of_hashes(ENTRY_TAG, hs.values());
        o.expect(h.hex() == en.hash, || format!("{} entry {:?}: hash", en.family, en.experts));
        rh.hash(&h);
    }
    let road = rh.finish();
    o.expect(road.hex() == mf.parts.road.hash, || "road part hash".into());
    let mut vh = Hasher::tagged(VOCAB_TAG);
    for name in m.names_in("vocab") {
        if name != "vocab/road_hash" {
            vh.hash(&m.tensor_hash(name)?);
        }
    }
    vh.bytes(&rt.vocab.road_hash);
    let vocab = vh.finish();
    o.expect(vocab.hex() == mf.parts.vocabulary.hash, || "vocabulary part hash".into());
    o.expect(model_id(&road, &vocab).hex() == mf.model_id, || "model_id".into());
    let hx = |s: &str| Hash::from_hex(s).unwrap_or_default();
    let ih = inputs_hash(&blobmap_format::hash::Inputs {
        pbf_sha256: hx(&mf.inputs.pbf.sha256),
        overture_release: &mf.inputs.overture.release,
        car_profile: hx(&mf.config.profiles.car.sha256),
        active_profile: hx(&mf.config.profiles.active.sha256),
        place_rules: hx(&mf.config.place_rules.sha256),
        extent: mf.inputs.extent.as_ref().map(|e| hx(&e.sha256)),
        spec_version: &mf.spec_version,
    });
    o.expect(ih.hex() == mf.inputs_hash, || "inputs_hash".into());
    let lic = std::fs::read(rt.model.root.join(blobmap_format::LICENSES_FILE)).unwrap_or_default();
    o.expect(Hash::of(&lic).hex() == mf.licenses_sha256, || "licenses_sha256".into());
    o.expect(m.verify_shard_hashes().is_ok(), || "shard SHA-256 (transport)".into());
    o.count();
    Ok(())
}
