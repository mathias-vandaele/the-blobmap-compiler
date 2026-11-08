//! The place rules of SPEC §3 (ported from the old `overture/places.rs`): the closed and
//! confidence filters, and duplicates merged into aliases.

use std::collections::HashMap;

use blobmap_osm::geo::{Local, dist_m};
use blobmap_profile::config::PlaceRules;

use crate::release::RawPlace;
use crate::taxonomy::Taxonomy;

/// Articles, prepositions and company forms that do not tell two places apart.
const STOP: [&str; 22] = [
    "le", "la", "les", "l", "de", "du", "des", "d", "et", "a", "au", "aux", "en", "the", "and", "of", "sarl", "sas", "sasu", "eurl", "sa",
    "sci",
];

/// Normalised tokens: ASCII-folded, lower case, split on anything not a letter or digit,
/// stop words dropped, sorted and deduplicated.
pub fn name_tokens(name: &str) -> Vec<String> {
    let folded = deunicode::deunicode(name).to_lowercase();
    let mut t: Vec<String> =
        folded.split(|c: char| !c.is_ascii_alphanumeric()).filter(|w| !w.is_empty() && !STOP.contains(w)).map(str::to_string).collect();
    t.sort();
    t.dedup();
    t
}

pub const FUZZY_LEN: usize = 5;

/// Compatible names (SPEC §3): either is missing, or every token of the shorter equals a
/// token of the longer or, both at least `FUZZY_LEN` long, is one edit away from it.
pub fn names_match(a: &[String], b: &[String]) -> bool {
    if a.is_empty() || b.is_empty() || a == b {
        return true;
    }
    let (s, l) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    s.iter().all(|w| l.iter().any(|x| x == w || (w.len() >= FUZZY_LEN && x.len() >= FUZZY_LEN && strsim::levenshtein(w, x) <= 1)))
}

/// Whether the closed or confidence rule filters the place.
pub fn filtered(p: &RawPlace, rules: &PlaceRules, tax: &Taxonomy) -> bool {
    if p.status.as_ref().is_some_and(|s| rules.rules.closed_statuses.contains(s)) {
        return true;
    }
    let node = p.primary.as_deref().and_then(|k| tax.id(k));
    let min = rules
        .confidence
        .under
        .iter()
        .find(|(k, _)| node.zip(tax.id(k)).is_some_and(|(c, n)| tax.is_under(c, n)))
        .map_or(rules.confidence.default, |x| x.1);
    p.confidence < min
}

/// Duplicates among `places` (indices into it): each merged place and the place kept.
/// Places are taken by confidence descending, ties to the smallest ID; one joins the
/// nearest place already kept (ties to the smallest ID) with the same basic category,
/// within the radius and with a compatible name, else it is kept. A merged place never
/// attracts another, so there is no chaining.
pub fn dedup(places: &[&RawPlace], radius_m: f64) -> Vec<(u32, u32)> {
    let tokens: Vec<Vec<String>> = places.iter().map(|p| name_tokens(&p.name)).collect();
    let mut order: Vec<u32> = (0..places.len() as u32).collect();
    order.sort_by(|&a, &b| {
        let (pa, pb) = (places[a as usize], places[b as usize]);
        pb.confidence.total_cmp(&pa.confidence).then_with(|| pa.id.cmp(&pb.id))
    });
    let cell_deg = radius_m.max(1.0) / 110_000.0;
    let cell = |x: f64| (x / cell_deg).floor() as i64;
    let mut grid: HashMap<(i64, i64), Vec<u32>> = HashMap::new();
    let mut merges = Vec::new();
    for &i in &order {
        let p = places[i as usize];
        if p.basic.is_none() {
            continue;
        }
        let (cx, cy) = (cell(p.lon), cell(p.lat));
        let w = cell_deg * Local::at((p.lat.abs() + cell_deg).min(89.9), 0.0).scale().0;
        let k = (radius_m / w).ceil() as i64;
        let mut best: Option<(f64, &str, u32)> = None;
        for gx in cx - k..=cx + k {
            for gy in cy - 1..=cy + 1 {
                for &r in grid.get(&(gx, gy)).map(Vec::as_slice).unwrap_or(&[]) {
                    let q = places[r as usize];
                    if q.basic != p.basic {
                        continue;
                    }
                    let d = dist_m((p.lat, p.lon), (q.lat, q.lon));
                    if d > radius_m || !names_match(&tokens[i as usize], &tokens[r as usize]) {
                        continue;
                    }
                    if best.is_none_or(|b| (d, q.id.as_str()) < (b.0, b.1)) {
                        best = Some((d, q.id.as_str(), r));
                    }
                }
            }
        }
        match best {
            Some((_, _, r)) => merges.push((i, r)),
            None => grid.entry((cx, cy)).or_default().push(i),
        }
    }
    merges.sort_unstable();
    merges
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names() {
        let t = |s: &str| name_tokens(s);
        assert!(names_match(&t("Chez Paul"), &t("Chez Paul SARL")));
        assert!(names_match(&t("Boulangerie"), &t("Boulangeries")));
        assert!(!names_match(&t("Chez Paul"), &t("Chez Marie")));
        assert!(names_match(&t(""), &t("Anything")));
    }
}
