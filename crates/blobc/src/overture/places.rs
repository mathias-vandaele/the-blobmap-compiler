//! The place rules of SPEC §3 (ported from the old `overture/places.rs`): the closed and
//! confidence filters, and duplicates merged into aliases.

use std::collections::HashMap;

use crate::osm::geo::{Local, dist_m};
use rayon::prelude::*;

use crate::overture::Id;
use crate::overture::release::{NO_STR, RawPlaces};
use crate::overture::taxonomy::Taxonomy;
use crate::profile::config::PlaceRules;

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
/// Tokens are numbers in `words`, in the order of their strings.
pub fn names_match(a: &[u32], b: &[u32], words: &[String]) -> bool {
    if a.is_empty() || b.is_empty() || a == b {
        return true;
    }
    let (s, l) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    s.iter().all(|&w| {
        let w = words[w as usize].as_str();
        l.iter().any(|&x| {
            let x = words[x as usize].as_str();
            x == w || (w.len() >= FUZZY_LEN && x.len() >= FUZZY_LEN && strsim::levenshtein(w, x) <= 1)
        })
    })
}

/// Whether the closed or confidence rule filters a place of this status, resolved primary
/// category and confidence.
pub fn filtered(status: Option<&str>, primary: Option<u16>, confidence: f64, rules: &PlaceRules, tax: &Taxonomy) -> bool {
    if status.is_some_and(|s| rules.rules.closed_statuses.iter().any(|c| c == s)) {
        return true;
    }
    let min = rules
        .confidence
        .under
        .iter()
        .find(|(k, _)| primary.zip(tax.id(k)).is_some_and(|(c, n)| tax.is_under(c, n)))
        .map_or(rules.confidence.default, |x| x.1);
    confidence < min
}

/// Places tokenised at a time: their token strings are held only until numbered.
const TOKEN_WINDOW: usize = 1 << 20;

/// Each place's name tokens as numbers in the returned words, numbered as first met.
fn tokens(raw: &RawPlaces, places: &[u32]) -> (Vec<u64>, Vec<u32>, Vec<String>) {
    let (mut first, mut toks, mut words) = (vec![0u64], Vec::new(), Vec::new());
    let mut index: HashMap<String, u32> = HashMap::new();
    for w in places.chunks(TOKEN_WINDOW) {
        let t: Vec<Vec<String>> = w.par_iter().map(|&i| name_tokens(raw.name(i as usize))).collect();
        for t in t {
            for s in t {
                let next = words.len() as u32;
                toks.push(*index.entry(s).or_insert_with_key(|s| {
                    words.push(s.clone());
                    next
                }));
            }
            first.push(toks.len() as u64);
        }
    }
    (first, toks, words)
}

/// Duplicates among `places` (indices into `raw`'s rows), as indices into `places`: each
/// merged place and the place kept. Places are taken by confidence descending, ties to the
/// smallest ID; one joins the nearest place already kept (ties to the smallest ID) with the
/// same basic category, within the radius and with a compatible name, else it is kept. A
/// merged place never attracts another, so there is no chaining.
pub fn dedup(raw: &RawPlaces, places: &[u32], radius_m: f64) -> Vec<(u32, u32)> {
    let row = |i: u32| &raw.rows[places[i as usize] as usize];
    let (tok_first, toks, words) = tokens(raw, places);
    let name = |i: u32| &toks[tok_first[i as usize] as usize..tok_first[i as usize + 1] as usize];
    let mut order: Vec<u32> = (0..places.len() as u32).collect();
    order.par_sort_unstable_by(|&a, &b| {
        let (pa, pb) = (row(a), row(b));
        pb.confidence.total_cmp(&pa.confidence).then_with(|| pa.id.cmp(&pb.id))
    });
    let cell_deg = radius_m.max(1.0) / 110_000.0;
    let cell = |x: f64| (x / cell_deg).floor() as i64;
    // Every place with a basic category in its grid cell; a place joins the grid, as one
    // kept, once taken and not merged.
    let mut grid: Vec<(i64, i64, u32)> =
        (0..places.len() as u32).filter(|&i| row(i).basic != NO_STR).map(|i| (cell(row(i).lon), cell(row(i).lat), i)).collect();
    grid.par_sort_unstable();
    let mut kept = vec![false; places.len()];
    let mut merges = Vec::new();
    for &i in &order {
        let p = row(i);
        if p.basic == NO_STR {
            continue;
        }
        let cy = cell(p.lat);
        let w = cell_deg * Local::at((p.lat.abs() + cell_deg).min(89.9), 0.0).scale().0;
        let k = (radius_m / w).ceil() as i64;
        // Near the antimeridian, the cells across it too.
        let reach = (k + 1) as f64 * cell_deg;
        let shifts = [0.0, -360.0, 360.0].into_iter().filter(|&s| s == 0.0 || (p.lon + s).abs() <= 180.0 + reach);
        let cols: Vec<i64> = shifts.flat_map(|s| cell(p.lon + s) - k..=cell(p.lon + s) + k).collect();
        let mut best: Option<(f64, &Id, u32)> = None;
        for gx in cols {
            // The column's three cells cy - 1 ..= cy + 1 follow one another in the grid.
            let lo = grid.partition_point(|c| (c.0, c.1) < (gx, cy - 1));
            for &(_, _, r) in grid[lo..].iter().take_while(|c| c.0 == gx && c.1 <= cy + 1) {
                let q = row(r);
                if !kept[r as usize] || q.basic != p.basic {
                    continue;
                }
                let d = dist_m((p.lat, p.lon), (q.lat, q.lon));
                if d > radius_m || !names_match(name(i), name(r), &words) {
                    continue;
                }
                if best.is_none_or(|b| (d, &q.id) < (b.0, b.1)) {
                    best = Some((d, &q.id, r));
                }
            }
        }
        match best {
            Some((_, _, r)) => merges.push((i, r)),
            None => kept[i as usize] = true,
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
        let mut words: Vec<String> = Vec::new();
        let mut t = |s: &str| -> Vec<u32> {
            name_tokens(s)
                .into_iter()
                .map(|w| match words.iter().position(|x| *x == w) {
                    Some(k) => k as u32,
                    None => {
                        words.push(w);
                        words.len() as u32 - 1
                    }
                })
                .collect()
        };
        let cases = [(t("Chez Paul"), t("Chez Paul SARL"), true), (t("Boulangerie"), t("Boulangeries"), true), (t("Chez Paul"), t("Chez Marie"), false), (t(""), t("Anything"), true)];
        for (a, b, m) in cases {
            assert_eq!(names_match(&a, &b, &words), m);
        }
    }
}
