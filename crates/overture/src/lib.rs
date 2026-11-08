//! The Overture inputs (SPEC §3, §6): addresses and places of the pinned release, the place
//! rules, duplicates as aliases, the taxonomy, and the extent polygon. The result is the
//! catalogue the vocabulary is built from, every ID checked to be a UUID.

pub mod extent;
pub mod places;
pub mod release;
pub mod taxonomy;

use std::collections::BTreeMap;

use anyhow::{Result, bail, ensure};
use blobmap_profile::config::PlaceRules;

pub use extent::Polygon;
pub use release::Release;
pub use taxonomy::{NONE16, Taxonomy};

/// An Overture ID: a UUID in canonical lower-case text form, as its 16 bytes in text order
/// (FORMAT §7.1).
pub type Id = [u8; 16];

pub fn parse_uuid(s: &str) -> Result<Id> {
    let b = s.as_bytes();
    let ok = b.len() == 36
        && b.iter()
            .enumerate()
            .all(|(i, &c)| if matches!(i, 8 | 13 | 18 | 23) { c == b'-' } else { c.is_ascii_digit() || (b'a'..=b'f').contains(&c) });
    ensure!(ok, "Overture ID {s:?} is not a UUID in canonical lower-case form (SPEC §3)");
    let hex: String = s.chars().filter(|&c| c != '-').collect();
    let mut id = [0u8; 16];
    for (k, x) in id.iter_mut().enumerate() {
        *x = u8::from_str_radix(&hex[2 * k..2 * k + 2], 16)?;
    }
    Ok(id)
}

pub fn uuid_string(id: &Id) -> String {
    let h: String = id.iter().map(|b| format!("{b:02x}")).collect();
    format!("{}-{}-{}-{}-{}", &h[0..8], &h[8..12], &h[12..16], &h[16..20], &h[20..32])
}

#[derive(Clone, Debug, PartialEq)]
pub struct Address {
    pub id: Id,
    pub lat: f64,
    pub lon: f64,
    pub country: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Place {
    pub id: Id,
    pub lat: f64,
    pub lon: f64,
    pub country: String,
    /// 0 kept, 4 `FilteredPlace` (FORMAT §7.4).
    pub status: u8,
    pub primary: u16,
    pub basic: u16,
    /// Ascending, without the primary.
    pub alt: Vec<u16>,
    /// The basic category's key as Overture gives it (the data report counts by it).
    pub basic_key: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CatalogStats {
    pub addresses: u64,
    pub places: u64,
    pub filtered: u64,
    pub merged: u64,
    pub primary_none: u64,
    /// Category keys absent from the taxonomy, with their place counts.
    pub missing_keys: BTreeMap<String, u64>,
}

/// The Overture IDs a model may hold, before snapping decides the extent without a polygon.
#[derive(Clone, Debug)]
pub struct Catalog {
    pub release: String,
    pub taxonomy: Taxonomy,
    /// Sorted by ID.
    pub addresses: Vec<Address>,
    /// Kept and filtered places, sorted by ID.
    pub places: Vec<Place>,
    /// Merged duplicates and the index in `places` of the place kept, sorted by ID.
    pub aliases: Vec<(Id, u32)>,
    /// Source licences, with the number of rows that carry each.
    pub licenses: BTreeMap<String, u64>,
    pub stats: CatalogStats,
}

impl Catalog {
    /// Reads the release, keeping the points `keep` accepts (the extent polygon, if any).
    pub fn read(release: &Release, rules: &PlaceRules, keep: &(dyn Fn(f64, f64) -> bool + Sync)) -> Result<Catalog> {
        let taxonomy = Taxonomy::read(&release.taxonomy_dir())?;
        let raw_a = release.read_addresses(keep)?;
        let raw_p = release.read_places(keep)?;
        let mut stats = CatalogStats { addresses: raw_a.len() as u64, places: raw_p.len() as u64, ..CatalogStats::default() };
        let mut licenses: BTreeMap<String, u64> = BTreeMap::new();
        let mut addresses = Vec::with_capacity(raw_a.len());
        for a in &raw_a {
            addresses.push(Address { id: parse_uuid(&a.id)?, lat: a.lat, lon: a.lon, country: a.country.clone() });
            for l in &a.licenses {
                *licenses.entry(l.clone()).or_default() += 1;
            }
        }
        addresses.sort_by_key(|a| a.id);
        let key = |k: &Option<String>, stats: &mut CatalogStats| -> u16 {
            match k {
                None => NONE16,
                Some(k) => taxonomy.id(k).unwrap_or_else(|| {
                    *stats.missing_keys.entry(k.clone()).or_default() += 1;
                    NONE16
                }),
            }
        };
        let mut places: Vec<(Place, usize)> = Vec::with_capacity(raw_p.len());
        for (i, p) in raw_p.iter().enumerate() {
            let id = parse_uuid(&p.id)?;
            let primary = key(&p.primary, &mut stats);
            let basic = key(&p.basic, &mut stats);
            let mut alt: Vec<u16> = p.alternates.iter().filter_map(|k| taxonomy.id(k)).filter(|&a| a != primary).collect();
            alt.sort_unstable();
            alt.dedup();
            let status = if places::filtered(p, rules, &taxonomy) { 4 } else { 0 };
            stats.filtered += (status == 4) as u64;
            stats.primary_none += (primary == NONE16) as u64;
            for l in &p.licenses {
                *licenses.entry(l.clone()).or_default() += 1;
            }
            places.push((
                Place { id, lat: p.lat, lon: p.lon, country: p.country.clone(), status, primary, basic, alt, basic_key: p.basic.clone() },
                i,
            ));
        }
        // Duplicates among the kept places.
        let kept: Vec<usize> = (0..places.len()).filter(|&k| places[k].0.status == 0).collect();
        let kept_raw: Vec<&release::RawPlace> = kept.iter().map(|&k| &raw_p[places[k].1]).collect();
        let merges = places::dedup(&kept_raw, rules.duplicates.radius_m);
        stats.merged = merges.len() as u64;
        let merged: std::collections::HashMap<usize, usize> = merges.iter().map(|&(m, k)| (kept[m as usize], kept[k as usize])).collect();
        let mut out: Vec<(Place, usize)> = Vec::new();
        let mut alias_raw: Vec<(Id, Id)> = Vec::new();
        for (k, p) in places.into_iter().enumerate() {
            match merged.get(&k) {
                Some(&to) => alias_raw.push((p.0.id, parse_uuid(&raw_p[to].id)?)),
                None => out.push(p),
            }
        }
        out.sort_by_key(|a| a.0.id);
        let places: Vec<Place> = out.into_iter().map(|p| p.0).collect();
        let index = |id: &Id| places.binary_search_by(|p| p.id.cmp(id)).ok().map(|i| i as u32);
        let mut aliases: Vec<(Id, u32)> = alias_raw.iter().map(|(a, k)| (*a, index(k).expect("kept place"))).collect();
        aliases.sort();
        // No ID twice across the three tables (FORMAT §7.1).
        let mut all: Vec<&Id> =
            addresses.iter().map(|a| &a.id).chain(places.iter().map(|p| &p.id)).chain(aliases.iter().map(|a| &a.0)).collect();
        all.sort_unstable();
        if let Some(w) = all.windows(2).find(|w| w[0] == w[1]) {
            bail!("Overture ID {} appears twice in the release", uuid_string(w[0]));
        }
        Ok(Catalog { release: release.name.clone(), taxonomy, addresses, places, aliases, licenses, stats })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uuids() {
        let s = "0123abcd-0000-4000-8000-00000000000f";
        assert_eq!(uuid_string(&parse_uuid(s).unwrap()), s);
        assert!(parse_uuid("0123ABCD-0000-4000-8000-00000000000f").is_err());
        assert!(parse_uuid("not-a-uuid").is_err());
    }

    #[test]
    fn town_catalog() {
        let dir = std::env::temp_dir().join(format!("blobmap-overture-{}", std::process::id()));
        let w = blobmap_fixtures::by_name("town").unwrap().write(&dir).unwrap();
        let cfg = blobmap_profile::config::Config::load(&w.configs[0].1).unwrap();
        let c = Catalog::read(&Release::open(&w.overture).unwrap(), &cfg.places.config, &|_, _| true).unwrap();
        assert_eq!(c.stats.filtered, 2);
        assert_eq!(c.stats.merged, 1);
        assert_eq!(c.aliases.len(), 1);
        assert_eq!(c.stats.missing_keys.get("space_elevator"), Some(&1));
        assert_eq!(c.stats.primary_none, 2);
        assert!(c.addresses.windows(2).all(|w| w[0].id < w[1].id));
        let t = &c.taxonomy;
        assert_eq!(t.resolve("pizzeria"), t.id("pizza_restaurant"));
        assert_eq!(t.resolve("gone_category"), None);
        assert!(t.is_under(t.id("pizza_restaurant").unwrap(), t.id("eat_and_drink").unwrap()));
        let pizza = c.places.iter().find(|p| p.primary == t.id("pizza_restaurant").unwrap()).unwrap();
        assert_eq!(pizza.alt, [t.id("cafe").unwrap()]);
        assert_eq!(pizza.basic, t.id("restaurant").unwrap());
    }
}
