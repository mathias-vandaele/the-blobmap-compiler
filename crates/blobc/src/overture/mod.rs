//! The Overture inputs (SPEC §3, §6): addresses and places of the pinned release, the place
//! rules, duplicates as aliases, the taxonomy, and the extent polygon. The result is the
//! catalogue the vocabulary is built from, every ID checked to be a UUID.

pub mod extent;
pub mod places;
pub mod release;
pub mod taxonomy;

use std::collections::BTreeMap;

use anyhow::{Result, bail, ensure};
use blobmap_format::NONE16;
use rayon::prelude::*;

use std::path::Path;

use crate::osm::column::{Column, Plain};
use crate::profile::config::PlaceRules;

pub use extent::Polygon;
pub use release::{NO_STR, Release};
pub use taxonomy::Taxonomy;

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

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct Address {
    pub id: Id,
    pub lat: f64,
    pub lon: f64,
    /// Index of the row's source licences in `Catalog::license_sets`.
    pub license: u32,
    /// Its country, as a number in `Catalog::strings` (`NO_STR` if none).
    pub country: u32,
}

// The planet's addresses are a column of the work directory, 40 bytes each (SPEC §10).
const _: () = assert!(std::mem::size_of::<Address>() == 40);
// SAFETY: `repr(C)` with no padding (40 bytes of fields), plain numbers only.
unsafe impl Plain for Address {}

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct Place {
    pub id: Id,
    pub lat: f64,
    pub lon: f64,
    /// Its first address's country, as a number in `Catalog::strings` (`NO_STR` if none).
    pub country: u32,
    /// The basic category's key as Overture gives it, as a number in `Catalog::strings`
    /// (the data report counts by it).
    pub basic_key: u32,
    /// The alternate categories are `Catalog::alts[alt_first..][..alt_len]`.
    pub alt_first: u32,
    /// Index of the row's source licences in `Catalog::license_sets`.
    pub license: u32,
    pub primary: u16,
    pub basic: u16,
    pub alt_len: u16,
    /// 0 kept, 4 `FilteredPlace` (FORMAT §7.4).
    pub status: u8,
    /// Always 0: the column holds no padding.
    pub pad: u8,
}

const _: () = assert!(std::mem::size_of::<Place>() == 56);
// SAFETY: `repr(C)` with no padding (56 bytes of fields), plain numbers only.
unsafe impl Plain for Place {}

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
#[derive(Clone, Debug, Default)]
pub struct Catalog {
    pub release: String,
    pub taxonomy: Taxonomy,
    /// Sorted by ID.
    pub addresses: Column<Address>,
    /// Kept and filtered places, sorted by ID.
    pub places: Column<Place>,
    /// The places' alternate categories: per place ascending, without the primary.
    pub alts: Vec<u16>,
    /// The country codes and category keys addresses and places point to.
    pub strings: Vec<String>,
    /// Merged duplicates and the index in `places` of the place kept, sorted by ID.
    pub aliases: Vec<(Id, u32)>,
    /// The source licences of each alias's own row, as `license_sets` indices.
    pub alias_license: Vec<u32>,
    /// The distinct sets of source licences that rows carry, sorted.
    pub license_sets: Vec<Vec<String>>,
    pub stats: CatalogStats,
}

impl Catalog {
    /// Reads the release, keeping the points `keep` accepts (the extent polygon, if any).
    /// The places come first, so that their rows as read are dropped before the addresses,
    /// the larger table, are read. Addresses and places go to `dir` if given.
    pub fn read(release: &Release, rules: &PlaceRules, keep: &(dyn Fn(f64, f64) -> bool + Sync), dir: Option<&Path>) -> Result<Catalog> {
        let taxonomy = Taxonomy::read(&release.taxonomy_dir())?;
        let lic = release::Licenses::default();
        let raw = release.read_places(keep, &lic)?;
        let mut stats = CatalogStats { places: raw.rows.len() as u64, ..CatalogStats::default() };
        // A place's keys go through the redirects, like query keys (SPEC §6).
        let resolved: Vec<Option<u16>> = raw.strings.iter().map(|k| taxonomy.resolve(k)).collect();
        let key = |k: u32, stats: &mut CatalogStats| -> u16 {
            if k == NO_STR {
                return NONE16;
            }
            resolved[k as usize].unwrap_or_else(|| {
                *stats.missing_keys.entry(raw.strings[k as usize].clone()).or_default() += 1;
                NONE16
            })
        };
        let mut places: Vec<Place> = Vec::with_capacity(raw.rows.len());
        let mut alts: Vec<u16> = Vec::new();
        let mut alt: Vec<u16> = Vec::new();
        for (i, p) in raw.rows.iter().enumerate() {
            let primary = key(p.primary, &mut stats);
            let basic = key(p.basic, &mut stats);
            alt.clear();
            alt.extend(raw.alternates(i).iter().filter_map(|&k| resolved[k as usize]).filter(|&a| a != primary));
            alt.sort_unstable();
            alt.dedup();
            let status = if places::filtered(raw.str(p.status), (primary != NONE16).then_some(primary), p.confidence, rules, &taxonomy) { 4 } else { 0 };
            stats.filtered += (status == 4) as u64;
            stats.primary_none += (primary == NONE16) as u64;
            let alt_first = u32::try_from(alts.len())?;
            alts.extend_from_slice(&alt);
            places.push(Place {
                id: p.id,
                lat: p.lat,
                lon: p.lon,
                country: p.country,
                status,
                primary,
                basic,
                basic_key: p.basic,
                alt_first,
                alt_len: u16::try_from(alt.len())?,
                license: p.license,
                pad: 0,
            });
        }
        // Duplicates among the kept places.
        let kept: Vec<u32> = (0..places.len() as u32).filter(|&k| places[k as usize].status == 0).collect();
        let merges = places::dedup(&raw, &kept, rules.duplicates.radius_m);
        stats.merged = merges.len() as u64;
        let mut strings = raw.strings.clone();
        drop(raw);
        let mut to = vec![u32::MAX; places.len()];
        for &(m, k) in &merges {
            to[kept[m as usize] as usize] = kept[k as usize];
        }
        let mut alias_raw: Vec<(Id, Id, u32)> = Vec::with_capacity(merges.len());
        for (k, &t) in to.iter().enumerate() {
            if t != u32::MAX {
                alias_raw.push((places[k].id, places[t as usize].id, places[k].license));
            }
        }
        let mut k = 0;
        places.retain(|_| {
            k += 1;
            to[k - 1] == u32::MAX
        });
        drop(to);
        places.par_sort_unstable_by_key(|p| p.id);
        let index = |id: &Id| places.binary_search_by(|p| p.id.cmp(id)).ok().map(|i| i as u32);
        let mut aliases: Vec<(Id, u32, u32)> = alias_raw.iter().map(|(a, k, l)| (*a, index(k).expect("kept place"), *l)).collect();
        drop(alias_raw);
        aliases.par_sort_unstable();
        let mut addresses = release.read_addresses(keep, &lic, &mut strings)?;
        stats.addresses = addresses.len() as u64;
        // Licence sets, in sorted order.
        let (license_sets, license_at) = lic.sorted();
        addresses.par_iter_mut().for_each(|a| a.license = license_at[a.license as usize]);
        places.par_iter_mut().for_each(|p| p.license = license_at[p.license as usize]);
        addresses.par_sort_unstable_by_key(|a| a.id);
        let alias_license = aliases.iter().map(|a| license_at[a.2 as usize]).collect();
        let aliases: Vec<(Id, u32)> = aliases.into_iter().map(|a| (a.0, a.1)).collect();
        // No ID twice across the three tables (FORMAT §7.1).
        if let Some(id) = twice([&mut addresses.iter().map(|a| &a.id), &mut places.iter().map(|p| &p.id), &mut aliases.iter().map(|a| &a.0)]) {
            bail!("Overture ID {} appears twice in the release", uuid_string(&id));
        }
        let (mut addresses, mut places) = (Column::from(addresses), Column::from(places));
        addresses.spill(dir, "addresses")?;
        places.spill(dir, "places")?;
        Ok(Catalog { release: release.name.clone(), taxonomy, addresses, places, alts, strings, aliases, alias_license, license_sets, stats })
    }

    /// A place's alternate categories.
    pub fn alt(&self, p: &Place) -> &[u16] {
        &self.alts[p.alt_first as usize..][..p.alt_len as usize]
    }

    /// A number in `strings`, `None` for `NO_STR`.
    pub fn str(&self, k: u32) -> Option<&str> {
        (k != NO_STR).then(|| self.strings[k as usize].as_str())
    }

    /// Source licences with the number of rows carrying each, over the given rows: address
    /// indices, place indices and alias indices (SPEC §12: the rows in the model).
    pub fn license_counts(&self, addresses: &[usize], places: &[usize], aliases: &[usize]) -> BTreeMap<String, u64> {
        let mut per_set = vec![0u64; self.license_sets.len()];
        let sets = addresses.iter().map(|&i| self.addresses[i].license).chain(places.iter().map(|&p| self.places[p].license));
        for l in sets.chain(aliases.iter().map(|&a| self.alias_license[a])) {
            per_set[l as usize] += 1;
        }
        let mut out: BTreeMap<String, u64> = BTreeMap::new();
        for (set, &n) in self.license_sets.iter().zip(&per_set) {
            for l in set {
                if n > 0 {
                    *out.entry(l.clone()).or_default() += n;
                }
            }
        }
        out
    }
}

/// The smallest ID that two of the sorted lists hold, or one holds twice.
fn twice<const N: usize>(mut lists: [&mut dyn Iterator<Item = &Id>; N]) -> Option<Id> {
    let mut head: [Option<&Id>; N] = std::array::from_fn(|k| lists[k].next());
    let mut last: Option<&Id> = None;
    loop {
        let k = (0..N).filter(|&k| head[k].is_some()).min_by_key(|&k| head[k])?;
        let x = head[k];
        if last == x {
            return x.copied();
        }
        last = x;
        head[k] = lists[k].next();
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
        let w = crate::fixtures::by_name("town").unwrap().write(&dir).unwrap();
        let cfg = crate::profile::config::Config::load(&w.configs[0].1).unwrap();
        let c = Catalog::read(&Release::open(&w.overture).unwrap(), &cfg.places.config, &|_, _| true, None).unwrap();
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
        assert_eq!(c.alt(pizza), [t.id("cafe").unwrap()]);
        assert_eq!(pizza.basic, t.id("restaurant").unwrap());
    }
}
