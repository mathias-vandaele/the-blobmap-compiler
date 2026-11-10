//! Reading the `.osm.pbf` (SPEC §3): the ways a family's profile keeps, split into pieces
//! between routing nodes, with their lengths, shapes, times per metric and direction
//! (delays included, SPEC §4), snapping eligibility (SPEC §6), and the car turn
//! restrictions, via nodes and via ways.

pub mod column;
pub mod geo;
pub mod network;
pub mod nodes;
pub mod restrictions;

use std::path::Path;

use anyhow::{Context, Result};

pub use network::{Network, Pieces};
pub use nodes::Osm;
pub use restrictions::{Restriction, Via};

/// The `osmosis_replication_timestamp` of the file's header, seconds since the epoch.
pub fn header_timestamp(path: &Path) -> Result<Option<i64>> {
    let mut r = osmpbf::BlobReader::from_path(path).with_context(|| format!("opening {}", path.display()))?;
    let blob = r.next().with_context(|| format!("{}: empty file", path.display()))??;
    anyhow::ensure!(matches!(blob.get_type(), osmpbf::BlobType::OsmHeader), "{}: does not start with an OSM header", path.display());
    Ok(blob.to_headerblock()?.osmosis_replication_timestamp())
}

/// Length of a straight segment as OSRM measures it: the cheap ruler on the WGS84
/// ellipsoid at the segment's mean latitude, longitudes taken across the antimeridian the
/// short way.
pub fn cheap_ruler_m(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let (kx, ky) = ruler(((lat1 + lat2) / 2.0).to_radians().cos());
    (geo::lon_diff(lon2, lon1) * kx).hypot((lat2 - lat1) * ky)
}

/// Metres per degree of longitude and of latitude where cos(latitude) = `cl`.
pub fn ruler(cl: f64) -> (f64, f64) {
    const RE: f64 = 6_378_137.0;
    const FE: f64 = 1.0 / 298.257223563;
    const E2: f64 = FE * (2.0 - FE);
    let m = RE * std::f64::consts::PI / 180.0;
    let w2 = 1.0 / (1.0 - E2 * (1.0 - cl * cl));
    let w = w2.sqrt();
    (m * w * cl, m * w * w2 * (1.0 - E2))
}

/// The longest master time a piece may have (SPEC §7, long pieces): its weight then stays
/// below `INF`.
pub const MAX_PIECE_S: f32 = 65534.0;

/// Master time (SPEC §7) of `len` metres at `kmh`, plus a delay in seconds.
pub fn arc_time(len: f64, kmh: f32, delay: f64) -> f32 {
    (len / (kmh as f64 / 3.6) + delay) as f32
}

/// Decimicro-degrees to degrees, as every reader of this compiler computes them.
pub fn degrees(decimicro: i32) -> f64 {
    decimicro as f64 / 1e7
}
