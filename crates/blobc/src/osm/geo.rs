//! The local tangent plane (ported from the old `geo/local.rs`): metres east and north of a
//! point on the WGS84 ellipsoid, distances and bearings, so the same code holds anywhere on
//! Earth.

/// The local tangent plane at a point: metres east (x) and north (y) of it, with the
/// WGS84 radii of curvature there (equirectangular around the origin; over the 2 km
/// of a snapping search its distance error is below 1e-6 relative).
#[derive(Clone, Copy, Debug)]
pub struct Local {
    lat0: f64,
    lon0: f64,
    /// Metres per degree east and north at the origin.
    kx: f64,
    ky: f64,
}

impl Local {
    pub fn at(lat: f64, lon: f64) -> Local {
        const A: f64 = 6_378_137.0;
        const E2: f64 = 0.006_694_379_990_141_317; // WGS84 first eccentricity squared
        let (s, c) = lat.to_radians().sin_cos();
        let w = 1.0 - E2 * s * s;
        let n = A / w.sqrt(); // prime vertical radius
        let m = A * (1.0 - E2) / (w * w.sqrt()); // meridional radius
        let d = std::f64::consts::PI / 180.0;
        Local { lat0: lat, lon0: lon, kx: n * c * d, ky: m * d }
    }

    /// Metres east and north of the origin (longitudes across the antimeridian wrap).
    pub fn xy(&self, lat: f64, lon: f64) -> (f64, f64) {
        let mut dl = lon - self.lon0;
        if dl > 180.0 {
            dl -= 360.0;
        } else if dl < -180.0 {
            dl += 360.0;
        }
        (dl * self.kx, (lat - self.lat0) * self.ky)
    }

    /// The point at (x, y) metres from the origin.
    pub fn latlon(&self, x: f64, y: f64) -> (f64, f64) {
        let lon = self.lon0 + x / self.kx;
        (
            self.lat0 + y / self.ky,
            if lon > 180.0 {
                lon - 360.0
            } else if lon < -180.0 {
                lon + 360.0
            } else {
                lon
            },
        )
    }

    /// Metres per degree of longitude and of latitude at the origin.
    pub fn scale(&self) -> (f64, f64) {
        (self.kx, self.ky)
    }
}

/// `a - b` in degrees of longitude, across the antimeridian the short way: the IEEE
/// remainder by 360, as OSRM's cheap ruler takes it (`longDiff`). Exact, and `a - b` itself
/// whenever |a - b| ≤ 180.
pub fn lon_diff(a: f64, b: f64) -> f64 {
    let d = a - b;
    d - 360.0 * (d / 360.0).round_ties_even()
}

/// A longitude brought into [-180, 180].
pub fn lon_wrap(lon: f64) -> f64 {
    if lon > 180.0 {
        lon - 360.0
    } else if lon < -180.0 {
        lon + 360.0
    } else {
        lon
    }
}

/// Rings of (lat, lon) that may cross the antimeridian, for point-in-polygon tests: each
/// ring is unwrapped so that consecutive longitudes differ by at most 180°, and a point is
/// tested at its longitude and at ±360° of it. A ring that does not cross is unchanged.
pub fn unwrap_ring(ring: &mut [(f64, f64)]) {
    for i in 1..ring.len() {
        ring[i].1 = ring[i - 1].1 + lon_diff(ring[i].1, ring[i - 1].1);
    }
}

/// The longitudes to test a point at against unwrapped rings spanning `lo ..= hi`.
pub fn lon_shifts(lon: f64, lo: f64, hi: f64) -> impl Iterator<Item = f64> {
    [lon, lon - 360.0, lon + 360.0].into_iter().filter(move |&x| x >= lo && x <= hi)
}

/// Distance between two (lat, lon) points, metres (in the tangent plane at their midpoint).
pub fn dist_m(a: (f64, f64), b: (f64, f64)) -> f64 {
    let l = Local::at(0.5 * (a.0 + b.0), a.1);
    let ((ax, ay), (bx, by)) = (l.xy(a.0, a.1), l.xy(b.0, b.1));
    (bx - ax).hypot(by - ay)
}

/// Bearing from a to b, degrees clockwise from north (in the tangent plane at a).
pub fn bearing_deg(a: (f64, f64), b: (f64, f64)) -> f64 {
    let (x, y) = Local::at(a.0, a.1).xy(b.0, b.1);
    x.atan2(y).to_degrees()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn antimeridian() {
        assert_eq!((lon_diff(-179.9, 179.9), lon_diff(179.9, -179.9), lon_diff(2.5, 1.25)), (-359.8 + 360.0, 359.8 - 360.0, 1.25));
        let (a, b) = (crate::osm::cheap_ruler_m(-17.0, 179.999, -17.0, -179.999), crate::osm::cheap_ruler_m(-17.0, -179.999, -17.0, 179.999));
        assert!(a == b && a > 200.0 && a < 220.0, "{a}");
        let mut r = vec![(0.0, 179.0), (0.0, -179.0), (1.0, -179.0), (1.0, 179.0), (0.0, 179.0)];
        unwrap_ring(&mut r);
        assert_eq!(r[1].1, 181.0);
        assert_eq!(lon_shifts(-179.5, 179.0, 181.0).collect::<Vec<_>>(), vec![180.5]);
    }
}
