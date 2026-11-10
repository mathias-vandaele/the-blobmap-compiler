//! The extent polygon (SPEC §3, `--extent`): an Osmosis `.poly` file (ported from the old
//! `geo/region.rs`), its rings under the even-odd rule.

use anyhow::{Context, Result, bail, ensure};

use crate::osm::geo::{lon_shifts, unwrap_ring};

#[derive(Clone, Debug, PartialEq)]
pub struct Polygon {
    /// Rings of (lat, lon), closed implicitly, unwrapped across the antimeridian
    /// (`geo::unwrap_ring`).
    pub rings: Vec<Vec<(f64, f64)>>,
    /// The longitudes the rings span.
    lo: f64,
    hi: f64,
}

impl Polygon {
    /// A name line, then sections (a name line, "lon lat" lines, END), then END. Hole
    /// sections start with `!` and are rings like the others.
    pub fn parse_poly(text: &str) -> Result<Polygon> {
        let mut lines = text.lines().map(str::trim).filter(|l| !l.is_empty());
        lines.next().context("empty polygon file")?;
        let mut rings = Vec::new();
        loop {
            let name = lines.next().context("polygon file without the final END")?;
            if name == "END" {
                break;
            }
            let mut ring = Vec::new();
            loop {
                let l = lines.next().context("ring without END")?;
                if l == "END" {
                    break;
                }
                let mut it = l.split_whitespace().map(str::parse::<f64>);
                match (it.next(), it.next()) {
                    (Some(Ok(lon)), Some(Ok(lat))) => ring.push((lat, lon)),
                    _ => bail!("bad polygon line {l:?}"),
                }
            }
            ensure!(ring.len() >= 3, "ring {name} has fewer than 3 points");
            rings.push(ring);
        }
        ensure!(!rings.is_empty(), "polygon without a ring");
        rings.iter_mut().for_each(|r: &mut Vec<(f64, f64)>| unwrap_ring(r));
        let lo = rings.iter().flatten().map(|p| p.1).fold(f64::INFINITY, f64::min);
        let hi = rings.iter().flatten().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
        Ok(Polygon { rings, lo, hi })
    }

    /// Even-odd rule over all rings (crossings of the ray towards +lon), the point taken at
    /// each of its longitudes the rings span.
    pub fn contains(&self, lat: f64, lon: f64) -> bool {
        lon_shifts(lon, self.lo, self.hi).any(|x| self.contains_at(lat, x))
    }

    fn contains_at(&self, lat: f64, lon: f64) -> bool {
        let mut inside = false;
        for r in &self.rings {
            let mut j = r.len() - 1;
            for i in 0..r.len() {
                let ((ya, xa), (yb, xb)) = (r[i], r[j]);
                if (ya > lat) != (yb > lat) && lon < xa + (lat - ya) / (yb - ya) * (xb - xa) {
                    inside = !inside;
                }
                j = i;
            }
        }
        inside
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_with_a_hole() {
        let p = Polygon::parse_poly("x\n1\n 0 0\n 10 0\n 10 10\n 0 10\nEND\n!2\n 4 4\n 6 4\n 6 6\n 4 6\nEND\nEND\n").unwrap();
        assert!(p.contains(1.0, 1.0) && !p.contains(5.0, 5.0) && !p.contains(11.0, 1.0));
    }

    #[test]
    fn across_the_antimeridian() {
        // A box from 179°E to 179°W, written with longitudes past 180 and with wrapped ones.
        for text in ["x\n1\n 179 -1\n 181 -1\n 181 1\n 179 1\nEND\nEND\n", "x\n1\n 179 -1\n -179 -1\n -179 1\n 179 1\nEND\nEND\n"] {
            let p = Polygon::parse_poly(text).unwrap();
            assert!(p.contains(0.0, 179.5) && p.contains(0.0, -179.5) && p.contains(0.0, 180.0) && p.contains(0.0, -180.0));
            assert!(!p.contains(0.0, 178.5) && !p.contains(0.0, -178.5) && !p.contains(0.0, 0.0));
        }
    }
}
