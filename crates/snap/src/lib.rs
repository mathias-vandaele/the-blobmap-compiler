//! Snapping (SPEC §6), ported from the old `snap` crate: a sparse grid over every piece's
//! segments, the nearest point in the local tangent plane at the ID, and the fraction of
//! the piece's length before it. A query asks for the nearest piece of three nested sets:
//! the family's graph, the pieces eligible for the mode, and those it may be served on (not
//! trapped). Their absence gives the reason codes; the offsets follow SPEC §7. A piece of a
//! small component is passed over for one of a larger component within the radius (SPEC §6).

use blobmap_format::{INF, rnd};
use blobmap_osm::Pieces;
use blobmap_osm::geo::Local;
use rayon::prelude::*;

pub mod modes;
pub use modes::Membership;

/// Cell size: 0.002° (222 m north-south; 143 m east-west at 50°N).
pub const CELL_DEG: f64 = 0.002;

/// The nearest point of a piece.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hit {
    pub piece: u32,
    pub dist_m: f64,
    /// Fraction of the piece's length from u (0) to v (1); exactly 0 or 1 when the closest
    /// point is an end node.
    pub frac: f64,
}

/// The nearest piece of each set, within the radius; and the servable piece chosen.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Found {
    pub any: Option<Hit>,
    pub eligible: Option<Hit>,
    /// The nearest servable piece.
    pub nearest: Option<Hit>,
    /// The servable piece the ID stands on: the nearest one of a component with more than
    /// `small_max` nodes, or `nearest` when there is none within the radius (SPEC §6).
    pub servable: Option<Hit>,
}

/// Reason codes (FORMAT §7.3).
pub const NO_ELIGIBLE_ROAD: u8 = 1;
pub const NO_ROAD: u8 = 2;
pub const TRAPPED_COMPONENT: u8 = 3;
pub const FILTERED_PLACE: u8 = 4;

impl Found {
    /// The hit to serve on, or the reason and its distance (FORMAT §7.3).
    pub fn outcome(&self) -> Result<Hit, (u8, f32)> {
        match (self.any, self.eligible, self.servable) {
            (_, _, Some(h)) => Ok(h),
            (_, Some(e), None) => Err((TRAPPED_COMPONENT, e.dist_m as f32)),
            (Some(a), None, None) => Err((NO_ELIGIBLE_ROAD, a.dist_m as f32)),
            (None, _, _) => Err((NO_ROAD, f32::INFINITY)),
        }
    }
}

/// What a build keeps of a `Found`: whether any piece of the family's graph is within the
/// radius, the outcome, and for the data report a trapped component and the nearest servable
/// piece when a larger component won over it (`u32::MAX` for none).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Snapped {
    pub any: bool,
    /// The reason code when not served, else 0.
    pub code: u8,
    pub piece: u32,
    pub trapped: u32,
    pub moved_from: u32,
    pub reason_m: f32,
    pub frac: f64,
}

impl Snapped {
    pub fn new(f: &Found, trapped: u32) -> Snapped {
        let moved_from = match (f.nearest, f.servable) {
            (Some(n), Some(s)) if n.piece != s.piece => n.piece,
            _ => u32::MAX,
        };
        let base = Snapped { any: f.any.is_some(), code: 0, piece: u32::MAX, trapped, moved_from, reason_m: 0.0, frac: 0.0 };
        match f.outcome() {
            Ok(h) => Snapped { piece: h.piece, frac: h.frac, ..base },
            Err((code, m)) => Snapped { code, reason_m: m, ..base },
        }
    }

    /// The piece and fraction to serve on, or the reason and its distance.
    pub fn outcome(&self) -> Result<(u32, f64), (u8, f32)> {
        if self.code == 0 { Ok((self.piece, self.frac)) } else { Err((self.code, self.reason_m)) }
    }
}

/// What a mode may stand on: membership of the three sets, by piece.
pub trait Sets: Sync {
    fn in_graph(&self, piece: u32) -> bool;
    fn eligible(&self, piece: u32) -> bool;
    fn servable(&self, piece: u32) -> bool;
    /// The node count of a servable piece's component.
    fn size(&self, _piece: u32) -> u32 {
        0
    }
}

pub struct Grid {
    cells: Vec<u64>,
    first: Vec<u32>,
    /// Segment = global index i of its first point (i → i + 1 inside one piece).
    seg: Vec<u32>,
    piece_of_point: Vec<u32>,
}

fn cell_key(row: i64, col: i64) -> u64 {
    (((row + (1 << 31)) as u64) << 32) | (col + (1 << 31)) as u64
}

fn grid_units(lat: f64, lon: f64) -> (f64, f64) {
    (lat / CELL_DEG, lon / CELL_DEG)
}

/// Every grid cell the segment touches (grid units).
fn supercover(ax: f64, ay: f64, bx: f64, by: f64, mut f: impl FnMut(i64, i64)) {
    let (mut cx, mut cy) = (ax.floor() as i64, ay.floor() as i64);
    let (ex, ey) = (bx.floor() as i64, by.floor() as i64);
    let (dx, dy) = (bx - ax, by - ay);
    let (sx, sy) = (if dx > 0.0 { 1 } else { -1 }, if dy > 0.0 { 1 } else { -1 });
    let inv = |d: f64| if d == 0.0 { f64::INFINITY } else { 1.0 / d.abs() };
    let (tdx, tdy) = (inv(dx), inv(dy));
    let next = |a: f64, c: i64, s: i64| if s > 0 { (c + 1) as f64 - a } else { a - c as f64 };
    // A segment parallel to an axis never crosses a line of the other; 0 × ∞ would be NaN
    // when it starts on one.
    let cross = |d: f64, a: f64, c: i64, s: i64, td: f64| if d == 0.0 { f64::INFINITY } else { next(a, c, s) * td };
    let (mut tx, mut ty) = (cross(dx, ax, cx, sx, tdx), cross(dy, ay, cy, sy, tdy));
    f(cx, cy);
    let steps = (ex - cx).abs() + (ey - cy).abs();
    for _ in 0..steps {
        if tx < ty {
            cx += sx;
            tx += tdx;
        } else {
            cy += sy;
            ty += tdy;
        }
        f(cx, cy);
    }
}

/// Distance from the origin to the segment a-b, and the parameter t in [0, 1] of the foot.
fn seg_dist(ax: f64, ay: f64, bx: f64, by: f64) -> (f64, f64) {
    let (dx, dy) = (bx - ax, by - ay);
    let l2 = dx * dx + dy * dy;
    let t = if l2 == 0.0 { 0.0 } else { ((-ax * dx - ay * dy) / l2).clamp(0.0, 1.0) };
    ((-ax - t * dx).hypot(-ay - t * dy), t)
}

fn seg_len(p: &Pieces, i: usize) -> f64 {
    blobmap_osm::cheap_ruler_m(p.pt_lat[i], p.pt_lon[i], p.pt_lat[i + 1], p.pt_lon[i + 1])
}

/// Fraction of piece p's length before the point at parameter t of its segment i; exactly
/// 0 at the start node and 1 at the end node.
fn frac_along(p: &Pieces, piece: usize, i: usize, t: f64) -> f64 {
    let r = p.points(piece);
    if i == r.start && t == 0.0 {
        return 0.0;
    }
    if i + 2 == r.end && t == 1.0 {
        return 1.0;
    }
    let before: f64 = (r.start..i).map(|j| seg_len(p, j)).sum::<f64>() + t * seg_len(p, i);
    let total: f64 = (r.start..r.end - 1).map(|j| seg_len(p, j)).sum();
    if total == 0.0 { 0.0 } else { (before / total).clamp(0.0, 1.0) }
}

impl Grid {
    pub fn new(p: &Pieces) -> Grid {
        let mut piece_of_point = vec![0u32; p.pt_lat.len()];
        for k in 0..p.len() {
            piece_of_point[p.points(k)].fill(k as u32);
        }
        let wrap = (360.0 / CELL_DEG).round() as i64;
        let half = wrap / 2;
        let mut pairs: Vec<(u64, u32)> = (0..p.len())
            .into_par_iter()
            .flat_map_iter(|k| {
                let r = p.points(k);
                let mut out = Vec::new();
                for i in r.start..r.end - 1 {
                    let (ay, ax) = grid_units(p.pt_lat[i], p.pt_lon[i]);
                    let mut dl = p.pt_lon[i + 1] - p.pt_lon[i];
                    if dl > 180.0 {
                        dl -= 360.0;
                    } else if dl < -180.0 {
                        dl += 360.0;
                    }
                    let (by, bx) = grid_units(p.pt_lat[i + 1], p.pt_lon[i] + dl);
                    supercover(ax, ay, bx, by, |cx, cy| {
                        let cx = (cx + half).rem_euclid(wrap) - half;
                        out.push((cell_key(cy, cx), i as u32))
                    });
                }
                out
            })
            .collect();
        pairs.par_sort_unstable();
        pairs.dedup();
        let (mut cells, mut first) = (Vec::new(), Vec::new());
        for (k, &(c, _)) in pairs.iter().enumerate() {
            if cells.last() != Some(&c) {
                cells.push(c);
                first.push(k as u32);
            }
        }
        first.push(pairs.len() as u32);
        Grid { cells, first, seg: pairs.into_iter().map(|(_, i)| i).collect(), piece_of_point }
    }

    fn cell(&self, row: i64, col: i64) -> &[u32] {
        match self.cells.binary_search(&cell_key(row, col)) {
            Ok(k) => &self.seg[self.first[k] as usize..self.first[k + 1] as usize],
            Err(_) => &[],
        }
    }

    /// The nearest piece of each set within `radius_m`, ties to the smaller piece index; and
    /// the servable piece chosen, passing over components of at most `small_max` nodes
    /// (SPEC §6).
    pub fn find(&self, p: &Pieces, lat: f64, lon: f64, radius_m: f64, small_max: u32, sets: &dyn Sets) -> Found {
        let local = Local::at(lat, lon);
        let (gy, gx) = grid_units(lat, lon);
        let (cy, cx) = (gy.floor() as i64, gx.floor() as i64);
        let cell_h = CELL_DEG * local.scale().1;
        let wrap = (360.0 / CELL_DEG).round() as i64;
        let half = wrap / 2;
        let cols = |r: i64| -> i64 {
            let lat_max = (lat.abs() + (r + 1) as f64 * CELL_DEG).min(89.9);
            let w = CELL_DEG * Local::at(lat_max, 0.0).scale().0;
            ((r as f64 * cell_h / w).ceil() as i64).min(half)
        };
        let mut f = Found::default();
        let mut large: Option<Hit> = None;
        let max_ring = (radius_m / cell_h).ceil() as i64 + 1;
        let mut prev_cols = -1i64;
        for r in 0..=max_ring {
            let c = cols(r);
            for dy in -r..=r {
                for dx in -c..=c {
                    if dy.abs() < r && dx.abs() <= prev_cols {
                        continue;
                    }
                    let col = (cx + dx + half).rem_euclid(wrap) - half;
                    for &i in self.cell(cy + dy, col) {
                        let i = i as usize;
                        let k = self.piece_of_point[i];
                        if !sets.in_graph(k) {
                            continue;
                        }
                        let (ax, ay) = local.xy(p.pt_lat[i], p.pt_lon[i]);
                        let (bx, by) = local.xy(p.pt_lat[i + 1], p.pt_lon[i + 1]);
                        let (d, t) = seg_dist(ax, ay, bx, by);
                        if d > radius_m {
                            continue;
                        }
                        let better = |h: &Option<Hit>| h.is_none_or(|h| d < h.dist_m || (d == h.dist_m && k < h.piece));
                        let mk = || Some(Hit { piece: k, dist_m: d, frac: frac_along(p, k as usize, i, t) });
                        if better(&f.any) {
                            f.any = mk();
                        }
                        if sets.eligible(k) {
                            if better(&f.eligible) {
                                f.eligible = mk();
                            }
                            if sets.servable(k) {
                                if better(&f.nearest) {
                                    f.nearest = mk();
                                }
                                if sets.size(k) > small_max && better(&large) {
                                    large = mk();
                                }
                            }
                        }
                    }
                }
            }
            prev_cols = c;
            // Every piece nearer than the nearest of a large component has been met.
            if large.is_some_and(|h| h.dist_m <= r as f64 * cell_h * 0.999) {
                break;
            }
        }
        f.servable = large.or(f.nearest);
        f
    }

    /// `find` for many points, in parallel, results in input order.
    pub fn find_all(&self, p: &Pieces, points: &[(f64, f64)], radius_m: f64, small_max: u32, sets: &dyn Sets) -> Vec<Found> {
        points.par_iter().map(|&(lat, lon)| self.find(p, lat, lon, radius_m, small_max, sets)).collect()
    }
}

/// `Grid::find` without the grid: every segment of every piece, for checking the grid.
pub fn find_brute(p: &Pieces, lat: f64, lon: f64, radius_m: f64, small_max: u32, sets: &dyn Sets) -> Found {
    let local = Local::at(lat, lon);
    let mut f = Found::default();
    let mut large: Option<Hit> = None;
    for k in 0..p.len() as u32 {
        if !sets.in_graph(k) {
            continue;
        }
        let r = p.points(k as usize);
        for i in r.start..r.end - 1 {
            let (ax, ay) = local.xy(p.pt_lat[i], p.pt_lon[i]);
            let (bx, by) = local.xy(p.pt_lat[i + 1], p.pt_lon[i + 1]);
            let (d, t) = seg_dist(ax, ay, bx, by);
            if d > radius_m {
                continue;
            }
            let better = |h: &Option<Hit>| h.is_none_or(|h| d < h.dist_m);
            let mk = || Some(Hit { piece: k, dist_m: d, frac: frac_along(p, k as usize, i, t) });
            if better(&f.any) {
                f.any = mk();
            }
            if sets.eligible(k) {
                if better(&f.eligible) {
                    f.eligible = mk();
                }
                if sets.servable(k) {
                    if better(&f.nearest) {
                        f.nearest = mk();
                    }
                    if sets.size(k) > small_max && better(&large) {
                        large = mk();
                    }
                }
            }
        }
    }
    f.servable = large.or(f.nearest);
    f
}

/// The two parts of a road of weight w around an ID at fraction g of its length from the
/// road's tail (SPEC §7, FORMAT §7.2): before = rnd(w × g), after = w − before; both `INF`
/// where w is.
pub fn parts(w: u16, g: f64) -> (u16, u16) {
    if w == INF {
        return (INF, INF);
    }
    let before = (rnd(w as f64 * g) as u16).min(w);
    (before, w - before)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct All;

    impl Sets for All {
        fn in_graph(&self, _: u32) -> bool {
            true
        }
        fn eligible(&self, k: u32) -> bool {
            k != 0
        }
        fn servable(&self, k: u32) -> bool {
            k == 2
        }
    }

    fn pieces() -> Pieces {
        // Three straight pieces along a parallel at 50°N: 0 → 1 → 2 → 3, 100 m each.
        let (kx, _) = blobmap_osm::ruler(50f64.to_radians().cos());
        let lon = |x: f64| 3.0 + x / kx;
        let mut p = Pieces { pt_first: vec![0], ..Pieces::default() };
        for k in 0..3 {
            p.way.push(1);
            p.seq.push(k);
            p.u.push(k);
            p.v.push(k + 1);
            p.len_m.push(100.0);
            p.eligible.push(true);
            p.pt_lat.extend([50.0, 50.0]);
            p.pt_lon.extend([lon(100.0 * k as f64), lon(100.0 * (k + 1) as f64)]);
            p.pt_first.push(p.pt_lat.len() as u32);
        }
        p
    }

    #[test]
    fn supercover_on_a_grid_line() {
        // Horizontal and vertical segments starting exactly on a grid line stay in their row
        // or column.
        let mut cells = Vec::new();
        supercover(0.5, 0.0, 3.5, 0.0, |x, y| cells.push((x, y)));
        assert_eq!(cells, [(0, 0), (1, 0), (2, 0), (3, 0)]);
        cells.clear();
        supercover(2.0, 1.5, 2.0, -1.5, |x, y| cells.push((x, y)));
        assert_eq!(cells, [(2, 1), (2, 0), (2, -1), (2, -2)]);
    }

    #[test]
    fn nearest_by_set_and_reasons() {
        let p = pieces();
        let g = Grid::new(&p);
        let (kx, ky) = blobmap_osm::ruler(50f64.to_radians().cos());
        let at = |x: f64, y: f64| (50.0 + y / ky, 3.0 + x / kx);
        let (lat, lon) = at(30.0, 10.0);
        let f = g.find(&p, lat, lon, 500.0, 0, &All);
        assert_eq!(f.any.unwrap().piece, 0);
        assert_eq!(f.eligible.unwrap().piece, 1);
        assert_eq!(f.servable.unwrap().piece, 2);
        assert!((f.any.unwrap().frac - 0.3).abs() < 1e-6 && (f.any.unwrap().dist_m - 10.0).abs() < 1e-3);
        assert!(g.find(&p, lat, lon, 150.0, 0, &All).outcome() == Err((TRAPPED_COMPONENT, f.eligible.unwrap().dist_m as f32)));
        let (lat, lon) = at(0.0, 0.0);
        assert_eq!(g.find(&p, lat, lon, 500.0, 0, &All).any.unwrap().frac, 0.0);
        let (lat, lon) = at(5000.0, 0.0);
        assert_eq!(g.find(&p, lat, lon, 500.0, 0, &All).outcome(), Err((NO_ROAD, f32::INFINITY)));
    }

    /// Pieces 1 and 2 servable, in components of 10 and 100 nodes.
    struct TwoComponents;

    impl Sets for TwoComponents {
        fn in_graph(&self, _: u32) -> bool {
            true
        }
        fn eligible(&self, _: u32) -> bool {
            true
        }
        fn servable(&self, k: u32) -> bool {
            k > 0
        }
        fn size(&self, k: u32) -> u32 {
            if k == 1 { 10 } else { 100 }
        }
    }

    #[test]
    fn a_small_component_yields_within_the_radius() {
        let p = pieces();
        let g = Grid::new(&p);
        let (kx, ky) = blobmap_osm::ruler(50f64.to_radians().cos());
        let at = |x: f64, y: f64| (50.0 + y / ky, 3.0 + x / kx);
        // 10 m from piece 1 (10 nodes); piece 2 (100 nodes) is 61 m away, at its start.
        let (lat, lon) = at(140.0, 10.0);
        for (small_max, radius, want) in [(0, 500.0, 1), (9, 500.0, 1), (10, 500.0, 2), (99, 500.0, 2), (100, 500.0, 1), (10, 50.0, 1)] {
            let f = g.find(&p, lat, lon, radius, small_max, &TwoComponents);
            assert_eq!(f.nearest.unwrap().piece, 1);
            assert_eq!(f.servable.unwrap().piece, want, "{small_max} {radius}");
            assert_eq!(f, find_brute(&p, lat, lon, radius, small_max, &TwoComponents), "{small_max} {radius}");
            let s = Snapped::new(&f, u32::MAX);
            assert_eq!(s.moved_from, if want == 2 { 1 } else { u32::MAX });
        }
    }

    #[test]
    fn offsets() {
        assert_eq!(parts(10, 0.25), (3, 7));
        assert_eq!(parts(10, 0.0), (0, 10));
        assert_eq!(parts(10, 1.0), (10, 0));
        assert_eq!(parts(INF, 0.5), (INF, INF));
        assert_eq!(parts(5, 0.5), (3, 2));
    }
}
