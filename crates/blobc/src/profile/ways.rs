//! What a network needs of a profile (research `extract.rs`): the ways it keeps, their
//! per-direction attributes, node obstacles, and per-mode delays.

use crate::profile::car::{CarProfile, Obstacle, ObstacleDir, WayAttrs};

/// Which ways a network keeps and what it records of them (`Attrs`), and where traffic keeps
/// left on them. (The node obstacles come with the node locations, `osm::Osm`.)
pub trait WayProfile: Sync {
    type Attrs: Copy + Send + Sync + Speeds;
    fn way_attrs<'a>(&self, tags: impl Iterator<Item = (&'a str, &'a str)>) -> Option<Self::Attrs>;
    /// Whether traffic keeps left on a way, from its tags alone: None when its location
    /// decides (`keeps_left_at` of its last node). A profile without turn costs keeps right.
    fn driving_side(&self, _tags: &[(&str, &str)]) -> Option<bool> {
        Some(false)
    }
    fn keeps_left_at(&self, _lat: f64, _lon: f64) -> bool {
        false
    }
}

/// One mode of a network (France ACTIVE phase, Step 2): the attributes of a way it uses,
/// and what an obstacle on its way costs it. A closure `Fn(&A) -> WayAttrs` is a mode with
/// the car's rule: OSRM's obstacle delay, in the obstacle's direction.
pub trait NetworkMode<A> {
    fn attrs(&self, a: &A) -> WayAttrs;
    /// Delay (s) of `o` for a traversal of its way along (`forward`) or against the node
    /// order; None = not charged.
    fn delay(&self, o: &Obstacle, forward: bool) -> Option<f64> {
        car_delay(o, forward)
    }
}

impl<A, F: Fn(&A) -> WayAttrs> NetworkMode<A> for F {
    fn attrs(&self, a: &A) -> WayAttrs {
        self(a)
    }
}

/// The car's rule: OSRM's duration, only in the obstacle's direction if it has one.
pub fn car_delay(o: &Obstacle, forward: bool) -> Option<f64> {
    let applies = match o.dir {
        ObstacleDir::Any => true,
        ObstacleDir::Forward => forward,
        ObstacleDir::Backward => !forward,
    };
    applies.then_some(o.duration_s as f64)
}

/// The speed a way is used at in each direction (for the obstacle rules).
pub trait Speeds {
    fn speeds(&self) -> (f32, f32);
}

impl Speeds for WayAttrs {
    fn speeds(&self) -> (f32, f32) {
        (self.speed_fwd_kmh, self.speed_bwd_kmh)
    }
}

impl WayProfile for CarProfile {
    type Attrs = WayAttrs;
    fn way_attrs<'a>(&self, tags: impl Iterator<Item = (&'a str, &'a str)>) -> Option<WayAttrs> {
        CarProfile::way_attrs(self, tags)
    }
    fn driving_side(&self, tags: &[(&str, &str)]) -> Option<bool> {
        CarProfile::driving_side(self, tags)
    }
    fn keeps_left_at(&self, lat: f64, lon: f64) -> bool {
        self.left_polygons.contains(lat, lon)
    }
}
