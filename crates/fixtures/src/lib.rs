//! Synthetic inputs for the tiny tier (PLAN milestone 2): real `.osm.pbf` files, Overture
//! parquet and taxonomy CSVs, and config directories, so that nothing in the pipeline is
//! test-only. Each scenario also lists facts the conformance suite checks (`Check`).

pub mod build;
pub mod overture;
pub mod pbf;
pub mod scenarios;

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

/// The Overture release name of every scenario (its directory name).
pub const RELEASE: &str = "tiny-2026-11";

/// The repository's config files; every scenario uses them unchanged.
pub const CAR_TOML: &str = include_str!("../../../config/car.toml");
pub const ACTIVE_TOML: &str = include_str!("../../../config/active.toml");
pub const PLACES_TOML: &str = include_str!("../../../config/places.toml");

/// One way of building a scenario: its `build.toml` (build options never change answers).
pub struct Variant {
    pub name: &'static str,
    pub build_toml: String,
}

/// A fact about a scenario's model, beyond the suite's general checks (SPEC §11).
#[derive(Clone, Debug)]
pub enum Check {
    /// Component `component` of `family` has these links and levels (the toy town).
    Hierarchy {
        family: &'static str,
        component: u32,
        links: u64,
        levels: u32,
    },
    /// `travel_time(ids[i], ids[j])` is `times[i][j]`.
    Grid {
        mode: &'static str,
        ids: Vec<String>,
        times: Vec<Vec<u16>>,
    },
    /// The true time a → b is below the cap (`under`), or at or above it.
    Cap {
        mode: &'static str,
        a: String,
        b: String,
        under: bool,
    },
    Reason {
        id: String,
        mode: &'static str,
        reason: &'static str,
    },
    Served {
        id: String,
        mode: &'static str,
    },
    Alias {
        alias: String,
        kept: String,
    },
    Unknown {
        id: String,
    },
    DifferentComponent {
        mode: &'static str,
        a: String,
        b: String,
    },
    /// Same component, but no route for the mode.
    Beyond {
        mode: &'static str,
        a: String,
        b: String,
    },
    Reachable {
        mode: &'static str,
        a: String,
        b: String,
    },
    /// `nearest(category, id, mode, from)` is an equal-time tie won by `place`.
    Tie {
        category: &'static str,
        id: String,
        mode: &'static str,
        place: String,
    },
    SamePiece {
        mode: &'static str,
        a: String,
        b: String,
    },
    /// The place matches the taxonomy category (its own keys went through the redirects).
    Matches {
        category: &'static str,
        place: String,
    },
    AtNode {
        id: String,
        mode: &'static str,
    },
    Trapped {
        family: &'static str,
        at_least: usize,
    },
    /// At least this many via-way restrictions applied.
    Restrictions {
        at_least: usize,
    },
    /// At least this many via-way restrictions have a prefix copy with no exit (SPEC §5).
    DeadEnds {
        at_least: usize,
    },
    /// The ID stands on a segment that has via-way copies.
    OnVia {
        id: String,
    },
    /// Built with the `cells` variant, the family's component 0 is built as cells.
    Cells {
        family: &'static str,
    },
}

pub struct Scenario {
    pub name: &'static str,
    pub osm: pbf::OsmData,
    pub release: overture::Release,
    /// An Osmosis `.poly` file, passed as `--extent`.
    pub extent: Option<String>,
    pub variants: Vec<Variant>,
    pub checks: Vec<Check>,
}

/// Where a written scenario's inputs are.
pub struct Written {
    pub pbf: PathBuf,
    pub overture: PathBuf,
    /// One config directory per variant, in variant order.
    pub configs: Vec<(String, PathBuf)>,
    pub extent: Option<PathBuf>,
}

impl Scenario {
    pub fn pbf_bytes(&self) -> Result<Vec<u8>> {
        self.osm.to_bytes()
    }

    /// Writes the inputs under `dir/<name>/`.
    pub fn write(&self, dir: &Path) -> Result<Written> {
        let root = dir.join(self.name);
        std::fs::create_dir_all(&root).with_context(|| format!("creating {}", root.display()))?;
        let pbf = root.join(format!("{}.osm.pbf", self.name));
        self.osm.write(&pbf)?;
        let overture = root.join("overture").join(RELEASE);
        self.release.write(&overture)?;
        let mut configs = Vec::new();
        for v in &self.variants {
            let c = root.join(if v.name == "default" { "config".to_string() } else { format!("config-{}", v.name) });
            std::fs::create_dir_all(&c)?;
            std::fs::write(c.join("car.toml"), CAR_TOML)?;
            std::fs::write(c.join("active.toml"), ACTIVE_TOML)?;
            std::fs::write(c.join("places.toml"), PLACES_TOML)?;
            std::fs::write(c.join("build.toml"), &v.build_toml)?;
            configs.push((v.name.to_string(), c));
        }
        let extent = match &self.extent {
            Some(text) => {
                let p = root.join(format!("{}.poly", self.name));
                std::fs::write(&p, text)?;
                Some(p)
            }
            None => None,
        };
        Ok(Written { pbf, overture, configs, extent })
    }
}

pub fn by_name(name: &str) -> Option<Scenario> {
    scenarios::all().into_iter().find(|s| s.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pbf_reads_back_with_osmpbf() {
        for s in scenarios::all() {
            let bytes = s.pbf_bytes().unwrap();
            let r = osmpbf::ElementReader::new(std::io::Cursor::new(bytes));
            let (mut nodes, mut ways, mut rels) = (Vec::new(), Vec::new(), Vec::new());
            r.for_each(|e| match e {
                osmpbf::Element::DenseNode(n) => nodes.push((n.id(), n.decimicro_lat(), n.decimicro_lon(), n.tags().count())),
                osmpbf::Element::Node(n) => nodes.push((n.id(), n.decimicro_lat(), n.decimicro_lon(), n.tags().count())),
                osmpbf::Element::Way(w) => ways.push((w.id(), w.refs().collect::<Vec<_>>(), w.tags().count())),
                osmpbf::Element::Relation(r) => rels.push((r.id(), r.members().count(), r.tags().count())),
            })
            .unwrap();
            assert_eq!(nodes.len(), s.osm.nodes.len(), "{}", s.name);
            for (n, m) in nodes.iter().zip(&s.osm.nodes) {
                assert_eq!((n.0, n.1, n.2, n.3), (m.id, (m.lat * 1e7).round() as i32, (m.lon * 1e7).round() as i32, m.tags.len()));
            }
            assert_eq!(ways.len(), s.osm.ways.len());
            for (w, m) in ways.iter().zip(&s.osm.ways) {
                assert_eq!((w.0, &w.1, w.2), (m.id, &m.refs, m.tags.len()));
            }
            assert_eq!(rels.len(), s.osm.relations.len());
            let header = osmpbf::BlobReader::new(std::io::Cursor::new(s.pbf_bytes().unwrap())).next().unwrap().unwrap();
            assert_eq!(header.to_headerblock().unwrap().osmosis_replication_timestamp(), s.osm.timestamp);
        }
    }

    #[test]
    fn same_scenario_same_bytes() {
        assert_eq!(by_name("town").unwrap().pbf_bytes().unwrap(), by_name("town").unwrap().pbf_bytes().unwrap());
    }

    #[test]
    fn parquet_reads_back() {
        use parquet::file::reader::{FileReader, SerializedFileReader};
        let dir = std::env::temp_dir().join(format!("blobmap-fixtures-{}", std::process::id()));
        let w = by_name("town").unwrap().write(&dir).unwrap();
        let f = std::fs::File::open(w.overture.join("raw/place.parquet")).unwrap();
        let r = SerializedFileReader::new(f).unwrap();
        let cols: Vec<String> = r.metadata().file_metadata().schema_descr().columns().iter().map(|c| c.path().string()).collect();
        for c in ["id", "geometry", "confidence", "names.primary", "operating_status", "basic_category", "taxonomy.primary"] {
            assert!(cols.contains(&c.to_string()), "{c} in {cols:?}");
        }
        assert!(r.metadata().file_metadata().num_rows() > 10);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
