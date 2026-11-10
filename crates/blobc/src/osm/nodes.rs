//! Node locations on disk (SPEC §10, memory). One pass over the `.pbf` writes every node's
//! id and packed coordinates, in id order, to two files of a work directory, and keeps the
//! obstacles each family's profile charges. A family then looks up the nodes its ways use,
//! in id order, through memory maps of the two files, a window at a time, so that resident
//! memory stays bounded whatever the size of the file.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use memmap2::{Mmap, UncheckedAdvice};
use osmpbf::{BlobDecode, BlobReader};
use rayon::prelude::*;

use crate::profile::active::ActiveProfile;
use crate::profile::car::{CarProfile, Obstacle};
use crate::profile::config::Config;

/// A node absent from the file.
pub const MISSING: u64 = u64::MAX;

/// Ids per page of the id file: the directory's step.
const STEP: usize = 512;

/// Blobs decoded at once while writing, in file order.
const BATCH: usize = 1024;
/// Ids looked up per task, and tasks per window of the maps.
const CHUNK: usize = 1 << 18;
const WINDOW: usize = 64;

const _: () = assert!(cfg!(target_endian = "little"), "the node files are read in place as little-endian");

/// (lat, lon) in decimicro-degrees, packed as the networks keep them.
pub fn pack(lat: i32, lon: i32) -> u64 {
    ((lat as u32 as u64) << 32) | lon as u32 as u64
}

pub fn unpack(c: u64) -> (f64, f64) {
    (crate::osm::degrees((c >> 32) as u32 as i32), crate::osm::degrees(c as u32 as i32))
}

/// A `.pbf` with its nodes on disk, and the obstacles of each family.
pub struct Osm {
    pub pbf: PathBuf,
    dir: PathBuf,
    ids: Mmap,
    coords: Mmap,
    n: usize,
    /// Every `STEP`-th id, so that a lookup searches one page of the id file.
    directory: Vec<i64>,
    /// Nodes with an obstacle, by OSM id, for CAR and for ACTIVE.
    pub car_obstacles: Vec<(i64, Obstacle)>,
    pub active_obstacles: Vec<(i64, Obstacle)>,
}

/// What one blob gives: its nodes in order, and their obstacles per family.
#[derive(Default)]
struct Block {
    ids: Vec<i64>,
    coords: Vec<u64>,
    car: Vec<(i64, Obstacle)>,
    active: Vec<(i64, Obstacle)>,
}

impl Osm {
    /// Reads the nodes of `pbf` into `dir` (created; any earlier files there are replaced).
    /// The file's nodes must be sorted by id, as Geofabrik extracts and the planet are.
    pub fn open(pbf: &Path, dir: &Path, config: &Config) -> Result<Osm> {
        let (car, active) = (CarProfile::new(&config.car.config), ActiveProfile::new(&config.active.config));
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        let (ids_path, coords_path) = (dir.join("node_ids.bin"), dir.join("node_coords.bin"));
        let mut ids_out = BufWriter::with_capacity(1 << 22, File::create(&ids_path)?);
        let mut coords_out = BufWriter::with_capacity(1 << 22, File::create(&coords_path)?);
        let (mut n, mut last) = (0usize, i64::MIN);
        let mut directory = Vec::new();
        let (mut car_obstacles, mut active_obstacles) = (Vec::new(), Vec::new());
        let mut blobs = BlobReader::from_path(pbf).with_context(|| format!("opening {}", pbf.display()))?;
        loop {
            let batch: Vec<_> = blobs.by_ref().take(BATCH).collect::<std::result::Result<_, _>>()?;
            if batch.is_empty() {
                break;
            }
            let blocks: Vec<Block> = batch
                .into_par_iter()
                .map(|blob| -> Result<Block> {
                    let mut b = Block::default();
                    if let BlobDecode::OsmData(block) = blob.decode()? {
                        // An untagged node (most of them) allocates nothing here.
                        let mut node = |id: i64, lat: i32, lon: i32, tags: Vec<(&str, &str)>| {
                            b.ids.push(id);
                            b.coords.push(pack(lat, lon));
                            if !tags.is_empty() {
                                if let Some(o) = car.obstacle(tags.iter().copied()) {
                                    b.car.push((id, o));
                                }
                                if let Some(o) = active.obstacle(tags.iter().copied()) {
                                    b.active.push((id, o));
                                }
                            }
                        };
                        for g in block.groups() {
                            for x in g.dense_nodes() {
                                node(x.id(), x.decimicro_lat(), x.decimicro_lon(), x.tags().collect());
                            }
                            for x in g.nodes() {
                                node(x.id(), x.decimicro_lat(), x.decimicro_lon(), x.tags().collect());
                            }
                        }
                    }
                    Ok(b)
                })
                .collect::<Result<_>>()?;
            for b in blocks {
                for (&id, &c) in b.ids.iter().zip(&b.coords) {
                    ensure!(id > last, "{}: nodes are not sorted by id (node {id} after {last}); sort the file first (osmium sort)", pbf.display());
                    last = id;
                    if n.is_multiple_of(STEP) {
                        directory.push(id);
                    }
                    n += 1;
                    ids_out.write_all(&id.to_le_bytes())?;
                    coords_out.write_all(&c.to_le_bytes())?;
                }
                car_obstacles.extend(b.car);
                active_obstacles.extend(b.active);
            }
        }
        ids_out.flush()?;
        coords_out.flush()?;
        drop((ids_out, coords_out));
        // SAFETY: the files were written above, are owned by this process's work directory,
        // and are not changed while mapped.
        let (ids, coords) = unsafe { (Mmap::map(&File::open(&ids_path)?)?, Mmap::map(&File::open(&coords_path)?)?) };
        Ok(Osm { pbf: pbf.to_path_buf(), dir: dir.to_path_buf(), ids, coords, n, directory, car_obstacles, active_obstacles })
    }

    /// The work directory, where the families' columns go too.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Nodes in the file.
    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    fn id_slice(&self) -> &[i64] {
        if self.n == 0 {
            return &[];
        }
        // SAFETY: the map holds `n` little-endian i64 written by `open`, and is page-aligned.
        unsafe { std::slice::from_raw_parts(self.ids.as_ptr() as *const i64, self.n) }
    }

    fn coord_slice(&self) -> &[u64] {
        if self.n == 0 {
            return &[];
        }
        // SAFETY: as `id_slice`, for the coordinates.
        unsafe { std::slice::from_raw_parts(self.coords.as_ptr() as *const u64, self.n) }
    }

    /// The packed coordinates of one node, `MISSING` if it is not in the file: one page of
    /// the id file searched, found by the directory.
    pub fn coord(&self, id: i64) -> u64 {
        let k = self.directory.partition_point(|&x| x <= id);
        if k == 0 {
            return MISSING;
        }
        let lo = (k - 1) * STEP;
        let all = self.id_slice();
        let page = &all[lo..(lo + STEP).min(self.n)];
        match page.binary_search(&id) {
            Ok(j) => self.coord_slice()[lo + j],
            Err(_) => MISSING,
        }
    }

    /// Drops every mapped page from the resident set (they stay in the page cache), as a
    /// long run of `coord` lookups goes on.
    pub fn release_all(&self) {
        self.release(0, self.n);
    }

    /// The packed coordinates of sorted, distinct `ids`, `MISSING` for a node not in the file.
    pub fn coords(&self, ids: &[i64]) -> Vec<u64> {
        let (all, at) = (self.id_slice(), self.coord_slice());
        let mut out = vec![MISSING; ids.len()];
        for (window, o) in ids.chunks(CHUNK * WINDOW).zip(out.chunks_mut(CHUNK * WINDOW)) {
            window.par_chunks(CHUNK).zip(o.par_chunks_mut(CHUNK)).for_each(|(q, o)| {
                let mut k = all.partition_point(|&x| x < q[0]);
                for (id, c) in q.iter().zip(o) {
                    while k < all.len() && all[k] < *id {
                        k += 1;
                    }
                    if k < all.len() && all[k] == *id {
                        *c = at[k];
                    }
                }
            });
            // The window's pages leave this process (they stay in the page cache).
            let lo = all.partition_point(|&x| x < window[0]);
            let hi = all.partition_point(|&x| x <= *window.last().unwrap());
            self.release(lo, hi);
        }
        out
    }

    /// Drops the mapped pages of entries `lo .. hi` from the resident set.
    fn release(&self, lo: usize, hi: usize) {
        const PAGE: usize = 4096;
        for m in [&self.ids, &self.coords] {
            let (a, b) = ((lo * 8) / PAGE * PAGE, (hi * 8).div_ceil(PAGE) * PAGE);
            let b = b.min(m.len());
            if b > a {
                // SAFETY: the maps are read-only views of files; dropping their pages only
                // makes later reads fault them back in from the page cache.
                let _ = unsafe { m.unchecked_advise_range(UncheckedAdvice::DontNeed, a, b - a) };
            }
        }
    }
}

impl Drop for Osm {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(self.dir.join("node_ids.bin"));
        let _ = std::fs::remove_file(self.dir.join("node_coords.bin"));
        let _ = std::fs::remove_dir(&self.dir);
    }
}
