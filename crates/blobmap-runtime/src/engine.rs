//! Where sweeps run: on the CPU, inline, or on a GPU through one worker thread that owns the
//! device. Requests queue there; the worker takes the oldest and every queued request on the
//! same component, metric and direction, up to `b_max` columns, and runs them as one
//! batched sweep. Components load onto the device on first use, inside the VRAM budget,
//! evicting the least recently used ones.

use std::collections::{BTreeMap, VecDeque};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex, OnceLock};

use anyhow::{Result, anyhow, bail};

use crate::gpu::{DevGraph, DevHeads, DevIds, Gpu, HeadCol, Kind};
use crate::graph::Graph;
use crate::heads::{Labels, Times};
use crate::matrix::{DevMatrix, MatrixIn, MatrixOut};
use crate::sweep::{Bounds, Column, Key, all_bounds, cpu_down};
use crate::vocab::{Dir, Mode};

/// A loaded component: its graph, its entry bounds once a pruned CPU sweep needs them, and
/// per mode the IDs of a small component once its heads need them.
pub struct Comp {
    pub g: Graph,
    pub bounds: OnceLock<Result<Bounds, String>>,
    pub members: Mutex<BTreeMap<usize, Arc<Vec<u32>>>>,
}

/// A component of fewer than 1/this of its family's nodes keeps its IDs' list; a larger one's
/// heads scan the IDs' layout ids, which they read anyway.
const SMALL: usize = 16;

impl Comp {
    pub fn new(g: Graph) -> Comp {
        Comp { g, bounds: OnceLock::new(), members: Mutex::new(BTreeMap::new()) }
    }

    /// The IDs of component x in mode `mode`, ascending, for a small component (found by one
    /// scan of the mode's IDs, then kept); `None` for a large one.
    pub fn members(&self, mode: usize, mv: &Mode, x: u32, family_nodes: usize) -> Option<Arc<Vec<u32>>> {
        if self.g.n * SMALL >= family_nodes {
            return None;
        }
        if let Some(v) = self.members.lock().unwrap().get(&mode) {
            return Some(v.clone());
        }
        use rayon::prelude::*;
        let (a, b) = (mv.first(x), mv.first(x) + self.g.n as u32);
        let ids: Vec<u32> = (0..mv.len() as u32)
            .into_par_iter()
            .with_min_len(1 << 16)
            .filter(|&i| {
                let nd = mv.raw_node(i as usize);
                let v = if nd[0] != u32::MAX { nd[0] } else { nd[1] };
                v >= a && v < b
            })
            .collect();
        let ids = Arc::new(ids);
        self.members.lock().unwrap().insert(mode, ids.clone());
        Some(ids)
    }

    pub fn bounds(&self) -> Result<&Bounds> {
        self.bounds.get_or_init(|| all_bounds(&self.g).map_err(|e| format!("{e:#}"))).as_ref().map_err(|e| anyhow!("{e}"))
    }
}

pub fn cpu_sweep(c: &Comp, metric: usize, dir: Dir, col: &Column, prune: bool) -> Result<Vec<u16>> {
    let b = if prune && col.pruned() { Some(&c.bounds()?[2 * metric + crate::sweep::dir_index(dir)][..]) } else { None };
    Ok(cpu_down(&c.g, metric, dir, col, b))
}

/// A column's result from the worker: its heads, computed on the device, or its labels for
/// the host to compute them.
pub enum ColOut {
    Times(Times),
    Labels(Labels),
    /// Labels at the new ids the job listed for this column.
    Gathered(Vec<u16>),
}

type Reply = mpsc::Sender<Result<Vec<ColOut>, String>>;

/// A mode borrowed by the worker. Safety: made only by `Runtime`, whose `gpu` (the worker,
/// joined on drop) is dropped before its vocabulary.
#[derive(Clone, Copy)]
pub struct ModeRef(pub &'static Mode);

/// What the device needs for a job's heads: the mode, the component's first layout id,
/// and each column's sources and wants.
pub struct JobHeads {
    pub mode: ModeRef,
    pub first: u32,
    pub cols: Vec<HeadCol>,
}

struct Job {
    key: Key,
    comp: Arc<Comp>,
    cols: Vec<Column>,
    heads: Option<JobHeads>,
    /// Per column, the new ids whose labels alone come back (instead of heads or labels).
    gather: Option<Vec<Vec<u32>>>,
    reply: Reply,
}

/// Columns at one list of targets for many sources, as one device job (`matrix`).
struct MatJob {
    key: Key,
    comp: Arc<Comp>,
    input: MatrixIn,
    timed: bool,
    reply: mpsc::Sender<Result<MatrixOut, String>>,
}

#[derive(Default)]
struct Queue {
    jobs: VecDeque<Job>,
    mats: VecDeque<MatJob>,
    stop: bool,
}

pub struct GpuEngine {
    q: Arc<(Mutex<Queue>, Condvar)>,
    worker: Option<std::thread::JoinHandle<()>>,
    pub kind: Kind,
    pub describe: String,
    pub stats: Arc<Mutex<Stats>>,
}

#[derive(Default, Clone, Debug)]
pub struct Stats {
    pub sweeps: u64,
    pub columns: u64,
    pub loads: u64,
    pub evictions: u64,
    pub vram_used: u64,
    pub vram_peak: u64,
    /// Seconds on the worker: uploads (with entry bounds), seeding and launches up to a
    /// device sync, label read-back.
    pub load_s: f64,
    pub down_s: f64,
    pub read_s: f64,
    /// Batches whose heads ran on the device (the others' labels were read back), the
    /// seconds of those heads with their read-back, and the ID tables' uploads.
    pub dev_heads: u64,
    pub heads_s: f64,
    pub id_loads: u64,
    /// Batches run again after a device out of memory, everything evicted first.
    pub oom_retries: u64,
    /// Matrix jobs, their hierarchies' uploads, and their seconds on the worker.
    pub matrices: u64,
    pub matrix_loads: u64,
    pub matrix_s: f64,
}

pub struct GpuConfig {
    pub kind: Kind,
    pub b_max: u32,
    pub budget: u64,
    pub prune: bool,
}

impl GpuEngine {
    pub fn start(cfg: GpuConfig) -> Result<GpuEngine> {
        let q: Arc<(Mutex<Queue>, Condvar)> = Arc::default();
        let stats: Arc<Mutex<Stats>> = Arc::default();
        let (tx, rx) = mpsc::channel::<Result<String, String>>();
        let (q2, s2) = (q.clone(), stats.clone());
        let kind = cfg.kind;
        let worker = std::thread::Builder::new()
            .name(format!("blobmap-gpu-{kind:?}"))
            .spawn(move || {
                let mut gpu = match Gpu::open(cfg.kind) {
                    Ok(g) => g,
                    Err(e) => {
                        let _ = tx.send(Err(format!("{e:#}")));
                        return;
                    }
                };
                // Phase times for Stats: one device sync a sweep, which the read-back waits
                // for anyway.
                gpu.timing = true;
                let _ = tx.send(Ok(gpu.describe()));
                run(gpu, &cfg, &q2, &s2);
            })?;
        let describe = rx.recv().map_err(|_| anyhow!("the GPU worker ended"))?.map_err(|e| anyhow!("{e}"))?;
        Ok(GpuEngine { q, worker: Some(worker), kind, describe, stats })
    }

    /// Queues a sweep; the receiver gets each column's heads or labels.
    pub fn submit(&self, key: Key, comp: Arc<Comp>, cols: Vec<Column>, heads: Option<JobHeads>, gather: Option<Vec<Vec<u32>>>) -> mpsc::Receiver<Result<Vec<ColOut>, String>> {
        let (tx, rx) = mpsc::channel();
        let (m, cv) = &*self.q;
        m.lock().unwrap().jobs.push_back(Job { key, comp, cols, heads, gather, reply: tx });
        cv.notify_one();
        rx
    }
}

impl GpuEngine {
    /// Queues a matrix job (it runs before queued sweeps); `timed` syncs between phases.
    pub fn matrix(&self, key: Key, comp: Arc<Comp>, input: MatrixIn, timed: bool) -> mpsc::Receiver<Result<MatrixOut, String>> {
        let (tx, rx) = mpsc::channel();
        let (m, cv) = &*self.q;
        m.lock().unwrap().mats.push_back(MatJob { key, comp, input, timed, reply: tx });
        cv.notify_one();
        rx
    }
}

impl Drop for GpuEngine {
    fn drop(&mut self) {
        let (m, cv) = &*self.q;
        m.lock().unwrap().stop = true;
        cv.notify_all();
        if let Some(w) = self.worker.take() {
            let _ = w.join();
        }
    }
}

struct Loaded {
    dev: DevGraph,
    heads: Option<DevHeads>,
    used_at: u64,
}

/// ID tables unused for this many batches give way to another one.
const STALE: u64 = 64;

/// A device out of memory, as cuda (at the allocation) or wgpu (at the read of a buffer
/// whose allocation failed) reports it through CubeCL.
fn oom(msg: &str) -> bool {
    msg.contains("out of device memory") || msg.contains("The bytes were never written")
}

/// What CubeCL's pools hold on the device: the buffers in use, their padding, and the
/// pages freed buffers left in the pools. The budget's own accounting counts only the
/// first; a sweep's transient buffers and the pools' slack count here too.
fn reserved(gpu: &Gpu) -> u64 {
    gpu.client.memory_report(cubecl_runtime::memory_management::MemoryScope::Device).usage().bytes_reserved
}

#[allow(clippy::map_entry)]
fn run(gpu: Gpu, cfg: &GpuConfig, q: &(Mutex<Queue>, Condvar), stats: &Mutex<Stats>) {
    let p_max = cfg.b_max.div_ceil(2).max(1);
    let mut loaded: BTreeMap<(u8, u32), Loaded> = BTreeMap::new();
    // Per mode (family, metric), its ID table.
    let mut ids: BTreeMap<(u8, u8), (DevIds, u64)> = BTreeMap::new();
    // Per component, its hierarchy for matrix jobs.
    let mut mats: BTreeMap<(u8, u32), (DevMatrix, u64)> = BTreeMap::new();
    let mut used = 0u64;
    let mut clock = 0u64;
    let sync_cleanup = |gpu: &Gpu| -> Result<()> {
        cubecl::future::block_on(gpu.client.sync()).map_err(|e| anyhow!("sync: {e:?}"))?;
        gpu.client.memory_cleanup().map_err(|e| anyhow!("memory cleanup: {e:?}"))
    };
    loop {
        // A matrix job first; else the oldest job and every queued one with its key, up to
        // b_max columns.
        let batch: Vec<Job> = {
            let (m, cv) = q;
            let mut g = m.lock().unwrap();
            while g.jobs.is_empty() && g.mats.is_empty() && !g.stop {
                g = cv.wait(g).unwrap();
            }
            if let Some(mj) = g.mats.pop_front() {
                drop(g);
                clock += 1;
                let t = std::time::Instant::now();
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<MatrixOut> {
                    let ck = (mj.key.fam, mj.key.component);
                    let g = &mj.comp.g;
                    if !mats.contains_key(&ck) {
                        let need = DevMatrix::estimate(g) + g.links() as u64 * 4;
                        while used.max(reserved(&gpu)) + need > cfg.budget {
                            if let Some(k) = ids.iter().min_by_key(|(_, t)| t.1).map(|(&k, _)| k) {
                                used -= ids.remove(&k).unwrap().0.bytes;
                                sync_cleanup(&gpu)?;
                                continue;
                            }
                            let Some((k, _)) = loaded.iter().min_by_key(|(_, l)| l.used_at).map(|(&k, l)| (k, l.used_at)) else { break };
                            let l = loaded.remove(&k).unwrap();
                            used -= l.dev.bytes + l.heads.as_ref().map_or(0, |h| h.bytes);
                            stats.lock().unwrap().evictions += 1;
                            sync_cleanup(&gpu)?;
                        }
                        if used + need > cfg.budget {
                            bail!("component {}'s hierarchy for matrices needs {:.2} GiB on the device, over the {:.2} GiB budget", mj.key.component, need as f64 / (1u64 << 30) as f64, cfg.budget as f64 / (1u64 << 30) as f64);
                        }
                        let dm = gpu.load_matrix(g)?;
                        used += need;
                        let mut s = stats.lock().unwrap();
                        s.matrix_loads += 1;
                        s.load_s += t.elapsed().as_secs_f64();
                        s.vram_used = used;
                        s.vram_peak = s.vram_peak.max(used);
                        mats.insert(ck, (dm, clock));
                    }
                    let (dm, at) = mats.get_mut(&ck).unwrap();
                    *at = clock;
                    let dir = if mj.key.dir == 0 { Dir::From } else { Dir::To };
                    gpu.matrix(dm, g, mj.key.metric as usize, dir, &mj.input, mj.timed)
                }))
                .unwrap_or_else(|e| Err(anyhow!("GPU panic: {}", e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default())));
                let mut s = stats.lock().unwrap();
                s.matrices += 1;
                s.matrix_s += t.elapsed().as_secs_f64();
                drop(s);
                let _ = mj.reply.send(r.map_err(|e| format!("{e:#}")));
                continue;
            }
            if g.jobs.is_empty() {
                // Free the device buffers and wait for the device, so that nothing is left
                // to free after the process starts to exit.
                drop(g);
                loaded.clear();
                ids.clear();
                mats.clear();
                let _ = cubecl::future::block_on(gpu.client.sync());
                return;
            }
            let first = g.jobs.pop_front().unwrap();
            let key = first.key;
            let mut cols = first.cols.len();
            let mut batch = vec![first];
            let mut i = 0;
            while i < g.jobs.len() {
                // One batch either gathers labels for every column or for none.
                if g.jobs[i].key == key && g.jobs[i].gather.is_some() == batch[0].gather.is_some() && cols + g.jobs[i].cols.len() <= 2 * p_max as usize {
                    cols += g.jobs[i].cols.len();
                    batch.push(g.jobs.remove(i).unwrap());
                } else {
                    i += 1;
                }
            }
            batch
        };
        let key = batch[0].key;
        clock += 1;
        // A panic in CubeCL or a kernel becomes this batch's error: the worker lives on, so
        // that no later request waits on a queue nobody drains.
        // A device out of memory beyond the budget's accounting (the driver's context, other
        // processes, the pool's slack): everything leaves the device and the batch runs once
        // more with only what it needs.
        let mut tries = 0;
        let r = loop {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<Vec<ColOut>> {
            let ck = (key.fam, key.component);
            let mk = (key.fam, key.metric);
            // The pools grown past the budget by earlier sweeps' transient buffers give their
            // free pages back before this one allocates.
            if reserved(&gpu) > cfg.budget {
                sync_cleanup(&gpu)?;
            }
            if !loaded.contains_key(&ck) {
                let g = &batch[0].comp.g;
                let need = Gpu::estimate(g, p_max, cfg.prune);
                // ID tables go first, the least recently used first (an ID table is an
                // optional accelerator, reloaded in a second, where a component takes many:
                // under LRU alone, Europe active evicted its 13 GB main component 608 times
                // for small ones); then components, the least recently used first.
                while used.max(reserved(&gpu)) + need > cfg.budget {
                    if let Some(k) = ids.iter().min_by_key(|(_, t)| t.1).map(|(&k, _)| k) {
                        used -= ids.remove(&k).unwrap().0.bytes;
                    } else if let Some(k) = loaded.iter().min_by_key(|(_, l)| l.used_at).map(|(&k, _)| k) {
                        let l = loaded.remove(&k).unwrap();
                        used -= l.dev.bytes + l.heads.as_ref().map_or(0, |h| h.bytes);
                    } else {
                        break;
                    }
                    stats.lock().unwrap().evictions += 1;
                    // Give the evicted buffers back to the device, not to CubeCL's pool.
                    sync_cleanup(&gpu)?;
                }
                if used + need > cfg.budget {
                    bail!(
                        "component {} ({} nodes, {} links) needs {:.2} GiB on the device, over the {:.2} GiB budget by {:.2} GiB",
                        key.component,
                        g.n,
                        g.links(),
                        need as f64 / (1u64 << 30) as f64,
                        cfg.budget as f64 / (1u64 << 30) as f64,
                        (used + need - cfg.budget) as f64 / (1u64 << 30) as f64
                    );
                }
                let t = std::time::Instant::now();
                let bounds = if cfg.prune { Some(all_bounds(g)?) } else { None };
                let dev = gpu.load(g, bounds.as_ref(), p_max)?;
                used += dev.bytes;
                // Its heads data, when it fits beside what is there.
                let heads = match &batch[0].heads {
                    Some(h) if used + Gpu::heads_bytes(g, p_max) <= cfg.budget => {
                        let d = gpu.load_heads(g, h.first, p_max)?;
                        used += d.bytes;
                        Some(d)
                    }
                    _ => None,
                };
                let mut s = stats.lock().unwrap();
                s.load_s += t.elapsed().as_secs_f64();
                s.loads += 1;
                s.vram_used = used;
                s.vram_peak = s.vram_peak.max(used);
                loaded.insert(ck, Loaded { dev, heads, used_at: clock });
            }
            // The mode's ID table, when every job wants device heads and the component has
            // its heads data: kept, or loaded where it fits, after stale tables leave.
            let want_heads = batch.iter().all(|j| j.heads.is_some()) && loaded[&ck].heads.is_some();
            if want_heads && !ids.contains_key(&mk) {
                let mv = batch[0].heads.as_ref().unwrap().mode.0;
                let need = Gpu::ids_bytes(mv.len());
                let stale: Vec<(u8, u8)> = ids.iter().filter(|(_, t)| t.1 + STALE < clock).map(|(&k, _)| k).collect();
                for k in stale {
                    if used + need <= cfg.budget {
                        break;
                    }
                    used -= ids.remove(&k).unwrap().0.bytes;
                    sync_cleanup(&gpu)?;
                }
                if used + need <= cfg.budget {
                    let t = std::time::Instant::now();
                    let d = gpu.load_ids(mv)?;
                    used += d.bytes;
                    let mut s = stats.lock().unwrap();
                    s.load_s += t.elapsed().as_secs_f64();
                    s.id_loads += 1;
                    s.vram_used = used;
                    s.vram_peak = s.vram_peak.max(used);
                    ids.insert(mk, (d, clock));
                }
            }
            let l = loaded.get_mut(&ck).unwrap();
            l.used_at = clock;
            let cols: Vec<&Column> = batch.iter().flat_map(|j| j.cols.iter()).collect();
            let dir = if key.dir == 0 { Dir::From } else { Dir::To };
            let prune = gpu.sweep(&mut l.dev, key.metric as usize, dir, &cols)?;
            let [down, _] = gpu.times.get();
            let mut out = None;
            if batch[0].gather.is_some() {
                let lists: Vec<&[u32]> = batch.iter().flat_map(|j| j.gather.as_ref().unwrap().iter().map(|l| &l[..])).collect();
                let got = gpu.gather(&l.dev, &lists)?;
                stats.lock().unwrap().read_s += gpu.times.get()[1];
                out = Some(got.into_iter().map(ColOut::Gathered).collect());
            } else if want_heads && let Some((dids, at)) = ids.get_mut(&mk) {
                *at = clock;
                let t = std::time::Instant::now();
                let hc: Vec<&HeadCol> = batch.iter().flat_map(|j| j.heads.as_ref().unwrap().cols.iter()).collect();
                let car = key.fam == crate::vocab::Fam::Car as u8;
                if let Some(times) = gpu.heads(&l.dev, l.heads.as_ref().unwrap(), dids, car, dir, prune, &hc)? {
                    out = Some(times.into_iter().map(ColOut::Times).collect());
                    let mut s = stats.lock().unwrap();
                    s.dev_heads += 1;
                    s.heads_s += t.elapsed().as_secs_f64();
                }
            }
            let out = match out {
                Some(o) => o,
                None => {
                    let labels = gpu.labels(&l.dev, cols.len())?;
                    stats.lock().unwrap().read_s += gpu.times.get()[1];
                    labels.into_iter().map(ColOut::Labels).collect()
                }
            };
            let mut s = stats.lock().unwrap();
            s.down_s += down;
            s.sweeps += 1;
            s.columns += cols.len() as u64;
            Ok(out)
        }))
        .unwrap_or_else(|e| Err(anyhow!("GPU panic: {}", e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default())));
        match &r {
            Err(e) if tries == 0 && oom(&format!("{e:#}")) => {
                tries += 1;
                loaded.clear();
                ids.clear();
                mats.clear();
                used = 0;
                let _ = sync_cleanup(&gpu);
                let mut s = stats.lock().unwrap();
                s.oom_retries += 1;
                s.vram_used = 0;
            }
            _ => break r,
        }
        };
        match r {
            Ok(mut out) => {
                for j in batch {
                    let rest = out.split_off(j.cols.len());
                    let _ = j.reply.send(Ok(std::mem::replace(&mut out, rest)));
                }
            }
            Err(e) => {
                let msg = format!("{e:#}");
                for j in batch {
                    let _ = j.reply.send(Err(msg.clone()));
                }
            }
        }
    }
}
