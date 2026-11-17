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
use crate::sweep::{Bounds, Column, Key, all_bounds, cpu_down};
use crate::vocab::{Dir, Mode};

/// A loaded component: its graph, and its entry bounds once a pruned CPU sweep needs them.
pub struct Comp {
    pub g: Graph,
    pub bounds: OnceLock<Result<Bounds, String>>,
}

impl Comp {
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
    reply: Reply,
}

#[derive(Default)]
struct Queue {
    jobs: VecDeque<Job>,
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
    pub fn submit(&self, key: Key, comp: Arc<Comp>, cols: Vec<Column>, heads: Option<JobHeads>) -> mpsc::Receiver<Result<Vec<ColOut>, String>> {
        let (tx, rx) = mpsc::channel();
        let (m, cv) = &*self.q;
        m.lock().unwrap().jobs.push_back(Job { key, comp, cols, heads, reply: tx });
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

fn run(gpu: Gpu, cfg: &GpuConfig, q: &(Mutex<Queue>, Condvar), stats: &Mutex<Stats>) {
    let p_max = cfg.b_max.div_ceil(2).max(1);
    let mut loaded: BTreeMap<(u8, u32), Loaded> = BTreeMap::new();
    // Per mode (family, metric), its ID table.
    let mut ids: BTreeMap<(u8, u8), (DevIds, u64)> = BTreeMap::new();
    let mut used = 0u64;
    let mut clock = 0u64;
    let sync_cleanup = |gpu: &Gpu| -> Result<()> {
        cubecl::future::block_on(gpu.client.sync()).map_err(|e| anyhow!("sync: {e:?}"))?;
        gpu.client.memory_cleanup().map_err(|e| anyhow!("memory cleanup: {e:?}"))
    };
    loop {
        // The oldest job and every queued one with its key, up to b_max columns.
        let batch: Vec<Job> = {
            let (m, cv) = q;
            let mut g = m.lock().unwrap();
            while g.jobs.is_empty() && !g.stop {
                g = cv.wait(g).unwrap();
            }
            if g.jobs.is_empty() {
                // Free the device buffers and wait for the device, so that nothing is left
                // to free after the process starts to exit.
                drop(g);
                loaded.clear();
                ids.clear();
                let _ = cubecl::future::block_on(gpu.client.sync());
                return;
            }
            let first = g.jobs.pop_front().unwrap();
            let key = first.key;
            let mut cols = first.cols.len();
            let mut batch = vec![first];
            let mut i = 0;
            while i < g.jobs.len() {
                if g.jobs[i].key == key && cols + g.jobs[i].cols.len() <= 2 * p_max as usize {
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
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<Vec<ColOut>> {
            let ck = (key.fam, key.component);
            let mk = (key.fam, key.metric);
            if !loaded.contains_key(&ck) {
                let g = &batch[0].comp.g;
                let need = Gpu::estimate(g, p_max, cfg.prune);
                // The least recently used component or ID table goes first.
                while used + need > cfg.budget {
                    let lc = loaded.iter().min_by_key(|(_, l)| l.used_at).map(|(&k, l)| (k, l.used_at));
                    let li = ids.iter().min_by_key(|(_, t)| t.1).map(|(&k, t)| (k, t.1));
                    match (lc, li) {
                        (None, None) => break,
                        (Some((k, a)), Some((_, b))) if a <= b => {
                            let l = loaded.remove(&k).unwrap();
                            used -= l.dev.bytes + l.heads.as_ref().map_or(0, |h| h.bytes);
                        }
                        (Some((k, _)), None) => {
                            let l = loaded.remove(&k).unwrap();
                            used -= l.dev.bytes + l.heads.as_ref().map_or(0, |h| h.bytes);
                        }
                        (_, Some((k, _))) => used -= ids.remove(&k).unwrap().0.bytes,
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
            if want_heads && let Some((dids, at)) = ids.get_mut(&mk) {
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
