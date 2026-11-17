//! Where sweeps run: on the CPU, inline, or on a GPU through one worker thread that owns the
//! device. Requests queue there; the worker takes the oldest and every queued request on the
//! same component, metric and direction, up to `b_max` columns, and runs them as one
//! batched sweep. Components load onto the device on first use, inside the VRAM budget,
//! evicting the least recently used ones.

use std::collections::{BTreeMap, VecDeque};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex, OnceLock};

use anyhow::{Result, anyhow, bail};

use crate::gpu::{DevGraph, Gpu, Kind};
use crate::graph::Graph;
use crate::sweep::{Bounds, Column, Key, all_bounds, cpu_down};
use crate::vocab::Dir;

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

type Reply = mpsc::Sender<Result<Vec<Vec<u16>>, String>>;

struct Job {
    key: Key,
    comp: Arc<Comp>,
    cols: Vec<Column>,
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
}

pub struct GpuConfig {
    pub kind: Kind,
    pub b_max: u32,
    pub budget: u64,
    pub prune: bool,
    pub graphs: bool,
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
                gpu.use_graphs &= cfg.graphs;
                let _ = tx.send(Ok(gpu.describe()));
                run(gpu, &cfg, &q2, &s2);
            })?;
        let describe = rx.recv().map_err(|_| anyhow!("the GPU worker ended"))?.map_err(|e| anyhow!("{e}"))?;
        Ok(GpuEngine { q, worker: Some(worker), kind, describe, stats })
    }

    /// Queues a sweep; the receiver gets one label vector per column.
    pub fn submit(&self, key: Key, comp: Arc<Comp>, cols: Vec<Column>) -> mpsc::Receiver<Result<Vec<Vec<u16>>, String>> {
        let (tx, rx) = mpsc::channel();
        let (m, cv) = &*self.q;
        m.lock().unwrap().jobs.push_back(Job { key, comp, cols, reply: tx });
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
    used_at: u64,
}

fn run(gpu: Gpu, cfg: &GpuConfig, q: &(Mutex<Queue>, Condvar), stats: &Mutex<Stats>) {
    let p_max = cfg.b_max.div_ceil(2).max(1);
    let mut loaded: BTreeMap<(u8, u32), Loaded> = BTreeMap::new();
    let mut used = 0u64;
    let mut clock = 0u64;
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
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<Vec<Vec<u16>>> {
            let ck = (key.fam, key.component);
            if !loaded.contains_key(&ck) {
                let g = &batch[0].comp.g;
                let need = Gpu::estimate(g, p_max, cfg.prune);
                while used + need > cfg.budget {
                    let Some((&victim, _)) = loaded.iter().min_by_key(|(_, l)| l.used_at) else { break };
                    used -= loaded.remove(&victim).unwrap().dev.bytes;
                    stats.lock().unwrap().evictions += 1;
                    // Give the evicted buffers back to the device, not to CubeCL's pool.
                    cubecl::future::block_on(gpu.client.sync()).map_err(|e| anyhow!("sync: {e:?}"))?;
                    gpu.client.memory_cleanup().map_err(|e| anyhow!("memory cleanup: {e:?}"))?;
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
                let bounds = if cfg.prune { Some(all_bounds(g)?) } else { None };
                let dev = gpu.load(g, bounds.as_ref(), p_max)?;
                used += dev.bytes;
                let mut s = stats.lock().unwrap();
                s.loads += 1;
                s.vram_used = used;
                s.vram_peak = s.vram_peak.max(used);
                loaded.insert(ck, Loaded { dev, used_at: clock });
            }
            let l = loaded.get_mut(&ck).unwrap();
            l.used_at = clock;
            let cols: Vec<&Column> = batch.iter().flat_map(|j| j.cols.iter()).collect();
            let out = gpu.sweep(&mut l.dev, key.metric as usize, if key.dir == 0 { Dir::From } else { Dir::To }, &cols)?;
            let mut s = stats.lock().unwrap();
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
