//! The runtime: a model opened for serving, its columns and pairs (FORMAT §8), and the
//! operations of SPEC §9 on them.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail, ensure};
use blobmap_model::manifest::Manifest;
use blobmap_model::{INF, Model, NONE};
use rayon::prelude::*;

use crate::engine::{Comp, GpuConfig, GpuEngine, cpu_sweep};
use crate::gpu::Kind;
use crate::graph::{Graph, meet, upward};
use crate::heads::{self, Labels, Times, Want};
use crate::sweep::{Column, DEAD, Key, dir_index};
use crate::vocab::{self, Dir, Fam, Family, Mode, Reason, Vocab};

/// Which device sweeps run on. `Auto` takes CubeCL's CUDA runtime, else its wgpu one, else
/// the CPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Auto,
    Cpu,
    Cuda,
    Wgpu,
}

impl Backend {
    pub fn parse(s: &str) -> Result<Backend> {
        Ok(match s {
            "auto" => Backend::Auto,
            "cpu" => Backend::Cpu,
            "cuda" => Backend::Cuda,
            "wgpu" => Backend::Wgpu,
            _ => bail!("backend is auto, cpu, cuda or wgpu, not {s:?}"),
        })
    }
}

#[derive(Clone, Debug)]
pub struct Options {
    pub backend: Backend,
    /// Components smaller than this sweep on the CPU even with a GPU (0: all on the GPU).
    pub gpu_min_nodes: usize,
    /// Columns per batched sweep at most.
    pub b_max: u32,
    /// Device bytes the runtime may use.
    pub vram_budget: u64,
    /// Pruned sweeps for answers that compare times with a bound (entry bounds on the
    /// device: 2 bytes per link, metric and direction).
    pub prune: bool,
    /// Per family (car, active), whether its sweeps go to the GPU (else the CPU).
    pub gpu_families: [bool; 2],
}

impl Default for Options {
    fn default() -> Options {
        Options { backend: Backend::Auto, gpu_min_nodes: 1 << 16, b_max: 16, vram_budget: 14 << 30, prune: true, gpu_families: [true, true] }
    }
}

/// A source of a column: internal index and start time.
#[derive(Clone, Copy, Debug)]
pub struct Src {
    pub id: u32,
    pub start: u16,
}

/// One column to compute: sources, mode, direction, a bound for pruning (`DEAD` or
/// above: none), and the IDs it answers.
#[derive(Clone, Debug)]
pub struct ColReq {
    pub sources: Vec<Src>,
    pub mode: usize,
    pub dir: Dir,
    pub bound: u16,
    pub want: Want,
}

/// Where `columns` spends its time, summed over calls (nanoseconds, every thread).
#[derive(Default, Debug)]
pub struct Profile {
    pub calls: AtomicU64,
    /// Components loaded on the host on first use (read and relayer).
    pub comp_load: AtomicU64,
    pub upward: AtomicU64,
    /// Waiting on the GPU worker: its queue, upload on first use, sweep, read-back.
    pub gpu_wait: AtomicU64,
    pub cpu_sweep: AtomicU64,
    /// Heads at the wanted IDs (`heads`): the output vectors, the passes; the page faults
    /// (minor, major) of the process meanwhile, and the threads of the pool they ran on.
    pub heads: AtomicU64,
    pub heads_init: AtomicU64,
    pub heads_ids: AtomicU64,
    pub heads_minflt: AtomicU64,
    pub heads_majflt: AtomicU64,
    pub heads_threads: AtomicU64,
}

/// The process's page faults so far: minor, major (/proc/self/stat).
pub fn faults() -> (u64, u64) {
    let s = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    let f: Vec<&str> = s.rsplit_once(')').map_or("", |x| x.1).split_whitespace().collect();
    let g = |i: usize| f.get(i).and_then(|x| x.parse().ok()).unwrap_or(0);
    // After the command: state is field 3, minflt field 10, majflt field 12.
    (g(7), g(9))
}

fn since(t: Instant) -> u64 {
    t.elapsed().as_nanos() as u64
}

pub struct Runtime {
    pub manifest: Manifest,
    pub fams: Vec<Family>,
    pub vocab: Vocab,
    /// Per mode, the IDs of each component (CSR).
    members: Vec<(Vec<u32>, Vec<u32>)>,
    comps: Mutex<BTreeMap<(u8, u32), Arc<Comp>>>,
    loading: Mutex<()>,
    gpu: Option<GpuEngine>,
    pub opts: Options,
    pub prof: Profile,
    /// Last, so that it is dropped after the vocabulary's tables that borrow its maps.
    pub model: Model,
}

impl Runtime {
    pub fn open(dir: &Path, opts: Options) -> Result<Runtime> {
        let model = Model::open(dir)?;
        let manifest = Manifest::from_value(&model.manifest).context("manifest")?;
        ensure!(manifest.parts.vocabulary.road_hash == manifest.parts.road.hash, "the vocabulary names another road part (FORMAT §9)");
        // Safety: `model` moves into the Runtime, which drops it after `vocab` (its maps do
        // not move with it).
        let (fams, vocab) = unsafe { vocab::load(&model, &manifest)? };
        let rh = model.get_len::<u8>("vocab/road_hash", 32)?;
        ensure!(hex::encode(&rh[..]) == manifest.parts.road.hash, "vocab/road_hash differs from the manifest (FORMAT §9)");
        let members = vocab
            .modes
            .iter()
            .map(|mv| {
                let k = fams[mv.fam as usize].components();
                let mut first = vec![0u32; k + 1];
                for &x in &mv.component {
                    if x != NONE {
                        first[x as usize + 1] += 1;
                    }
                }
                for i in 0..k {
                    first[i + 1] += first[i];
                }
                let mut at = first.clone();
                let mut ids = vec![0u32; first[k] as usize];
                for (i, &x) in mv.component.iter().enumerate() {
                    if x != NONE {
                        ids[at[x as usize] as usize] = i as u32;
                        at[x as usize] += 1;
                    }
                }
                (first, ids)
            })
            .collect();
        let gpu = match opts.backend {
            Backend::Cpu => None,
            Backend::Cuda | Backend::Wgpu | Backend::Auto => {
                let kinds: &[Kind] = match opts.backend {
                    Backend::Cuda => &[Kind::Cuda],
                    Backend::Wgpu => &[Kind::Wgpu],
                    _ => &[Kind::Cuda, Kind::Wgpu],
                };
                let mut got = None;
                let mut why = Vec::new();
                for &kind in kinds {
                    match GpuEngine::start(GpuConfig { kind, b_max: opts.b_max, budget: opts.vram_budget, prune: opts.prune }) {
                        Ok(e) => {
                            got = Some(e);
                            break;
                        }
                        Err(e) => why.push(format!("{kind:?}: {e:#}")),
                    }
                }
                if got.is_none() && opts.backend != Backend::Auto {
                    bail!("no GPU backend: {}", why.join("; "));
                }
                got
            }
        };
        Ok(Runtime { model, manifest, fams, vocab, members, comps: Mutex::new(BTreeMap::new()), loading: Mutex::new(()), gpu, opts, prof: Profile::default() })
    }

    /// The backend sweeps run on, described.
    pub fn backend(&self) -> String {
        match &self.gpu {
            Some(g) => g.describe.clone(),
            None => "CPU".into(),
        }
    }

    pub fn gpu_stats(&self) -> Option<crate::engine::Stats> {
        self.gpu.as_ref().map(|g| g.stats.lock().unwrap().clone())
    }

    pub fn mode(&self, name: &str) -> Result<usize> {
        self.vocab.modes.iter().position(|m| m.name == name).with_context(|| format!("unknown mode {name:?}"))
    }

    pub fn comp(&self, fam: Fam, x: u32) -> Result<Arc<Comp>> {
        let k = (fam as u8, x);
        if let Some(c) = self.comps.lock().unwrap().get(&k) {
            return Ok(c.clone());
        }
        let _l = self.loading.lock().unwrap();
        if let Some(c) = self.comps.lock().unwrap().get(&k) {
            return Ok(c.clone());
        }
        let g = Graph::load(&self.model, &self.fams[fam as usize], x)?;
        let c = Arc::new(Comp { g, bounds: std::sync::OnceLock::new() });
        self.comps.lock().unwrap().insert(k, c.clone());
        Ok(c)
    }

    /// Drops the components loaded on the host (the device keeps its own until evicted).
    pub fn release(&self) {
        self.comps.lock().unwrap().clear();
    }

    pub fn served(&self, i: u32, mode: usize) -> Result<u32, Reason> {
        self.vocab.served(i, mode)
    }

    /// Columns: for each, the wanted IDs' times from (`From`) or to (`To`) the nearest source
    /// plus its start, `INF` for BEYOND or unserved. Sweeps on one component, metric and
    /// direction go as one batch.
    pub fn columns(&self, reqs: &[ColReq]) -> Result<Vec<Times>> {
        // Per request and component: the sources, then the upward half.
        struct Part {
            req: usize,
            comp: Arc<Comp>,
            x: u32,
            srcs: Vec<Src>,
            col: Column,
        }
        let pr = &self.prof;
        pr.calls.fetch_add(1, Relaxed);
        let t = Instant::now();
        let mut parts = Vec::new();
        for (r, q) in reqs.iter().enumerate() {
            let mv = &self.vocab.modes[q.mode];
            let mut by: BTreeMap<u32, Vec<Src>> = BTreeMap::new();
            for s in &q.sources {
                let x = mv.component[s.id as usize];
                ensure!(x != NONE, "a column source must be served (internal index {})", s.id);
                by.entry(x).or_default().push(*s);
            }
            for (x, srcs) in by {
                let comp = self.comp(mv.fam, x)?;
                parts.push(Part { req: r, comp, x, srcs, col: Column { entries: Vec::new(), bound: q.bound } });
            }
        }
        pr.comp_load.fetch_add(since(t), Relaxed);
        let t = Instant::now();
        parts.par_iter_mut().for_each(|p| {
            let q = &reqs[p.req];
            let mv = &self.vocab.modes[q.mode];
            let g = &p.comp.g;
            let mut seeds = Vec::new();
            for s in &p.srcs {
                seed(g, mv, s, q.dir, &mut |v, t| seeds.push((v, t)));
            }
            seeds.sort_unstable();
            seeds.dedup_by_key(|s| s.0);
            p.col.entries = upward(g, mv.metric, q.dir, &seeds);
        });
        pr.upward.fetch_add(since(t), Relaxed);
        // Sweeps: GPU jobs first (one per component, metric and direction), then the CPU's.
        let mut groups: BTreeMap<Key, Vec<usize>> = BTreeMap::new();
        for (i, p) in parts.iter().enumerate() {
            let mv = &self.vocab.modes[reqs[p.req].mode];
            groups.entry(Key { fam: mv.fam as u8, component: p.x, metric: mv.metric as u8, dir: dir_index(reqs[p.req].dir) as u8 }).or_default().push(i);
        }
        let mut labels: Vec<Option<Labels>> = (0..parts.len()).map(|_| None).collect();
        let mut pending = Vec::new();
        let mut on_cpu = Vec::new();
        for (key, idx) in &groups {
            let comp = &parts[idx[0]].comp;
            match &self.gpu {
                Some(gpu) if comp.g.n >= self.opts.gpu_min_nodes && self.opts.gpu_families[key.fam as usize] => {
                    let cols: Vec<Column> = idx.iter().map(|&i| parts[i].col.clone()).collect();
                    pending.push((idx.clone(), gpu.submit(*key, comp.clone(), cols)));
                }
                _ => on_cpu.extend(idx.iter().map(|&i| (*key, i))),
            }
        }
        let t = Instant::now();
        let cpu: Vec<(usize, Result<Vec<u16>>)> = on_cpu
            .par_iter()
            .map(|&(key, i)| {
                let p = &parts[i];
                let dir = if key.dir == 0 { Dir::From } else { Dir::To };
                (i, cpu_sweep(&p.comp, key.metric as usize, dir, &p.col, self.opts.prune))
            })
            .collect();
        for (i, l) in cpu {
            labels[i] = Some(Labels::Plain(l?));
        }
        pr.cpu_sweep.fetch_add(since(t), Relaxed);
        let t = Instant::now();
        for (idx, rx) in pending {
            let out = rx.recv().map_err(|_| anyhow!("the GPU worker ended"))?.map_err(|e| anyhow!("{e}"))?;
            for (i, l) in idx.into_iter().zip(out) {
                labels[i] = Some(l);
            }
        }
        pr.gpu_wait.fetch_add(since(t), Relaxed);
        // Heads at the wanted IDs: dense wants into one vector per request, sparse ones
        // gathered per request.
        let t = Instant::now();
        let f0 = faults();
        pr.heads_threads.fetch_max(rayon::current_num_threads() as u64, Relaxed);
        let mut dense: Vec<Option<Vec<u16>>> = reqs.iter().map(|q| q.want.dense().then(|| vec![INF; (q.want.hi - q.want.lo) as usize])).collect();
        let mut sparse: Vec<Vec<(u32, u16)>> = reqs.iter().map(|_| Vec::new()).collect();
        pr.heads_init.fetch_add(since(t), Relaxed);
        let ti = Instant::now();
        for (p, lab) in parts.iter().zip(labels) {
            let q = &reqs[p.req];
            let (first, ids) = &self.members[q.mode];
            let part = heads::Part {
                g: &p.comp.g,
                mv: &self.vocab.modes[q.mode],
                x: p.x,
                ids: &ids[first[p.x as usize] as usize..first[p.x as usize + 1] as usize],
                srcs: &p.srcs,
                dir: q.dir,
                bound: q.bound,
            };
            let got = heads::heads(&part, &lab.unwrap(), q.want, dense[p.req].as_deref_mut());
            sparse[p.req].extend(got);
        }
        pr.heads_ids.fetch_add(since(ti), Relaxed);
        let out = dense
            .into_iter()
            .zip(sparse)
            .zip(reqs)
            .map(|((d, mut s), q)| match d {
                Some(t) => Times::Dense { lo: q.want.lo, t },
                None => {
                    s.sort_unstable();
                    Times::Sparse(s)
                }
            })
            .collect();
        let f1 = faults();
        pr.heads_minflt.fetch_add(f1.0 - f0.0, Relaxed);
        pr.heads_majflt.fetch_add(f1.1 - f0.1, Relaxed);
        pr.heads.fetch_add(since(t), Relaxed);
        Ok(out)
    }

    pub fn column(&self, sources: &[Src], mode: usize, dir: Dir, bound: u16, want: Want) -> Result<Times> {
        Ok(self.columns(&[ColReq { sources: sources.to_vec(), mode, dir, bound, want }])?.pop().unwrap())
    }

    /// The time from o to d (o ≠ d, both served in one component), from their ancestors
    /// alone: the meet of o's `From` and d's `To` upward labels, and the direct term on a
    /// shared piece.
    pub fn pair(&self, o: u32, d: u32, mode: usize) -> Result<u16> {
        let mv = &self.vocab.modes[mode];
        let x = mv.component[o as usize];
        ensure!(x != NONE && x == mv.component[d as usize], "a pair must be served in one component (internal indices {o}, {d})");
        let c = self.comp(mv.fam, x)?;
        let g = &c.g;
        let up = |i: u32, dir: Dir| {
            let mut seeds = Vec::new();
            seed(g, mv, &Src { id: i, start: 0 }, dir, &mut |v, t| seeds.push((v, t)));
            seeds.sort_unstable();
            seeds.dedup_by_key(|s| s.0);
            upward(g, mv.metric, dir, &seeds)
        };
        let mut t = meet(&up(o, Dir::From), &up(d, Dir::To));
        if same_piece(mv, o as usize, d as usize) {
            t = t.min(direct(mv, o as usize, d as usize));
        }
        Ok(t)
    }
}

/// The bound of a pruned column for answers that compare times with t.
pub fn bound_of(t: u64) -> u16 {
    if t < DEAD as u64 { t as u16 } else { INF }
}

/// Puts each seed of a source (FORMAT §8): a new id and its label.
fn seed(g: &Graph, mv: &Mode, s: &Src, dir: Dir, put: &mut impl FnMut(u32, u16)) {
    let i = s.id as usize;
    let (nodes, off) = (mv.node(i), mv.off(i));
    let nw = |v: u32| g.new[v as usize];
    match (mv.fam, dir) {
        (Fam::Car, Dir::From) => {
            for k in 0..2 {
                if nodes[k] == NONE {
                    continue;
                }
                for a in g.arcs(nodes[k]) {
                    put(nw(g.arc_head[a]), s.start.saturating_add(off[2 * k + 1]).saturating_add(g.turn_cost[a]));
                }
            }
        }
        (Fam::Car, Dir::To) => {
            for k in 0..2 {
                if nodes[k] == NONE {
                    continue;
                }
                for y in g.arrivals(nodes[k]) {
                    put(nw(y), s.start.saturating_add(off[2 * k]));
                }
            }
        }
        (Fam::Active, Dir::From) => {
            put(nw(nodes[0]), s.start.saturating_add(off[1]));
            put(nw(nodes[1]), s.start.saturating_add(off[3]));
        }
        (Fam::Active, Dir::To) => {
            put(nw(nodes[0]), s.start.saturating_add(off[0]));
            put(nw(nodes[1]), s.start.saturating_add(off[2]));
        }
    }
}

/// Two IDs on the same piece (FORMAT §8 "Direct drive").
pub fn same_piece(mv: &Mode, a: usize, b: usize) -> bool {
    mv.component[a] == mv.component[b] && mv.node(a) == mv.node(b) && mv.road(a) == mv.road(b)
}

/// The direct term from a to b on their shared piece.
pub fn direct(mv: &Mode, a: usize, b: usize) -> u16 {
    let (oa, ob) = (mv.off(a), mv.off(b));
    let before: [(usize, bool); 2] = match mv.fam {
        Fam::Car => [(0, mv.node(a)[0] != NONE), (2, mv.node(a)[1] != NONE)],
        Fam::Active => [(0, true), (2, true)],
    };
    let mut t = INF;
    for (k, present) in before {
        if present && oa[k] != INF && ob[k] >= oa[k] {
            t = t.min(ob[k] - oa[k]);
        }
    }
    t
}
