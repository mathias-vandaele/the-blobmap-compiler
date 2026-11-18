//! Columns at one list of targets for many sources, as one device job (an N×T block of
//! SPEC §9 `column` answers, one row a source): the time of a target reads only the labels of
//! its head nodes (FORMAT §8), and a node's label only those of its elimination-tree
//! ancestors, so the job labels the union of the head nodes' chains alone:
//!
//! 1. union: the chains marked from the head nodes, then numbered in new-id order (a prefix
//!    scan), so that the union keeps the sweep's level order;
//! 2. upward: one cube per source, its upward search space being its seeds' chains, labelled
//!    bottom-up in shared memory; the labels at union nodes seed the sweep;
//! 3. sweep: the downward half over the union, one launch per level, every source a column,
//!    a node's links split among groups of units;
//! 4. heads: h of each target node, then each target's time per source, written row by row;
//! 5. read-back of the block; the host adds the direct terms of sources and targets that
//!    share a piece.
//!
//! The same labels and times as one column per source, exactly: no pruning, saturating sums.

#![allow(clippy::unnecessary_cast, clippy::collapsible_if, clippy::too_many_arguments, clippy::manual_div_ceil, clippy::if_same_then_else)]

use std::collections::BTreeMap;

use anyhow::{Result, anyhow, ensure};
use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::gpu::{Gpu, arg, bytes_u16, bytes_u32, gid, grid, sadd, vadd2, vmin2, wpair};
use crate::graph::Graph;
use crate::vocab::Dir;

const NONE: u32 = u32::MAX;
const UP_THREADS: u32 = 64;
const SCAN_BLOCK: u32 = 1024;
/// Groups of 32 units sharing a union node's links in the sweep.
const MM_SPLIT: u32 = 8;

// ---- kernels -------------------------------------------------------------------------

/// Marks the chains of `need` (new ids): upos 0 on every ancestor, from NONE.
#[cube(launch)]
fn mm_mark(need: &[u32], count: u32, parent: &[u32], upos: &mut [u32]) {
    let t = gid();
    if t < count {
        let mut u = need[t as usize];
        let mut go = true;
        while go {
            if upos[u as usize] != NONE {
                go = false;
            } else {
                upos[u as usize] = 0u32;
                u = parent[u as usize];
                if u == NONE {
                    go = false;
                }
            }
        }
    }
}

/// Marked nodes per block of `SCAN_BLOCK` new ids (256 units, 4 ids each).
#[cube(launch)]
fn mm_count(upos: &[u32], n: u32, bsum: &mut [u32]) {
    let c = Shared::<[Atomic<u32>]>::new_slice(1usize);
    let b = CUBE_POS_Y as u32 * CUBE_COUNT_X as u32 + CUBE_POS_X as u32;
    let t = UNIT_POS_X;
    if t == 0u32 {
        c[0usize].store(0u32);
    }
    sync_cube();
    let base = b * 1024u32 + t * 4u32;
    let mut k = 0u32;
    for e in 0..4u32 {
        let v = base + e;
        if v < n && upos[v as usize] != NONE {
            k += 1u32;
        }
    }
    c[0usize].fetch_add(k);
    sync_cube();
    if t == 0u32 {
        bsum[b as usize] = c[0usize].load();
    }
}

/// The blocks' counts, exclusive prefix sums in place (one cube of 1024 units); the total
/// at `total[0]`.
#[cube(launch)]
fn mm_scan(bsum: &mut [u32], nb: u32, total: &mut [u32]) {
    let mut sa = Shared::<[u32]>::new_slice(1024usize);
    let mut sb = Shared::<[u32]>::new_slice(1024usize);
    let t = UNIT_POS_X;
    let chunk = (nb + 1023u32) / 1024u32;
    let lo = t * chunk;
    let mut s = 0u32;
    for i in 0..chunk {
        let j = lo + i;
        if j < nb {
            s += bsum[j as usize];
        }
    }
    sa[t as usize] = s;
    sync_cube();
    // Hillis–Steele inclusive scan, ping-pong between sa and sb.
    let mut d = 1u32;
    let mut in_a = true;
    while d < 1024u32 {
        if in_a {
            let x = if t >= d { sa[(t - d) as usize] } else { 0u32 };
            sb[t as usize] = sa[t as usize] + x;
        } else {
            let x = if t >= d { sb[(t - d) as usize] } else { 0u32 };
            sa[t as usize] = sb[t as usize] + x;
        }
        sync_cube();
        in_a = !in_a;
        d *= 2u32;
    }
    let inc = if in_a { sa[t as usize] } else { sb[t as usize] };
    let mut run = inc - s;
    for i in 0..chunk {
        let j = lo + i;
        if j < nb {
            let x = bsum[j as usize];
            bsum[j as usize] = run;
            run += x;
        }
    }
    if t == 1023u32 {
        total[0usize] = inc;
    }
}

/// Numbers the marked nodes in new-id order: upos[v] its position, ulist the inverse.
#[cube(launch)]
fn mm_place(upos: &mut [u32], n: u32, bsum: &[u32], ulist: &mut [u32]) {
    let mut sa = Shared::<[u32]>::new_slice(256usize);
    let mut sb = Shared::<[u32]>::new_slice(256usize);
    let b = CUBE_POS_Y as u32 * CUBE_COUNT_X as u32 + CUBE_POS_X as u32;
    let t = UNIT_POS_X;
    let base = b * 1024u32 + t * 4u32;
    let mut k = 0u32;
    for e in 0..4u32 {
        let v = base + e;
        if v < n && upos[v as usize] != NONE {
            k += 1u32;
        }
    }
    sa[t as usize] = k;
    sync_cube();
    let mut d = 1u32;
    let mut in_a = true;
    while d < 256u32 {
        if in_a {
            let x = if t >= d { sa[(t - d) as usize] } else { 0u32 };
            sb[t as usize] = sa[t as usize] + x;
        } else {
            let x = if t >= d { sb[(t - d) as usize] } else { 0u32 };
            sa[t as usize] = sb[t as usize] + x;
        }
        sync_cube();
        in_a = !in_a;
        d *= 2u32;
    }
    let inc = if in_a { sa[t as usize] } else { sb[t as usize] };
    let mut pos = bsum[b as usize] + inc - k;
    for e in 0..4u32 {
        let v = base + e;
        if v < n && upos[v as usize] != NONE {
            upos[v as usize] = pos;
            ulist[pos as usize] = v;
            pos += 1u32;
        }
    }
}

/// Per level l (and the end), the union's first position at or past `level_first[l]`.
#[cube(launch)]
fn mm_levels(level_first: &[u32], levels: u32, ulist: &[u32], total: &[u32], ul_first: &mut [u32]) {
    let l = gid();
    if l <= levels {
        let x = level_first[l as usize];
        let mut lo = 0u32;
        let mut hi = total[0usize];
        while lo < hi {
            let mid = (lo + hi) / 2u32;
            if ulist[mid as usize] < x {
                lo = mid + 1u32;
            } else {
                hi = mid;
            }
        }
        ul_first[l as usize] = lo;
    }
}

#[cube(launch)]
fn mm_fill(x: &mut [u32], count: u32) {
    let t = gid();
    if t < count {
        x[t as usize] = 0xFFFF_FFFFu32;
    }
}

/// Puts back upos to NONE on the union.
#[cube(launch)]
fn mm_clear(ulist: &[u32], total: &[u32], upos: &mut [u32]) {
    let t = gid();
    if t < total[0usize] {
        upos[ulist[t as usize] as usize] = NONE;
    }
}

/// Position of v in `chain[0..len]` (descending), or `len` if absent.
#[cube]
fn chain_find(chain: &[u32], len: u32, v: u32) -> u32 {
    let mut lo = 0u32;
    let mut hi = len;
    while lo < hi {
        let mid = (lo + hi) / 2u32;
        if chain[mid as usize] > v {
            lo = mid + 1u32;
        } else {
            hi = mid;
        }
    }
    let mut r = len;
    if lo < len {
        if chain[lo as usize] == v {
            r = lo;
        }
    }
    r
}

/// The upward half of source s = this cube: its seeds' chains merged (descending new ids),
/// labelled bottom-up (a node's lower neighbours on the chain come before it), then the
/// labels at union nodes written into column s of `state` (all INF before).
#[cube(launch)]
fn mm_up(
    seeds_first: &[u32],
    seeds: &[u32],
    parent: &[u32],
    up_first: &[u32],
    up_head: &[u32],
    w: &[u32],
    upos: &[u32],
    state: &mut [Atomic<u32>],
    p: u32,
    flag: &mut [Atomic<u32>],
    #[comptime] cap: usize,
) {
    let mut chain = Shared::<[u32]>::new_slice(cap);
    let mut run = Shared::<[u32]>::new_slice(cap);
    let lab = Shared::<[Atomic<u32>]>::new_slice(cap);
    let mut kend = Shared::<[u32]>::new_slice(cap);
    let mut len_s = Shared::<[u32]>::new_slice(1usize);
    let s = CUBE_POS_X as u32;
    let t = UNIT_POS_X;
    let nt = CUBE_DIM_X;
    let capu = cap as u32;
    let (s0, s1) = (seeds_first[s as usize], seeds_first[(s + 1u32) as usize]);
    if t == 0u32 {
        let mut len = 0u32;
        let mut over = false;
        for k in s0..s1 {
            // The seed's chain up to the first node already in it.
            let mut r = 0u32;
            let mut u = seeds[(2u32 * k) as usize];
            let mut go = true;
            while go {
                if u == NONE {
                    go = false;
                } else if chain_find(&chain, len, u) < len {
                    go = false;
                } else if len + r >= capu {
                    over = true;
                    go = false;
                } else {
                    run[r as usize] = u;
                    r += 1u32;
                    u = parent[u as usize];
                }
            }
            // Merge from the end, both descending.
            let mut ia = len;
            let mut jb = r;
            while jb > 0u32 {
                if ia > 0u32 && chain[(ia - 1u32) as usize] < run[(jb - 1u32) as usize] {
                    chain[(ia + jb - 1u32) as usize] = chain[(ia - 1u32) as usize];
                    ia -= 1u32;
                } else {
                    chain[(ia + jb - 1u32) as usize] = run[(jb - 1u32) as usize];
                    jb -= 1u32;
                }
            }
            len += r;
        }
        if over {
            flag[0usize].fetch_or(1u32);
            len = 0u32;
        }
        len_s[0usize] = len;
    }
    sync_cube();
    let len = len_s[0usize];
    // Each chain node's row bounds, in shared memory (`run` is free past the build).
    let mut i = t;
    while i < len {
        let c = chain[i as usize];
        run[i as usize] = up_first[c as usize];
        kend[i as usize] = up_first[(c + 1u32) as usize];
        lab[i as usize].store(0xFFFFu32);
        i += nt;
    }
    sync_cube();
    if t == 0u32 {
        for k in s0..s1 {
            let j = chain_find(&chain, len, seeds[(2u32 * k) as usize]);
            if j < len {
                lab[j as usize].fetch_min(seeds[(2u32 * k + 1u32) as usize]);
            }
        }
    }
    sync_cube();
    // Bottom-up, each unit's first link of the next row loaded before the barrier.
    let steps = len;
    let mut hn = 0xFFFF_FFFFu32;
    let mut wn = 0u32;
    if steps > 0u32 {
        let k = run[0usize] + t;
        if k < kend[0usize] {
            hn = up_head[k as usize];
            wn = wpair(w, k) & 0xFFFFu32;
        }
    }
    for i in 0..steps {
        let h = hn;
        let wk = wn;
        hn = 0xFFFF_FFFFu32;
        if i + 1u32 < steps {
            let k = run[(i + 1u32) as usize] + t;
            if k < kend[(i + 1u32) as usize] {
                hn = up_head[k as usize];
                wn = wpair(w, k) & 0xFFFFu32;
            }
        }
        let li = lab[i as usize].load();
        if li < 0xFFFFu32 {
            if h != NONE {
                let j = chain_find(&chain, len, h);
                if j < len {
                    lab[j as usize].fetch_min(sadd(li, wk));
                }
            }
            let k1 = kend[i as usize];
            let mut k = run[i as usize] + t + nt;
            while k < k1 {
                let j = chain_find(&chain, len, up_head[k as usize]);
                if j < len {
                    lab[j as usize].fetch_min(sadd(li, wpair(w, k) & 0xFFFFu32));
                }
                k += nt;
            }
        }
        sync_cube();
    }
    let q = s / 2u32;
    let sh = (s % 2u32) * 16u32;
    let mut i = t;
    while i < len {
        let l = lab[i as usize].load();
        let up = upos[chain[i as usize] as usize];
        if l < 0xFFFFu32 && up != NONE {
            let m = if sh == 0u32 { l | 0xFFFF_0000u32 } else { (l << 16u32) | 0xFFFFu32 };
            state[(up * p + q) as usize].fetch_and(m);
        }
        i += nt;
    }
}

/// Each union link's head as a union position: `uhead[k] = upos[up_head[k]]` on the union's rows.
#[cube(launch)]
fn mm_map(ulist: &[u32], total: &[u32], up_first: &[u32], up_head: &[u32], upos: &[u32], uhead: &mut [u32]) {
    let i = gid();
    if i < total[0usize] {
        let v = ulist[i as usize];
        for k in up_first[v as usize]..up_first[(v + 1u32) as usize] {
            uhead[k as usize] = upos[up_head[k as usize] as usize];
        }
    }
}

/// One union level, a cube per (node, 32 words), its `s` groups of 32 units taking every
/// s-th link, then a minimum over the groups in shared memory.
#[cube(launch)]
fn mm_split(u0: u32, cnt: u32, ulist: &[u32], up_first: &[u32], uhead: &[u32], w: &[u32], state: &mut [u32], p: u32, #[comptime] s: u32) {
    let mut part = Shared::<[u32]>::new_slice((s * 32u32) as usize);
    let c = CUBE_POS_Y as u32 * CUBE_COUNT_X as u32 + CUBE_POS_X as u32;
    let wq = (p + 31u32) / 32u32;
    let li = c / wq;
    let i = u0 + li;
    let grp = UNIT_POS_X / 32u32;
    let lane = UNIT_POS_X % 32u32;
    let q = (c % wq) * 32u32 + lane;
    let live = li < cnt && q < p;
    let mut acc = 0xFFFF_FFFFu32;
    let mut k1 = 0u32;
    let mut k = 0u32;
    if live {
        let v = ulist[i as usize];
        k = up_first[v as usize] + grp;
        k1 = up_first[(v + 1u32) as usize];
    }
    while k < k1 {
        acc = vmin2(acc, vadd2(state[(uhead[k as usize] * p + q) as usize], wpair(w, k)));
        k += s;
    }
    part[UNIT_POS_X as usize] = acc;
    sync_cube();
    if grp == 0u32 && live {
        let mut m = state[(i * p + q) as usize];
        for j in 0..s {
            m = vmin2(m, part[(j * 32u32 + lane) as usize]);
        }
        state[(i * p + q) as usize] = m;
    }
}

/// h of target node j, word q: the least of its terms' labels ⊕ cost.
#[cube(launch)]
fn mm_h(hn_first: &[u32], terms: &[u32], nh: u32, upos: &[u32], state: &[u32], p: u32, h: &mut [u32]) {
    let t = gid();
    if t < nh * p {
        let j = t / p;
        let q = t % p;
        let mut acc = 0xFFFF_FFFFu32;
        for e in hn_first[j as usize]..hn_first[(j + 1u32) as usize] {
            let up = upos[terms[(2u32 * e) as usize] as usize];
            let c = terms[(2u32 * e + 1u32) as usize];
            acc = vmin2(acc, vadd2(state[(up * p + q) as usize], c | (c << 16u32)));
        }
        h[(j * p + q) as usize] = acc;
    }
}

/// Times of targets 2jp, 2jp + 1 for sources 2q, 2q + 1, into rows of `tw` words (u16
/// pairs): `tgt` is 4 words a target (h index of node 0 and 1 or NONE, their offsets).
#[cube(launch)]
fn mm_times(tgt: &[u32], nt: u32, ns: u32, h: &[u32], p: u32, tw: u32, out: &mut [u32]) {
    let t = gid();
    if t < p * tw {
        let q = t / tw;
        let jp = t % tw;
        let mut acc0 = 0xFFFF_FFFFu32;
        let mut acc1 = 0xFFFF_FFFFu32;
        for e in 0..2u32 {
            let j = 2u32 * jp + e;
            if j < nt {
                let ja = tgt[(4u32 * j) as usize];
                let jb = tgt[(4u32 * j + 1u32) as usize];
                let oa = tgt[(4u32 * j + 2u32) as usize];
                let ob = tgt[(4u32 * j + 3u32) as usize];
                let mut acc = 0xFFFF_FFFFu32;
                if ja != NONE {
                    acc = vmin2(acc, vadd2(h[(ja * p + q) as usize], oa | (oa << 16u32)));
                }
                if jb != NONE {
                    acc = vmin2(acc, vadd2(h[(jb * p + q) as usize], ob | (ob << 16u32)));
                }
                if e == 0u32 {
                    acc0 = acc;
                } else {
                    acc1 = acc;
                }
            }
        }
        let r0 = 2u32 * q;
        out[(r0 * tw + jp) as usize] = (acc0 & 0xFFFFu32) | (acc1 << 16u32);
        if r0 + 1u32 < ns {
            out[((r0 + 1u32) * tw + jp) as usize] = (acc0 >> 16u32) | (acc1 & 0xFFFF_0000u32);
        }
    }
}

// ---- host side -----------------------------------------------------------------------

/// A component's hierarchy for matrix jobs: rows by new id (u32 offsets), parents, the
/// union positions (all NONE between jobs), and per metric and direction the weights.
pub struct DevMatrix {
    n: usize,
    level_first: Handle,
    levels: usize,
    up_first: Handle,
    up_head: Handle,
    parent: Handle,
    upos: Handle,
    /// Per link, its head's union position, written on the union's rows by each job.
    uhead: Handle,
    w: BTreeMap<usize, (Handle, Handle)>,
    links: usize,
    cap: usize,
    pub bytes: u64,
}

/// A job's input, built on the host from the sources and targets (no per-pair work).
pub struct MatrixIn {
    /// Per source, its seeds (new id, label) from `seeds_first[s]`.
    pub seeds_first: Vec<u32>,
    pub seeds: Vec<u32>,
    /// The new ids whose chains form the union (every term below).
    pub need: Vec<u32>,
    /// Per target node, its terms (new id, cost) from `hn_first[j]`.
    pub hn_first: Vec<u32>,
    pub terms: Vec<u32>,
    /// Per target, 4 words: h index of node 0, node 1 (or NONE), their offsets.
    pub tgt: Vec<u32>,
}

/// Seconds of a job's phases (device syncs between them when timed): input upload, union
/// (with its level read-back), upward, sweep, heads, read-back.
pub type Phases = [f64; 6];

pub struct MatrixOut {
    /// Row-major, a source per row, `tw` words (u16 pairs) a row.
    pub words: Vec<u32>,
    pub tw: usize,
    pub union: usize,
    pub launches: usize,
    pub phases: Phases,
    /// An upward chain was longer than the shared buffer: nothing was answered.
    pub overflow: bool,
    /// The union's labels for every source exceed a device binding: nothing was answered,
    /// fewer sources a job would.
    pub split: bool,
}

impl DevMatrix {
    pub fn estimate(g: &Graph) -> u64 {
        (g.n as u64 + 1) * 4 * 3 + g.levels() as u64 * 4 + g.links() as u64 * 12
    }
}

impl Gpu {
    fn read_u32(&self, h: &Handle, len: usize) -> Result<Vec<u32>> {
        let desc = h.clone().copy_descriptor(cubecl::zspace::Shape::new([len]), cubecl::zspace::Strides::new(&[1]), 4);
        let bytes = self.client.read_tensor(vec![desc]).map_err(|e| anyhow!("read: {e:?}"))?.remove(0);
        ensure!(bytes.len() >= len * 4, "read {} bytes of {}", bytes.len(), len * 4);
        Ok(bytemuck::pod_collect_to_vec(&bytes[..len * 4]))
    }

    fn sync(&self) -> Result<()> {
        cubecl::future::block_on(self.client.sync()).map_err(|e| anyhow!("sync: {e:?}"))
    }

    pub fn load_matrix(&self, g: &Graph) -> Result<DevMatrix> {
        ensure!(g.links() < (1usize << 32) && g.n < (1usize << 31), "a component of {} links is too large for 32-bit rows", g.links());
        ensure!((g.links() as u64) * 4 <= self.max_chunk, "a component of {} links exceeds a device binding", g.links());
        let up_first: Vec<u32> = g.up_first.iter().map(|&x| x as u32).collect();
        let cl = &self.client;
        // Room for the longest upward chain: the levels and then some (several seeds).
        let cap = ((g.levels() + 128).div_ceil(128) * 128).clamp(256, (self.max_shared / 16).min(3000));
        Ok(DevMatrix {
            n: g.n,
            level_first: cl.create_from_slice(bytes_u32(&g.level_first)),
            levels: g.levels(),
            up_first: cl.create_from_slice(bytes_u32(&up_first)),
            up_head: cl.create_from_slice(bytes_u32(if g.links() == 0 { &[0] } else { &g.up_head })),
            parent: cl.create_from_slice(bytes_u32(&g.parent)),
            upos: cl.create_from_slice(bytes_u32(&vec![NONE; g.n.max(1)])),
            uhead: cl.empty(g.links().max(1) * 4),
            w: BTreeMap::new(),
            links: g.links(),
            cap,
            bytes: DevMatrix::estimate(g),
        })
    }

    /// One job: N = sources, T = targets.
    pub fn matrix(&self, d: &mut DevMatrix, g: &Graph, metric: usize, dir: Dir, job: &MatrixIn, timed: bool) -> Result<MatrixOut> {
        let cl = self.client.clone();
        let mut ph = [0.0f64; 6];
        let mut t = std::time::Instant::now();
        let lap = |ph: &mut Phases, k: usize, t: &mut std::time::Instant| -> Result<()> {
            if timed {
                self.sync()?;
                ph[k] += t.elapsed().as_secs_f64();
                *t = std::time::Instant::now();
            }
            Ok(())
        };
        let ns = job.seeds_first.len() - 1;
        let nt = job.tgt.len() / 4;
        let nh = job.hn_first.len() - 1;
        ensure!((1..=65535).contains(&ns) && nt >= 1, "{ns} sources and {nt} targets");
        let wi = 2 * metric + crate::sweep::dir_index(dir);
        if !d.w.contains_key(&wi) {
            let down = cl.create_from_slice(&bytes_u16(g.going_down(metric, dir)));
            let up = cl.create_from_slice(&bytes_u16(g.going_up(metric, dir)));
            d.bytes += 2 * (d.links as u64) * 2;
            d.w.insert(wi, (down, up));
        }
        let (wd, wu) = d.w[&wi].clone();
        let wl = d.links.div_ceil(2).max(1);
        let up = |x: &[u32]| cl.create_from_slice(bytes_u32(if x.is_empty() { &[0] } else { x }));
        let (sf, sd, nd, hf, tm, tg) = (up(&job.seeds_first), up(&job.seeds), up(&job.need), up(&job.hn_first), up(&job.terms), up(&job.tgt));
        lap(&mut ph, 0, &mut t)?;
        // 1. Union.
        let n = d.n as u32;
        let dim = CubeDim::new_1d(256);
        mm_mark::launch(&cl, grid(job.need.len() as u32, 256, self.max_x), dim, arg(&nd, job.need.len()), job.need.len() as u32, arg(&d.parent, d.n), arg(&d.upos, d.n));
        let nb = n.div_ceil(SCAN_BLOCK).max(1);
        let bsum = cl.empty(nb as usize * 4);
        let total = cl.empty(4);
        let nbx = grid(nb, 1, self.max_x);
        mm_count::launch(&cl, nbx.clone(), dim, arg(&d.upos, d.n), n, arg(&bsum, nb as usize));
        mm_scan::launch(&cl, CubeCount::Static(1, 1, 1), CubeDim::new_1d(1024), arg(&bsum, nb as usize), nb, arg(&total, 1));
        let ulist = cl.empty(d.n.max(1) * 4);
        mm_place::launch(&cl, nbx, dim, arg(&d.upos, d.n), n, arg(&bsum, nb as usize), arg(&ulist, d.n));
        let ulf = cl.empty((d.levels + 1) * 4);
        mm_levels::launch(&cl, grid(d.levels as u32 + 1, 256, self.max_x), dim, arg(&d.level_first, d.levels + 1), d.levels as u32, arg(&ulist, d.n), arg(&total, 1), arg(&ulf, d.levels + 1));
        mm_map::launch(&cl, grid(d.n as u32, 256, self.max_x), dim, arg(&ulist, d.n), arg(&total, 1), arg(&d.up_first, d.n + 1), arg(&d.up_head, d.links), arg(&d.upos, d.n), arg(&d.uhead, d.links));
        let ul = self.read_u32(&ulf, d.levels + 1)?;
        let u = ul[d.levels] as usize;
        lap(&mut ph, 1, &mut t)?;
        // 2. Upward, into the union's state (all INF first).
        let p = ns.div_ceil(2) as u32;
        let words = u.max(1) * p as usize;
        if (words as u64) * 4 > self.max_chunk {
            // Too many sources for one binding: the caller splits them.
            mm_clear::launch(&cl, grid(u.max(1) as u32, 256, self.max_x), dim, arg(&ulist, d.n), arg(&total, 1), arg(&d.upos, d.n));
            return Ok(MatrixOut { words: Vec::new(), tw: 0, union: u, launches: 0, phases: ph, overflow: false, split: true });
        }
        let state = cl.empty(words * 4);
        mm_fill::launch(&cl, grid(words as u32, 256, self.max_x), dim, arg(&state, words), words as u32);
        let flag = cl.create_from_slice(bytes_u32(&[0]));
        mm_up::launch(
            &cl,
            CubeCount::Static(ns as u32, 1, 1),
            CubeDim::new_1d(UP_THREADS),
            arg(&sf, ns + 1),
            arg(&sd, job.seeds.len()),
            arg(&d.parent, d.n),
            arg(&d.up_first, d.n + 1),
            arg(&d.up_head, d.links),
            arg(&wu, wl),
            arg(&d.upos, d.n),
            arg(&state, words),
            p,
            arg(&flag, 1),
            d.cap,
        );
        lap(&mut ph, 2, &mut t)?;
        // 3. Sweep, level by level.
        // A launch a union level: a cube per (node, 32 words), its links split 8 ways (a
        // node near the top has every ancestor in the top clique as an upper neighbour).
        let mut launches = 0;
        for l in 0..d.levels {
            let cnt = ul[l + 1] - ul[l];
            if cnt > 0 {
                mm_split::launch(&cl, grid(cnt * p.div_ceil(32), 1, self.max_x), CubeDim::new_1d(32 * MM_SPLIT), ul[l], cnt, arg(&ulist, d.n), arg(&d.up_first, d.n + 1), arg(&d.uhead, d.links), arg(&wd, wl), arg(&state, words), p, MM_SPLIT);
                launches += 1;
            }
        }
        lap(&mut ph, 3, &mut t)?;
        // 4. Heads.
        let hh = cl.empty(nh.max(1) * p as usize * 4);
        mm_h::launch(&cl, grid(nh as u32 * p, 256, self.max_x), dim, arg(&hf, nh + 1), arg(&tm, job.terms.len()), nh as u32, arg(&d.upos, d.n), arg(&state, words), p, arg(&hh, nh.max(1) * p as usize));
        let tw = nt.div_ceil(2);
        let out = cl.empty(ns * tw * 4);
        mm_times::launch(&cl, grid(p * tw as u32, 256, self.max_x), dim, arg(&tg, nt * 4), nt as u32, ns as u32, arg(&hh, nh.max(1) * p as usize), p, tw as u32, arg(&out, ns * tw));
        mm_clear::launch(&cl, grid(u.max(1) as u32, 256, self.max_x), dim, arg(&ulist, d.n), arg(&total, 1), arg(&d.upos, d.n));
        lap(&mut ph, 4, &mut t)?;
        // 5. Read-back.
        let words_out = self.read_u32(&out, ns * tw)?;
        let overflow = self.read_u32(&flag, 1)?[0] != 0;
        ph[5] += t.elapsed().as_secs_f64();
        // A chain past the buffer: again with a larger one, while shared memory allows.
        let max_cap = (self.max_shared / 16).min(3000);
        if overflow && d.cap < max_cap {
            d.cap = (d.cap * 2).min(max_cap);
            return self.matrix(d, g, metric, dir, job, timed);
        }
        Ok(MatrixOut { words: words_out, tw, union: u, launches, phases: ph, overflow, split: false })
    }
}
