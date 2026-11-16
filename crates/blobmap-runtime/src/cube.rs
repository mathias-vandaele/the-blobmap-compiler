//! The research downward sweep (`cu/research/sweep.cu`: seeding, `top_resident` with the
//! warp merge, `wide_level` and `wide_level_split`) ported to CubeCL, on its CUDA runtime.
//! Same state layout (u16 labels, two columns per u32 word, a seed bit per node), same launch
//! sequence (one top launch, one per wide level, optionally captured as a graph), same split
//! of high-degree nodes. One difference: a heavy node's lane groups merge through shared
//! memory and two barriers instead of `__shfl_xor_sync` on a partial mask, because CubeCL's
//! plane shuffles are whole-plane operations.

use anyhow::{Context, Result, ensure};
use cubecl::cuda::{CudaDevice, CudaRuntime};
use cubecl_runtime::runtime::Runtime;
use cubecl::prelude::*;
use cubecl::client::{Client, Graph};
use cubecl::server::Handle;

use crate::layered::{Entry, Layered};

/// Threads of a wide-level cube, and of the resident top (research `RES_THREADS`).
const WIDE_THREADS: u32 = 256;
const TOP_THREADS: u32 = 1024;
/// Research `WIDE_SPLIT_DEFAULT`, per lane width 1, 2, 4, 8, 16 (32: no split).
const SPLIT_DEFAULT: [u32; 5] = [4, 8, 16, 16, 0];

#[cube]
fn seeded(mask: &[u32], v: u32) -> bool {
    ((mask[(v >> 5u32) as usize] >> (v & 31u32)) & 1u32) == 1u32
}

#[cube]
fn vmin2(a: u32, b: u32) -> u32 {
    let (al, bl, ah, bh) = (a & 0xFFFFu32, b & 0xFFFFu32, a >> 16u32, b >> 16u32);
    let lo = if al < bl { al } else { bl };
    let hi = if ah < bh { ah } else { bh };
    lo | (hi << 16u32)
}

#[cube]
fn vadd2(a: u32, b: u32) -> u32 {
    let lo = (a & 0xFFFFu32) + (b & 0xFFFFu32);
    let hi = (a >> 16u32) + (b >> 16u32);
    let lo = if lo < 0xFFFFu32 { lo } else { 0xFFFFu32 };
    let hi = if hi < 0xFFFFu32 { hi } else { 0xFFFFu32 };
    lo | (hi << 16u32)
}

#[cube]
fn wpair(w: &[u16], k: u32) -> u32 {
    let x = u32::cast_from(w[k as usize]);
    x | (x << 16u32)
}

#[cube(launch)]
fn seed_rows(e: &[u32], count: u32, d32: &mut [u32], p: u32, mask: &[Atomic<u32>]) {
    let i = ABSOLUTE_POS as u32;
    if i < count {
        let v = e[(3u32 * i) as usize];
        mask[(v >> 5u32) as usize].fetch_or(1u32 << (v & 31u32));
        for q in 0..p {
            d32[(v * p + q) as usize] = 0xFFFF_FFFFu32;
        }
    }
}

#[cube(launch)]
fn seed_values(e: &[u32], count: u32, d16: &mut [u16], p: u32) {
    let i = ABSOLUTE_POS as u32;
    if i < count {
        let v = e[(3u32 * i) as usize];
        let c = e[(3u32 * i + 1u32) as usize];
        d16[(v * 2u32 * p + c) as usize] = u16::cast_from(e[(3u32 * i + 2u32) as usize]);
    }
}

#[cube(launch)]
fn clear_mask(e: &[u32], count: u32, mask: &mut [u32]) {
    let i = ABSOLUTE_POS as u32;
    if i < count {
        mask[(e[(3u32 * i) as usize] >> 5u32) as usize] = 0u32;
    }
}

/// The narrow top, one cube per column, its labels resident in shared memory; the arc loop
/// handles two arcs per thread and merges a warp's arcs with one reduction when they share a
/// tail (research `top_resident<false, MERGE_WARP>`).
#[cube(launch)]
fn top_resident(
    level_first: &[u32],
    up_first: &[u32],
    top_arc: &[u32],
    w: &[u16],
    d16: &mut [u16],
    mask: &[u32],
    p: u32,
    #[comptime] top_levels: usize,
    #[comptime] max_level: usize,
    #[comptime] top_nodes: usize,
) {
    let mut lf = Shared::<[u32]>::new_slice(top_levels + 1);
    let mut af = Shared::<[u32]>::new_slice(top_levels + 1);
    let acc = Shared::<[Atomic<u32>]>::new_slice(2 * max_level);
    let mut dist = Shared::<[u32]>::new_slice(top_nodes);
    let c = CUBE_POS as u32;
    let nt = CUBE_DIM;
    let tid = UNIT_POS;
    let lane = UNIT_POS_PLANE;
    let stride = 2u32 * p;
    let tl = top_levels as u32;
    let ml = max_level as u32;
    let mut i = tid;
    while i <= tl {
        let x = level_first[i as usize];
        lf[i as usize] = x;
        af[i as usize] = up_first[x as usize];
        i += nt;
    }
    sync_cube();
    let tn = lf[tl as usize];
    let mut v = tid;
    while v < tn {
        let s = seeded(mask, v);
        dist[v as usize] = if s { u32::cast_from(d16[(v * stride + c) as usize]) } else { 0xFFFFu32 };
        v += nt;
    }
    sync_cube();
    let mut i = tid;
    while i < lf[1usize] {
        acc[i as usize].store(dist[i as usize]);
        i += nt;
    }
    sync_cube();
    for l in 0..tl {
        let off = (l & 1u32) * ml;
        let a1 = af[(l + 1u32) as usize];
        let mut i = af[l as usize] + tid;
        // Warp-uniform trip count (i - lane is the same for the whole warp).
        while i - lane < a1 {
            let k0 = i;
            let k1 = i + nt;
            let live0 = k0 < a1;
            let live1 = k1 < a1;
            let r0 = top_arc[(if live0 { k0 } else { 0u32 }) as usize];
            let x0 = u32::cast_from(w[(if live0 { k0 } else { 0u32 }) as usize]);
            let r1 = top_arc[(if live1 { k1 } else { 0u32 }) as usize];
            let x1 = u32::cast_from(w[(if live1 { k1 } else { 0u32 }) as usize]);
            let mut y0 = 0xFFFFu32;
            let mut t0 = 0xFFFF_FFFFu32;
            if live0 {
                let s = dist[(r0 & 0xFFFFu32) as usize] + x0;
                y0 = if s < 0xFFFFu32 { s } else { 0xFFFFu32 };
                t0 = r0 >> 16u32;
            }
            if plane_all(t0 == plane_broadcast(t0, 0u32)) {
                let m = plane_min(y0);
                if lane == 0u32 && live0 {
                    acc[(off + t0) as usize].fetch_min(m);
                }
            } else if live0 {
                acc[(off + t0) as usize].fetch_min(y0);
            }
            let mut y1 = 0xFFFFu32;
            let mut t1 = 0xFFFF_FFFFu32;
            if live1 {
                let s = dist[(r1 & 0xFFFFu32) as usize] + x1;
                y1 = if s < 0xFFFFu32 { s } else { 0xFFFFu32 };
                t1 = r1 >> 16u32;
            }
            if plane_all(t1 == plane_broadcast(t1, 0u32)) {
                let m = plane_min(y1);
                if lane == 0u32 && live1 {
                    acc[(off + t1) as usize].fetch_min(m);
                }
            } else if live1 {
                acc[(off + t1) as usize].fetch_min(y1);
            }
            i += 2u32 * nt;
        }
        sync_cube();
        let base = lf[l as usize];
        let cnt = lf[(l + 1u32) as usize] - base;
        let mut i = tid;
        while i < cnt {
            let val = acc[(off + i) as usize].load();
            dist[(base + i) as usize] = val;
            d16[((base + i) * stride + c) as usize] = u16::cast_from(val);
            i += nt;
        }
        if l + 1u32 < tl {
            let noff = ml - off;
            let nb = lf[(l + 1u32) as usize];
            let ncnt = lf[(l + 2u32) as usize] - nb;
            let mut i = tid;
            while i < ncnt {
                acc[(noff + i) as usize].store(dist[(nb + i) as usize]);
                i += nt;
            }
        }
        sync_cube();
    }
}

/// One wide level, `lp` lanes per node, each lane owning column pairs (research `wide_level`).
#[cube(launch)]
fn wide_level(base: u32, cnt: u32, up_first: &[u32], up_head: &[u32], w: &[u16], d32: &mut [u32], p: u32, mask: &[u32], #[comptime] lp: u32) {
    let t = ABSOLUTE_POS as u32;
    if t < cnt * lp {
        let v = base + t / lp;
        let lane = t % lp;
        let s = seeded(mask, v);
        let k0 = up_first[v as usize];
        let k1 = up_first[(v + 1u32) as usize];
        let mut q = lane;
        while q < p {
            let mut acc = if s { d32[(v * p + q) as usize] } else { 0xFFFF_FFFFu32 };
            for k in k0..k1 {
                let x = d32[(up_head[k as usize] * p + q) as usize];
                acc = vmin2(acc, vadd2(x, wpair(w, k)));
            }
            d32[(v * p + q) as usize] = acc;
            q += lp;
        }
    }
}

/// A wide level with heavy nodes (research `wide_level_split`): items are the heavy classes
/// (32u32, 16u32, 8u32, 4u32, 2u32 lane groups of `lp` lanes, each group taking every G-th arc), then every
/// node with `lp` lanes, skipping those of more than `skip_t` arcs. `split` holds, per wide
/// level, the first heavy node and the five class counts; `li` is this level's index there.
#[cube(launch)]
fn wide_split(
    base: u32,
    cnt: u32,
    li: u32,
    split: &[u32],
    heavy: &[u32],
    skip_t: u32,
    up_first: &[u32],
    up_head: &[u32],
    w: &[u16],
    d32: &mut [u32],
    p: u32,
    mask: &[u32],
    #[comptime] lp: u32,
) {
    let mut part_buf = Shared::<[u32]>::new_slice(256usize);
    let mut t = ABSOLUTE_POS as u32;
    let tid = UNIT_POS;
    // Role: 0u32 none, 1u32 heavy, 2u32 light.
    let mut role = 0u32;
    let mut v = 0u32;
    let mut sub = 0u32;
    let mut g = 1u32;
    let mut first = split[(6u32 * li) as usize];
    let mut found = false;
    for c in 0..5u32 {
        if !found {
            let gc = if (32u32 >> c) < (32u32 / lp) { 32u32 >> c } else { 32u32 / lp };
            let n_c = split[(6u32 * li + 1u32 + c) as usize];
            let span = n_c * gc * lp;
            if t < span {
                role = 1u32;
                v = heavy[(first + t / (gc * lp)) as usize];
                sub = t % (gc * lp);
                g = gc;
                found = true;
            } else {
                t -= span;
                first += n_c;
            }
        }
    }
    if !found && t < cnt * lp {
        v = base + t / lp;
        sub = t % lp;
        let deg = up_first[(v + 1u32) as usize] - up_first[v as usize];
        if deg <= skip_t {
            role = 2u32;
        }
    }
    let lane = sub % lp;
    let part = sub / lp;
    let s = role != 0u32 && seeded(mask, v);
    let k0 = if role != 0u32 { up_first[v as usize] } else { 0u32 };
    let k1 = if role != 0u32 { up_first[(v + 1u32) as usize] } else { 0u32 };
    let mut q0 = 0u32;
    // The same trip count for every thread of the cube: the barriers are uniform.
    while q0 < p {
        let q = q0 + lane;
        let mut acc = 0xFFFF_FFFFu32;
        if role == 1u32 && q < p {
            let mut k = k0 + part;
            while k < k1 {
                let x = d32[(up_head[k as usize] * p + q) as usize];
                acc = vmin2(acc, vadd2(x, wpair(w, k)));
                k += g;
            }
        }
        if role == 2u32 && q < p {
            acc = if s { d32[(v * p + q) as usize] } else { 0xFFFF_FFFFu32 };
            for k in k0..k1 {
                let x = d32[(up_head[k as usize] * p + q) as usize];
                acc = vmin2(acc, vadd2(x, wpair(w, k)));
            }
            d32[(v * p + q) as usize] = acc;
        }
        part_buf[tid as usize] = acc;
        sync_cube();
        if role == 1u32 && part == 0u32 && q < p {
            for j in 1..g {
                acc = vmin2(acc, part_buf[(tid + j * lp) as usize]);
            }
            if s {
                acc = vmin2(acc, d32[(v * p + q) as usize]);
            }
            d32[(v * p + q) as usize] = acc;
        }
        sync_cube();
        q0 += lp;
    }
}

fn lanes_for(p: u32) -> u32 {
    let mut lp = 1;
    while lp < p && lp < 32 {
        lp <<= 1;
    }
    lp
}

fn u32s(x: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 4) }
}
fn u16s(x: &[u16]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 2) }
}

/// Per lane width: the split table on the device and on the host.
struct Split {
    table: Handle,
    heavy: Handle,
    /// Per wide level, the heavy nodes per class.
    counts: Vec<[u32; 5]>,
    t: u32,
}

pub struct CubeSweep {
    client: Client,
    n: usize,
    level_first: Vec<u32>,
    top: usize,
    max_top_level: usize,
    top_nodes: usize,
    max_cols: u32,
    d_level_first: Handle,
    d_up_first: Handle,
    d_up_head: Handle,
    d_w: [Handle; 2],
    d_top_arc: Handle,
    d_state: Handle,
    d_mask: Handle,
    state_words: usize,
    mask_words: usize,
    splits: Vec<Option<Split>>,
    last: Option<(Handle, u32)>,
    graphs: Vec<((u32, bool), Graph)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Schedule {
    PerLevel,
    Graph,
}

impl CubeSweep {
    pub fn new(c: &Layered, max_cols: u32) -> Result<CubeSweep> {
        let client = CudaRuntime::client(&CudaDevice::new(0));
        let p = max_cols.div_ceil(2) as usize;
        let state_words = c.n * p;
        let mask_words = c.n.div_ceil(32);
        let d_state = client.empty(state_words * 4);
        let d_mask = client.create_from_slice(u32s(&vec![0u32; mask_words]));
        let top_arc = c.top_arcs();
        let mut splits = Vec::new();
        for (i, &t) in SPLIT_DEFAULT.iter().enumerate() {
            let lp = 1u32 << i;
            let (table, heavy) = c.split(lp, t);
            let mut flat = Vec::with_capacity(table.len() * 6);
            let mut counts = Vec::with_capacity(table.len());
            for (first, count) in &table {
                flat.push(*first);
                flat.extend_from_slice(count);
                counts.push(*count);
            }
            flat.push(0);
            let heavy = if heavy.is_empty() { vec![0] } else { heavy };
            splits.push(Some(Split { table: client.create_from_slice(u32s(&flat)), heavy: client.create_from_slice(u32s(&heavy)), counts, t }));
        }
        Ok(CubeSweep {
            n: c.n,
            level_first: c.level_first.clone(),
            top: c.top,
            max_top_level: c.max_top_level().max(1) as usize,
            top_nodes: c.level_first[c.top].max(1) as usize,
            max_cols,
            d_level_first: client.create_from_slice(u32s(&c.level_first)),
            d_up_first: client.create_from_slice(u32s(&c.up_first)),
            d_up_head: client.create_from_slice(u32s(&c.up_head)),
            d_w: [client.create_from_slice(u16s(&c.w_fwd)), client.create_from_slice(u16s(&c.w_rev))],
            d_top_arc: client.create_from_slice(u32s(&top_arc)),
            d_state,
            d_mask,
            state_words,
            mask_words,
            splits,
            last: None,
            graphs: Vec::new(),
            client,
        })
    }

    fn arg(h: &Handle, len: usize) -> BufferArg {
        unsafe { BufferArg::from_raw_parts(h.clone(), len) }
    }

    /// The downward sweep's launches: the top, then one per wide level.
    fn enqueue_down(&self, cols: u32, reversed: bool) {
        let p = cols.div_ceil(2);
        let lp = lanes_for(p);
        let cl = &self.client;
        let w = &self.d_w[reversed as usize];
        let m = cl_len(w, 2);
        if self.top > 0 {
            top_resident::launch(
                cl,
                CubeCount::Static(cols, 1, 1),
                CubeDim::new_1d(TOP_THREADS),
                Self::arg(&self.d_level_first, self.level_first.len()),
                Self::arg(&self.d_up_first, self.n + 1),
                Self::arg(&self.d_top_arc, cl_len(&self.d_top_arc, 4)),
                Self::arg(w, m),
                Self::arg(&self.d_state, self.state_words * 2),
                Self::arg(&self.d_mask, self.mask_words),
                p,
                self.top,
                self.max_top_level,
                self.top_nodes,
            );
        }
        let sp = if lp < 32 { self.splits[lp.trailing_zeros() as usize].as_ref() } else { None };
        for l in self.top..self.level_first.len() - 1 {
            let base = self.level_first[l];
            let cnt = self.level_first[l + 1] - base;
            let li = l - self.top;
            match sp {
                Some(s) if s.t != 0 && s.counts[li].iter().any(|&x| x > 0) => {
                    // Items: as in research `wide_items`, computed on the host from the table.
                    let items = cnt * lp + (0..5).map(|c| s.counts[li][c] * (32u32 >> c).min(32 / lp) * lp).sum::<u32>();
                    wide_split::launch(
                        cl,
                        CubeCount::Static(items.div_ceil(WIDE_THREADS), 1, 1),
                        CubeDim::new_1d(WIDE_THREADS),
                        base,
                        cnt,
                        li as u32,
                        Self::arg(&s.table, cl_len(&s.table, 4)),
                        Self::arg(&s.heavy, cl_len(&s.heavy, 4)),
                        s.t,
                        Self::arg(&self.d_up_first, self.n + 1),
                        Self::arg(&self.d_up_head, cl_len(&self.d_up_head, 4)),
                        Self::arg(w, m),
                        Self::arg(&self.d_state, self.state_words),
                        p,
                        Self::arg(&self.d_mask, self.mask_words),
                        lp,
                    );
                }
                _ => {
                    wide_level::launch(
                        cl,
                        CubeCount::Static((cnt * lp).div_ceil(WIDE_THREADS), 1, 1),
                        CubeDim::new_1d(WIDE_THREADS),
                        base,
                        cnt,
                        Self::arg(&self.d_up_first, self.n + 1),
                        Self::arg(&self.d_up_head, cl_len(&self.d_up_head, 4)),
                        Self::arg(w, m),
                        Self::arg(&self.d_state, self.state_words),
                        p,
                        Self::arg(&self.d_mask, self.mask_words),
                        lp,
                    );
                }
            }
        }
    }

    /// One sweep of `cols` columns from `entries` (new ids): seeds, then the downward
    /// launches, then a wait for the device.
    pub fn sweep(&mut self, cols: u32, entries: &[Entry], reversed: bool, sched: Schedule) -> Result<()> {
        ensure!(cols >= 1 && cols <= self.max_cols, "cols {cols} outside 1..{}", self.max_cols);
        let p = cols.div_ceil(2);
        let cl = self.client.clone();
        if let Some((h, count)) = self.last.take() {
            clear_mask::launch(&cl, CubeCount::Static(count.div_ceil(256), 1, 1), CubeDim::new_1d(256), Self::arg(&h, 3 * count as usize), count, Self::arg(&self.d_mask, self.mask_words));
        }
        let count = entries.len() as u32;
        if count > 0 {
            let flat: Vec<u32> = entries.iter().flat_map(|e| [e.node, e.col, e.value]).collect();
            let h = cl.create_from_slice(u32s(&flat));
            let cubes = CubeCount::Static(count.div_ceil(256), 1, 1);
            seed_rows::launch(&cl, cubes.clone(), CubeDim::new_1d(256), Self::arg(&h, flat.len()), count, Self::arg(&self.d_state, self.state_words), p, Self::arg(&self.d_mask, self.mask_words));
            seed_values::launch(&cl, cubes, CubeDim::new_1d(256), Self::arg(&h, flat.len()), count, Self::arg(&self.d_state, self.state_words * 2), p);
            self.last = Some((h, count));
        }
        match sched {
            Schedule::PerLevel => self.enqueue_down(cols, reversed),
            Schedule::Graph => {
                let key = (cols, reversed);
                if !self.graphs.iter().any(|g| g.0 == key) {
                    // CubeCL's capture protocol: prepare, one warm-up run (it sweeps for real,
                    // from the same seeds, so the labels stay those of this sweep), record.
                    cl.graph_prepare().map_err(|e| anyhow::anyhow!("graph prepare: {e:?}"))?;
                    self.enqueue_down(cols, reversed);
                    cl.start_capture().map_err(|e| anyhow::anyhow!("start capture: {e:?}"))?;
                    self.enqueue_down(cols, reversed);
                    let g = cl.stop_capture().map_err(|e| anyhow::anyhow!("stop capture: {e:?}"))?;
                    self.graphs.push((key, g));
                }
                let g = &self.graphs.iter().find(|g| g.0 == key).context("graph")?.1;
                unsafe { g.replay() }.map_err(|e| anyhow::anyhow!("replay: {e:?}"))?;
            }
        }
        cubecl::future::block_on(cl.sync()).map_err(|e| anyhow::anyhow!("sync: {e:?}"))?;
        Ok(())
    }

    /// Column 0's labels, in new ids.
    pub fn download(&self, cols: u32) -> Result<Vec<u16>> {
        let p = cols.div_ceil(2) as usize;
        let bytes = self.client.read_one(self.d_state.clone()).map_err(|e| anyhow::anyhow!("read: {e:?}"))?;
        let all: &[u16] = unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const u16, bytes.len() / 2) };
        Ok((0..self.n).map(|v| all[v * 2 * p]).collect())
    }
}

fn cl_len(h: &Handle, elem: usize) -> usize {
    h.size() as usize / elem
}
