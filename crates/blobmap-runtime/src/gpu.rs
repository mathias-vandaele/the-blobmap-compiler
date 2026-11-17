//! The downward sweep on a GPU, in CubeCL, on its CUDA runtime or its wgpu one (Vulkan,
//! Metal, DX12). A port of the research kernels (`cuda/sweep.cu`: the resident top with the
//! warp merge, one launch per wide level, lane groups for high-degree nodes), in 32-bit words
//! only, as WGSL has no 16-bit storage:
//!
//! - labels are u16 pairs, two columns per u32 word, P = ceil(columns / 2) words per node;
//! - the model's u16 weights are read as packed u32 pairs, uploaded as they are;
//! - the resident top runs one cube per word, with a lower and an upper accumulator;
//! - links are bound one wide level at a time (a slice of a chunk at an aligned offset),
//!   with row offsets relative to the level, so no index needs more than 32 bits and no
//!   buffer more than the device's largest binding;
//! - pruned columns follow `sweep`'s rules, with the top never pruned.

// Inside `#[cube]` code a cast that clippy sees as u32 → u32 is the usize builtin's, and a
// comptime `if` cannot merge with a runtime one.
#![allow(clippy::unnecessary_cast, clippy::collapsible_if)]

use anyhow::{Result, anyhow, ensure};
use cubecl::client::Client;
use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::graph::Graph;
use crate::heads::Labels;
use crate::sweep::{Bounds, Column, DEAD};
use crate::vocab::Dir;

const WIDE_THREADS: u32 = 256;
/// Research `TOP_CAP` and `WIDE_SPLIT_DEFAULT` (per lane width 1, 2, 4, 8, 16).
const TOP_CAP: u32 = 512;
const SPLIT_DEFAULT: [u32; 5] = [4, 8, 16, 16, 0];
/// Level link ranges start at multiples of this many links (256 bytes of u16 weights, the
/// largest storage offset alignment WebGPU allows).
const ALIGN_LINKS: u64 = 128;
const NONE: u32 = u32::MAX;

// ---- kernels -------------------------------------------------------------------------

#[cube]
fn gid() -> u32 {
    let x = CUBE_POS_Y as u32 * CUBE_COUNT_X as u32 + CUBE_POS_X as u32;
    x * CUBE_DIM_X + UNIT_POS_X
}

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

/// Link k's u16 weight of a packed array, in both halves.
#[cube]
fn wpair(w: &[u32], k: u32) -> u32 {
    let x = (w[(k >> 1u32) as usize] >> ((k & 1u32) << 4u32)) & 0xFFFFu32;
    x | (x << 16u32)
}

/// One half of a pruned label (`sweep`'s rules): above the bound → INF; an unreached node
/// whose entry bound is above it, or whose parent is dead → DEAD.
#[cube]
fn prune_half(acc: u32, lb: u32, t: u32, s: bool, dead: bool) -> u32 {
    let mut a = if acc > t { 0xFFFFu32 } else { acc };
    if (!s && lb > t) || dead {
        a = 0xFFFEu32;
    }
    a
}

#[cube]
fn prune_word(acc: u32, lb: u32, t: u32, s: bool, dl: bool, dh: bool) -> u32 {
    let lo = prune_half(acc & 0xFFFFu32, lb & 0xFFFFu32, t & 0xFFFFu32, s, dl);
    let hi = prune_half(acc >> 16u32, lb >> 16u32, t >> 16u32, s, dh);
    lo | (hi << 16u32)
}

/// Whether a parent's word marks a pruned half: a wide parent, DEAD in a pruned column.
#[cube]
fn dead_half(pd: u32, t: u32) -> bool {
    pd == 0xFFFEu32 && t < 0xFFFEu32
}

/// Seeds: per seeded node, its id and its P words.
#[cube(launch)]
fn seed_write(seeds: &[u32], count: u32, d32: &mut [u32], p: u32) {
    let i = gid();
    if i < count {
        let r = i * (p + 1u32);
        let v = seeds[r as usize];
        for q in 0..p {
            d32[(v * p + q) as usize] = seeds[(r + 1u32 + q) as usize];
        }
    }
}

/// The seed mask: (word, bits) pairs, set, or cleared when `clear` is 1.
#[cube(launch)]
fn mask_write(upd: &[u32], count: u32, mask: &mut [u32], clear: u32) {
    let i = gid();
    if i < count {
        let x = upd[(2u32 * i + 1u32) as usize];
        mask[upd[(2u32 * i) as usize] as usize] = if clear == 1u32 { 0u32 } else { x };
    }
}

#[cube(launch)]
fn copy_words(src: &[u32], count: u32, dst: &mut [u32]) {
    let i = gid();
    if i < count {
        dst[i as usize] = src[i as usize];
    }
}

/// The narrow top, one cube per word (two columns), labels resident in shared memory; a
/// thread takes two arcs per round and a plane's arcs merge in one reduction when they share
/// a tail (research `top_resident<false, MERGE_WARP>`).
#[cube(launch)]
fn top_resident(
    level_first: &[u32],
    arc_first: &[u32],
    top_arc: &[u32],
    top_w: &[u32],
    d32: &mut [u32],
    mask: &[u32],
    p: u32,
    #[comptime] top_levels: usize,
    #[comptime] max_level: usize,
    #[comptime] top_nodes: usize,
    #[comptime] merge: bool,
) {
    let mut lf = Shared::<[u32]>::new_slice(top_levels + 1);
    let mut af = Shared::<[u32]>::new_slice(top_levels + 1);
    let acc_lo = Shared::<[Atomic<u32>]>::new_slice(2 * max_level);
    let acc_hi = Shared::<[Atomic<u32>]>::new_slice(2 * max_level);
    let mut dist = Shared::<[u32]>::new_slice(top_nodes);
    let q = CUBE_POS_X as u32;
    let nt = CUBE_DIM_X;
    let tid = UNIT_POS_X;
    let lane = UNIT_POS_PLANE;
    let tl = top_levels as u32;
    let ml = max_level as u32;
    let mut i = tid;
    while i <= tl {
        lf[i as usize] = level_first[i as usize];
        af[i as usize] = arc_first[i as usize];
        i += nt;
    }
    sync_cube();
    let tn = lf[tl as usize];
    let mut v = tid;
    while v < tn {
        dist[v as usize] = if seeded(mask, v) { d32[(v * p + q) as usize] } else { 0xFFFF_FFFFu32 };
        v += nt;
    }
    sync_cube();
    let mut i = tid;
    while i < lf[1usize] {
        acc_lo[i as usize].store(dist[i as usize] & 0xFFFFu32);
        acc_hi[i as usize].store(dist[i as usize] >> 16u32);
        i += nt;
    }
    sync_cube();
    for l in 0..tl {
        let off = (l & 1u32) * ml;
        let a1 = af[(l + 1u32) as usize];
        let mut i = af[l as usize] + tid;
        // A plane-uniform trip count (i - lane is the same across the plane).
        while i - lane < a1 {
            for h in 0..2u32 {
                let k = i + h * nt;
                let live = k < a1;
                let kk = if live { k } else { 0u32 };
                let r = top_arc[kk as usize];
                let mut y = 0xFFFF_FFFFu32;
                let mut t = 0xFFFF_FFFFu32;
                if live {
                    y = vadd2(dist[(r & 0xFFFFu32) as usize], wpair(top_w, kk));
                    t = r >> 16u32;
                }
                let ylo = y & 0xFFFFu32;
                let yhi = y >> 16u32;
                if merge {
                    if plane_all(t == plane_broadcast(t, 0u32)) {
                        let mlo = plane_min(ylo);
                        let mhi = plane_min(yhi);
                        if lane == 0u32 && live {
                            acc_lo[(off + t) as usize].fetch_min(mlo);
                            acc_hi[(off + t) as usize].fetch_min(mhi);
                        }
                    } else if live {
                        acc_lo[(off + t) as usize].fetch_min(ylo);
                        acc_hi[(off + t) as usize].fetch_min(yhi);
                    }
                } else if live {
                    acc_lo[(off + t) as usize].fetch_min(ylo);
                    acc_hi[(off + t) as usize].fetch_min(yhi);
                }
            }
            i += 2u32 * nt;
        }
        sync_cube();
        let base = lf[l as usize];
        let cnt = lf[(l + 1u32) as usize] - base;
        let mut i = tid;
        while i < cnt {
            let val = acc_lo[(off + i) as usize].load() | (acc_hi[(off + i) as usize].load() << 16u32);
            dist[(base + i) as usize] = val;
            d32[((base + i) * p + q) as usize] = val;
            i += nt;
        }
        if l + 1u32 < tl {
            let noff = ml - off;
            let nb = lf[(l + 1u32) as usize];
            let ncnt = lf[(l + 2u32) as usize] - nb;
            let mut i = tid;
            while i < ncnt {
                acc_lo[(noff + i) as usize].store(dist[(nb + i) as usize] & 0xFFFFu32);
                acc_hi[(noff + i) as usize].store(dist[(nb + i) as usize] >> 16u32);
                i += nt;
            }
        }
        sync_cube();
    }
}

/// One wide level, `lp` lanes per node, each lane owning words q ≡ lane (mod lp). `up_head`,
/// `w` and `eb` are the level's slices; `rel_first` is indexed by node, relative to the
/// level, with `links` its link count.
#[cube(launch)]
fn wide_level(
    base: u32,
    cnt: u32,
    links: u32,
    rel_first: &[u32],
    up_head: &[u32],
    w: &[u32],
    d32: &mut [u32],
    p: u32,
    mask: &[u32],
    parent: &[u32],
    eb: &[u32],
    bound: &[u32],
    top_nodes: u32,
    #[comptime] lp: u32,
    #[comptime] prune: bool,
) {
    let t = gid();
    if t < cnt * lp {
        let li = t / lp;
        let v = base + li;
        let lane = t % lp;
        let s = seeded(mask, v);
        let k0 = rel_first[v as usize];
        let k1 = if li + 1u32 < cnt { rel_first[(v + 1u32) as usize] } else { links };
        let mut par = 0xFFFF_FFFFu32;
        if prune {
            par = parent[v as usize];
        }
        let mut q = lane;
        while q < p {
            let mut acc = if s { d32[(v * p + q) as usize] } else { 0xFFFF_FFFFu32 };
            let mut lb = 0xFFFF_FFFFu32;
            let mut dl = false;
            let mut dh = false;
            let mut tq = 0xFFFF_FFFFu32;
            if prune {
                tq = bound[q as usize];
                if par != 0xFFFF_FFFFu32 && par >= top_nodes {
                    let pd = d32[(par * p + q) as usize];
                    dl = dead_half(pd & 0xFFFFu32, tq & 0xFFFFu32);
                    dh = dead_half(pd >> 16u32, tq >> 16u32);
                }
            }
            if !(dl && dh) {
                for k in k0..k1 {
                    let x = d32[(up_head[k as usize] * p + q) as usize];
                    acc = vmin2(acc, vadd2(x, wpair(w, k)));
                    if prune {
                        lb = vmin2(lb, vadd2(x, wpair(eb, k)));
                    }
                }
            }
            if prune {
                acc = prune_word(acc, lb, tq, s, dl, dh);
            }
            d32[(v * p + q) as usize] = acc;
            q += lp;
        }
    }
}

/// A wide level with heavy nodes (research `wide_level_split`): items are the heavy classes
/// (32, 16, 8, 4, 2 lane groups of `lp` lanes, a group taking every G-th arc), then every
/// node with `lp` lanes, skipping those of more than `skip_t` arcs. `split` holds, per wide
/// level, the first heavy node and the five class counts; `li` is this level's row there.
#[cube(launch)]
fn wide_split(
    base: u32,
    cnt: u32,
    links: u32,
    li: u32,
    split: &[u32],
    heavy: &[u32],
    skip_t: u32,
    rel_first: &[u32],
    up_head: &[u32],
    w: &[u32],
    d32: &mut [u32],
    p: u32,
    mask: &[u32],
    parent: &[u32],
    eb: &[u32],
    bound: &[u32],
    top_nodes: u32,
    #[comptime] lp: u32,
    #[comptime] prune: bool,
) {
    let mut part_acc = Shared::<[u32]>::new_slice(256usize);
    let mut part_lb = Shared::<[u32]>::new_slice(256usize);
    let mut t = gid();
    let tid = UNIT_POS_X;
    // Role: 0 none, 1 heavy, 2 light.
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
    let mut k0 = 0u32;
    let mut k1 = 0u32;
    if !found && t < cnt * lp {
        v = base + t / lp;
        sub = t % lp;
        role = 2u32;
    }
    if role != 0u32 {
        k0 = rel_first[v as usize];
        k1 = if v + 1u32 < base + cnt { rel_first[(v + 1u32) as usize] } else { links };
        if role == 2u32 && k1 - k0 > skip_t {
            role = 0u32;
        }
    }
    let lane = sub % lp;
    let part = sub / lp;
    let s = role != 0u32 && seeded(mask, v);
    let mut par = 0xFFFF_FFFFu32;
    if prune {
        if role != 0u32 {
            par = parent[v as usize];
        }
    }
    let mut q0 = 0u32;
    // The same trip count for every thread of the cube: the barriers are uniform.
    while q0 < p {
        let q = q0 + lane;
        let mut acc = 0xFFFF_FFFFu32;
        let mut lb = 0xFFFF_FFFFu32;
        let mut dl = false;
        let mut dh = false;
        let mut tq = 0xFFFF_FFFFu32;
        if prune && role != 0u32 && q < p {
            tq = bound[q as usize];
            if par != 0xFFFF_FFFFu32 && par >= top_nodes {
                let pd = d32[(par * p + q) as usize];
                dl = dead_half(pd & 0xFFFFu32, tq & 0xFFFFu32);
                dh = dead_half(pd >> 16u32, tq >> 16u32);
            }
        }
        if role == 1u32 && q < p && !(dl && dh) {
            let mut k = k0 + part;
            while k < k1 {
                let x = d32[(up_head[k as usize] * p + q) as usize];
                acc = vmin2(acc, vadd2(x, wpair(w, k)));
                if prune {
                    lb = vmin2(lb, vadd2(x, wpair(eb, k)));
                }
                k += g;
            }
        }
        if role == 2u32 && q < p {
            acc = if s { d32[(v * p + q) as usize] } else { 0xFFFF_FFFFu32 };
            if !(dl && dh) {
                for k in k0..k1 {
                    let x = d32[(up_head[k as usize] * p + q) as usize];
                    acc = vmin2(acc, vadd2(x, wpair(w, k)));
                    if prune {
                        lb = vmin2(lb, vadd2(x, wpair(eb, k)));
                    }
                }
            }
            if prune {
                acc = prune_word(acc, lb, tq, s, dl, dh);
            }
            d32[(v * p + q) as usize] = acc;
        }
        part_acc[tid as usize] = acc;
        part_lb[tid as usize] = lb;
        sync_cube();
        if role == 1u32 && part == 0u32 && q < p {
            for j in 1..g {
                acc = vmin2(acc, part_acc[(tid + j * lp) as usize]);
                lb = vmin2(lb, part_lb[(tid + j * lp) as usize]);
            }
            if s {
                acc = vmin2(acc, d32[(v * p + q) as usize]);
            }
            if prune {
                acc = prune_word(acc, lb, tq, s, dl, dh);
            }
            d32[(v * p + q) as usize] = acc;
        }
        sync_cube();
        q0 += lp;
    }
}


// ---- heads (FORMAT §8; `heads` has the host passes and their meaning) ----------------

/// Car `From`: each copy's words, gathered (pair j's copy `cs_c[j]`).
#[cube(launch)]
fn copy_gather(cs_c: &[u32], count: u32, d32: &[u32], tmp: &mut [u32], p: u32) {
    let j = gid();
    if j < count {
        let c = cs_c[j as usize];
        for q in 0..p {
            tmp[(j * p + q) as usize] = d32[(c * p + q) as usize];
        }
    }
}

/// Car `From`: a node with copies takes the least of its label and theirs (h = its arrivals).
#[cube(launch)]
fn copy_fold(cs_s: &[u32], cs_first: &[u32], count: u32, tmp: &[u32], d32: &mut [u32], p: u32) {
    let r = gid();
    if r < count {
        let s = cs_s[r as usize];
        let j0 = cs_first[r as usize];
        let j1 = cs_first[(r + 1u32) as usize];
        for q in 0..p {
            let mut acc = d32[(s * p + q) as usize];
            for j in j0..j1 {
                acc = vmin2(acc, tmp[(j * p + q) as usize]);
            }
            d32[(s * p + q) as usize] = acc;
        }
    }
}

/// Car `To`: h(u) = the least exit turn ⊕ its head's label, for new ids u0 .. u0 + count
/// (`tf` relative to the chunk, heads `th` in new ids, costs `tc` as u16 pairs).
#[cube(launch)]
fn turn_pass(u0: u32, count: u32, tf: &[u32], th: &[u32], tc: &[u32], d32: &[u32], h: &mut [u32], p: u32) {
    let j = gid();
    if j < count {
        let u = u0 + j;
        let k0 = tf[j as usize];
        let k1 = tf[(j + 1u32) as usize];
        for q in 0..p {
            let mut acc = 0xFFFF_FFFFu32;
            for k in k0..k1 {
                acc = vmin2(acc, vadd2(wpair(tc, k), d32[(th[k as usize] * p + q) as usize]));
            }
            h[(u * p + q) as usize] = acc;
        }
    }
}

#[cube]
fn sadd(a: u32, b: u32) -> u32 {
    let s = a + b;
    if s < 0xFFFFu32 { s } else { 0xFFFFu32 }
}

/// The direct term (FORMAT §8) from offsets oa to ob (offsets 0 and 2 of each ID).
#[cube]
fn direct_term(oa0: u32, oa2: u32, ob0: u32, ob2: u32, p0: bool, p1: bool) -> u32 {
    let mut d = 0xFFFFu32;
    if p0 && oa0 != 0xFFFFu32 && ob0 >= oa0 {
        d = ob0 - oa0;
    }
    if p1 && oa2 != 0xFFFFu32 && ob2 >= oa2 {
        let x = ob2 - oa2;
        if x < d {
            d = x;
        }
    }
    d
}

/// Heads at IDs i0 .. i0 + count, the first at row `voff` of `vt`: 5 words an ID (layout ids a, b; offsets 0|1, 2|3;
/// road). `h`: per new id, P words of h (CAR `To`: the turn pass; else the labels). `src`:
/// the sources, 8 words each, sorted by piece: [a, b, road, column, start, offsets 0|2,
/// index, 0]. Dense: out[j * p + q]. Sparse: (index, column << 16 | time) pairs for times at
/// most the column's `mx` half, appended at counter[0], kept while below `cap`.
#[cube(launch)]
fn heads_ids(
    vt: &[u32],
    voff: u32,
    i0: u32,
    count: u32,
    first: u32,
    n: u32,
    pos: &[u32],
    h: &[u32],
    p: u32,
    cols: u32,
    koff: u32,
    car: u32,
    bound: &[u32],
    prune: u32,
    src: &[u32],
    nsrc: u32,
    mx: &[u32],
    sparse: u32,
    out: &mut [u32],
    counter: &mut [Atomic<u32>],
    cap: u32,
) {
    let j = gid();
    if j < count {
        let i = i0 + j;
        let e = (voff + j) * 5u32;
        let a = vt[e as usize];
        let b = vt[(e + 1u32) as usize];
        let o01 = vt[(e + 2u32) as usize];
        let o23 = vt[(e + 3u32) as usize];
        let road = vt[(e + 4u32) as usize];
        let x = if a != NONE { a } else { b };
        let member = x != NONE && x >= first && x - first < n;
        if !member {
            if sparse == 0u32 {
                for q in 0..p {
                    out[(j * p + q) as usize] = 0xFFFF_FFFFu32;
                }
            }
        } else {
            let sh = koff * 16u32;
            let oa = (o01 >> sh) & 0xFFFFu32;
            let ob = (o23 >> sh) & 0xFFFFu32;
            let offa = oa | (oa << 16u32);
            let offb = ob | (ob << 16u32);
            let mut ha = 0u32;
            let mut hb = 0u32;
            if a != NONE {
                ha = pos[(a - first) as usize];
            }
            if b != NONE {
                hb = pos[(b - first) as usize];
            }
            // The first source on this ID's piece, by binary search.
            let mut lo = 0u32;
            let mut hi = nsrc;
            while lo < hi {
                let mid = (lo + hi) / 2u32;
                let m = mid * 8u32;
                let sa = src[m as usize];
                let sb = src[(m + 1u32) as usize];
                let sr = src[(m + 2u32) as usize];
                let less = sa < a || (sa == a && (sb < b || (sb == b && sr < road)));
                if less {
                    lo = mid + 1u32;
                } else {
                    hi = mid;
                }
            }
            let s0 = lo;
            let p0 = car == 0u32 || a != NONE;
            let p1 = car == 0u32 || b != NONE;
            let io0 = o01 & 0xFFFFu32;
            let io2 = o23 & 0xFFFFu32;
            for q in 0..p {
                let mut acc = 0xFFFF_FFFFu32;
                if a != NONE {
                    acc = vmin2(acc, vadd2(h[(ha * p + q) as usize], offa));
                }
                if b != NONE {
                    acc = vmin2(acc, vadd2(h[(hb * p + q) as usize], offb));
                }
                for r in s0..nsrc {
                    let m = r * 8u32;
                    if src[m as usize] != a || src[(m + 1u32) as usize] != b || src[(m + 2u32) as usize] != road {
                        break;
                    }
                    let c = src[(m + 3u32) as usize];
                    if c / 2u32 == q {
                        let start = src[(m + 4u32) as usize];
                        let mut t = start;
                        if src[(m + 6u32) as usize] != i {
                            let so = src[(m + 5u32) as usize];
                            let so0 = so & 0xFFFFu32;
                            let so2 = so >> 16u32;
                            let d = if koff == 0u32 { direct_term(so0, so2, io0, io2, p0, p1) } else { direct_term(io0, io2, so0, so2, p0, p1) };
                            t = sadd(start, d);
                        }
                        let tt = if (c & 1u32) == 0u32 { t | 0xFFFF_0000u32 } else { (t << 16u32) | 0xFFFFu32 };
                        acc = vmin2(acc, tt);
                    }
                }
                if prune == 1u32 {
                    let tq = bound[q as usize];
                    let l = acc & 0xFFFFu32;
                    let u = acc >> 16u32;
                    let l2 = if l > (tq & 0xFFFFu32) { 0xFFFFu32 } else { l };
                    let u2 = if u > (tq >> 16u32) { 0xFFFFu32 } else { u };
                    acc = l2 | (u2 << 16u32);
                }
                if sparse == 0u32 {
                    out[(j * p + q) as usize] = acc;
                } else {
                    let m = mx[q as usize];
                    let c0 = 2u32 * q;
                    let t0 = acc & 0xFFFFu32;
                    if c0 < cols && t0 <= (m & 0xFFFFu32) {
                        let k = counter[0].fetch_add(1u32);
                        if k < cap {
                            out[(2u32 * k) as usize] = i;
                            out[(2u32 * k + 1u32) as usize] = (c0 << 16u32) | t0;
                        }
                    }
                    let t1 = acc >> 16u32;
                    if c0 + 1u32 < cols && t1 <= (m >> 16u32) {
                        let k = counter[0].fetch_add(1u32);
                        if k < cap {
                            out[(2u32 * k) as usize] = i;
                            out[(2u32 * k + 1u32) as usize] = ((c0 + 1u32) << 16u32) | t1;
                        }
                    }
                }
            }
        }
    }
}

// ---- host side -----------------------------------------------------------------------

fn bytes_u32(x: &[u32]) -> &[u8] {
    bytemuck::cast_slice(x)
}

/// u16s padded to whole u32 words.
fn bytes_u16(x: &[u16]) -> Vec<u8> {
    let mut b = bytemuck::cast_slice::<u16, u8>(x).to_vec();
    while !b.len().is_multiple_of(4) || b.is_empty() {
        b.push(0xFF);
    }
    b
}

fn lanes_for(p: u32) -> u32 {
    let mut lp = 1;
    while lp < p && lp < 32 {
        lp <<= 1;
    }
    lp
}

/// A grid of `items` threads in cubes of `dim`, two-dimensional past the device's limit.
fn grid(items: u32, dim: u32, max_x: u32) -> CubeCount {
    let cubes = items.div_ceil(dim).max(1);
    if cubes <= max_x { CubeCount::Static(cubes, 1, 1) } else { CubeCount::Static(max_x, cubes.div_ceil(max_x), 1) }
}

fn arg(h: &Handle, len: usize) -> BufferArg {
    unsafe { BufferArg::from_raw_parts(h.clone(), len.max(1)) }
}

/// Which device kind a sweeper runs on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Cuda,
    Wgpu,
}

/// A wide level's links: a chunk, an aligned offset in it, and a count.
struct LevelAt {
    chunk: usize,
    off: u64,
    links: u32,
}

struct Chunk {
    head: Handle,
    /// Per metric and direction (`2 * metric + dir`), the going-down weights and the entry
    /// bounds.
    w: Vec<Handle>,
    eb: Vec<Handle>,
}

struct Split {
    table: Handle,
    heavy: Handle,
    counts: Vec<[u32; 5]>,
    t: u32,
}

impl DevGraph {
    /// Kernel launches of a sweep's downward half: the top, then one per wide level.
    pub fn launches(&self) -> usize {
        (self.top > 0) as usize + self.levels.len()
    }
}

/// One component on the device.
pub struct DevGraph {
    n: usize,
    level_first: Vec<u32>,
    top: usize,
    top_nodes: u32,
    max_top_level: u32,
    p_max: u32,
    rel_first: Handle,
    parent: Handle,
    top_lf: Handle,
    top_af: Handle,
    top_arc: Handle,
    top_w: Vec<Handle>,
    chunks: Vec<Chunk>,
    levels: Vec<LevelAt>,
    splits: Vec<Split>,
    state: Handle,
    mask: Handle,
    bound: Handle,
    last_mask: Option<(Handle, u32)>,
    pub bytes: u64,
    pub pruning: bool,
}

pub struct Gpu {
    pub kind: Kind,
    pub client: Client,
    merge: bool,
    max_shared: usize,
    top_threads: u32,
    max_x: u32,
    max_chunk: u64,
    /// Time each sweep (a device sync after the launches): `times` holds the seconds of the
    /// last sweep's seeding and launches, and of its read-back.
    pub timing: bool,
    pub times: std::cell::Cell<[f64; 2]>,
    /// The last sweep's seconds up to its launches' submission, before the device sync.
    pub enqueue: std::cell::Cell<f64>,
}

impl Gpu {
    /// The device's client, or why there is none (no driver, no adapter).
    pub fn open(kind: Kind) -> Result<Gpu> {
        let client = std::panic::catch_unwind(|| match kind {
            Kind::Cuda => {
                use cubecl::cuda::{CudaDevice, CudaRuntime};
                <CudaRuntime as cubecl_runtime::runtime::Runtime>::client(&CudaDevice::new(0))
            }
            Kind::Wgpu => {
                use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
                <WgpuRuntime as cubecl_runtime::runtime::Runtime>::client(&WgpuDevice::default())
            }
        })
        .map_err(|e| anyhow!("no {kind:?} device: {}", e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default()))?;
        let props = client.properties();
        let hw = &props.hardware;
        let merge = props.features.plane.contains(cubecl::ir::features::Plane::Ops);
        let max_shared = hw.max_shared_memory_size.min(48 * 1024);
        let top_threads = hw.max_units_per_cube.min(1024);
        let max_x = hw.max_cube_count.0.min(65535);
        let max_chunk = props.memory.max_page_size.min(1 << 30);
        Ok(Gpu { kind, merge, max_shared, top_threads, max_x, max_chunk, timing: false, times: std::cell::Cell::new([0.0; 2]), enqueue: std::cell::Cell::new(0.0), client })
    }

    pub fn describe(&self) -> String {
        let p = self.client.properties();
        format!(
            "{:?}: plane merge {}, shared {} B, top threads {}, max chunk {} MiB",
            self.kind,
            self.merge,
            self.max_shared,
            self.top_threads,
            self.max_chunk >> 20
        ) + &format!(", planes {}..{}", p.hardware.plane_size_min, p.hardware.plane_size_max)
    }

    fn upload(&self, bytes: &[u8], used: &mut u64) -> Handle {
        *used += bytes.len() as u64;
        let h = self.client.create_from_slice(bytes);
        // A large upload's staging memory goes back before the next one (wgpu kept it,
        // and a continent's component went past the device).
        if bytes.len() >= 64 << 20 {
            let _ = cubecl::future::block_on(self.client.sync());
            let _ = self.client.memory_cleanup();
        }
        h
    }

    /// The bytes a graph would take on the device (hot set, state for `p_max` words, entry
    /// bounds when given).
    pub fn estimate(g: &Graph, p_max: u32, bounds: bool) -> u64 {
        let m = g.links() as u64;
        let pairs = 2 * g.down.len() as u64;
        let mut b = m * 4 + pairs * m * 2 + g.n as u64 * 8 + g.n as u64 * p_max as u64 * 4 + (g.n as u64).div_ceil(8);
        if bounds {
            b += pairs * m * 2;
        }
        b
    }

    /// Uploads a component: its hot set in the device layout, entry bounds if given.
    pub fn load(&self, g: &Graph, bounds: Option<&Bounds>, p_max: u32) -> Result<DevGraph> {
        let mut used = 0u64;
        let n = g.n;
        ensure!(n < (1 << 31) && (n as u64) * (p_max as u64) < (1u64 << 32), "a component of {n} nodes is too large for 32-bit state indices at {p_max} words a node");
        let levels = g.levels();
        let pairs = 2 * g.down.len();
        // The narrow top, fitted to shared memory.
        let mut top = (0..levels).take_while(|&l| g.width(l) <= TOP_CAP).count();
        let shared = |top: usize| {
            let maxw = (0..top).map(|l| g.width(l) as usize).max().unwrap_or(0);
            (2 * (top + 1) + 4 * maxw + g.level_first[top] as usize) * 4
        };
        while top > 0 && (shared(top) > self.max_shared || g.level_first[top] > 65536) {
            top -= 1;
        }
        let top_nodes = g.level_first[top];
        let max_top_level = (0..top).map(|l| g.width(l)).max().unwrap_or(0);
        // The top's arcs, compact: head | tail's index in its level << 16.
        let top_links = g.up_first[top_nodes as usize] as usize;
        let mut top_arc = vec![0u32; top_links];
        let mut top_af = Vec::with_capacity(top + 1);
        for l in 0..top {
            top_af.push(g.up_first[g.level_first[l] as usize] as u32);
            for v in g.level_first[l]..g.level_first[l + 1] {
                for k in g.row(v) {
                    top_arc[k] = g.up_head[k] | ((v - g.level_first[l]) << 16);
                }
            }
        }
        top_af.push(top_links as u32);
        let mut top_w = Vec::new();
        for m in 0..g.down.len() {
            for d in [Dir::From, Dir::To] {
                top_w.push(self.upload(&bytes_u16(&g.going_down(m, d)[..top_links]), &mut used));
            }
        }
        // Wide levels: chunks of whole levels at aligned offsets.
        let mut rel_first = vec![0u32; n.max(1)];
        let mut at = Vec::new();
        let mut chunk_ranges: Vec<(u64, u64)> = Vec::new(); // link range of the graph per chunk
        let (mut c_start, mut c_len) = (g.up_first[top_nodes as usize], 0u64);
        let max_links = (self.max_chunk / 4).max(ALIGN_LINKS * 2);
        for l in top..levels {
            let (a, b) = (g.level_first[l] as usize, g.level_first[l + 1] as usize);
            let (s, e) = (g.up_first[a], g.up_first[b]);
            ensure!(e - s < (1 << 31), "level {l} has {} links", e - s);
            for v in a..b {
                rel_first[v] = (g.up_first[v] - s) as u32;
            }
            let need = (e - s).div_ceil(ALIGN_LINKS) * ALIGN_LINKS + ALIGN_LINKS;
            if c_len > 0 && c_len + need > max_links {
                chunk_ranges.push((c_start, c_len));
                c_start = s;
                c_len = 0;
            }
            ensure!(need <= max_links, "level {l}'s {} links exceed a device binding", e - s);
            at.push(LevelAt { chunk: chunk_ranges.len(), off: c_len, links: (e - s) as u32 });
            c_len += need;
        }
        chunk_ranges.push((c_start, c_len.max(ALIGN_LINKS)));
        let mut chunks = Vec::new();
        for (ci, &(_, len)) in chunk_ranges.iter().enumerate() {
            let mut head = vec![0u32; len as usize];
            let mut w: Vec<Vec<u16>> = vec![vec![0xFFFF; len as usize]; pairs];
            let mut eb: Vec<Vec<u16>> = if bounds.is_some() { vec![vec![0xFFFF; len as usize]; pairs] } else { Vec::new() };
            for (li, la) in at.iter().enumerate().filter(|(_, la)| la.chunk == ci) {
                let l = top + li;
                let s = g.up_first[g.level_first[l] as usize] as usize;
                let o = la.off as usize;
                let k = la.links as usize;
                head[o..o + k].copy_from_slice(&g.up_head[s..s + k]);
                for m in 0..g.down.len() {
                    for (di, d) in [Dir::From, Dir::To].into_iter().enumerate() {
                        w[2 * m + di][o..o + k].copy_from_slice(&g.going_down(m, d)[s..s + k]);
                        if let Some(bs) = bounds {
                            eb[2 * m + di][o..o + k].copy_from_slice(&bs[2 * m + di][s..s + k]);
                        }
                    }
                }
            }
            chunks.push(Chunk {
                head: self.upload(bytes_u32(&head), &mut used),
                w: w.iter().map(|x| self.upload(&bytes_u16(x), &mut used)).collect(),
                eb: eb.iter().map(|x| self.upload(&bytes_u16(x), &mut used)).collect(),
            });
        }
        // Degree splits per lane width.
        let mut splits = Vec::new();
        for (i, &t) in SPLIT_DEFAULT.iter().enumerate() {
            let lp = 1u32 << i;
            let mut heavy = Vec::new();
            let mut flat = Vec::new();
            let mut counts = Vec::new();
            for l in top..levels {
                let first = heavy.len() as u32;
                let mut count = [0u32; 5];
                if t != 0 && (g.width(l) as u64) * (lp as u64) <= 65536 {
                    let mut h: Vec<(u32, u32)> = (g.level_first[l]..g.level_first[l + 1]).map(|v| (g.row(v).len() as u32, v)).filter(|&(d, _)| d > t).collect();
                    h.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
                    for &(d, _) in &h {
                        let mut groups = 2u32;
                        while groups < 32 && d.div_ceil(groups) > t {
                            groups <<= 1;
                        }
                        count[(32 / groups).trailing_zeros() as usize] += 1;
                    }
                    heavy.extend(h.iter().map(|x| x.1));
                }
                flat.push(first);
                flat.extend_from_slice(&count);
                counts.push(count);
            }
            flat.push(0);
            heavy.push(0);
            splits.push(Split { table: self.upload(bytes_u32(&flat), &mut used), heavy: self.upload(bytes_u32(&heavy), &mut used), counts, t });
        }
        let parent: Vec<u32> = if bounds.is_some() { g.parent.clone() } else { vec![NONE] };
        let state_bytes = (n.max(1) as u64) * p_max as u64 * 4;
        used += state_bytes;
        let mask_words = n.div_ceil(32).max(1);
        Ok(DevGraph {
            n,
            level_first: g.level_first.clone(),
            top,
            top_nodes,
            max_top_level,
            p_max,
            rel_first: self.upload(bytes_u32(&rel_first), &mut used),
            parent: self.upload(bytes_u32(&parent), &mut used),
            top_lf: self.upload(bytes_u32(&g.level_first[..=top]), &mut used),
            top_af: self.upload(bytes_u32(&top_af), &mut used),
            top_arc: self.upload(bytes_u32(if top_arc.is_empty() { &[0] } else { &top_arc }), &mut used),
            top_w,
            chunks,
            levels: at,
            splits,
            state: self.client.empty(state_bytes as usize),
            mask: self.upload(bytes_u32(&vec![0u32; mask_words]), &mut used),
            bound: self.upload(bytes_u32(&vec![u32::MAX; p_max as usize]), &mut used),
            last_mask: None,
            bytes: used,
            pruning: bounds.is_some(),
        })
    }

    fn enqueue_down(&self, d: &DevGraph, p: u32, wi: usize, prune: bool) {
        let cl = &self.client;
        let lp = lanes_for(p);
        let words = d.n * p as usize;
        if d.top > 0 {
            top_resident::launch(
                cl,
                CubeCount::Static(p, 1, 1),
                CubeDim::new_1d(self.top_threads),
                arg(&d.top_lf, d.top + 1),
                arg(&d.top_af, d.top + 1),
                arg(&d.top_arc, (d.top_arc.size() / 4) as usize),
                arg(&d.top_w[wi], (d.top_w[wi].size() / 4) as usize),
                arg(&d.state, words),
                arg(&d.mask, d.n.div_ceil(32)),
                p,
                d.top,
                (d.max_top_level as usize).max(1),
                (d.top_nodes as usize).max(1),
                self.merge,
            );
        }
        let sp = if lp < 32 { Some(&d.splits[lp.trailing_zeros() as usize]) } else { None };
        for (li, la) in d.levels.iter().enumerate() {
            let l = d.top + li;
            let base = d.level_first[l];
            let cnt = d.level_first[l + 1] - base;
            let c = &d.chunks[la.chunk];
            let head = c.head.clone().offset_start(la.off * 4);
            let w = c.w[wi].clone().offset_start(la.off * 2);
            let eb = if prune { c.eb[wi].clone().offset_start(la.off * 2) } else { w.clone() };
            let (hl, wl) = (la.links as usize, (la.links as usize).div_ceil(2));
            match sp {
                Some(s) if s.t != 0 && s.counts[li].iter().any(|&x| x > 0) => {
                    let items = cnt * lp + (0..5).map(|c| s.counts[li][c] * (32u32 >> c).min(32 / lp) * lp).sum::<u32>();
                    wide_split::launch(
                        cl,
                        grid(items, WIDE_THREADS, self.max_x),
                        CubeDim::new_1d(WIDE_THREADS),
                        base,
                        cnt,
                        la.links,
                        li as u32,
                        arg(&s.table, (s.table.size() / 4) as usize),
                        arg(&s.heavy, (s.heavy.size() / 4) as usize),
                        s.t,
                        arg(&d.rel_first, d.n),
                        arg(&head, hl),
                        arg(&w, wl),
                        arg(&d.state, words),
                        p,
                        arg(&d.mask, d.n.div_ceil(32)),
                        arg(&d.parent, (d.parent.size() / 4) as usize),
                        arg(&eb, wl),
                        arg(&d.bound, d.p_max as usize),
                        d.top_nodes,
                        lp,
                        prune,
                    );
                }
                _ => {
                    wide_level::launch(
                        cl,
                        grid(cnt * lp, WIDE_THREADS, self.max_x),
                        CubeDim::new_1d(WIDE_THREADS),
                        base,
                        cnt,
                        la.links,
                        arg(&d.rel_first, d.n),
                        arg(&head, hl),
                        arg(&w, wl),
                        arg(&d.state, words),
                        p,
                        arg(&d.mask, d.n.div_ceil(32)),
                        arg(&d.parent, (d.parent.size() / 4) as usize),
                        arg(&eb, wl),
                        arg(&d.bound, d.p_max as usize),
                        d.top_nodes,
                        lp,
                        prune,
                    );
                }
            }
        }
    }

    /// A batched sweep, left on the device (`labels` or `heads` read it). Whether it pruned.
    pub fn sweep(&self, d: &mut DevGraph, metric: usize, dir: Dir, cols: &[&Column]) -> Result<bool> {
        let t0 = std::time::Instant::now();
        let k = cols.len() as u32;
        ensure!(k >= 1 && k <= 2 * d.p_max, "{k} columns, outside 1..{}", 2 * d.p_max);
        let p = k.div_ceil(2);
        let cl = self.client.clone();
        let dim = CubeDim::new_1d(256);
        if let Some((h, count)) = d.last_mask.take() {
            mask_write::launch(&cl, grid(count, 256, self.max_x), dim, arg(&h, 2 * count as usize), count, arg(&d.mask, d.n.div_ceil(32)), 1u32);
        }
        // Seeds: each seeded node's P words, built on the host.
        let mut rows: Vec<(u32, u32, u16)> = Vec::new();
        for (c, col) in cols.iter().enumerate() {
            rows.extend(col.entries.iter().map(|&(v, t)| (v, c as u32, t)));
        }
        rows.sort_unstable_by_key(|r| (r.0, r.1));
        let mut seeds: Vec<u32> = Vec::new();
        let mut upd: Vec<u32> = Vec::new();
        let mut i = 0;
        while i < rows.len() {
            let v = rows[i].0;
            seeds.push(v);
            let at = seeds.len();
            seeds.extend(std::iter::repeat_n(u32::MAX, p as usize));
            while i < rows.len() && rows[i].0 == v {
                let (_, c, t) = rows[i];
                let x = &mut seeds[at + (c / 2) as usize];
                let sh = 16 * (c % 2);
                *x = (*x & !(0xFFFF << sh)) | ((t as u32) << sh);
                i += 1;
            }
            let wd = v / 32;
            if upd.len() >= 2 && upd[upd.len() - 2] == wd {
                *upd.last_mut().unwrap() |= 1 << (v % 32);
            } else {
                upd.extend([wd, 1 << (v % 32)]);
            }
        }
        let count = (seeds.len() / (p as usize + 1)) as u32;
        if count > 0 {
            let sh = cl.create_from_slice(bytes_u32(&seeds));
            seed_write::launch(&cl, grid(count, 256, self.max_x), dim, arg(&sh, seeds.len()), count, arg(&d.state, d.n * p as usize), p);
            let uc = (upd.len() / 2) as u32;
            let uh = cl.create_from_slice(bytes_u32(&upd));
            mask_write::launch(&cl, grid(uc, 256, self.max_x), dim, arg(&uh, upd.len()), uc, arg(&d.mask, d.n.div_ceil(32)), 0u32);
            d.last_mask = Some((uh, uc));
        }
        let prune = d.pruning && cols.iter().any(|c| c.pruned());
        if prune {
            let mut b = vec![u32::MAX; p as usize];
            for (c, col) in cols.iter().enumerate() {
                let t = if col.pruned() { col.bound as u32 } else { 0xFFFF };
                let sh = 16 * (c % 2);
                b[c / 2] = (b[c / 2] & !(0xFFFF << sh)) | (t << sh);
            }
            let bh = cl.create_from_slice(bytes_u32(&b));
            copy_words::launch(&cl, CubeCount::Static(1, 1, 1), CubeDim::new_1d(p.max(1)), arg(&bh, b.len()), p, arg(&d.bound, d.p_max as usize));
        }
        let wi = 2 * metric + crate::sweep::dir_index(dir);
        self.enqueue_down(d, p, wi, prune);
        self.enqueue.set(t0.elapsed().as_secs_f64());
        if self.timing {
            cubecl::future::block_on(cl.sync()).map_err(|e| anyhow!("sync: {e:?}"))?;
            self.times.set([t0.elapsed().as_secs_f64(), 0.0]);
        }
        Ok(prune)
    }

    /// The last sweep's labels of k columns, read back.
    pub fn labels(&self, d: &DevGraph, k: usize) -> Result<Vec<Labels>> {
        let t0 = std::time::Instant::now();
        let cl = &self.client;
        let p = k.div_ceil(2) as u32;
        // The used words, read straight from the state as a one-dimensional tensor.
        let words = d.n * p as usize;
        let desc = d.state.clone().copy_descriptor(cubecl::zspace::Shape::new([words]), cubecl::zspace::Strides::new(&[1]), 4);
        let bytes = cl.read_tensor(vec![desc]).map_err(|e| anyhow!("read: {e:?}"))?.remove(0);
        ensure!(bytes.len() >= words * 4, "read {} bytes of {}", bytes.len(), words * 4);
        // As words, without a copy when the buffer allows it.
        let mut all: Vec<u32> = bytes.try_into_vec::<u32>().unwrap_or_else(|b| bytemuck::pod_collect_to_vec(&b[..words * 4]));
        all.truncate(words);
        if self.timing {
            let [a, _] = self.times.get();
            self.times.set([a, t0.elapsed().as_secs_f64()]);
        }
        let _ = DEAD;
        let all = std::sync::Arc::new(all);
        Ok((0..k).map(|c| Labels::Packed { words: all.clone(), p: p as usize, c }).collect())
    }
}

/// A mode's ID table on the device (FORMAT §7: layout ids, offsets, road), in chunks of
/// `per` IDs, 5 words each.
pub struct DevIds {
    chunks: Vec<(Handle, usize)>,
    per: usize,
    n: usize,
    pub bytes: u64,
}

/// A component's heads data on the device: the new id of each layout id; CAR: exit turns in
/// new-id order (chunks of nodes), copies (nodes with copies, CSR, their copies), and room
/// for h of the turn pass and the gathered copies.
pub struct DevHeads {
    first: u32,
    n: usize,
    pos: Handle,
    /// (first new id, nodes, CSR, heads, costs, turns).
    turns: Vec<(u32, u32, Handle, Handle, Handle, usize)>,
    copies: Option<(Handle, Handle, Handle, u32, u32)>,
    h: Option<Handle>,
    tmp: Option<Handle>,
    pub bytes: u64,
}

/// One column's heads on the device: its sources' rows (`heads_ids`'s `src`, column 0) and
/// what it wants.
#[derive(Clone, Debug)]
pub struct HeadCol {
    pub src: Vec<[u32; 8]>,
    pub want: crate::heads::Want,
}

/// The sources of a column as device rows.
pub fn source_rows(mv: &crate::vocab::Mode, srcs: &[(u32, u16)]) -> Vec<[u32; 8]> {
    srcs.iter()
        .map(|&(i, start)| {
            let r = mv.device_row(i as usize);
            let o = mv.off(i as usize);
            [r[0], r[1], r[4], 0, start as u32, o[0] as u32 | (o[2] as u32) << 16, i, 0]
        })
        .collect()
}

/// Sparse answers kept per batch at most; past it the batch's heads run on the host.
const SPARSE_CAP: u32 = 1 << 22;

impl Gpu {
    fn ids_per(&self) -> usize {
        (self.max_chunk / 32) as usize
    }

    pub fn ids_bytes(n: usize) -> u64 {
        n as u64 * 20
    }

    pub fn load_ids(&self, mv: &crate::vocab::Mode) -> Result<DevIds> {
        let (n, per) = (mv.len(), self.ids_per());
        let mut used = 0u64;
        let mut chunks = Vec::new();
        let mut rows: Vec<u32> = Vec::new();
        for a in (0..n.max(1)).step_by(per) {
            let b = (a + per).min(n);
            rows.clear();
            rows.reserve((b - a) * 5);
            for i in a..b {
                rows.extend_from_slice(&mv.device_row(i));
            }
            if rows.is_empty() {
                rows.push(NONE);
            }
            chunks.push((self.upload(bytes_u32(&rows), &mut used), rows.len()));
        }
        Ok(DevIds { chunks, per, n, bytes: used })
    }

    pub fn heads_bytes(g: &Graph, p_max: u32) -> u64 {
        let n = g.n as u64;
        let mut b = n * 4;
        if g.fam == crate::vocab::Fam::Car {
            b += (n + 1) * 4 + g.arc_head.len() as u64 * 6 + 64 * 1024 + g.copies.len() as u64 * (12 + 4 * p_max as u64) + n * p_max as u64 * 4;
        }
        b
    }

    pub fn load_heads(&self, g: &Graph, first: u32, p_max: u32) -> Result<DevHeads> {
        let mut used = 0u64;
        let n = g.n;
        let pos = self.upload(bytes_u32(if g.new.is_empty() { &[0] } else { &g.new }), &mut used);
        let (mut turns, mut copies, mut h, mut tmp) = (Vec::new(), None, None, None);
        if g.fam == crate::vocab::Fam::Car {
            let mut old = vec![0u32; n];
            for (v, &u) in g.new.iter().enumerate() {
                old[u as usize] = v as u32;
            }
            // Exit turns by new id, in chunks of whole nodes within a binding.
            let max_k = (self.max_chunk / 4) as usize;
            let (mut u0, mut tf, mut th, mut tc) = (0u32, vec![0u32], Vec::new(), Vec::new());
            let mut flush = |u: u32, u0: &mut u32, tf: &mut Vec<u32>, th: &mut Vec<u32>, tc: &mut Vec<u16>, used: &mut u64| {
                let k = th.len();
                if th.is_empty() {
                    th.push(0);
                }
                turns.push((*u0, u - *u0, self.upload(bytes_u32(tf), used), self.upload(bytes_u32(th), used), self.upload(&bytes_u16(tc), used), k));
                *u0 = u;
                tf.clear();
                tf.push(0);
                th.clear();
                tc.clear();
            };
            for u in 0..n as u32 {
                let r = g.arcs(old[u as usize]);
                ensure!(r.len() < max_k, "a node of {} exit turns", r.len());
                if th.len() + r.len() > max_k {
                    flush(u, &mut u0, &mut tf, &mut th, &mut tc, &mut used);
                }
                for a in r {
                    th.push(g.new[g.arc_head[a] as usize]);
                    tc.push(g.turn_cost[a]);
                }
                tf.push(th.len() as u32);
            }
            flush(n as u32, &mut u0, &mut tf, &mut th, &mut tc, &mut used);
            // Copies: per node with copies (new ids, ascending), its copies.
            let mut pairs: Vec<(u32, u32)> = Vec::new();
            for v in 0..n as u32 {
                for y in g.arrivals(v).skip(1) {
                    pairs.push((g.new[v as usize], g.new[y as usize]));
                }
            }
            pairs.sort_unstable();
            if !pairs.is_empty() {
                let (mut cs_s, mut cs_first, mut cs_c) = (Vec::new(), Vec::new(), Vec::new());
                for (k, &(s, c)) in pairs.iter().enumerate() {
                    if k == 0 || pairs[k - 1].0 != s {
                        cs_s.push(s);
                        cs_first.push(k as u32);
                    }
                    cs_c.push(c);
                }
                cs_first.push(pairs.len() as u32);
                let (ns, np) = (cs_s.len() as u32, pairs.len() as u32);
                copies = Some((self.upload(bytes_u32(&cs_s), &mut used), self.upload(bytes_u32(&cs_first), &mut used), self.upload(bytes_u32(&cs_c), &mut used), ns, np));
                let tb = pairs.len() * p_max as usize * 4;
                used += tb as u64;
                tmp = Some(self.client.empty(tb));
            }
            let hb = n.max(1) * p_max as usize * 4;
            used += hb as u64;
            h = Some(self.client.empty(hb));
        }
        Ok(DevHeads { first, n, pos, turns, copies, h, tmp, bytes: used })
    }

    /// Heads of the last sweep's columns on the device (`car`, `dir` and `prune` as its).
    /// `None` when a sparse batch kept more answers than it has room for.
    pub fn heads(&self, d: &DevGraph, dh: &DevHeads, ids: &DevIds, car: bool, dir: Dir, prune: bool, cols: &[&HeadCol]) -> Result<Option<Vec<crate::heads::Times>>> {
        use crate::heads::{PackedOut, Times};
        let cl = self.client.clone();
        let dim = CubeDim::new_1d(256);
        let k = cols.len() as u32;
        let p = k.div_ceil(2);
        let koff = (dir == Dir::To) as u32;
        // h: the labels, with car copies folded in going from; the turn pass going to.
        let mut h = &d.state;
        if car && dir == Dir::From {
            if let (Some((cs_s, cs_first, cs_c, ns, np)), Some(tmp)) = (&dh.copies, &dh.tmp) {
                copy_gather::launch(&cl, grid(*np, 256, self.max_x), dim, arg(cs_c, *np as usize), *np, arg(&d.state, d.n * p as usize), arg(tmp, *np as usize * p as usize), p);
                copy_fold::launch(&cl, grid(*ns, 256, self.max_x), dim, arg(cs_s, *ns as usize), arg(cs_first, *ns as usize + 1), *ns, arg(tmp, *np as usize * p as usize), arg(&d.state, d.n * p as usize), p);
            }
        } else if car {
            let hh = dh.h.as_ref().unwrap();
            for (u0, cnt, tf, th, tc, nk) in &dh.turns {
                if *cnt > 0 {
                    turn_pass::launch(&cl, grid(*cnt, 256, self.max_x), dim, *u0, *cnt, arg(tf, *cnt as usize + 1), arg(th, *nk), arg(tc, nk.div_ceil(2)), arg(&d.state, d.n * p as usize), arg(hh, d.n * p as usize), p);
                }
            }
            h = hh;
        }
        // Sources of every column, sorted by piece.
        let mut src: Vec<[u32; 8]> = Vec::new();
        for (c, col) in cols.iter().enumerate() {
            src.extend(col.src.iter().map(|r| {
                let mut r = *r;
                r[3] = c as u32;
                r
            }));
        }
        src.sort_unstable();
        let nsrc = src.len() as u32;
        let src_h = cl.create_from_slice(bytes_u32(if src.is_empty() { &[0u32; 8] } else { bytemuck::cast_slice(&src) }));
        let lo = cols.iter().map(|c| c.want.lo).min().unwrap();
        let hi = cols.iter().map(|c| c.want.hi).max().unwrap().min(ids.n as u32);
        let sparse = cols.iter().all(|c| !c.want.dense());
        let mut mx = vec![0u32; p as usize];
        for (c, col) in cols.iter().enumerate() {
            mx[c / 2] |= (col.want.max as u32) << (16 * (c % 2));
        }
        let mx_h = cl.create_from_slice(bytes_u32(&mx));
        let counter = cl.create_from_slice(bytes_u32(&[0u32]));
        let cap = SPARSE_CAP;
        let sp_out = if sparse { Some(cl.empty(cap as usize * 8)) } else { None };
        let mut dense_out: Vec<(u32, u32, Handle)> = Vec::new();
        let per = ids.per as u32;
        let mut c0 = lo / per;
        while c0 * per < hi {
            let (a, b) = ((c0 * per).max(lo), ((c0 + 1) * per).min(hi));
            let count = b - a;
            let out = match &sp_out {
                Some(o) => o.clone(),
                None => cl.empty((count as usize * p as usize * 4).max(4)),
            };
            let out_len = if sparse { cap as usize * 2 } else { count as usize * p as usize };
            heads_ids::launch(
                &cl,
                grid(count, 256, self.max_x),
                dim,
                arg(&ids.chunks[c0 as usize].0, ids.chunks[c0 as usize].1),
                a - c0 * per,
                a,
                count,
                dh.first,
                dh.n as u32,
                arg(&dh.pos, dh.n),
                arg(h, d.n * p as usize),
                p,
                k,
                koff,
                car as u32,
                arg(&d.bound, d.p_max as usize),
                prune as u32,
                arg(&src_h, src.len() * 8),
                nsrc,
                arg(&mx_h, p as usize),
                sparse as u32,
                arg(&out, out_len),
                arg(&counter, 1),
                cap,
            );
            if !sparse {
                dense_out.push((a, count, out));
            }
            c0 += 1;
        }
        let read = |h: &Handle, words: usize| -> Result<Vec<u32>> {
            let desc = h.clone().copy_descriptor(cubecl::zspace::Shape::new([words]), cubecl::zspace::Strides::new(&[1]), 4);
            let bytes = cl.read_tensor(vec![desc]).map_err(|e| anyhow!("read: {e:?}"))?.remove(0);
            ensure!(bytes.len() >= words * 4, "read {} bytes of {}", bytes.len(), words * 4);
            let mut v: Vec<u32> = bytes.try_into_vec::<u32>().unwrap_or_else(|b| bytemuck::pod_collect_to_vec(&b[..words * 4]));
            v.truncate(words);
            Ok(v)
        };
        if sparse {
            let got = read(&counter, 1)?[0];
            if got > cap {
                return Ok(None);
            }
            let pairs = if got > 0 { read(sp_out.as_ref().unwrap(), got as usize * 2)? } else { Vec::new() };
            let mut per_col: Vec<Vec<(u32, u16)>> = vec![Vec::new(); cols.len()];
            for e in pairs.as_chunks::<2>().0 {
                let c = (e[1] >> 16) as usize;
                if (cols[c].want.lo..cols[c].want.hi).contains(&e[0]) {
                    per_col[c].push((e[0], e[1] as u16));
                }
            }
            return Ok(Some(
                per_col
                    .into_iter()
                    .map(|mut v| {
                        v.sort_unstable();
                        Times::Sparse(v)
                    })
                    .collect(),
            ));
        }
        let mut chunks = Vec::new();
        for (a, count, hd) in &dense_out {
            chunks.push((*a, read(hd, *count as usize * p as usize)?));
        }
        let out = std::sync::Arc::new(PackedOut { per, p: p as usize, chunks });
        Ok(Some(
            cols.iter()
                .enumerate()
                .map(|(c, col)| {
                    let t = Times::Packed { lo, hi, out: out.clone(), c, patch: None };
                    if col.want.dense() {
                        // Each column covers its own range only.
                        if col.want.lo == lo && col.want.hi.min(ids.n as u32) == hi { t } else { Times::Dense { lo: col.want.lo, t: (col.want.lo..col.want.hi).map(|i| t.get(i)).collect() } }
                    } else {
                        Times::Sparse(t.sparse(col.want.lo, col.want.hi, col.want.max))
                    }
                })
                .collect(),
        ))
    }
}
