// Downward sweep + heads on the GPU layout (see src/layout.rs, src/field.rs).
//
// State: d[node][col] as u16 seconds, 65535 = infinity, row stride = 2*P
// columns where P = ceil(cols/2) u32 words (two columns per word, SIMD
// __vaddus2 / __vminu2 = saturating add / min per 16-bit half).
//
// No full-state memset per batch: every node's row is written exactly once by
// the sweep. Rows of seeded nodes (the sources' ancestors, from the CPU
// upward sweep) are initialized explicitly and flagged in a bitmask; every
// other node starts from infinity inside the kernels.
//
// Levels are numbered top-first: level l = nodes [level_first[l], level_first[l+1]).
// Levels [0, top) are the narrow top, run by independent blocks that each own
// a slice of columns and sync with __syncthreads. Levels [top, L) are wide.

#include <cooperative_groups.h>
#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <vector>

namespace cg = cooperative_groups;

#define INF16 0xFFFFu

struct Entry {
    uint32_t node, col, value;
};

static __device__ __forceinline__ bool seeded(const uint32_t* mask, uint32_t v) {
    return (mask[v >> 5] >> (v & 31)) & 1u;
}

// ---- weights ----------------------------------------------------------------
// A sweep runs on one u16 metric (the weight is the same for every column: W = uint16_t),
// or on a packed pair of metrics (W = uint32_t, modes phase Step 4: half h of w[k] is the
// weight of column 2p + h, so a column pair is one source in two modes, e.g. the ACTIVE
// expert's [bike, walk]). The u16 path is the instructions of before.
static __device__ __forceinline__ uint32_t wpair(const uint16_t* __restrict__ w, uint32_t k) {
    const uint32_t x = w[k];
    return x | (x << 16);
}
static __device__ __forceinline__ uint32_t wpair(const uint32_t* __restrict__ w, uint32_t k) { return w[k]; }
// The weight of arc k for column c.
static __device__ __forceinline__ uint32_t wcol(const uint16_t* __restrict__ w, uint32_t k, uint32_t) { return w[k]; }
static __device__ __forceinline__ uint32_t wcol(const uint32_t* __restrict__ w, uint32_t k, uint32_t c) {
    return (w[k] >> (16 * (c & 1))) & 0xFFFFu;
}

// ---- seeding --------------------------------------------------------------

__global__ void seed_rows(const Entry* e, uint32_t count, uint32_t* d32, uint32_t P, uint32_t* mask) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    uint32_t v = e[i].node;
    atomicOr(&mask[v >> 5], 1u << (v & 31));
    for (uint32_t p = 0; p < P; ++p) d32[(size_t)v * P + p] = 0xFFFFFFFFu;
}

__global__ void seed_values(const Entry* e, uint32_t count, uint16_t* d16, uint32_t P) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    // Entries are unique per (node, col) by construction (CPU upward sweep).
    d16[(size_t)e[i].node * 2 * P + e[i].col] = (uint16_t)e[i].value;
}

__global__ void clear_mask(const Entry* e, uint32_t count, uint32_t* mask) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) mask[e[i].node >> 5] = 0;
}

// ---- narrow top: one block per column slice, __syncthreads between levels --
// Per level: (arc, column) work items, shared-memory atomicMin per (node, column).
//
// V selects profiling variants (production uses TOP_RUN):
//   TOP_STAMP     same work, block 0 records clock64() at every level boundary;
//   TOP_NO_ARCS   skips the arc phase (init + write-back + syncs only): the per-level floor;
//   TOP_NO_ATOMIC arc phase without atomics (per-thread min, one store at the end): contention.
// The two ablations leave a wrong state; they exist only for cchgpu_profile.

enum TopVariant { TOP_RUN = 0, TOP_STAMP = 1, TOP_NO_ARCS = 2, TOP_NO_ATOMIC = 3 };

template <int V, class W>
__global__ void top_kernel(const uint32_t* __restrict__ level_first, uint32_t top_levels,
                           const uint32_t* __restrict__ up_first, const uint32_t* __restrict__ up_head,
                           const W* __restrict__ w, const uint16_t* __restrict__ top_tail, uint16_t* d16,
                           uint32_t P, uint32_t cols, uint32_t cols_per_block, const uint32_t* __restrict__ mask,
                           long long* stamps) {
    extern __shared__ uint32_t acc[];
    const uint32_t c0 = blockIdx.x * cols_per_block;
    const uint32_t C = min(cols_per_block, cols - c0);
    const size_t stride = 2 * (size_t)P;
    const bool stamp = V != TOP_RUN && blockIdx.x == 0 && threadIdx.x == 0;
    uint32_t sink = INF16;
    if (stamp) stamps[0] = clock64();
    for (uint32_t l = 0; l < top_levels; ++l) {
        const uint32_t base = level_first[l], cnt = level_first[l + 1] - base;
        for (uint32_t i = threadIdx.x; i < cnt * C; i += blockDim.x) {
            uint32_t v = base + i / C, c = c0 + i % C;
            acc[i] = seeded(mask, v) ? d16[v * stride + c] : INF16;
        }
        __syncthreads();
        if (V != TOP_NO_ARCS) {
            const uint32_t a0 = up_first[base], na = up_first[base + cnt] - a0;
            for (uint32_t i = threadIdx.x; i < na * C; i += blockDim.x) {
                uint32_t k = a0 + i / C, c = i % C;
                uint32_t x = (uint32_t)d16[(size_t)up_head[k] * stride + c0 + c] + wcol(w, k, c0 + c);
                if (V == TOP_NO_ATOMIC)
                    sink = min(sink, x + top_tail[k]);
                else
                    atomicMin(&acc[top_tail[k] * C + c], min(x, (uint32_t)INF16));
            }
            __syncthreads();
        }
        for (uint32_t i = threadIdx.x; i < cnt * C; i += blockDim.x) {
            uint32_t v = base + i / C, c = c0 + i % C;
            d16[v * stride + c] = (uint16_t)acc[i];
        }
        __syncthreads();
        if (stamp) stamps[l + 1] = clock64();
    }
    if (V == TOP_NO_ATOMIC && sink == 0x12345678u) d16[0] = 0;  // keep the loads alive
}

// ---- narrow top, shared-memory resident ------------------------------------
// One column per block. The whole top's distances (top_nodes u16) live in
// shared memory for the kernel's lifetime, so the per-arc gather never touches
// global memory. Arc records are packed: top_arc[k] = head | (tail local id << 16),
// both < 65536 (checked on the host); w[k] is the layout weight (top arcs are the
// first arcs of the CSR). Each thread issues up to RES_ILP arc records before
// consuming them. Per level: arc phase, sync, write-back of this level and
// accumulator init of the next level (double-buffered), sync.
//
// AGG (degree-aware): a warp's 32 lanes hold 32 consecutive arcs, sorted by tail,
// so arcs of one node form contiguous lane segments. A segmented shuffle min
// collapses each segment and only its first lane issues the shared atomicMin:
// a high-degree node gets whole warps, a 1-arc node a single lane. Without AGG,
// every arc of a high-degree node hits the same shared address.

#define RES_ILP 2
#define RES_THREADS 1024

// Segmented suffix min over lanes with equal, contiguous keys; true on each segment's first lane.
static __device__ __forceinline__ bool segment_min(uint32_t key, uint32_t& x) {
    const uint32_t lane = threadIdx.x & 31;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        uint32_t y = __shfl_down_sync(0xFFFFFFFFu, x, o);
        uint32_t ky = __shfl_down_sync(0xFFFFFFFFu, key, o);
        if (lane + o < 32 && ky == key) x = min(x, y);
    }
    uint32_t kp = __shfl_up_sync(0xFFFFFFFFu, key, 1);
    return lane == 0 || kp != key;
}

// How the arc phase merges a warp's 32 arcs into the shared accumulators.
enum TopMerge { MERGE_ATOMIC = 0, MERGE_SEGMENT = 1, MERGE_WARP = 2 };

template <bool STAMP, int MERGE = MERGE_WARP, class W = uint16_t>
__global__ void __launch_bounds__(1024) top_resident(const uint32_t* __restrict__ level_first, uint32_t top_levels,
                                                     const uint32_t* __restrict__ up_first,
                                                     const uint32_t* __restrict__ top_arc,
                                                     const W* __restrict__ w, uint16_t* d16, uint32_t P,
                                                     uint32_t max_level, const uint32_t* __restrict__ mask,
                                                     long long* stamps) {
    extern __shared__ uint32_t sm[];
    uint32_t* lf = sm;                     // level_first[0..top]
    uint32_t* af = lf + top_levels + 1;    // first arc of each level
    uint32_t* acc0 = af + top_levels + 1;  // two accumulators of max_level entries
    uint32_t* acc1 = acc0 + max_level;
    uint16_t* dist = (uint16_t*)(acc1 + max_level);
    const uint32_t c = blockIdx.x, T = blockDim.x, tid = threadIdx.x;
    const size_t stride = 2 * (size_t)P;
    const bool stamp = STAMP && c == 0 && tid == 0;
    if (stamp) stamps[0] = clock64();
    for (uint32_t i = tid; i <= top_levels; i += T) {
        lf[i] = level_first[i];
        af[i] = up_first[level_first[i]];
    }
    __syncthreads();
    const uint32_t top_nodes = lf[top_levels];
    for (uint32_t v = tid; v < top_nodes; v += T) dist[v] = seeded(mask, v) ? d16[v * stride + c] : INF16;
    __syncthreads();
    for (uint32_t i = tid; i < lf[1]; i += T) acc0[i] = dist[i];
    __syncthreads();
    const uint32_t lane = tid & 31;
    for (uint32_t l = 0; l < top_levels; ++l) {
        uint32_t* acc = (l & 1) ? acc1 : acc0;
        uint32_t* next = (l & 1) ? acc0 : acc1;
        const uint32_t a1 = af[l + 1];
        // Warp-uniform trip count (i - lane is the same for the whole warp): shuffles need all lanes.
        for (uint32_t i = af[l] + tid; i - lane < a1; i += RES_ILP * T) {
            uint32_t r[RES_ILP], x[RES_ILP];
#pragma unroll
            for (int j = 0; j < RES_ILP; ++j) {
                // Clamped index: loads write registers directly; dead lanes are masked below.
                const uint32_t k = i + j * T < a1 ? i + j * T : 0;
                r[j] = top_arc[k];
                x[j] = wcol(w, k, c);
            }
#pragma unroll
            for (int j = 0; j < RES_ILP; ++j) {
                const bool live = i + j * T < a1;
                uint32_t y = live ? min((uint32_t)dist[r[j] & 0xFFFFu] + x[j], (uint32_t)INF16) : INF16;
                const uint32_t t = live ? r[j] >> 16 : 0xFFFFFFFFu;
                if (MERGE == MERGE_SEGMENT) {
                    bool lead = segment_min(t, y);
                    if (live && lead) atomicMin(&acc[t], y);
                } else if (MERGE == MERGE_WARP) {
                    // Whole warp on one node (a high-degree node): one reduction, one atomic.
                    if (__all_sync(0xFFFFFFFFu, t == __shfl_sync(0xFFFFFFFFu, t, 0))) {
                        y = __reduce_min_sync(0xFFFFFFFFu, y);
                        if (lane == 0 && live) atomicMin(&acc[t], y);
                    } else if (live) {
                        atomicMin(&acc[t], y);
                    }
                } else if (live) {
                    atomicMin(&acc[t], y);
                }
            }
        }
        __syncthreads();
        const uint32_t base = lf[l], cnt = lf[l + 1] - base;
        for (uint32_t i = tid; i < cnt; i += T) {
            uint16_t val = (uint16_t)acc[i];
            dist[base + i] = val;
            d16[(base + i) * stride + c] = val;
        }
        if (l + 1 < top_levels)
            for (uint32_t i = tid; i < lf[l + 2] - lf[l + 1]; i += T) next[i] = dist[lf[l + 1] + i];
        __syncthreads();
        if (stamp) stamps[l + 1] = clock64();
    }
}

// ---- wide levels: LP lanes per node, each lane owns column pairs ----------

template <class W>
static __device__ __forceinline__ void relax_node(uint32_t v, uint32_t lane, uint32_t LP,
                                                  const uint32_t* __restrict__ up_first,
                                                  const uint32_t* __restrict__ up_head,
                                                  const W* __restrict__ w, uint32_t* d32, uint32_t P,
                                                  const uint32_t* __restrict__ mask) {
    const bool s = seeded(mask, v);
    const uint32_t k0 = up_first[v], k1 = up_first[v + 1];
    for (uint32_t p = lane; p < P; p += LP) {
        uint32_t acc = s ? d32[(size_t)v * P + p] : 0xFFFFFFFFu;
        for (uint32_t k = k0; k < k1; ++k) {
            uint32_t x = d32[(size_t)up_head[k] * P + p];
            acc = __vminu2(acc, __vaddus2(x, wpair(w, k)));
        }
        d32[(size_t)v * P + p] = acc;
    }
}

// ---- degree-aware split of high-degree wide nodes ---------------------------
// Nodes with more than split_t upper arcs are listed per level (heavy[], by
// decreasing degree) in classes of 32, 16, 8, 4, 2 arc groups (enough groups for
// ~split_t arcs each). A heavy node gets G = min(class, 32 / LP) groups of LP
// lanes, each group takes every G-th arc, and a shuffle min merges the groups;
// light nodes keep one group of LP lanes and skip the heavy ones. When LP = 32
// (B >= 63) there is no room in a warp and every node is light. Only levels that
// cannot fill the GPU by themselves (latency-bound: the serial neighbour loop is
// the critical path) are split.

#define SPLIT_CLASSES 5
// Measured on France (RESULTS.md, Part 2.2): best threshold per LP = 1, 2, 4, 8, 16; 0 = off.
static const uint32_t WIDE_SPLIT_DEFAULT[5] = {4, 8, 16, 16, 0};
// Only levels with at most this many (node, lane) items are split (same measurement).
#define WIDE_SPLIT_MAX_ITEMS 65536

struct LevelSplit {
    uint32_t first;                   // into heavy[]
    uint32_t count[SPLIT_CLASSES];    // heavy nodes with 32 >> c groups
};

template <class W>
static __device__ __forceinline__ void relax_heavy(uint32_t v, uint32_t sub, uint32_t G, uint32_t LP,
                                                   const uint32_t* __restrict__ up_first,
                                                   const uint32_t* __restrict__ up_head,
                                                   const W* __restrict__ w, uint32_t* d32, uint32_t P,
                                                   const uint32_t* __restrict__ mask) {
    const uint32_t lane = sub % LP, part = sub / LP, width = G * LP, wl = threadIdx.x & 31;
    const uint32_t gmask = width == 32 ? 0xFFFFFFFFu : ((1u << width) - 1) << (wl & ~(width - 1));
    const bool s = seeded(mask, v);
    const uint32_t k0 = up_first[v], k1 = up_first[v + 1];
    for (uint32_t p0 = 0; p0 < P; p0 += LP) {  // same trip count for the whole group
        const uint32_t p = p0 + lane;
        uint32_t acc = 0xFFFFFFFFu;
        if (p < P)
            for (uint32_t k = k0 + part; k < k1; k += G) {
                uint32_t x = d32[(size_t)up_head[k] * P + p];
                acc = __vminu2(acc, __vaddus2(x, wpair(w, k)));
            }
        for (uint32_t o = LP; o < width; o <<= 1) acc = __vminu2(acc, __shfl_xor_sync(gmask, acc, o));
        if (part == 0 && p < P) {
            if (s) acc = __vminu2(acc, d32[(size_t)v * P + p]);
            d32[(size_t)v * P + p] = acc;
        }
    }
}

// Work item t of a wide level: heavy classes first, then every node of the level with LP lanes.
// skip_t = degree above which a node was handled as heavy (UINT32_MAX: no split at this LP).
template <class W>
static __device__ __forceinline__ void wide_item(uint64_t t, uint32_t base, uint32_t cnt, uint32_t LP,
                                                 const LevelSplit& sp, const uint32_t* __restrict__ heavy,
                                                 uint32_t skip_t, const uint32_t* __restrict__ up_first,
                                                 const uint32_t* __restrict__ up_head,
                                                 const W* __restrict__ w, uint32_t* d32, uint32_t P,
                                                 const uint32_t* __restrict__ mask) {
    if (skip_t != 0xFFFFFFFFu) {
        uint32_t first = sp.first;
        for (int c = 0; c < SPLIT_CLASSES; ++c) {
            const uint32_t G = min(32u >> c, 32u / LP), span = sp.count[c] * G * LP;
            if (t < span) {
                relax_heavy(heavy[first + (uint32_t)t / (G * LP)], (uint32_t)t % (G * LP), G, LP, up_first, up_head,
                            w, d32, P, mask);
                return;
            }
            t -= span;
            first += sp.count[c];
        }
    }
    if (t >= (uint64_t)cnt * LP) return;
    const uint32_t v = base + (uint32_t)(t / LP);
    if (up_first[v + 1] - up_first[v] > skip_t) return;
    relax_node(v, (uint32_t)(t % LP), LP, up_first, up_head, w, d32, P, mask);
}

// Levels without heavy nodes take the plain path (no item mapping, no skip test).
static __host__ __device__ __forceinline__ uint32_t level_skip(const LevelSplit& sp, uint32_t skip_t) {
    uint32_t heavy = 0;
    for (int c = 0; c < SPLIT_CLASSES; ++c) heavy += sp.count[c];
    return heavy ? skip_t : 0xFFFFFFFFu;
}

static __host__ __device__ __forceinline__ uint64_t wide_items(const LevelSplit& sp, uint32_t cnt, uint32_t LP,
                                                               uint32_t skip_t) {
    uint64_t total = (uint64_t)cnt * LP;
    if (skip_t != 0xFFFFFFFFu)
        for (int c = 0; c < SPLIT_CLASSES; ++c) total += (uint64_t)sp.count[c] * min(32u >> c, 32u / LP) * LP;
    return total;
}

template <class W>
__global__ void wide_level(uint32_t base, uint32_t cnt, uint32_t LP, const uint32_t* __restrict__ up_first,
                           const uint32_t* __restrict__ up_head, const W* __restrict__ w, uint32_t* d32,
                           uint32_t P, const uint32_t* __restrict__ mask) {
    uint64_t t = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (uint64_t)cnt * LP) return;
    relax_node(base + (uint32_t)(t / LP), (uint32_t)(t % LP), LP, up_first, up_head, w, d32, P, mask);
}

// A level with heavy nodes (skip_t = level_skip(sp, ...) != UINT32_MAX).
template <class W>
__global__ void wide_level_split(uint32_t base, uint32_t cnt, uint32_t LP, LevelSplit sp,
                                 const uint32_t* __restrict__ heavy, uint32_t skip_t,
                                 const uint32_t* __restrict__ up_first, const uint32_t* __restrict__ up_head,
                                 const W* __restrict__ w, uint32_t* d32, uint32_t P,
                                 const uint32_t* __restrict__ mask) {
    uint64_t t = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= wide_items(sp, cnt, LP, skip_t)) return;
    wide_item(t, base, cnt, LP, sp, heavy, skip_t, up_first, up_head, w, d32, P, mask);
}

// One wide level as its own launch (per-level and graph schedules, profiling).
template <class W>
static void launch_level(uint32_t base, uint32_t cnt, uint32_t LP, const LevelSplit& sp, const uint32_t* heavy,
                         uint32_t skip_t, const uint32_t* up_first, const uint32_t* up_head, const W* w,
                         uint32_t* d32, uint32_t P, const uint32_t* mask, cudaStream_t stream) {
    const uint32_t st = level_skip(sp, skip_t);
    const uint32_t blocks = (uint32_t)((wide_items(sp, cnt, LP, st) + 255) / 256);
    if (st == 0xFFFFFFFFu)
        wide_level<<<blocks, 256, 0, stream>>>(base, cnt, LP, up_first, up_head, w, d32, P, mask);
    else
        wide_level_split<<<blocks, 256, 0, stream>>>(base, cnt, LP, sp, heavy, st, up_first, up_head, w, d32, P,
                                                     mask);
}

// splits[l - l0] for each wide level.
template <class W>
__global__ void wide_coop(const uint32_t* __restrict__ level_first, uint32_t l0, uint32_t l1, uint32_t LP,
                          const LevelSplit* __restrict__ splits, const uint32_t* __restrict__ heavy, uint32_t skip_t,
                          const uint32_t* __restrict__ up_first, const uint32_t* __restrict__ up_head,
                          const W* __restrict__ w, uint32_t* d32, uint32_t P,
                          const uint32_t* __restrict__ mask) {
    cg::grid_group grid = cg::this_grid();
    const uint64_t nthreads = (uint64_t)gridDim.x * blockDim.x;
    for (uint32_t l = l0; l < l1; ++l) {
        const uint32_t base = level_first[l], cnt = level_first[l + 1] - base;
        const LevelSplit sp = splits[l - l0];
        const uint32_t st = level_skip(sp, skip_t);
        const uint64_t items = wide_items(sp, cnt, LP, st);
        for (uint64_t t = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; t < items; t += nthreads)
            if (st == 0xFFFFFFFFu)
                relax_node(base + (uint32_t)(t / LP), (uint32_t)(t % LP), LP, up_first, up_head, w, d32, P, mask);
            else
                wide_item(t, base, cnt, LP, sp, heavy, st, up_first, up_head, w, d32, P, mask);
        grid.sync();
    }
}

// ---- heads ----------------------------------------------------------------

// bits[c][word]: one warp per 32 nodes, ballot per column.
__global__ void head_threshold(const uint16_t* __restrict__ d16, uint32_t n, uint32_t P, uint32_t cols, uint16_t t,
                               uint32_t* bits) {
    uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t words = (n + 31) / 32;
    for (uint32_t c = 0; c < cols; ++c) {
        bool on = v < n && d16[(size_t)v * 2 * P + c] <= t;
        uint32_t b = __ballot_sync(0xFFFFFFFFu, on);
        if ((threadIdx.x & 31) == 0 && v / 32 < words) bits[(size_t)c * words + v / 32] = b;
    }
}

__global__ void head_reduce(const uint16_t* __restrict__ d16, uint32_t n, uint32_t P, uint32_t cols, int op,
                            uint16_t* r) {
    uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    const uint16_t* row = d16 + (size_t)v * 2 * P;
    uint32_t acc = row[0];
    for (uint32_t c = 1; c < cols; ++c) acc = op == 0 ? min(acc, (uint32_t)row[c]) : max(acc, (uint32_t)row[c]);
    r[v] = (uint16_t)acc;
}

__global__ void head_bits(const uint16_t* __restrict__ r, uint32_t n, uint16_t t, uint32_t* bits) {
    uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    bool on = v < n && r[v] <= t;
    uint32_t b = __ballot_sync(0xFFFFFFFFu, on);
    if ((threadIdx.x & 31) == 0 && v / 32 < (n + 31) / 32) bits[v / 32] = b;
}

// ---- compact heads ------------------------------------------------------------
// Each block owns a tile of TILE consecutive items (nodes, or addresses for the
// address head). The tile's rows (P u32 words each) are loaded coalesced into
// shared memory once, with the row stride padded to P+1 words against bank
// conflicts; all per-column work then reads shared memory. Shared bytes =
// TILE*(P+1)*4 + cols*WARPS*4.

#define TILE 256
#define WARPS (TILE / 32)

// Rows of the state itself: item = layout node.
struct NodeRows {
    const uint32_t* __restrict__ d32;
    uint32_t n, P;
    __device__ uint32_t count() const { return n; }
    __device__ uint32_t word(uint32_t v, uint32_t p) const { return d32[(size_t)v * P + p]; }
};

// Same-piece data of a column (field::ColPiece): the column address's segments and packed
// "to" offsets; u = v = to = ~0 for a column seeded at vertices.
struct ColPiece {
    uint32_t u, v, to;
};

// What the address head reads of one address: piece ends, packed offsets of this sweep's
// direction, packed "to" offsets (same-piece rule); the same of the second table (pairs).
struct AddrItem {
    uint32_t u, v, off, to, u2, v2, off2, to2;
};

// Address head: item = address a on piece (u, v); per 16-bit half (column)
// t = min(d[u] + o_u, d[v] + o_v), saturating. off[a] = o_u | o_v << 16
// (the "to" offsets after a forward sweep, the "from" offsets after a reversed one).
// With column pieces (cchgpu_set_col_pieces), also the direct drive along the piece shared
// with the column's address (same ends u, v; field::ColPiece): forward q -> a, reversed
// a -> q, to(dest) - to(origin) when the destination is ahead.
struct AddrRows {
    const uint32_t* __restrict__ d32;
    const uint32_t* __restrict__ u;
    const uint32_t* __restrict__ v;
    const uint32_t* __restrict__ off;
    uint32_t n, P;
    const uint32_t* __restrict__ to;  // "to" offsets, either direction
    const ColPiece* __restrict__ cp;  // 2 * P entries, or null
    int rev;
    // Pair sweeps (cchgpu_set_addresses2): the table of the upper half of every word (the
    // second mode snaps the same addresses onto its own pieces); null otherwise.
    const uint32_t* __restrict__ u2 = nullptr;
    const uint32_t* __restrict__ v2 = nullptr;
    const uint32_t* __restrict__ off2 = nullptr;
    const uint32_t* __restrict__ to2 = nullptr;
    __device__ uint32_t count() const { return n; }
    static __device__ __forceinline__ uint32_t at(const uint32_t* d32, uint32_t P, uint32_t u, uint32_t v, uint32_t o,
                                                  uint32_t p) {
        const uint32_t ou = (o & 0xFFFFu) * 0x10001u, ov = (o >> 16) * 0x10001u;
        return __vminu2(__vaddus2(d32[(size_t)u * P + p], ou), __vaddus2(d32[(size_t)v * P + p], ov));
    }
    // Everything the head reads of address a, whatever the column (load_tile stages it in
    // shared memory once per tile).
    __device__ AddrItem item(uint32_t a) const {
        AddrItem it{u[a], v[a], off[a], to[a], 0, 0, 0, 0};
        if (u2) it.u2 = u2[a], it.v2 = v2[a], it.off2 = off2[a], it.to2 = to2[a];
        return it;
    }
    __device__ uint32_t word(uint32_t a, uint32_t p) const { return word_of(item(a), p); }
    __device__ uint32_t word_of(const AddrItem& it, uint32_t p) const {
        uint32_t w;
        if (!u2) {
            w = at(d32, P, it.u, it.v, it.off, p);
        } else if (it.u2 == it.u && it.v2 == it.v) {
            // Same piece for both modes (96 % of the HdF addresses): one gather per end,
            // each half with its own mode's offsets.
            const uint32_t ou = (it.off & 0xFFFFu) | (it.off2 << 16), ov = (it.off >> 16) | (it.off2 & 0xFFFF0000u);
            w = __vminu2(__vaddus2(d32[(size_t)it.u * P + p], ou), __vaddus2(d32[(size_t)it.v * P + p], ov));
        } else {
            w = (at(d32, P, it.u, it.v, it.off, p) & 0xFFFFu) | (at(d32, P, it.u2, it.v2, it.off2, p) & 0xFFFF0000u);
        }
        return cp ? __vminu2(w, direct(it, p)) : w;
    }
    __device__ uint32_t direct(const AddrItem& it, uint32_t p) const {
        uint32_t out = 0;
        for (int h = 0; h < 2; ++h) {
            const bool second = h && u2;
            const uint32_t ta = second ? it.to2 : it.to, sa[2] = {second ? it.u2 : it.u, second ? it.v2 : it.v};
            const ColPiece q = cp[2 * p + h];
            uint32_t best = 0xFFFFu;
            if (q.u == sa[0] && q.v == sa[1])
                for (int x = 0; x < 2; ++x) {
                    const uint32_t oq = (q.to >> (16 * x)) & 0xFFFFu, oa = (ta >> (16 * x)) & 0xFFFFu;
                    const uint32_t from = rev ? oa : oq, dst = rev ? oq : oa;
                    if (oq != 0xFFFFu && oa != 0xFFFFu && dst >= from) best = min(best, dst - from);
                }
            out |= best << (16 * h);
        }
        return out;
    }
};

// Thresholds: one for every column, or one per column.
struct OneT {
    uint16_t t;
    __device__ uint32_t operator()(uint32_t) const { return t; }
};
struct ColT {
    const uint16_t* __restrict__ t;
    __device__ uint32_t operator()(uint32_t c) const { return t[c]; }
};

template <class Rows>
static __device__ __forceinline__ void load_tile(const Rows& rows, uint32_t P, uint32_t* tile) {
    const uint32_t base = blockIdx.x * TILE, n = rows.count();
    for (uint32_t i = threadIdx.x; i < TILE * P; i += TILE) {
        uint32_t vl = i / P, p = i % P, v = base + vl;
        tile[vl * (P + 1) + p] = v < n ? rows.word(v, p) : 0xFFFFFFFFu;
    }
    __syncthreads();
}

// Address rows (France ACTIVE phase, Step 2): the tile's addresses are read once, into
// shared memory (TILE AddrItem, 8 KiB static), then every word of the tile from there;
// before, every one of the P words of an address read its 4 (pairs: 8) table entries again.
static __device__ __forceinline__ void load_tile(const AddrRows& rows, uint32_t P, uint32_t* tile) {
    __shared__ AddrItem items[TILE];
    const uint32_t base = blockIdx.x * TILE, n = rows.count();
    if (base + threadIdx.x < n) items[threadIdx.x] = rows.item(base + threadIdx.x);
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < TILE * P; i += TILE) {
        uint32_t vl = i / P, p = i % P, v = base + vl;
        tile[vl * (P + 1) + p] = v < n ? rows.word_of(items[vl], p) : 0xFFFFFFFFu;
    }
    __syncthreads();
}

static __device__ __forceinline__ uint32_t tile_at(const uint32_t* tile, uint32_t P, uint32_t vl, uint32_t c) {
    return ((const uint16_t*)tile)[vl * 2 * (P + 1) + c];
}

// Output k of a head: one per column (d <= t(c)), or one per group of columns.
template <class Thr>
struct ColPred {
    Thr thr;
    __device__ bool operator()(const uint32_t* tile, uint32_t P, uint32_t vl, uint32_t c) const {
        return tile_at(tile, P, vl, c) <= thr(c);
    }
};
// Group k = columns [bounds[k], bounds[k + 1]): d <= t[c] for some (op 0, union) / every (op 1,
// intersection) column c of the group.
struct GroupPred {
    const uint16_t* __restrict__ t;
    const uint32_t* __restrict__ bounds;
    int op;
    __device__ bool operator()(const uint32_t* tile, uint32_t P, uint32_t vl, uint32_t k) const {
        bool any = false, all = true;
        for (uint32_t c = bounds[k]; c < bounds[k + 1]; ++c) {
            bool in = tile_at(tile, P, vl, c) <= t[c];
            any |= in;
            all &= in;
        }
        return op == 0 ? any : all;
    }
};

// counts[k] += items of this tile in output k (k < outs); chunk_counts[k][tile] = same (if not null);
// warp_bits[k][warp] = the warp's ballot word for output k (if not null: bits_compact then needs
// no second pass over the rows).
template <class Rows, class Pred>
__global__ void head_count(Rows rows, uint32_t P, uint32_t outs, Pred pred, uint32_t* counts, uint32_t* chunk_counts,
                           uint32_t* warp_bits = nullptr) {
    extern __shared__ uint32_t sm[];
    uint32_t* tile = sm;
    uint32_t* wc = sm + TILE * (P + 1);
    load_tile(rows, P, tile);
    const uint32_t vl = threadIdx.x, lane = vl & 31, wid = vl >> 5;
    const bool live = blockIdx.x * TILE + vl < rows.count();
    const uint32_t cols = outs;
    for (uint32_t c = 0; c < cols; ++c) {
        uint32_t b = __ballot_sync(0xFFFFFFFFu, live && pred(tile, P, vl, c));
        if (lane == 0) {
            wc[c * WARPS + wid] = __popc(b);
            if (warp_bits) warp_bits[(size_t)c * gridDim.x * WARPS + blockIdx.x * WARPS + wid] = b;
        }
    }
    __syncthreads();
    for (uint32_t c = threadIdx.x; c < cols; c += TILE) {
        uint32_t x = 0;
        for (uint32_t w = 0; w < WARPS; ++w) x += wc[c * WARPS + w];
        if (x) atomicAdd(&counts[c], x);
        if (chunk_counts) chunk_counts[(size_t)c * gridDim.x + blockIdx.x] = x;
    }
}

// Ordered stream compaction: out[chunk_offsets[k][tile] + rank in tile] = ids[v] (v itself if ids is null).
template <class Rows, class Pred>
__global__ void head_compact(Rows rows, uint32_t P, uint32_t outs, Pred pred, const uint32_t* __restrict__ chunk_offsets,
                             const uint32_t* __restrict__ ids, uint32_t* out) {
    extern __shared__ uint32_t sm[];
    uint32_t* tile = sm;
    uint32_t* wc = sm + TILE * (P + 1);
    load_tile(rows, P, tile);
    const uint32_t vl = threadIdx.x, lane = vl & 31, wid = vl >> 5, v = blockIdx.x * TILE + vl;
    const bool live = v < rows.count();
    const uint32_t cols = outs;
    for (uint32_t c = 0; c < cols; ++c) {
        uint32_t b = __ballot_sync(0xFFFFFFFFu, live && pred(tile, P, vl, c));
        if (lane == 0) wc[c * WARPS + wid] = __popc(b);
    }
    __syncthreads();
    for (uint32_t c = 0; c < cols; ++c) {
        bool on = live && pred(tile, P, vl, c);
        uint32_t b = __ballot_sync(0xFFFFFFFFu, on);
        if (on) {
            uint32_t pre = 0;
            for (uint32_t w = 0; w < wid; ++w) pre += wc[c * WARPS + w];
            out[chunk_offsets[(size_t)c * gridDim.x + blockIdx.x] + pre + __popc(b & ((1u << lane) - 1))] =
                ids ? ids[v] : v;
        }
    }
}

// Ordered compaction from head_count's warp_bits: out[chunk_offsets[k][tile] + rank in tile] = v.
__global__ void bits_compact(const uint32_t* __restrict__ warp_bits, uint32_t outs, const uint32_t* __restrict__ chunk_offsets,
                             uint32_t* out) {
    const uint32_t vl = threadIdx.x, lane = vl & 31, wid = vl >> 5, v = blockIdx.x * TILE + vl;
    for (uint32_t k = 0; k < outs; ++k) {
        const uint32_t* w = warp_bits + (size_t)k * gridDim.x * WARPS + blockIdx.x * WARPS;
        uint32_t b = w[wid];
        if (b >> lane & 1) {
            uint32_t pre = 0;
            for (uint32_t x = 0; x < wid; ++x) pre += __popc(w[x]);
            out[chunk_offsets[(size_t)k * gridDim.x + blockIdx.x] + pre + __popc(b & ((1u << lane) - 1))] = v;
        }
    }
}

// counts[0] += items with d <= t(c) for some column (op 0, union) / for every column (op 1,
// intersection). With one threshold this is (min | max over columns) <= t.
template <class Rows, class Thr>
__global__ void head_reduce_count(Rows rows, uint32_t P, uint32_t cols, int op, Thr thr, uint32_t* counts) {
    extern __shared__ uint32_t sm[];
    load_tile(rows, P, sm);
    const uint32_t vl = threadIdx.x;
    bool any = false, all = true;
    for (uint32_t c = 0; c < cols; ++c) {
        bool in = tile_at(sm, P, vl, c) <= thr(c);
        any |= in;
        all &= in;
    }
    uint32_t b = __ballot_sync(0xFFFFFFFFu, blockIdx.x * TILE + vl < rows.count() && (op == 0 ? any : all));
    if ((vl & 31) == 0 && b) atomicAdd(&counts[0], __popc(b));
}

// Address times for addresses [first, first + count): out[i][p] = word p of address first + i (tests).
__global__ void addr_times(AddrRows rows, uint32_t first, uint32_t count, uint32_t* out) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)count * rows.P) return;
    out[i] = rows.word(first + (uint32_t)(i / rows.P), (uint32_t)(i % rows.P));
}

// Meeting point: argmin over nodes of max (sum = 0) or sum (sum = 1) of the row; rows with an
// unreachable column are skipped. best = min of (cost << 32) | layout id.
__global__ void head_meet(const uint32_t* __restrict__ d32, uint32_t n, uint32_t P, uint32_t cols, int sum,
                          unsigned long long* best) {
    extern __shared__ uint32_t sm[];
    __shared__ unsigned long long warp_best[WARPS];
    load_tile(NodeRows{d32, n, P}, P, sm);
    const uint32_t vl = threadIdx.x, lane = vl & 31, wid = vl >> 5, v = blockIdx.x * TILE + vl;
    unsigned long long key = ~0ull;
    if (v < n) {
        uint32_t cost = 0;
        bool inf = false;
        for (uint32_t c = 0; c < cols; ++c) {
            uint32_t x = tile_at(sm, P, vl, c);
            inf |= x == INF16;
            cost = sum ? cost + x : max(cost, x);
        }
        if (!inf) key = ((unsigned long long)cost << 32) | v;
    }
    for (int o = 16; o > 0; o >>= 1) {
        unsigned long long y = __shfl_down_sync(0xFFFFFFFFu, key, o);
        key = y < key ? y : key;
    }
    if (lane == 0) warp_best[wid] = key;
    __syncthreads();
    if (wid == 0) {
        key = lane < WARPS ? warp_best[lane] : ~0ull;
        for (int o = 16; o > 0; o >>= 1) {
            unsigned long long y = __shfl_down_sync(0xFFFFFFFFu, key, o);
            key = y < key ? y : key;
        }
        if (lane == 0 && key != ~0ull) atomicMin(best, key);
    }
}

// ---- host side -------------------------------------------------------------

enum Mode { PER_LEVEL = 0, GRAPH = 1, COOP = 2 };

struct CchGpu {
    uint32_t n = 0, m = 0, levels = 0, top_levels = 0, max_cols = 0, cols = 0, top_cols_per_block = 1;
    // Dynamic shared memory a head kernel may use (head_opt_in).
    size_t head_smem = 48 * 1024;
    std::vector<uint32_t> level_first;
    uint32_t max_top_level = 0;
    // Shared-memory resident top (one column per block): bytes of dynamic shared memory, 0 if unavailable.
    size_t resident_smem = 0;
    bool use_resident = true;
    uint32_t* d_top_arc = nullptr;
    // Degree-aware wide split (see LevelSplit), one table per LP = 1, 2, 4, 8, 16 (none at LP = 32).
    struct SplitTable {
        uint32_t t = 0;  // threshold, 0 = off
        std::vector<LevelSplit> splits;  // per wide level
        LevelSplit* d_splits = nullptr;
        uint32_t* d_heavy = nullptr;
        size_t heavy_cap = 0;
    } split[5];
    std::vector<uint32_t> h_up_first;
    uint32_t *d_level_first = nullptr, *d_up_first = nullptr, *d_up_head = nullptr, *d_mask = nullptr;
    uint16_t *d_w_fwd = nullptr, *d_w_rev = nullptr, *d_top_tail = nullptr, *d_reduce = nullptr;
    // Packed pair metric (cchgpu_set_pair_weights): when set, every sweep runs on it.
    uint32_t *d_w2_fwd = nullptr, *d_w2_rev = nullptr;
    uint32_t *d_state = nullptr, *d_bits = nullptr;
    uint32_t *d_ids = nullptr, *d_counts = nullptr, *d_chunk_counts = nullptr, *d_chunk_offsets = nullptr;
    uint32_t *d_list = nullptr, *h_list = nullptr, *h_counts = nullptr;
    unsigned long long *d_best = nullptr, *h_best = nullptr;
    void* d_scan_tmp = nullptr;
    size_t list_cap = 0, scan_tmp_bytes = 0;
    uint64_t list_total_pending = 0;
    uint16_t list_t = 0;
    // Address head (cchgpu_set_addresses): piece ends, packed offsets, list scratch per address tile.
    uint32_t n_addr = 0;
    uint32_t *d_addr_u = nullptr, *d_addr_v = nullptr, *d_addr_to = nullptr, *d_addr_from = nullptr;
    uint32_t *d_addr2_u = nullptr, *d_addr2_v = nullptr, *d_addr2_to = nullptr, *d_addr2_from = nullptr;
    uint32_t *d_addr_chunk_counts = nullptr, *d_addr_chunk_offsets = nullptr, *d_addr_out = nullptr;
    void* d_addr_scan_tmp = nullptr;
    size_t addr_scan_bytes = 0, addr_out_cap = 0, addr_bytes = 0;
    uint16_t* d_thresh = nullptr;  // per-column thresholds, max_cols
    ColPiece* d_col_pieces = nullptr;  // max_cols + 1 (cchgpu_set_col_pieces)
    uint32_t* d_bounds = nullptr;      // max_cols + 1: column groups of the pending group head
    uint32_t* d_warp_bits = nullptr;   // max_cols x address tiles x WARPS ballot words (group lists)
    uint32_t list_groups = 0;          // pending list over column groups (0: one list per column)
    int list_op = 0;
    bool has_col_pieces = false;       // for the current sweep
    int last_reversed = 0;
    bool list_addr = false;  // the pending list is over addresses
    Entry* d_entries = nullptr;
    Entry* h_entries = nullptr;  // pinned staging
    uint32_t* h_bits = nullptr;  // pinned
    size_t entries_cap = 0, bits_cap = 0, mem_used = 0;
    uint32_t last_count = 0;
    int coop_blocks = 0, coop_blocks2 = 0;
    cudaStream_t stream = nullptr;
    cudaEvent_t ev[6] = {};
    std::map<std::pair<uint32_t, int>, cudaGraphExec_t> graphs;  // (cols, reversed) -> graph
    char err[512] = {};
};

static thread_local char g_err[512];

#define CK(x)                                                                              \
    do {                                                                                   \
        cudaError_t e_ = (x);                                                              \
        if (e_ != cudaSuccess) {                                                           \
            snprintf(g_err, sizeof g_err, "%s:%d %s: %s", __FILE__, __LINE__, #x,          \
                     cudaGetErrorString(e_));                                              \
            return (int)e_;                                                                \
        }                                                                                  \
    } while (0)

template <class T>
static int dalloc(CchGpu* g, T** p, size_t count) {
    CK(cudaMalloc((void**)p, count * sizeof(T)));
    g->mem_used += count * sizeof(T);
    return 0;
}

static uint32_t lanes_for(uint32_t P) {
    uint32_t lp = 1;
    while (lp < P && lp < 32) lp <<= 1;
    return lp;
}

extern "C" {

const char* cchgpu_last_error() { return g_err; }

int cchgpu_mem_info(size_t* free_b, size_t* total_b) {
    CK(cudaMemGetInfo(free_b, total_b));
    return 0;
}

size_t cchgpu_mem_used(const CchGpu* g) { return g->mem_used; }

// Columns owned by each narrow-top block (default 1). Invalidates captured graphs.
int cchgpu_set_top_cols(CchGpu* g, uint32_t c) {
    if (c == 0 || c > 64) return -1;
    g->top_cols_per_block = c;
    for (auto& kv : g->graphs) cudaGraphExecDestroy(kv.second);
    g->graphs.clear();
    return 0;
}

static int build_split(CchGpu* g, CchGpu::SplitTable& tb, uint32_t t) {
    const uint32_t l0 = g->top_levels, nl = g->levels - l0;
    std::vector<uint32_t> heavy;
    tb.splits.assign(nl, LevelSplit{});
    for (uint32_t l = l0; l < g->levels; ++l) {
        LevelSplit& sp = tb.splits[l - l0];
        sp.first = (uint32_t)heavy.size();
        if (t == 0) continue;
        // Levels that fill the GPU on their own are throughput-bound: no split there.
        const uint32_t LP = 1u << (&tb - g->split);
        if ((uint64_t)(g->level_first[l + 1] - g->level_first[l]) * LP > WIDE_SPLIT_MAX_ITEMS) continue;
        std::vector<std::pair<uint32_t, uint32_t>> h;  // (degree, node)
        for (uint32_t v = g->level_first[l]; v < g->level_first[l + 1]; ++v) {
            uint32_t d = g->h_up_first[v + 1] - g->h_up_first[v];
            if (d > t) h.push_back({d, v});
        }
        std::sort(h.begin(), h.end(), [](auto a, auto b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
        for (auto [d, v] : h) {
            uint32_t groups = 2;  // smallest power of two with ceil(d / groups) <= t, capped at 32
            while (groups < 32 && (d + groups - 1) / groups > t) groups <<= 1;
            int c = 0;
            while ((32u >> c) != groups) ++c;
            sp.count[c]++;
        }
        // Decreasing degree is also decreasing class: heavy[] holds the classes in order.
        for (auto [d, v] : h) heavy.push_back(v);
    }
    if (heavy.size() > tb.heavy_cap) {
        if (tb.d_heavy) cudaFree(tb.d_heavy), g->mem_used -= tb.heavy_cap * 4;
        tb.d_heavy = nullptr;
        size_t cap = std::max<size_t>(heavy.size(), 1);
        if (int rc = dalloc(g, &tb.d_heavy, cap)) return rc;
        tb.heavy_cap = cap;
    }
    if (!tb.d_splits)
        if (int rc = dalloc(g, &tb.d_splits, std::max<uint32_t>(nl, 1))) return rc;
    if (!heavy.empty()) CK(cudaMemcpy(tb.d_heavy, heavy.data(), heavy.size() * 4, cudaMemcpyHostToDevice));
    if (nl) CK(cudaMemcpy(tb.d_splits, tb.splits.data(), nl * sizeof(LevelSplit), cudaMemcpyHostToDevice));
    tb.t = t;
    return 0;
}

// The split table for lane width LP, or null (LP = 32: no room for lane groups).
static const CchGpu::SplitTable* split_for(const CchGpu* g, uint32_t LP) {
    int i = 0;
    while (i < 5 && (1u << i) != LP) ++i;
    return i < 5 ? &g->split[i] : nullptr;
}

// Degree-aware wide levels: nodes with more than t upper arcs get several lane groups.
// t = 0: off; t = UINT32_MAX: the measured default per LP (WIDE_SPLIT_DEFAULT); otherwise t at
// every LP < 32. Invalidates captured graphs.
int cchgpu_set_split(CchGpu* g, uint32_t t) {
    for (int i = 0; i < 5; ++i)
        if (int rc = build_split(g, g->split[i], t == 0xFFFFFFFFu ? WIDE_SPLIT_DEFAULT[i] : t)) return rc;
    for (auto& kv : g->graphs) cudaGraphExecDestroy(kv.second);
    g->graphs.clear();
    return 0;
}

// Narrow-top kernel: 1 = shared-memory resident (default, when it fits), 0 = the global-memory
// kernel. Returns the resident kernel's shared-memory bytes (0 if unavailable). Invalidates graphs.
size_t cchgpu_set_top_resident(CchGpu* g, int on) {
    g->use_resident = on != 0;
    for (auto& kv : g->graphs) cudaGraphExecDestroy(kv.second);
    g->graphs.clear();
    return g->resident_smem;
}

void cchgpu_destroy(CchGpu* g) {
    if (!g) return;
    for (auto& kv : g->graphs) cudaGraphExecDestroy(kv.second);
    void* ptrs[] = {g->d_level_first, g->d_up_first, g->d_up_head, g->d_mask, g->d_w_fwd, g->d_w_rev,
                    g->d_w2_fwd, g->d_w2_rev, g->d_top_tail, g->d_reduce, g->d_state, g->d_bits, g->d_entries, g->d_ids,
                    g->d_counts, g->d_chunk_counts, g->d_chunk_offsets, g->d_list, g->d_best, g->d_scan_tmp,
                    g->d_top_arc, g->d_addr_u, g->d_addr_v, g->d_addr_to, g->d_addr_from,
                    g->d_addr2_u, g->d_addr2_v, g->d_addr2_to, g->d_addr2_from,
                    g->d_addr_chunk_counts, g->d_addr_chunk_offsets, g->d_addr_out, g->d_addr_scan_tmp,
                    g->d_thresh, g->d_col_pieces, g->d_bounds, g->d_warp_bits};
    for (void* p : ptrs)
        if (p) cudaFree(p);
    for (auto& tb : g->split) {
        if (tb.d_splits) cudaFree(tb.d_splits);
        if (tb.d_heavy) cudaFree(tb.d_heavy);
    }
    void* hptrs[] = {g->h_entries, g->h_bits, g->h_list, g->h_counts, g->h_best};
    for (void* p : hptrs)
        if (p) cudaFreeHost(p);
    for (auto& e : g->ev)
        if (e) cudaEventDestroy(e);
    if (g->stream) cudaStreamDestroy(g->stream);
    delete g;
}

// Head kernels: opt in to the device's per-block shared memory, less the address tile's
// static AddrItem stage, so that wide batches fit (a pair batch has 2 columns per place;
// 64 columns use 36 KiB, 128 columns 70 KiB).
static int head_opt_in(CchGpu* g) {
    int dev, optin;
    CK(cudaGetDevice(&dev));
    CK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    const int bytes = optin - (int)(TILE * sizeof(AddrItem));
    if (bytes <= 48 * 1024) return 0;
    const auto a = cudaFuncAttributeMaxDynamicSharedMemorySize;
    CK(cudaFuncSetAttribute(head_count<NodeRows, ColPred<OneT>>, a, bytes));
    CK(cudaFuncSetAttribute(head_count<AddrRows, ColPred<ColT>>, a, bytes));
    CK(cudaFuncSetAttribute(head_count<AddrRows, GroupPred>, a, bytes));
    CK(cudaFuncSetAttribute(head_compact<NodeRows, ColPred<OneT>>, a, bytes));
    CK(cudaFuncSetAttribute(head_compact<AddrRows, ColPred<ColT>>, a, bytes));
    CK(cudaFuncSetAttribute(head_reduce_count<NodeRows, OneT>, a, bytes));
    CK(cudaFuncSetAttribute(head_reduce_count<AddrRows, ColT>, a, bytes));
    CK(cudaFuncSetAttribute(head_meet, a, bytes));
    g->head_smem = bytes;
    return 0;
}

// w_fwd = weights used going down for forward queries (metric.down); w_rev = metric.up.
int cchgpu_create(uint32_t n, uint32_t m, uint32_t levels, const uint32_t* level_first, const uint32_t* up_first,
                  const uint32_t* up_head, const uint16_t* w_fwd, const uint16_t* w_rev, const uint32_t* ids,
                  uint32_t top_levels, uint32_t max_cols, CchGpu** out) {
    *out = nullptr;
    CchGpu* g = new CchGpu();
    g->n = n, g->m = m, g->levels = levels, g->top_levels = std::min(top_levels, levels), g->max_cols = max_cols;
    g->level_first.assign(level_first, level_first + levels + 1);
    g->h_up_first.assign(up_first, up_first + (size_t)n + 1);
    for (uint32_t l = 0; l < g->top_levels; ++l)
        g->max_top_level = std::max(g->max_top_level, level_first[l + 1] - level_first[l]);
    if (g->max_top_level > 65535) {
        snprintf(g_err, sizeof g_err, "top level too wide (%u nodes) for u16 local ids", g->max_top_level);
        delete g;
        return -1;
    }
    // Local index of each top arc's tail inside its level.
    uint32_t top_nodes = level_first[g->top_levels];
    uint32_t top_arcs = up_first[top_nodes];
    std::vector<uint16_t> tail(std::max(top_arcs, 1u));
    for (uint32_t l = 0; l < g->top_levels; ++l)
        for (uint32_t v = level_first[l]; v < level_first[l + 1]; ++v)
            for (uint32_t k = up_first[v]; k < up_first[v + 1]; ++k) tail[k] = (uint16_t)(v - level_first[l]);

    int rc;
    auto fail = [&](int code) { cchgpu_destroy(g); return code; };
    if ((rc = cudaStreamCreateWithFlags(&g->stream, cudaStreamNonBlocking))) return fail(rc);
    for (auto& e : g->ev)
        if ((rc = cudaEventCreate(&e))) return fail(rc);
    uint32_t P = (max_cols + 1) / 2;
    if ((rc = dalloc(g, &g->d_level_first, levels + 1)) || (rc = dalloc(g, &g->d_up_first, (size_t)n + 1)) ||
        (rc = dalloc(g, &g->d_up_head, m)) || (rc = dalloc(g, &g->d_w_fwd, m)) ||
        (rc = dalloc(g, &g->d_w_rev, m)) || (rc = dalloc(g, &g->d_top_tail, tail.size())) ||
        (rc = dalloc(g, &g->d_mask, (n + 31) / 32)) || (rc = dalloc(g, &g->d_reduce, n)) ||
        (rc = dalloc(g, &g->d_state, (size_t)n * P)) ||
        (rc = dalloc(g, &g->d_bits, (size_t)((n + 31) / 32) * max_cols)) || (rc = dalloc(g, &g->d_ids, n)) ||
        (rc = dalloc(g, &g->d_counts, max_cols)) || (rc = dalloc(g, &g->d_thresh, max_cols)) ||
        (rc = dalloc(g, &g->d_chunk_counts, (size_t)((n + TILE - 1) / TILE) * max_cols)) ||
        (rc = dalloc(g, &g->d_chunk_offsets, (size_t)((n + TILE - 1) / TILE) * max_cols)) ||
        (rc = dalloc(g, &g->d_best, 1)))
        return fail(rc);
    if ((rc = cudaMallocHost((void**)&g->h_counts, max_cols * 4)) || (rc = cudaMallocHost((void**)&g->h_best, 8)))
        return fail(rc);
    cudaMemcpy(g->d_ids, ids, (size_t)n * 4, cudaMemcpyHostToDevice);
    {
        size_t bytes = 0;
        cub::DeviceScan::ExclusiveSum(nullptr, bytes, g->d_chunk_counts, g->d_chunk_offsets,
                                      (int)(((n + TILE - 1) / TILE) * max_cols));
        if ((rc = dalloc(g, (uint8_t**)&g->d_scan_tmp, bytes))) return fail(rc);
        g->scan_tmp_bytes = bytes;
    }
    g->bits_cap = (size_t)((n + 31) / 32) * max_cols;
    if ((rc = cudaMallocHost((void**)&g->h_bits, g->bits_cap * 4))) return fail(rc);
    cudaMemcpy(g->d_level_first, level_first, (levels + 1) * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(g->d_up_first, up_first, ((size_t)n + 1) * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(g->d_up_head, up_head, (size_t)m * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(g->d_w_fwd, w_fwd, (size_t)m * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(g->d_w_rev, w_rev, (size_t)m * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(g->d_top_tail, tail.data(), tail.size() * 2, cudaMemcpyHostToDevice);
    {
        // Resident top: needs u16 head ids (top_nodes <= 65536) and the shared memory to hold them.
        int dev, optin;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        size_t bytes = (2 * ((size_t)g->top_levels + 1) + 2 * (size_t)g->max_top_level) * 4 + ((size_t)top_nodes * 2 + 3) / 4 * 4;
        if (g->top_levels > 0 && top_nodes <= 65536 && bytes <= (size_t)optin) {
            std::vector<uint32_t> packed(std::max(top_arcs, 1u));
            for (uint32_t k = 0; k < top_arcs; ++k) packed[k] = up_head[k] | ((uint32_t)tail[k] << 16);
            if ((rc = dalloc(g, &g->d_top_arc, packed.size()))) return fail(rc);
            cudaMemcpy(g->d_top_arc, packed.data(), packed.size() * 4, cudaMemcpyHostToDevice);
            // The limit is the kernel's, for the process: with several graphs (one per mode,
            // France ACTIVE phase Step 4) it only grows, to the largest graph's need.
            static std::mutex attr_mu;
            static size_t attr_bytes = 0;
            std::lock_guard<std::mutex> lock(attr_mu);
            if (bytes > attr_bytes) {
                for (auto f : {top_resident<false>, top_resident<true>, top_resident<true, MERGE_ATOMIC>,
                               top_resident<true, MERGE_SEGMENT>})
                    if ((rc = cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes)))
                        return fail(rc);
                if ((rc = cudaFuncSetAttribute(top_resident<false, MERGE_WARP, uint32_t>,
                                               cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes)))
                    return fail(rc);
                attr_bytes = bytes;
            }
            g->resident_smem = bytes;
        }
    }
    if ((rc = cudaMemset(g->d_mask, 0, ((n + 31) / 32) * 4))) return fail(rc);
    int dev, sms, per_sm;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, wide_coop<uint16_t>, 256, 0);
    g->coop_blocks = sms * per_sm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, wide_coop<uint32_t>, 256, 0);
    g->coop_blocks2 = sms * per_sm;
    if ((rc = cchgpu_set_split(g, 0xFFFFFFFFu))) return fail(rc);
    if ((rc = head_opt_in(g))) return fail(rc);
    if ((rc = cudaDeviceSynchronize())) return fail(rc);
    *out = g;
    return 0;
}

}  // extern "C"

// Enqueue top + wide levels for the current batch, on weights w (u16, or a packed pair).
template <class W>
static int enqueue_down_w(CchGpu* g, const W* w, int coop) {
    const uint32_t P = (g->cols + 1) / 2, LP = lanes_for(P);
    uint16_t* d16 = (uint16_t*)g->d_state;
    if (g->top_levels > 0 && g->use_resident && g->resident_smem && g->top_cols_per_block == 1) {
        top_resident<false, MERGE_WARP, W><<<g->cols, RES_THREADS, g->resident_smem, g->stream>>>(
            g->d_level_first, g->top_levels, g->d_up_first, g->d_top_arc, w, d16, P, g->max_top_level, g->d_mask,
            nullptr);
        CK(cudaGetLastError());
    } else if (g->top_levels > 0) {
        uint32_t cpb = g->top_cols_per_block;
        uint32_t blocks = (g->cols + cpb - 1) / cpb;
        size_t smem = (size_t)g->max_top_level * cpb * 4;
        if (smem > 48 * 1024) {
            snprintf(g_err, sizeof g_err, "top accumulator %zu B exceeds 48 KiB (lower cap or top cols)", smem);
            return -1;
        }
        top_kernel<TOP_RUN, W><<<blocks, 1024, smem, g->stream>>>(g->d_level_first, g->top_levels, g->d_up_first,
                                                               g->d_up_head, w, g->d_top_tail, d16, P, g->cols,
                                                               cpb, g->d_mask, nullptr);
        CK(cudaGetLastError());
    }
    CK(cudaEventRecord(g->ev[2], g->stream));
    const CchGpu::SplitTable* tb = split_for(g, LP);
    const uint32_t skip_t = tb && tb->t ? tb->t : 0xFFFFFFFFu;
    if (!tb) tb = &g->split[0];  // never consulted: skip_t is UINT32_MAX
    LevelSplit* d_splits = tb->d_splits;
    uint32_t* d_heavy = tb->d_heavy;
    if (coop) {
        uint32_t l0 = g->top_levels, l1 = g->levels;
        uint32_t* d32 = g->d_state;
        void* args[] = {&g->d_level_first, &l0, &l1, (void*)&LP, &d_splits, &d_heavy, (void*)&skip_t,
                        &g->d_up_first, &g->d_up_head, (void*)&w, &d32, (void*)&P, &g->d_mask};
        const int blocks = sizeof(W) == 2 ? g->coop_blocks : g->coop_blocks2;
        CK(cudaLaunchCooperativeKernel((void*)wide_coop<W>, blocks, 256, args, 0, g->stream));
    } else {
        for (uint32_t l = g->top_levels; l < g->levels; ++l) {
            uint32_t base = g->level_first[l], cnt = g->level_first[l + 1] - base;
            launch_level(base, cnt, LP, tb->splits[l - g->top_levels], d_heavy, skip_t, g->d_up_first,
                         g->d_up_head, w, g->d_state, P, g->d_mask, g->stream);
            CK(cudaGetLastError());
        }
    }
    return 0;
}

extern "C" {

static int enqueue_down(CchGpu* g, int reversed, int coop) {
    if (g->d_w2_fwd) return enqueue_down_w(g, reversed ? g->d_w2_rev : g->d_w2_fwd, coop);
    return enqueue_down_w(g, reversed ? g->d_w_rev : g->d_w_fwd, coop);
}

// Modes phase, Step 4: sweep on a packed pair metric from now on (half h of word k = the
// weight of arc k for columns 2p + h; the batch width must then be even). Same layout as
// the u16 weights: w_fwd for forward sweeps, w_rev for reversed ones.
int cchgpu_set_pair_weights(CchGpu* g, const uint32_t* w_fwd, const uint32_t* w_rev) {
    int rc;
    if (!g->d_w2_fwd && ((rc = dalloc(g, &g->d_w2_fwd, g->m)) || (rc = dalloc(g, &g->d_w2_rev, g->m)))) {
        snprintf(g_err, sizeof g_err, "pair weights: allocation failed (%d)", rc);
        return rc;
    }
    CK(cudaMemcpy(g->d_w2_fwd, w_fwd, (size_t)g->m * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(g->d_w2_rev, w_rev, (size_t)g->m * 4, cudaMemcpyHostToDevice));
    for (auto& kv : g->graphs) cudaGraphExecDestroy(kv.second);
    g->graphs.clear();
    return 0;
}

// timings_ms[4] = {upload (H2D entries + seeding), narrow top, wide levels, total}.
int cchgpu_sweep(CchGpu* g, uint32_t cols, const Entry* entries, uint32_t count, int reversed, int mode,
                 float* timings_ms) {
    if (cols == 0 || cols > g->max_cols) {
        snprintf(g_err, sizeof g_err, "cols %u outside 1..%u", cols, g->max_cols);
        return -1;
    }
    if (count > g->entries_cap) {
        // The old entries (needed to clear the previous seed bits) are about to go: clear the whole mask.
        CK(cudaMemsetAsync(g->d_mask, 0, ((g->n + 31) / 32) * 4, g->stream));
        CK(cudaStreamSynchronize(g->stream));
        g->last_count = 0;
        if (g->d_entries) cudaFree(g->d_entries), g->mem_used -= g->entries_cap * sizeof(Entry);
        if (g->h_entries) cudaFreeHost(g->h_entries);
        size_t cap = std::max<size_t>(count, 4096) * 2;
        CK(cudaMalloc((void**)&g->d_entries, cap * sizeof(Entry)));
        CK(cudaMallocHost((void**)&g->h_entries, cap * sizeof(Entry)));
        g->mem_used += cap * sizeof(Entry);
        g->entries_cap = cap;
    }
    if (g->d_w2_fwd && cols % 2) {
        snprintf(g_err, sizeof g_err, "pair metric: cols %u must be even", cols);
        return -1;
    }
    const uint32_t P = (cols + 1) / 2;
    CK(cudaEventRecord(g->ev[0], g->stream));
    // Clear the previous batch's seed bits (only the words it touched).
    if (g->last_count > 0) {
        clear_mask<<<(g->last_count + 255) / 256, 256, 0, g->stream>>>(g->d_entries, g->last_count, g->d_mask);
        CK(cudaGetLastError());
    }
    g->cols = cols;
    g->last_reversed = reversed;
    g->has_col_pieces = false;
    std::memcpy(g->h_entries, entries, (size_t)count * sizeof(Entry));
    CK(cudaMemcpyAsync(g->d_entries, g->h_entries, (size_t)count * sizeof(Entry), cudaMemcpyHostToDevice,
                       g->stream));
    if (count > 0) {
        seed_rows<<<(count + 255) / 256, 256, 0, g->stream>>>(g->d_entries, count, g->d_state, P, g->d_mask);
        seed_values<<<(count + 255) / 256, 256, 0, g->stream>>>(g->d_entries, count, (uint16_t*)g->d_state, P);
        CK(cudaGetLastError());
    }
    g->last_count = count;
    CK(cudaEventRecord(g->ev[1], g->stream));
    if (mode == GRAPH) {
        auto key = std::make_pair(cols, reversed);
        auto it = g->graphs.find(key);
        if (it == g->graphs.end()) {
            cudaGraph_t graph;
            CK(cudaStreamBeginCapture(g->stream, cudaStreamCaptureModeThreadLocal));
            int rc = enqueue_down(g, reversed, 0);
            CK(cudaStreamEndCapture(g->stream, &graph));
            if (rc) return rc;
            cudaGraphExec_t exec;
            CK(cudaGraphInstantiate(&exec, graph, 0));
            cudaGraphDestroy(graph);
            it = g->graphs.emplace(key, exec).first;
        }
        CK(cudaGraphLaunch(it->second, g->stream));
        // ev[2] was recorded inside the captured graph; measure top+wide together in GRAPH mode.
        CK(cudaEventRecord(g->ev[2], g->stream));
    } else {
        int rc = enqueue_down(g, reversed, mode == COOP);
        if (rc) return rc;
    }
    CK(cudaEventRecord(g->ev[3], g->stream));
    CK(cudaEventSynchronize(g->ev[3]));
    if (timings_ms) {
        CK(cudaEventElapsedTime(&timings_ms[0], g->ev[0], g->ev[1]));
        if (mode == GRAPH) {
            CK(cudaEventElapsedTime(&timings_ms[2], g->ev[1], g->ev[3]));
            timings_ms[1] = 0.0f;  // not separable inside the graph
        } else {
            CK(cudaEventElapsedTime(&timings_ms[1], g->ev[1], g->ev[2]));
            CK(cudaEventElapsedTime(&timings_ms[2], g->ev[2], g->ev[3]));
        }
        CK(cudaEventElapsedTime(&timings_ms[3], g->ev[0], g->ev[3]));
    }
    return 0;
}

// Tests only: the full state, [node][col] u16, cols = current batch width.
int cchgpu_download(CchGpu* g, uint16_t* out) {
    const uint32_t P = (g->cols + 1) / 2;
    std::vector<uint16_t> tmp((size_t)g->n * 2 * P);
    CK(cudaMemcpy(tmp.data(), g->d_state, tmp.size() * 2, cudaMemcpyDeviceToHost));
    for (size_t v = 0; v < g->n; ++v) std::memcpy(out + v * g->cols, tmp.data() + v * 2 * P, g->cols * 2);
    return 0;
}

// bits[c][word] for all columns of the current batch. ms = kernel + D2H.
int cchgpu_threshold_bits(CchGpu* g, uint16_t t, uint32_t* out, float* ms) {
    const uint32_t P = (g->cols + 1) / 2, words = (g->n + 31) / 32;
    CK(cudaEventRecord(g->ev[4], g->stream));
    head_threshold<<<(words * 32 + 255) / 256, 256, 0, g->stream>>>((uint16_t*)g->d_state, g->n, P, g->cols, t,
                                                                    g->d_bits);
    CK(cudaGetLastError());
    CK(cudaMemcpyAsync(g->h_bits, g->d_bits, (size_t)words * g->cols * 4, cudaMemcpyDeviceToHost, g->stream));
    CK(cudaEventRecord(g->ev[5], g->stream));
    CK(cudaEventSynchronize(g->ev[5]));
    std::memcpy(out, g->h_bits, (size_t)words * g->cols * 4);
    if (ms) CK(cudaEventElapsedTime(ms, g->ev[4], g->ev[5]));
    return 0;
}

// r[v] = min (op 0) or max (op 1) over columns; kept on the device. If out != NULL, also download (tests).
int cchgpu_reduce(CchGpu* g, int op, uint16_t* out) {
    const uint32_t P = (g->cols + 1) / 2;
    head_reduce<<<(g->n + 255) / 256, 256, 0, g->stream>>>((uint16_t*)g->d_state, g->n, P, g->cols, op, g->d_reduce);
    CK(cudaGetLastError());
    if (out) CK(cudaMemcpyAsync(out, g->d_reduce, (size_t)g->n * 2, cudaMemcpyDeviceToHost, g->stream));
    CK(cudaStreamSynchronize(g->stream));
    return 0;
}

// Bits of (reduce over columns) <= t. ms = both kernels + D2H.
int cchgpu_reduce_bits(CchGpu* g, int op, uint16_t t, uint32_t* out, float* ms) {
    const uint32_t P = (g->cols + 1) / 2, words = (g->n + 31) / 32;
    CK(cudaEventRecord(g->ev[4], g->stream));
    head_reduce<<<(g->n + 255) / 256, 256, 0, g->stream>>>((uint16_t*)g->d_state, g->n, P, g->cols, op, g->d_reduce);
    head_bits<<<(words * 32 + 255) / 256, 256, 0, g->stream>>>(g->d_reduce, g->n, t, g->d_bits);
    CK(cudaGetLastError());
    CK(cudaMemcpyAsync(g->h_bits, g->d_bits, (size_t)words * 4, cudaMemcpyDeviceToHost, g->stream));
    CK(cudaEventRecord(g->ev[5], g->stream));
    CK(cudaEventSynchronize(g->ev[5]));
    std::memcpy(out, g->h_bits, (size_t)words * 4);
    if (ms) CK(cudaEventElapsedTime(ms, g->ev[4], g->ev[5]));
    return 0;
}

// Head timings: ms[0] = kernels, ms[1] = device-to-host copies.
static size_t tile_smem(uint32_t P, uint32_t cols) { return (size_t)TILE * (P + 1) * 4 + (size_t)cols * WARPS * 4; }

static int head_counts_impl(CchGpu* g, bool chunks) {
    const uint32_t ntiles = (g->n + TILE - 1) / TILE, P = (g->cols + 1) / 2;
    size_t smem = tile_smem(P, g->cols);
    if (smem > g->head_smem) {
        snprintf(g_err, sizeof g_err, "head tile needs %zu B shared memory (B too large)", smem);
        return -1;
    }
    CK(cudaMemsetAsync(g->d_counts, 0, g->cols * 4, g->stream));
    head_count<<<ntiles, TILE, smem, g->stream>>>(NodeRows{g->d_state, g->n, P}, P, g->cols, ColPred<OneT>{OneT{g->list_t}},
                                                  g->d_counts, chunks ? g->d_chunk_counts : nullptr);
    CK(cudaGetLastError());
    return 0;
}

// counts[c] = #{v : d[v][c] <= t}.
int cchgpu_counts(CchGpu* g, uint16_t t, uint32_t* out, float* ms) {
    CK(cudaEventRecord(g->ev[4], g->stream));
    g->list_t = t;
    int rc = head_counts_impl(g, false);
    if (rc) return rc;
    CK(cudaEventRecord(g->ev[5], g->stream));
    CK(cudaMemcpyAsync(g->h_counts, g->d_counts, g->cols * 4, cudaMemcpyDeviceToHost, g->stream));
    CK(cudaEventRecord(g->ev[0], g->stream));
    CK(cudaEventSynchronize(g->ev[0]));
    std::memcpy(out, g->h_counts, g->cols * 4);
    if (ms) {
        CK(cudaEventElapsedTime(&ms[0], g->ev[4], g->ev[5]));
        CK(cudaEventElapsedTime(&ms[1], g->ev[5], g->ev[0]));
    }
    return 0;
}

// Two calls: lists_begin computes counts (out_counts[cols]) and returns the total via *total;
// lists_finish(out) writes the concatenated lists (column-major, ascending layout id). ms as above.
int cchgpu_lists_begin(CchGpu* g, uint16_t t, uint32_t* out_counts, uint64_t* total, float* ms) {
    CK(cudaEventRecord(g->ev[4], g->stream));
    g->list_t = t;
    int rc = head_counts_impl(g, true);
    if (rc) return rc;
    CK(cudaEventRecord(g->ev[5], g->stream));
    CK(cudaMemcpyAsync(g->h_counts, g->d_counts, g->cols * 4, cudaMemcpyDeviceToHost, g->stream));
    CK(cudaEventRecord(g->ev[0], g->stream));
    CK(cudaEventSynchronize(g->ev[0]));
    uint64_t sum = 0;
    for (uint32_t c = 0; c < g->cols; ++c) sum += g->h_counts[c];
    std::memcpy(out_counts, g->h_counts, g->cols * 4);
    *total = sum;
    if (ms) {
        CK(cudaEventElapsedTime(&ms[0], g->ev[4], g->ev[5]));
        CK(cudaEventElapsedTime(&ms[1], g->ev[5], g->ev[0]));
    }
    g->list_total_pending = sum;
    g->list_t = t;
    g->list_addr = false;
    g->list_groups = 0;
    return 0;
}

static int ensure_list_cap(CchGpu* g, uint64_t total) {
    if (total > g->list_cap) {
        if (g->d_list) cudaFree(g->d_list), g->mem_used -= g->list_cap * 4;
        if (g->h_list) cudaFreeHost(g->h_list);
        g->d_list = nullptr, g->h_list = nullptr, g->list_cap = 0;
        size_t cap = total + total / 4 + 1024;
        CK(cudaMalloc((void**)&g->d_list, cap * 4));
        g->mem_used += cap * 4;
        CK(cudaMallocHost((void**)&g->h_list, cap * 4));
        g->list_cap = cap;
    }
    return 0;
}

static AddrRows addr_rows(const CchGpu* g) {
    return AddrRows{g->d_state, g->d_addr_u, g->d_addr_v, g->last_reversed ? g->d_addr_from : g->d_addr_to,
                    g->n_addr, (g->cols + 1) / 2, g->d_addr_to, g->has_col_pieces ? g->d_col_pieces : nullptr,
                    g->last_reversed, g->d_addr2_u, g->d_addr2_v,
                    g->d_addr2_u ? (g->last_reversed ? g->d_addr2_from : g->d_addr2_to) : nullptr, g->d_addr2_to};
}

int cchgpu_lists_finish(CchGpu* g, uint32_t* out, float* ms) {
    const uint64_t total = g->list_total_pending;
    const uint32_t P = (g->cols + 1) / 2;
    int rc = ensure_list_cap(g, total);
    if (rc) return rc;
    CK(cudaEventRecord(g->ev[4], g->stream));
    if (g->list_addr) {
        const uint32_t nchunks = (g->n_addr + TILE - 1) / TILE, outs = g->list_groups ? g->list_groups : g->cols;
        size_t bytes = g->addr_scan_bytes;
        CK(cub::DeviceScan::ExclusiveSum(g->d_addr_scan_tmp, bytes, g->d_addr_chunk_counts, g->d_addr_chunk_offsets,
                                         (int)(nchunks * outs), g->stream));
        if (g->list_groups)
            bits_compact<<<nchunks, TILE, 0, g->stream>>>(g->d_warp_bits, outs, g->d_addr_chunk_offsets, g->d_list);
        else
            head_compact<<<nchunks, TILE, tile_smem(P, g->cols), g->stream>>>(
                addr_rows(g), P, g->cols, ColPred<ColT>{ColT{g->d_thresh}}, g->d_addr_chunk_offsets, nullptr, g->d_list);
    } else {
        const uint32_t nchunks = (g->n + TILE - 1) / TILE;
        size_t bytes = g->scan_tmp_bytes;
        CK(cub::DeviceScan::ExclusiveSum(g->d_scan_tmp, bytes, g->d_chunk_counts, g->d_chunk_offsets,
                                         (int)(nchunks * g->cols), g->stream));
        head_compact<<<nchunks, TILE, tile_smem(P, g->cols), g->stream>>>(
            NodeRows{g->d_state, g->n, P}, P, g->cols, ColPred<OneT>{OneT{g->list_t}}, g->d_chunk_offsets, g->d_ids, g->d_list);
    }
    CK(cudaGetLastError());
    CK(cudaEventRecord(g->ev[5], g->stream));
    if (total) CK(cudaMemcpyAsync(g->h_list, g->d_list, total * 4, cudaMemcpyDeviceToHost, g->stream));
    CK(cudaEventRecord(g->ev[0], g->stream));
    CK(cudaEventSynchronize(g->ev[0]));
    // out = null: the list stays in the pinned buffer (cchgpu_list_host) until the next head call.
    if (total && out) std::memcpy(out, g->h_list, total * 4);
    if (ms) {
        CK(cudaEventElapsedTime(&ms[0], g->ev[4], g->ev[5]));
        CK(cudaEventElapsedTime(&ms[1], g->ev[5], g->ev[0]));
    }
    return 0;
}

// #{v : (min|max over columns of d[v][c]) <= t}: union (op 0) / intersection (op 1) count.
int cchgpu_reduce_count(CchGpu* g, int op, uint16_t t, uint32_t* out, float* ms) {
    const uint32_t P = (g->cols + 1) / 2;
    CK(cudaEventRecord(g->ev[4], g->stream));
    CK(cudaMemsetAsync(g->d_counts, 0, 4, g->stream));
    head_reduce_count<<<(g->n + TILE - 1) / TILE, TILE, tile_smem(P, 0), g->stream>>>(
        NodeRows{g->d_state, g->n, P}, P, g->cols, op, OneT{t}, g->d_counts);
    CK(cudaGetLastError());
    CK(cudaEventRecord(g->ev[5], g->stream));
    CK(cudaMemcpyAsync(g->h_counts, g->d_counts, 4, cudaMemcpyDeviceToHost, g->stream));
    CK(cudaEventRecord(g->ev[0], g->stream));
    CK(cudaEventSynchronize(g->ev[0]));
    *out = g->h_counts[0];
    if (ms) {
        CK(cudaEventElapsedTime(&ms[0], g->ev[4], g->ev[5]));
        CK(cudaEventElapsedTime(&ms[1], g->ev[5], g->ev[0]));
    }
    return 0;
}

// Argmin over nodes of max (sum = 0) or sum (sum = 1) across columns. *key = (cost << 32) | layout id,
// or ~0 if every node is unreachable from some column.
int cchgpu_meeting_point(CchGpu* g, int sum, unsigned long long* key, float* ms) {
    const uint32_t P = (g->cols + 1) / 2;
    CK(cudaEventRecord(g->ev[4], g->stream));
    CK(cudaMemsetAsync(g->d_best, 0xFF, 8, g->stream));
    head_meet<<<(g->n + TILE - 1) / TILE, TILE, tile_smem(P, 0), g->stream>>>(g->d_state, g->n, P, g->cols, sum,
                                                                             g->d_best);
    CK(cudaGetLastError());
    CK(cudaEventRecord(g->ev[5], g->stream));
    CK(cudaMemcpyAsync(g->h_best, g->d_best, 8, cudaMemcpyDeviceToHost, g->stream));
    CK(cudaEventRecord(g->ev[0], g->stream));
    CK(cudaEventSynchronize(g->ev[0]));
    *key = *g->h_best;
    if (ms) {
        CK(cudaEventElapsedTime(&ms[0], g->ev[4], g->ev[5]));
        CK(cudaEventElapsedTime(&ms[1], g->ev[5], g->ev[0]));
    }
    return 0;
}

// ---- address head ------------------------------------------------------------
// Address a: piece ends u[a], v[a] (layout ids); to[a] = to_u | to_v << 16 (used after a forward
// sweep), from[a] = from_u | from_v << 16 (after a reversed sweep). Replaces any previous table.
int cchgpu_set_addresses(CchGpu* g, uint32_t count, const uint32_t* u, const uint32_t* v, const uint32_t* to,
                         const uint32_t* from) {
    uint32_t** bufs[] = {&g->d_addr_u, &g->d_addr_v, &g->d_addr_to, &g->d_addr_from, &g->d_addr_chunk_counts,
                         &g->d_addr_chunk_offsets};
    const size_t nchunks = (count + TILE - 1) / TILE, sizes[] = {count, count, count, count,
                                                                 nchunks * g->max_cols, nchunks * g->max_cols};
    for (int i = 0; i < 6; ++i)
        if (*bufs[i]) cudaFree(*bufs[i]), *bufs[i] = nullptr;
    if (g->d_addr_scan_tmp) cudaFree(g->d_addr_scan_tmp), g->d_addr_scan_tmp = nullptr;
    g->mem_used -= g->addr_bytes;
    // A second table (cchgpu_set_addresses2) belongs to the old first one.
    for (uint32_t** b : {&g->d_addr2_u, &g->d_addr2_v, &g->d_addr2_to, &g->d_addr2_from})
        if (*b) cudaFree(*b), *b = nullptr, g->mem_used -= std::max<size_t>(g->n_addr, 1) * 4;
    const size_t before = g->mem_used;
    g->n_addr = 0;
    for (int i = 0; i < 6; ++i) {
        int rc = dalloc(g, bufs[i], std::max<size_t>(sizes[i], 1));
        if (rc) return rc;
    }
    const uint32_t* src[] = {u, v, to, from};
    for (int i = 0; i < 4; ++i) CK(cudaMemcpy(*bufs[i], src[i], (size_t)count * 4, cudaMemcpyHostToDevice));
    size_t bytes = 0;
    CK(cub::DeviceScan::ExclusiveSum(nullptr, bytes, g->d_addr_chunk_counts, g->d_addr_chunk_offsets,
                                     (int)(nchunks * g->max_cols)));
    int rc = dalloc(g, (uint8_t**)&g->d_addr_scan_tmp, std::max<size_t>(bytes, 1));
    if (rc) return rc;
    g->addr_scan_bytes = bytes;
    g->addr_bytes = g->mem_used - before;
    g->n_addr = count;
    return 0;
}

// Pair sweeps: the address table of the upper half of every word (same addresses, same
// count as cchgpu_set_addresses, snapped by the second mode).
int cchgpu_set_addresses2(CchGpu* g, uint32_t count, const uint32_t* u, const uint32_t* v, const uint32_t* to,
                          const uint32_t* from) {
    if (count != g->n_addr) {
        snprintf(g_err, sizeof g_err, "second address table: %u addresses, first has %u", count, g->n_addr);
        return -1;
    }
    uint32_t** bufs[] = {&g->d_addr2_u, &g->d_addr2_v, &g->d_addr2_to, &g->d_addr2_from};
    const uint32_t* src[] = {u, v, to, from};
    for (int i = 0; i < 4; ++i) {
        if (!*bufs[i]) {
            int rc = dalloc(g, bufs[i], std::max<size_t>(count, 1));
            if (rc) return rc;
        }
        CK(cudaMemcpy(*bufs[i], src[i], (size_t)count * 4, cudaMemcpyHostToDevice));
    }
    return 0;
}

static int addr_head_begin(CchGpu* g, const uint16_t* t) {
    if (!g->n_addr) {
        snprintf(g_err, sizeof g_err, "no address table (cchgpu_set_addresses)");
        return -1;
    }
    if (tile_smem((g->cols + 1) / 2, g->cols) > g->head_smem) {
        snprintf(g_err, sizeof g_err, "head tile needs %zu B shared memory, %zu available (B too large)",
                 tile_smem((g->cols + 1) / 2, g->cols), g->head_smem);
        return -1;
    }
    CK(cudaMemcpyAsync(g->d_thresh, t, g->cols * 2, cudaMemcpyHostToDevice, g->stream));
    return 0;
}

static int addr_counts_impl(CchGpu* g, bool chunks) {
    const uint32_t P = (g->cols + 1) / 2;
    CK(cudaMemsetAsync(g->d_counts, 0, g->cols * 4, g->stream));
    head_count<<<(g->n_addr + TILE - 1) / TILE, TILE, tile_smem(P, g->cols), g->stream>>>(
        addr_rows(g), P, g->cols, ColPred<ColT>{ColT{g->d_thresh}}, g->d_counts, chunks ? g->d_addr_chunk_counts : nullptr);
    CK(cudaGetLastError());
    return 0;
}

// counts[c] = #{a : t_addr[a][c] <= t[c]}. ms[0] = kernels, ms[1] = D2H.
int cchgpu_addr_counts(CchGpu* g, const uint16_t* t, uint32_t* out, float* ms) {
    int rc = addr_head_begin(g, t);
    if (rc) return rc;
    CK(cudaEventRecord(g->ev[4], g->stream));
    if ((rc = addr_counts_impl(g, false))) return rc;
    CK(cudaEventRecord(g->ev[5], g->stream));
    CK(cudaMemcpyAsync(g->h_counts, g->d_counts, g->cols * 4, cudaMemcpyDeviceToHost, g->stream));
    CK(cudaEventRecord(g->ev[0], g->stream));
    CK(cudaEventSynchronize(g->ev[0]));
    std::memcpy(out, g->h_counts, g->cols * 4);
    if (ms) {
        CK(cudaEventElapsedTime(&ms[0], g->ev[4], g->ev[5]));
        CK(cudaEventElapsedTime(&ms[1], g->ev[5], g->ev[0]));
    }
    return 0;
}

// Address lists, two calls like cchgpu_lists_begin / cchgpu_lists_finish (ids = address index, ascending).
int cchgpu_addr_lists_begin(CchGpu* g, const uint16_t* t, uint32_t* out_counts, uint64_t* total, float* ms) {
    int rc = addr_head_begin(g, t);
    if (rc) return rc;
    CK(cudaEventRecord(g->ev[4], g->stream));
    if ((rc = addr_counts_impl(g, true))) return rc;
    CK(cudaEventRecord(g->ev[5], g->stream));
    CK(cudaMemcpyAsync(g->h_counts, g->d_counts, g->cols * 4, cudaMemcpyDeviceToHost, g->stream));
    CK(cudaEventRecord(g->ev[0], g->stream));
    CK(cudaEventSynchronize(g->ev[0]));
    uint64_t sum = 0;
    for (uint32_t c = 0; c < g->cols; ++c) sum += g->h_counts[c];
    std::memcpy(out_counts, g->h_counts, g->cols * 4);
    *total = sum;
    if (ms) {
        CK(cudaEventElapsedTime(&ms[0], g->ev[4], g->ev[5]));
        CK(cudaEventElapsedTime(&ms[1], g->ev[5], g->ev[0]));
    }
    g->list_total_pending = sum;
    g->list_addr = true;
    g->list_groups = 0;
    return 0;
}

// #{a : t_addr[a][c] <= t[c] for some (op 0) / every (op 1) column c}.
int cchgpu_addr_reduce_count(CchGpu* g, int op, const uint16_t* t, uint32_t* out, float* ms) {
    int rc = addr_head_begin(g, t);
    if (rc) return rc;
    const uint32_t P = (g->cols + 1) / 2;
    CK(cudaEventRecord(g->ev[4], g->stream));
    CK(cudaMemsetAsync(g->d_counts, 0, 4, g->stream));
    head_reduce_count<<<(g->n_addr + TILE - 1) / TILE, TILE, tile_smem(P, 0), g->stream>>>(
        addr_rows(g), P, g->cols, op, ColT{g->d_thresh}, g->d_counts);
    CK(cudaGetLastError());
    CK(cudaEventRecord(g->ev[5], g->stream));
    CK(cudaMemcpyAsync(g->h_counts, g->d_counts, 4, cudaMemcpyDeviceToHost, g->stream));
    CK(cudaEventRecord(g->ev[0], g->stream));
    CK(cudaEventSynchronize(g->ev[0]));
    *out = g->h_counts[0];
    if (ms) {
        CK(cudaEventElapsedTime(&ms[0], g->ev[4], g->ev[5]));
        CK(cudaEventElapsedTime(&ms[1], g->ev[5], g->ev[0]));
    }
    return 0;
}

// Column groups (France reference phase, Step 2): group k = columns [bounds[k], bounds[k + 1]),
// bounds[0] = 0, bounds[groups] = cols, every group non-empty. An address is in group k when
// t_addr[a][c] <= t[c] for some (op 0, union) / every (op 1, intersection) column c of the group:
// one answer per query of the batch, merged on the GPU.
static int group_head_begin(CchGpu* g, int op, const uint16_t* t, uint32_t groups, const uint32_t* bounds) {
    int rc = addr_head_begin(g, t);
    if (rc) return rc;
    bool ok = groups > 0 && groups <= g->cols && bounds[0] == 0 && bounds[groups] == g->cols && (op == 0 || op == 1);
    for (uint32_t k = 0; ok && k < groups; ++k) ok = bounds[k] < bounds[k + 1];
    if (!ok) {
        snprintf(g_err, sizeof g_err, "bad column groups (%u groups over %u columns, op %d)", groups, g->cols, op);
        return -1;
    }
    if (!g->d_bounds && (rc = dalloc(g, &g->d_bounds, g->max_cols + 1))) return rc;
    CK(cudaMemcpyAsync(g->d_bounds, bounds, (groups + 1) * 4, cudaMemcpyHostToDevice, g->stream));
    return 0;
}

// With `lists`, also the chunk counts and the warp ballot words for bits_compact.
static int group_counts_impl(CchGpu* g, int op, uint32_t groups, bool lists) {
    const uint32_t P = (g->cols + 1) / 2, nchunks = (g->n_addr + TILE - 1) / TILE;
    int rc;
    if (lists && !g->d_warp_bits && (rc = dalloc(g, &g->d_warp_bits, (size_t)g->max_cols * nchunks * WARPS))) return rc;
    uint32_t* cc = lists ? g->d_addr_chunk_counts : nullptr;
    uint32_t* wb = lists ? g->d_warp_bits : nullptr;
    CK(cudaMemsetAsync(g->d_counts, 0, groups * 4, g->stream));
    // One column per group (single-source queries): the plain column test, same answer.
    if (groups == g->cols)
        head_count<<<nchunks, TILE, tile_smem(P, groups), g->stream>>>(addr_rows(g), P, groups, ColPred<ColT>{ColT{g->d_thresh}},
                                                                     g->d_counts, cc, wb);
    else
        head_count<<<nchunks, TILE, tile_smem(P, groups), g->stream>>>(
            addr_rows(g), P, groups, GroupPred{g->d_thresh, g->d_bounds, op}, g->d_counts, cc, wb);
    CK(cudaGetLastError());
    return 0;
}

// out[k] = number of addresses in group k. ms[0] = kernels, ms[1] = D2H.
int cchgpu_addr_group_counts(CchGpu* g, int op, const uint16_t* t, uint32_t groups, const uint32_t* bounds, uint32_t* out,
                             float* ms) {
    int rc = group_head_begin(g, op, t, groups, bounds);
    if (rc) return rc;
    CK(cudaEventRecord(g->ev[4], g->stream));
    if ((rc = group_counts_impl(g, op, groups, false))) return rc;
    CK(cudaEventRecord(g->ev[5], g->stream));
    CK(cudaMemcpyAsync(g->h_counts, g->d_counts, groups * 4, cudaMemcpyDeviceToHost, g->stream));
    CK(cudaEventRecord(g->ev[0], g->stream));
    CK(cudaEventSynchronize(g->ev[0]));
    std::memcpy(out, g->h_counts, groups * 4);
    if (ms) {
        CK(cudaEventElapsedTime(&ms[0], g->ev[4], g->ev[5]));
        CK(cudaEventElapsedTime(&ms[1], g->ev[5], g->ev[0]));
    }
    return 0;
}

// Group lists (ascending address ids per group), two calls: this one, then cchgpu_lists_finish.
int cchgpu_addr_group_lists_begin(CchGpu* g, int op, const uint16_t* t, uint32_t groups, const uint32_t* bounds,
                                  uint32_t* out_counts, uint64_t* total, float* ms) {
    int rc = group_head_begin(g, op, t, groups, bounds);
    if (rc) return rc;
    CK(cudaEventRecord(g->ev[4], g->stream));
    if ((rc = group_counts_impl(g, op, groups, true))) return rc;
    CK(cudaEventRecord(g->ev[5], g->stream));
    CK(cudaMemcpyAsync(g->h_counts, g->d_counts, groups * 4, cudaMemcpyDeviceToHost, g->stream));
    CK(cudaEventRecord(g->ev[0], g->stream));
    CK(cudaEventSynchronize(g->ev[0]));
    uint64_t sum = 0;
    for (uint32_t k = 0; k < groups; ++k) sum += g->h_counts[k];
    std::memcpy(out_counts, g->h_counts, groups * 4);
    *total = sum;
    if (ms) {
        CK(cudaEventElapsedTime(&ms[0], g->ev[4], g->ev[5]));
        CK(cudaEventElapsedTime(&ms[1], g->ev[5], g->ev[0]));
    }
    g->list_total_pending = sum;
    g->list_addr = true;
    g->list_groups = groups;
    g->list_op = op;
    return 0;
}

// The pinned host buffer cchgpu_lists_finish(g, nullptr, ...) left the last list in.
const uint32_t* cchgpu_list_host(const CchGpu* g) { return g->h_list; }

// Column pieces for the current sweep (count = cols, or 0 to drop them); cleared by cchgpu_sweep.
int cchgpu_set_col_pieces(CchGpu* g, uint32_t count, const ColPiece* p) {
    if (count == 0) {
        g->has_col_pieces = false;
        return 0;
    }
    if (count != g->cols) {
        snprintf(g_err, sizeof g_err, "%u column pieces for %u columns", count, g->cols);
        return -1;
    }
    if (!g->d_col_pieces) {
        int rc = dalloc(g, (uint8_t**)&g->d_col_pieces, (g->max_cols + 1) * sizeof(ColPiece));
        if (rc) return rc;
    }
    std::vector<ColPiece> h(p, p + count);
    if (count % 2) h.push_back(ColPiece{~0u, ~0u, ~0u});  // the padding half
    CK(cudaMemcpyAsync(g->d_col_pieces, h.data(), h.size() * sizeof(ColPiece), cudaMemcpyHostToDevice, g->stream));
    CK(cudaStreamSynchronize(g->stream));
    g->has_col_pieces = true;
    return 0;
}

// Tests only: t_addr[a][c] for a in [first, first + count), out = [count][cols] u16.
int cchgpu_addr_times(CchGpu* g, uint32_t first, uint32_t count, uint16_t* out) {
    if (!g->n_addr || (uint64_t)first + count > g->n_addr) {
        snprintf(g_err, sizeof g_err, "address range %u+%u outside the table (%u)", first, count, g->n_addr);
        return -1;
    }
    const uint32_t P = (g->cols + 1) / 2;
    const size_t words = (size_t)count * P;
    if (words > g->addr_out_cap) {
        if (g->d_addr_out) cudaFree(g->d_addr_out), g->mem_used -= g->addr_out_cap * 4;
        g->d_addr_out = nullptr, g->addr_out_cap = 0;
        int rc = dalloc(g, &g->d_addr_out, words);
        if (rc) return rc;
        g->addr_out_cap = words;
    }
    addr_times<<<(unsigned)((words + 255) / 256), 256, 0, g->stream>>>(addr_rows(g), first, count, g->d_addr_out);
    CK(cudaGetLastError());
    std::vector<uint16_t> tmp(words * 2);
    CK(cudaMemcpyAsync(tmp.data(), g->d_addr_out, words * 4, cudaMemcpyDeviceToHost, g->stream));
    CK(cudaStreamSynchronize(g->stream));
    for (size_t a = 0; a < count; ++a) std::memcpy(out + a * g->cols, tmp.data() + a * 2 * P, g->cols * 2);
    return 0;
}

// Profiling, on the current batch (call after cchgpu_sweep; the sweep is idempotent, so variant 0
// and 3 leave the state valid; 1 and 2 leave it wrong). top variant: 0 = stamped global-memory
// kernel, 1 = no arc phase, 2 = no atomics (see TopVariant); shared-memory resident, stamped:
// 3 = production merge (TopMerge default), 4 = one atomic per arc, 5 = segmented shuffle min. Outputs: cycles[top_levels + 1] = block 0's clock64 at
// each top level boundary; *top_ms = event time of the top kernel; wide_ms[levels - top_levels] =
// event time of each wide level, one launch per level.
int cchgpu_profile(CchGpu* g, int reversed, int variant, long long* cycles, float* top_ms, float* wide_ms) {
    const uint32_t P = (g->cols + 1) / 2, LP = lanes_for(P), top = g->top_levels;
    const uint16_t* w = reversed ? g->d_w_rev : g->d_w_fwd;
    uint16_t* d16 = (uint16_t*)g->d_state;
    long long* d_stamps = nullptr;
    CK(cudaMalloc((void**)&d_stamps, (top + 1) * sizeof(long long)));
    const uint32_t cpb = g->top_cols_per_block, blocks = (g->cols + cpb - 1) / cpb;
    const size_t smem = (size_t)g->max_top_level * cpb * 4;
    auto kern = variant == 1   ? top_kernel<TOP_NO_ARCS, uint16_t>
                : variant == 2 ? top_kernel<TOP_NO_ATOMIC, uint16_t>
                               : top_kernel<TOP_STAMP, uint16_t>;
    if (variant >= 3 && !g->resident_smem) {
        snprintf(g_err, sizeof g_err, "resident top unavailable");
        return -1;
    }
    const uint32_t nw = g->levels - top;
    std::vector<cudaEvent_t> ev(nw + 2);
    for (auto& e : ev) CK(cudaEventCreate(&e));
    CK(cudaEventRecord(ev[0], g->stream));
    if (top > 0 && variant >= 3) {
        auto res = variant == 3   ? top_resident<true>
                   : variant == 4 ? top_resident<true, MERGE_ATOMIC>
                                  : top_resident<true, MERGE_SEGMENT>;
        res<<<g->cols, RES_THREADS, g->resident_smem, g->stream>>>(g->d_level_first, top, g->d_up_first, g->d_top_arc, w,
                                                            d16, P, g->max_top_level, g->d_mask, d_stamps);
        CK(cudaGetLastError());
    } else if (top > 0) {
        kern<<<blocks, 1024, smem, g->stream>>>(g->d_level_first, top, g->d_up_first, g->d_up_head, w,
                                                g->d_top_tail, d16, P, g->cols, cpb, g->d_mask, d_stamps);
        CK(cudaGetLastError());
    }
    CK(cudaEventRecord(ev[1], g->stream));
    const CchGpu::SplitTable* tb = split_for(g, LP);
    const uint32_t skip_t = tb && tb->t ? tb->t : 0xFFFFFFFFu;
    if (!tb) tb = &g->split[0];
    for (uint32_t l = top; l < g->levels; ++l) {
        uint32_t base = g->level_first[l], cnt = g->level_first[l + 1] - base;
        launch_level(base, cnt, LP, tb->splits[l - top], tb->d_heavy, skip_t, g->d_up_first, g->d_up_head, w,
                     g->d_state, P, g->d_mask, g->stream);
        CK(cudaEventRecord(ev[l - top + 2], g->stream));
    }
    CK(cudaGetLastError());
    CK(cudaStreamSynchronize(g->stream));
    CK(cudaEventElapsedTime(top_ms, ev[0], ev[1]));
    for (uint32_t i = 0; i < nw; ++i) CK(cudaEventElapsedTime(&wide_ms[i], ev[i + 1], ev[i + 2]));
    if (top > 0) CK(cudaMemcpy(cycles, d_stamps, (top + 1) * sizeof(long long), cudaMemcpyDeviceToHost));
    for (auto& e : ev) cudaEventDestroy(e);
    cudaFree(d_stamps);
    return 0;
}

}  // extern "C"
