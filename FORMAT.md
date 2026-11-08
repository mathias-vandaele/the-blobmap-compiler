# BlobMap model format 2.0.0

This is the byte-level contract between `blobc` and every runtime. SPEC.md says what a model
means; this document says only how it is laid out in bytes. Section references such as
"SPEC §7" point to SPEC.md. Where this document seems to say more than SPEC about meaning,
SPEC wins and this document is wrong.

- **Format name:** `blobmap-model`. **Version:** `2.0.0` (§12).
- **Status:** approved at milestone 1 (SPEC §13), 2026-11-08.

Contents: §1 conventions · §2 files · §3 manifest · §4 storage units and names · §5 road part ·
§6 re-customization · §7 vocabulary part · §8 meaning of the bytes · §9 hashes ·
§10 validation · §11 golden queries · §12 versioning.

## 1. Conventions

- **Byte order:** little-endian everywhere.
- **Dtypes** (safetensors names): `U8`, `U16`, `U32`, `U64`, `I64`, `F32`. A tensor's dtype
  and shape are part of the contract; a reader refuses a mismatch.
- **`NONE` = `0xFFFFFFFF`** in a `U32` index that may be absent. **`NONE16` = `0xFFFF`** in a
  `U16` taxonomy index.
- **Times** are `U16` seconds with `INF = 65535` (SPEC §7). `a ⊕ b` is
  `min(a + b, 65535)`, computed in at least 17 bits; it is the only addition used below.
- **Master times** are `F32` seconds. A road a metric may not use has `+∞` (`0x7F800000`).
  No master time is NaN or negative.
- **`rnd(x)`**, for a non-negative `f64` x: the nearest integer, ties away from zero (SPEC §7).
- **Layout id:** a `U32` node number within one expert (§4). **Row:** a position in a pack's
  per-node tensors. **Local link index:** a `U32` position in one pack's per-link tensors.
- **Strings:** UTF-8 bytes concatenated in a `U8` tensor `<x>.bytes`, plus `U32` offsets
  `<x>.first` of length n + 1; string i is `bytes[first[i] .. first[i+1]]`.
- **CSR:** `<x>_first` of length n + 1, ascending, `<x>_first[0] = 0`; item i owns the range
  `<x>_first[i] .. <x>_first[i+1]` of the tensors that `<x>_first` indexes.
- **JSON files** are UTF-8, keys sorted, 2-space indentation, one trailing newline, integers
  without exponent. The same input gives the same bytes.
- **Hashes:** SHA-256, written as 64 lower-case hex characters. Inside another hash, a hash is
  its 32 raw bytes. `‖` is concatenation; a string is its UTF-8 bytes; `0x00` is one zero byte.

## 2. Files

A model is one directory (SPEC §8):

```
manifest.json                       §3
model.safetensors.index.json        §2.3
model-00001-of-0000N.safetensors    the shards, numbered from 1, five digits each
LICENSES                            UTF-8 text generated from the inputs (SPEC §8, §12)
```

Nothing else is in the directory (SPEC §8).

### 2.1 Shards

Each shard is a standard safetensors file:
- 8 bytes: header length H as `U64`;
- H bytes: the JSON header, padded with spaces (`0x20`) to a multiple of 8;
- the tensor data.

The header holds:
- `__metadata__` = `{"format":"blobmap-shard","format_version":"2.0.0"}`;
- one entry per tensor, `{"dtype","shape","data_offsets"}`;
- no whitespace other than the padding. `__metadata__` comes first, then the tensors in
  data order.

**Data order inside a shard:** by element size, descending (8, 4, 2, 1 bytes), then by
full name in byte order. Tensors are contiguous, with no gaps. Since the data starts at a
multiple of 8, **every tensor is aligned to its element size**. A runtime may map a shard
and use its tensors in place.

### 2.2 Chunked tensors

A logical tensor whose data exceeds the shard target (manifest `build.shard_target_bytes`)
is stored as chunks along dimension 0:
- the chunks are physical tensors named `<name>@0`, `<name>@1`, …;
- every chunk except the last holds `floor(target / row_bytes)` rows (`row_bytes` = bytes per
  index of dimension 0); the last chunk holds the rest;
- other dimensions are unchanged;
- the index (§2.3) lists the chunks.

Everywhere else in this document a tensor name means the logical tensor, whole. Hashes (§9)
cover logical tensors, so chunking never changes a hash.

### 2.3 Index: `model.safetensors.index.json`

```json
{
  "chunks": { "<logical name>": ["<name>@0", "<name>@1"] },
  "metadata": {
    "format": "blobmap-index",
    "format_version": "2.0.0",
    "total_size": 0
  },
  "shards": [ { "bytes": 0, "file": "model-00001-of-00003.safetensors", "sha256": "…" } ],
  "weight_map": { "<physical name>": "model-00001-of-00003.safetensors" }
}
```

- **`weight_map`:** every physical tensor to its shard, as in a Hugging Face sharded
  checkpoint. A reader that ignores `chunks` still finds every byte.
- **`total_size`:** bytes of tensor data over all shards, headers excluded.
- **`shards`:** in file order: file size in bytes and the SHA-256 of the file. These are
  transport checks, not identity (§9).

### 2.4 Filling shards

A **unit** is one of:
- the road tables of one family (§5.1, and `road/active/cross/` for ACTIVE);
- one pack (§4);
- the vocabulary part (§7).

A manifest entry is a run of consecutive pack units. "Never mixes" in SPEC §8 holds because
no shard holds two units.

The compiler assigns tensors to shards in this order:
1. **Units, in order:** each family's road tables, `car` then `active`; then the packs,
   in manifest-entry order and pack order inside an entry; then the vocabulary part.
2. **Within a unit,** tensors in full-name order.
3. **Filling:** a unit's tensors fill shards one after the other. A new shard starts when
   the next tensor (or chunk) would push the current one past `build.shard_target_bytes`,
   or when the next unit starts.

So a shard never mixes units (SPEC §8). Chunks are at most the target, so no shard exceeds
`build.shard_max_bytes`.

## 3. Manifest: `manifest.json`

| key | type | meaning |
| --- | --- | --- |
| `format` | string | `"blobmap-model"` |
| `format_version` | string | `"2.0.0"` |
| `model_id` | string | §9; the identity of the model's content |
| `inputs_hash` | string | §9; the identity of what the answers depend on, which keys golden files (§11) |
| `spec_version` | string | the version of SPEC.md the model was built to, read from its "SPEC version" line (`"2.0.0"`) |
| `compiler` | object | `{name: "blobc", version, commit, dirty}`: commit is the 40-hex git commit, dirty a boolean |
| `inputs` | object | below |
| `config` | object | below |
| `build` | object | `{shard_target_bytes, shard_max_bytes, memory_budget_bytes, group_max_nodes}`: options that shape the output without changing answers |
| `families` | object | family → `{graph, metrics}` |
| `modes` | object | mode → `{family, metric, snap_radius_m}` |
| `experts` | array | the manifest entries, below |
| `parts` | object | `{road: {hash}, vocabulary: {hash, road_hash}}` (§9) |
| `attribution` | array of strings | attribution strings (SPEC §12) |
| `licenses_sha256` | string | SHA-256 of the `LICENSES` file |

- **`inputs`:**
  - `pbf`: `{name, bytes, sha256, timestamp}`. `name` is the file name without its directory;
    `timestamp` is the header's `osmosis_replication_timestamp` (ISO 8601 UTC), or null.
  - `overture`: `{release, files: [{name, sha256}]}`.
  - `taxonomy`: `{version, files: [{name, sha256}]}`.
  - `extent`: `{name, sha256}` of the polygon file passed as `--extent`, or null (SPEC §3).
  - A manifest holds no machine paths.
- **`config`:** `{profiles: {car: {name, sha256}, active: {name, sha256}}, place_rules: {name,
  sha256}}`. Each `sha256` covers the config file's bytes.
- **`families`:**
  - `car`: `{"graph": "edge", "metrics": ["car"]}`;
  - `active`: `{"graph": "node", "metrics": ["bike", "walk"]}`.
  - The metric list is data. A later model may list more metrics (SPEC §4); a runtime ignores
    metrics it does not serve.
- **`modes`:** `car → {family: car, metric: car}`, `bike` and `walk → {family: active, metric:
  <same name>}`. `snap_radius_m` is copied from the profile and is informative.
- **`experts`:** one object per manifest entry. An entry is a single expert, or an expert
  group (SPEC §2):

| key | meaning |
| --- | --- |
| `family` | `car` or `active` |
| `kind` | `"expert"` or `"group"` |
| `experts` | `[first, count]`: the family expert indices it holds (§4) |
| `components` | `count`; 1 for `kind = expert` |
| `nodes`, `links` | layout nodes and links (CCH arcs), summed over its experts |
| `levels` | the longest top-down chain of levels in any of its experts (§5.2) |
| `cells` | number of cell blocks; 0 for a group or an expert built in one piece |
| `packs` | `[first, count]`: its packs (§4) |
| `shards` | its shard file names, in order |
| `hash` | the entry hash (§9) |

  Entries are ordered by family (`car`, then `active`), then by first expert index.

## 4. Storage units and names

- **Expert index:** per family, the experts (SPEC §2, §5) are numbered from 0, by node
  count descending. Ties go to the expert holding the smallest road key `(way, way_seq,
  way_dir)` (§5.3, §5.4). Expert 0 is the largest.
- **Layout ids:** each expert numbers its own nodes 0 .. n−1 (§5.2). Every reference to a
  node is a layout id within the expert at hand.
- **Block:** a contiguous range of one expert's layout ids, with its own levels.
  - An expert built in one piece is one block (kind 0, `whole`).
  - An expert built as cells (SPEC §5) is block 0 (kind 1, `top`: every separator node),
    then one block per cell (kind 2, `cell`), in layout-id order. The top block starts at
    layout id 0.
- **Pack:** one group of safetensors tensors; the unit that tensors are stored in.
  - A `kind = expert` entry has one pack per block.
  - A `kind = group` entry has one pack holding every block of its experts. Each of those
    experts is one whole block.
  - Packs are numbered per family from 0, in entry order.
  - A pack's per-node tensors are its blocks' nodes concatenated in block order; a node's
    position there is its row. Per-link and per-road tensors use local indices within the
    pack.
- **Link:** a CCH arc (SPEC §2 "links"). It is stored in the pack holding its lower node:
  the endpoint with the larger layout id.

**Link references.** No tensor stores a pack number for a link. A reference to a link is a
local index, and the pack it indexes is always the pack of the link's lower node, which
the referencing tensor already names:
- `turn_link` and `road_link` (§5.3, §5.4): the pack of max(tail, head);
- `tri_t` (§6.3): the pack of x, the lower node of t.

To resolve one, find the lower node's block and pack through the block table (§5.1). That
pack's `up_first[row] .. up_first[row + 1]` must contain the index, and `up_head` there must
be the other endpoint. Equivalently, link (v, u) with v > u is at
`up_first[row(v)] + j` in v's pack, where j is the position of u in v's sorted row.

**Which references cross packs.** Only cell → top:
- An expert built in one piece, or one inside a group, has all its links in one pack.
- In an expert built as cells, a cell node's upper neighbours are in its own cell or in the
  top block (§5.2). So a reference from a cell pack names either the cell pack or the top
  pack, never another cell.
- When a cell is customized, its triangles write onto top-block links (§6.3). A runtime that
  customizes cells one at a time keeps the top pack's metric tensors writable until every
  cell is done.

**Limits.** Each pack's row, link, turn (CAR) and road (ACTIVE) counts must each be below
2³², since they are U32 CSR offsets. The build fails if any count reaches 2³². The build
report (SPEC §10) gives, per family, the largest pack and each of these counts for it.
Milestone 6 (SPEC §13) checks these counts for Eurasia's top block.

**Tensor names.** A full name is `/`-separated words, and its last word is the tensor. Metric
tensors are prefixed by the metric name (`car.w`, `bike.up`), so a new metric only adds
tensors (§12).

| prefix | holds |
| --- | --- |
| `road/<family>/experts/` | the expert table (§5.1) |
| `road/<family>/blocks/` | the block table (§5.1) |
| `road/active/cross/` | ACTIVE roads between experts (§5.4) |
| `road/<family>/p<k>/` | pack k (§5.2–§5.4); k in decimal without padding |
| `vocab/` | the vocabulary part (§7) |

## 5. Road part

### 5.1 Family tables

**`road/<family>/experts/`** (E experts):

| tensor | dtype | shape | meaning |
| --- | --- | --- | --- |
| `nodes` | U32 | [E] | layout nodes of the expert |
| `links` | U64 | [E] | links of the expert |
| `levels` | U32 | [E] | its longest top-down chain of levels |
| `trapped` | U8 | [E] | 1 if it is a trapped component (SPEC §5), else 0 |
| `entry` | U32 | [E] | index of its manifest entry in `experts` |
| `block_first` | U32 | [E + 1] | CSR into the block table; its blocks in layout-id order |

**`road/<family>/blocks/`** (B blocks, ordered by expert, then layout id):

| tensor | dtype | shape | meaning |
| --- | --- | --- | --- |
| `expert` | U32 | [B] | owning expert |
| `kind` | U8 | [B] | 0 whole, 1 top, 2 cell |
| `id_first` | U32 | [B] | first layout id of the block |
| `nodes` | U32 | [B] | node count; the block is layout ids `id_first .. id_first + nodes` |
| `pack` | U32 | [B] | pack holding the block |
| `row_first` | U32 | [B] | row of `id_first` in that pack |
| `levels` | U32 | [B] | number of levels L of the block |
| `level_base` | U32 | [B] | where the block's L + 1 level bounds start in the pack's `level_first` |

To find node v of expert e: among the blocks `block_first[e] .. block_first[e+1]`, take the
last one with `id_first ≤ v`. Its row is `row_first + (v − id_first)` in pack `pack`.

### 5.2 Pack: the hierarchy

Pack `road/<family>/p<k>/` with n rows, A links:

| tensor | dtype | shape | meaning |
| --- | --- | --- | --- |
| `level_first` | U32 | [Σ (L_b + 1)] | per block, L_b + 1 ascending layout ids: level l of the block is ids `level_first[base + l] .. level_first[base + l + 1]`; level 0 is the top |
| `up_first` | U32 | [n + 1] | CSR by row over the links whose lower node is that row |
| `up_head` | U32 | [A] | layout id of the link's upper node |

**Invariants:**
- **Order:** for every link, `up_head` < its lower node's layout id. Within a row, `up_head`
  is strictly ascending, so there are no parallel links.
- **Levels:** a block's levels cover its ids exactly, top level first. Both ends of a link
  are never in the same level of the same block.
  - Inside a block, an upper neighbour is in a smaller level.
  - An upper neighbour in another block is in the top block (kind 1). Cells share no links.
  - A pass over layout ids in increasing order therefore visits every upper neighbour of v
    before v.
- **Chordality:** if x and y are upper neighbours of v with y < x, the link (x, y) exists.
- **Elimination tree:** the parent of v is its largest upper neighbour (the last entry of
  its row), or none if v has no upper neighbour.
- **Free choices of the compiler:** the order of nodes inside a level, and the numbering of
  cells. They are fixed per build (SPEC §10, determinism), but a runtime must not rely on
  them.

**Metric weights.** For each metric m, two `U16 [A]` tensors:
- `<m>.up`: the weight lower → upper;
- `<m>.down`: the weight upper → lower;
- `INF` where there is no path.

They are the exact output of §6 for that metric.

### 5.3 CAR packs: segments and turns

CAR layout nodes are directed road segments (SPEC §4); its roads are the segments. Per row:

| tensor | dtype | shape | meaning |
| --- | --- | --- | --- |
| `car.time_s` | F32 | [n] | master time of the segment (SPEC §7), including its delays (SPEC §4, §7) |
| `car.w` | U16 | [n] | integer weight of the segment, §6.1 |
| `way` | I64 | [n] | OSM way id |
| `way_dir` | U8 | [n] | 0 along the way's node order, 1 against it |
| `way_seq` | U32 | [n] | position of the piece on the way, 0-based in node order |
| `kind` | U8 | [n] | 0 road; 1 ferry: reserved, never written in 2.0 (SPEC §4); 2 via-way copy (below) |
| `copy_of` | U32 | [n] | for a copy: layout id of its original, if that is in the same expert; else `NONE`. `NONE` for every original |
| `copy_first` | U32 | [n + 1] | CSR by row into `copies` |
| `copies` | U32 | [C] | for an original: layout ids of its copies in the same expert, ascending; empty for a copy |

Turns, by the row of the segment they leave (T turns):

| tensor | dtype | shape | meaning |
| --- | --- | --- | --- |
| `turn_first` | U32 | [n + 1] | CSR by row |
| `turn_head` | U32 | [T] | layout id of the segment the turn enters |
| `car.turn_cost` | U16 | [T] | turn cost in whole seconds (SPEC §7). Never `INF` in 2.0, because a turn forbidden to car is absent (SPEC §4); `INF` is kept for a later metric of the family |
| `turn_link` | U32 | [T] | local index of link {tail, head}, in the pack holding max(tail, head); `NONE` if tail = head |

- Turns between different experts are not stored; no route within an expert uses them.
- Inside a row, turns are ordered by `turn_head` ascending. There is at most one turn per
  pair of segments.

**Turn restrictions** (SPEC §4) are not stored as such. They are compiled into the turn
graph:
- **Via a node:** a forbidden turn is absent.
- **Via one or more ways:** the via segments are copied (the via-way construction).
  - `only_*` restrictions are first rewritten as the equivalent set of forbidden
    sequences.
  - Each forbidden sequence [a₀, a₁, …, a_k] is a sequence of segments, k ≥ 2.
  - **States:** every original segment is a state. Every proper prefix [a₀ … a_j], with
    1 ≤ j < k, of a forbidden sequence is a state too: a copy of a_j, stored as a row with
    `kind = 2`.
  - **Turns out of a state** P whose last segment is x: for each turn x → b, the turn goes
    to the state of the longest suffix of P + [b] that is a state (an Aho–Corasick
    automaton over segments). It is absent if some suffix of P + [b] is a whole forbidden
    sequence. Its cost is the turn cost of x → b.
- **Rows of a copy:** a copy has its original's `way`, `way_dir`, `way_seq`, `car.time_s`
  and `car.w`.
  - A copy is a layout node like any other, with links and turns of its own.
  - It is not a road of its own: it is not a member of any rounding group (§6.1), and no ID
    snaps to it (§7.2).
  - Heads and seeds take copies into account as §8 says.
- **One view.** The CAR metric is stored in its *start* view only: turn a → b weighs
  `w[a] ⊕ c(a, b)`, so a label is the time at a segment's start (§6.2, §8).
  - Both query directions are exact with it and these turns, which are indexed by the
    segment they leave (§8).
  - An *end* view (`c ⊕ w[b]`), with turns indexed by the segment they enter, would make
    the heads of `to` queries one lookup instead of one per exit turn. It costs 4 bytes per
    link, plus 6 bytes per turn and 4 per segment for the entering index. A later minor
    version may add it as optional tensors (§12).

### 5.4 ACTIVE packs: roads

ACTIVE layout nodes are intersections; its roads are directed road pieces (SPEC §4). One
union graph serves every metric, and a road a metric may not use has weight `INF` for it.
Roads are stored by the row of their tail (R roads):

| tensor | dtype | shape | meaning |
| --- | --- | --- | --- |
| `road_first` | U32 | [n + 1] | CSR by row |
| `road_head` | U32 | [R] | layout id of the head |
| `road_link` | U32 | [R] | local index of link {tail, head}, in the pack holding max(tail, head); `NONE` if tail = head |
| `way`, `way_dir`, `way_seq`, `kind` | as §5.3 | [R] | as §5.3 |
| `<m>.time_s` | F32 | [R] | master time for metric m, with the node costs charged to the road (SPEC §4, §7); `+∞` where m may not use it |
| `<m>.w` | U16 | [R] | integer weight, §6.1; `INF` where m may not use it |

- Inside a row, roads are ordered by (`road_head`, `way`, `way_seq`). Parallel roads are kept:
  each is a separate piece that IDs can stand on.
- **Roads between experts:** these roads have ends in different ACTIVE experts and never lie
  on a route. They are kept only because a way's rounding (§6.1) spans all its pieces. They
  live in `road/active/cross/` as `way`, `way_dir`, `way_seq`, `kind`, `<m>.time_s`, `<m>.w`,
  each [R_x], sorted by (`way`, `way_dir`, `way_seq`).
- **CAR needs no cross table**, including for one-way roads into trapped components:
  - A CAR road is a segment, which is a node of the turn graph. Every node belongs to
    exactly one strongly connected component, so every segment is stored in some expert.
  - A segment that cannot be returned to is a component of its own: a trapped expert.
  - What crosses experts in CAR is a turn. Turns carry no master time and take no part in
    rounding (§6.1), so dropping them loses nothing that re-customization needs.

## 6. Re-customization

From the bytes alone, a runtime can recompute every metric tensor from new master times
(SPEC §8). The compiler computes them this way, and the conformance suite checks that this
procedure reproduces the stored bytes.

### 6.1 Integer road weights

SPEC §7 defines the rounding groups and the rule. In bytes:
- **Members:** a group of metric m is all roads of one family with the same (`way`,
  `way_dir`) and a finite `<m>.time_s`, over every expert and the cross table (§5.4). Every
  member of every group is stored, so re-rounding needs no data outside the model.
  - CAR rows with `kind = 2` (via-way copies, §5.3) are not members.
  - A road whose new time is `+∞` leaves its group; one whose time becomes finite joins it.
- **Copies:** after rounding, each CAR copy takes the `car.time_s` and `car.w` of its
  original. The original is the member with the same (`way`, `way_dir`, `way_seq`). When
  `copy_of = NONE`, the original is in another expert, and re-rounding finds it by this
  way key, since rounding groups span the family.
- **Travel order:** `way_seq` ascending if `way_dir = 0`, descending if 1.

For each group, the exact integer computation:
1. Let tᵢ = the `f64` of each `time_s` (exact widening). Sum S = Σ tᵢ in `f64`, in travel
   order. T = rnd(S).
2. Let fᵢ = ⌊tᵢ⌋ and extra = T − Σ fᵢ (0 ≤ extra ≤ group size).
3. Sort the roads by tᵢ − fᵢ descending, ties in travel order. The first `extra` roads get
   fᵢ + 1, the others fᵢ.
4. `w = min(that, 65535)`. A road with `time_s = +∞` has `w = INF`.

Turn costs (`car.turn_cost`) are stored integers (SPEC §7). Re-customization takes them as
given.

### 6.2 Link initialisation

1. Every link starts at `INF` in both directions, for each metric.
2. **CAR.** For every stored turn a → b with `turn_link ≠ NONE` and `car.turn_cost = c ≠
   INF`:
   - its weight is `car.w[a] ⊕ c`;
   - if a > b (a is the lower node), it relaxes `up` (min); else `down`.
3. **ACTIVE.** For every road t → h with `road_link ≠ NONE`, for each metric m, its
   `<m>.w` relaxes `up` if t > h, else `down`.

### 6.3 Triangles and output

Let a node v have upper neighbours h₀ < h₁ < … (its row of `up_head`), with links ℓ₀, ℓ₁, ….
**The triangles of v** are the pairs (i, j) with j < i, enumerated by i ascending, then j
ascending:
- x = hᵢ, y = hⱼ (y is further up);
- a = ℓᵢ, b = ℓⱼ;
- t = the link (x, y), stored in the pack holding x (§4, link references). It exists by
  chordality and is found by walking x's sorted row. When v is in a cell and x is in the
  top block, t is a top-block link.

**Order:**
- blocks of an expert from the last to the first (cells before the top, SPEC §5);
- inside a block, levels from the bottom (largest l) to the top;
- inside a level, any order.

For each triangle:
- `up[t] = min(up[t], down[a] ⊕ up[b])`;
- `down[t] = min(down[t], down[b] ⊕ up[a])`.

Why the result does not depend on order:
- inside a level, no node writes a link that another node of the same level reads;
- writes are `min`;
- cells write only their own links and links of the top block, and the top block is
  processed after every cell.

Saturating at 65535 throughout gives the same bytes as working in wider integers and
clamping at the end. The result is the metric tensors of §5.2.

**Optional triangle tensors** (SPEC §8), per pack. The compiler does not write them by
default:

| tensor | dtype | shape | meaning |
| --- | --- | --- | --- |
| `tri_first` | U64 | [n + 1] | CSR by row: the triangles of each node, in the enumeration order above |
| `tri_a`, `tri_b` | U32 | [K] | local indices of a and b, in this pack |
| `tri_t` | U32 | [K] | local index of t, in the pack holding x |

If present, they hold exactly the enumeration above. A runtime may use them, or enumerate
from `up_head`; both give the same bytes.

## 7. Vocabulary part

The vocabulary covers every ID of the model, across families and experts. Which Overture
IDs are in it depends on the extent (SPEC §3).

### 7.1 IDs

| tensor | dtype | shape | meaning |
| --- | --- | --- | --- |
| `vocab/ids/address` | U8 | [N_a, 16] | Overture address IDs, strictly ascending |
| `vocab/ids/place` | U8 | [N_p, 16] | Overture place IDs, kept and filtered (§7.4), strictly ascending |
| `vocab/ids/alias` | U8 | [N_x, 16] | aliases: IDs of places merged as duplicates (SPEC §3, §6), strictly ascending |
| `vocab/ids/alias_to` | U32 | [N_x] | internal index of the kept place each one was merged into |
| `vocab/road_hash` | U8 | [32] | the road part hash this vocabulary was built on (§9) |

- **ID bytes:** an Overture ID is a UUID in its canonical text form: 36 characters,
  lower-case hex with dashes. It is stored as its 16 bytes, in text order. Byte order then
  equals text order, which is the order of the smallest-ID rule (SPEC §7). Every ID is a
  UUID (SPEC §3).
- **Internal index i:** addresses are 0 .. N_a, places N_a .. N_a + N_p. N = N_a + N_p.
  - No ID appears twice across the three tables.
  - An alias has no internal index of its own; `alias_to` always names a kept place.
- **Lookup:**
  1. Binary-search `address`, then `place`; a hit gives i.
  2. Otherwise binary-search `alias`; a hit resolves to `alias_to` (SPEC §6, aliases).
  3. Otherwise the ID is absent from the model: `UnknownId` (SPEC §6).

### 7.2 Entries per mode

Each mode has its own entries (SPEC §6). In bytes, "the mode's graph" (SPEC §6) is:
- for car, the pieces whose segments are CAR layout nodes;
- for an ACTIVE mode m, the pieces with at least one road of finite `<m>.w`.

**CAR, `vocab/car/`:**

| tensor | dtype | shape | meaning |
| --- | --- | --- | --- |
| `car.expert` | U32 | [N] | CAR expert of the ID, `NONE` if not served for car |
| `car.seg` | U32 | [N, 2] | layout ids of the piece's two directed segments, [A → B, B → A]; always original segments (`kind ≠ 2`, §5.3); `NONE` for a direction absent from the expert |
| `car.off` | U16 | [N, 2, 2] | for slot k: `[k, 0]` = time from the start of `seg[k]` to the ID, `[k, 1]` = time from the ID to the end of `seg[k]` |

**ACTIVE, `vocab/active/`,** for each mode m (`bike`, `walk`):

| tensor | dtype | shape | meaning |
| --- | --- | --- | --- |
| `<m>.expert` | U32 | [N] | ACTIVE expert of the ID for mode m, `NONE` if not served for m |
| `<m>.node` | U32 | [N, 2] | layout ids [U, V] of the piece's ends; U = V for a loop |
| `<m>.road` | U32 | [N] | local index, in the pack holding U, of the road U → V of the piece; it identifies the piece for the same-piece rule. U and V are ordered so that this road exists |
| `<m>.off` | U16 | [N, 4] | for the mode's metric: [U → ID, ID → U, V → ID, ID → V] |

**Offsets** (SPEC §7). For each directed road r of the piece, with w = `<metric>.w` of the
mode's metric, the tensors hold the two parts of r around the ID:
- **the part before:** `car.off[k, 0]` for r = `seg[k]`, or U → ID / V → ID;
- **the part after:** `car.off[k, 1]`, or ID → V / ID → U.

The part before is rnd(w × g). g is the fraction of the piece's length from r's tail to
the ID's snap point, so f from A (or U) for r = A → B (or U → V), and 1 − f the other way.

**Invariants:**
- part before + part after = w exactly.
- An ID standing at a node (SPEC §6) has g = 0 or 1, so its offsets at that node are 0.
- Where w = `INF` (a direction the mode may not use, SPEC §6, or a `NONE` slot), both parts
  are `INF`. For ACTIVE, that is U → ID and ID → V, or V → ID and ID → U.

**Other invariants:**
- An unserved ID has `expert = NONE`, `NONE` nodes and `INF` offsets for that mode.
- A served ID's nodes belong to its expert for the mode, and that expert has `trapped = 0`.
- At least one direction of a served ID's piece is finite for the mode.
- **CAR:** if the piece's two segments are in different experts, the ID takes the one with
  the smaller expert index, and the other slot is `NONE`.

### 7.3 Reason codes

Per mode, sorted by internal index (K entries):

| tensor | dtype | shape | meaning |
| --- | --- | --- | --- |
| `vocab/<family>/<mode>.reason_id` | U32 | [K] | internal index of an ID not served for the mode |
| `vocab/<family>/<mode>.reason_code` | U8 | [K] | the code, below |
| `vocab/<family>/<mode>.reason_m` | F32 | [K] | a distance in metres, below |

Codes, with their meaning in SPEC §6:

| code | name | `reason_m` |
| --- | --- | --- |
| 1 | `NoEligibleRoad` | distance to the nearest piece of the family's graph |
| 2 | `NoRoad` | `+∞` |
| 3 | `TrappedComponent` | distance to the nearest eligible piece (all are in trapped components) |
| 4 | `FilteredPlace` | never in these lists; stored in `vocab/places/status` (§7.4) |

- An ID is served for a mode exactly when its `<mode>.expert ≠ NONE`.
- An unserved ID that is not a filtered place has exactly one entry in the mode's list.
- Code 0 and codes 5–255 are not used in 2.0. `UnknownId` and `DifferentComponent` are
  never stored.

### 7.4 Places and categories

Per place p = i − N_a:

| tensor | dtype | shape | meaning |
| --- | --- | --- | --- |
| `vocab/places/status` | U8 | [N_p] | 0 kept; 4 `FilteredPlace` (SPEC §3, §6) |
| `vocab/places/primary` | U16 | [N_p] | taxonomy node of the primary category; `NONE16` if the place has none, or if its key is absent from the taxonomy |
| `vocab/places/basic` | U16 | [N_p] | taxonomy node of the basic category; `NONE16` if Overture gives none, or if its key is absent from the taxonomy |
| `vocab/places/alt_first` | U32 | [N_p + 1] | CSR into `alt` |
| `vocab/places/alt` | U16 | [·] | alternate categories, ascending, without the primary; keys absent from the taxonomy are left out |

- **A filtered place** has `NONE` entries for every mode and no entry in any reason list.
  It never matches a category; its category tensors are still filled.
- **A place with `primary = NONE16`** never matches a category, in either namespace. It
  is still served as a location. Such places, and keys absent from the taxonomy, are
  counted in the data report (SPEC §3); they do not fail the build.

The taxonomy of the pinned release (T nodes, T < 65535):

| tensor | dtype | shape | meaning |
| --- | --- | --- | --- |
| `vocab/taxonomy/name.bytes`, `name.first` | U8, U32 | strings [T] | the category key (`primary` in `taxonomy.csv`), sorted in byte order; a node's index is its position |
| `vocab/taxonomy/parent` | U16 | [T] | parent node, `NONE16` at a root |
| `vocab/taxonomy/basic` | U16 | [T] | its basic category, `NONE16` if none |
| `vocab/taxonomy/is_basic` | U8 | [T] | 1 for a basic category |
| `vocab/taxonomy/redirect.bytes`, `redirect.first` | U8, U32 | strings [D] | retired category keys that Overture redirects, sorted |
| `vocab/taxonomy/redirect_to` | U16 | [D] | the node each one redirects to |

**Basic categories are taxonomy nodes.** `basic` indexes the same node table, so every
basic category must be a taxonomy node with the same key, and the build fails otherwise. In
release 2026-10-21.0 this holds: 298 basic categories, all of them among the 2,302 nodes.

**Evaluating a category query** (SPEC §6) on these tensors:
1. **Resolve the key:** a key found in `redirect` is replaced by the key of `redirect_to`.
   Then it is looked up in `name`. A basic-category query also needs `is_basic = 1`.
2. **A taxonomy query for node X** matches a kept place p with `primary[p] ≠ NONE16` when
   `primary[p]` or an `alt` of p reaches X by following `parent` zero or more times.
3. **A basic-category query for B** matches a kept place p with `primary[p] ≠ NONE16` when
   `basic[p] = B`.

No coordinates and no place names are stored (SPEC §7, §8).

## 8. Meaning of the bytes

This section restates no rule of SPEC §9. It says which bytes a runtime combines to get the
quantities those rules are written in. All sums use ⊕. All terms are non-negative, so any
true time below the cap is reached without saturating, and is exact.

**Road graph distances.** For an expert and metric m, d(x, y) is the shortest time from node
x to node y:
- **ACTIVE:** over its roads, with weights `<m>.w`.
- **CAR:** over its turns, where turn a → b weighs `car.w[a] ⊕ c(a, b)`, c being
  `car.turn_cost`. d is the time from the start of segment x to the start of segment y.

**Hierarchy property.** For any x, y of one expert,
d(x, y) = min over z of (up(x, z) ⊕ down(z, y)), where:
- up(x, z) is the shortest path from x to z using only lower → upper links and `up` weights;
- down(z, y) is the shortest path from z to y using only upper → lower links and `down`
  weights;
- z ranges over the common ancestors of x and y.

This is what the metric tensors guarantee. Any CCH sweep that relies only on it is exact.

**Time between two IDs.** Take o ≠ d, both served for the mode and with the same
`<mode>.expert`. Let `off` be the mode's offsets.

- **ACTIVE:** the route term is
  min over x ∈ {U_o, V_o} and y ∈ {U_d, V_d} of (o → x) ⊕ d(x, y) ⊕ (y → d).
- **CAR:** for a segment y, let arr(y) be y together with its `copies` (§5.3). A trip
  leaves an ID from the original segment, since no restricted prefix has been driven yet.
  It may reach an ID on the original or on any copy. The route term is
  min over slots k of o, turns s → b out of s = `seg_o[k]`, slots k′ of d, and
  y ∈ arr(`seg_d[k′]`) of `off_o[k, 1] ⊕ c(s, b) ⊕ d(b, y) ⊕ off_d[k′, 0]`.
  - This includes y = s or a copy of s, the way around the block.
  - Both directions of SPEC §9 evaluate this sum with the stored turns, which are indexed by
    the segment they leave. Every term is added, never subtracted.
  - **`from` (sources → IDs):**
    - seed each b reached by a turn out of a source's segment s with
      `off_o[k, 1] ⊕ c(s, b)`;
    - sweep forward;
    - the head reads, per slot of each ID, the min over y ∈ arr(`seg_d[k′]`) of
      `label[y] ⊕ off_d[k′, 0]`.
  - **`to` (IDs → sources):**
    - seed each y ∈ arr(`seg_o[k′]`) of each source with `off_o[k′, 0]`;
    - sweep the reversed hierarchy (`down` weights going up, `up` going down);
    - the head of an ID reads, for each slot k and each turn s → b out of `seg[k]`,
      `off[k, 1] ⊕ c(s, b) ⊕ label[b]`.
- **Direct drive** (SPEC §9 "Same road piece"): o and d share a directed road r of the same
  piece, in the same slot. For CAR that is `seg_o[k] = seg_d[k]`. For ACTIVE it is the same
  `road` and nodes, in either direction.
  - The direct term exists only when r's weight is finite for the mode.
  - Let p_o and p_d be the parts before the ID (§7.2) on r.
  - When p_d ≥ p_o, the direct term is p_d − p_o; equal parts give 0 s (SPEC §9).
- **The answer** is the min of the route term and the direct term, then SPEC §7 and §9
  (cap, `BEYOND`, same ID, `DifferentComponent`).

This is exactly a shortest path in the road graph with o and d inserted at their offsets.
Golden answers are computed that way (§11).

The delays charged to a road (SPEC §4) are spread over its offsets by length, like the
rest of its time. This is intended.

## 9. Hashes

- **Tensor hash** = sha256(`"blobmap-tensor-2"` ‖ 0x00 ‖ full logical name ‖ 0x00 ‖ dtype
  name ‖ 0x00 ‖ rank as U8 ‖ each dimension as U64 ‖ the data bytes).
- **Entry hash** of a manifest entry = sha256(`"blobmap-entry-2"` ‖ 0x00 ‖ the tensor hashes
  of all its packs, in full-name order).
- **Road part hash** = sha256(`"blobmap-road-2"` ‖ 0x00 ‖ the tensor hashes of every
  `road/<family>/experts/`, `road/<family>/blocks/` and `road/active/cross/` tensor, in
  full-name order ‖ the entry hashes, in manifest order).
- **Vocabulary part hash** = sha256(`"blobmap-vocab-2"` ‖ 0x00 ‖ the tensor hashes of every
  `vocab/` tensor except `vocab/road_hash`, in full-name order ‖ the raw `vocab/road_hash`).
- **`model_id`** = hex sha256(`"blobmap-model-2"` ‖ 0x00 ‖ road part hash ‖ vocabulary part
  hash). It changes with layout: the same inputs built as cells or in one piece give
  different model IDs.
- **`inputs_hash`** = hex sha256 of:
  - `"blobmap-inputs-2"` ‖ 0x00;
  - the `.pbf`'s sha256;
  - the Overture `release` string ‖ 0x00;
  - the sha256 of the car profile, of the active profile, and of the place rules, in that
    order (§3 `config`);
  - the extent polygon's sha256, or 32 zero bytes without `--extent`;
  - `spec_version` ‖ 0x00.

  Sha256 values are taken as 32 raw bytes. Every part is a manifest field, so a reader can
  recompute it. It leaves out the compiler version and the build options. Answers are
  defined by SPEC and the inputs alone (SPEC §9), so two models with the same inputs hash
  must give the same answers.
- **The link** (SPEC §8): `vocab/road_hash` must equal `parts.road.hash`, and
  `parts.vocabulary.road_hash` repeats it.
  - A reader refuses a model where these three differ.
  - A new Overture release changes the vocabulary hash and `model_id`, never the road hash.

Shard SHA-256s (§2.3) depend on shard layout and are for transport only.

## 10. Validation

A reader **must** refuse:
- another `format`, or another major version (§12);
- a tensor missing, or with a dtype or shape other than this document's;
- chunks that do not concatenate to the logical shape;
- a road-hash link mismatch (§9).

A reader **must ignore** manifest keys it does not know (SPEC §8).

A reader **should** check, at least per pack it loads:
- the CSR shapes, and that `up_head` is ascending and below the node's id (§5.2);
- the level bounds;
- that every layout id and local index is in range.

The conformance suite (SPEC §11) checks every invariant of §5–§8 on every model, and
mutated models must fail it.

## 11. Golden queries

Per tier, a directory with `golden.json` (the header) and `queries.jsonl` (one query per
line).

**`golden.json`:** `{format: "blobmap-golden", format_version: "2.0.0", tier, inputs_hash,
queries_sha256}`.
- A replay refuses a model whose `inputs_hash` (§9) differs, and accepts any model with the
  same one: whatever its layout, cells, shards or compiler version.
- Answers are layout-independent. They name IDs only, and the only order they use is the
  internal-index order of the vocabulary (§7.1), which comes from the IDs.

**Each line of `queries.jsonl`** is `{"op", "args", "answer"}`:
- **IDs** are canonical UUID strings.
- **A time** is an integer below 65535, or the string `"BEYOND"`.
- **A reason** is `{"reason": "UnknownId" | "DifferentComponent" | "NoEligibleRoad" |
  "NoRoad" | "TrappedComponent" | "FilteredPlace", "id": <the ID it concerns>}`.
- **A category** in `nearest` and `chained` is `{"category": <key>}` (a taxonomy node) or
  `{"basic": <key>}` (SPEC §6).
- **`expr`** is the JSON tree of SPEC §9, as is. `kind` defaults to `"addresses"`.
- **`top`:** the weighted answer, best first, at most k entries. `score` is a decimal
  string, because a `u64` may exceed the integers JSON represents exactly.
- **An alias** may appear in `args`. Answers name kept IDs only (§7.1).

| op | args | answer |
| --- | --- | --- |
| `travel_time` | `{a, b, mode}` | `{time}` or a reason |
| `isochrone` | `{sources: [{id, start?}], t, mode, dir}` | `{count, ids_sha256}` |
| `nearest` | `{category, id, mode, dir}` | `{time, place}`, or `{time: "BEYOND", place: null}` |
| `column` | `{sources: [{id, start?}], mode, dir}` | `{times_sha256, beyond}` |
| `reachable_from_all`, `reachable_from_any` | `{sources, t, mode}` | `{count, ids_sha256}` |
| `multi_criteria`, boolean form | `{expr, kind?}` | `{count, ids_sha256}` |
| `multi_criteria`, weighted form | `{expr, kind?, k}` | `{top: [{id, score}]}` |
| `chained` | `{id, transfer_category, penalty, mode_out: "car", mode_in: "walk", dir}` | `{time, place}` |

- **`ids_sha256`:** the SHA-256 over the 16-byte IDs of the answer, ascending.
- **`times_sha256`:** over every ID of the model in internal-index order: the U16 time
  (INF for `BEYOND`, unreachable or unserved).
- **`beyond`:** the number of INF entries in that column.
- **`dir`:** `"from"` or `"to"`. `start` is in seconds, default 0.
- Small tiers may add `"ids"` (the full list) next to a hash.

## 12. Versioning

`format_version` is `MAJOR.MINOR.PATCH` (SPEC §8):
- **PATCH:** this document's wording only; same bytes.
- **MINOR:** adds optional tensors, or optional manifest keys, that a reader of a lower
  minor ignores (SPEC §8). Examples: the triangle tensors of §6.3, or a CAR end view
  (§5.3).
  - Readers ignore unknown manifest keys (§10).
  - A reader of 2.y reads any 2.x. When x < y, the tensors and keys added after 2.x are
    absent.
- **MAJOR:** anything else. That includes a changed name, dtype, shape, meaning or
  invariant; a removed tensor; a required key; and any new key in the index or in shard
  headers. A reader refuses any major but its own.

Not format changes:
- **A new metric** in a family (SPEC §4): new `<m>.*` tensors and a longer `metrics` list,
  which readers of 2.0 ignore.
- **Ferries** (SPEC §4): roads with `kind = 1`, a profile change; runtimes route on them
  like any road.

The compiler promises byte-identical output only for the same compiler version (SPEC §10).
