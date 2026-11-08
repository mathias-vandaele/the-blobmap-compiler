# BlobMap World Model — Compiler & Format Spec (v2 draft)

Nov 8, 2026 · @Mathias Vandaele

## 1. Purpose and scope

`blobc` compiles one OpenStreetMap `.pbf` and one pinned Overture release into one model: a manifest plus sharded safetensors that any runtime can execute to answer exact travel-time queries by car, bike and walk.

- **In scope:** the compiler, this spec and the tensor-level format spec (FORMAT.md), a CPU reference runtime, the conformance suite.
- **Out of scope:** GPU runtimes (CUDA, Metal, wgpu), servers, the BlobMap API, precomputed category fields, runtime benchmarks.
- **Consumer:** BlobMap servers only for now. The spec is written so it could be published.
- **Scale:** designed for the world. The compiler compiles exactly the `.pbf` it is given; the first target is France.
- **The repository stays lean:** one workspace, small crates, no experiments.

## 2. Concepts

No boundary in the model is drawn by a human: experts come from the connectivity of the road network, cells from the compiler's own nested dissection. Country borders, polygons and file splits never change a result.

| Term | Definition | Decided by |
| --- | --- | --- |
| Model | One versioned artifact: a manifest, an index and shards. One build produces one model. | the build |
| Mode family | CAR (car) or ACTIVE (bike and walk, one shared graph). | this spec |
| Metric | One set of travel times for one mode: car, bike or walk. | this spec (3 metrics) |
| Expert | One strongly connected component of one mode family's graph, with one hierarchy (CCH). Every trip inside an expert is exact; trips between experts are unreachable. | the compiler, from the data |
| Expert group | Small experts packed together into shared shards. Each component inside stays its own expert for routing. | the compiler, by size |
| Cell | A build unit: a piece of an expert, chosen by nested dissection so the build fits in memory. Never a routing boundary. | the compiler, from the memory budget |
| Road part | The graph, hierarchy, metrics and re-customization data of an expert. Changes with OSM. | the build |
| Vocabulary part | Overture address and place IDs mapped to positions in each expert, plus place categories. Changes with each Overture release, and depends on the road part's hash. | the build |
| Router | The mapping from an input ID and a mode to its expert and entries. Part of the vocabulary. | the build |

## 3. Inputs and data rules

Roads come only from the `.pbf`; addresses and places come only from Overture. Every routable road in the `.pbf` is compiled and served; nothing is dropped.

- **Roads:** all ways routable by at least one mode, from the whole `.pbf` passed in. The extent of the model is the extent of the `.pbf`.
- **Overture:** one pinned release per model, recorded in the manifest. The compiler reads the addresses and places that fall inside the `.pbf`'s extent.
- **Addresses:** every Overture address in the extent. An address is served for a mode only if it snaps (§6).
- **Places:** Overture places, filtered by the rules below, with their categories (§6).
- **No other source:** no OSM POIs, no Overture roads, no BAN.

Place rules (defaults, kept in versioned config; the config hash goes into the manifest):

| Rule | Default |
| --- | --- |
| Closed places | dropped (Overture operating status) |
| Confidence, categories under `node:health_care` | ≥ 0.5 |
| Confidence, all other categories | ≥ 0.6 |
| Duplicates | merged when same basic category, within 30 m, compatible names; the highest confidence is kept, ties to the smallest ID |
| Categories | Overture's taxonomy of the pinned release: basic category and taxonomy hierarchy, with Overture's redirect rules |

Each build writes a data report: place counts per country and basic category against the previous build, flagging any change above 20 %, plus the number of trapped components (§5), which usually point at OSM errors. Provider score distributions drift between releases, so the rules are reviewed against this report every month.

## 4. Modes and profiles

Three metrics exist in this version, car, bike and walk, in two mode families. Profiles are config files; their hashes go into the manifest. Travel times are static: no time of day.

|  | CAR | ACTIVE |
| --- | --- | --- |
| Modes | car | bike, walk (one graph, two metrics) |
| Graph | edge-based: nodes are directed road segments, arcs are allowed turns | node-based: nodes are intersections, arcs are road pieces |
| Network | ways usable by cars | union of the ways usable by bike or on foot |
| Speeds | OSRM v6 car profile: speeds by road class, `maxspeed`, access and one-way rules | bike 15 km/h, walk 5 km/h, flat |
| Turns | OSRM v6-style turn penalties (angle, u-turns) | none |
| Restrictions | all OSM turn restrictions, `no_*` and `only_*`, via a node and via one or more ways | access per mode; steps for walk only; bike contraflow (`oneway:bicycle=no`) |
| Delays | traffic signals, stops and give-ways, as in OSRM v6 car | traffic signals and crossings, as node costs |
| Forbidden for a mode | arc absent | arc weight ∞ for that metric |
| Reference accuracy | OSRM v6 car | OSRM bike and foot (bike differs by its missing turn costs, about 5 %, accepted) |

- **Ferries and car shuttles:** excluded in this version. The format reserves a `ferry` link type (fixed crossing time plus an average wait) so they can be added later as a profile change.
- **Format room:** an expert family may carry more than its current metrics later (e-bike, slow walker), without a format change. This version defines exactly three.

## 5. Experts and cells

The compiler splits the network into experts by connectivity and splits large experts into cells by memory, then builds one exact hierarchy per expert. No user input decides either.

1. **Build each mode family's graph** over the whole `.pbf` (CAR: the turn graph; ACTIVE: the bike and walk union).
2. **Compute strongly connected components.** Each component is an expert: Eurasia plus Africa is one CAR expert; Britain is its own while ferries and shuttles are excluded; every island is its own. CAR and ACTIVE components are computed separately and need not line up.
3. **Keep every component.** Small ones are packed into expert groups by size, to keep shard counts reasonable. A component attached to a larger one only by one-way roads (a one-way street into a dead end, a mis-tagged junction) is a trapped component, usually a data error: it stays in the model, but no ID snaps to it (§6). A component with no road at all to the rest is an island, served whatever its size.
4. **ACTIVE reachability per metric.** An ACTIVE expert is a component of the union graph. Inside it, a pair one mode cannot connect gets ∞ for that metric, which is exact.
5. **Cells, when an expert is too large to build at once.** The compiler coarsens the expert, finds its top nested-dissection separators on the coarse graph, and splits it into cells that fit the memory budget. Each cell is ordered and contracted on its own; the separators are ordered above all cells and contracted last.
6. **One hierarchy per expert.** Contracting a cell's interior only creates shortcuts among that cell's own nodes and its separator nodes, so the assembled hierarchy is exact. Its shards follow the cells.

The assembly must give the same answers as building the expert in one piece. Its only acceptable cost is a slightly worse ordering (more levels or shortcuts), which is measured and reported (§13, milestone 4).

&#91;embedded content: model anatomy · experts by connectivity, cells by memory\]

Experts (top row) are separate road networks with no road between them; cells (bottom) are only how one large expert gets built within memory, tied back together by the separators.

## 6. Vocabulary: what a query can name

The only inputs the model accepts are Overture address IDs and Overture place IDs, anywhere a query takes a location (source, target, group member). It never accepts coordinates.

- **Addresses:** a street number, street and postcode; no name, no category.
- **Places:** something people go to, with a name and categories. A place counts for its primary category and all its alternates. Addresses and places are separate Overture datasets and are not linked by ID.
- **Entries per mode family:** each ID maps to its expert and its entries in that expert's graph. CAR: the two directed segments of its road piece, with the offset in seconds to each end. ACTIVE: the two ends of its road piece, with offsets per metric.
- **Snapping happens once, at compile time.** An ID snaps to the nearest eligible road piece within 2 km for car and 500 m for bike and walk. An ID whose closest point falls exactly on a node stands at that node. Pieces in trapped components (§5) are skipped: the ID snaps to the next nearest eligible piece within the radius, or gets the reason code TrappedComponent if there is none. Only the result is stored; no geometry or coordinate index ships in the model.
- **Eligibility:** only road pieces in the mode's graph are eligible. For car: no motorways, trunk roads, or their links and ramps. For every mode: no bridges, tunnels or covered pieces, so an address never lands on an overpass or in a tunnel. Service roads stay eligible. The exact rules live in the profile config.
- **Not served:** an ID that fails to snap for a mode keeps a reason code for that mode (`NoEligibleRoad` with the nearest distance, `NoRoad`). An ID absent from the pinned release answers `UnknownId`. There is no lookup outside Overture IDs; client-side mappings (BAN, coordinates) are out of scope.

## 7. Numeric rules

Every time in the model and in every answer is an integer number of seconds in `u16`, capped at 65,534 s (18 h 12 min 14 s); 65,535 means beyond the cap or unreachable.

- **Exact below the cap.** Any true time under 65,535 s is reported exactly. Any time at or above it is reported as `BEYOND`, for every operation, point-to-point included (Lisbon to Moscow answers `BEYOND`).
- **Saturating arithmetic.** Every runtime must add with saturation: ∞ plus anything stays ∞. Shortcuts longer than the cap saturate too, which is consistent with the cap.
- **Rounding once.** Each OSM way's total time is computed in floating point from the profile and rounded once to the nearest second (ties away from zero). Its whole seconds are then distributed across the way's pieces by the largest-remainder method, so the pieces sum exactly to the rounded total and short pieces do not all round down to zero. Shortcuts are exact integer sums; no other rounding happens anywhere.
- **Master times.** The road part also stores each road's time as `f32`, for re-customization and later training. Runtimes answer from the `u16` metrics only.
- **Indices** are `u32`.
- **Tie-break.** When several answers share the same time, the smallest Overture ID wins: the nearest place, the meeting point, any reported argmin. Routes themselves are not exposed, so their ties do not matter.
- **Coordinates** are WGS84 where the compiler uses them; none are exposed.

## 8. Model format

A model is one directory: `manifest.json`, `model.safetensors.index.json` (tensor name to shard) and the shards, like a large LLM checkpoint. Runtimes memory-map the shards and read only the experts they use.

- **Manifest:** format version, model ID, compiler version with commit and dirty flag, input provenance (`.pbf` name, timestamp, SHA-256; Overture release; taxonomy version), config hashes (profiles, place rules), the list of experts (family, component count, node and link counts, levels, cells, shards), and the hash of every part.
- **Road part, per expert:** the hierarchy (nodes in level order, upper-neighbour lists, level boundaries), the `u16` metrics per mode, the master `f32` road times, and the re-customization data: which road initializes each link, and the triangles per level. With these, a runtime can recompute every link weight from new road times without the compiler.
- **Vocabulary part:** the ID table (Overture ID to internal index, stored as one byte buffer), the entries per mode family, the reason codes, and the place categories (basic category, hierarchy, primary and alternates). It records the hash of the road part it was built on and refuses any other.
- **Shards** are split by expert, part and cell, with a target of 2 GB and a hard maximum of 4 GB (configurable). A shard holds one expert group or a slice of one expert, never a mix of unrelated parts.
- **Never in the model:** snap geometry, coordinate indexes, category fields, runtime tuning (tile sizes, block splits, batch sizes, the narrow-top cutoff).
- **Versioning:** semantic versions. A minor version may add optional tensors that old runtimes can ignore; anything else is a major version.

Every tensor's name, dtype, shape, meaning and invariants are specified in FORMAT.md, written and reviewed in milestone 1 (§13) before any compiler code.

## 9. Query semantics

The spec defines each operation by its exact result, never by an algorithm. A runtime may limit sizes or time bounds, but within its limits it must return exactly the reference answer.

| Operation | Result |
| --- | --- |
| `travel_time(a, b, mode)` | the shortest time from `a` to `b`, `BEYOND`, or a reason |
| `isochrone(sources, T, mode, dir)` | every address and place ID whose time is ≤ T |
| `nearest(category, id, mode, dir)` | the time to the nearest place of that category, and its ID |
| multi-source column | for each ID, the time to the nearest of the sources, each source with an optional start time |
| `reachable_from_all` / `_any(sources, T, mode)` | IDs within T of every source / of at least one |
| `multi_criteria(expr)` | IDs satisfying a boolean expression over (category or source, mode, threshold) terms; or a weighted score with top-k |
| chained trip | car to a place of a transfer category (car parks), plus a fixed penalty, then walk: the minimum over transfer places |

- **Direction:** `from` is the time from the sources to each ID; `to` is the time from each ID to the sources (the reversed graph). One-way streets make them differ.
- **Same ID:** 0 s.
- **Same road piece:** the minimum of driving straight along the shared piece, when allowed and the target is ahead, and the best route through at least one other node (around the block).
- **Different experts:** unreachable, answered without computation (`DifferentComponent`).
- **Mixed modes:** terms of one expression may use different modes; they combine per ID.
- **Ties:** the smallest Overture ID wins (§7).

## 10. Build

One command builds a model from a `.pbf` and an Overture release, deterministically, within 62 GB of RAM whatever the input size.

```
blobc build   --pbf <file.osm.pbf> --overture <release dir> --config <config dir> --out <model dir>
blobc verify  <model dir> [--tier tiny|small|region|country]
blobc inspect <model dir>
```

- **Memory:** peak RSS of at most 62 GB for any input. Cells (§5) and memory-mapped intermediates make this hold for a continent. Every build reports each stage's wall time and peak RSS.
- **Determinism:** the same inputs, config and compiler version give byte-identical output, with any thread count. CI checks it.
- **Time:** no target; up to a week for the world is acceptable.
- **Cadence:** a full rebuild every month. No incremental build machinery is required.
- **Code:** a Rust workspace of small crates, traits and composition, no inheritance-style hierarchies. Configs (profiles, place rules) are files, not code.

## 11. Conformance

A model is correct when the CPU reference runtime matches Dijkstra on the original graphs for every check below, with zero mismatches; a runtime is correct when it replays the golden queries and matches their answers exactly.

| Tier | Content | When |
| --- | --- | --- |
| tiny | the 3×3 toy town (17 links, 5 levels, its known distance grid) and synthetic graphs | every commit |
| small | a city-sized `.pbf` | every commit |
| region | a region-sized `.pbf` | by hand |
| country | France | by hand, every release |

Checks:

- CCH against Dijkstra: single and multi-source columns, start times, the reversed graph, both ACTIVE metrics, turn-aware Dijkstra for CAR.
- Re-customization with random metrics on the same hierarchy.
- The address and place heads, including same-piece pairs and same-ID queries.
- Snapping re-check: re-snapping reproduces the stored entries exactly.
- Experts: components found as expected on synthetic graphs (islands, one-way traps, a footbridge linking two parts for walkers only); cross-expert pairs unreachable.
- Cells: an expert built as cells gives the same answers as the same expert built in one piece.
- The cap: synthetic graphs with routes longer than 18 h give `BEYOND`, and routes just under it stay exact.
- Ties: the smallest-ID rule on synthetic equal-time cases.
- Deliberately broken models and runtimes must fail.

Golden queries and their answers are stored per tier, so any runtime can be checked without the compiler.

## 12. Licensing

A model built from OSM is a derived database under the ODbL; API answers computed from it are produced works that need attribution.

- **OSM roads:** ODbL. Distributing model files (for example to browsers) means offering them under the ODbL with attribution. Serving answers only needs attribution.
- **Overture addresses and places:** the licences of each theme and of each address source in the pinned release.
- **In every model:** a `LICENSES` entry generated from the inputs at build time, and the attribution strings in the manifest.

## 13. Milestones

Each milestone ends at a gate, and the next starts only once the gate passes. The world stays the target throughout: no milestone may take a shortcut that would not scale to it.

1. **Spec and FORMAT.md.** This spec approved; FORMAT.md written (every tensor, invariant and version rule) and approved. Gate: both signed off, before any compiler code.
2. **Toy and synthetic.** The compiler, reference runtime and suite on the toy town and synthetic graphs (islands, one-way traps, the cap, ties, cells). Gate: tiny tier green, byte-identical builds across thread counts.
3. **France.** A full build from the France `.pbf` within 62 GB. Gate: country tier green; accuracy against OSRM at least as good as the research engine (car MAE about 19 s over France); every component served, Corsica and islands included.
4. **Cells.** France built with a memory budget small enough to force cells, against France built in one piece. Gate: identical answers; levels, shortcuts and build time compared and reported.
5. **France and its neighbours.** A `.pbf` covering France, Belgium, Luxembourg, Switzerland and Germany. Gate: exact cross-border trips and isochrones; Overture data quality reported per country.
6. **Europe.** Gate: built within 62 GB; measured bytes per link, per address and per column, and the extrapolation to each continent.
7. **World.** Gate: built; experts per continent listed; the hardware choice for serving made from the measured sizes.

## 14. Deferred

Deferred, with room left for them in the format: ferries and car shuttles, time-dependent metrics, extra metrics (e-bike, slow walker), slope for bikes, public transport, training road times on observed trips, GPU runtimes, precomputed fields, incremental builds.

No question remains open. The last four were settled in review: trapped components (§5, §6), shard size (§8), snapping eligibility (§6) and rounding (§7).
