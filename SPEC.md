# BlobMap World Model — Compiler & Format Spec (v2 draft)

Nov 8, 2026 · @Mathias Vandaele

SPEC version 2.0.0

## 1. Purpose and scope

`blobc` compiles one OpenStreetMap `.pbf` and one pinned Overture release into one model: a manifest plus sharded safetensors that any runtime can execute to answer exact travel-time queries by car, bike and walk.

- **In scope:** the compiler, this spec and the tensor-level format spec (FORMAT.md), a CPU reference runtime, the conformance suite.
- **Out of scope:** GPU runtimes (CUDA, Metal, wgpu), servers, the BlobMap API, precomputed category fields, runtime benchmarks.
- **Consumer:** BlobMap servers only for now. The spec is written so it could be published.
- **Scale:** designed for the world. The compiler compiles exactly the `.pbf` it is given; the first target is France.
- **The repository stays lean:** one workspace of two crates (§10), no experiments.

## 2. Concepts

No boundary in the model is drawn by a human: components come from the connectivity of the road network, cells from the compiler's own nested dissection. Country borders, polygons and file splits never change a result.

| Term | Definition | Decided by |
| --- | --- | --- |
| Model | One versioned artifact: a manifest, an index and shards. One build produces one model. | the build |
| Mode family | CAR (car) or ACTIVE (bike and walk, one shared graph). | this spec |
| Metric | One set of travel times for one mode: car, bike or walk. | this spec (3 metrics) |
| Hierarchy | One CCH per mode family, over all its components. Its elimination tree is a forest with one tree per component, so components never share links or levels. | the compiler |
| Component | One strongly connected component of one mode family's graph: one tree of the family's hierarchy. Every trip inside a component is exact; trips between components are unreachable. The model stores the component of every node. | the compiler, from the data |
| Cell | A build unit: a piece of a component, chosen by nested dissection so the build fits in memory. Never a routing boundary. | the compiler, from the memory budget |
| Road part | The graph, hierarchy, metrics and re-customization data of each mode family. Changes with OSM. | the build |
| Vocabulary part | Overture address and place IDs mapped directly to their nodes and offsets in each mode family's graph, plus place categories. Changes with each Overture release, and depends on the road part's hash. | the build |

## 3. Inputs and data rules

Roads come only from the `.pbf`; addresses and places come only from Overture. Every routable road in the `.pbf` is compiled and served; nothing is dropped.

- **Roads:** all ways routable by at least one mode, from the whole `.pbf` passed in. The extent of the model is the extent of the `.pbf`.
- **Overture:** one pinned release per model, recorded in the manifest.
- **Extent:** `blobc build` takes an optional `--extent <poly>`, a polygon file recorded in the manifest. With it, the model holds only the Overture IDs inside the polygon. Without it, the model holds every ID within the snap radius (§6) of a road: within some mode's radius of a piece of that mode's family graph. France builds use Geofabrik's `france.poly`. An ID outside the extent is absent from the model.
- **IDs:** every Overture ID is a UUID. The build fails if any ID it reads is not a valid UUID.
- **Addresses:** every Overture address in the extent. An address is served for a mode only if it snaps (§6).
- **Places:** Overture places, filtered by the rules below, with their categories (§6).
  - A place dropped as closed or below the confidence threshold stays in the model, unserved, with the reason `FilteredPlace` (§6).
  - A duplicate merged into another place stays in the model as an alias of the place kept (§6).
- **No other source:** no OSM POIs, no Overture roads, no BAN.

Place rules (defaults, kept in versioned config; the config hash goes into the manifest):

| Rule | Default |
| --- | --- |
| Closed places | dropped (Overture operating status) |
| Confidence, categories under `node:health_care` | ≥ 0.5 |
| Confidence, all other categories | ≥ 0.6 |
| Duplicates | merged when same basic category, within 30 m, compatible names; the highest confidence is kept, ties to the smallest ID |
| Categories | Overture's taxonomy of the pinned release: basic category and taxonomy hierarchy, with Overture's redirect rules |

Each build writes a data report, outside the model directory (§8): `<model dir>.data-report.json`, next to the model directory.
- **Place counts** per country and basic category, compared with the previous build's data report and flagging any change above 20 %. The previous report is passed as `--previous-report <file>` (§10); without it there is no comparison.
- **Coverage** per country: the addresses and places in the model, and the countries with places but no address. Overture has no addresses for some countries (Great Britain and India in release 2026-10-21.0), so the gap shows at every scale.
- **Trapped components** (§5), which usually point at OSM errors: every one, sorted by size, with its node count, the number of IDs whose nearest eligible piece lies in it, and one OSM way id, so the large ones can be fixed in OSM. Provider score distributions drift between releases, so the rules are reviewed against this report every month.
- **Small components** (§6), per mode, each with its node count and one OSM way id, sorted by IDs:
  - every component that IDs were moved away from, with the IDs moved and the IDs left on it;
  - every component other than the largest that still serves IDs: the real islands, and fragments with no larger component in reach, usually OSM errors to fix;
  - every real island the rule emptied: a small island within the radius of a larger component loses its IDs to it (§6), as Île de la Dérivation, Île Callot and Île des Embiez do for car in France.
  - Each component listed is marked with the real island it stands on, if any. Each of its servable pieces counts for the smallest OSM island polygon (`place=island|islet`, or a closed coastline way) around it, or for none; the polygon holding the most of them, ties to none and then to the first polygon read, is a real island when it holds no servable piece of a larger component. So a component mostly on the mainland, with a few pieces on an island it reaches by bridge, stands on no island.
- **Driving side** (§4, turns): the car segments keeping left and keeping right, each split by what decided it (the way's `driving_side` tag, or the polygon its last node lies in), and the 1° cells holding the side fewer segments keep, with one way each.
- **Dead-end restrictions** (§5): every via-way restriction whose prefix copy has no turn out, with its relation id and the copied segment's way, usually a mis-tagged restriction.

## 4. Modes and profiles

Three metrics exist in this version, car, bike and walk, in two mode families. Profiles are config files; their hashes go into the manifest. Travel times are static: no time of day.

|  | CAR | ACTIVE |
| --- | --- | --- |
| Modes | car | bike, walk (one graph, two metrics) |
| Graph | edge-based: nodes are directed road segments, arcs are allowed turns | node-based: nodes are intersections, arcs are road pieces |
| Network | ways usable by cars | union of the ways usable by bike or on foot |
| Speeds | OSRM v6 car profile: speeds by road class, `maxspeed`, access and one-way rules | bike 15 km/h, walk 5 km/h, flat |
| Turns | OSRM v6-style turn penalties (angle, u-turns), mirrored where traffic keeps left | none |
| Restrictions | all OSM turn restrictions, `no_*` and `only_*`, via a node and via one or more ways | access per mode; steps for walk only; bike contraflow (`oneway:bicycle=no`) |
| Delays | traffic signals, stops and give-ways, as in OSRM v6 car | traffic signals and crossings, as node costs |
| Forbidden for a mode | arc absent | arc weight ∞ for that metric |
| Reference accuracy | OSRM v6 car | OSRM bike and foot (bike differs by its missing turn costs, about 5 %, accepted) |

- **Delays** are part of the time of the road that arrives at them, in the direction they apply to (§7).
- **Driving side:** as OSRM v6 given its location-dependent data. A way keeps left when its `driving_side` tag says `left`; without the tag, when its last node lies in one of OSRM v6's left-hand polygons (`data/driving_side.geojson`, held in the car profile). Elsewhere traffic keeps right. A turn takes the side of the road it arrives by, and keeping left inverts the turn bias, so right turns cost what left turns cost elsewhere.
- **The 180° meridian:** OSM splits ways at the antimeridian by convention, so a road across it ends at one node at longitude +180° and goes on from another at −180°. Nodes at exactly +180° and exactly −180° with the same stored latitude (OSM's integer of 1e-7°, so the same point, not a tolerance) are one routing node in both families: the one with the smallest OSM id, which also takes their obstacles and their turn restrictions. When several nodes share such a latitude, all of them are that one node. Everything that uses coordinates (lengths, angles, snapping, island polygons, nested dissection) takes longitudes across the meridian the short way, so a component on both sides is one region.
- **Ferries and car shuttles:** excluded in this version. The format reserves a `ferry` link type (fixed crossing time plus an average wait) so they can be added later as a profile change.
- **Format room:** a mode family may carry more than its current metrics later (e-bike, slow walker), without a format change. This version defines exactly three.

## 5. Components and cells

The compiler finds each mode family's components by connectivity and splits large components into cells by memory, then builds one exact hierarchy per family whose elimination tree has one tree per component. No user input decides either.

1. **Build each mode family's graph** over the whole `.pbf` (CAR: the turn graph; ACTIVE: the bike and walk union).
2. **Compute strongly connected components.** Eurasia plus Africa is one CAR component; Britain is its own while ferries and shuttles are excluded; every island is its own. CAR and ACTIVE components are computed separately and need not line up.
   - **CAR via-way copies** (§4, FORMAT §5.3): components are determined by the original segments only: they are those of the originals, a turn into or out of a copy counting as one into or out of its original. Every copy belongs to its original's component, whatever its own strong connectivity. A copy is an arrival state and no trip starts from one, so this stays exact; a turn from a copy into another component is dropped like any turn between components. A restriction whose prefix copy has no turn out is likely mis-tagged and is listed in the data report (§3).
3. **Keep every component.** Small ones share packs, to keep shard counts reasonable; packs are storage only. A component attached to a larger one only by one-way roads (a one-way street into a dead end, a mis-tagged junction) is a trapped component, usually a data error: it stays in the model, but no ID snaps to it (§6). A component with no road at all to the rest is an island, served whatever its size.
   - **Precisely:** group the components that roads join, ignoring road direction. In each group, the largest component is served and every other one is trapped, with ties to the smaller component index. A group of one component is an island.
4. **ACTIVE reachability per metric.** An ACTIVE component is a component of the union graph. Inside it, a pair one mode cannot connect gets ∞ for that metric, which is exact.
5. **Cells, when a component is too large to build at once.** The compiler orders the component's fine graph by the same nested dissection as in one piece, and cuts the elimination tree of that order where subtrees fit the memory budget: the subtrees below the cut, packed together, are the cells; the nodes above it, the top separators, are the top block. Every block keeps that order. Each cell is contracted on its own; the separators are ordered above all cells and contracted last.
   - **Fallback:** only a component whose fine graph is too large to order within the build's memory (§10) is coarsened first. Its top separators then come from the coarse graph's order, and each block is ordered on its own.
6. **One hierarchy per family.** Each component is ordered and contracted on its own, so no link joins two components and each is one tree of the family's elimination forest. Contracting a cell's interior only creates shortcuts among that cell's own nodes and its separator nodes, so the assembled hierarchy is exact. Its shards follow the cells.

The assembly must give the same answers as building the component in one piece. Cut from the fine order, it is the one-piece hierarchy: a reordering of the same elimination tree, with the same links. Only the coarse fallback may cost a slightly worse ordering (more levels or shortcuts), which is measured and reported (§13, milestone 4).

&#91;embedded content: model anatomy · components by connectivity, cells by memory\]

Components (top row) are separate road networks with no road between them; cells (bottom) are only how one large component gets built within memory, tied back together by the separators.

## 6. Vocabulary: what a query can name

The only inputs the model accepts are Overture address IDs and Overture place IDs, anywhere a query takes a location (source, target, group member). It never accepts coordinates.

- **Addresses:** a street number, street and postcode; no name, no category.
- **Places:** something people go to, with a name and categories. A place counts for its primary category and all its alternates. Addresses and places are separate Overture datasets and are not linked by ID.
- **Category queries:** names first go through Overture's redirect rules. A name that matches no category after that is an error, not an empty answer. A place's own category keys go through the same redirect rules at build time. There are two namespaces:
  - **Taxonomy:** a query names a taxonomy node. It matches every place whose primary category, or any alternate, is that node or one of its descendants.
  - **Basic categories:** a query names a basic category. It matches the places whose basic category is that one.
- **Aliases:** the ID of a duplicate merged into another place (§3) is an alias of the kept place.
  - A query naming an alias is answered exactly as if it named the kept place; an alias and its kept place are the same ID (§9).
  - Answers name kept IDs only.
- **Entries per mode:** for each mode, each ID maps directly to its nodes and offsets in the family's graph; its component is that of its nodes.
  - **CAR:** the two directed segments of its road piece, with the offset in seconds to each end.
  - **ACTIVE:** bike and walk snap separately on the family's shared graph. One ID may stand on different pieces, even in different components, for the two modes. Each entry holds the two ends of its piece, with offsets for that mode's metric.
  - A direction the mode may not use has offset ∞. An example is a bike against a one-way it can neither ride nor push along.
  - Offsets are whole seconds (§7).
- **Snapping happens once, at compile time.** An ID snaps to the nearest eligible road piece within 2 km for car and 500 m for bike and walk. An ID whose closest point falls exactly on a node stands at that node. Pieces in trapped components (§5) are skipped: the ID snaps to the next nearest eligible piece within the radius, or gets the reason code TrappedComponent if there is none. An ACTIVE piece whose two ends lie in different components lies on no route, and is skipped the same way. Only the result is stored; no geometry or coordinate index ships in the model.
- **Small components yield, in every mode.** A component of at most N nodes is never chosen when a piece of a larger component, left after the skips above, lies within the radius: the ID snaps to the nearest such piece. Otherwise the nearest piece wins, as above. So an ID beside a tiny fragment (a car park, a mapping gap) snaps to the street beyond it. N is a profile setting per family. For CAR it is 1000, mirroring OSRM's default small-component size, so an ID in an enclave that access tags cut off (a gated estate, a pedestrian old town) also snaps to the street beyond; no access tag is a special case. For ACTIVE it is 10, since on foot a small network is the real one. The tiny-tier fixtures keep 10 for CAR, to fit their synthetic scale.
  - Islands stay served whatever their size; a small one loses its IDs only to a larger component within the radius. Fragments stay in the model and are never dropped.
  - No other rule applies to fragments. The IDs they keep are reported as OSM errors (§3).
- **Eligibility:** only road pieces in the mode's graph are eligible. For bike or walk, those are the pieces of the shared graph with at least one direction that mode may use. For car: no motorways, trunk roads, or their links and ramps. For every mode: no bridges, tunnels or covered pieces, so an address never lands on an overpass or in a tunnel. Service roads stay eligible. The exact rules live in the profile config.
- **Not served:** an ID that fails to snap for a mode keeps a reason code for that mode:
  - `NoRoad`: no piece of the family's graph within the mode's radius;
  - `NoEligibleRoad`: pieces of the family's graph lie within the radius, but none is eligible for the mode; the distance to the nearest one is stored;
  - `TrappedComponent`: eligible pieces lie within the radius, but all of them are in trapped components (§5).
- **Other answers:**
  - A place dropped by the place rules (§3) answers `FilteredPlace`, for every mode.
  - Only an ID absent from the pinned release, or outside the extent (§3), answers `UnknownId`.
  - There is no lookup outside Overture IDs; client-side mappings (BAN, coordinates) are out of scope.

## 7. Numeric rules

Every time in the model and in every answer is an integer number of seconds in `u16`, capped at 65,534 s (18 h 12 min 14 s); 65,535 means beyond the cap or unreachable.

- **Exact below the cap.** Any true time under 65,535 s is reported exactly. Any time at or above it is reported as `BEYOND`, for every operation, point-to-point included (Lisbon to Moscow answers `BEYOND`).
- **Saturating arithmetic.** Every runtime must add with saturation: ∞ plus anything stays ∞. Shortcuts longer than the cap saturate too, which is consistent with the cap.
- **Rounding once.** A road's time includes the delays charged to it (§4).
  - **Ways:** each OSM way's total time, per direction and per metric, is computed in floating point from the profile and rounded once to the nearest second (ties away from zero). Its whole seconds are then distributed across the way's pieces in that direction by the largest-remainder method. So the pieces sum exactly to the rounded total, and short pieces do not all round down to zero.
  - **Groups and ties:** a group is one way, one direction and one metric, over the pieces that metric may use. Equal remainders go to the piece that comes first in the direction of travel.
  - **Inputs:** the computation starts from the stored `f32` master times, summed in `f64` in travel order. A runtime can therefore re-round new master times exactly.
  - **Turn costs (CAR):** each is rounded once, to the nearest second.
  - **Offsets of an ID (§6):** the offset before the ID is the road's whole-second time times the fraction of the piece's length before the ID, rounded once. The offset after the ID is the rest, so the two always sum to the road's time.
  - Shortcuts are exact integer sums. No other rounding happens anywhere.
- **Long pieces.** A piece whose master time, in some direction and for some metric, exceeds 65,534 s is cut at compile time into the fewest equal-length parts whose master times are all at most 65,534 s, with new nodes along its geometry.
  - A delay (§4) stays with the part that arrives at it.
  - The parts keep the piece's way and stay in its rounding groups; they are numbered in order along the way, like any other pieces.
  - Rounding adds less than one second to a road, so every finite road weight is below 65,535 s. An ID on a long piece therefore keeps finite offsets, and its times stay exact below the cap.
- **Master times.** The road part also stores each road's time as `f32`, for re-customization and later training. Runtimes answer from the `u16` metrics only.
- **Indices** are `u32`.
- **Tie-break.** When several answers share the same time, the smallest Overture ID wins: the nearest place, the meeting point, any reported argmin. IDs compare as UUIDs in their canonical lower-case text form. Routes themselves are not exposed, so their ties do not matter.
- **Coordinates** are WGS84 where the compiler uses them; none are exposed.

## 8. Model format

A model is one directory: `manifest.json`, `model.safetensors.index.json` (tensor name to shard), the shards (like a large LLM checkpoint) and a `LICENSES` file (§12). Nothing else is in it. The build and data reports (§3, §10) are written outside it. Runtimes memory-map the shards and read only the packs they use.

- **Manifest:** format version, model ID, compiler version with commit and dirty flag, input provenance (`.pbf` name, timestamp, SHA-256; Overture release; taxonomy version), config hashes (profiles, place rules), per family its node, link, component and pack counts, and the hash of every part.
- **Road part, per family:** the component of every node, the hierarchy (nodes in level order, upper-neighbour lists, level boundaries), the `u16` metrics per mode, the master `f32` road times, and the re-customization data.
  - **Re-customization data:** which road initializes each link, each road's rounding group (way, direction, metric; §7), and the triangles per level.
  - **Triangles:** they are derived from the sorted upper-neighbour lists. Storing them is optional.
  - With these, a runtime can recompute every link weight from new road times without the compiler.
- **Vocabulary part:** the ID table (Overture ID to internal index, stored as one byte buffer), the aliases, the entries per mode, the reason codes, the filtered places, and the place categories (basic category, hierarchy, primary and alternates, redirects). It records the hash of the road part it was built on and refuses any other.
- **Shards** are split by part, pack and cell, with a target of 2 GB and a hard maximum of 4 GB (configurable). A shard holds one pack (small components together, or one block of a large one) or a slice of one, never a mix of unrelated parts.
- **Never in the model:** snap geometry, coordinate indexes, place names, category fields, runtime tuning (tile sizes, block splits, batch sizes, the narrow-top cutoff).
- **Versioning:** semantic versions. A minor version may add optional tensors and optional manifest keys. Old runtimes ignore them, and every runtime must ignore manifest keys it does not know. Anything else is a major version.

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
| `multi_criteria(expr)` | the IDs satisfying a boolean expression over (target, mode, direction, max) terms; or the top k by a weighted score (below) |
| `chained(a, b, transfer_category, penalty)` | min over transfer places p of car(a, p) + penalty + walk(p, b), and that p |
| chained column `(id, transfer_category, penalty, dir)` | for each ID, the chained time from `id` (`from`) or to `id` (`to`) |

- **Direction:** `from` is the time from the sources to each ID; `to` is the time from each ID to the sources (the reversed graph). One-way streets make them differ.
- **Same ID:** 0 s.
- **Same road piece:** the minimum of driving straight along the shared piece, when allowed and the target is ahead, and the best route through at least one other node (around the block). Two IDs at the same offset on the shared road count as ahead: 0 s.
- **Different components:** unreachable, answered by comparing the two IDs' component ids, without computation (`DifferentComponent`).
- **Mixed modes:** terms of one expression may use different modes; they combine per ID.
- **Ties:** the smallest Overture ID wins (§7).
- **Reasons:** an ID that is not served for the mode an operation needs answers its reason (§6).

**Chained trips.** A chained trip is always car, then walk: car to a transfer place (car parks, say), a fixed penalty, then walk.
- **Transfer places:** the kept places that match the transfer category (§6) and are served for both car and walk.
- **Point to point:** `chained(a, b, transfer_category, penalty)` = min over transfer places p of car(a, p) + penalty + walk(p, b).
  - `a` must be served for car and `b` for walk; otherwise the answer is that ID's reason.
  - The answer is the time and p, ties to the smallest ID (§7). When no p gives a time below the cap, it is `BEYOND`, with no place.
  - **Same ID:** when a = b, the answer is 0 s with no place (the same-ID rule above).
- **Column:** `dir` `from` gives, for every ID y, the chained time from `id` to y; `to` gives, for every y, the chained time from y to `id`. The trip is car then walk in both.
  - `id` must be served for its leg's mode: car for `from`, walk for `to`. An ID not served for the other leg's mode gets `BEYOND`, as in a column.
  - `id`'s own entry is 0 s when it is served for both modes (the same-ID rule).
- **Arithmetic:** `penalty` is whole seconds below 65,535, and the sum saturates (§7).

**`multi_criteria`.** The expression is a JSON tree:
- **A term:** `{"term": {"target": T, "mode": m, "dir": "from" | "to", "max": secs}}`, where T is one of:
  - `{"category": key}` (a taxonomy node, §6);
  - `{"basic": key}` (a basic category, §6);
  - `{"sources": [ids], "start": [secs]}`, where `start` is optional and gives one start time per source.
- **Combinators:** `{"and": [...]}`, `{"or": [...]}`, `{"not": x}`.
- **Kind:** the request has `"kind"`: `"addresses"`, `"places"` or `"both"` (default `"addresses"`).
- **Candidates:** IDs of that kind that are served in every mode the expression uses.
- **A term's time** for an ID:
  - for a category target, the time to the nearest place of it (as `nearest`);
  - for a sources target, the multi-source column with its start times;
  - both in the term's mode and direction.
- **Truth:** a term is true when that time is ≤ `max`; `BEYOND` is false.
- **Boolean form:** the result is the matching candidates, sorted by Overture ID (§7).
- **Weighted form:**
  - the request also has `"k"`, and every term has an integer `"weight"` w (`u32`);
  - an ID's score is the sum of w × seconds over all terms, computed in `u64`;
  - a candidate with `BEYOND` in any weighted term is excluded;
  - the expression, with its `max` values, still filters;
  - the result is the top k by lowest score, ties to the smallest ID.
  - No floating point is used anywhere in the operation.

## 10. Build

One command builds a model from a `.pbf` and an Overture release, deterministically, within 62 GB of RAM whatever the input size.

```
blobc build   --pbf <file.osm.pbf> --overture <release dir> --config <config dir> --out <model dir> [--extent <poly>] [--previous-report <file>]
blobc verify  <model dir> --tier tiny|small|region|country --data <dir> [--golden <dir>] [--mutants]
blobc inspect <model dir>
```

- **Memory:** peak RSS of at most 62 GB for any input. Cells (§5) and memory-mapped intermediates make this hold for a continent. Every build reports each stage's wall time and peak RSS, in a build report written outside the model directory, as `<model dir>.build-report.json` (these numbers are not deterministic).
- **Reports:** `--previous-report` reads the data report of an earlier build, `<model dir>.data-report.json` (§3).
- **Verify:** `--data` is the directory holding the tier's inputs. Each tier lists its input files, by path under that directory and SHA-256, in a tier file in the repository; `verify` checks every one before use and picks the inputs whose `.pbf` and extent match the model's manifest. `--golden` is the root of the golden files, `<root>/<tier>/<inputs>/` (§11).
- **Determinism:** the same inputs, config and compiler version give byte-identical output, with any thread count. CI checks it.
- **Time:** no target; up to a week for the world is acceptable.
- **Cadence:** a full rebuild every month. No incremental build machinery is required.
- **Code:** two crates: blobmap-format and blobc, traits and composition, no inheritance-style hierarchies. Configs (profiles, place rules) are files, not code.

## 11. Conformance

A model is correct when the CPU reference runtime matches Dijkstra on the original graphs for every check below, with zero mismatches; a runtime is correct when it replays the golden queries and matches their answers exactly.

| Tier | Content | When |
| --- | --- | --- |
| tiny | the 3×3 toy town (17 links, 5 levels, its known distance grid) and synthetic graphs, including via-way restrictions (`no_*` and `only_*`, with one and with two via ways) | every commit |
| small | a city-sized `.pbf` | every commit |
| region | a region-sized `.pbf` | by hand |
| country | France | by hand, every release |

Checks:

- CCH against Dijkstra: single and multi-source columns, start times, the reversed graph, both ACTIVE metrics, turn-aware Dijkstra for CAR.
- Via-way restrictions:
  - every forbidden sequence is avoided, against a turn-aware Dijkstra that tracks the segments driven;
  - an address on a via segment, reached through a restricted prefix, gets its exact time, in both directions.
- Re-customization with random metrics on the same hierarchy.
- The address and place heads, including same-piece pairs and same-ID queries.
- Snapping re-check: re-snapping, trapped skips and the small-component rule included, reproduces the stored entries exactly; an ID beside a fragment snaps to the street, and IDs on a large island within the radius and on a small one beyond it stay on their islands.
- Components: found as expected on synthetic graphs (islands, one-way traps, a footbridge linking two parts for walkers only); pairs in different components unreachable.
- The 180° meridian (§4): on a synthetic graph across it, joined only at a pair of twin nodes, every mode goes across; on every input, the twins found in the `.pbf` route as one node in each family.
- Cells: a component built as cells gives the same answers as the same component built in one piece.
- The cap: synthetic graphs with routes longer than 18 h give `BEYOND`, and routes just under it stay exact; a piece longer than the cap is cut (§7), and an ID on it gets its exact time.
- Car `to` queries (from each ID to the sources):
  - same-piece pairs on one-way and two-way pieces, in both orders, including equal offsets and the way around the block;
  - routes just under the cap and just over it, in the `to` direction.
- Ties: the smallest-ID rule on synthetic equal-time cases.
- Deliberately broken models and runtimes must fail.

Golden queries and their answers are stored per tier, so any runtime can be checked without the compiler.

## 12. Licensing

A model built from OSM is a derived database under the ODbL; API answers computed from it are produced works that need attribution.

- **OSM roads:** ODbL. Distributing model files (for example to browsers) means offering them under the ODbL with attribution. Serving answers only needs attribution.
- **Overture addresses and places:** the licences of each theme and of each address source in the pinned release.
- **In every model:** a `LICENSES` file in the model directory, generated from the inputs at build time, and the attribution strings in the manifest. Its counts are of the rows present in the model.

## 13. Milestones

Each milestone ends at a gate, and the next starts only once the gate passes. The world stays the target throughout: no milestone may take a shortcut that would not scale to it.

1. **Spec and FORMAT.md.** This spec approved; FORMAT.md written (every tensor, invariant and version rule) and approved. Gate: both signed off, before any compiler code.
2. **Toy and synthetic.** The compiler, reference runtime and suite on the toy town and synthetic graphs (islands, one-way traps, the cap, ties, cells). Gate: tiny tier green, byte-identical builds across thread counts.
3. **France.** A full build from the France `.pbf` within 62 GB. Gate: country tier green; accuracy against OSRM at least as good as the research engine (car MAE about 19 s over France), with pairs in different components only where a real island or a ferry separates them; no ID unserved because of its component, Corsica and islands served.
4. **Any pbf.** Extracts that stress what France does not: Great Britain and Japan (left-hand traffic: turn penalties follow the driving side as in OSRM v6), Fiji (the 180° meridian), Norway (ferries excluded), India (sparse tagging) and a large US state (California). Each is built and verified with the small-tier checks, the via-way copy check exhaustive. The planet runs through the input stages only (the planet `.pbf` and all of Overture), with node locations in a disk-backed, memory-mapped index, measuring peak RSS per stage. Gate: every extract at zero mismatches, its data report reviewed; the planet input stages within 62 GB.
5. **Continent.** The coarsening fallback (§5) at most 1.25× the sweep layers of one piece (FORMAT's level count: top block plus deepest cell), forced on Hauts-de-France, France, and a `.pbf` covering France, Belgium, Luxembourg, Switzerland and Germany. Attempts, in order: flow-based refinement of the lifted separators; then, for CAR, an order from the node-based graph. Verify split per component on disk, so it scales past 1.5× France. Then Europe. Gate: the fallback within 1.25× at zero mismatches and peak RSS ≤ 62 GB; Europe built within 62 GB and verified, with its bytes per link, per address and per column and the extrapolation to the world.
6. **World.** Gate: the planet built within 62 GB, verified at the tiers that apply, and its sizes reported for choosing the serving hardware.

## 14. Deferred

Deferred, with room left for them in the format: ferries and car shuttles, time-dependent metrics, extra metrics (e-bike, slow walker), slope for bikes, public transport, training road times on observed trips, GPU runtimes, precomputed fields, incremental builds.

No question remains open. The last four were settled in review: trapped components (§5, §6), shard size (§8), snapping eligibility (§6) and rounding (§7).

The format review of milestone 1 (Nov 8, 2026) settled twelve more:
- the rounding of turn costs, delays and offsets (§4, §7);
- rounding groups (§7, §8);
- derived triangles (§8);
- per-mode snapping for bike and walk (§6);
- the extent (§3) and the UUID rule (§3);
- equal offsets on a shared piece (§9);
- aliases and filtered places (§3, §6);
- reason codes (§6);
- versioning (§8);
- place names (§8);
- `LICENSES` and the build report (§8, §10);
- category queries (§6);
- `multi_criteria` (§9).
