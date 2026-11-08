# BlobMap world-model compiler (`blobc`)

## What is authoritative

- **SPEC.md** is the source of truth for meaning: what is built, what a model means, what
  every query answers. Its decisions are closed; do not reopen them.
- **FORMAT.md** is the source of truth for bytes: files, tensors, dtypes, shapes,
  invariants, hashes, versioning. It references SPEC sections instead of restating them.
- Nothing else is authoritative: old code, old docs, the reference repos, commit messages,
  memory notes. Where they disagree with SPEC.md or FORMAT.md, those two win.
- If you find a real contradiction or gap in SPEC.md or FORMAT.md, stop and ask. Do not
  pick an answer silently.
- **PLAN.md** orders the work by milestone. It decides nothing about meaning or bytes.
- Do not create other docs. A change of meaning goes into SPEC.md, a change of bytes into
  FORMAT.md, and only once it is approved.

## Constraints

- One Rust workspace of small crates. Use traits and composition, with no
  inheritance-style hierarchies. Configs (profiles, place rules) are files, not code.
- Output is deterministic: the same inputs, config and compiler version give
  byte-identical models with any thread count. Never iterate a `HashMap` into output.
- Peak RSS stays within 62 GB for any input (SPEC §10). Nothing may take a shortcut that
  would not scale to the world (SPEC §13).
- Never kill or restart other processes.
- Never query the live BlobMap API.
- Commit with clear messages. Work in milestone order (SPEC §13); the next milestone
  starts only when the current gate passes.

## Reference material (read-only)

Never modify these folders or their data. Port code from them where it helps, but never a
concept that contradicts SPEC.md.

- `~/Work/blobmap/the-new-blobmap-model`: the research repository. Its `RESULTS.md`
  documents every experiment, decision and measurement behind the spec.
- `~/Work/blobmap/cch-data`: research data and reference models. It holds the OSM
  extracts, the Overture release 2026-10-21.0 (`overture/2026-10-21.0`), BAN, and the
  compiled models.
- `~/Work/blobmap/blobmap-compiler`: a previous attempt. Its crates reproduce the research
  compiler byte for byte and are good porting material. Two things in it are wrong and
  must not be carried over:
  - its experts, one per region polygon (its DECISIONS D14): experts are connected
    components, and cells are compiler-chosen build units;
  - its FORMAT v1.

## Crate map

Build order (PLAN.md), each crate under `crates/`:
- `format`: the FORMAT.md writer and validating reader, manifest, hashes.
- `fixtures`: synthetic `.osm.pbf`, Overture parquet and taxonomy for the tiny tier; the
  `fixtures` binary writes every scenario's inputs.
- `profile`: car and active profiles, place rules, `build.toml`.
- `osm`: ways, pieces, restrictions. `graph`: CAR turn graph with via-way copies, ACTIVE
  union graph, rounding, experts.
- `order`: `Orderer` (inertial flow, fixed) and cells. `cch`: one hierarchy per expert.
- `overture`: release, taxonomy, place rules, aliases, extent. `snap`: IDs onto pieces.
- `build`: the `Stage`s, groups and packs, vocabulary, model writer, reports.
- `runtime`: the CPU reference runtime. `conformance`: the SPEC §11 suite, golden files,
  mutants. `cli`: `blobc build | verify | inspect`.

`ci.sh` runs the milestone 2 gate; `tools/determinism.sh` the thread-count check. Golden
files live in `golden/<tier>/<inputs>/`. `tiers/<tier>.toml` lists each tier's inputs by
path and SHA-256 (`blobc verify --data <dir>`); the `tier` binary of `conformance` writes
them (`tier tiny <dir> tiers/tiny.toml`, `tier add …`).
