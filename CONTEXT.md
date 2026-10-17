# SmartTile Agent Context

For coding agents working on SmartTile. It is more implementation-facing than
`README.md` (the user contract); use the 3Dtrees monorepo `CONTEXT.md` for
shared terminology. Keep both files short and aligned with `src/run.py`.

## Task modes

- `tile`: indexes LAS/LAZ/COPC inputs, plans buffered tiles and reads every
  source point once to write `subsampled_res1..N` (`--tile-outputs`; default
  1 cm COPC and 10 cm LAZ). `original` keeps every point. `--instance-tiles`
  writes whole-instance collections for inputs that already carry predictions.
- `merge`: optionally transfers predictions to dense target geometry, then
  filters core ownership, recovers, reconciles and deduplicates tile-local
  instances. `--segmented-remapped-folder` skips the transfer and only filters
  already-dense (or coarse-only, with `--resolution-1`) tiles; it replaced the
  removed `filter` task. With `--original-laz-input-dir` it also enriches the
  originals (Galaxy: Filter/Merge with Original LAZ/LAS files).
- `remap`: transfers prediction dimensions from one or more collections onto the
  uploaded LAS/LAZ originals (`--original-laz-input-dir`; the hidden legacy alias
  `--original-input-dir` is still accepted). `--original-copc-input-dir` only
  validates COPC twins; no enriched COPC original is written.
- `create_merged_file`: builds `prod_merged_*` / `merged.*` products from
  Original-with-predictions (or plain source) files.

## Merge contract (3DT-2101)

- `run.py` merge and remap use `remap_first_pipeline.py`, `merge_stages.py`,
  `dense_instance_ownership.py` and `bounded_point_index.py`. Do not reintroduce
  a second merge path.
- With target geometry, transfer each model's predictions before filtering and
  require 100% assignment (`--prediction-transfer-tolerance`; Galaxy uses the
  0.1732 m default, √3 × a 10 cm source voxel). Without targets use the supplied geometry; coarse-only inputs
  must keep samples from the same tiling run. Record the actual resolution. Do
  not enlarge the 5 cm correspondence or 1 cm duplicate radii to the voxel size.
- Remove whole instances whose anchor (`centroid`, `highest_point`,
  `lowest_point`; tested in XY) lies outside the core towards a declared
  neighbour, even when that neighbour is absent from a partial collection.
  Background uses half-open cores (east/north own shared edges).
- Non-RCT orphan recovery admits whole rejected instances that supply
  unsupported core samples, greedily by newly covered samples with stable ties,
  using a disk-backed claim index (`orphan_recovery` in the report).
- Reconcile IDs per model without tree sidecars: each accepted transitive group
  gets one ID and keeps every member's unique geometry; never drop a whole member.
- Shared points between retained positive instances (including one group)
  within 0.01 m go to the claimant nearest its closed XY core rectangle; ties
  by stable source filename order. Background or removed instances cannot win.
  Then retained trees override neighbouring background within 0.01 m.
- Deduplicate label-consistent cross-tile points against actual final survivors
  within 0.01 m XYZ; same-tile points are never thinned. Other label conflicts fail.
- After deduplication (and optional small-instance reassignment), every instance
  moves whole into one output tile; an ID in two tiles fails the merge.
- Declared overlaps require positive XY intersection; localized bounds carry an
  eight-ULP per-axis allowance from the global bounds and origin. Distance tests
  are inclusive with the same eight-ULP float64 guard; nearest ties within it use
  stable tile/point order, independent of batches, cache and workers.
- A tiling bypass writes one layout entry with actual bounds and no buffer.
  Historical unused grid plans are recovered only for one cloud matching the full
  extent within 1 cm with untouched planned bounds; record `layout_recovery`.
- Original coverage: 100% of every original within √3 × first-stage resolution
  (17.32 mm at 1 cm) for the baseline, when checked, and for final non-RCT
  survivors. RCT final misses get 0 in every prediction field and are reported.
  No 99% fallback.
- Preserve unfiltered dense geometry and `smarttile_merge.json` for remap. The
  remap baseline check is optional; a manifest baseline that no longer exists is
  skipped, and Galaxy checks final coverage only.
- Merge and remap refuse COPC-only originals (`original_files`).
- Index dense/filtered/shared-owner records during their write pass; do not
  reread compressed outputs to build an equivalent index.

## RayCloudTools (RCT)

- `PredInstance_RCT` with paired `_trees.txt`/`_trees_info.txt` per tile (and
  optional `_trees_mesh.ply`), or the same files from `--tree-tables-folder`,
  `--tree-info-folder` and `--tree-meshes-folder` (kind given by the folder,
  tile by name), selects the membership-preserving path: no
  reconciliation, point reassignment or deduplication; small-instance
  reassignment and an intermediate merged LAZ are rejected (`--skip-merged-file`).
- Retained IDs are `tile_id * 100000 + local_id` (one-based layout position;
  subsets keep the original ID). Local IDs below 100000, encoded IDs fit uint32.
  Tables keep matching rows with the same encoded `predinstance`; explicit IDs
  are read by ID, not row position. The namespace VLR (record 24002) prevents
  applying the offset twice on a repeat merge.
- Recover a rejected RCT instance only if it supplies unsupported core geometry
  and none of its points (including buffer tails) is within 1 cm XYZ of a
  retained or already recovered tree; report blocked candidates.
- Final remap prefers a positive tree within the radius over background and
  compacts IDs dataset-wide (VLR 24003, `rct_instance_mapping.json`, per-original
  tables and meshes). Standalone remap rejects several unencoded legacy tiles.
- See `docs/raycloud-filter-only.md`.

## Product contract

- User-facing products are `original_with_predictions/` and prod-merged files
  built from those originals. The processed `merged_tiles.laz` and
  center-of-mass tiles are intermediate/diagnostic; never use COM geometry as
  an analysis product.
- Two product goals: (1) a final merged product from the enriched originals
  after remap; (2) a source-file union of uploads that keeps CRS, scales,
  offsets, compatible dimensions and truthful metadata. Multi-source products
  must not present one source's specific metadata as describing the whole.
- Final original enrichment compacts surviving IDs per model across the
  collection to uint32 1..N (background 0), keeps memberships and every non-ID
  field, and writes `instance_mapping.json`. `--match-instances` keeps IDs.
- Labels: `0` background, positive instances, negative invalid. Intermediate
  labels use uint16 unless a value exceeds 65535, then uint32.
- Keep prediction dimension names exactly; duplicate names across collections
  fail instead of being renamed. `OriginalPointIndex` is never transferred.

## Metadata and CRS invariants

- Original-with-predictions files preserve their source header, CRS/projection
  records, scales, offsets, point format and non-prediction ExtraBytes. The
  uploaded LAS/LAZ is the metadata source; COPC twins only validate.
- Index every prediction collection sequentially into a disk-backed
  `PointIndex`, then stream originals in bounded batches; no COPC spatial
  queries, no full cloud in memory.
- One standardized WKT CRS record per LAS 1.4/COPC output; validation is
  semantic. Resolve duplicate projection records with the reader's last-record
  precedence (EVLRs after VLRs); repeated conversion must be idempotent. Do not
  promise byte-identical VLRs after LAZ → COPC.
- Tiling CRS check (`tile_crs`): reads WKT in EVLRs; rejects geographic,
  geocentric, non-metre horizontal or vertical units, unreadable CRS records,
  differing CRSs and mixed with/without CRS.
- ExtraBytes `no_data` is translated through scales/offsets via the canonical
  metadata helper; lossy or ambiguous declarations fail. Patch only the
  fixed-size descriptor payload so COPC hierarchy offsets stay valid. Mixed
  inputs need compatible metadata.
- Vector ExtraBytes are scalarized for COPC with a VLR/EVLR schema and a
  mandatory `.vector-extra-bytes.json` sidecar, and verified value by value.
- A staged COPC is reused only when the staging manifest records its source; a
  header match is not enough (it could hold old predictions).
- `--standardization-json` reads `collection.reference_attribute_names` from
  tool_standard `collection_summary.json`, ignores zero-variance dims and checks
  staged COPCs and LAS/COPC products.
- PLY carries no VLRs: CRS goes into `comment crs:`; restore exact property
  name case after PDAL. Do not claim VLR preservation for PLY.

## Tiling and subsampling

- Input discovery (`point_cloud_metadata.point_cloud_files`): COPC preferred
  over its LAS/LAZ twin; names equal ignoring case/extension fail; empty files fail.
- The grid is built from indices (`start + i * length`), so float steps cannot
  stall it and a zero-width extent gets one column/row. Points outside every
  planned core (stale header bounds) fail the read.
- Tile LAS offsets are whole metres (`tile_offsets`) with the finest source
  scale per axis, keeping voxel keys lattice-exact.
- `tile_bounds_tindex.json` bounds are the union over every LAS tile output.
- One read for both methods (`tile_centroids`): occupancy counted in the same
  read; exact integer voxel means rounded half-to-even once; every resolution
  from original points; results independent of workers, block size and order.
- `center-of-mass` averages XYZ only; never average other attributes. Where
  attributes must survive, copy them from a real point (`nearest-to-centroid`).
- Every process pool uses an explicit `spawn` context: `fork` inherits
  lazrs/Rayon locks and can deadlock on the first compressed read.
- Products (`create_merged_file`): strips of about 2 M points, planned from LAS
  headers, read with a margin; each strip owns whole voxel columns on an
  absolute grid (`select_voxel_points`), so no point is lost between strips and
  products do not depend on `--workers`. Finest input scale. A failing chunk
  cancels the rest. The full product streams inputs (COPC-only sources included).

## Workers and memory

- `--workers` is the only CPU setting: tiling read/reduce, merge per-tile and
  query processes, remap enrichment, product chunks. Default and cap: allocated
  CPUs (`GALAXY_SLOTS`, affinity, cgroup quota; `worker_budget`). Record the
  requested/effective budget in `parallelism`.
- Enrichment spills originals into 4 m-region blocks (`original_blocks`), sends a
  region always to the same worker, admits at most two batches per process and
  writes in original order with one writer; tiny inputs stay serial.
- Immutable nearest queries reuse 4 m XY-region trees in a process-wide LRU
  cache (512 MiB, capped by cgroup memory and process count;
  `SMARTTILE_SPATIAL_CACHE_MB=0` disables it). Cache keys include positive-label
  filters, overlap and core eligibility; results equal disk queries.
- Same-instance remap is the ordinary nearest query with an equality filter on
  the instance dimension, applied to cache identity and both tie passes.
- Keep runs bounded: chunked reads/writes, no giant in-memory cloud.

## Module map

- `run.py` (CLI, task routing), `parameters.py` (settings), `worker_budget.py`.
- Tile: `main_tile.py` (`plan_tiling`), `tile_crs.py`, `tile_tindex.py`,
  `get_bounds_from_tindex.py`, `prepare_tile_jobs.py`, `tile_core_occupancy.py`,
  `tile_centroids.py` (single read), `tile_original_geometry.py`, `voxel_grid.py`,
  `tile_copc.py` (Untwine/PDAL COPC), `output_names.py` (`file_stem`, the one
  helper that drops the extension for pairing and naming; output names and
  labels), `subsample_*.py`, `plot_tiles_and_copc.py`.
- Instance tiles: `instance_tiles.py`, `instance_tile_outputs.py`.
- Merge/remap: `remap_first_pipeline.py` (orchestration, `enrich_originals`,
  `strict_remap`), `merge_stages.py`, `dense_instance_ownership.py`,
  `ownership_spatial_blocks.py`, `orphan_instance_recovery.py`, `orphan_claims.py`,
  `instance_consolidation.py`, `small_instance_reassignment.py`,
  `instance_statistics.py`, `instance_finalization.py`, `instance_labels.py`,
  `tile_bounds_graph.py`, `tile_file_matching.py`, `prediction_collections.py`,
  `original_blocks.py`, `output_publication.py`, `collection_folders.py`
  (`--collection-dir`: hard links into one flat folder per Galaxy collection).
- Spatial index and parallelism: `bounded_point_index.py`,
  `spatial_query_cache.py`, `parallel_tiles.py`, `parallel_index_queries.py`,
  `parallel_remap.py`.
- RCT: `raycloud_instance_ids.py`, `raycloud_recovery.py`,
  `raycloud_tree_files.py`, `raycloud_meshes.py`, `raycloud_finalization.py`.
- Products: `main_create_merged_file.py`, `full_resolution_merge.py`,
  `attribute_join.py`, `product_formats.py`, `copc_staging.py`.
- Metadata: `point_cloud_metadata.py`, `copc_metadata.py`, `crs_records.py`,
  `ply_crs.py`, `vector_extra_bytes.py`.

Workflow diagrams: `docs/task-workflow.md`; merge ownership decisions:
`docs/filter-decision-flow.md`. Update them when task behaviour changes.

## Change safety checklist

- Original-with-predictions keeps source metadata, CRS records (including WKT),
  prediction dimensions and original ExtraBytes?
- With `--standardization-json`, do outputs keep the expected dimensions?
- 0/background and positive-instance semantics unchanged?
- No center-of-mass geometry in prod-merged products?
- Outputs independent of `--workers`, batch size and read order?
- Large inputs still chunked or streamed within production memory limits?
- Staged publication: nothing partial or overwritten on failure?
- README examples, this file and the docs still match `src/run.py --help`?

## Validation

In the locked environment (`conda-linux-64.lock`, plus `pytest` and
`pytest-subtests`):

```bash
python -m py_compile src/*.py
PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests
git diff --check
docker run --rm <image> python -u /src/run.py --show-params
```

Important test areas: output formats and `create_merged_file` products;
metadata/CRS/ExtraBytes preservation; COM and nearest-to-centroid selection;
voxel boundaries and offsets; one-pass multi-collection remap; label dtypes;
scale/offset preservation; `--workers` budget wiring; RCT sidecars and IDs;
COPC dimension policy (strip extras for tiles, preserve them for products).

For algorithm or performance changes, also run the candidate image on a real
multi-tile dataset through tile, merge (SAT and ForestMamba), remap and
`create_merged_file`; record wall time, peak memory and output equality against
a saved baseline, with read-only inputs and isolated runs. The historical GFZ
harness (`tests/integration/gfz/`) and the
[validation records](https://github.com/3dTrees-earth/3dtrees_Smart_Tile/tree/b2b84466158c528b75dffcbdfa4c04488261c06d/docs/validation)
remain at commit `b2b8446`.

## Watch items

- `main_create_merged_file.py` is still large; extract focused helpers instead
  of adding modes inline.
- COPC finalization is scratch-disk heavy even when memory is bounded.
- Dense-cell point targets are soft: very dense inputs can exceed them.
