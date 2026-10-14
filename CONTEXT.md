# SmartTile Agent Context

This file is for coding agents working on SmartTile. It is intentionally more
implementation-facing than `README.md`; use the README for user-facing behavior
and the repository root `CONTEXT.md` for shared 3Dtrees terminology.

## Current merge contract (2026-09-21 / 3DT-2101)

- `run.py` merge/filter/remap use `remap_first_pipeline.py`,
  `merge_stages.py`, `dense_instance_ownership.py` and
  `bounded_point_index.py`. The v2.3 centroid/orphan merge modules were removed;
  do not reintroduce a second merge path.
- RCT predictions (`PredInstance_RCT`) require paired `_trees.txt` and
  `_trees_info.txt` per tile. Filter whole instances by core ownership and encode
  retained IDs as `tile_id * 100000 + local_id`, with background 0. Tile IDs use
  one-based positions in the declared layout; subsets retain their original
  tile ID. Require local IDs below 100000 and reject uint32 overflow. Skip
  reconciliation, point deduplication, and membership changes. Recover a rejected
  whole RCT instance only when it supplies unsupported core geometry and none
  of its points is within 1 cm XYZ of a retained or previously recovered tree.
  Check its complete geometry, including buffer tails, before admitting it.
  Report blocked candidates and their competing instance IDs in orphan_recovery.
  Filter both tables to the retained IDs with the same encoded `predinstance`.
  Read existing explicit IDs rather than row positions. Preserve the per-tile
  namespace VLR on repeat filtering; never apply the offset twice. Record the
  local/global mapping in `instance_metadata.csv` and the merge manifest.
  Keep `--skip-merged-file` for the intermediate task. Standalone remap rejects
  multiple legacy unencoded RCT tiles; regenerate them through merge/filter.
  Validate collection namespaces from headers, then check labels during the
  indexing read; do not decompress prediction files in a separate preflight.
  Shared points can belong to rejected instances in both tiles even when the
  main neighboring counterparts survive; final RCT remap fills only unmatched
  predictions with 0. During original enrichment, prefer a positive RCT tree
  within the remap radius over background, preserving its attributes. See
  `docs/raycloud-filter-only.md` for the illustrated case.
- With target geometry, transfer each model's predictions before filtering and
  require 100% assignment. Galaxy derives transfer radius from the source voxel
  diagonal. Without targets, skip transfer and use supplied geometry for filtering;
  coarse-only inputs retain samples from the same tiling run. Record the actual
  resolution. Do not enlarge correspondence or duplicate radii to voxel size;
  see README's optional-target contract.
- Remove whole instances whose anchor on the processed geometry is outside their core;
  background uses half-open spatial cores. Reconcile IDs independently per
  model without tree sidecars: each accepted transitive group gets one ID and retains every member's
  unique geometry. Resolve shared points by tile ownership, retaining all
  attributes of the winning point; never discard a whole merged member.
- For retained positive instances (including the same merged group) sharing points within 0.01 m,
  assign disputed points to the retained claimant nearest its closed XY core
  rectangle; break equal-distance ties by stable source filename order. A tile
  predicting background or a removed instance cannot win a tree claim. Preserve
  the winner's instance ID and per-point attributes, and retain unshared buffer
  tails. Record these decisions in shared_point_ownership.
- Declared buffered overlaps require positive XY intersection before any
  roundoff allowance. When localizing their bounds, carry an eight-ULP
  per-axis allowance from the global XY bounds and common origin. Cached
  sources, disk sources and query points then share the same inclusive
  local-bound check and its eight-ULP allowance. Both guards are independent
  of batch composition and Z; preserve the geometric matching radius and
  core ownership boundaries.
- After tree/tree resolution, retained trees override neighboring background
  within 0.01 m, preserving the tree owner's semantic values and attributes.
  Query actual surviving trees; removed instances cannot override background.
- A tiling bypass writes one layout entry with actual cloud bounds and no buffer.
  Historical unused grid plans may be recovered only for one cloud matching the
  complete projected extent within 1 cm, with untouched planned bounds and an
  extent larger than any individual planned tile. Record the recovery; preserve
  missing-neighbor ownership for partial collections.
- Deduplicate label-consistent cross-tile points against actual final survivors
  within 0.01 m XYZ. Adjacent points in separate cores may retain different tree
  labels or background semantics. Other label conflicts still fail.
  Same-tile points are never thinned.
- Validate baseline original coverage at 100% within the first-stage voxel
  diagonal (17.32 mm for 1 cm resolution). Require 100% final coverage for
  non-RCT models. For RCT only, assign zero to every prediction field when no
  surviving point matches within that radius, and record the count and examples
  in the coverage report. Other sampling gaps, conflicts and missing predictions
  still fail before publication; the earlier 99% fallback is not used.
- Preserve unfiltered dense geometry and its manifest for the separate final
  remap task. The baseline check in remap is an optional diagnosis: the Galaxy
  remap no longer passes a baseline collection and checks final coverage only,
  and a manifest-inferred baseline folder that no longer exists is skipped.
  See the README for flags and diagnostics.
- `--workers` sets the merge/filter worker processes: per-tile processes for
  dense transfer, core filtering, reconciliation and shared-point ownership,
  plus query processes within tiles (dense transfer, recovery, deduplication),
  bounded by scheduler slots, affinity and detected cgroup CPU quotas. Small
  queries stay serial; index writes and model/tile ordering stay deterministic.
  Record the requested and effective budget in `parallelism`.
- Standalone strict remap and original enrichment during merge/filter share
  CPU-capped batch processes after serial indexing.
  Each process opens completed indexes read-only and uses one query thread.
  Admit at most two batches per process, preserve input order with one writer,
  and join workers before deleting scratch indexes. Tiny inputs stay serial.
  Keep coverage, stable ties, original fields and transactional publication
  identical to the serial path; report indexing/enrichment timings separately.
- Non-RCT orphan recovery uses a disk-backed spatial claim index and incremental
  candidate scores. Update coverage only near each newly admitted instance,
  keeping the same greedy ranking and stable ties. Report selection work as
  `orphan_recovery.selection_checked_locations`.
- Dense searches keep fixed-size disk-backed batches as the fallback. Immutable
  nearest queries reuse 4 m XY-region trees in a process-wide LRU cache. The
  default 512 MiB cache charge is capped by cgroup memory and the maximum remap
  process count; `SMARTTILE_SPATIAL_CACHE_MB=0` disables it. Reserve the read
  peak, then charge kept entries at their real size. Inserts invalidate
  cached regions. Keep source references, overlap masks, positive-label filters,
  numerical distance bounds and stable ties identical to disk queries.
- Shared-point ownership reuses bounded region trees filtered by positive labels,
  overlap and source-side core eligibility; the cache key includes both cores and
  the origin. Query-side nearest-core ranking is unchanged. Numerical nearest
  ties use the existing eight-ULP allowance relative to the true minimum distance,
  with stable tile/point order independent of cache or disk batch partitioning.
  Original baseline coverage needs only minimum distances, not source identity.
- Same-instance remapping delegates to the ordinary nearest query with an
  equality filter on the required instance dimension. Apply this filter to
  cache identity, candidate selection and both global numerical-tie passes;
  source batch partitioning must not change the selected prediction record.
- Index dense/filtered/shared-owner output records during their write pass;
  do not reread compressed outputs only to construct an equivalent index.
- CRS preservation resolves duplicate projection identities with the same
  last-record precedence as the reader (source EVLRs follow VLRs). Never append
  a shadowed normalized WKT over an already-preserved original WKT; repeated
  product conversion must be idempotent. Keep strict CRS validation enabled.
- Preserve ExtraBytes missingness through the canonical metadata helper before
  publishing LAS/LAZ/COPC conversions. Translate raw sentinels through source and
  destination scales/offsets; reject lossy or ambiguous declarations. Patch only
  the fixed-size descriptor payload so COPC hierarchy offsets remain valid.
  Require compatible collection metadata before mixing records, and include
  missingness/encoding in staged-COPC cache freshness.
- Exact production
  replays of 3110/3111 and resource benchmarking remain release obligations.

## Historical Handoff State (2026-07-08)

- Repo/branch: `/home/kg281/projects/3dtrees_smart_tile`, branch `v2.2`,
  tracking `upstream/v2.2`.
- Current code head: `bb13f7a Fix SmartTile v2.2 Docker tag docs`.
- Local checkout is expected to be clean before new work. Verify with
  `git status -sb`.
- Local validation command that worked in this checkout:
  `/home/kg281/anaconda3/bin/python -m pytest` -> `243 passed`.
- Published container smoke test that worked:
  `docker run --rm ghcr.io/3dtrees-earth/3dtrees_smart_tile:v2.2 python /src/run.py --show-params`.
- Published Docker image: `ghcr.io/3dtrees-earth/3dtrees_smart_tile:v2.2`
  from GitHub Actions run `28884470282`, package version id `1009022178`,
  updated `2026-07-07T17:07:37Z`.
- ToolShed has SmartTile `2.2+galaxy15` as revision `7:d839ad802f56`.
- UseGalaxy.eu still exposes only SmartTile `2.1.0+galaxy0`
  (`ctx_rev=6`, changeset `b05456e47259`) as of the last check.
- Production workflow pin PR exists but must stay draft until explicitly
  approved and until the Galaxy deployment plan is clear:
  `https://github.com/3dTrees-earth/3dtrees/pull/254`.
- A simple ToolShed wrapper rollback to `2.1.0+galaxy0` is not viable as a
  new ToolShed revision: PR `https://github.com/bgruening/galaxytools/pull/1911`
  was closed unmerged and CI failed with `ShedVersion` because ToolShed requires
  monotonic installable tool versions after `2.2+galaxy15`.
- Do not merge SmartTile wrapper, workflow, or production rollout PRs without
  explicit user approval.

## Product Contract

- Preserve uploaded point clouds as closely as the selected output format allows.
- User-facing analysis products are `original_with_predictions/` and
  prod-merged files created from those originals.
- SmartTile has two distinct user-facing merge goals:
  1. Create a final merged point-cloud product next to the enriched original
     files after prediction/remap.
  2. Merge multiple uploaded source files into one point-cloud product without
     losing CRS, source dimensions, or metadata that remains true for the merged
     product.
- COM/processed merged files are intermediate or diagnostic products. Do not use
  center-of-mass geometry as the authoritative merged product for further
  analysis.
- Final original enrichment compacts surviving IDs independently per model
  across the original collection into uint32 1..N, background 0. Preserve
  memberships and all non-ID fields; publish `instance_mapping.json`. RCT
  tree/treeinfo tables use the same final mapping, while intermediate tile
  namespaces remain reusable.
- Instance labels use the simple contract: `0` is background/no tree, positive
  values are tree instances, and negative labels are invalid.
- Keep prediction dimension names exactly as supplied. Multi-collection remap
  must fail on duplicate output dimension names instead of auto-renaming.
- For intermediate prediction labels, use `uint16` unless a positive instance value exceeds
  `65535`; then use `uint32`.

## Task Modes

- `tile`: indexes uploaded LAZ/LAS/COPC inputs, plans buffered tiles and reads
  every source point once to write the subsampled tiles directly
  (`subsampled_res1..N` from `--tile-outputs`); selecting Original resolution
  retains every point instead of subsampling. Without `--tile-outputs` the first output is 1cm COPC LAZ and the
  second 10cm regular LAZ.
- `merge`: optionally transfers predictions to target geometry, then filters,
  recovers, reconciles and deduplicates independent tile-local instances. Without
  targets it keeps the supplied resolution. The CLI can strictly enrich originals;
  Galaxy uses separate Remap/Create tasks for enriched originals and final products.
- `filter`: runs the same reconciliation/deduplication on already-dense tiles.
- `remap`: transfers prediction dimensions back to original source points. It
  supports multiple segmented prediction collections when their dimension names
  are already unique. The explicit production interface is
  `--original-laz-input-dir` for uploaded raw LAZ/LAS files. Optional
  `--original-copc-input-dir` may provide matching original COPCs for validation
  and source context, but remap does not create enriched COPC
  originals. Legacy `--original-input-dir` remains accepted as a LAZ/LAS source.
- `create_merged_file`: creates user-facing prod-merged outputs from
  Original-with-predictions files. It stages LAZ/LAS inputs to COPC, reuses
  existing staged COPCs when valid, and supports `copc.laz`, `laz`, and `ply`.

## Two Merge Goals

SmartTile's product merge behavior must support two related but different
workflows:

1. Final product after processing: after tiling, segmentation, filtering, and
   remap, users should receive `original_with_predictions/` plus one or more
   prod-merged files. These prod-merged files are built from the enriched
   originals so they use original uploaded points as product geometry.
2. Source-file union: users may also need multiple uploaded point-cloud files
   merged into one product even when the main objective is not segmentation
   cleanup. This path must keep CRS, scales, offsets, compatible source
   dimensions, and truthful metadata as far as the output format allows.

Do not confuse these goals with the processed/COM merged intermediate. A COM
merged file can be useful for diagnostics or model input/output inspection, but
it is not the analysis-grade source union and not the default final download
product.

## Metadata And CRS Invariants

- Original-with-predictions files represent one uploaded source file and should
  preserve that source file's header, CRS/projection VLRs, scales, offsets, point
  format, and non-prediction extra dimensions as far as LAS/COPC allows.
- Keep the product-enrichment source explicit: `--original-laz-input-dir` must
  use uploaded non-COPC LAZ/LAS files as the metadata source for faithful
  downloadable enriched originals. Optional `--original-copc-input-dir` must
  only describe matching original COPCs for validation or source context.
- In remap, validate matching COPC/LAZ source pairs when the optional COPC lane
  is configured, then enrich the uploaded LAZ/LAS originals directly from the
  prediction collections. The COPC lane must not produce an intermediate
  COPC-enriched original.
- For remap, index every prediction collection (LAS/LAZ/COPC, including a
  merged file) sequentially into a disk-backed `PointIndex`, then stream the
  uploaded LAZ/LAS originals in bounded batches against it. No COPC spatial
  queries are used, and no full original or prediction cloud is loaded at once.
- Create prod-merged `copc.laz`, `laz`, and `ply` outputs from the enriched LAZ
  originals rather than from COPC-original processing outputs.
- COPC output must preserve CRS metadata, including GeoTIFF/GeoKeyDirectory and
  WKT projection records. Do not only preserve one projection VLR record type.
- Do not promise byte-identical raw VLR preservation after LAZ -> COPC
  conversion. A COPC can preserve CRS semantically as WKT even when the uploaded
  raw LAZ represented CRS with GeoKeyDirectory/GeoAscii VLRs. Use the raw lane
  when the user-facing downloadable file should preserve the uploaded metadata
  representation as closely as possible. A LAZ written from COPC is not
  guaranteed to be identical to a LAZ enriched directly from the uploaded raw
  file.
- `--standardization-json` restores the v2.1 schema guard. It reads
  `collection.reference_attribute_names` from tool_standard
  `collection_summary.json`, maps R/LAS names to laspy names, ignores constant
  dims when global stats mark them as zero-variance, and validates that staged
  Original-with-predictions COPCs and LAS/COPC prod-merged outputs still expose
  those expected source dimensions.
- Multi-source prod-merged files should preserve CRS and run-true metadata, but
  must not pretend one source file's source-specific metadata describes the whole
  product.
- PLY is allowed as an output format, but PLY does not carry LAS/COPC VLR
  metadata. Do not claim CRS/VLR preservation for PLY products.
- SmartTile assumes upstream tools ensure CRS consistency across input files.
  SmartTile should preserve CRS, not perform semantic CRS reconciliation.

## Tiling process safety

- The single-read tile path decodes a compressed XY sample in the parent before
  its read/reduce workers start, and counts core occupancy in the workers' read.
  Every SmartTile process pool must use an explicit `spawn` context: Linux `fork`
  inherits lazrs/Rayon locks without their threads and can deadlock on the first
  worker read. Keep the compressed multi-chunk regression
  (`tests/test_tiling_process_start.py`, which exercises the older pre-scan
  occupancy-to-COPC path); uncompressed or single-chunk fixtures miss this.

## Subsampling Contract

- `center-of-mass` is the default subsampling method. It averages only XYZ inside
  each populated voxel.
- Non-coordinate attributes must not be averaged. When attributes need to remain
  on subsampled points, copy them from a real nearest source point.
- `nearest-to-centroid` in the tile task keeps the original point nearest each
  voxel's exact mean, on the same single read and fixed voxel grid as center of
  mass. `create_merged_file` products use the same rule on an absolute grid
  (`select_voxel_points`); PDAL only reads and merges the chunks.
- The tile task reads each source point once for both subsampling methods (`tile_centroids`):
  no full-resolution tiles, occupancy counted in the same read, exact integer
  voxel means rounded half-to-even once. Every resolution comes from original
  points. Outputs must not depend on worker count, block size or point order.
- `--workers` is the only CPU setting: tiling read/reduce processes,
  merge/filter queries, remap enrichment processes and prod-merged chunks. It
  defaults to the allocated CPUs, never the host core count.
- Enrichment queries originals in spatial blocks (`original_blocks`) and writes
  them back in original point order.
- Keep large runs memory bounded: prefer chunked COPC reads/writes, avoid one
  giant in-memory point cloud, and stream batches into final products whenever
  practical.

## Workflow Reference

See `docs/task-workflow.md` for flow diagrams of all five tasks, their input/output
contracts, model-specific recovery rules and matching distances. Keep these diagrams
aligned with `src/run.py` and the strict pipeline when task behavior changes.

## Module Map

- `src/run.py`: CLI entry point and task routing; `src/parameters.py`: settings.
- `src/main_tile.py`, `src/tile_*.py` (`tile_crs`, `tile_tindex`, `tile_copc`,
  `tile_bounds_graph`, `tile_core_occupancy`, `tile_file_matching`): tile
  planning (`plan_tiling`), occupancy publication, LAZ-to-COPC conversion and
  tile geometry/matching helpers.
- `src/tile_centroids.py`: the tile task's single read for both subsampling
  methods; `src/main_subsample.py`: output names/labels and bounds helpers;
  `src/subsample_*.py`: COM headers, COPC output conversion;
  `src/original_blocks.py`: block-ordered enrichment input.
- `src/remap_first_pipeline.py`: merge/filter/remap orchestration and original
  enrichment; stages in `src/merge_stages.py` (dense transfer, reconciliation,
  deduplication), `src/dense_instance_ownership.py` (core filter, shared-point
  ownership), `src/orphan_*.py`, `src/small_instance_reassignment.py`.
- `src/parallel_tiles.py`, `src/parallel_index_queries.py`,
  `src/parallel_remap.py`: worker processes; results always in tile/input order.
- `src/raycloud_*.py`, `src/instance_finalization.py`: RCT and final IDs.
- `src/prediction_collections.py`: collection discovery and dimension schemas;
  enrichment itself is `remap_first_pipeline.enrich_originals` with
  `src/parallel_remap.py`.
- `src/main_create_merged_file.py`: prod-merged product creation.
- `src/copc_*.py`, `src/crs_records.py`, `src/ply_crs.py`,
  `src/point_cloud_metadata.py`: metadata, CRS records and COPC staging.
- `src/instance_labels.py`, `src/worker_budget.py`: shared contracts/utilities.

## Change Safety Checklist

Before changing product behavior, check:

- Does the change preserve source metadata for Original-with-predictions?
- Does it preserve CRS VLRs for LAS/COPC, including WKT projection records?
- Does it keep prediction dimensions and original extra dimensions?
- If `--standardization-json` is supplied, does the output still contain the
  expected standardized source dimensions?
- Does it keep 0/background and positive-instance semantics?
- Does it avoid introducing center-of-mass geometry into prod-merged products?
- Does it keep large files chunked or streamed enough for production memory
  limits?
- Are README user examples and this context file still aligned?

## Validation

For optimization work, always validate the candidate image on the full GFZ
dataset: all four buffered tiles for SAT and ForestMamba, then original remap.
Small probes supplement this gate; they do not replace it. Record phase wall
time, container CPU time/peak memory, and exact output equality against the
saved baseline. Keep inputs read-only and each run isolated.

Fast local validation:

```bash
/home/kg281/anaconda3/bin/python -m py_compile src/*.py
/home/kg281/anaconda3/bin/python -m pytest
git diff --check
docker run --rm ghcr.io/3dtrees-earth/3dtrees_smart_tile:2.4.0 python -u /src/run.py --show-params
```

Important test areas:

- output format validation and `create_merged_file` products
- metadata/header/CRS preservation helpers
- COM and nearest-to-centroid subsampling selection
- scientific-notation bounds parsing
- EPSG:5650/zone-prefixed coordinate scale-offset safety
- one-pass multi-collection remap behavior
- prediction label dtype rules
- scale/offset preservation during remap
- CPU worker budget wiring: `--workers` defaults to the allocated CPUs
  (Galaxy slots, affinity, cgroup quota) and larger requests are capped
- tree sidecar pruning and disabling cross-tile matching/small-cluster
  reassignment when tree sidecars are present
- COPC conversion dimension policy: prefer Untwine, strip extra dimensions for
  non-prod subsampled/tiled COPCs, preserve enriched dimensions for prod-merged
  outputs

For production-like checks, use small real datasets first, then run a multi-file
dataset through tiling, segmentation, merge's built-in prediction filter/remap
lane, optional original remap, and `create_merged_file`. Compare output headers
and dimensions against the original source files.

## Current Watch Items

- Production deployment is intentionally not complete. UseGalaxy.eu has not
  installed v2.2, and the production workflow PR is draft. Dataset 2095 has only
  mathematical/local metadata validation for the EPSG:5650 overflow fix; it has
  not completed a live production v2.2 rerun.
- If the team wants to supersede the published ToolShed v2.2 wrapper, create a
  forward version such as `2.2+galaxy16` or `2.2.1+galaxy0`; do not attempt to
  publish a lower `2.1.0+galaxy0` wrapper as a new revision.
- `main_create_merged_file.py` is still large. Prefer
  extracting focused helpers instead of adding new modes inline.
- COPC finalization can be scratch-disk heavy even when memory is bounded.
  Preserve the current warnings and staged-COPC reuse behavior.
- The README is the user contract. This file is the agent/developer orientation;
  keep both short enough to stay useful.
