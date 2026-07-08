# SmartTile Agent Context

This file is for coding agents working on SmartTile. It is intentionally more
implementation-facing than `README.md`; use the README for user-facing behavior
and the repository root `CONTEXT.md` for shared 3Dtrees terminology.

## Current Handoff State (2026-07-08)

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
- Instance labels use the simple contract: `0` is background/no tree, positive
  values are tree instances, and negative labels are invalid.
- Keep prediction dimension names exactly as supplied. Multi-collection remap
  must fail on duplicate output dimension names instead of auto-renaming.
- Use `uint16` for prediction labels unless a positive instance value exceeds
  `65535`; then use `uint32`.

## Task Modes

- `tile`: converts uploaded LAZ/LAS/COPC inputs into spatial COPC tiles, then
  creates subsampled products. The default first resolution is 1cm COPC LAZ; the
  default second resolution is 10cm regular LAZ.
- `merge`: filters duplicate buffer-zone instances from segmented predictions,
  remaps the filtered predictions to the target resolution, merges the remapped
  predictions into per-tile 1cm products, and can enrich uploaded originals from
  those per-tile products before optional prod-merged creation.
- `filter`: removes duplicate buffer-zone instances from segmented/remapped
  tile files before downstream merge/remap workflows.
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
- For merged-COPC-to-original remap, stream uploaded LAZ/LAS originals in chunks
  and query the merged COPC by each chunk's spatial bounds before building a
  local KDTree. Do not load a full uploaded original or full merged COPC when a
  bounded spatial query can produce the same enriched original output.
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

## Subsampling Contract

- `center-of-mass` is the default subsampling method. It averages only XYZ inside
  each populated voxel.
- Non-coordinate attributes must not be averaged. When attributes need to remain
  on subsampled points, copy them from a real nearest source point.
- `nearest-to-centroid` preserves the previous PDAL voxel nearest-neighbor
  behavior.
- `--num-spatial-chunks` controls spatial parallelism for both subsampling
  strategies, COPC-original remap windows, and bounded prod-merged COPC reads.
- Keep large runs memory bounded: prefer chunked COPC reads/writes, avoid one
  giant in-memory point cloud, and stream batches into final products whenever
  practical.

## Module Map

- `src/run.py`: CLI entry point and task routing.
- `src/parameters.py`: Pydantic settings, CLI parameters, and validators.
- `src/main_tile.py`: tile task orchestration.
- `src/tile_copc.py`, `src/tile_tindex.py`, `src/tile_spatial.py`,
  `src/tile_bounds_graph.py`: tiling helpers.
- `src/main_subsample.py`: subsampling orchestration.
- `src/subsample_com.py`, `src/subsample_chunk_worker.py`,
  `src/subsample_methods.py`, `src/subsample_outputs.py`: subsampling helpers.
- `src/main_merge.py`, `src/merge_tiles.py`, `src/merge_tiles_cli.py`: merge
  task orchestration and compatibility entry points.
- `src/merge_*`: merge internals for overlap handling, instance matching,
  global IDs, orphan recovery, tile loading, and original dimension handling.
- `src/main_remap.py`, `src/prediction_collection_remap.py`,
  `src/output_remap.py`, `src/dimension_transfer.py`: remapping and dimension
  transfer.
- `src/main_create_merged_file.py`: prod-merged product creation.
- `src/copc_metadata.py`, `src/copc_staging.py`, `src/point_cloud_metadata.py`,
  `src/point_cloud_outputs.py`: metadata preservation, COPC staging, and output
  writing.
- `src/instance_labels.py`, `src/worker_budget.py`, `src/union_find.py`: shared
  contracts/utilities.

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

Fast local validation:

```bash
/home/kg281/anaconda3/bin/python -m py_compile src/*.py
/home/kg281/anaconda3/bin/python -m pytest
git diff --check
docker run --rm ghcr.io/3dtrees-earth/3dtrees_smart_tile:v2.2 python /src/run.py --show-params
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
- CPU/file worker budget wiring: file-level workers default to two; spatial
  chunking uses available CPUs/Galaxy slots
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
- `main_subsample.py` and `main_create_merged_file.py` are still large. Prefer
  extracting focused helpers instead of adding new modes inline.
- COPC finalization can be scratch-disk heavy even when memory is bounded.
  Preserve the current warnings and staged-COPC reuse behavior.
- The README is the user contract. This file is the agent/developer orientation;
  keep both short enough to stay useful.
