# Changelog

## v2.4.0 - 2026-10-15

### Correctness
- Merge output tiles have distinct instances: every instance ID is in exactly one tile (split reconciled instances move whole into the tile whose core holds their centroid; points are moved, never dropped), so DetailView sees each tree once. v2.3 retiled labels onto the buffered tiles, so border trees reached DetailView in two tiles. Output tiles are named `cXX_rYY.laz`.
- Remap-first merge: predictions go to 1 cm geometry before ownership, recovery, reconciliation and deduplication. Every original point must be covered by the final survivors (and by the unfiltered 1 cm baseline in merge, or in remap when baseline geometry is given), or the task fails without publishing (3DT-2101, 3DT-1566).
- Nearest-to-centroid tiling uses the same single read and fixed voxel grid as center of mass: each voxel keeps the original point nearest its exact mean, at any number of resolutions and in every format. It was not reproducible before (PDAL's grid followed the input point order, so repeated runs gave different point counts), and the second resolution was derived from the first. 3109 (330 M points, 10 CPUs, `--tile-outputs 0.01:laz,0.1:laz`): 62 s, the same as center of mass (63 s).
- Tiles are encoded with the finest scale of all sources per axis (previously the first source's), so no source loses precision.
- Center-of-mass tiles are exact voxel means of the original points at every resolution, rounded once onto the LAS grid. 10 cm is no longer a mean of 1 cm means (3DT-2203).
- Tiling rejects geographic, geocentric and non-metre CRSs, inputs with different CRSs, and inputs mixing files with and without a CRS (3DT-1898).
- PLY products keep the exact case of application properties (`treeID`, `PredSemantic`, `OriginalPointIndex`); PDAL lowercases them, so the header-only rewrite that adds `comment crs:` restores them, with no PDAL fork (3DT-2352).
- Original enrichment memory no longer depends on point density: a spill block above 8 M points is streamed cell by cell (2 m query cells, in point-index order) instead of loaded whole. 483 `3700.laz`, one 4 m region with about 99 M points: 10.9 -> 0.8 GiB for the block reads, same speed, identical results.
- Subsampled prod-merged products are reproducible: SmartTile selects the original point nearest each voxel's exact mean on an absolute grid instead of PDAL's voxel filter, whose grid followed the read order (645: two runs differed by 4,388 of 80 M points). Points are written in GPS-time order, since voxel order tripled the LAZ size (645 10 cm: 2.52 -> 0.81 GB, smaller than the originals).
- One CRS record per output; instance IDs widen to uint32 instead of wrapping (3DT-2200, 3DT-2172).

### Production review fixes
- Tiling fails instead of silently dropping points that lie outside the header bounds used to plan the grid (stale headers from third-party writers).
- Tile offsets are whole metres (and lattice-exact at original resolution), so points on voxel faces are no longer floored into the lower voxel when a float tile centre is not a decimal.
- The CRS check reads WKT stored in EVLRs, rejects vertical units other than metres and rejects unreadable CRS records.
- Inputs whose names differ only in case or extension (Plot1.las, plot1.laz) fail instead of one being ignored; empty input files fail with a clear message; a zero-width extent gets one tile column/row; the tile grid is built from indices, so no float rounding can stall it (3DT-1565).
- Subsampled prod-merged chunks are read with a margin and own whole voxel columns, so no point is lost between strips (pdal info rounded edge bounds; half-scale gaps with unaligned offsets) and products no longer depend on --workers. Chunk bounds come from LAS headers. Products use the finest input scale instead of the first input's.
- The full-resolution product includes COPC-only sources like the subsampled products, and its COPC accepts scalarized vector ExtraBytes (it failed on every vector input).
- A staged COPC is reused only when the staging manifest records its source; a header match could reuse old predictions.
- Remap and merge refuse COPC-only originals instead of skipping them with exit 0.
- Small-instance reassignment keeps tiles LAZ-compressed (it wrote uncompressed LAS into .laz files).
- tile_bounds_tindex.json bounds are the union over every LAS tile output, not only the first (a coarse first output could hide 1 cm points from merge's overlap checks).
- A failing product chunk cancels the remaining chunks.
- Parameters come only from the command line; job environment variables such as `WORKERS` or `TASK` no longer change a run.

### Performance (10 CPUs)
- Tiling reads every source point once (no full-resolution tile COPCs). 3109 (330 M points): 879–1,039 s → 160 s with 1 cm COPC + 10 cm LAZ (63 s with both as LAZ). 483 (3.68 B points): 641 s with 1 cm LAZ, 1,237 s with 1 cm COPC.
- Merge stages run in worker processes; enrichment queries originals in spatial blocks. 645: total 4 h 09 min → 46 min, enrichment 3 h → 170 s. 3109 SAT: 21,200 s → about 4,930 s. Outputs are identical for any worker count.

### Interface changes
- Removed the `filter` task: `merge --segmented-remapped-folder` does the same (production workflows never used it).
- Instance tiles for inputs that already carry predictions: `--instance-tiles PredInstance_SAT,...` writes one collection per dimension and resolution in which every instance is whole in exactly one tile (per-instance voxels at each of `--instance-resolutions`, nearest original point to the voxel mean; background by core ownership; many instance IDs found in several files more than 5 m apart fail as reused IDs; instances wider than `--max-instance-extent`, default 50 m, only warn). Command line only, not exposed in the Galaxy wrapper. `--match-instances` remaps per-point values (e.g. species and probability) from such tiles onto the originals within each instance, so neighbouring trees never exchange values.
- `--merged-resolutions full` writes a full-resolution prod-merged product of every input point in record order; `--original-point-index` (default on) adds `OriginalPointIndex` as the first ExtraByte so ForestStructure can restore source order after COPC and reproduce Julia's results. Merge and remap never transfer it onto originals.
- `create_merged_file` joins several versions of the same originals (comma-separated `--original-with-predictions-dir`, e.g. one remap output per segmentation run) by record order before merging, so one product carries every run's attributes.
- New `--tile-outputs "res:format,..."`: any number of tiled outputs in order (`laz`, `copc.laz`, `ply`) from the same read, written to `subsampled_res1..N`. The Galaxy wrapper exposes them as a repeat (default: one 10 cm LAZ output) with one collection each.
- Merge frees each intermediate stage as soon as the next one has read it (645: peak scratch was 25.4 GB for 0.4 GB of 1 cm input). `--skip-unfiltered-tiles` drops the unfiltered baseline copy (the Galaxy wrapper always passes it). Prod-merged chunks hold at most 2 M points each, so product memory no longer grows with the dataset (645: 30.7 GiB peak before).
- Standalone remap no longer needs the 1 cm tiles or a merge manifest: `--baseline-1cm-folders` is optional, and without it remap checks final coverage only. A baseline inferred from a merge manifest is skipped when its folder no longer exists. The Galaxy remap form drops the resolution-1 tiles input.
- Tile output labels are rounded: whole centimetres (0.29 m → `29cm`, previously truncated), otherwise millimetres (0.005 m → `5mm`).
- One CPU setting: `--workers` defaults to the allocated CPUs (`GALAXY_SLOTS`, affinity, cgroup quota) instead of 2 and drives every task. `--num-spatial-chunks` and `--tile-writer-workers` are removed; the tile task converts at most two outputs to COPC at once. Prod-merged product chunks (create_merged_file, and merge/remap with `--produce-merged-file`) are capped to the allocated CPUs too.
- Removed: the v2.3 merge/filter/remap path, 15 unused parameters, and `--chunk-size`, `--threads` and `--tile-source-workers` (only the old full-resolution tiling read them), and the Galaxy outputs `merge_report`, `remap_report` and `output_remap_mappings` (still written locally as evidence). The Galaxy wrapper no longer exposes a grid origin.

### Operations
- Scratch disk in the job directory: tiling spills about 16 B per source point (plus buffer overlap) and about 20 B per output point before assembly; 483 spilled 56 GiB. Enrichment spills 16 B per original point per file.
- Peak process memory measured at 10 CPUs: tiling 483 about 10.5 GiB; merge 3109 SAT about 6.2 GiB; merge 645 about 4.6 GiB.

## v2.0.1 - 2026-04-21

- Added filtered tile manifest output and annotated bounds JSON copies so omitted tiles are recorded explicitly.
- Skips writing buffer-only or empty filtered outputs, and skips remap follow-up work when no filtered tiles remain.
- Added `--output-copc-res2` and documented COPC support for segmented inputs, remap targets, and tiled outputs.
