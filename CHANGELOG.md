# Changelog

## v2.4.0 - unreleased

### Correctness
- Remap-first merge: predictions go to 1 cm geometry before ownership, recovery, reconciliation and deduplication. Every original point must be covered by the final survivors (and by the unfiltered 1 cm baseline in merge, or in remap when baseline geometry is given), or the task fails without publishing (3DT-2101, 3DT-1566).
- Nearest-to-centroid tiling uses the same single read and fixed voxel grid as center of mass: each voxel keeps the original point nearest its exact mean, at any number of resolutions and in every format. It was not reproducible before (PDAL's grid followed the input point order, so repeated runs gave different point counts), and the second resolution was derived from the first. 3109 (330 M points, 10 CPUs): 62 s, the same as center of mass (63 s).
- Center-of-mass tiles are exact voxel means of the original points at every resolution, rounded once onto the LAS grid. 10 cm is no longer a mean of 1 cm means (3DT-2203).
- Tiling rejects geographic, geocentric and non-metre CRSs, inputs with different CRSs, and inputs mixing files with and without a CRS (3DT-1898).
- One CRS record per output; instance IDs widen to uint32 instead of wrapping (3DT-2200, 3DT-2172).

### Performance (10 CPUs)
- Tiling reads every source point once (no full-resolution tile COPCs). 3109 (330 M points): 879–1,039 s → 160 s. 483 (3.68 B points): 641 s with 1 cm LAZ, 1,237 s with 1 cm COPC.
- Merge stages run in worker processes; enrichment queries originals in spatial blocks. 645: total 4 h 09 min → 46 min, enrichment 3 h → 170 s. 3109 SAT: 21,200 s → about 4,930 s. Outputs are identical for any worker count.

### Interface changes
- New `--tile-outputs "res:format,..."`: any number of tiled outputs in order (`laz`, `copc.laz`, `ply`) from the same read, written to `subsampled_res1..N`. The Galaxy wrapper exposes them as a repeat (default: one 10 cm LAZ output) with one collection each.
- Standalone remap no longer needs the 1 cm tiles or a merge manifest: `--baseline-1cm-folders` is optional, and without it remap checks final coverage only. The Galaxy remap form drops the resolution-1 tiles input.
- One CPU setting: `--workers` defaults to the allocated CPUs (`GALAXY_SLOTS`, affinity, cgroup quota) instead of 2 and drives every task. `--num-spatial-chunks` and `--tile-writer-workers` are removed; the tile task converts at most two outputs to COPC at once.
- Removed: the v2.3 merge/filter/remap path, 15 unused parameters, and `--chunk-size`, `--threads` and `--tile-source-workers` (only the old full-resolution tiling read them), and the Galaxy outputs `merge_report`, `remap_report` and `output_remap_mappings` (still written locally as evidence). The Galaxy wrapper no longer exposes a grid origin.

### Operations
- Scratch disk in the job directory: tiling spills about 16 B per source point (plus buffer overlap) and about 20 B per output point before assembly; 483 spilled 56 GiB. Enrichment spills 16 B per original point per file.
- Peak process memory measured at 10 CPUs: tiling 483 about 10.5 GiB; merge 3109 SAT about 6.2 GiB; merge 645 about 4.6 GiB.

## v2.0.1 - 2026-04-21

- Added filtered tile manifest output and annotated bounds JSON copies so omitted tiles are recorded explicitly.
- Skips writing buffer-only or empty filtered outputs, and skips remap follow-up work when no filtered tiles remain.
- Added `--output-copc-res2` and documented COPC support for segmented inputs, remap targets, and tiled outputs.
