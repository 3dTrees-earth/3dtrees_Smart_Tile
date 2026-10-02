# Spatial ownership experiment — isolated side-conversation implementation

## Scope

Implements conservative occupied-block pruning and spatial ordering across file chunks for `assign_shared_points`. SAT and ForestMamba remain separate collections. RCT with paired sidecars continues to bypass this function. After successful validation, the exact three-file patch was applied to the main branch working copy. No commit, image replacement or running full-test modification was made. The supplied `spatial-ownership.patch` and source snapshot preserve the tested implementation.

## Method

A bounded positive-tree occupancy summary uses 2 m XYZ voxels. A query whose radius-expanded bounds contain no occupied competitor voxel avoids the expensive nearest search. Background does not mark occupancy. If the complete summary exceeds 200,000 voxels, pruning is disabled rather than using an incomplete summary.

Eligible query groups are staged to a disk-backed SQLite spool and processed by 4 m XY region. Original 2 m query groups, source row positions and numerical tolerances are preserved. All groups in a region use a common maximum halo for loading cached source geometry, while exact matching uses each original group's own tolerance. Existing source/query core eligibility and stable tile precedence are unchanged. Oversized source regions retain the original bounded batch fallback. A disk-backed removal mask permits output in the original record order, retaining original attributes. This adds one source-file pass and temporary disk storage proportional to eligible query points; the benchmark includes that cost.

## Regression validation

395 tests passed in the locked `smarttile:v2.4a-reconcile5` runtime, including five new tests for XYZ absence, cache enabled/disabled spatial ordering, neighboring-voxel halos, occupancy budget fallback and preservation of each query group's numerical tolerance. The first three regression cases failed on the old implementation. The first full suite run used disk scratch and was stopped because filesystem waits dominated; the completed run used bounded 2 GiB RAM scratch and took 11.95 s. No large-data processing was retried.

## Medium real-data benchmark

Two adjacent GFZ tiles (0 and 1) for each model, cropped to a 20 m square: X 398713–398733, Y 5646197–5646217. These are completed, immutable recovered point files from the existing full run, immediately before shared-point ownership. Original tile cores/overlaps are preserved. Cropping is only for this ownership-phase differential benchmark; it is not a new centroid/recovery or end-to-end validation.

- SAT: 2,492,787 + 2,309,578 = 4,802,365 records.
- ForestMamba: 2,274,988 + 2,614,232 = 4,889,220 records.
- Total: 9,691,585 records.

The baseline and candidate run sequentially with identical inputs, a 512 MiB spatial cache, 10 CPU / 50 GiB container limits, user kg281, no GPU and SSD scratch. Full tests in the main task continue concurrently, so storage/CPU contention can affect wall time. The same locked runtime is used with immutable source snapshots mounted read-only. Commands, cgroup CPU/peak-memory samples and per-model query counts are saved. Exact comparison covers all four output clouds: raw point-record bytes, point order/count, dimensions, scales and offsets. No tolerance-based comparison is used.

State and timings are in `manifest.json`, `outputs/baseline.json`, `outputs/candidate.json` and `outputs/equality.json`. All stages passed; all four output clouds matched exactly (6,151,540 output records).

## Results

| Model | Previous ownership | Candidate ownership | Reduction | Previous query calls | Candidate query calls |
|---|---:|---:|---:|---:|---:|
| SAT | 58.033 s | 41.795 s | 28.0% | 29,442 | 29,441 |
| FM | 62.461 s | 52.053 s | 16.7% | 38,336 | 38,219 |

Combined ownership: 120.494 s → 93.848 s (22.1% lower). These times include occupancy construction, query staging, matching, output writing and indexing, but exclude preparation of the common source index.

- baseline: container wall 138.269 s; CPU 138.740 CPU-s; observed peak 912.9 MiB. Container totals include source indexing and output hashing.
- candidate: container wall 112.134 s; CPU 110.918 CPU-s; observed peak 912.4 MiB. Container totals include source indexing and output hashing.

Spatial ordering/cache reuse provided most of the observed improvement. SAT region cache hits rose from 1,157 to 1,542; FM from 1,542 to 1,751. The occupancy guard pruned few actual queries on this dense crop: SAT query calls fell by one, FM by 117. This is not evidence that occupancy pruning alone is a substantial speedup. Oversized cache regions still use the bounded fallback and can perform repeated source queries.

This is one sequential ownership-phase comparison, with other full runs active on the host. It establishes exact output parity on this medium sample; it does not establish a full-pipeline or full-four-tile speedup. Existing large runs do not include this new patch. For the next full validation, rebuild the image from the updated source.

## Commit isolation

The focused optimization commit includes the ownership-region cache interface and write-pass indexing helper it needs, while preserving the existing committed core-filter and whole-instance-owner APIs. Other pending pipeline, ID-compaction, CRS and remap changes remain uncommitted in the working copy. The exact focused commit snapshot independently passed **373 tests in 10.00 s** in the locked Docker runtime; the complete candidate working snapshot passed the 395 tests reported above. The ownership implementation used in the medium benchmark is identical in both snapshots.
