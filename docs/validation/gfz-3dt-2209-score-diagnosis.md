# GFZ single-score discrepancy: confirmed source records (28 September 2026)

Affected original: `398720_5646180.laz`, zero-based row 2249272.
XYZ at the stored 1 mm resolution: (398735.321, 5646187.674, 411.103).
Both surviving FM records have instance 65 and semantic 2.

| Source | Zero-based row | Score | Computed distance to original in combined frame (m) |
|---|---:|---:|---:|
| tile_00001.laz | 3876910 | 0.6418741345405579 | 4.6192952765852895e-10 |
| tile_00002.laz | 2813136 | 0.8754913806915283 | 4.619277513099283e-10 |

At decimal LAS scale/offset precision these are the same location; their local
float64 Y values differ by one ULP (1.7763568394002505e-15 m). Their ~4.62e-10 m
distance to the original also reflects different LAS offset representations,
not meaningful spatial separation. A third nearby point is 0.022561028 m away,
outside the sqrt(3)*0.01 m remap radius.

## Minimal replay

`replay.py` loads the three extracted records, preserving headers and original
row indices, and runs each image's real PointIndex.nearest implementation.
The baseline union1 image selects tile 2/row 2813136 with score 0.87549138 in
both original-enrichment coordinate frames. The candidate 2209 image selects
tile 1/row 3876910 with score 0.64187413 in both frames, with spatial caching
both enabled and disabled. This reproduces the saved full-run score mismatch.

The older implementation compares distances exactly. The candidate treats
numerical differences within its 8-ULP budget as ties and chooses stable
(tile, source-row) order. Thus tile 1 wins the remap tie. Scores are copied from
source records, never averaged or recomputed. Model inference was not rerun.

## Underlying overlap-boundary defect

`boundary.py` uses the actual dense-file headers, tile layout and candidate
ownership/overlap functions. The tile-1/tile-2 overlap begins at local Y=10.0.
The tile 1 record becomes Y=9.999999999999998 and fails `inside_xy`; the tile 2
record becomes Y=10.0 and passes. Expanding only the lower bound outward by one
ULP admits both records. The exact-bound comparison therefore lets the first
copy evade shared-point ownership and deduplication. Both old and new filtered
clouds contain it; this is not a changed instance merge or a changed tree ID.

The point is in tile 2's core (distance 0); tile 1's core is 23.26894 m away.
Under the intended nearest-core ownership rule, tile 2 should supply its
attributes, including confidence 0.8754913806915283. The old final remap happened
to choose that record due to numerical distance; it did not repair the duplicate.

## Recommended correction and limits

Use a consistent outward numerical tolerance for declared overlap membership,
without expanding the geometric matching radius. Add this real-header boundary
case as a regression through shared-point ownership/deduplication, then rerun
full GFZ filtering and remap. Do not revert stable nearest-neighbor ties or
force scores to match a baseline. No production-code changes or full rerun were
performed during this diagnostic. Whether additional boundary records change
must be measured by that validation.

Evidence: neighbors.json, baseline_replay.json, candidate_replay.json,
boundary.json, query.las, near_*.las and near_*_indices.json in this directory.

## Boundary correction implemented — 28 September 2026

`bounded_point_index.inside_xy` now expands inclusive overlap bounds by eight
float64 ULPs per XY axis. The allowance depends only on the fixed local bounds,
so query batching, source height and cache configuration cannot change it.
`spatial_query_cache.region_entry` applies that same helper to source records.
Core ownership and geometric matching radii are unchanged.

Seven new regressions failed in the previous image. After the fix, all 35 targeted
ownership, cache, nearest-roundoff and merged-group tests pass. The real GFZ
record pair now retains only tile 2's record and remaps score 0.8754913806915283,
instance 65 and semantic 2 with caching on/off and 1/4 query workers. Additional
cases cover all four XY bounds, empty queries, batch independence, high-Z input,
and exclusion of records 0.1 mm outside the overlap.

Candidate image: `smarttile:v2.4a-2209-boundary`. A fresh full GFZ run is queued
behind the packaged suite: both models' four buffered clouds, original remap,
an explicit check of the affected point, and all 16 output comparisons.
Only the two source modules above differ from the preceding tested image.
Run evidence is in the sibling `fix-3dt-2209-boundary-20260928` directory.
Previously escaped duplicates may now be removed from filtered tiles; comparisons
remain strict and every difference will be reported for evaluation. Full-data
validation is pending.

Packaged result: **402 tests passed in 18.66 s**. Image source hashes verified;
immutable image ID: `sha256:c19a47678e04024e0c2c9859cbd719f0d72e1dee1d1bb2f4d64badf3b3106ef0`.
Full GFZ SAT + ForestMamba processing is now running; final remap, affected-point
verification and complete output comparison follow sequentially.
