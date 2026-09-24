# RayCloudTools: core ownership and unmatched points

SmartTile keeps RCT tree IDs aligned with the accompanying tree and treeinfo
tables. When processing `PredInstance_RCT`, it transfers predictions to the
dense target tiles and filters whole instances by the selected ownership anchor
(`centroid` by default, calculated on the remapped points). An instance whose
anchor lies outside its core toward a declared neighboring tile is initially removed.
Retained instances keep their membership and per-point attributes. Their IDs
are encoded with a reversible tile prefix. RCT processing skips cross-tile
reconciliation, point deduplication, and membership changes. It can recover
whole rejected trees under the strict ownership check below.

## Recovery without ownership conflicts

A rejected RCT instance is eligible when it supplies unsupported tree geometry
inside the source or overlapping neighboring cores. Recover it only if **none
of its points is within 0.01 m XYZ of any retained or already recovered tree**.
Check the entire instance, including its buffer tails. One competing positive
tree point blocks recovery of the whole instance; background does not block it.
The identity check uses both tile and local ID, so equal local IDs in different
tiles still identify different trees.

Candidates use the normal recovery ranking: most distinct unsupported core
samples first, then stable source-tile order and local ID. A recovered tree
becomes a blocker for subsequent candidates. Removed instances do not block
each other unless one is recovered. Recovery keeps the full instance geometry,
attributes and tile-prefixed ID, and restores its row in both tree tables.
It does not merge trees or transfer points between trees. Missing neighbors in
a partial collection do not make their cores eligible for recovery.

The report's `orphan_recovery` section records admissions, the conflict radius,
blocked candidates and a competing tile/instance with an example coordinate.
Recovered geometry is validated after output encoding. The 488/94 example
below remains blocked because both candidates overlap retained trees.

## Tile-prefixed IDs

Every retained positive RCT ID is written as `tile_id * 100000 + local_id` in
both the filtered LAZ and the two tree tables. `tile_id` is the one-based position
in the tile-bounds JSON; the report's processing `tile` index remains zero-based.
A subset uses the original layout position. For example, local ID 78 becomes
100078 in tile 1 and 200078 in tile 2, so it remains distinguishable after remap
to one original cloud. Decode a positive ID with integer division by 100000
for the tile, and remainder modulo 100000 for the original local ID.

Zero remains background. Positive local IDs must be 1 through 99999; larger
local IDs and encoded values exceeding uint32 fail before publication. The
fixed stride is never silently changed. Encoded LAZ dimensions use uint32.

Both `*_trees.txt` and `*_trees_info.txt` are filtered to retained IDs. Their
leading `predinstance` column contains the same encoded ID as the corresponding
LAZ. `instance_metadata.csv` records `tile`, `tile_id`, `local_instance_id`,
`PredInstance_RCT`, and `has_added_clusters`. Background 0 creates no tree row.
Missing rows, duplicate explicit IDs, or different ID sets in the two tables
fail validation. Original RCT tables without an explicit `predinstance` column
use their one-based row positions as local IDs.

Each output tile carries a versioned namespace VLR (`3DTrees`, record 24002),
containing the stride, tile ID and original source filename. The merge manifest
and report also record the mapping. A later filter pass uses this metadata to
preserve encoded IDs and uses explicit table IDs, so gaps do not shift rows and
an offset is never added twice. Co-locate the tree tables with the LAZ inputs
for another merge/filter pass. Unfiltered baseline files retain source labels;
they supply geometry for coverage validation, not final output IDs.

Standalone remap accepts the encoded tiles and transfers their IDs unchanged.
It checks collection namespace headers first and validates labels during the
indexing read, avoiding a second decompression pass over the prediction files.
A legacy collection of multiple unencoded RCT tiles is rejected with a request
to rerun merge/filter with its tree files. The intermediate merge task still
requires `--skip-merged-file`; original enrichment is the supported path to put
predictions from multiple RCT tiles into one original cloud.

## Why both tiles can remove the same points

An overlapping point can receive different tree assignments in two tile runs.
Each tile independently checks the centroid of the instance it predicted. If
both assigned instances have their centroids in their respective buffers, both
copies of the point are filtered out. This can occur while the main counterpart
of each tree is retained in its owning tile.

The 10-million-point local RCT validation on 2026-09-23 used 180 m cores with
20 m of buffer on each side, giving a 40 m overlap between adjacent tiles.
Inspection of its unfiltered dense tiles found these **local input IDs**,
before the tile-prefix encoding:

| Removed instance | Dense points | Main retained counterpart | Points shared with that counterpart |
| --- | ---: | --- | ---: |
| Tile 0, ID 488 | 25,851 | Tile 1, ID 120 | 25,391 |
| Tile 1, ID 94 | 16,876 | Tile 0, ID 433 | 14,727 |

Every coordinate of 488 was present in tile 1, and every coordinate of 94 was
present in tile 0. Most remaining points were assigned to other retained
instances; a small number were background. However, **443 exact XYZ positions
belonged to 488 in tile 0 and 94 in tile 1**. Both instances were removed:
488's centroid lay 6.18 m beyond tile 0's core, while 94's centroid lay 9.11 m
beyond tile 1's core in the opposite direction. Neither retained counterpart
claimed those 443 positions. With the new encoding, the retained counterparts
are 200120 (tile 1 in this zero-based report) and 100433 (tile 0).

![Top and side views of the RCT ownership disagreement](images/rct-instance-ownership.png)

Blue and orange follow the western and eastern tree across tiles. Pink shows
the 443 disputed positions; crosses mark the centroids. The dashed line is the
core boundary, and dotted lines show the overlap limits. All points of the four
shown instances are plotted; other neighboring instances are omitted. Z is
original elevation, not height above ground.

This example confirms complete spatial coverage for the inspected points while
showing disagreement about tree membership. Determinism alone does not imply
identical assignments when the two buffered input clouds differ.

## Final remap and background 0

When a positive RCT tree and background both lie within the final remap radius,
choose the nearest positive tree and keep its attributes. Use background only
when no positive tree matches. This applies to serial and parallel enrichment;
filtered tile membership stays intact. The 1 cm recovery-conflict radius and
the resolution-dependent original-remap radius are separate checks.

The RCT final remap preserves original point geometry and source attributes.
For each original point:

1. Require a match in the unfiltered dense baseline within the final remap
   radius. This remains a 100% coverage requirement for every model.
2. Search the filtered predictions within the same radius. When a surviving
   prediction is found, copy its prediction values, including its retained ID
   and semantics.
3. If no surviving prediction is found, write `PredInstance_RCT=0` and zero
   every other selected RCT prediction field, including semantics. Keep the
   original point. These points mean background/no assigned tree in the output;
   the coverage report distinguishes fallback zeros from matched background.

The final radius defaults to `sqrt(3) * --resolution-1`: approximately
**0.01732 m (1.732 cm / 17.32 mm)** for a 0.01 m first-stage resolution.
`--remap-tolerance` overrides it in meters. This differs from the earlier
coarse-model-to-dense transfer tolerance, whose default is **0.1732 m
(17.32 cm)**, controlled by `--prediction-transfer-tolerance`.

The 443 rejected dense positions are not a count of final background originals.
Original and dense point counts can differ, and a removed position can still
find a nearby surviving prediction inside the final radius. Only failed final
matches receive fallback 0. This exception applies to RCT; other models still
require complete final coverage. Missing baseline coverage remains an error.

The coverage report records `original_radius_m`, per-file `original_coverage`
entries for `unfiltered_1cm` and `final_survivors`, the final
`unmatched_policy: background_zero`, `background_assigned`, and the aggregate
`background_assigned_points`. Ownership decisions are recorded under
`models[].instance_ownership.tiles[].instances` in the merge report.

## Regression coverage

`tests/test_raycloud_filter_only.py` covers tile-prefixed IDs and uint32 limits,
subset stability, repeat filtering without double offsets, explicit sidecar ID
validation, both tree tables, missing sidecars, baseline coverage failure, and
zeroed instance and semantic fields for unmatched originals. Its
`test_shared_point_rejected_by_both_tiles_becomes_background` case models the
opposite-core ownership disagreement above: both tiles contain the same point,
both assigned instances are removed, both main trees survive, and final remap
writes background 0 only for the point with no surviving match. A nearby
original point still receives the retained tree's ID and semantics.
