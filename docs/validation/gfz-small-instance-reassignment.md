# GFZ: small-instance reassignment and instance summaries (2026-09-29)

Candidate: `smarttile:v2.4a-reassign-xy` (sha256:da6406e452cb47d51622af939ef08908e0aa9c34581d3965d421d132700eeedb), built from this branch.
Resources: 10 CPUs / 50 GiB per container, no GPU, local workstation only.

## Rule

After deduplication on the final 1 cm tiles, an instance with fewer than
`--max-cluster-size` points (3000) and an axis-aligned bounding box below
`--max-volume-for-merge` (4 m3) takes the ID of the non-small instance with the
nearest **XY** centroid (horizontal distance, height ignored) within 5 m.
Statistics are collected during deduplication; tiles are relabelled in one pass;
the final label index is built once, only when in-task original enrichment needs it.

## Run

The `filter` task was rerun with `--reassign-small-instances` on the validated
two-tile GFZ dense inputs (SAT from the three-model run, ForestMamba from the
two-model run; identical tile bounds). Outputs were compared with the validated
filter outputs without reassignment (`compare.py` in the run directory).

| | SAT | ForestMamba |
|---|---:|---:|
| Instances | 98 | 114 |
| Small (< 3000 pts and bbox < 4 m3) | 6 | 3 |
| Reassigned | 6 (6,986 points) | 3 (1,187 points) |
| Kept small | 0 | 0 |
| Reassignment stage | 5.2 s | 6.1 s |

SAT: 17 → 16 (0.44 m), 39 → 22 (2.22 m), 62 → 79 (4.50 m), 74 → 44 (1.42 m),
77 → 79 (0.72 m), 82 → 83 (0.26 m). ForestMamba: 5 → 6 (0.91 m), 32 → 48 (1.19 m),
38 → 39 (2.48 m). Distances are horizontal, between XY centroids.

Checks passed for both models: identical point counts; identical coordinates and
all non-instance dimensions; every changed point belongs to a reassigned source
ID and carries its planned target; no reassigned ID survives; changed point
counts equal the reported counts. `PredInstance_<model>_summary.json` lists every
final instance with `reassigned_from`.

An earlier run with 3D centroid distance left 5 of 9 small instances unassigned
(centroids of crowns lie high above ground-level fragments); the XY rule replaced it.

Known behaviour: a fragment goes to the tree whose crown centre is above or next
to it. SAT 82 touches the stem of tree 69 (0.01 m) but goes to tree 83, whose
XY centroid is 0.26 m away; SAT 17 and 77 go to sparse high instances (16, 79).

Evidence: `/mnt/ssds/kg281/smarttile-v2.4a-gfz-reassign-20260929/` (commands, logs, reports, comparison JSON, plots).
