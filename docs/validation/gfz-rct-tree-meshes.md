# GFZ: RCT QSM tree meshes through merge and final remap (3DT-2232, 2026-09-29)

Candidate: `smarttile:v2.4a-meshes` (sha256:2a2b0eda909a53a187fc099d51737c8f18ab2c07cd5f49ae74c20cc2b32d3d29).
RCT: release source v1.3.0 (`235fc9e`) with the natively rebuilt mesh-ID tools of the
validated 10M-point RCT run, on the two GFZ 10 cm tiles (same parameters as the earlier
GFZ RCT branch, treeinfo on). Local only; 10 CPUs / 50 GiB per container; no GPU.

## Result

| Tile | Source trees (all with faces) | Source faces | Retained trees | Removed trees | Filtered faces |
|---|---:|---:|---:|---:|---:|
| c00_r00 | 213 | 1,691,136 | 107 | 106 | 1,109,544 |
| c00_r01 | 277 | 1,448,340 | 202 | 75 | 806,508 |

Final remap: 309 dataset-wide trees over 8 originals; every original's
`<stem>_trees_mesh.ply` holds all its trees (6–109 per file, 114,624–738,624 faces,
108 MiB in total); 50 trees are shared by 2–4 originals and have the same ID and
identical geometry in each. Mesh filtering took 0.33 s for both tiles.

## Checks (all passed; `validate_meshes.py` in the run directory)

- Source: every mesh `tree_id` is a native tree-table row; `rct_tree_count` equals the rows.
- Merge: filtered mesh IDs equal the retained table IDs; every retained tree's faces
  (vertex XYZ and RGBA, face order) equal the source faces; removed trees are absent.
- Final: per original, mesh IDs = `_trees.txt` IDs = `_trees_info.txt` IDs = positive
  `PredInstance_RCT` values; every final tree's faces equal its source tree through
  `rct_instance_mapping.json` provenance; retained models are not clipped per original.

Terrain meshes are not processed. GLB packing is not part of this step.

Evidence: `/mnt/ssds/kg281/smarttile-v2.4a-gfz-rct-meshes-20260929/` (commands, logs, reports, `validation.json`).
