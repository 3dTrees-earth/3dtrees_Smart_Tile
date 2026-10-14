# Shared tile outputs and original-resolution remap — local validation

Validated on 2026-10-14 with wrapper `2.4.0+galaxy3` and local image
`sha256:96f1279650c537461777bb908ddd6876c0828be9cbf92373322b6ca721db4775`.
All 56 embedded Python source files match the working source. This records local
acceptance, not a published release image or deployed production workflow.

## Behavior

Spatial and whole-instance Tile use the same repeatable Output controls.
Each entry selects Subsampled with a voxel size, or Original resolution, plus
LAZ, COPC LAZ or binary PLY. Original resolution retains duplicate points and
preserves compatible source coordinate grids. Whole-instance output retains its
ID and selected attributes, with one owner tile per tree. PLY checks scalar
integer ranges and cannot represent LAS missing-value declarations.

Remap has a matching Prediction resolution → Original resolution selection.
It uses each prediction collection's LAS encoding precision, requires complete
coverage for every model, and preserves original fields and record order.
Same-instance mode restricts species transfer to the existing tree ID.
Subsampled behavior and the legacy multi-dimension instance CLI remain supported.

## Results

| Check | Result |
| --- | --- |
| Complete SmartTile scientific suite | 553 passed, zero failures/errors/skips, 170.60 seconds. |
| Real Galaxy/Planemo wrapper suite | All 21 cases passed across the five tasks: 17 in the initial run, plus four targeted reruns after the label correction. |
| Linked Galaxy original COPC → same-instance Remap | Both jobs passed; all 20 points, original fields/order, species, CRS and missing declarations checked. Remapped LAZ byte-matches the aligned fixture. |
| GFZ SAT and ForestMamba original-resolution round trips | 867,599 points per model, two COPC tiles each, 100% remap coverage and zero species/probability mismatches. |
| GFZ instance ownership | SAT 12 and FM 17 positive instances; none appears in multiple tiles. |
| GFZ original preservation | All 36 original fields, coordinates, record order, CRS and ExtraBytes declarations unchanged; source checksums unchanged. |
| Native Galaxy collection labels | 54 cases, 224 rendered labels and 124 filtered labels passed; covers all six slots in both tiling modes. |
| Packaging | XML lint and offline ToolShed package lint passed. Published-version lookup was not performed because ToolShed DNS was unavailable. |

GFZ validation reused existing segmentation and species results; model inference
was not rerun. Four CLI commands completed in 91.87 seconds, excluding container
startup. The matched-coordinate allowance was approximately 0.866 mm for these
1 mm-encoded files. Galaxy jobs used the rebuilt image without a source override;
the GFZ run used the identical source in the accepted scientific runtime.

## Reproduction and boundaries

Run the scientific suite with `pytest -q -p no:cacheprovider
-o faulthandler_timeout=300 tests` in the scientific image, with the offline test
dependencies available. The successful run used 10 CPUs, a 50 GiB container cap,
no network and 8 GiB RAM-backed `/tmp`, matching the prior acceptance setup.
These are test-environment limits, not a SmartTile internal memory cap.

Run `planemo test` against an isolated local Galaxy configured for the exact
candidate image. Run `planemo lint smart_tile.xml` and, for offline packaging,
`planemo shed_lint --tools --skip version_bumped` on the wrapper directory.
The new aligned species fixture has a reproducible generator beside it.

Initial Galaxy submissions exposed a missing Cheetah variable marker in output
labels. Four Planemo cases hit that error before the watched reload; their targeted
rerun passes after the correction. The final result matrix uses the latest result
per test ID and keeps both raw reports. The negative label test reproduces the old
error. A GFZ control exposed metadata loss in fixture preparation, corrected
before the final replay. An incomplete full-suite run stalled on Docker overlay
SQLite journal writes; the successful rerun used the baseline RAM-backed scratch.
These controls remain separate from passing evidence.

Raw reports are retained under
`/tmp/smarttile-original-resolution-validation-20261014` and
`/tmp/smarttile-original-resolution-gfz-20261014/validated`.
The adjacent JSON records source/report hashes and measured results.

Whole-instance Galaxy workflows must bind `output_subsampled_res1` … `res6`
instead of the former `output_instance_tiles`. Existing workflow parameters must
also bind the nested resolution choices. Immutable image publication, production
workflow rebinding and the coordinated release canary remain release work.
