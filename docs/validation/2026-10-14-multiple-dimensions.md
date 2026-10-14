# Multiple instance dimensions — local validation

Validated on 2026-10-14 with wrapper `2.4.0+galaxy4` and local image
`sha256:cc3e4b7245a480a78c4f06daf1494db5fba7b2d636e7e4e5cee9c47a48534a2e`.
All 56 embedded Python source files match the working source.

## Behavior

Whole-instance Tile accepts up to six comma-separated instance dimensions, for
example `PredInstance_SAT,PredInstance_FM`, through the shared Output controls.
Each dimension and resolution produces its own top-level list collection.
Two fields with Original resolution and 10 cm produce four collections.
The first field preserves `output_subsampled_res1` … `output_subsampled_res6`;
additional fields use `output_instance_dim2_res1`, and so on. Collection labels
include the field and resolution. Spatial outputs retain their existing names.
Empty and duplicate field selections fail validation.

## Checks

- 569 scientific tests passed, zero failures/errors/skips.
- Six targeted live Galaxy cases passed: spatial, single-dimension and
  multi-dimension tiling; sampled/original selections; LAZ/COPC/PLY.
- The independent live Galaxy oracle found exactly four list collections.
  Original COPC retained all 20 fixture points for each field, with exact
  coordinate/attribute inventories. At 10 cm, SAT retained four representatives
  and ForestMamba three. The independent ideal-grid check expected two FM points
  and failed: the existing floating-point voxel calculation routes X=500001.1
  just below a 10 cm boundary. This limitation was unresolved in that image. The subsequent
  [source fix and focused validation](2026-10-14-voxel-boundaries.md) correct it;
  this historical image result is preserved.
  The failed check is retained in the evidence; this is not a full numeric pass.
  IDs, CRS and Amplitude missing-value declarations were preserved; neither
  collection accidentally carried the other model's ID field.
- Native Galaxy validation passed 189 cases, rendering
  2464 labels and filtering 4484 unused outputs,
  across all six dimension and six resolution slots; 16 input-validation cases passed.
- Planemo XML/schema lint and independent code review passed.

Artifacts: `/tmp/smarttile-multi-dimensions-20261014`.
This is local validation of the extension, not certification of a published
release image or production workflow. Prior original-resolution/GFZ results
remain frozen in their separate validation record and were not rerun here.
