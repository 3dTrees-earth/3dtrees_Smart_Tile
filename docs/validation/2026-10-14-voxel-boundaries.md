# Exact voxel boundaries — SmartTile local validation

Issue: [3DT-2391](https://linear.app/geosense-ufr/issue/3DT-2391).

## Fix

Voxel grouping now uses LAS integer XYZ and the decimal scale, offset and
resolution metadata directly. Checked integer arithmetic computes a world-zero,
half-open voxel grid; unusual encodings use an exact integer fallback. Genuine
points below a boundary are not snapped into the next voxel. Negative values
use floor division. Keys outside int64 fail clearly.

Spatial spill routing, block splitting, both spatial reduction methods,
whole-instance reduction and merged-product sampling share the calculation.
Merged-product chunk edges advance by exact grid cells, avoiding cumulative
floating drift and the reproduced gap at X=0.3 m in a ten-chunk input.

## Validation

**Final image: 1,016 tests and 94 subtests passed** (1,110 JUnit cases), zero
failures/errors/skips, in 142.73 seconds. The full suite ran without network
access, with 10 CPUs, 50 GiB RAM and SSD scratch. All 57 source files match the
image below and source commit `a539134`. Reports: `final-full-suite.xml` and
`final-full-suite.log` under `/tmp/smarttile-premerge-20261015`; checksums are in
the accompanying JSON record.

The earlier targeted suite passed **515 tests**, zero failures/errors/skips, in
the existing local scientific environment (Python 3.11, NumPy 2.3.5, laspy 2.6.1,
pytest 8.3.5).

The 20-point fixture from the earlier Galaxy failure now produces **4 SAT and
2 ForestMamba representatives at 10 cm**; ordinary spatial sampling produces 2.
Original resolution retains all 20 points per instance dimension. Real output
checks cover IDs, source attributes, missing-value metadata, CRS and original
point multiplicity. Reversing input order, splitting spill blocks, changing
chunk sizes and changing worker counts preserve results. Existing LAS/LAZ/COPC
and PLY output regressions also pass.

Independent rational expectations cover every integer centimetre resolution
from 1 through 100, negative and projected coordinates, fractional offsets,
points immediately below/on/above boundaries, equivalent LAS encodings,
anisotropic scales and overflow handling. The former float formula independently
reproduces 3 FM groups, while exact grouping yields 2.

Commands and test reports are under `/tmp/smarttile-voxel-boundary-20261014`:
`retile-focused-tests.xml`, `retile-focused-tests.log`,
`original-arithmetic-reproduction.json`, and `source-manifest.json`.
The source hashes are also embedded in the accompanying JSON record.

## Image and workflow scope

The local runtime recovered on 2026-10-15. The corrected image is now built and
active in local Galaxy, with all 57 embedded Python files matching the reviewed
source:
`sha256:41cd03e0f991669efdac8b0128ed53fc31fdae0f9f56528c48526b207ccbf018`.

The independent live Galaxy replay passed on wrapper `2.4.0+galaxy4`: exactly
four collections; SAT 4 / ForestMamba 2 representatives at 10 cm; all 20 original
points per dimension; retained IDs/attributes, CRS and missing-value declarations.
History `924bf55bd1f85561`, job `f87a121a672694df`. The original failed-image
record and interrupted build logs remain preserved. New evidence is under
`/tmp/smarttile-premerge-20261015`, including `galaxy-final-oracle/evidence.json` and
`final-image-source-check.json`.

These local tests do not replace the full release canary or production
promotion under 3DT-1903. No new full GFZ inference/remap run is claimed.
