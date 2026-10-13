# Combined SmartTile v2.4 branch validation — 2026-10-13

## Integration

The working branch is `codex/3dt-2101-remap-first`. It includes the earlier
CRS fix (`29ff07d`), full-resolution products and attribute joins (`7e75ef7`),
and instance tiles with same-instance remapping (`6e5178e`). Instance tiling
remains CLI-only. The related Galaxy wrapper branch is
`claude/smarttile-v2.4-wrapper`, including `cdfe22ca`.

The combined patch removes the unused tiling/subsampling path, bounds dense
original-remap batches, restores application property names in PLY exports,
preserves merge/filter output-directory conventions, and makes subsampled
product selection and equal-GPS-time ordering deterministic. PLY fields whose
names collide after lowercasing are rejected, including `treeID` with `treeid`.

## Final tool suite

The final source was copied to a fixed snapshot, mounted read-only in the
existing SmartTile scientific-runtime container, and tested without network
access. Limits: 10 CPUs, 50 GiB memory including swap, and an 8 GiB temporary
filesystem for test scratch. The pytest dependencies were also mounted read-only.

```sh
python -m pytest -q -p no:cacheprovider -o faulthandler_timeout=300 tests
```

**392 passed, 91 subtests passed**, exit 0, in 141.87 seconds. This includes the
new lowercase PLY-name collision cases and deterministic product-order tests.
An earlier run using the container's ordinary temporary filesystem was stopped
while waiting on filesystem I/O; that incomplete run is not validation evidence.
`git diff --check` also passes.

## Dataset 645 reproducibility

Two local builds of the 10 cm LAZ product, using the same existing enriched
originals and 10 workers, produced:

- 80,036,292 selected points;
- 808,809,039 bytes in each LAZ;
- identical SHA-256:
  `3a76d350b5a028e550a4bf9b1edd24958877c3793bb8b4376189ceace5ef7947`.

Each build completed in about 9–10 minutes; measured peak memory was 14.34 GiB
and 13.39 GiB. Before the equal-GPS-time tie fix, the point sets matched but
record order and compressed bytes differed.

This establishes repeated LAZ byte identity for the tested inputs, runtime,
and chunk configuration. It does not certify byte identity across different
COPC encoders, runtime versions, or chunk layouts. Product chunk planning
uses an average point-count target, not a hard density-independent memory cap.

## Release boundary

These are local integration and validation results. They do not establish a
production deployment, recovery of individual production datasets, or completion
of the wider SmartTile v2.4 release certification.

## Review fixes — 2026-10-13

Five reviewed data-safety defects are fixed on top of `7f496bb`:

- Instance tiling checks the complete input collection before processing and
  promotes compatible integer ExtraBytes without truncation. Regressions cover
  ID 70000 with uint16/uint32 inputs in both file orders, carried attributes,
  and incompatible metadata.
- Remap scratch files use internal identifiers instead of attribute names.
  Instance tile directories and summary filenames encode attribute names;
  absolute paths and traversal names cannot select files outside the job.
- Product and tile publication share destination validation. Input aliases,
  duplicate destinations, and existing results are rejected before writing.
  Products, including vector sidecars, are staged and rolled back on ordinary
  publication failure; tile jobs publish their complete directory only after
  all requested outputs and layout files succeed. This is not a power-loss or
  process-crash durability guarantee. Repeated builds need fresh destinations.
- Disabling OriginalPointIndex generation preserves existing values.
- Full-resolution merge and attribute join retain ExtraBytes scales, offsets,
  and no_data through the canonical metadata helpers, rejecting incompatible
  shared-field definitions.

The final source and regression tests passed the complete local tool suite:
**429 passed, 91 subtests passed**, exit 0, in 117.52 seconds. The command was the
same as above, with JUnit output enabled. The scientific-runtime image was
`smarttile:v2.4a-fast`, image ID
`sha256:b8c79e45dc52a363f8273f83a27b57c9954e8befaa8ea2fe20883bbfff119a52`.
Source and test dependencies were mounted read-only; networking was disabled;
the limits remained 10 CPUs, 50 GiB memory including swap, and 8 GiB scratch.
Regressions include input symlink/hardlink aliases, concurrent destination
creation, late publication failure, and failure after ordinary tiles but
before instance tiles complete. `git diff --check` passes.

The unchanged Galaxy wrapper at `d51eeb2b` also passed all **13 Planemo cases**
against this fixed source: 0 errors, 0 failures, 0 skips, exit 0. The final report
confirms output assertions for four tile cases, three merge cases, four product
exports (full-resolution LAZ, subsampled COPC and PLY), and two remap cases. Summed
test-case time was 557.79 seconds. This used an isolated local Galaxy instance
and disposable PostgreSQL database; tool jobs mounted the source read-only and
ran without network access, with 10 CPUs and a 50 GiB memory limit.

These fixes do not close the review's same-instance nearest-point tie issue:
numerically tied candidates can still be selected differently across prediction
batches. Dense spatial cells also lack a strict worst-case memory bound. Wider
downstream release certification remains incomplete, so this is not production
approval or certification of all three end-to-end scenarios.

## SmartTile-only scenario replays — 2026-10-13

The user clarified the three flows as:

1. Original → tile → SAT → merge → species → remap → visualization.
2. Original-with-predictions → tile + filter → species → remap → visualization.
3. Multiple segmentation outputs → merge + source-order index → ForestStructure.

This pass executes only SmartTile at commit `53ba60e`. Actual cached GFZ SAT,
ForestMamba, RCT and species predictions supply the external model boundaries;
no model inference, visualization, ForestStructure or production service runs.
All original data and source code are mounted read-only. Containers use the
same scientific runtime identified above, networking disabled, and explicit
CPU/memory caps. Machine-readable evidence is summarized in
[`2026-10-13-scenarios.json`](2026-10-13-scenarios.json).

| Scenario | Local fixture and checks | Result |
| --- | --- | --- |
| 1 | 553,030 real points in a 9 × 9 m crop crossing four historical tile cores; fresh tiling, cached SAT merge, cached species transfer, strict remap; workers 1 and 4 | Passed: 100% coverage, five compact SAT IDs, no conflicting positive geometry claims, identical final records across worker settings, retained source fields/order/CRS/ExtraBytes metadata. |
| 2, conventional tile/filter | Two adjoining enriched files, 867,599 points | Unsupported: ordinary nearest-to-centroid tiling omits instance fields; filter exits with `expected one model instance dimension, got []`. |
| 2, instance tiles | Same 867,599 points; SAT and FM at 1 cm and 10 cm; real species annotations replayed onto selected source points | Passed: 100% own-instance remap coverage, zero independent species ID/probability oracle mismatches, unchanged original coordinates/order/IDs/retained metadata. Each positive ID has one owner tile within the tested crop. |
| 3, full-resolution LAZ | Same two files, both a combined collection and three model-specific attribute views joined by SmartTile | Passed: all 867,599 points and fields preserved, uint64 `OriginalPointIndex` is the first ExtraByte with complete 0..N−1 values in source record order; source `point_source_id`, CRS and missing-value metadata retained. |
| 3, full-resolution COPC | Same inputs and attribute joins | Points, interpreted field values, CRS and index-based order recovery pass. Metadata fails: `Amplitude`, `Reflectance` and `Deviation` lose their missing-value declarations. |
| 3, full-resolution PLY | Default generated `OriginalPointIndex` | Fails: PDAL rejects `uint64_t`. Both combined requests leave no partially published products and inputs are unchanged. |
| 3, 10 cm products | LAZ, COPC and binary PLY, direct and attribute-joined inputs | 139,401 selected points in each format, matching geometry and prediction values; source-record membership and unique output voxels pass. LAZ and COPC lose the same three missing-value declarations. |

Scenario 1 took 202.49 seconds and observed 2.65 GiB container peak memory under
4 CPUs/20 GiB. The corrected scenario 2 harness took 84.86 seconds with the same
caps. Scenario 3's supported-format checks took 62.46 seconds under 2 CPUs/8 GiB,
with additional isolated failed-format probes. CLI exit success is not counted
as acceptance where independent metadata assertions fail.

Full COPC and 10 cm LAZ/COPC materialize the Amplitude/Reflectance scale into
floating-point values; interpreted values remain equal, so this representation
change is not reported as value corruption. Their missing-value declarations
are nevertheless absent (source sentinels: Amplitude/Reflectance 0, Deviation
−1). The tested files contain no actual sentinel-valued points in these fields;
the reproduced failure is loss of declarations. One-worker and two-worker
10 cm LAZ products have identical multisets across all 36 source fields and
point multiplicities. Physical record order differs as chunk planning changes
from one to two chunks; the existing byte-identity guarantee is scoped to the
same chunk configuration, so this ordering difference is not a new blocker.

The cached species values are replay fixtures, not fresh inference. Scenario 1
verifies that fresh and historical tile cores match and that species fixture
coordinates match within 1 micrometre (observed maximum 9.34e-10 m). Both crop
tests may truncate trees at omitted source boundaries; ownership claims concern
only the tested points. Model-view and species-removal fixtures preserve the raw
retained ExtraBytes descriptors and are checked against untouched sources. An
initial fixture preparation that lost missing-value declarations was corrected;
its metadata results are superseded.

### Tie and density stress results

- Same-instance remap remains batch-dependent for numerical nearest ties:
  identical two-point input selects species 10 in one insertion batch and 20
  in separate batches; a three-point case selects 20 versus 30. Ordinary nearest
  and exact-distance tie controls remain stable. Cache on/off and query worker
  settings do not remove the defect.
- The dense tiler accepts an 8-million-point reducer leaf despite a configured
  4-million-point target. Peak process RSS rises from 0.65 GiB at 4 million to
  1.86 GiB at 8 million. Point accounting and the 400 output voxel coordinates
  remain correct, but the target is not a hard memory bound.
- Product selection reaches 0.83 GiB peak process RSS at 4 million concentrated
  input points, with identical selected records across the density ladder.
- Dense original-remap streaming handles 9 million records exactly once, in
  order, in batches of at most 32,768, at 0.19 GiB peak process RSS.
- A bounded systematic sample of local dataset 483 found 196,608 of 524,288
  sampled records in one 0.1 m XY cell. This demonstrates a relevant local dense
  case; it is not an exact full-dataset population estimate.

All ten isolated stress processes completed under 2 CPUs/8 GiB without OOM or
timeout. Successful stress execution does not establish a density-independent
memory guarantee. The SmartTile-only readiness gate remains open because of the
unsupported conventional scenario 2 route, export failures/metadata loss, and
the reproduced numerical-tie and dense-cell resource gaps. Source code was not
changed during this validation pass.
