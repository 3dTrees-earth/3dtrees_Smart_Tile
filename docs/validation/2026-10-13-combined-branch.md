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
| 2, control without `--instance-tiles` | Two adjoining enriched files, 867,599 points | Ordinary nearest-to-centroid tiling omits instance fields; filter exits with `expected one model instance dimension, got []`. This control did not enable the existing instance-preserving flag and is not a failure of the requested mode. |
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

For scenario 2, the existing `--instance-tiles PredInstance_SAT,PredInstance_FM`
mode carries each required instance attribute automatically, regardless of the
optional `--instance-tile-dims` allowlist. It also places each whole instance in
one owner tile, so a separate buffer-filter task is unnecessary. The successful
instance-tile replay above is the applicable SmartTile path for the user's
clarified scenario. The earlier classification of ordinary tiling without this
flag as a scenario 2 blocker was too broad.

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
memory guarantee. The SmartTile-only readiness gate remains open because of
export failures/metadata loss and the reproduced numerical-tie and dense-cell
resource gaps. Source code was not
changed during this validation pass.


### Required instance attribute follow-up

The existing `--instance-tiles` implementation already includes the selected
instance field automatically; no tiling or remapping algorithm change was needed.
CLI help, dimension logging and the README now make this requirement explicit.
The scenario 2 interpretation above is corrected accordingly.

A fresh local replay of the same 867,599-point GFZ crop passed for SAT and FM at
1 cm and 10 cm, both with no optional attributes and with intensity/Amplitude
plus simultaneous ordinary tile outputs. Every selected record has its source
coordinates and instance ID, all source labels are represented, and each positive
ID has exactly one owner tile. The ordinary outputs retain their separate
geometry-only contract. Both CLI runs exited 0 under 4 CPUs/20 GiB with no
network; source code and original files remained unchanged.

The targeted instance, parameter and tile-publication suite passed **38 tests
and 16 subtests** in 6.64 seconds. Three CLI-parsing regression cases cover
omitted/empty/selected optional attributes and coexistence with ordinary outputs.
The tests ran under 4 CPUs/12 GiB with read-only source and no network. Commands
and artifact hashes are recorded in `2026-10-13-scenarios.json`.

## Full GFZ filtering and instance tiles — 2026-10-13

The uncropped GFZ runs at source commit `39b8791` close the missing full-size
test coverage, but **standalone buffered-tile filtering fails independent
acceptance for both SAT and FM**. The separate `--instance-tiles` route passes.
No SmartTile algorithm was changed during this validation pass.

All eight original files contain **16,426,961 points**. The complete historical
GFZ cache supplies four dense 1 cm tiles per model, **53,265,082 buffered records
per model**, with matching 60 m cores and 20 m buffers. Both real SAT and FM
predictions were reused; no inference or production services ran. The historical
dense geometry uses center-of-mass voxels, so original coverage is checked at
the 1 cm voxel diagonal rather than requiring original-coordinate identity.
The incomplete October 12 dense cache was explicitly excluded.

Each filter ran in the previously identified scientific runtime with 8 CPUs,
32 GiB memory including swap, SSD scratch, read-only sources/inputs, and network
disabled. The two filter jobs ran concurrently. CPU measurements include worker
processes; scratch peaks are sampled every 10 seconds.

| Standalone filter | SAT | FM |
| --- | ---: | ---: |
| CLI result | Exit 0 | Exit 0 |
| Final points | 15,916,646 | 15,916,718 |
| Positive instance IDs | 100 | 114 |
| CLI wall time | 607.68 s | 908.31 s |
| Cgroup CPU time | 1,658.81 s | 2,181.96 s |
| Peak cgroup memory | 7.17 GiB | 7.93 GiB |
| Sampled scratch peak | 3.61 GiB | 4.45 GiB |
| Missing original points within 17.321 mm | 0 | 0 |
| Maximum original-to-output distance | 11.314 mm | 14.457 mm |
| Exact coordinate locations with conflicting positive IDs | **30** | **1** |
| Independent acceptance | **Failed** | **Failed** |

Independent validation uses laspy, NumPy and SciPy without SmartTile imports.
It checks every output record and its multiplicity against the cached source,
including all retained attributes and the declared instance-ID mapping. All
metadata/CRS checks, stage accounting, reconciliation graph checks, one output
file per positive ID, and exhaustive original coverage passed. All 36,256,804 SAT
and 37,148,305 FM positive dense core records have positive final geometry within
1 cm, plus the documented 1e-8 m numerical allowance. Spatial matching scores and
biological segmentation quality are not independently certified.

### Blocking outer-boundary defect

The conflicting points have bit-identical decoded XYZ and distinct positive IDs
from different source tiles. They are neither quantization aliases introduced by
the validator nor existing same-tile label ambiguity. Both records survive while
the CLI reports no conflicts. For example, SAT retains IDs 66 and 94 at
`(398739.999, 5646206.642, 409.683)`.

The overlap bound conversion in `src/remap_first_pipeline.py:299` yields local
upper X `46.570999999996275`, whereas the LAS integer-coordinate conversion yields
`46.571`. The difference, `3.723e-12` m, exceeds `inside_xy`'s local-coordinate
allowance of `5.684e-14` m (`src/bounded_point_index.py:511`). Consequently, valid
boundary points are excluded before ownership resolution and conflict checks.
A two-record reproduction detects the zero-distance conflict without the overlap
mask, misses it with the current bounds, and detects it with an experimental
guard derived from the global-coordinate precision. That experiment is diagnostic
evidence, **not an implemented or fully validated repair**. The bounds conversion
needs a regression-tested fix and both full GFZ filter runs must pass afterward.

### Whole-instance route

The full original-with-predictions collection was also tiled with
`--instance-tiles PredInstance_SAT,PredInstance_FM` at 1 cm and 10 cm, using 10 m
cores, 2 m buffers, and eleven optional source/model/species attributes. The
required instance attribute is carried automatically. All **34 independent
checks passed** across 26 occupied output tiles per collection:

- All selected records match source coordinates, instance labels and raw
  attribute values; CRS and missing-value declarations are preserved.
- All 100 SAT and 114 FM positive instances have one correct centroid owner.
- Every original point has retained geometry of the same instance within the
  respective voxel diagonal; zero uncovered points at either resolution.
- SAT selected 15,918,455 / 2,484,151 points at 1 / 10 cm; FM selected
  15,918,439 / 2,481,853 points. Subsampling preserves representation and ownership,
  not every original record.

The tile command took 68.00 s under 4 CPUs/20 GiB, with peak cgroup memory
2.91 GiB and sampled scratch 1.44 GiB. Exhaustive verification took a further
597.45 s. All original/source hashes remained unchanged. No completed SSD run
had an OOM or memory-limit event. Earlier HDD attempts were interrupted because
of storage stalls and are excluded from pass/fail and performance evidence.

Commands, checks, source/runtime identity, artifact paths and hashes are recorded
in [`2026-10-13-full-gfz.json`](2026-10-13-full-gfz.json). The original failed filter
reports and diagnostic reproductions are retained. These results do not change
the earlier remap/export defects or certify the three downstream workflows.
