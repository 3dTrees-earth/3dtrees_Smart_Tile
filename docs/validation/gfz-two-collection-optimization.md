# GFZ two-collection optimization validation

## Candidate

`smarttile:v2.4a-cache1`:
`sha256:808a1848cfa6901eb9d9fa210cb322504b6a0cd943eb75c33cbec86925292e44`.
Baseline: `smarttile:v2.4a-union1`.

SAT and ForestMamba are processed together through `merge_collections`, keeping
separate model schemas, reconciliation groups and final instance mappings.

Changes:

- Process-wide LRU cache of immutable 4 m XY-region trees, conservative byte
  accounting, default 512 MiB cap, cgroup/process-derived ceiling, oversized
  region disk fallback, and invalidation on index writes/close.
- Exactly preserve distance-boundary grouping, equal-distance source references,
  positive-only queries, overlap restrictions and per-tile filters.
- Index emitted dense/core-owned/shared-owner records during writing instead
  of reopening their LAZ outputs. References remain output-row positions.
- Record per-model phase times in merge reports.

## Gates

- 367 tests passed in the built image (11.09 s).
- An additional seeded randomized cache/disk equivalence and eviction test
  passed afterward; the focused cache suite contains six passing tests.
- No production inference checkpoint, ownership criterion, merge threshold,
  namespace rule or tree-file behavior changed.

## Real-data comparison

Run directory: `optimization-two-collections/` beneath the GFZ validation run.
The first fixture uses the same 8 m square from all four dense tiles and both
models (442,681 records per tile), with 275,253 originals in its inner 6 m square.
All original cores are retained; cropped bounds are recorded in fixture JSON.
Cropping changes whole-tree anchors, so this fixture establishes differential
behavior and timing, not full-dataset correctness.

The initial attempt stopped before processing because its fixture JSON retained
full-cloud bounds. Corrected fixture metadata and failed evidence are preserved;
no production geometry or layout was changed.

Old and new images run sequentially with 10 CPUs, 50 GiB each, as UID 1010/GID
1012, on read-only input mounts. Equality checks compare every output point field,
record count/order, coordinate scales and offsets for both collections and the
combined enriched originals. After that gate, the optimized image processes the
full GFZ inputs and compares all produced point dimensions against the existing
validated outputs (species and RCT fields are absent from this SAT/FM-only run).

The native v2.3 two-collection entry point is separately queued against full GFZ
inputs under `comparison-v23-two-collections/`; its parameters and results are
recorded there and do not contaminate the sequential small-fixture timings.

Results are recorded in the run manifests, `outputs/equality_subset.json`,
`outputs/equality_full.json`, per-model `phase_seconds`, and resource logs.
Full validation is pending until those artifacts report success.

## Completed subset comparison

Both versions processed SAT and ForestMamba in one invocation. All nine output
clouds passed exact field, count/order, scale and offset equality (1,160,523
output records including enriched originals).

| Measurement | union1 baseline | cache1 candidate |
|---|---:|---:|
| Container wall time | 173.205 s | 106.086 s |
| Cgroup CPU time | 220.934 s | 139.553 s |
| Observed cgroup peak | 723.44 MiB | 808.64 MiB |

Measured speedup: 1.63x (38.8% lower elapsed time), 36.8% lower CPU time,
with approximately 85.2 MiB additional peak memory. These are small real-fixture
measurements; full GFZ validation remains separately tracked.

## Native v2.3 joint result

Image: `ghcr.io/3dtrees-earth/3dtrees_smart_tile:2.3`.
Full saved GFZ dense collections, original tiles and eight original files were
mounted read-only. Ran `run.py --task merge --segmented-folders SAT,FM` with
`--instance-dimension PredInstance_SAT`, 10 requested workers, 20 m buffer,
and enforced Docker limits of 10 CPUs / 50 GiB, as UID 1010/GID 1012.

Result: failed after 186.2 s; observed cgroup peak
42.11 GiB. All four dense retile targets
had 13,630 points without a match within 0.1 m. These are overlapping tile counts,
not a sum of unique missing original points. Original remap was never reached.
No retry was performed.

This legacy multi-collection entry point first remaps additional model fields
onto SAT reference geometry, then filters/reconciles the selected SAT instance
field. FM IDs are passenger attributes in this merge. It is not equivalent to
the current independent per-model reconciliation in one run. Logs additionally
show legacy inner parallelism requested 128 workers despite the outer workers
argument; the actual CPU usage remained constrained by the 10-CPU container quota.

See `commands/v23_joint.json`, `logs/v23_joint.log`, resource JSONL and
`manifest.json` for the exact invocation and evidence. Do not compare this
failed-run time directly to a successful two-model pipeline's total time.
