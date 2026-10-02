# GFZ: SmartTile v2.3 comparison and performance investigation

## Scope and provenance

Compared cached `ghcr.io/3dtrees-earth/3dtrees_smart_tile:2.3`
(`sha256:94791e7dfd2911333677e670d32c748149e1a17af21b3aa6d0a39c8e60901353`)
with `smarttile:v2.4a-union1`
(`sha256:459bce743eb21f5290053e9a8dfd97ee2831a15063fc38f472c7fe7855f75878`).
The v2.3 image's `run.py`, `merge_tiles.py`, and `merge_instance_matching.py`
SHA-256 values match the corresponding files at tag `2.3`, commit `62229a4`.

Both used the same saved dense predictions: four 60 m cores with 20 m buffers,
SAT, updated ForestMamba, and RCT, originating from 16,426,961 original points.
Overlapping dense tiles contain 53,265,082 records per model. Segmentation and
initial dense transfer were reused. Input files were mounted read-only and
aliased to the original grid names for v2.3. Tile metadata was unchanged.

Each container ran as UID 1010/GID 1012 with 10 CPUs and 50 GiB memory; three
comparison branches ran concurrently, using no GPU. Outputs were isolated.

This isolates post-segmentation behavior, not a replay of the entire historical
v2.3 workflow. SAT/FM used native `merge_tiles.py` defaults (30% overlap, 10 cm
correspondence grid, border matching and small-cluster reassignment), including
retiling and requested original remap. RCT used native `run.py --task filter`
with a 20 m buffer and paired tree files, preserving the no-merge requirement.
The versions perform different work; failed-run times are not successful
pipeline speed comparisons. No failed processing stage was retried.

## Results

| Branch | v2.3 outcome | Wall time to exit | Peak observed cgroup memory | v2.4a outcome |
|---|---|---:|---:|---|
| SAT | Failed dense retiling: 13,630 unmatched points per tile at 10 cm | 146.3 s | 8.13 GiB | Filter and full original remap passed |
| ForestMamba | Failed dense retiling: 210 unmatched points per tile at 10 cm | 150.8 s | 8.74 GiB | Filter and full original remap passed |
| RCT | Native filtering completed | 46.5 s | 1.68 GiB | Filter, DetailView, original remap and sidecar validation passed |

Counts per tile include repeated overlapping geometry and must not be summed
as unique original points. The independent diagnostic below queries every
original point against the preserved v2.3 merged cloud. It is not a successful
v2.3 final remap, which was never reached.

| Original-point diagnostic | SAT | ForestMamba |
|---|---:|---:|
| No v2.3 match within 10 cm | 14,100 | 211 |
| No v2.3 match within 17.32 mm | 14,986 | 303 |
| Maximum nearest distance | 1.707 m | 0.571 m |
| v2.3 merged instance count | 109 | 123 |
| v2.4a final instance count | 100 | 114 |

All SAT originals missing at 10 cm belong to v2.4a tree 47, whose group contains
source instances `(tile 0, 63)`, `(tile 1, 43)`, `(tile 2, 63)`. Tile 1 was the
normal retained owner; tiles 0 and 2 supplied recovered geometry. ForestMamba's
211 missing originals belong to tree 36, grouping `(0,40)`, `(1,31)`, `(3,27)`;
again tile 1 was retained and the other two members recovered. This gives a
concrete example of why retaining all accepted members' geometry matters.
The gaps are too large to explain as centimetre-scale rounding alone. This
comparison does not isolate which individual legacy filtering/recovery decision
caused each missing point.

For RCT, v2.3 retained 448 tile-local instances (65/97/108/178), versus 308
namespaced instances (50/71/55/132) after v2.4 filtering and safe recovery.
These counts alone do not measure biological accuracy. v2.3 prunes matching
sidecar rows but does not add explicit IDs; after pruning, row positions no
longer identify the unchanged cloud IDs. v2.4 explicitly records IDs and validates
both sidecars against the retained cloud. Native v2.3 RCT filtering success
therefore does not establish the current sidecar-identity contract.

## Resource investigation

Current v2.4 SAT filtering took 5,381.5 s with 6.00 GiB observed cgroup peak;
ForestMamba took 6,006.4 s with 7.46 GiB. CPU/wall ratios were approximately
1.18/1.17 cores despite a ten-CPU allowance. This indicates limited effective
parallelism; it does not by itself identify I/O versus CPU bottlenecks.

A bounded exploratory query fixture uses the first 524,288 real GFZ SAT dense
points and 131,072 nearby shifted queries, at a 5 cm radius. A cProfile trace
shows 788 small index queries and repeated Python/thread operations. Unprofiled
three-run medians prevent profiler overhead from being mistaken for a speedup:

| Variant | Query median |
|---|---:|
| Current disk-backed spatial batches, current thread policy | 0.662 s |
| Same queries, serial native calls | 0.660 s |
| Same queries, higher threading threshold (8192) | 0.658 s |
| One reusable in-memory tree | 0.0107 s |

The reusable tree needed an additional 0.134 s to build. All variants returned
identical distances, instance labels and source point references on this fixture.
The initial benchmark process peaked at approximately 171 MiB RSS; this is a
subset measurement, not a full-cloud memory estimate. The roughly 62x query-only
improvement is not an end-to-end speedup. Thread-threshold changes showed no
material benefit and should not be adopted on this evidence.

## Recommended improvements, in order

1. **Prototype a memory-budgeted spatial-index cache.** Reuse immutable KD trees
   for tiles or spatial regions across queries. Keep the disk index as fallback,
   cap total bytes across active indexes and workers using the cgroup limit,
   and handle invalidation on inserts. Preserve stable equal-distance tie rules,
   tile/overlap restrictions, positive-label selection, and all ownership gates.
   Validate exact results on regression fixtures and this full GFZ comparison
   before shipping it. A whole-cloud tree without a memory limit is only a probe.
2. **Remove redundant LAZ reads and intermediate index construction.** For
   example, `filter_owned_instances` writes each retained tile and then reads it
   again in `index_file`; indexing the written records during that pass could
   avoid a decompression pass. Preserve output point references, schemas, and
   publication/rollback behavior. Similarly avoid copying already-dense input
   solely to rebuild an equivalent transient index where ownership permits it.
3. **Measure phases explicitly before broader parallelism.** Record transfer,
   statistics, recovery-support queries, reconciliation, point ownership,
   deduplication, indexing and output times separately. Current overall timers
   cannot establish which stage accounts for the 90–100 minute filtering runs.
   Parallelize independent partitions only after preserving stable ordering.

Keep the v2.4 correctness rules. Do not trade them for v2.3's faster but incomplete
outputs. This turn changes comparison tooling/documentation only; no production
merge algorithm or validated image was changed.

## Evidence

Run directory: `comparison-v23/` under the existing GFZ validation run.

- `manifest.json`: exact images, commands, resource samples and exit states.
- `logs/v23_sat.log`, `logs/v23_fm.log`, `logs/v23_rct.log`: native results.
- `outputs/comparison_{SAT,FM,RCT}.json`: coverage, partition and sidecar diagnostics.
- `outputs/query-profile.txt`: exploratory profile.
- `outputs/query-variants.json`: unprofiled timing repetitions and equality checks.
- `commands/`: reproducible runners and analysis/benchmark scripts.

v2.4 reference success remains recorded in the parent `manifest.json` and
`outputs/final_validation.json`. Historical failure classes other than this GFZ
fixture were not rerun by this comparison.
