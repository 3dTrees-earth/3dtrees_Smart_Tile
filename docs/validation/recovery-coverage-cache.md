# Recovery coverage cache validation — 2026-09-24

Recovery support checks now call `PointIndex.covered`: a positive-tree-only,
boolean query using the existing bounded region cache. It does not collect
winning attributes or resolve nearest-point ties. It skips already supported
queries, retaining the original XY group's 1 cm distance bound and 8-ULP
roundoff allowance. Mutable indexes and oversized regions use a bounded disk
fallback. Cache invalidation and the recovery admission policy are unchanged.

## Validation

- 372 tests passed in 18.06 s, including four new regression tests covering
  boundaries, background exclusion, cache reuse, fallback/eviction, model
  separation, insertion invalidation and read-only reopening.
- Real GFZ SAT support benchmark: 442,631 indexed points and 219,844 queries.
  Old support check: 4.044 s. Cached coverage: 1.284 s with a cold cache,
  1.283 s on repeat. All support booleans identical; 3.15× speedup for this
  query workload only. Cache budget: 256 MiB; peak charged bytes: 266,053,284.
- Real GFZ SAT + ForestMamba subset processed through filtering, recovery,
  reconciliation and original remap: all nine output clouds exactly match the
  saved pre-change outputs, including all point fields/order, scales and
  offsets (1,160,523 output records).
- Tests ran with the locked `smarttile:v2.4a-cache1` runtime and the modified
  source mounted read-only; 2 CPUs, 4 GiB RAM, UID/GID 1010:1012, no GPU.
  The image itself and the existing full-cloud benchmark were not modified.
- Full-cloud speedup for this additional change has not been measured. A new
  image build is required to include it in subsequent container runs.

Local evidence and reproducible real-data validator:
`/tmp/smarttile-recovery-coverage-20260924/` (`tests.log`, `real.log`,
`validate_real.py`, `evidence/coverage_benchmark.json`,
`evidence/pipeline_equality.json`).

## Commit isolation

The cache/recovery-only commit was also tested on base `3d714ca`, excluding
the other pending merge/tiling/finalization changes: **360 tests passed** in
16.19 s. The separate two-collection cache integration test depends on the
pending merge-geometry implementation and remains with those changes. The
372-test and real-data results above describe the complete working tree.
