# Combined original-remap worker optimization — 2026-09-25

## Change

Combined merge/filter with originals previously called `enrich_originals` with its default of one process even when `--workers` permitted more. Standalone remap already used the bounded process pool. Both paths now select the enrichment process count in the shared function, using the effective CPU budget and available source batches. There must be at least one full source batch per worker; tiny inputs stay serial.

Each process opens completed indexes read-only and performs single-thread spatial queries. At most two batches per worker remain pending. The parent retains the ordered writer and final coverage/publication checks. Query-side rules, instance/semantic ownership, source metadata and tie handling are unchanged. The report records effective enrichment processes, query threads and pending batches.

## Sequential real-data comparison

Same GFZ SAT/FM subset: 275,253 original points, four dense tile inputs per model, nine output clouds. Locked ownership3 runtime; each container limited to 10 CPUs / 50 GiB; 4 GiB RAM scratch reduces disk-sync variance. The diagnostic wrapper varied only the already-existing process pool's worker count. Every run matched the saved point fields, order/count, scales and offsets exactly.

| Remap processes | Enrichment | Pipeline | Container CPU time | Observed peak RAM |
|---|---:|---:|---:|---:|
| 1 | 28.861 s | 59.806 s | 75.236 CPU-s | 799.5 MiB |
| 2 | 15.292 s | 46.694 s | 73.724 CPU-s | 1148.1 MiB |
| 4 | 9.065 s | 39.842 s | 74.568 CPU-s | 1549.7 MiB |

Four processes gave about 3.18x faster enrichment without increasing total CPU time in this single comparison. Two processes used less memory. Pipeline times exclude container startup and the subsequent equality scan; container CPU totals include the equality scan. These subset measurements do not establish a full-cloud speedup.

## Packaged validation

Image: `smarttile:v2.4a-remap4`  
ID: `sha256:2de4811a2f3d92c8a9561cfe4380c67170cf039e62d4390e5dd237b10bbe2d23`

385 tests passed in the packaged image (12.73 s), including new combined merge/remap tests comparing one and four processes, coverage-failure cleanup, and existing standalone/RCT worker tests. The new combined-worker regression failed before the change.

The packaged image's normal execution path with `workers=4` then processed the same real subset without the diagnostic override. All nine output clouds matched exactly. Its report confirms four enrichment processes, one query thread each and eight pending batches. Enrichment took 9.055 s; complete pipeline 43.097 s; observed peak RAM 1528.5 MiB; container CPU 74.860 s. Variation in other phases explains the difference from the first four-process diagnostic run.

The full ownership3 validation remains running separately. It is pinned to its original image and does not include this subsequent worker-selection change. No extra full filtering run was launched for this remap-only change.

Evidence: `/mnt/ssds/kg281/smarttile-v2.4a-gfz-20260924/optimization-remap-workers/` includes immutable image IDs, source snapshot, exact commands, resource samples, phase timings and output comparison results. All inputs remained read-only.

## Next target

Profile reconciliation on the completed full run. It previously cost 21.7 / 25.7 minutes for SAT/FM. It currently rereads each tile for every earlier neighbor; a single reading pass and avoiding irrelevant queries are candidates, but must preserve overlap counts, per-group numerical guards and mutual-best matching decisions before adoption.

## Full-dataset follow-up

The combined remap-worker change is now included in `smarttile:v2.4a-reconcile5` and its full GFZ SAT/FM validation. See [reconciliation validation](gfz-reconciliation-optimization.md). That full run remains pending; the subset measurements above are unchanged.
