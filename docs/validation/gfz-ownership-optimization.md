# GFZ ownership optimization and remap discrepancy — 2026-09-25

## Diagnosis

The previous full SAT/FM run completed processing but failed final equality. A complete scan found 81 differing `PredScore_FM` values and one differing `PredInstance_FM` / `PredSemantic_FM` assignment, all in `398720_5646180.laz`. All original source fields and SAT predictions matched. Comparing all eight filtered SAT/FM tiles found identical point counts, order, dimensions, scales and offsets.

Combined remap used the first prediction tile's header origin; standalone remap used the first original's header origin. At numerically coincident boundary points this changed distances at the float64 ULP scale. For example, one pair differed by only 1.776e-15 m with the combined origin, but tied exactly with the standalone origin. Disabling the region cache did not change the discrepancy. All 81 affected query locations were reproduced with small extracted prediction neighborhoods.

Nearest selection now treats candidates within the existing eight-local-coordinate-ULP roundoff allowance of the true minimum as a numerical tie, keeping stable tile / source-point order. A bounded second pass resolves rare non-exact ties against the global minimum, so a chain of pairwise-close distances cannot make cache or batch partitions change the winner. The spatial matching radius is unchanged. Every affected-point replay matches reference score, instance and semantics with either origin and with caching on/off.

## Ownership optimization

Shared-point ownership previously rebuilt competitor trees from many SQLite batches and repeatedly calculated the same source-side core eligibility. Reuse the existing bounded LRU cache for positive competitor points inside the overlap and eligible side of the core boundary. Keep query-side nearest-core ranking, source-side eligibility, stable tile priority and background rules. Skip competitors that cannot win before loading their points. Mutable or oversized regions use bounded disk batches. Cache entries include both cores and the origin and are invalidated by writes.

Baseline coverage in original remap now requests minimum distances only; it does not pay for attributes or source-identity tie resolution on unfiltered overlapping copies. Prediction remap still applies the full identity rule. This avoided the regression seen in the first diagnostic tie implementation; that trial was stopped and its logs preserved.

## Controlled real-data comparison

Same SAT/FM subset, same locked runtime, 10 CPUs / 50 GiB, sequential runs with 4 GiB bounded RAM scratch to reduce disk-sync variation. 275,253 original points; four dense tile inputs per model; all nine output clouds compared exactly (1,160,523 output records). No biological accuracy claim is implied by equivalence.

| Measurement | Previous recovery2 | Candidate |
|---|---:|---:|
| SAT shared-point ownership | 10.284 s | 1.908 s |
| FM shared-point ownership | 8.738 s | 1.795 s |
| Complete subset filtering + remap | 76.900 s | 60.660 s |

Ownership improved 5.39x / 4.87x; complete subset time decreased 21.1%. All output point fields, counts/order, scales and offsets matched. This is one controlled sequential comparison, not a full-cloud speedup claim.

383 regression tests passed in 10.83 s, including the real GFZ numerical tie, chained-near-distance batch partitioning, cached/disk ownership eligibility and baseline-distance equivalence. A separate all-81-point replay passed for both origins with caching enabled and disabled.

## Full validation

Image: `smarttile:v2.4a-ownership3`. Sources and immutable image ID are captured in the run manifest. The full rerun uses both model collections, the same four buffered tiles, kg281, 10 CPUs, 50 GiB RAM and no GPU. Packaged-image tests precede the full run and final original-output equality check. Full-cloud validation remains pending; do not claim it has passed.

Evidence: `/mnt/ssds/kg281/smarttile-v2.4a-gfz-20260924/optimization-ownership3/` (`evidence/`, source snapshot, commands, manifest and logs).

## Remaining performance opportunities

Reconciliation still takes 21.7 / 25.7 minutes in the previous full run. Combined merge-with-originals currently uses one enrichment process, while standalone remap supports bounded process parallelism; this explains part of the 88.7-minute combined enrichment stage. Those are separate next targets, not changes included here.
