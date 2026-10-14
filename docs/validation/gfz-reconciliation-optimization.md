# Full GFZ reconciliation optimization — 2026-09-25

## Change

Reconciliation previously decompressed each query tile once per earlier overlapping neighbor. It now reads that tile once, accumulating separate pair counts for every earlier neighbor. It skips complete 2 m XY query groups without a positive instance eligible for that neighbor's overlap. The report adds `files_read`, `points_read`, `query_points` and `skipped_query_points` to `reconciliation`; query counters are per neighbor, whereas read counters count actual input reads.

Whole groups are retained when selected: background or high-Z points can set the group's existing eight-ULP numerical allowance. Target-side background remains a nearest-neighbor candidate, so a closer background point still blocks a farther tree match. Mutual-best counts, overlap threshold, stable edge order, same-tile exclusion and recovered-bridge restrictions are unchanged. Paired RCT tree-sidecar processing continues to bypass reconciliation.

The immutable candidate also includes the preceding bounded spatial-cache, ownership/tie and parallel original-enrichment changes. This full run therefore validates the combined candidate; phase metrics isolate reconciliation and enrichment, but overall speed cannot be attributed to reconciliation alone.

## Regression evidence

The new three-tile regression failed before implementation because tile 2 was read twice. It now checks one read per query file and unchanged accepted pair counts. A second test covers high-Z numerical allowance and nearest-background blocking. The locked-runtime suite passed all 387 tests before building. Packaged tests are the first gate of the full driver.

## Required large-data validation

Image: `smarttile:v2.4a-reconcile5`  
ID: `sha256:936e7faaa9d7262ffc8628739eaf7807a37d3062e1c825fe9eeaed0ae2131a6b`

Inputs are the existing read-only full GFZ inference outputs: SAT and ForestMamba, each four buffered tiles and 53,265,082 dense records, with 60 m cores and 20 m buffers. All four cores contain original points. Original enrichment processes all eight files / 16,426,961 points. This reruns filtering, recovery, reconciliation, ownership, deduplication and original remap; it reuses completed tiling and model inference.

Run directory: `/mnt/ssds/kg281/smarttile-v2.4a-gfz-20260924/optimization-reconcile5/`.

The driver runs packaged tests, full processing, then exact field/count/order/scale/offset comparisons for all eight filtered tiles and eight enriched original files against the successful union1 baseline. Reference-only RCT/species fields are excluded because this run processes SAT/FM. Mismatches fail the validation; no automatic costly retry is scheduled.

Each container runs as kg281 with 10 CPUs / 50 GiB and no GPU. The previous ownership3 run remains active as a separate comparison, so shared disk contention can affect wall times. The manifest records this limitation. Commands, source snapshot/hashes/diff, logs and cgroup CPU/peak-memory samples are saved. Processing and full equality are pending; no full speedup or production-readiness claim is established by the unit tests.

Future optimization changes must also run this full GFZ validation. Small probes are supplementary and cannot replace the larger dataset gate.
