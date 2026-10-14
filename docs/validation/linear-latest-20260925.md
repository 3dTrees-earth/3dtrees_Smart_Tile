# Latest Linear SmartTile failures — 25 September 2026

## Scope and runtime

Read-only Linear and Galaxy inspection identified two new reports created on 25 September. Both failed under SmartTile `2.3+galaxy0`; neither production invocation was rerun or modified. Local validation uses the PR3/remap-first branch, not a deployed release.

Candidate before fix: `smarttile:v2.4a-reconcile5`, `sha256:936e7faaa9d7262ffc8628739eaf7807a37d3062e1c825fe9eeaed0ae2131a6b`.

Candidate after fix: `smarttile:v2.4a-crs6`, `sha256:f19317dfb561373d021301c9deda7c23049578687d65aba7323839cc7fbd78f8`.

Evidence directory: `/mnt/ssds/kg281/smarttile-linear-20260925/`. It contains invocation/job evidence, input IDs and checksums, spatial crop commands, original tile layout, source snapshot/hashes/diff, regression logs, and validation reports. Tests run locally as kg281 with 10 CPUs / 50 GiB, no GPU, and at most five concurrent containers. Read-only Galaxy access downloaded artifacts; it did not submit jobs.

## 3118 — coverage gaps (3DT-2201 / 3DT-1566)

[Issue 3DT-2201](https://linear.app/geosense-ufr/issue/3DT-2201) reports failed final remapping of the 21-file original collection. The smallest failing original, `9598.laz`, has 491,448 points. Galaxy reported 491,173 matches within 0.125 m for SAT: exactly 275 missing points.

An initial interior crop passed in both models and was rejected as a failure reproduction. Expanding to the complete `9598.laz` footprint and cropping *both* intersecting prediction tiles reproduced the exact 275 SAT gaps; an independent FM check found 21 gaps. All other prediction tile headers lie outside this footprint. Prediction crops include a 1 m halo. At 0.1732 m there are still 252 SAT gaps and 21 FM gaps; nearest distances reach 1.195 m / 1.414 m. Increasing tolerance does not repair lost geometry.

The candidate reruns transfer from the actual unfiltered SAT/FM predictions onto the original 1 cm targets, filtering, recovery, reconciliation, shared ownership, deduplication and final original enrichment. COPC range queries extracted 520,535 / 189,931 dense points from the two relevant tiles. Both model collections use the original core rectangles and full neighbor layout; only declared point bounds are updated for the crop. Raw prediction crops include additional halo for transfer.

Result: **491,448/491,448 original points matched for each model**, both against unfiltered geometry and final survivors. Maximum matched distance is 0.00734915 m, below the strict 0.01732051 m voxel-diagonal bound. Original point count/order, all existing fields, scales, offsets and parsed CRS match. The local pipeline took 24.80 s with 2.34 GiB observed cgroup peak memory. It recovered unsupported positive geometry in both models; no tolerance relaxation or zero-filling was needed.

Limits: this validates the original's full point set against cropped neighboring collections. Instance anchors are computed from cropped geometry, so it does not certify full-dataset ownership, instance counts or all 21 originals. DetailView was not rerun; renewed filtering requires species inference afterward before final species enrichment. Existing v2.3 filtered outputs still contain the gaps and must not be reused as proof of v2.4 recovery. Production recovery remains pending.

## 3433 — duplicate WKT precedence (3DT-2200 / 3DT-1898)

[Issue 3DT-2200](https://linear.app/geosense-ufr/issue/3DT-2200) fails while producing the 10 cm merged LAZ after original enrichment succeeded. The enriched original has 77,966,458 points. The validator reports missing or changed CRS/projection metadata.

An initial 50,000-point subset of the standardized input passed in both v2.3 and v2.4 and was rejected as a failure reproduction. A 50,000-point subset of the **actual enriched output from the failed job** reproduces the exact error in the v2.4 candidate's ten-chunk product merge.

Diagnosis: the staged COPC carries normalized WKT in a VLR (634 bytes) and original WKT in an EVLR (590 bytes), both identified as `LASF_Projection/2112`. The destination already holds the original 590-byte WKT. Preservation incorrectly appends the shadowed 634-byte variant, making it the effective last record; strict validation then rejects it. This is a serialization-precedence bug, not an identified coordinate transformation or unknown CRS.

Fix: collapse source records by `(user_id, record_id)` using the same last-record precedence as the existing projection reader, then append only missing effective records. Preserve the original authoritative WKT, point data and strict validator. Repeated preservation is idempotent; genuinely changed CRS is still rejected.

Regression: two precedence/idempotence cases fail before the patch; after it, all three focused checks pass. The fixed packaged image passes **390 tests** in 11.88 s. The exact failing enriched subset now completes in 1.50 s and produces 31,735 nearest-to-centroid representatives with source CRS preserved.

Larger validation: the fixed image is running the same product stage on the **complete 77,966,458-point enriched cloud**. Inputs are mounted read-only, output is isolated in `3433/full-crs6`, and `3433/full-status.json` / `full-crs6-validation.json` record completion. Full result pending at report creation; no production readiness or full-dataset success is inferred from the crop.

## Other recent issue classes

- [3DT-2167](https://linear.app/geosense-ufr/issue/3DT-2167), missing `main_subsample`: the current image copies the complete source tree; the real product-creation replay exercises this path successfully after the CRS fix. Dataset 1668 itself was not replayed here.
- [3DT-2172](https://linear.app/geosense-ufr/issue/3DT-2172), mixed uint16/uint32 instance IDs: collection-wide promotion/overflow protection is already present and covered by the packaged suite. Existing corrupted dataset-2924 products are not repaired merely by changing images.
- Recent exit-125 / `unexpected EOF` reports are container/node failures. A SmartTile algorithm change does not establish that the Galaxy runtime issue is fixed; those require a healthy node and a separately verified rerun.

Neither issue was marked resolved in Linear. Evidence supports a passing 3118 crop and a newly fixed 3433 defect, not blanket closure of historical errors or production recovery.
