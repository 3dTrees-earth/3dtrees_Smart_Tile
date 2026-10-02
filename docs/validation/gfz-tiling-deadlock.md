# GFZ tiling deadlock after core occupancy

## Cause

The core-occupancy scan reads LAZ in the parent, initializing the native
lazrs/Rayon thread pool. Python 3.12 on Linux defaults to fork for
ProcessPoolExecutor. Forked children inherit the pool state but not its threads,
then wait indefinitely on their first parallel decompression. This happens
before model inference and is independent of point ownership or merge logic.

The two-tile GFZ run stalled at source distribution: both workers waited on
futexes, CPU time remained at 14.68 seconds, peak cgroup memory was 2.44 GiB.
The container was explicitly stopped; its exit 137 was not an OOM kill.

## Controlled probes

- One ten-point GFZ read in a child without a preceding parent read: passed.
- Same child read after a parent read, using fork: timed out after 8 seconds.
- Same sequence using spawn: passed in 0.64 seconds.
- Actual core occupancy -> two source workers -> two COPC workers, using a
  100,019-point compressed fixture: old image timed out after 25 seconds.
- Same regression with spawn for both pools: passed in 2.57 seconds, retaining
  coordinates, intensity and classification in both buffered outputs.

A 19-point single-chunk LAZ did not reproduce this failure. The regression uses
multiple compressed chunks and kills its subprocess group on timeout, so CI
cannot hang indefinitely. It requires lazrs parallel, PDAL and Untwine; run it
inside the packaged image to ensure these dependencies are present.

## Fix and scope

Both process pools in main_tile.py now request multiprocessing.get_context("spawn").
No geometry, filtering, ownership, checkpoint, resource limit or sampling
parameter changed. Subsampling already uses spawn.

Candidate: smarttile:v2.4a-2209-spawn, built from the exact boundary-fix image
with only main_tile.py replaced. Regression:
`tests/test_tiling_process_start.py`.

Original evidence and failure logs:
`/mnt/ssds/kg281/smarttile-v2.4a-gfz-two-tiles-20260928/diagnosis`.
The fresh GFZ workflow and final model/sidecar validations must be checked
separately; this diagnosis does not assert production readiness.

## Packaged validation and GFZ restart

The rebuilt image passed all **403 tests in 15.30 seconds**, including the new
regression. Immutable image ID:
`sha256:6d46c8a2c8ed1d8d67755d9fcf8056e0a5a03156d8640e783d7600e1dc94f84f`.

The fresh full GFZ run is tracked at
`/mnt/ssds/kg281/smarttile-v2.4a-gfz-two-tiles-20260928-spawn/manifest.json`.
It includes ForestMamba, RCT, both DetailView branches, original remap, and
RCT sidecar validation. At launch those workflow results remain pending.

The real rerun passed the original failure point: all eight GFZ source files
completed distribution and both COPC finalizers started within the first
22 seconds of the tile stage. Full subsampling and downstream model processing
remain tracked by the run manifest.
