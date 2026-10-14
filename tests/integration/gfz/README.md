# GFZ local integration suite

This suite exercises the supplied GFZ eight-file collection (16,426,961 points)
with a four-core layout. It is a dataset-specific, local-only integration test.
Inputs are local fixtures; the harness does not call API, Galaxy or storage
services. This is the historical inference harness, not the runner for every
later SmartTile-only acceptance check below.

## Current acceptance and the three workflows

The canonical [combined validation record](../../../docs/validation/2026-10-13-combined-branch.md)
keeps the failed controls and subsequent successful fixes. As of 2026-10-13:

| User workflow | Accepted local evidence |
| --- | --- |
| Original → tile → SAT → merge → species → remap → visualization | Fresh tiling and a compatible cached SAT/species chain pass on a 553,030-point seam crop at 1/4 workers. The fresh and historical layouts were compared explicitly. [Scenario evidence](../../../docs/validation/2026-10-13-scenarios.json). |
| Original with predictions → tile + filter → species → remap → visualization | `tile --instance-tiles PredInstance_SAT,PredInstance_FM --instance-resolutions 0.01,0.1` passes on all 16,426,961 original points. These per-instance outputs need no further buffer filter. Species replay and `remap --match-instances` pass on 867,599 points at 1/10 cm, 1/4 workers and cache off/on. [Instance evidence](../../../docs/validation/2026-10-13-full-gfz.json), [remap acceptance](../../../docs/validation/2026-10-13-remap-export-fixes.json). |
| Multiple segmentation outputs → merge + source attribution/order → ForestStructure | `create_merged_file --merged-resolutions full` and attribute joins preserve existing source fields and generate `OriginalPointIndex`; all 12 full/10 cm LAZ/COPC/PLY products pass on 867,599 points. The real ForestStructure consumer accepts joined full COPC with 16 artifacts and seven independent checks. [Export evidence](../../../docs/validation/2026-10-13-remap-export-fixes.json), [consumer evidence](../../../docs/validation/2026-10-13-foreststructure-handoff.json). |

Separately, standalone buffered filtering passes for all four dense GFZ tiles
per model (**53,265,082 buffered records each for SAT and FM**) after the
[overlap-boundary fix](../../../docs/validation/2026-10-13-gfz-boundary-fix.json).
The earlier failed full-filter reports remain diagnostic evidence; the later
acceptance supersedes that defect. Whole-instance tiling preserves ownership
and per-voxel representation, not every source record after subsampling.

The final review source `f9c01a1` passes **520 tests and 90 subtests**. The
unpublished Galaxy wrapper `2.4.0+galaxy1` passes 18 functional cases and nine
independent numeric checks against its exact local candidate image. See the
[final review record](../../../docs/validation/2026-10-13-final-review-wrapper.md)
for source/image identities and the additional metadata-cache/LAS 1.2 PLY fixes.
These small acceptance checks do not rerun the GFZ corpus.

The earlier accepted remap/export source `f2d8ecb` passed **513 tests and 90 subtests** in
the pinned scientific runtime. Later replays reuse cached real predictions;
they do not rerun segmentation/species inference or visualization. Full-GFZ
filter/instance checks and the later cropped remap/export checks have distinct
source snapshots and are not one uninterrupted end-to-end run. ForestStructure
acceptance establishes artifact/point/voxel consistency, not independent Julia
metric parity or ecological accuracy.

Remaining release checks are the exact release-image workflow/visualization
canary and exact dataset 3110/3111 replays. The `2.4.0+galaxy1` wrapper is an
unpublished candidate. Dense-cell memory targets remain soft; measured runs
completed within their container caps, without establishing a universal bound.

## Run the historical inference suite

Build `smarttile:v2.4a` from the intended source snapshot (or set
`GFZ_SMART_IMAGE` to an explicit local candidate) and run the complete pytest
suite in its scientific environment first (see the root README). Record the
source commit and immutable image ID; the tag alone is not acceptance evidence.
Make the pinned SAT, ForestMamba, RCT and DetailView images available
(see `runner.py`). Check their actual CLI and GPU access before a new run.

```sh
python tests/integration/gfz/suite.py \
  --archive '/path/to/10_ gfz_test.zip' --run-root /path/to/new-run
```

Requires Docker access, local Python 3, GPUs 0 and 1, 10 CPUs and 50 GiB per
container. Model branches use separate GPUs and RCT uses CPU; at most three
stage containers run concurrently in this suite, within the approved cap of five.
Every run uses a new root. A stopped stage is resumed only by explicit invocation
of `--stage models` or `--stage postprocess` after diagnosing the failure; existing
stage/container names prevent accidental duplicate execution. There is no
automatic expensive retry.

## Inference-suite stages

1. Four 60 × 60 m cores, each buffered by 20 m, with a central junction inside
   the known GFZ extent; 1 cm COPC baseline and 10 cm LAZ model inputs.
2. Validate CRS, four populated cores and point counts.
3. SAT (GPU 0), ForestMamba (GPU 1), RCT (CPU), independently for all four tiles.
4. Transfer each model onto its own dense baseline with SmartTile's transfer
   implementation; invoke the actual filter CLI. RCT carries both tree tables.
5. DetailView on all three branches, retaining instance IDs and model-specific
   species fields. Preserve baseline and tree-table links in staged manifests.
6. Remap all three prediction families onto the untouched originals, with
   dataset-wide compaction independently for each model.
7. Compare all original point fields, count/order, headers and CRS metadata;
   validate final consecutive IDs, species fields and RCT table alignment.

The grid origin is specific to this GFZ fixture. Do not use these coordinates for
an unrelated dataset. Original inputs are mounted read-only inside every stage.
RCT outputs a terrain mesh, not a raster DTM.

## Evidence

`manifest.json`, `commands/*.json`, `logs/*.log`, per-stage resource JSONL and
`outputs/*validation.json` describe the actual run. Resource observations include
cgroup CPU usage and memory peak sampled while the container exists; GPU usage
is sampled and may miss short peaks. Model CSVs and RCT tables retain their
intermediate forms; final exported tables must use the final model ID mapping.
Image IDs are recorded at stage launch. The suite validates artifact consistency;
it does not establish biological prediction accuracy or all historical failure
classes. Small regression fixtures complement the real four-tile input.

## Alternate ForestMamba checkpoint

The GFZ run uses model-author revision
`thanhhff/ForestMamba@f0c5239d0d83e95f56508cdc20287f025a0b98db`
with `v6_expand_1_epoch_3000_fix.pth` and the upstream
`forestmamba_chm_radius16_qp300_2many_v6_expand_1.py` configuration.
Checkpoint SHA-256:
`89724eaffdd140782b9e553dcd55a1e01107cb685fb6853aae80bd89b52cd193`.
The local image is `forestmamba:upstream-f0c5239-expand1-epoch3000-gfz-20260924`.
Source: [model author's repository](https://github.com/thanhhff/ForestMamba/tree/f0c5239d0d83e95f56508cdc20287f025a0b98db).
Weights: [published checkpoint](https://drive.google.com/file/d/188KEscG4aB8s-Li5QtXdOTjF2THkv28y/view).

`GFZ_FM_IMAGE`, `GFZ_FM_CONFIG`, and `GFZ_FM_CHECKPOINT` select an explicit
image/config/checkpoint combination. The suite performs a strict CPU weight load
before starting model inference; partial weight loading is a failure. Both
preflight and inference use the same paths. For an authorized rerun after a
failure, set `GFZ_FM_ATTEMPT` to a fresh suffix so failed evidence is preserved.
Completed SAT and RCT stages are reused. This alternate checkpoint is a separate
validation candidate; its epoch count alone does not establish better accuracy.

Containers run with the intended non-root owner’s UID/GID and writable caches under `/work/cache`. Launch the suite as the user who should own the results. `GFZ_RUN_ATTEMPT` gives an explicitly authorized restart new container, log and scratch names while retaining previous stage records in `stage_history`. Archive interrupted output directories before restarting; completed validated stages are reused.

## DetailView permissions in the historical local image

The cached DetailView 1.1.1 image has root-only Python source files. Build the
permission-only derivative with `Dockerfile.detailview-kg281`, recording the
resolved base and resulting image IDs. It makes `/app` readable and uses UID
1010/GID 1012 by default; model contents and inference parameters are unchanged.
Set `GFZ_DV_IMAGE` to the derivative tag. The runner still sets the launching
user's UID/GID and writable caches explicitly. Validate CLI imports and GPU
access as that user before resuming failed DetailView stages with a fresh
`GFZ_RUN_ATTEMPT` and `GFZ_RESUME_PASSED=1`.
