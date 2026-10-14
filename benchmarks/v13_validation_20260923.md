# SmartTile V13 representative validation — 2026-09-23

Image: `smarttile:pr3-v13-orphan-recovery-20260923`

Image ID: `sha256:c812b1e14f1f45dbcdad02a4ce6b88cd8b973d0633b58197ebcf7c472bbd49fb`

The operator accepted one passing representative per confirmed failure group.
A group marked complete below means local representative validation; it does
not assert that every dataset or its production workflow completed.

## Real-data crops

| Dataset / file | Model | Previous coverage | V13 coverage | What this checks |
| --- | --- | --- | --- | --- |
| 3142 / 10500 | SAT | 6/12 | 12/12 | Gaps left by instance filtering |
| 3142 / 10500 | ForestMamba | 2/12 | 12/12 | Gaps left by instance filtering |
| 3148 / 10518 | SAT | 0/5 | 5/5 | Gaps left by instance filtering |
| 3148 / 10518 | ForestMamba | 5/5 | 5/5 | Previously covered control |
| 3148 / 10519 | SAT | 8/8 | 8/8 | Previously covered control |
| 3148 / 10519 | ForestMamba | 0/8 | 8/8 | Gaps left by instance filtering |
| 2595 / 8650 | Both models independently | 6/11 | 11/11 | LAS 1.0 output and voxel-diagonal matching |

For 3142 and 3148, both old and new filtering were checked at the same
17.320508 mm final remap radius. Increasing tolerance alone did not fill the
filtering gaps. V13 recovered one whole instance per model per dataset;
post-deduplication support checks reported zero missing samples. The replay used
small XYZ crops with saved full-instance anchor/ownership decisions. Cross-tile
instance reconciliation was disabled because crop overlap fractions do not
represent whole instances. This validates recovery, point ownership,
deduplication and final remap on those crops, not a full-dataset merge.

For 2595, the actual LAS 1.0 source header was passed through the original
reader; a wrapper restricted yielded records to crops around five saved failure
locations. All five locations were verified in the 11-point original crop.
The old 10 mm radius reproduced the failure; the automatic sqrt(3) times 1 cm
radius passed both baseline and final coverage for both models. Maximum matched
distance was 14.142135 mm. Output LAS 1.4 retained all original raw point fields,
scales and offsets exactly. Only five of the 300 previously reported missing
points per model were sampled; the full 22,434,879-point original was not remapped.
An output-verification harness accessor was corrected and verification rerun
against the successful remap output.

Related local source-code probe: 2591 improved from 10/15 to 15/15 original crop
points in both models at the voxel-diagonal radius. This was a radius probe on
saved products, not a full V13 Docker replay.

## Resources

Both Docker crop replays used a 2-CPU / 4-GiB limit, no network, and read-only
corpus inputs. For 3142 and 3148 together: 37.94 seconds, peak Python process RSS
128.20 MiB. For 2595: 59.72 seconds, peak Python process RSS 87.68 MiB, plus a
9.01-second output verification pass. RSS is process memory, not total container
memory. These crop timings do not estimate full-dataset throughput.

## Worktree regression check

The current worktree was mounted read-only into V13:

- 312 unittest cases passed. Initial discovery also produced two loader errors
  because the runtime image omits pytest; these were test-environment errors.
- The two pytest-only modules (`test_parallel_remap.py` and
  `test_legacy_las_output.py`) then passed all 13 cases with existing local pytest
  dependencies mounted read-only. No runtime image packages were changed.
- Total: 325 test cases passed. `git diff --check` passed.

The 55 source-file hashes match the V13 manifest except for `main_merge.py`,
whose only additional change corrects the final-radius docstring. No executable
source difference was found.

The historical full-run results and unresolved/unclassified dataset groups remain
separate from these representative checks. No production reruns were submitted.
