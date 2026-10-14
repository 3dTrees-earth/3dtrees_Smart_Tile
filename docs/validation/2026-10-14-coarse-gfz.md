# Coarse-only GFZ filtering — local acceptance, 2026-10-14

Both cached GFZ segmentation collections passed filtering and instance reconciliation
at 10 cm without high-resolution targets. The local Galaxy wrapper now exposes
this route through **Filter / merge segmented tiles**, with finer targets optional.
This validates the stated coarse-input contract, not the complete production release.

## Candidate and inputs

- Wrapper: `2.4.0+galaxy4`, XML SHA-256
  `60f8a6db24b2af0b424b618bba76e70a7472935056a03101d70a76edd8be0fba`.
- Existing image: `sha256:cc3e4b7245a480a78c4f06daf1494db5fba7b2d636e7e4e5cee9c47a48534a2e`.
  No image rebuild or executable source change was needed for this acceptance.
  All 56 embedded Python source files matched the working source before each run.
  The working tree includes changes beyond base commit `b397214`; the companion
  JSON records the exact per-file hashes rather than treating that commit as the candidate.
- Fixture: the `smarttile-v2.4a-gfz-20260924` run, four buffered tiles per model,
  60 m cores, 20 m buffers, center-of-mass sampling at 10 cm. SAT and ForestMamba
  were inferred independently per tile; these are cached model results, not
  synthetic labels assigned to previously merged geometry.
- All eight prediction files preserve the corresponding pre-segmentation tile's
  coordinate multiset and order at 1 µm comparison precision. Input and output
  CRS match. Original input/layout hashes remain unchanged.
- Runs were sequential, CPU only, with 10 CPUs, a 50 GiB container RAM/swap ceiling,
  read-only source data and network disabled.

## Results

| Measurement | SegmentAnyTree | ForestMamba |
| --- | ---: | ---: |
| Buffered input records | 8,280,988 | 8,280,988 |
| Output records | 2,479,599 | 2,479,643 |
| Final positive instance IDs | 102 | 114 |
| CLI duration | 43.31 s | 49.43 s |
| Peak container RAM | 2.12 GiB | 2.41 GiB |
| Positive input core records checked | 6,447,843 | 6,640,632 |
| Records without final positive support within 1 cm | 0 | 0 |
| Recovered candidate fragments | 126 of 159 | 152 of 198 |

An independent validator reads the source/output point records, without importing
SmartTile or trusting its processing report. Every output coordinate and source
attribute tuple exists in the inputs, excluding the deliberately renumbered
instance ID. No final positive ID spans multiple output files. Core-support
checks cover every positive source record inside the participating core union;
this is a spatial support claim, not proof that every segmentation label is correct.

The 10 cm spacing does not require a wider tree-matching radius for these inputs.
The overlapping tiles reuse sampled coordinates: 99.9324–100% of overlap samples
have a counterpart within the existing 5 cm correspondence radius. Tree matching
also requires reciprocal best matches and the overlap threshold. The 1 cm
recovery/duplicate distances remain unchanged. Enlarging these distances to a
coarse voxel diagonal could join neighboring trees or remove distinct samples.

Every per-tile transfer record has a null transfer radius: the run really skips
coarse-to-fine transfer. The output manifest records `resolution_1_m = 0.1`.
Legacy internal names such as `remapped_1cm` do not describe the actual spacing.
The report remains `intermediate; final original coverage pending`, because this
task returns processed tiles and does not perform original remap.

## Boundary and wrapper regressions

`tests/test_coarse_filtering.py` covers reconciliation enabled/disabled at one and
two workers: four pipeline cases. Its 30 input records produce 16 known output
records. With reconciliation enabled there are four known tree groups; disabled
there are six expected fragment groups. Two trees only 8 cm apart stay distinct,
a unique recovered tip survives, and record values, IDs and file placement agree
across worker counts. All cases passed in the unchanged image.

Missing-value declarations were checked directly in the raw ExtraBytes VLR.
The controlled fixture preserves `Confidence`'s `65535` sentinel on both inputs
and outputs. GFZ inputs contain no declared missing-value fields, so they cannot
prove sentinel preservation by themselves. An initial regression assertion read
laspy's `DimensionInfo.no_data`, which omits this declaration on reread; it was
corrected to inspect the actual VLR. The original failure evidence is retained.
No runtime metadata fix was required.

The updated wrapper passed Planemo lint, 22 native render/routing cases and seven
live Galaxy functional cases (zero-based indices 4, 5, 6, 8, 14, 22, 23).
Ten Remap/Create render cases are unchanged. The live form exposes segmented
files and their resolution, optional finer files and their resolution, the
prediction model/field and tile layout, then collapsed Advanced controls.
Filter/Merge returns tiles and applicable RCT companions; original enrichment
and merged products use the separate Remap and Create tasks.

A separate live Create merged file smoke check selected two real Galaxy list
collections in repeat entries. One SmartTile job combined all 11,147 points,
preserving coordinates/order, source attributes, CRS, `OriginalPointIndex` and
both instance fields. This closes the earlier locally blocked collection check.

## Reproduction and scope

Use the existing image and a new output directory. The SAT command is:

```bash
python /src/run.py --task merge \
  --segmented-remapped-folder /fixture/collections/SAT \
  --output-dir /work/SAT \
  --tile-bounds-json /fixture/01_tiles/tile_bounds_tindex.json \
  --instance-dimension PredInstance_SAT --resolution-1 0.1 \
  --filter-anchor centroid --overlap-threshold 0.3 \
  --skip-merged-file --no-transfer-original-dims-to-merged --workers 10
```

For ForestMamba, replace `SAT` with `FM`. The [machine-readable record](2026-10-14-coarse-gfz.json)
contains commands, input/source hashes, output record digests and checked invariants.
The local `smarttile-coarse-gfz-20261014` artifact directory contains the actual
`SAT/output_tiles` and `FM/output_tiles`, processing reports, independent oracles,
resource logs, metadata audit and copied Galaxy acceptance evidence.

Same-run provenance and coordinate-preserving segmentation are caller requirements:
the current CLI does not automatically compare predictions against their original
tile files. This replay used the September 24 tiler, so it does not establish a
new run of the current tiler's overlap consistency. Biological tree truth,
equivalence to dense-first processing, downstream visualization and a complete
release-image staging workflow are outside this acceptance. Earlier release
validation records retain their own scope and unresolved findings.
