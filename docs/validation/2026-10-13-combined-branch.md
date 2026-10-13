# Combined SmartTile v2.4 branch validation — 2026-10-13

## Integration

The working branch is `codex/3dt-2101-remap-first`. It includes the earlier
CRS fix (`29ff07d`), full-resolution products and attribute joins (`7e75ef7`),
and instance tiles with same-instance remapping (`6e5178e`). Instance tiling
remains CLI-only. The related Galaxy wrapper branch is
`claude/smarttile-v2.4-wrapper`, including `cdfe22ca`.

The combined patch removes the unused tiling/subsampling path, bounds dense
original-remap batches, restores application property names in PLY exports,
preserves merge/filter output-directory conventions, and makes subsampled
product selection and equal-GPS-time ordering deterministic. PLY fields whose
names collide after lowercasing are rejected, including `treeID` with `treeid`.

## Final tool suite

The final source was copied to a fixed snapshot, mounted read-only in the
existing SmartTile scientific-runtime container, and tested without network
access. Limits: 10 CPUs, 50 GiB memory including swap, and an 8 GiB temporary
filesystem for test scratch. The pytest dependencies were also mounted read-only.

```sh
python -m pytest -q -p no:cacheprovider -o faulthandler_timeout=300 tests
```

**392 passed, 91 subtests passed**, exit 0, in 141.87 seconds. This includes the
new lowercase PLY-name collision cases and deterministic product-order tests.
An earlier run using the container's ordinary temporary filesystem was stopped
while waiting on filesystem I/O; that incomplete run is not validation evidence.
`git diff --check` also passes.

## Dataset 645 reproducibility

Two local builds of the 10 cm LAZ product, using the same existing enriched
originals and 10 workers, produced:

- 80,036,292 selected points;
- 808,809,039 bytes in each LAZ;
- identical SHA-256:
  `3a76d350b5a028e550a4bf9b1edd24958877c3793bb8b4376189ceace5ef7947`.

Each build completed in about 9–10 minutes; measured peak memory was 14.34 GiB
and 13.39 GiB. Before the equal-GPS-time tie fix, the point sets matched but
record order and compressed bytes differed.

This establishes repeated LAZ byte identity for the tested inputs, runtime,
and chunk configuration. It does not certify byte identity across different
COPC encoders, runtime versions, or chunk layouts. Product chunk planning
uses an average point-count target, not a hard density-independent memory cap.

## Release boundary

These are local integration and validation results. They do not establish a
production deployment, recovery of individual production datasets, or completion
of the wider SmartTile v2.4 release certification.
