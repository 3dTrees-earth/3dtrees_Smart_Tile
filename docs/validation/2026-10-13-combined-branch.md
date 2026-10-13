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

## Review fixes — 2026-10-13

Five reviewed data-safety defects are fixed on top of `7f496bb`:

- Instance tiling checks the complete input collection before processing and
  promotes compatible integer ExtraBytes without truncation. Regressions cover
  ID 70000 with uint16/uint32 inputs in both file orders, carried attributes,
  and incompatible metadata.
- Remap scratch files use internal identifiers instead of attribute names.
  Instance tile directories and summary filenames encode attribute names;
  absolute paths and traversal names cannot select files outside the job.
- Product and tile publication share destination validation. Input aliases,
  duplicate destinations, and existing results are rejected before writing.
  Products, including vector sidecars, are staged and rolled back on ordinary
  publication failure; tile jobs publish their complete directory only after
  all requested outputs and layout files succeed. This is not a power-loss or
  process-crash durability guarantee. Repeated builds need fresh destinations.
- Disabling OriginalPointIndex generation preserves existing values.
- Full-resolution merge and attribute join retain ExtraBytes scales, offsets,
  and no_data through the canonical metadata helpers, rejecting incompatible
  shared-field definitions.

The final source and regression tests passed the complete local tool suite:
**429 passed, 91 subtests passed**, exit 0, in 117.52 seconds. The command was the
same as above, with JUnit output enabled. The scientific-runtime image was
`smarttile:v2.4a-fast`, image ID
`sha256:b8c79e45dc52a363f8273f83a27b57c9954e8befaa8ea2fe20883bbfff119a52`.
Source and test dependencies were mounted read-only; networking was disabled;
the limits remained 10 CPUs, 50 GiB memory including swap, and 8 GiB scratch.
Regressions include input symlink/hardlink aliases, concurrent destination
creation, late publication failure, and failure after ordinary tiles but
before instance tiles complete. `git diff --check` passes.

The unchanged Galaxy wrapper at `d51eeb2b` also passed all **13 Planemo cases**
against this fixed source: 0 errors, 0 failures, 0 skips, exit 0. The final report
confirms output assertions for four tile cases, three merge cases, four product
exports (full-resolution LAZ, subsampled COPC and PLY), and two remap cases. Summed
test-case time was 557.79 seconds. This used an isolated local Galaxy instance
and disposable PostgreSQL database; tool jobs mounted the source read-only and
ran without network access, with 10 CPUs and a 50 GiB memory limit.

These fixes do not close the review's same-instance nearest-point tie issue:
numerically tied candidates can still be selected differently across prediction
batches. Dense spatial cells also lack a strict worst-case memory bound. Wider
downstream release certification remains incomplete, so this is not production
approval or certification of all three end-to-end scenarios.
