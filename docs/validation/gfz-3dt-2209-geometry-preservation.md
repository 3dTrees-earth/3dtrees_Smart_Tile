# 3DT-2209: preserve merged member geometry with spatial ownership

Issue: https://linear.app/geosense-ufr/issue/3DT-2209

## Failure and root cause

The exact `6540d08` image passed 373 packaged tests but failed the full four-tile
GFZ SAT stage after 2,562.77 seconds: 5,158 of 87,712 recovered core samples no
longer had positive-tree support. Peak memory was 6,855,778,304 bytes. ForestMamba
and original enrichment were never reached.

Reconciliation identifies one tree ID for an accepted group; it does not prove
that any single member contains the entire group's geometry. The old pipeline
called `instance_owners` and `retain_instance_owners` after recovery and removed
all records belonging to the non-winning members. Distinct recovered tips were
therefore discarded before shared-point ownership or deduplication could protect
them. Increasing the matching radius or disabling validation would hide this loss.

## Deterministic evidence

Running `tests/test_merged_group_geometry.py` against the exact failing image
produced three failures:

- Recovered members: 1 of 2 required core samples lost after ownership/deduplication.
- Normal transitive A-B-C group: only 2 of 4 original points retained coverage.
- Distinct points across a core boundary: only 3 of 4 original points retained coverage.

A single-variable probe changed only the pipeline to bypass whole-instance owner
removal. Both the recovered-tip and normal-transitive regressions then passed.
The remaining boundary test exposed a semantic conflict between distinct points
at x=0.999 and x=1.001, each owned by its own core. This confirms the union fix
must retain the existing core-aware conflict handling too.

## Correction packaged for validation

The working branch already had the union correction, but it was absent from the
exact-commit image. The candidate packages that implementation with `6540d08`'s
spatial-query optimization:

1. Keep every accepted group member until point-level ownership is evaluated.
2. Assign one ID to the transitive group; retain unique geometry from every member.
3. Resolve shared records using nearest-core ownership and stable tile order.
4. Preserve each winning record's attributes. Distinct points owned by opposite
   cores may retain different semantics even inside the same reconciled tree.
5. Retain strict recovered-geometry and final original-coverage checks.
6. Paired RCT sidecars continue to select the separate filter-only path, without
   reconciliation or cross-tile membership changes.

The recovered-tip regression now also asserts two admitted members, one merged
ID, original-coordinate preservation and exact final per-point semantics with
one and four query workers.

## Validation run

Candidate tag: `smarttile:v2.4a-2209-union-spatial`.
The run manifest stores the immutable image ID and all packaged source hashes.
This is a frozen working-branch snapshot, not a claim that the image is pristine
`6540d08`; it also contains the existing remap-worker, CRS and ID-finalization work.

Scope: full GFZ, four 60 m cores / 20 m buffers for SAT and ForestMamba together,
53,265,082 dense records per model, then 16,426,961 original points. Reuse saved
model predictions; no new inference or DetailView run. Run as the host user with
10 CPU / 50 GiB memory per container, no GPU, read-only inputs, fresh output and
scratch directories.

Run gates: packaged tests, full processing with strict geometry validation,
then all 16 output clouds compared with the saved union baseline. Comparison
checks point count/order, scales/offsets and all candidate dimensions. Reference-only
RCT/species fields are outside this two-model scope. Every other difference stays
a failure; the earlier FM-score discrepancy is not waived.

Packaged validation: **395 tests passed in 23.68 s**. The immutable image is
`sha256:8c02e8825efc4ce0afd89634623ef04aecc24d0bc5bf7fe9afe09732ed7afb95`;
all `/src` hashes match the frozen snapshot. Full GFZ processing has started.
Full-data correctness and speedup are not yet established. Keep the issue
In Progress until full processing and output-comparison results are evaluated.
