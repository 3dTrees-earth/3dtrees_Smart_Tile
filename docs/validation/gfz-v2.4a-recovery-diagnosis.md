> Updated contract: the recommendations and recovery1 repair below are historical
> and superseded. Accepted groups now share one ID and retain combined geometry;
> ownership selects attributes only at shared points. See the current README and
> filter decision flow. The sole-owner functions and coverage gate were removed.

# GFZ recovery coverage failure — diagnosis

## Confirmed cause

SAT and ForestMamba independently recover several rejected instances that each
supply previously unsupported core geometry. Cross-tile reconciliation then merges
some of those recovered instances. `instance_owners` selects a single member,
favoring the member with the most new recovered locations. `retain_instance_owners`
drops every point belonging to the other members, including the unique points
that justified recovering them. The selected owner is never checked for covering
all required recovered geometry before the other members are removed.

The final 1 cm support check correctly rejects this loss. The demonstrated loss
occurs before shared-point resolution and deduplication. Increasing the matching
radius or removing the final check would hide it.

## Real run evidence

- SAT: 120 recovered candidates admitted, 52 discarded by sole-owner selection;
  final support check failed for 5,158 / 87,712 checked recovery samples.
- ForestMamba: 150 admitted, 53 discarded; final support check failed for
  4,600 / 128,030 checked recovery samples.
- These are per-candidate validation samples, not a count of unique original
  uploaded points. The discarded-instance counts do not equal missing samples.
- SAT example: tile_00003 local ID 4 was recovered (696 new locations), then
  grouped with tile_00001 local ID 4 (2,416 new locations) and tile_00002 local
  ID 19 (80 new locations), under reconciled ID 63. Tile_00001 wins and both
  other source copies are discarded.
- FM example: tile_00002 local ID 57 was recovered (396 new locations), then
  grouped into ID 37 with tile_00001 local ID 49 (1,108 new locations),
  tile_00000 local ID 41 (78 new locations), and tile_00003 local ID 50.
  Tile_00001 wins.
- A streaming inspection at the first missing coordinate of each model finds
  an exact recovered-source point and zero authoritative tree points within
  1 cm. Evidence: sat_missing_point.json and fm_missing_point.json.
- RCT filtering passes; it skips cross-tile instance reconciliation.

## Fast reproduction

`gfz_recovery_repro.py` exercises the actual merge/filter pipeline, with explicit
core/overlap metadata injected to isolate the ownership behavior. Three tiles
contain five records:

- a: tree 1 at x=0.90; normally retained.
- b: tree 2 at x=0.90 and 1.05; rejected by its anchor, then recovered for 1.05.
- c: tree 3 at x=0.90 and 1.15; rejected by its anchor, then recovered for 1.15.

The overlap rule merges all three into one group. Stable owner selection keeps
b; x=1.15 vanishes. The command exits 1 with:
`recovered tree geometry lost after ownership and deduplication: 1/2 core samples unsupported`.
The eight-point precursor and the five-point reproduction both fail consistently.
`gfz_recovery_trace.py` shows this geometry loss directly at the output of
`retain_instance_owners`, before shared-point/deduplication stages.
`gfz_recovery_no_merge.py` changes only `matching=False` and passes.

Example command (mount this directory as /diagnosis):

```sh
docker run --user 1010:1012 --cpus 10 --memory 50g --memory-swap 50g \
  -v /path/to/smarttile:/repo:ro -v /path/to/investigation_recovery:/diagnosis:ro \
  --entrypoint python smarttile:v2.4a /diagnosis/gfz_recovery_repro.py
```

## Repair recommendation

Before discarding a recovered member, require that the proposed surviving
owner covers the geometry that justified that recovery within the existing
1 cm support radius. If no single owner covers the necessary geometry, keep
those recovered instances in separate reconciliation groups; the existing
nearest-core rule can still resolve genuinely shared points while preserving
unique tails and their source semantics. Apply the check to the complete group,
including transitive merges, not just one pair.

Preserve the existing no-bridge rule between distinct normal retained groups,
the final support check, RCT membership rules and semantic ownership. A blanket
radius increase or a blanket ban on all instance merging is not a repair.

No production algorithm was changed and no full dataset retries were started by
this investigation. These saved scripts are diagnostic fixtures, not passing
regression tests. Convert the failing case into a regression when implementing
the repair; also cover transitive groups and a safe merge with complete support.

## Implemented repair

`RecoveryMergeCoverage` now gates each proposed reconciliation union against
its eventual sole owner. It checks every recovered member's recorded core claims
at the existing 1 cm radius using bounded spatial batches, filters support by
both source tile and local instance ID, and caches per-instance verdicts.
The gate checks the entire component on every union, so an owner change during
a transitive merge cannot invalidate earlier members' coverage. Owner ranking
is shared with final output selection. Unsafe merges stay separate; shared-point
ownership and the final recovery support check remain active.

The five-point pipeline case now includes original remap and tests preservation
of both tips, their source semantics, separate IDs, and shared-point ownership
with one and four query workers. Additional cases cover safe transitive merges,
unsafe transitive owner changes, source ID isolation, and the XYZ radius boundary.
Saved SAT group 63 and FM group 37 coordinate probes both reject the previously
unsafe merge and preserve the necessary source. These focused probes complement,
and do not replace, the pending full real-dataset rerun.
