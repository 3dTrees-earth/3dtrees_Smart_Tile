# GFZ 3DT-2209 final ownership audit — 28 September 2026

**Result: passed after source-level audit.** The strict equality check against
the old buggy output remains failed as historical evidence. All differences are
explained by the overlap-boundary correction; none are unaccounted for.

Image: `smarttile:v2.4a-2209-boundary`, immutable ID
`sha256:c19a47678e04024e0c2c9859cbd719f0d72e1dee1d1bb2f4d64badf3b3106ef0`.
402 packaged tests passed. Full four-tile SAT + FM processing, recovered-geometry
validation and remap of all 16,426,961 original points completed successfully.
Processing wall time: 7,689.51 s (2 h 8 min); peak memory: 16.66 GiB.

## Every intermediate difference checked

The eight new filtered clouds are exact subsequences of the old clouds: all
surviving point records are byte-identical and retain their original order,
scales, offsets and schema. Only 234 records were removed (65 SAT, 169 FM).

All 234 had previously been excluded by the strict overlap-boundary comparison.
For each, the audit finds a positive surviving claimant from another tile within
1 cm, with a better core rank at both the query and source location, and verifies
both records belong to the allowed overlap under the corrected numerical guard.

- 233 counterparts represent the same location within 1 nanometre.
- One FM counterpart is 9.219544 mm away, within the existing 10 mm matching rule.
  It retains the same instance 65 and semantic 2.
- No unmatched removals; no arbitrary clipping or new geometry.
- Accepted merge-pair membership and local-to-global ID mappings are unchanged:
  SAT has 100 groups / 178 accepted pairs; FM has 114 groups / 209 accepted pairs.
- Some correspondence counts increase because formerly excluded boundary records
  now participate. The accepted graph and group IDs do not change.

## Every changed original prediction checked

There are 82 affected original points, with 84 changed fields: 82 confidence
scores, one semantic label and one instance ID. Every old changed value traces
to a removed boundary claimant. Replaying nearest-point selection against the
new survivors reproduces every new value exactly. All 82 selected records come
from tile 2, whose core contains the original point.

### Tree ID 58 to 65

`398720_5646180.laz`, zero-based row 2082721, XYZ
(398729.466, 5646187.674, 399.668):

- Removed tile 1 claim: instance 58, semantic 2, score 0.7611703276634216.
- Retained tile 2 claim: instance 65, semantic 2, score 0.8754913806915283.
- Tile 2 core distance is 0; tile 1 core distance is 20.89156 m.

This is corrected point ownership between two existing instances, not a new
instance merge or ID renumbering.

### Semantic 2 to 0

Same file, zero-based row 2394486, XYZ
(398734.649, 5646187.674, 388.068):

- Removed tile 1/tile 3 claims: instance 58, semantic 2.
- Retained tile 2 claim: instance 58, semantic 0, score 0.8860500454902649.
- Tile 2 core distance is 0; tile 1 is 22.93275 m away and tile 3 is 20 m away.

The semantic 0 was already present in the owning tile's prediction. SmartTile
preserves it as requested. The point still has positive tree ID 58; it was not
lost or assigned background instance ID 0. This audit verifies attribution, not
the biological correctness of ForestMamba's original semantic prediction.

## Evidence and limits

Evidence lives in `fix-3dt-2209-boundary-20260928/audit/`: `extraction.json`
(complete deletion alignment and changed fields), `verification.json`
(source references and distances), `assessment.json` (all acceptance checks),
LAS/NPZ diagnostic fixtures, and the three reproducible audit scripts.
The raw verification report initially flags exact accepted-pair payload inequality;
`assessment.json` explicitly accounts for increased match counts and verifies
identical accepted pair membership and ID mappings. No raw results were erased.
The run manifest records the completed audit separately from strict equality.

The local GFZ geometry/ownership defect is verified fixed. No production rollout,
new model inference, or general biological prediction-quality validation is claimed.
