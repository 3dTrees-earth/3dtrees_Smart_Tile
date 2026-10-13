# SmartTile 2.4 final review and Galaxy wrapper

This record supplements the historical GFZ acceptance. It validates the local
SmartTile source and a Galaxy wrapper candidate, not a published production
image or deployed workflow.

## Review changes

- Product metadata preflight now selects the canonical raw source when a
  same-source COPC cache is present. Distinct COPC-only inputs remain subject to
  schema validation and are included exactly once.
- A new LAS 1.2 fixture exposed full-resolution PLY silently dropping ExtraBytes.
  Explicit PDAL ExtraBytes reading now preserves exact species, instance, measurement
  and uint32 OriginalPointIndex values for legacy and modern LAS inputs.
- The wrapper uses a single preflighted JSON staging manifest and a small Python
  helper instead of interpolating dataset names into shell commands. Duplicate
  names, traversal and directory symlink escapes fail before staging files.
- Wrapper 2.4.0+galaxy1 exposes whole-instance tiling, standalone dense filtering,
  same-instance remapping and an explicit custom instance field. Existing RCT
  sidecars, attribute-view joins and optional product formats remain covered.

## Acceptance boundary

Real GFZ SAT/FM results, full-GFZ filtering and whole-instance tiling, cropped
species remap/products and the ForestStructure COPC consumer have prior local
acceptance records. This pass adds a small adversarial fixture: another tree's
point is closer for 19 of 20 originals, yet same-instance remap retains
species 10/20 and instance IDs 7/70000. LAZ/COPC preserve Amplitude's missing
sentinel declaration; PLY preserves its values and a uint32 order key.

The exact published release image, production-workflow parameter rebinding,
visualization staging canary, coordinated RCT rollout and exact 3110/3111 replays
remain release work. No image, ToolShed revision or production workflow is
published by this validation. Dense-cell processing targets remain soft rather
than a universal RAM bound.

## Final local results

| Check | Result |
| --- | --- |
| Complete scientific suite | 520 tests and 90 subtests passed; zero failures, errors or skips; 123.15 seconds. |
| Real Galaxy/Planemo | All 18 functional cases passed across five tasks; zero failures, errors or skips. |
| Safe input staging | 44 tests passed, including collisions, traversal, literal shell characters and rollback. |
| Offline Galaxy macro/Cheetah rendering | 13 variants passed, including default and custom instance fields. |
| Independent numeric oracle | Nine checks passed across five CLI commands; all six fixtures unchanged. |
| Galaxy-discovered datasets | Six published point datasets pass exact IDs, species, original attributes, missing declarations, CRS and PLY/COPC order-index checks. |
| Packaging | Planemo XML lint, ToolShed lint and git diff checks passed. |

SmartTile runtime commit: `f9c01a14ec8351e796253847a83bc232824dc5c6`. Wrapper implementation commit:
`357064879abccc590ae117d0b48db2358901c5c3`. Wrapper XML SHA-256:
`d1c785b29e7822f47a021d2effd7a48d5e4b5468d0fcca4efeeda79d632d416b`. The 54 embedded source files match the frozen
accepted source. Candidate image:
`sha256:b8298326df51a401df6cab9893b7a7935f36bbf498675c99f463bb4fa0fae814`.

Galaxy tested XML `94fb13a07aa2c100fb870370e943a674280374531aeb5bc2936463bf23219ff3`. The final wrapper differs only in one
help example, corrected from `Intensity` to the exact LAS field name `intensity`;
commands, parameter semantics, outputs and tests are unchanged. Final XML lint
and package checks pass after that documentation correction.

The scientific suite and Galaxy jobs used 10 CPUs and a 50 GiB container cap,
without additional swap or network access. Scientific-suite peak memory was
0.488 GiB, with no OOM events. These limits
describe the test environment; they are not a SmartTile internal memory cap.
The numeric oracle used two CPUs and 8 GiB. Galaxy used an isolated local server
and disposable localhost PostgreSQL; the embedded tool source had no override.

The first 17-case Galaxy run passed 16 cases and reproduced the LAS 1.2 PLY loss
in case 16 despite the tool exiting successfully. Its evidence remains separate;
the final rebuilt image passes that case and the added custom-field case 18.

Machine-readable evidence and report hashes are in the adjacent JSON record.
Raw local logs and artifacts are retained under
`/tmp/smarttile-final-review-fixed-20261013` and
`/tmp/smarttile-final-wrapper-fixed-20261013`. Prior failed controls remain in
the corresponding paths without `-fixed`.

The six dataset checks resolve Galaxy result UUIDs into its dataset storage,
not the job working outputs. Uploaded originals byte-match the fixtures. Both
disposable databases, Galaxy processes and tool containers were cleaned up;
local evidence files remain available.
