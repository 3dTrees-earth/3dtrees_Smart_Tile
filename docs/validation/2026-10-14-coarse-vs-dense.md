# GFZ: coarse-only versus 1 cm filtering

The two routes produce different predictions on the same originals. Both remaps passed complete coverage and preserved original coordinates, record order, source attributes, missing-value metadata and CRS.

| Model | Original points | IDs: coarse / dense | Different tree assignments after ID alignment | Tree/background flips | Semantic differences |
| --- | ---: | ---: | ---: | ---: | ---: |
| SAT | 16,426,961 | 102 / 100 | 6,821 (0.0415%) | 301 (0.0018%) | 292 (0.0018%) |
| FM | 16,426,961 | 114 / 114 | 556 (0.0034%) | 296 (0.0018%) | 624 (0.0038%) |

The tree-assignment difference includes tree/background flips. It is the residual after an optimal one-to-one mapping between positive IDs; numeric renumbering alone does not count as a difference.

ForestMamba score values (`PredScore_FM`) differ at 61,702 points (0.3756%); the mean absolute difference across all originals is 0.00013037. Instance counts describe segmentation groups, not verified biological tree counts.

Routes:

- Coarse: existing independently segmented 10 cm tiles -> coarse filter/reconcile -> originals, with a 17.3205 cm original remap radius.
- Dense: the same segmented tiles -> 1 cm targets, with a 17.3205 cm transfer radius -> filter/reconcile -> originals, with a 1.73205 cm original remap radius.

Both use centroid ownership, overlap threshold 0.3, the same four tile cores/buffers and eight originals. Tree correspondence remains 5 cm and duplicate/recovery support remains 1 cm. The dense route was regenerated with the exact existing candidate image; previous dense outputs were not used.

Filtering on different geometry can change centroid ownership, recovery and reconciliation. Nearest-neighbor transfer through an intermediate geometry is also not equivalent to direct transfer. The test measures their combined effect, without claiming which segmentation is biologically better.

The pinned image, stage resource statistics, coverage and comparison results are
recorded in [the machine-readable summary](2026-10-14-coarse-vs-dense.json).
The local `smarttile-coarse-vs-dense-20261014` artifact directory contains full
SAT/FM contingency tables, per-file preservation checks, exact commands, input
hashes, the independent comparator and its seven passing self-checks. Outputs
are under `coarse/<model>/original_with_predictions/` and
`dense/<model>/original_with_predictions/`.

No runtime source, wrapper or image changes were needed for this comparison.
