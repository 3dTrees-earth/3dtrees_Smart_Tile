"""Shared coverage policy and bounded diagnostics for point remapping."""

from __future__ import annotations

from typing import Dict

import numpy as np


MIN_ORIGINAL_REMAP_MATCH_FRACTION = 0.99

# Nearest-neighbour distances are recorded as multiples of the configured
# tolerance. This keeps diagnostics bounded regardless of input size while
# still making p50/p95/p99 useful across differently scaled datasets.
DISTANCE_RATIO_BOUNDS = np.array(
    [0.0, 0.10, 0.25, 0.50, 0.75, 1.0, 2.0, 5.0, 10.0, np.inf],
    dtype=np.float64,
)


def empty_coverage() -> Dict[str, object]:
    return {
        "matched": 0,
        "total": 0,
        "histogram": np.zeros(len(DISTANCE_RATIO_BOUNDS) - 1, dtype=np.int64),
        "finite_distances": 0,
        "max_distance": 0.0,
    }


def distance_coverage(distances: np.ndarray, tolerance: float) -> Dict[str, object]:
    stats = empty_coverage()
    values = np.asarray(distances, dtype=np.float64)
    finite = values[np.isfinite(values)]
    stats["total"] = int(values.size)
    if finite.size:
        ratios = finite / float(tolerance)
        stats["histogram"] = np.histogram(ratios, bins=DISTANCE_RATIO_BOUNDS)[0]
        stats["finite_distances"] = int(finite.size)
        stats["max_distance"] = float(np.max(finite))
    stats["matched"] = int(np.count_nonzero(values <= tolerance))
    return stats


def merge_coverage(target: Dict[str, object], addition: Dict[str, object]) -> None:
    target["matched"] = int(target["matched"]) + int(addition["matched"])
    target["total"] = int(target["total"]) + int(addition["total"])
    target["finite_distances"] = (
        int(target["finite_distances"]) + int(addition["finite_distances"])
    )
    target["histogram"] = np.asarray(target["histogram"]) + np.asarray(addition["histogram"])
    target["max_distance"] = max(
        float(target["max_distance"]),
        float(addition["max_distance"]),
    )


def approximate_percentile_bound(
    stats: Dict[str, object],
    percentile: float,
    tolerance: float,
) -> str:
    count = int(stats["finite_distances"])
    if count <= 0:
        return "n/a"
    target = max(1, int(np.ceil(count * percentile)))
    cumulative = 0
    for idx, amount in enumerate(np.asarray(stats["histogram"])):
        cumulative += int(amount)
        if cumulative >= target:
            upper_ratio = DISTANCE_RATIO_BOUNDS[idx + 1]
            if np.isinf(upper_ratio):
                return f">{DISTANCE_RATIO_BOUNDS[idx] * tolerance:.6g}m"
            return f"<={upper_ratio * tolerance:.6g}m"
    return "n/a"


def coverage_diagnostic(
    collection: object,
    stats: Dict[str, object],
    tolerance: float,
) -> str:
    matched = int(stats["matched"])
    total = int(stats["total"])
    fraction = matched / total if total else 1.0
    return (
        f"{collection}: {matched:,}/{total:,} matched ({fraction:.3%}); "
        f"unmatched={total - matched:,}; nearest-distance "
        f"p50 {approximate_percentile_bound(stats, 0.50, tolerance)}, "
        f"p95 {approximate_percentile_bound(stats, 0.95, tolerance)}, "
        f"p99 {approximate_percentile_bound(stats, 0.99, tolerance)}, "
        f"max={float(stats['max_distance']):.6g}m"
    )
