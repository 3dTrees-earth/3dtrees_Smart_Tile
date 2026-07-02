#!/usr/bin/env python3
"""Point deduplication helpers for SmartTile merged products."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def deduplicate_points(
    points: np.ndarray,
    instances: np.ndarray,
    extra_dims: Dict[str, np.ndarray],
    tolerance: float = 0.01,
    grid_size: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Remove duplicate points from overlapping tiles.

    Duplicate keys are computed on a tolerance grid. When duplicate points exist,
    the point with the higher instance ID is kept.
    """
    n_points = len(points)
    scale = 1.0 / tolerance

    min_coords = points.min(axis=0)
    grid_indices = ((points[:, :2] - min_coords[:2]) / grid_size).astype(np.int32)

    max_grid_y = grid_indices[:, 1].max() + 1
    cell_keys = grid_indices[:, 0] * max_grid_y + grid_indices[:, 1]

    rounded = np.floor(points * scale).astype(np.int64)
    point_hash = rounded[:, 0] + rounded[:, 1] * 73856093 + rounded[:, 2] * 19349669

    sort_order = np.lexsort((-instances, point_hash, cell_keys))

    sorted_cell_keys = cell_keys[sort_order]
    sorted_point_hash = point_hash[sort_order]

    is_duplicate = np.zeros(n_points, dtype=bool)
    is_duplicate[1:] = (sorted_cell_keys[1:] == sorted_cell_keys[:-1]) & (
        sorted_point_hash[1:] == sorted_point_hash[:-1]
    )

    keep_mask = np.ones(n_points, dtype=bool)
    keep_mask[sort_order[is_duplicate]] = False

    unique_points = points[keep_mask]
    unique_instances = instances[keep_mask]
    unique_extras = {name: arr[keep_mask] for name, arr in extra_dims.items()}

    return unique_points, unique_instances, unique_extras
