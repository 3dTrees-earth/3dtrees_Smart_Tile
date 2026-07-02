#!/usr/bin/env python3
"""Instance-overlap algorithms for SmartTile tile merging."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def compute_ff3d_overlap_ratios(
    instances_a: np.ndarray,
    instances_b: np.ndarray,
    points_a: np.ndarray,
    points_b: np.ndarray,
    correspondence_tolerance: float = 0.1,
) -> Tuple[Dict[Tuple[int, int], float], Dict[int, int], Dict[int, int]]:
    """Compute FF3D-style overlap ratios between instance pairs.

    The metric is `max(intersection / size_a, intersection / size_b)`. Point
    correspondence is approximated by hashing points onto the requested
    tolerance grid, which keeps the overlap check linear after sorting the
    lookup cloud.
    """
    unique_a, counts_a = np.unique(instances_a[instances_a > 0], return_counts=True)
    unique_b, counts_b = np.unique(instances_b[instances_b > 0], return_counts=True)
    size_a = dict(zip(unique_a, counts_a))
    size_b = dict(zip(unique_b, counts_b))

    scale = 1.0 / correspondence_tolerance

    grid_b = np.floor(points_b * scale).astype(np.int64)
    hash_b = grid_b[:, 0] + grid_b[:, 1] * 73856093 + grid_b[:, 2] * 19349669

    grid_a = np.floor(points_a * scale).astype(np.int64)
    hash_a = grid_a[:, 0] + grid_a[:, 1] * 73856093 + grid_a[:, 2] * 19349669

    sort_idx_b = np.argsort(hash_b)
    sorted_hash_b = hash_b[sort_idx_b]
    sorted_inst_b = instances_b[sort_idx_b]

    unique_hash_b, first_idx = np.unique(sorted_hash_b, return_index=True)
    unique_inst_b = sorted_inst_b[first_idx]

    insert_pos = np.searchsorted(unique_hash_b, hash_a)
    insert_pos_clamped = np.clip(insert_pos, 0, len(unique_hash_b) - 1)
    matches_mask = unique_hash_b[insert_pos_clamped] == hash_a

    matched_inst_b = np.zeros(len(hash_a), dtype=instances_b.dtype)
    matched_inst_b[matches_mask] = unique_inst_b[insert_pos_clamped[matches_mask]]

    valid_mask = matches_mask & (instances_a > 0) & (matched_inst_b > 0)
    valid_inst_a = instances_a[valid_mask]
    valid_inst_b = matched_inst_b[valid_mask]

    if len(valid_inst_a) > 0:
        max_inst = max(instances_a.max(), instances_b.max()) + 1
        pair_keys = valid_inst_a.astype(np.int64) * max_inst + valid_inst_b.astype(np.int64)
        unique_pairs, pair_counts = np.unique(pair_keys, return_counts=True)
        intersection_counts = {
            (int(key // max_inst), int(key % max_inst)): count
            for key, count in zip(unique_pairs, pair_counts)
        }
    else:
        intersection_counts = {}

    overlap_ratios = {}
    for (inst_a, inst_b), intersection in intersection_counts.items():
        ratio_a = intersection / size_a[inst_a] if size_a.get(inst_a, 0) > 0 else 0
        ratio_b = intersection / size_b[inst_b] if size_b.get(inst_b, 0) > 0 else 0
        overlap_ratios[(inst_a, inst_b)] = max(ratio_a, ratio_b)

    return overlap_ratios, size_a, size_b
