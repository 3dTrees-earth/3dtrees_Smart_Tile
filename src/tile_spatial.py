#!/usr/bin/env python3
"""Spatial geometry helpers for SmartTile tile merge/remap workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import laspy
import numpy as np


Bounds = Tuple[float, float, float, float]


def compute_tile_bounds(points: np.ndarray) -> Bounds:
    """Return the XY bounding box of a point cloud."""
    return (
        points[:, 0].min(),
        points[:, 0].max(),
        points[:, 1].min(),
        points[:, 1].max(),
    )


def get_tile_bounds_from_header(filepath: Path) -> Optional[Bounds]:
    """Read XY bounds from a LAS/LAZ header without loading points."""
    try:
        with laspy.open(str(filepath), laz_backend=laspy.LazBackend.LazrsParallel) as las:
            return (las.header.x_min, las.header.x_max, las.header.y_min, las.header.y_max)
    except Exception:
        return None


def find_overlap_region(bounds_a: Bounds, bounds_b: Bounds) -> Optional[Bounds]:
    """Return the XY overlap region between two bounding boxes."""
    minx_a, maxx_a, miny_a, maxy_a = bounds_a
    minx_b, maxx_b, miny_b, maxy_b = bounds_b
    overlap = (
        max(minx_a, minx_b),
        min(maxx_a, maxx_b),
        max(miny_a, miny_b),
        min(maxy_a, maxy_b),
    )
    if overlap[0] < overlap[1] and overlap[2] < overlap[3]:
        return overlap
    return None


def compute_centroids_vectorized(points: np.ndarray, instances: np.ndarray) -> Dict[int, np.ndarray]:
    """Compute positive-instance centroids with sorting and cumulative sums."""
    valid_mask = instances > 0
    valid_points = points[valid_mask]
    valid_instances = instances[valid_mask]
    if len(valid_instances) == 0:
        return {}

    sort_idx = np.argsort(valid_instances)
    sorted_instances = valid_instances[sort_idx]
    sorted_points = valid_points[sort_idx]
    unique_instances, first_indices, counts = np.unique(
        sorted_instances, return_index=True, return_counts=True
    )

    cumsum = np.zeros((len(sorted_points) + 1, 3), dtype=np.float64)
    cumsum[1:] = np.cumsum(sorted_points, axis=0)
    centroids = {}
    for inst_id, start_idx, count in zip(unique_instances, first_indices, counts):
        end_idx = start_idx + count
        centroids[int(inst_id)] = (cumsum[end_idx] - cumsum[start_idx]) / count
    return centroids


def _edge_alignment_score(
    overlap_span: float,
    size_a: float,
    size_b: float,
    min_a: float,
    max_a: float,
    min_b: float,
    max_b: float,
) -> Tuple[float, float]:
    ratio_a = overlap_span / size_a if size_a > 0 else 0.0
    ratio_b = overlap_span / size_b if size_b > 0 else 0.0
    max_ratio = max(ratio_a, ratio_b)
    axis_alignment = (
        1.0
        if (ratio_a > 0.8 and ratio_b > 0.8)
        or (min_b == min_a and max_b == max_a)
        or (max_ratio > 0.8)
        else 0.5
    )
    edge_tolerance = 0.1
    low_edge_align = abs(min_a - min_b) < edge_tolerance
    high_edge_align = abs(max_a - max_b) < edge_tolerance
    edge_alignment = (
        2.0
        if (low_edge_align and high_edge_align)
        else (1.0 if (low_edge_align or high_edge_align) else 0.0)
    )
    return axis_alignment, edge_alignment


def find_spatial_neighbors(
    tile_boundary: Bounds,
    tile_name: str,
    all_tiles: Dict[str, Bounds],
    tolerance: float = 1.0,
) -> Dict[str, Optional[str]]:
    """Find east/west/north/south neighbors from actual spatial overlaps."""
    minx_a, maxx_a, miny_a, maxy_a = tile_boundary
    tile_width_a = maxx_a - minx_a
    tile_height_a = maxy_a - miny_a
    neighbors = {"east": None, "west": None, "north": None, "south": None}
    candidates = {"east": [], "west": [], "north": [], "south": []}

    for other_name, other_bounds in all_tiles.items():
        if other_name == tile_name:
            continue
        overlap = find_overlap_region(tile_boundary, other_bounds)
        if overlap is None:
            continue

        minx_b, maxx_b, miny_b, maxy_b = other_bounds
        overlap_minx, overlap_maxx, overlap_miny, overlap_maxy = overlap
        overlap_width = overlap_maxx - overlap_minx
        overlap_height = overlap_maxy - overlap_miny
        overlap_area = overlap_width * overlap_height

        if minx_b > minx_a and minx_b <= maxx_a + tolerance and overlap_width >= tolerance:
            if not (maxy_b < miny_a or miny_b > maxy_a):
                axis_alignment, edge_alignment = _edge_alignment_score(
                    overlap_height, tile_height_a, maxy_b - miny_b, miny_a, maxy_a, miny_b, maxy_b
                )
                candidates["east"].append((overlap_area, axis_alignment, minx_b - minx_a, edge_alignment, other_name))

        if minx_b < minx_a and maxx_b >= minx_a - tolerance and overlap_width >= tolerance:
            if not (maxy_b < miny_a or miny_b > maxy_a):
                axis_alignment, edge_alignment = _edge_alignment_score(
                    overlap_height, tile_height_a, maxy_b - miny_b, miny_a, maxy_a, miny_b, maxy_b
                )
                candidates["west"].append((overlap_area, axis_alignment, minx_a - minx_b, edge_alignment, other_name))

        if miny_b > miny_a and miny_b <= maxy_a + tolerance and overlap_height >= tolerance:
            if not (maxx_b < minx_a or minx_b > maxx_a):
                axis_alignment, edge_alignment = _edge_alignment_score(
                    overlap_width, tile_width_a, maxx_b - minx_b, minx_a, maxx_a, minx_b, maxx_b
                )
                candidates["north"].append((overlap_area, axis_alignment, miny_b - miny_a, edge_alignment, other_name))

        if miny_b < miny_a and maxy_b >= miny_a - tolerance and overlap_height >= tolerance:
            if not (maxx_b < minx_a or minx_b > maxx_a):
                axis_alignment, edge_alignment = _edge_alignment_score(
                    overlap_width, tile_width_a, maxx_b - minx_b, minx_a, maxx_a, minx_b, maxx_b
                )
                candidates["south"].append((overlap_area, axis_alignment, miny_a - miny_b, edge_alignment, other_name))

    for direction, values in candidates.items():
        if not values:
            continue
        best = max(values, key=lambda x: (x[1], x[3], x[0], -x[2]))
        neighbors[direction] = best[4]

    return neighbors


def filter_by_centroid_in_buffer(
    points: np.ndarray,
    instances: np.ndarray,
    boundary: Bounds,
    tile_name: str,
    all_tiles: Dict[str, Bounds],
    buffer: float = 10.0,
    precomputed_neighbors: Optional[Dict[str, Optional[str]]] = None,
) -> Tuple[Set[int], Dict[int, str]]:
    """Return instances whose centroid falls in an overlapping tile buffer zone."""
    min_x, max_x, min_y, max_y = boundary
    neighbors = (
        {direction: precomputed_neighbors.get(direction) for direction in ("east", "west", "north", "south")}
        if precomputed_neighbors is not None
        else find_spatial_neighbors(boundary, tile_name, all_tiles, tolerance=buffer)
    )

    buf_min_x = min_x + (buffer if neighbors["west"] is not None else 0)
    buf_max_x = max_x - (buffer if neighbors["east"] is not None else 0)
    buf_min_y = min_y + (buffer if neighbors["south"] is not None else 0)
    buf_max_y = max_y - (buffer if neighbors["north"] is not None else 0)

    instances_to_remove = set()
    instance_buffer_direction = {}
    for inst_id, centroid in compute_centroids_vectorized(points, instances).items():
        if inst_id <= 0:
            continue
        cx, cy = centroid[0], centroid[1]
        in_west_buffer = neighbors["west"] is not None and cx < buf_min_x
        in_east_buffer = neighbors["east"] is not None and cx > buf_max_x
        in_south_buffer = neighbors["south"] is not None and cy < buf_min_y
        in_north_buffer = neighbors["north"] is not None and cy > buf_max_y

        if in_west_buffer or in_east_buffer or in_south_buffer or in_north_buffer:
            instances_to_remove.add(inst_id)
            if in_west_buffer:
                instance_buffer_direction[inst_id] = "west"
            elif in_south_buffer:
                instance_buffer_direction[inst_id] = "south"
            elif in_east_buffer:
                instance_buffer_direction[inst_id] = "east"
            else:
                instance_buffer_direction[inst_id] = "north"

    return instances_to_remove, instance_buffer_direction


def get_border_region_mask(
    points: np.ndarray,
    boundary: Bounds,
    inner_dist: float,
    outer_dist: float,
    neighbors: Dict[str, Optional[str]],
) -> np.ndarray:
    """Return a mask for points in the edge band for directions with neighbors."""
    min_x, max_x, min_y, max_y = boundary
    x, y = points[:, 0], points[:, 1]
    mask = np.zeros(len(points), dtype=bool)

    if neighbors.get("east") is not None:
        mask |= (x > max_x - outer_dist) & (x <= max_x - inner_dist)
    if neighbors.get("west") is not None:
        mask |= (x >= min_x + inner_dist) & (x < min_x + outer_dist)
    if neighbors.get("north") is not None:
        mask |= (y > max_y - outer_dist) & (y <= max_y - inner_dist)
    if neighbors.get("south") is not None:
        mask |= (y >= min_y + inner_dist) & (y < min_y + outer_dist)
    return mask
