#!/usr/bin/env python3
"""Tile loading helpers for SmartTile merge."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import laspy
import numpy as np

from instance_labels import validate_prediction_instance_labels
from point_cloud_metadata import point_cloud_source_key
from tile_spatial import compute_tile_bounds, filter_by_centroid_in_buffer


@dataclass
class TileData:
    """Container for tile point-cloud data used during merge."""

    name: str
    points: np.ndarray
    instances: np.ndarray
    boundary: Tuple[float, float, float, float]
    extra_dims: Dict[str, np.ndarray] = field(default_factory=dict)


def merge_tile_name(filepath: Path) -> str:
    """Return the logical tile id shared by LAZ/LAS/COPC merge inputs."""
    tile_name = point_cloud_source_key(filepath)
    for suffix in ["_segmented_remapped", "_segmented", "_remapped"]:
        tile_name = tile_name.replace(suffix, "")
    return tile_name


def load_tile(
    filepath: Path,
    all_tiles: Dict[str, Tuple[float, float, float, float]],
    buffer: float,
    neighbors_by_tile: Optional[Dict[str, Dict[str, Optional[str]]]] = None,
    chunk_size: int = 1_000_000,
    instance_dimension: str = "PredInstance",
) -> Optional[Tuple[TileData, Set[int], Set[int], Dict[int, str]]]:
    """Load one segmented tile and compute buffer-filter metadata."""
    print(f"Loading {filepath.name}...")

    try:
        with laspy.open(str(filepath), laz_backend=laspy.LazBackend.Lazrs) as reader:
            n_points = reader.header.point_count
            header_extra_dims = {dim.name: dim for dim in reader.header.point_format.extra_dimensions}
            has_instance_dim = instance_dimension in header_extra_dims
            has_tree_id = "treeID" in header_extra_dims

            points = np.empty((n_points, 3), dtype=np.float64)
            instances = np.zeros(n_points, dtype=np.int32)
            extra_dims: Dict[str, np.ndarray] = {}
            for dim in reader.header.point_format.extra_dimensions:
                if dim.name == instance_dimension or (not has_instance_dim and dim.name == "treeID"):
                    continue
                extra_dims[dim.name] = np.zeros(n_points, dtype=dim.dtype)

            offset = 0
            for chunk in reader.chunk_iterator(chunk_size):
                chunk_len = len(chunk)
                end = offset + chunk_len

                points[offset:end, 0] = chunk.x
                points[offset:end, 1] = chunk.y
                points[offset:end, 2] = chunk.z

                if has_instance_dim:
                    instances[offset:end] = getattr(chunk, instance_dimension)
                elif has_tree_id:
                    instances[offset:end] = chunk.treeID

                for dim_name in extra_dims:
                    extra_dims[dim_name][offset:end] = getattr(chunk, dim_name)

                offset = end
    except Exception as exc:
        print(f"  Error loading {filepath}: {exc}")
        return None

    if not has_instance_dim and not has_tree_id:
        print(f"  Warning: No instance attribute ({instance_dimension}/treeID) found in {filepath}")
    elif has_instance_dim:
        validate_prediction_instance_labels(instances, instance_dimension, filepath)
    else:
        validate_prediction_instance_labels(instances, "treeID", filepath)

    boundary = compute_tile_bounds(points)
    tile_name = merge_tile_name(filepath)

    neighbors_for_tile = neighbors_by_tile.get(tile_name) if neighbors_by_tile is not None else None
    instances_to_remove, instance_buffer_direction = filter_by_centroid_in_buffer(
        points,
        instances,
        boundary,
        tile_name,
        all_tiles,
        buffer,
        precomputed_neighbors=neighbors_for_tile,
    )
    kept_instances = set(np.unique(instances)) - instances_to_remove - {0}

    print(
        f"  {len(points):,} points, {len(kept_instances)} instances kept, "
        f"{len(instances_to_remove)} filtered"
    )

    return (
        TileData(
            name=tile_name,
            points=points,
            instances=instances,
            boundary=boundary,
            extra_dims=extra_dims,
        ),
        instances_to_remove,
        kept_instances,
        instance_buffer_direction,
    )


def load_tile_wrapper(args):
    """ProcessPool wrapper for load_tile."""
    filepath, tile_boundaries, buffer, neighbors_by_tile, instance_dimension = args
    return load_tile(filepath, tile_boundaries, buffer, neighbors_by_tile, instance_dimension=instance_dimension)
