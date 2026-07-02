#!/usr/bin/env python3
"""Load and mutate already-merged SmartTile point clouds."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import laspy
import numpy as np

from instance_labels import validate_prediction_instance_labels
from merge_small_instances import merge_small_volume_instances
from point_cloud_metadata import extra_bytes_params_from_dimension_info


def load_merged_file(
    merged_file: Path,
    chunk_size: int = 1_000_000,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, laspy.ExtraBytesParams]]:
    """Load merged point coordinates and all non-XYZ dimensions from a LAZ file."""
    print(f"Loading existing merged file: {merged_file}")

    try:
        with laspy.open(str(merged_file), laz_backend=laspy.LazBackend.LazrsParallel) as f:
            n_points = f.header.point_count
            points = np.empty((n_points, 3), dtype=np.float64)
            all_dims: Dict[str, np.ndarray] = {}
            extra_dim_params: Dict[str, laspy.ExtraBytesParams] = {}
            offset = 0
            for chunk in f.chunk_iterator(chunk_size):
                chunk_len = len(chunk)
                end = offset + chunk_len
                points[offset:end, 0] = chunk.x
                points[offset:end, 1] = chunk.y
                points[offset:end, 2] = chunk.z
                for dim_name in f.header.point_format.dimension_names:
                    if dim_name in ("X", "Y", "Z"):
                        continue
                    arr = getattr(chunk, dim_name, None)
                    if arr is not None:
                        if dim_name not in all_dims:
                            all_dims[dim_name] = np.zeros(n_points, dtype=arr.dtype)
                        all_dims[dim_name][offset:end] = arr
                for dim in f.header.point_format.extra_dimensions:
                    if dim.name not in extra_dim_params:
                        extra_dim_params[dim.name] = extra_bytes_params_from_dimension_info(dim)
                    if dim.name not in all_dims:
                        all_dims[dim.name] = np.zeros(n_points, dtype=dim.dtype)
                    all_dims[dim.name][offset:end] = getattr(chunk, dim.name)
                offset = end

        print(f"  Loaded {len(points):,} points")
        if all_dims:
            print(f"  Dimensions from merged: {', '.join(sorted(all_dims.keys()))}")

        return points, all_dims, extra_dim_params
    except Exception as exc:
        raise ValueError(f"Error loading merged file {merged_file}: {exc}") from exc


def reassign_small_instances_in_dims(
    points: np.ndarray,
    all_dims: Dict[str, np.ndarray],
    instance_dimension: str,
    min_cluster_size: int = 250,
    hull_point_threshold: int = 5000,
    max_volume_for_merge: float = 5.0,
    max_search_radius: float = 5.0,
    num_threads: int = 1,
    verbose: bool = False,
) -> Dict[str, int]:
    """Reassign small instances in one loaded segmented/merged point cloud dimension."""
    if instance_dimension not in all_dims:
        raise ValueError(
            f"Cannot pre-remap reassign instances: dimension '{instance_dimension}' "
            f"not found. Available dimensions: {', '.join(sorted(all_dims.keys()))}"
        )

    original_instances = np.asarray(all_dims[instance_dimension])
    validate_prediction_instance_labels(
        original_instances,
        instance_dimension,
        Path("<loaded point cloud>"),
    )

    reassigned_instances = original_instances.astype(np.int64, copy=True)
    before_unique = np.unique(reassigned_instances[reassigned_instances > 0])

    reassigned_instances, _ = merge_small_volume_instances(
        points,
        reassigned_instances,
        min_points_for_hull_check=hull_point_threshold,
        min_cluster_size=min_cluster_size,
        max_volume_for_merge=max_volume_for_merge,
        max_search_radius=max_search_radius,
        num_threads=num_threads,
        verbose=verbose,
    )

    changed_points = int(np.count_nonzero(reassigned_instances != original_instances))
    after_unique = np.unique(reassigned_instances[reassigned_instances > 0])
    all_dims[instance_dimension] = reassigned_instances.astype(original_instances.dtype, copy=False)

    return {
        "changed_points": changed_points,
        "instances_before": int(len(before_unique)),
        "instances_after": int(len(after_unique)),
        "instances_removed": int(len(before_unique) - len(after_unique)),
    }
