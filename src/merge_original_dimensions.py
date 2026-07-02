#!/usr/bin/env python3
"""Add original point-cloud dimensions back onto merged SmartTile products."""

from __future__ import annotations

import gc
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional

import laspy
import numpy as np
from scipy.spatial import cKDTree

from dimension_transfer import plan_dimension_transfer
from point_cloud_metadata import (
    extra_bytes_params_from_dimension_info,
    point_cloud_files,
)


def add_original_dimensions_to_merged(
    merged_laz: Path,
    original_input_dir: Path,
    output_path: Path,
    tolerance: float = 0.1,
    retile_buffer: float = 2.0,
    distance_threshold: Optional[float] = None,
    num_threads: int = 4,
) -> None:
    """Enrich a merged point cloud with dimensions from original input files."""
    if output_path.resolve() == Path(merged_laz).resolve():
        raise ValueError("output_path must differ from merged_laz to avoid overwriting input")

    original_files = point_cloud_files(original_input_dir)
    if not original_files:
        print("  No original input files found; skipping merged-with-originals output.", flush=True)
        return

    print(f"\n{'=' * 60}", flush=True)
    print("Adding original-file dimensions to merged point cloud", flush=True)
    print(f"{'=' * 60}", flush=True)

    merged = laspy.read(str(merged_laz), laz_backend=laspy.LazBackend.LazrsParallel)
    n_merged = len(merged.points)
    merged_points = np.column_stack([merged.x, merged.y, merged.z])
    merged_dim_names = set(merged.point_format.dimension_names)
    for dim in merged.point_format.extra_dimensions:
        merged_dim_names.add(dim.name)

    skip_core = {"X", "Y", "Z"}
    orig_dims: Dict[str, np.dtype] = {}
    orig_extra_dim_info: Dict[str, object] = {}
    for orig_path in original_files:
        with laspy.open(str(orig_path), laz_backend=laspy.LazBackend.LazrsParallel) as f:
            pf = f.header.point_format
            pt_dtype = None
            try:
                one = f.read_points(1)
                if one is not None and one.size > 0:
                    arr = getattr(one, "array", one)
                    dt = getattr(arr, "dtype", None)
                    if dt is not None and getattr(dt, "names", None) is not None:
                        pt_dtype = dt
                if one is not None and one.size > 0:
                    for dim_name in pf.dimension_names:
                        if dim_name in skip_core or dim_name in orig_dims:
                            continue
                        dim_view = getattr(one, dim_name, None)
                        if dim_view is not None and hasattr(dim_view, "dtype"):
                            orig_dims[dim_name] = np.dtype(dim_view.dtype)
                        elif pt_dtype is not None and dim_name in pt_dtype.names:
                            orig_dims[dim_name] = pt_dtype.fields[dim_name][0]
                else:
                    one = None
            except Exception:
                one = None
            if one is None or pt_dtype is None:
                for dim_name in pf.dimension_names:
                    if dim_name in skip_core or dim_name in orig_dims:
                        continue
                    orig_dims[dim_name] = np.float64
            for dim in pf.extra_dimensions:
                if dim.name in skip_core:
                    continue
                if dim.name not in orig_dims:
                    orig_dims[dim.name] = dim.dtype
                if dim.name not in orig_extra_dim_info:
                    orig_extra_dim_info[dim.name] = dim

    transfer_plan = plan_dimension_transfer(
        orig_dims,
        merged_dim_names,
        lambda name: getattr(merged, name, None),
        skip=skip_core,
    )
    dims_to_add = transfer_plan.output_dtypes
    orig_dim_to_read = transfer_plan.output_to_source

    if not dims_to_add:
        print("  No dimensions to add or replace from originals; writing copy of merged file.", flush=True)
        merged.write(str(output_path), do_compress=True, laz_backend=laspy.LazBackend.LazrsParallel)
        del merged
        gc.collect()
        return

    for dim_name in sorted(transfer_plan.add_new.keys()):
        print(f"  Adding dimension: {dim_name}", flush=True)
    for dim_name in sorted(transfer_plan.overwrite.keys()):
        print(f"  Replacing dimension (was empty/constant): {dim_name}", flush=True)
    for orig_name, out_name in sorted(transfer_plan.renamed.items()):
        print(f"  Adding dimension from originals (collision with merged): {orig_name} -> {out_name}", flush=True)

    spatial_buffer = max(tolerance * 2, 1.0) + retile_buffer
    max_dist = distance_threshold if distance_threshold is not None else spatial_buffer

    best_dist = np.full(n_merged, np.inf, dtype=np.float64)
    new_arrays: Dict[str, np.ndarray] = {
        name: np.zeros(n_merged, dtype=dtype)
        for name, dtype in dims_to_add.items()
    }

    def process_one_original(orig_path: Path):
        try:
            orig_las = laspy.read(str(orig_path), laz_backend=laspy.LazBackend.LazrsParallel)
        except Exception:
            return None
        bounds = (
            orig_las.header.x_min,
            orig_las.header.x_max,
            orig_las.header.y_min,
            orig_las.header.y_max,
        )
        orig_points = np.column_stack([orig_las.x, orig_las.y, orig_las.z])
        mask = (
            (merged_points[:, 0] >= bounds[0] - spatial_buffer)
            & (merged_points[:, 0] <= bounds[1] + spatial_buffer)
            & (merged_points[:, 1] >= bounds[2] - spatial_buffer)
            & (merged_points[:, 1] <= bounds[3] + spatial_buffer)
        )
        merged_idx = np.where(mask)[0]
        if len(merged_idx) == 0:
            del orig_las
            return None
        tree = cKDTree(orig_points)
        distances, orig_idx = tree.query(merged_points[merged_idx], k=1, workers=1)
        if distances.ndim == 2:
            distances = distances[:, 0]
            orig_idx = orig_idx[:, 0]
        orig_dim_arrays = {}
        for out_name, orig_name in orig_dim_to_read.items():
            arr = getattr(orig_las, orig_name, None)
            if arr is not None:
                orig_dim_arrays[out_name] = np.asarray(arr)
        del orig_las
        return (merged_idx, distances, orig_idx, orig_dim_arrays)

    if num_threads > 1:
        print(f"  Processing {len(original_files)} original files with {num_threads} workers...", flush=True)
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = list(executor.map(process_one_original, original_files))
    else:
        results = [process_one_original(path) for path in original_files]

    for result in results:
        if result is None:
            continue
        merged_idx, distances, orig_idx, orig_dim_arrays = result
        within_max = distances <= max_dist
        better = distances < best_dist[merged_idx]
        accept = within_max & better
        if np.any(accept):
            acc_merged = merged_idx[accept]
            acc_orig = orig_idx[accept]
            best_dist[acc_merged] = distances[accept]
            for dim_name, arr_np in orig_dim_arrays.items():
                new_arrays[dim_name][acc_merged] = arr_np[acc_orig]
    gc.collect()

    for name, arr in new_arrays.items():
        arr = np.asarray(arr)
        vmin, vmax = float(np.min(arr)), float(np.max(arr))
        n_nonzero = int(np.count_nonzero(arr))
        if vmin == vmax or n_nonzero == 0:
            print(f"  Warning: {name} has no variation (min=max={vmin}, non-zero={n_nonzero})", flush=True)
        else:
            print(f"  {name}: min={vmin}, max={vmax}, non-zero={n_nonzero}", flush=True)

    extra_params = []
    for name, dtype in transfer_plan.add_new.items():
        if name in orig_extra_dim_info:
            extra_params.append(
                extra_bytes_params_from_dimension_info(orig_extra_dim_info[name], name=name)
            )
        else:
            extra_params.append(laspy.ExtraBytesParams(name=name, type=dtype))
    for orig_name, out_name in transfer_plan.renamed.items():
        dtype = transfer_plan.output_dtypes[out_name]
        if orig_name is not None and orig_name in orig_extra_dim_info:
            extra_params.append(
                extra_bytes_params_from_dimension_info(orig_extra_dim_info[orig_name], name=out_name)
            )
        else:
            extra_params.append(laspy.ExtraBytesParams(name=out_name, type=dtype))
    if extra_params:
        merged.add_extra_dims(extra_params)

    for name, arr in new_arrays.items():
        arr = np.asarray(arr)
        try:
            target_dtype = getattr(merged.points, name).dtype
            if arr.dtype != target_dtype:
                arr = arr.astype(target_dtype)
        except (AttributeError, KeyError, TypeError):
            pass
        setattr(merged, name, arr)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.write(str(output_path), do_compress=True, laz_backend=laspy.LazBackend.LazrsParallel)
    del merged
    gc.collect()
    print(f"  Saved merged with original dimensions: {output_path}", flush=True)
