#!/usr/bin/env python3
"""Retile/remap merged SmartTile predictions back onto original point clouds."""

from __future__ import annotations

import gc
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import laspy
import numpy as np
from scipy.spatial import cKDTree

from dimension_transfer import next_available_suffix, suffixes_for_collision
from instance_labels import cast_instances_for_output, instance_extra_bytes_params
from point_cloud_metadata import (
    copy_single_source_header,
    extra_bytes_params_from_dimension_info,
    extra_bytes_params_from_params,
    point_cloud_files,
    raw_point_cloud_files,
)


MIN_ORIGINAL_REMAP_MATCH_FRACTION = 0.99


def _match_fraction_is_acceptable(matched: int, total: int, min_fraction: float) -> bool:
    if total <= 0:
        return True
    return (matched / total) >= min_fraction


def _match_quality_message(
    matched: int,
    total: int,
    tolerance: float,
    min_fraction: float,
    context: str = "points",
) -> str:
    min_pct = min_fraction * 100.0
    return (
        f"Merged points matched {matched:,}/{total:,} {context} "
        f"within tolerance {tolerance} m; requires at least {min_pct:.1f}%"
    )


def _process_single_tile(args):
    """Process one original tile when retiling merged predictions."""
    (
        orig_file,
        output_file,
        merged_points,
        merged_instances,
        merged_extra_dims,
        merged_extra_dim_params,
        tolerance,
        spatial_buffer,
        kdtree_workers,
        instance_dimension,
    ) = args

    try:
        with laspy.open(str(orig_file), laz_backend=laspy.LazBackend.LazrsParallel) as f:
            bounds = (f.header.x_min, f.header.x_max, f.header.y_min, f.header.y_max)
            n_orig_points = f.header.point_count

        mask = (
            (merged_points[:, 0] >= bounds[0] - spatial_buffer)
            & (merged_points[:, 0] <= bounds[1] + spatial_buffer)
            & (merged_points[:, 1] >= bounds[2] - spatial_buffer)
            & (merged_points[:, 1] <= bounds[3] + spatial_buffer)
        )

        local_merged_points = merged_points[mask]
        local_merged_instances = merged_instances[mask]
        local_merged_extras = {name: arr[mask] for name, arr in merged_extra_dims.items()}

        if len(local_merged_points) == 0:
            return (orig_file.name, 0, n_orig_points, 0, False, "No merged points in tile region")

        local_tree = cKDTree(local_merged_points)
        orig_las = laspy.read(str(orig_file), laz_backend=laspy.LazBackend.LazrsParallel)
        orig_points = np.empty((n_orig_points, 3), dtype=np.float64)
        orig_points[:, 0] = orig_las.x
        orig_points[:, 1] = orig_las.y
        orig_points[:, 2] = orig_las.z

        distances, indices = local_tree.query(orig_points, workers=kdtree_workers)
        matched_mask = distances <= tolerance
        matched = int(np.count_nonzero(matched_mask))
        if matched != n_orig_points:
            return (
                orig_file.name,
                matched,
                n_orig_points,
                0,
                False,
                f"Merged points matched {matched:,}/{n_orig_points:,} "
                f"points within tolerance {tolerance} m",
            )
        new_instances = local_merged_instances[indices]
        new_extras = {name: arr[indices] for name, arr in local_merged_extras.items()}

        new_header = copy_single_source_header(
            orig_las.header,
            preserve_extra_dimensions=False,
        )
        output_las = laspy.LasData(new_header)

        orig_standard_names = set(orig_las.point_format.dimension_names)
        orig_extra_dim_names = {dim.name for dim in orig_las.point_format.extra_dimensions}
        orig_dim_names = orig_standard_names | orig_extra_dim_names
        merged_dim_names = {instance_dimension} | set(new_extras.keys())
        collision = orig_dim_names & merged_dim_names

        used_names = set(orig_dim_names)
        orig_rename = {}
        merged_rename = {}
        for name in sorted(collision):
            original_name, merged_name = suffixes_for_collision(name, used_names)
            orig_rename[name] = original_name
            merged_rename[name] = merged_name

        output_extra_dim_names = {dim.name for dim in output_las.point_format.extra_dimensions}
        extra_dims_to_add = []
        for dim in orig_las.point_format.extra_dimensions:
            out_name = orig_rename.get(dim.name, dim.name)
            if out_name not in output_extra_dim_names:
                extra_dims_to_add.append(extra_bytes_params_from_dimension_info(dim, name=out_name))
                output_extra_dim_names.add(out_name)

        inst_out_name = merged_rename.get(instance_dimension, instance_dimension)
        if inst_out_name not in output_extra_dim_names:
            extra_dims_to_add.append(instance_extra_bytes_params(inst_out_name, new_instances))
            output_extra_dim_names.add(inst_out_name)

        for dim_name, values in new_extras.items():
            out_name = merged_rename.get(dim_name, dim_name)
            if out_name not in output_extra_dim_names:
                if merged_extra_dim_params and dim_name in merged_extra_dim_params:
                    params = merged_extra_dim_params[dim_name]
                    extra_dims_to_add.append(extra_bytes_params_from_params(params, name=out_name))
                else:
                    extra_dims_to_add.append(laspy.ExtraBytesParams(name=out_name, type=values.dtype))
                output_extra_dim_names.add(out_name)

        if extra_dims_to_add:
            output_las.add_extra_dims(extra_dims_to_add)

        for dim_name in orig_las.point_format.dimension_names:
            try:
                if hasattr(orig_las, dim_name):
                    setattr(output_las, dim_name, getattr(orig_las, dim_name))
            except Exception:
                pass
        for dim in orig_las.point_format.extra_dimensions:
            name = dim.name
            out_name = orig_rename.get(name, name)
            if hasattr(orig_las, name):
                try:
                    setattr(output_las, out_name, getattr(orig_las, name))
                except Exception:
                    pass

        setattr(
            output_las,
            merged_rename.get(instance_dimension, instance_dimension),
            cast_instances_for_output(new_instances, instance_dimension),
        )
        for dim_name, values in new_extras.items():
            setattr(output_las, merged_rename.get(dim_name, dim_name), values)

        output_las.write(
            str(output_file),
            do_compress=True,
            laz_backend=laspy.LazBackend.LazrsParallel,
        )

        del orig_las
        del output_las

        unique_inst = len(np.unique(new_instances[new_instances > 0]))
        return (orig_file.name, matched, n_orig_points, unique_inst, True, "OK")

    except Exception as e:
        return (orig_file.name, 0, 0, 0, False, str(e))


def retile_to_original_files(
    merged_points: np.ndarray,
    merged_instances: np.ndarray,
    merged_extra_dims: Dict[str, np.ndarray],
    merged_extra_dim_params: Optional[Dict[str, laspy.ExtraBytesParams]],
    original_tiles_dir: Path,
    output_dir: Path,
    tolerance: float = 0.1,
    num_threads: int = 8,
    chunk_size: int = 1_000_000,
    parallel_tiles: int = 1,
    retile_buffer: float = 2.0,
    instance_dimension: str = "PredInstance",
):
    """Map merged instance IDs back to original tile point clouds."""
    print(f"\n{'=' * 60}", flush=True)
    print("Retiling merged results to original tile files", flush=True)
    print(f"{'=' * 60}", flush=True)

    original_files = point_cloud_files(original_tiles_dir)

    if len(original_files) == 0:
        print(f"  No LAZ/LAS files found in {original_tiles_dir}", flush=True)
        return

    print(f"  Found {len(original_files)} original tile files", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    spatial_buffer = max(tolerance * 2, 1.0) + retile_buffer
    tiles_to_process = []
    skipped = 0
    for orig_file in original_files:
        output_name = orig_file.name.replace(".copc.laz", ".laz")
        output_file = output_dir / output_name
        if output_file.exists():
            skipped += 1
        else:
            tiles_to_process.append((orig_file, output_file))

    if skipped > 0:
        print(f"  Skipping {skipped} already processed tiles", flush=True)

    if len(tiles_to_process) == 0:
        print("  All tiles already processed!", flush=True)
        return

    print(f"  Processing {len(tiles_to_process)} tiles...", flush=True)

    kdtree_workers = -1
    process_args = [
        (
            orig_file,
            output_file,
            merged_points,
            merged_instances,
            merged_extra_dims,
            merged_extra_dim_params,
            tolerance,
            spatial_buffer,
            kdtree_workers,
            instance_dimension,
        )
        for orig_file, output_file in tiles_to_process
    ]

    failures = []
    if parallel_tiles > 1:
        completed = 0
        with ThreadPoolExecutor(max_workers=parallel_tiles) as executor:
            for result in executor.map(_process_single_tile, process_args):
                filename, matched, total, unique_inst, success, message = result
                completed += 1
                match_pct = (matched / total * 100) if total > 0 else 0
                if success:
                    print(
                        f"  [{completed}/{len(tiles_to_process)}] {filename}: "
                        f"{matched:,}/{total:,} matched ({match_pct:.1f}%), "
                        f"{unique_inst} instances",
                        flush=True,
                    )
                else:
                    print(
                        f"  [{completed}/{len(tiles_to_process)}] {filename}: FAILED - {message}",
                        flush=True,
                    )
                    failures.append(f"{filename}: {message}")
    else:
        for i, args in enumerate(process_args):
            filename, matched, total, unique_inst, success, message = _process_single_tile(args)
            if success:
                match_pct = (matched / total * 100) if total > 0 else 0
                print(
                    f"  [{i+1}/{len(tiles_to_process)}] {matched:,}/{total:,} "
                    f"matched ({match_pct:.1f}%), {unique_inst} instances -> {filename}",
                    flush=True,
                )
            else:
                print(f"  [{i+1}/{len(tiles_to_process)}] FAILED: {message} -> {filename}", flush=True)
                failures.append(f"{filename}: {message}")
            gc.collect()

    if failures:
        raise RuntimeError("Retile to original files failed:\n  " + "\n  ".join(failures))

    print(f"\n  ✓ Retiling complete: {len(tiles_to_process)} tiles processed", flush=True)
    gc.collect()


def _copy_record_dimensions(source_points, out_record) -> None:
    """Copy dimensions from a source point record into an output record."""
    output_names = set(out_record.point_format.dimension_names)
    output_names.update(dim.name for dim in out_record.point_format.extra_dimensions)
    for dim_name in source_points.point_format.dimension_names:
        if dim_name in output_names:
            out_record[dim_name] = source_points[dim_name]
    for dim in source_points.point_format.extra_dimensions:
        if dim.name in output_names:
            out_record[dim.name] = source_points[dim.name]


def _branded_prediction_name(dim_name: str, threedtrees_suffix: str) -> str:
    """Apply the 3DTrees suffix unless the prediction dimension already has it."""
    if not threedtrees_suffix:
        return dim_name
    suffix = f"_{threedtrees_suffix}"
    return dim_name if dim_name.endswith(suffix) else f"{dim_name}{suffix}"


def _original_remap_output_header(source_header, merged_extra_dims, merged_extra_dim_params, threedtrees_dims, threedtrees_suffix):
    """Build output header for original-with-predictions remap products."""
    new_header = copy_single_source_header(
        source_header,
        preserve_extra_dimensions=False,
    )
    output_las = laspy.LasData(new_header)
    output_standard_dim_names = set(output_las.point_format.dimension_names)
    output_extra_dim_names = {dim.name for dim in output_las.point_format.extra_dimensions}
    extra_dims_to_add = []
    added_extra_names = output_standard_dim_names | output_extra_dim_names

    for dim in source_header.point_format.extra_dimensions:
        if dim.name not in added_extra_names:
            extra_dims_to_add.append(extra_bytes_params_from_dimension_info(dim))
            added_extra_names.add(dim.name)

    selected_names = [
        name for name in merged_extra_dims
        if not threedtrees_dims or name in threedtrees_dims
    ]
    branded_names = {}
    for dim_name in selected_names:
        desired_name = _branded_prediction_name(dim_name, threedtrees_suffix)
        out_name = (
            next_available_suffix(desired_name, added_extra_names)
            if desired_name in added_extra_names
            else desired_name
        )
        branded_names[dim_name] = out_name
        if out_name not in added_extra_names:
            values = merged_extra_dims[dim_name]
            if merged_extra_dim_params and dim_name in merged_extra_dim_params:
                params = merged_extra_dim_params[dim_name]
                extra_dims_to_add.append(extra_bytes_params_from_params(params, name=out_name))
            else:
                extra_dims_to_add.append(laspy.ExtraBytesParams(name=out_name, type=values.dtype))
            added_extra_names.add(out_name)

    if extra_dims_to_add:
        output_las.add_extra_dims(extra_dims_to_add)
    return output_las.header, branded_names


def _original_remap_output_header_from_params(
    source_header,
    merged_extra_dim_params: Dict[str, laspy.ExtraBytesParams],
    threedtrees_dims,
    threedtrees_suffix,
):
    """Build an original-with-predictions header from merged COPC header metadata."""
    new_header = copy_single_source_header(
        source_header,
        preserve_extra_dimensions=False,
    )
    output_las = laspy.LasData(new_header)
    output_standard_dim_names = set(output_las.point_format.dimension_names)
    output_extra_dim_names = {dim.name for dim in output_las.point_format.extra_dimensions}
    extra_dims_to_add = []
    added_extra_names = output_standard_dim_names | output_extra_dim_names

    for dim in source_header.point_format.extra_dimensions:
        if dim.name not in added_extra_names:
            extra_dims_to_add.append(extra_bytes_params_from_dimension_info(dim))
            added_extra_names.add(dim.name)

    selected_names = [
        name for name in merged_extra_dim_params
        if not threedtrees_dims or name in threedtrees_dims
    ]
    branded_names = {}
    for dim_name in selected_names:
        desired_name = _branded_prediction_name(dim_name, threedtrees_suffix)
        out_name = (
            next_available_suffix(desired_name, added_extra_names)
            if desired_name in added_extra_names
            else desired_name
        )
        branded_names[dim_name] = out_name
        if out_name not in added_extra_names:
            params = merged_extra_dim_params[dim_name]
            extra_dims_to_add.append(extra_bytes_params_from_params(params, name=out_name))
            added_extra_names.add(out_name)

    if extra_dims_to_add:
        output_las.add_extra_dims(extra_dims_to_add)
    return output_las.header, branded_names


def _extra_dim_params_from_header(header) -> Dict[str, laspy.ExtraBytesParams]:
    return {
        dim.name: extra_bytes_params_from_dimension_info(dim)
        for dim in header.point_format.extra_dimensions
    }


def _chunk_xy_bounds(points: np.ndarray):
    return (
        float(np.min(points[:, 0])),
        float(np.max(points[:, 0])),
        float(np.min(points[:, 1])),
        float(np.max(points[:, 1])),
    )


def _existing_output_is_reusable(
    input_file: Path,
    output_file: Path,
    expected_prediction_dims,
) -> bool:
    """Return True when an existing enriched original matches this remap request."""
    if not output_file.exists() or output_file.stat().st_size == 0:
        return False
    try:
        with laspy.open(str(input_file), laz_backend=laspy.LazBackend.LazrsParallel) as input_reader:
            expected_count = int(input_reader.header.point_count)
        with laspy.open(str(output_file), laz_backend=laspy.LazBackend.LazrsParallel) as output_reader:
            if int(output_reader.header.point_count) != expected_count:
                return False
            output_dims = set(output_reader.header.point_format.dimension_names)
            output_dims.update(dim.name for dim in output_reader.header.point_format.extra_dimensions)
        return set(expected_prediction_dims).issubset(output_dims)
    except Exception as exc:
        print(f"  Existing output is not reusable ({output_file.name}): {exc}", flush=True)
        return False


def _queue_original_outputs(
    original_files,
    output_dir: Path,
    expected_prediction_dims,
):
    """Build remap work items, keeping only outputs valid for this request."""
    files_to_process = []
    skipped = 0
    stale = 0
    for input_file in original_files:
        output_name = input_file.name.replace(".copc.laz", ".laz")
        output_file = output_dir / output_name
        if output_file.exists():
            expected_dims = (
                expected_prediction_dims(input_file)
                if callable(expected_prediction_dims)
                else expected_prediction_dims
            )
            if _existing_output_is_reusable(input_file, output_file, expected_dims):
                skipped += 1
                continue
            stale += 1
            try:
                output_file.unlink()
            except OSError as exc:
                raise RuntimeError(f"Could not replace stale output {output_file}: {exc}") from exc
        files_to_process.append((input_file, output_file))
    return files_to_process, skipped, stale


def _expected_output_dims_from_loaded_merge(
    input_file: Path,
    merged_extra_dims,
    merged_extra_dim_params,
    threedtrees_dims,
    threedtrees_suffix,
):
    with laspy.open(str(input_file), laz_backend=laspy.LazBackend.LazrsParallel) as input_reader:
        _header, branded_names = _original_remap_output_header(
            input_reader.header,
            merged_extra_dims,
            merged_extra_dim_params,
            threedtrees_dims,
            threedtrees_suffix,
        )
    return list(branded_names.values())


def _expected_output_dims_from_merged_copc(
    input_file: Path,
    merged_extra_dim_params,
    threedtrees_dims,
    threedtrees_suffix,
):
    with laspy.open(str(input_file), laz_backend=laspy.LazBackend.LazrsParallel) as input_reader:
        _header, branded_names = _original_remap_output_header_from_params(
            input_reader.header,
            merged_extra_dim_params,
            threedtrees_dims,
            threedtrees_suffix,
        )
    return list(branded_names.values())


def _merged_copc_points_for_chunk(copc_reader, merged_header, xy_bounds, spatial_buffer, selected_dims):
    query_bounds = laspy.copc.Bounds(
        mins=np.array(
            [
                xy_bounds[0] - spatial_buffer,
                xy_bounds[2] - spatial_buffer,
                float(merged_header.z_min),
            ],
            dtype=np.float64,
        ),
        maxs=np.array(
            [
                xy_bounds[1] + spatial_buffer,
                xy_bounds[3] + spatial_buffer,
                float(merged_header.z_max),
            ],
            dtype=np.float64,
        ),
    )
    merged_record = copc_reader.spatial_query(query_bounds)
    if len(merged_record) == 0:
        return None, {}

    merged_points = np.column_stack([merged_record.x, merged_record.y, merged_record.z])
    merged_dims = {
        dim_name: np.asarray(merged_record[dim_name])
        for dim_name in selected_dims
    }
    return merged_points, merged_dims


def _process_single_original_input_file_from_merged_copc(args):
    """Stream one uploaded LAZ/LAS original and query merged COPC windows per chunk."""
    (
        input_file,
        output_file,
        merged_copc_file,
        merged_extra_dim_params,
        tolerance,
        spatial_buffer,
        kdtree_workers,
        threedtrees_dims,
        threedtrees_suffix,
        chunk_size,
        min_match_fraction,
    ) = args

    try:
        selected_dims = [
            name for name in merged_extra_dim_params
            if not threedtrees_dims or name in threedtrees_dims
        ]
        if not selected_dims:
            return (input_file.name, 0, 0, 0, False, "No selected prediction dimensions in merged COPC")

        output_file.parent.mkdir(parents=True, exist_ok=True)
        matched_count = 0
        total_points = 0
        unique_instances = 0

        with laspy.open(str(input_file), laz_backend=laspy.LazBackend.LazrsParallel) as input_reader:
            source_header = input_reader.header
            n_input_points = int(source_header.point_count)
            with laspy.CopcReader.open(str(merged_copc_file)) as copc_reader:
                merged_header = copc_reader.header
                output_header, branded_names = _original_remap_output_header_from_params(
                    source_header,
                    merged_extra_dim_params,
                    threedtrees_dims,
                    threedtrees_suffix,
                )
                with laspy.open(
                    str(output_file),
                    mode="w",
                    header=output_header,
                    laz_backend=laspy.LazBackend.LazrsParallel,
                ) as writer:
                    for input_chunk in input_reader.chunk_iterator(chunk_size):
                        input_points = np.column_stack([input_chunk.x, input_chunk.y, input_chunk.z])
                        xy_bounds = _chunk_xy_bounds(input_points)
                        merged_points, merged_dims = _merged_copc_points_for_chunk(
                            copc_reader,
                            merged_header,
                            xy_bounds,
                            spatial_buffer,
                            selected_dims,
                        )
                        if merged_points is None or len(merged_points) == 0:
                            raise ValueError(
                                f"No merged COPC points near chunk bounds {xy_bounds} "
                                f"from {input_file.name}"
                            )

                        local_tree = cKDTree(merged_points)
                        distances, indices = local_tree.query(input_points, workers=kdtree_workers)
                        matched = distances <= tolerance
                        window_matched = int(np.count_nonzero(matched))

                        out_chunk = laspy.ScaleAwarePointRecord.zeros(len(input_chunk), header=output_header)
                        _copy_record_dimensions(input_chunk, out_chunk)
                        for dim_name, values in merged_dims.items():
                            remapped = values[indices]
                            out_chunk[branded_names[dim_name]] = remapped
                            if np.issubdtype(remapped.dtype, np.integer) and len(remapped) > 0:
                                unique_instances = max(
                                    unique_instances,
                                    len(np.unique(remapped[remapped > 0])),
                                )

                        writer.write_points(out_chunk)
                        matched_count += window_matched
                        total_points += len(input_chunk)
                        del input_points, merged_points, merged_dims, local_tree, distances, indices, out_chunk

        if total_points != n_input_points:
            return (
                input_file.name,
                matched_count,
                n_input_points,
                unique_instances,
                False,
                f"Wrote {total_points:,}/{n_input_points:,} original points",
            )
        if not _match_fraction_is_acceptable(matched_count, n_input_points, min_match_fraction):
            output_file.unlink(missing_ok=True)
            return (
                input_file.name,
                matched_count,
                n_input_points,
                unique_instances,
                False,
                _match_quality_message(matched_count, n_input_points, tolerance, min_match_fraction),
            )
        return (input_file.name, matched_count, n_input_points, unique_instances, True, "Success")
    except Exception as e:
        try:
            output_file.unlink(missing_ok=True)
        except Exception:
            pass
        return (input_file.name, 0, 0, 0, False, str(e))


def _copc_windows_from_header(header, num_windows: int):
    min_x = float(header.x_min)
    max_x = float(header.x_max)
    if max_x <= min_x:
        return [(min_x, max_x, True)]
    count = max(1, int(num_windows or 1))
    step = (max_x - min_x) / count
    return [
        (
            min_x + idx * step,
            max_x if idx == count - 1 else min_x + (idx + 1) * step,
            idx == count - 1,
        )
        for idx in range(count)
    ]


def _process_single_original_copc_file(
    input_file: Path,
    output_file: Path,
    merged_points: np.ndarray,
    merged_extra_dims: Dict[str, np.ndarray],
    merged_extra_dim_params: Optional[Dict[str, laspy.ExtraBytesParams]],
    tolerance: float,
    spatial_buffer: float,
    kdtree_workers: int,
    threedtrees_dims,
    threedtrees_suffix,
    num_spatial_chunks: int = 1,
    min_match_fraction: float = MIN_ORIGINAL_REMAP_MATCH_FRACTION,
):
    """Process one COPC original with native spatial queries."""
    with laspy.CopcReader.open(str(input_file)) as copc_reader:
        source_header = copc_reader.header
        n_input_points = int(source_header.point_count)
        output_header, branded_names = _original_remap_output_header(
            source_header,
            merged_extra_dims,
            merged_extra_dim_params,
            threedtrees_dims,
            threedtrees_suffix,
        )
        windows = _copc_windows_from_header(source_header, num_spatial_chunks)

        output_file.parent.mkdir(parents=True, exist_ok=True)
        matched_count = 0
        total_points = 0
        unique_instances = 0
        with laspy.open(
            str(output_file),
            mode="w",
            header=output_header,
            laz_backend=laspy.LazBackend.LazrsParallel,
        ) as writer:
            for window_idx, (min_x, max_x, include_upper) in enumerate(windows, start=1):
                query_bounds = laspy.copc.Bounds(
                    mins=np.array([min_x, float(source_header.y_min), float(source_header.z_min)], dtype=np.float64),
                    maxs=np.array([max_x, float(source_header.y_max), float(source_header.z_max)], dtype=np.float64),
                )
                source_points = copc_reader.spatial_query(query_bounds)
                if len(source_points) == 0:
                    continue
                xs = np.asarray(source_points.x)
                point_mask = (xs >= min_x) & ((xs <= max_x) if include_upper else (xs < max_x))
                if not np.any(point_mask):
                    continue
                if not np.all(point_mask):
                    source_points = source_points[point_mask]

                input_points = np.column_stack([source_points.x, source_points.y, source_points.z])
                bounds = (
                    float(np.min(input_points[:, 0])),
                    float(np.max(input_points[:, 0])),
                    float(np.min(input_points[:, 1])),
                    float(np.max(input_points[:, 1])),
                )
                merge_mask = (
                    (merged_points[:, 0] >= bounds[0] - spatial_buffer)
                    & (merged_points[:, 0] <= bounds[1] + spatial_buffer)
                    & (merged_points[:, 1] >= bounds[2] - spatial_buffer)
                    & (merged_points[:, 1] <= bounds[3] + spatial_buffer)
                )
                local_merged_points = merged_points[merge_mask]
                if len(local_merged_points) == 0:
                    raise ValueError(f"No merged points in COPC window {window_idx}/{len(windows)}")
                local_tree = cKDTree(local_merged_points)
                distances, indices = local_tree.query(input_points, workers=kdtree_workers)
                matched = distances <= tolerance
                window_matched = int(np.count_nonzero(matched))
                out_chunk = laspy.ScaleAwarePointRecord.zeros(len(source_points), header=output_header)
                _copy_record_dimensions(source_points, out_chunk)

                for dim_name, values in merged_extra_dims.items():
                    if threedtrees_dims and dim_name not in threedtrees_dims:
                        continue
                    remapped = values[merge_mask][indices]
                    out_chunk[branded_names[dim_name]] = remapped
                    if np.issubdtype(remapped.dtype, np.integer) and len(remapped) > 0:
                        unique_instances = max(unique_instances, len(np.unique(remapped[remapped > 0])))

                writer.write_points(out_chunk)
                matched_count += window_matched
                total_points += len(source_points)
                print(
                    f"    COPC original remap {input_file.name}: "
                    f"window {window_idx}/{len(windows)}, {matched_count:,} points",
                    flush=True,
                )

    if total_points != n_input_points:
        output_file.unlink(missing_ok=True)
        return (
            input_file.name,
            matched_count,
            n_input_points,
            unique_instances,
            False,
            f"Wrote {total_points:,}/{n_input_points:,} original points",
        )
    if not _match_fraction_is_acceptable(matched_count, n_input_points, min_match_fraction):
        output_file.unlink(missing_ok=True)
        return (
            input_file.name,
            matched_count,
            n_input_points,
            unique_instances,
            False,
            _match_quality_message(matched_count, n_input_points, tolerance, min_match_fraction),
        )
    return (input_file.name, matched_count, n_input_points, unique_instances, True, "Success")


def _process_single_original_input_file(args):
    """Process one original input file for final prediction remapping."""
    (
        input_file,
        output_file,
        merged_points,
        merged_extra_dims,
        merged_extra_dim_params,
        tolerance,
        spatial_buffer,
        kdtree_workers,
        threedtrees_dims,
        threedtrees_suffix,
        num_spatial_chunks,
        min_match_fraction,
    ) = args

    try:
        if input_file.name.lower().endswith(".copc.laz"):
            return _process_single_original_copc_file(
                input_file,
                output_file,
                merged_points,
                merged_extra_dims,
                merged_extra_dim_params,
                tolerance,
                spatial_buffer,
                kdtree_workers,
                threedtrees_dims,
                threedtrees_suffix,
                num_spatial_chunks=num_spatial_chunks,
                min_match_fraction=min_match_fraction,
            )

        with laspy.open(str(input_file), laz_backend=laspy.LazBackend.LazrsParallel) as f:
            bounds = (f.header.x_min, f.header.x_max, f.header.y_min, f.header.y_max)
            n_input_points = f.header.point_count

        mask = (
            (merged_points[:, 0] >= bounds[0] - spatial_buffer)
            & (merged_points[:, 0] <= bounds[1] + spatial_buffer)
            & (merged_points[:, 1] >= bounds[2] - spatial_buffer)
            & (merged_points[:, 1] <= bounds[3] + spatial_buffer)
        )

        local_merged_points = merged_points[mask]
        local_merged_extras = {name: arr[mask] for name, arr in merged_extra_dims.items()}

        if len(local_merged_points) == 0:
            return (input_file.name, 0, n_input_points, 0, False, "No merged points in file region")

        local_tree = cKDTree(local_merged_points)
        input_las = laspy.read(str(input_file), laz_backend=laspy.LazBackend.LazrsParallel)
        input_points = np.empty((n_input_points, 3), dtype=np.float64)
        input_points[:, 0] = input_las.x
        input_points[:, 1] = input_las.y
        input_points[:, 2] = input_las.z

        distances, indices = local_tree.query(input_points, workers=kdtree_workers)
        matched = distances <= tolerance
        matched_count = int(np.count_nonzero(matched))
        if not _match_fraction_is_acceptable(matched_count, n_input_points, min_match_fraction):
            return (
                input_file.name,
                matched_count,
                n_input_points,
                0,
                False,
                _match_quality_message(matched_count, n_input_points, tolerance, min_match_fraction),
            )

        if threedtrees_dims:
            filtered_extras = {
                name: arr[indices]
                for name, arr in local_merged_extras.items()
                if name in threedtrees_dims
            }
        else:
            filtered_extras = {name: arr[indices] for name, arr in local_merged_extras.items()}

        unique_instances = 0
        for arr in filtered_extras.values():
            if np.issubdtype(arr.dtype, np.integer) and len(arr) > 0:
                unique_instances = max(unique_instances, len(np.unique(arr[arr > 0])))
                break

        branded_names = {}
        for dim_name in filtered_extras:
            branded_names[dim_name] = _branded_prediction_name(dim_name, threedtrees_suffix)

        new_header = copy_single_source_header(
            input_las.header,
            preserve_extra_dimensions=False,
        )
        output_las = laspy.LasData(new_header)
        output_standard_dim_names = set(output_las.point_format.dimension_names)
        output_extra_dim_names = {dim.name for dim in output_las.point_format.extra_dimensions}

        extra_dims_to_add = []
        added_extra_names = output_standard_dim_names | output_extra_dim_names
        for dim in input_las.point_format.extra_dimensions:
            if dim.name not in added_extra_names:
                extra_dims_to_add.append(extra_bytes_params_from_dimension_info(dim))
                added_extra_names.add(dim.name)

        for dim_name in input_las.point_format.dimension_names:
            if dim_name not in added_extra_names:
                arr = getattr(input_las, dim_name, None)
                dtype = arr.dtype if arr is not None else np.int32
                extra_dims_to_add.append(laspy.ExtraBytesParams(name=dim_name, type=dtype))
                added_extra_names.add(dim_name)

        for dim_name, values in filtered_extras.items():
            desired_name = branded_names[dim_name]
            out_name = (
                next_available_suffix(desired_name, added_extra_names)
                if desired_name in added_extra_names
                else desired_name
            )
            branded_names[dim_name] = out_name
            if out_name not in added_extra_names:
                if merged_extra_dim_params and dim_name in merged_extra_dim_params:
                    params = merged_extra_dim_params[dim_name]
                    extra_dims_to_add.append(extra_bytes_params_from_params(params, name=out_name))
                else:
                    extra_dims_to_add.append(laspy.ExtraBytesParams(name=out_name, type=values.dtype))
                added_extra_names.add(out_name)

        if extra_dims_to_add:
            output_las.add_extra_dims(extra_dims_to_add)

        for dim_name in output_las.point_format.dimension_names:
            try:
                if hasattr(input_las, dim_name):
                    setattr(output_las, dim_name, getattr(input_las, dim_name))
            except Exception:
                pass
        for dim in input_las.point_format.extra_dimensions:
            if hasattr(input_las, dim.name):
                try:
                    setattr(output_las, dim.name, getattr(input_las, dim.name))
                except Exception:
                    pass

        for dim_name, values in filtered_extras.items():
            setattr(output_las, branded_names[dim_name], values)

        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_las.write(
            str(output_file),
            do_compress=True,
            laz_backend=laspy.LazBackend.LazrsParallel,
        )

        del input_las
        del output_las
        return (input_file.name, matched_count, n_input_points, unique_instances, True, "Success")

    except Exception as e:
        try:
            output_file.unlink(missing_ok=True)
        except Exception:
            pass
        return (input_file.name, 0, 0, 0, False, str(e))


def validate_common_dimensions_minmax(original_path: Path, output_path: Path, rel_tol: float = 1e-5) -> None:
    """Warn when common non-coordinate dimensions changed range after remap."""
    try:
        orig = laspy.read(str(original_path), laz_backend=laspy.LazBackend.LazrsParallel)
        out = laspy.read(str(output_path), laz_backend=laspy.LazBackend.LazrsParallel)
    except Exception as e:
        print(f"  Validation skip: could not read files ({e})", flush=True)
        return
    try:
        orig_names = set(orig.point_format.dimension_names) | {d.name for d in orig.point_format.extra_dimensions}
        out_names = set(out.point_format.dimension_names) | {d.name for d in out.point_format.extra_dimensions}
        common = orig_names & out_names - {"X", "Y", "Z"}
        if not common:
            return
        diffs = []
        for name in sorted(common):
            oa = getattr(orig, name, None)
            oo = getattr(out, name, None)
            if oa is None or oo is None or len(oa) != len(oo):
                continue
            oa, oo = np.asarray(oa), np.asarray(oo)
            omin, omax = float(np.min(oa)), float(np.max(oa))
            wmin, wmax = float(np.min(oo)), float(np.max(oo))
            if np.issubdtype(oa.dtype, np.integer) and np.issubdtype(oo.dtype, np.integer):
                if (omin != wmin or omax != wmax) and (int(omin) != int(wmin) or int(omax) != int(wmax)):
                    diffs.append((name, omin, omax, wmin, wmax))
            else:
                span = max(omax - omin, 1e-12)
                if abs(omin - wmin) > rel_tol * span or abs(omax - wmax) > rel_tol * span:
                    diffs.append((name, omin, omax, wmin, wmax))
        if diffs:
            print("  Warning: common dimensions min/max differ (output may have rounding/loss):", flush=True)
            for name, omin, omax, wmin, wmax in diffs:
                print(f"    {name}: original [{omin}, {omax}] vs output [{wmin}, {wmax}]", flush=True)
            print("  Tip: dimensions above were not overwritten by merged where range would be lost.", flush=True)
        del orig
        del out
    except Exception as e:
        print(f"  Validation skip: {e}", flush=True)


def remap_to_original_input_files(
    merged_points: np.ndarray,
    merged_extra_dims: Dict[str, np.ndarray],
    merged_extra_dim_params: Optional[Dict[str, laspy.ExtraBytesParams]],
    original_input_dir: Path,
    output_dir: Path,
    tolerance: float = 0.1,
    num_threads: int = 8,
    retile_buffer: float = 2.0,
    threedtrees_dims: Optional[List[str]] = None,
    threedtrees_suffix: str = "SAT",
    num_spatial_chunks: int = 1,
    prefer_copc_sources: bool = True,
    min_match_fraction: float = MIN_ORIGINAL_REMAP_MATCH_FRACTION,
):
    """Transfer selected merged prediction dimensions back to original input files."""
    print(f"\n{'=' * 60}", flush=True)
    print("Remapping to original input files", flush=True)
    print(f"{'=' * 60}", flush=True)

    original_files = (
        point_cloud_files(original_input_dir)
        if prefer_copc_sources
        else raw_point_cloud_files(original_input_dir)
    )
    if len(original_files) == 0:
        print(f"  No LAZ/LAS files found in {original_input_dir}", flush=True)
        return
    if not prefer_copc_sources:
        print("  Raw-original mode: ignoring COPC twins in original input dir", flush=True)

    print(f"  Found {len(original_files)} original input files", flush=True)
    print(f"  Output: {output_dir}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    spatial_buffer = max(tolerance * 2, 1.0) + retile_buffer

    if threedtrees_dims is None:
        threedtrees_dims = ["PredInstance", "PredSemantic"]
    threedtrees_dims_set = set(threedtrees_dims)

    available_3dt = sorted(threedtrees_dims_set & set(merged_extra_dims.keys()))
    expected_output_dims = [
        _branded_prediction_name(d, threedtrees_suffix)
        for d in available_3dt
    ]
    if available_3dt:
        print(f"  3DTrees dimensions to transfer: {', '.join(available_3dt)} -> {', '.join(expected_output_dims)}", flush=True)
    else:
        print(
            "  Warning: No 3DTrees dimensions found in merged file "
            f"(looked for: {', '.join(sorted(threedtrees_dims_set))})",
            flush=True,
        )

    files_to_process, skipped, stale = _queue_original_outputs(
        original_files,
        output_dir,
        lambda input_file: _expected_output_dims_from_loaded_merge(
            input_file,
            merged_extra_dims,
            merged_extra_dim_params,
            threedtrees_dims_set,
            threedtrees_suffix,
        ),
    )
    if skipped > 0:
        print(f"  Skipping {skipped} already processed files", flush=True)
    if stale > 0:
        print(f"  Reprocessing {stale} stale existing output file(s)", flush=True)

    if len(files_to_process) == 0:
        print("  All files already processed!", flush=True)
        return

    print(f"  Processing {len(files_to_process)} files...", flush=True)

    kdtree_workers = -1
    process_args = [
        (
            input_file,
            output_file,
            merged_points,
            merged_extra_dims,
            merged_extra_dim_params,
            tolerance,
            spatial_buffer,
            kdtree_workers,
            threedtrees_dims_set,
            threedtrees_suffix,
            num_spatial_chunks,
            min_match_fraction,
        )
        for input_file, output_file in files_to_process
    ]

    total_matched = 0
    total_points = 0
    failures = []
    parallel_workers = min(num_threads, len(files_to_process)) if num_threads > 1 else 1
    if parallel_workers > 1:
        print(f"  Processing with {parallel_workers} parallel workers...", flush=True)
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            results = executor.map(_process_single_original_input_file, process_args)
            for i, result in enumerate(results):
                filename, matched, total, unique_inst, success, message = result
                if success:
                    match_pct = (matched / total * 100) if total > 0 else 0
                    print(
                        f"  [{i+1}/{len(files_to_process)}] {matched:,}/{total:,} "
                        f"matched ({match_pct:.1f}%), {unique_inst} instances -> {filename}",
                        flush=True,
                    )
                    total_matched += matched
                    total_points += total
                else:
                    print(f"  [{i+1}/{len(files_to_process)}] FAILED: {message} -> {filename}", flush=True)
                    failures.append(f"{filename}: {message}")
    else:
        for i, args in enumerate(process_args):
            filename, matched, total, unique_inst, success, message = _process_single_original_input_file(args)
            if success:
                match_pct = (matched / total * 100) if total > 0 else 0
                print(
                    f"  [{i+1}/{len(files_to_process)}] {matched:,}/{total:,} "
                    f"matched ({match_pct:.1f}%), {unique_inst} instances -> {filename}",
                    flush=True,
                )
                total_matched += matched
                total_points += total
            else:
                print(f"  [{i+1}/{len(files_to_process)}] FAILED: {message} -> {filename}", flush=True)
                failures.append(f"{filename}: {message}")
            gc.collect()

    if failures:
        raise RuntimeError("Original remap failed:\n  " + "\n  ".join(failures))

    overall_match_pct = (total_matched / total_points * 100) if total_points > 0 else 0
    print(
        f"\n  ✓ Remap complete: {len(files_to_process)} files, "
        f"{total_matched:,}/{total_points:,} matched ({overall_match_pct:.1f}%)",
        flush=True,
    )

    if files_to_process and total_matched > 0:
        first_input, first_output = files_to_process[0]
        if first_output.exists():
            validate_common_dimensions_minmax(first_input, first_output)

    gc.collect()


def remap_merged_file_to_original_input_files(
    merged_file: Path,
    original_input_dir: Path,
    output_dir: Path,
    tolerance: float = 0.1,
    num_threads: int = 8,
    retile_buffer: float = 2.0,
    threedtrees_dims: Optional[List[str]] = None,
    threedtrees_suffix: str = "SAT",
    num_spatial_chunks: int = 1,
    chunk_size: int = 5_000_000,
    prefer_copc_sources: bool = True,
    min_match_fraction: float = MIN_ORIGINAL_REMAP_MATCH_FRACTION,
):
    """Transfer merged prediction dimensions back to originals.

    COPC merged sources use a streaming raw-original path: each uploaded
    original LAZ/LAS chunk queries the merged COPC by chunk XY bounds and builds
    a small local KDTree. Non-COPC merged sources fall back to the legacy loaded
    merged-cloud path.
    """
    merged_file = Path(merged_file)
    if not merged_file.name.lower().endswith(".copc.laz"):
        from merge_loaded_cloud import load_merged_file

        merged_points, merged_extra_dims, merged_extra_dim_params = load_merged_file(merged_file)
        return remap_to_original_input_files(
            merged_points,
            merged_extra_dims,
            merged_extra_dim_params,
            original_input_dir,
            output_dir,
            tolerance=tolerance,
            num_threads=num_threads,
            retile_buffer=retile_buffer,
            threedtrees_dims=threedtrees_dims,
            threedtrees_suffix=threedtrees_suffix,
            num_spatial_chunks=num_spatial_chunks,
            prefer_copc_sources=prefer_copc_sources,
            min_match_fraction=min_match_fraction,
        )

    print(f"\n{'=' * 60}", flush=True)
    print("Remapping merged COPC to original input files", flush=True)
    print(f"{'=' * 60}", flush=True)
    original_files = (
        point_cloud_files(original_input_dir)
        if prefer_copc_sources
        else raw_point_cloud_files(original_input_dir)
    )
    if len(original_files) == 0:
        print(f"  No LAZ/LAS files found in {original_input_dir}", flush=True)
        return
    if not prefer_copc_sources:
        print("  Raw-original mode: ignoring COPC twins in original input dir", flush=True)

    with laspy.open(str(merged_file), laz_backend=laspy.LazBackend.LazrsParallel) as merged_reader:
        merged_extra_dim_params = _extra_dim_params_from_header(merged_reader.header)

    if threedtrees_dims is None:
        threedtrees_dims = ["PredInstance", "PredSemantic"]
    threedtrees_dims_set = set(threedtrees_dims)
    available_3dt = sorted(threedtrees_dims_set & set(merged_extra_dim_params.keys()))
    if available_3dt:
        branded = [_branded_prediction_name(d, threedtrees_suffix) for d in available_3dt]
        print(f"  3DTrees dimensions to transfer: {', '.join(available_3dt)} -> {', '.join(branded)}", flush=True)
    else:
        print(
            "  Warning: No 3DTrees dimensions found in merged COPC "
            f"(looked for: {', '.join(sorted(threedtrees_dims_set))})",
            flush=True,
        )

    print(f"  Merged COPC source: {merged_file}", flush=True)
    print(f"  Original input files: {len(original_files)}", flush=True)
    print(f"  Output: {output_dir}", flush=True)
    print(f"  Original chunk size: {chunk_size:,} points", flush=True)
    print("  Per-chunk strategy: original LAZ chunk -> merged COPC spatial query -> local KDTree", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    spatial_buffer = max(tolerance * 2, 1.0) + retile_buffer
    expected_output_dims = [
        _branded_prediction_name(d, threedtrees_suffix)
        for d in available_3dt
    ]
    files_to_process, skipped, stale = _queue_original_outputs(
        original_files,
        output_dir,
        lambda input_file: _expected_output_dims_from_merged_copc(
            input_file,
            merged_extra_dim_params,
            threedtrees_dims_set,
            threedtrees_suffix,
        ),
    )

    if skipped > 0:
        print(f"  Skipping {skipped} already processed files", flush=True)
    if stale > 0:
        print(f"  Reprocessing {stale} stale existing output file(s)", flush=True)
    if not files_to_process:
        print("  All files already processed!", flush=True)
        return

    parallel_workers = min(max(1, num_threads), len(files_to_process))
    kdtree_workers = -1 if parallel_workers == 1 else max(1, num_threads // parallel_workers)
    process_args = [
        (
            input_file,
            output_file,
            merged_file,
            merged_extra_dim_params,
            tolerance,
            spatial_buffer,
            kdtree_workers,
            threedtrees_dims_set,
            threedtrees_suffix,
            chunk_size,
            min_match_fraction,
        )
        for input_file, output_file in files_to_process
    ]

    total_matched = 0
    total_points = 0
    failures = []
    if parallel_workers > 1:
        print(f"  Processing with {parallel_workers} parallel original-file workers...", flush=True)
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            results = list(executor.map(_process_single_original_input_file_from_merged_copc, process_args))
    else:
        results = [_process_single_original_input_file_from_merged_copc(args) for args in process_args]

    for i, (filename, matched, total, unique_inst, success, message) in enumerate(results):
        if success:
            match_pct = (matched / total * 100) if total > 0 else 0
            print(
                f"  [{i+1}/{len(files_to_process)}] {matched:,}/{total:,} "
                f"matched ({match_pct:.1f}%), {unique_inst} instances -> {filename}",
                flush=True,
            )
            total_matched += matched
            total_points += total
        else:
            print(f"  [{i+1}/{len(files_to_process)}] FAILED: {message} -> {filename}", flush=True)
            failures.append(f"{filename}: {message}")

    if failures:
        raise RuntimeError("Merged COPC remap failed:\n  " + "\n  ".join(failures))

    overall_match_pct = (total_matched / total_points * 100) if total_points > 0 else 0
    print(
        f"\n  ✓ Merged COPC remap complete: {len(files_to_process)} files, "
        f"{total_matched:,}/{total_points:,} matched ({overall_match_pct:.1f}%)",
        flush=True,
    )

    if files_to_process and total_matched > 0:
        first_input, first_output = files_to_process[0]
        if first_output.exists():
            validate_common_dimensions_minmax(first_input, first_output)

    gc.collect()
