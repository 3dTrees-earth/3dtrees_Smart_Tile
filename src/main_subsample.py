#!/usr/bin/env python3
"""
Main subsampling script: Parallel subsampling to resolution 1 (1cm) and resolution 2 (10cm).

This script handles subsampling of tiled point clouds:
1. Subsample tiles to resolution 1 (default: 1cm)
2. Subsample resolution 1 files to resolution 2 (default: 10cm)

Files are split across available CPU cores for parallel processing.

COPC Optimizations:
- Uses COPC native bounds filtering in readers.copc (more efficient than filters.crop)
- Writes output as COPC format when input is COPC (better performance for subsequent steps)
- Leverages COPC's spatial indexing for efficient chunk-based processing
- Multi-threaded COPC writing for improved performance

Usage:
    python main_subsample.py --tiles_dir /path/to/tiles --res1 0.01 --res2 0.1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import laspy
import numpy as np

# Import parameters
from parameters import TILE_PARAMS


SUBSAMPLING_METHOD_CENTER_OF_MASS = "center-of-mass"
SUBSAMPLING_METHOD_NEAREST_TO_CENTROID = "nearest-to-centroid"
SUBSAMPLING_METHODS = {
    SUBSAMPLING_METHOD_CENTER_OF_MASS,
    SUBSAMPLING_METHOD_NEAREST_TO_CENTROID,
}
COPC_COM_TARGET_WINDOW_CELLS = 500
COPC_COM_MAX_WINDOW_SIZE = 20.0
COPC_COM_DENSE_BIN_LIMIT = 5_000_000


def get_pdal_path() -> str:
    """Get the path to pdal executable."""
    import shutil
    # Use shutil.which to find pdal in PATH
    pdal_path = shutil.which("pdal")
    return pdal_path if pdal_path else "pdal"


def get_cpu_count() -> int:
    """Get available CPU count."""
    try:
        return os.cpu_count() or 4
    except Exception:
        return 4


def _parse_bounds_metadata(metadata) -> Optional[Tuple[float, float, float, float]]:
    """Extract finite XY bounds from PDAL metadata."""
    if not isinstance(metadata, dict):
        return None

    keys = ("minx", "maxx", "miny", "maxy")
    if all(key in metadata for key in keys):
        try:
            minx, maxx, miny, maxy = (float(metadata[key]) for key in keys)
        except (TypeError, ValueError):
            return None
        if (
            all(math.isfinite(value) for value in (minx, maxx, miny, maxy))
            and maxx >= minx
            and maxy >= miny
        ):
            return (minx, maxx, miny, maxy)
        return None

    for value in metadata.values():
        if isinstance(value, dict):
            bounds = _parse_bounds_metadata(value)
            if bounds is not None:
                return bounds
        elif isinstance(value, list):
            for item in value:
                bounds = _parse_bounds_metadata(item)
                if bounds is not None:
                    return bounds
    return None


def get_file_bounds(filepath: Path) -> Optional[Tuple[float, float, float, float]]:
    """
    Get spatial bounds of a point cloud file using pdal info.
    
    Returns:
        Tuple of (minx, maxx, miny, maxy) or None on error
    """
    try:
        pdal_cmd = get_pdal_path()
        result = subprocess.run(
            [pdal_cmd, "info", "--metadata", str(filepath)],
            capture_output=True,
            text=True,
            check=True
        )

        payload = json.loads(result.stdout)
        return _parse_bounds_metadata(payload.get("metadata", payload))
    except Exception:
        return None


def get_file_xy_scales(filepath: Path) -> Tuple[float, float]:
    """Return LAS/COPC XY scales for half-open chunk bounds."""
    try:
        with laspy.open(str(filepath), laz_backend=laspy.LazBackend.LazrsParallel) as reader:
            scales = reader.header.scales
            return (float(scales[0]), float(scales[1]))
    except Exception:
        return (0.001, 0.001)


def normalize_subsampling_method(method: Optional[str]) -> str:
    """Return a canonical SmartTile subsampling method name."""
    normalized = (method or SUBSAMPLING_METHOD_CENTER_OF_MASS).strip().lower()
    if normalized in {"centroid", "voxelcentroidnearestneighbor", "voxel-centroid-nearest-neighbor"}:
        return SUBSAMPLING_METHOD_NEAREST_TO_CENTROID
    if normalized in {"com", "center_of_mass", "center-of-mass"}:
        return SUBSAMPLING_METHOD_CENTER_OF_MASS
    if normalized not in SUBSAMPLING_METHODS:
        allowed = ", ".join(sorted(SUBSAMPLING_METHODS))
        raise ValueError(f"Unknown subsampling method {method!r}; expected one of: {allowed}")
    return normalized


def _voxel_subsampling_filter(resolution: float, method: str) -> dict:
    """Return the PDAL filter stage for methods implemented by PDAL."""
    method = normalize_subsampling_method(method)
    if method == SUBSAMPLING_METHOD_NEAREST_TO_CENTROID:
        return {"type": "filters.voxelcentroidnearestneighbor", "cell": resolution}
    raise ValueError(f"{method} is implemented by SmartTile, not a PDAL filter stage")


def _is_copc_file(path: Path) -> bool:
    return path.name.lower().endswith(".copc.laz")


def _is_stale_copc_vlr(vlr) -> bool:
    return getattr(vlr, "user_id", "") == "copc" and getattr(vlr, "record_id", None) in (1, 2)


def _is_extra_bytes_vlr(vlr) -> bool:
    return getattr(vlr, "user_id", "") == "LASF_Spec" and getattr(vlr, "record_id", None) == 4


def _make_center_of_mass_header(source_header: laspy.LasHeader, dimension_reduction: bool) -> laspy.LasHeader:
    """Create an output header for center-of-mass subsampling."""
    from laspy.vlrs.vlrlist import VLRList

    if dimension_reduction:
        header = laspy.LasHeader(point_format=0, version="1.2")
        for attr in (
            "file_source_id",
            "global_encoding",
            "uuid",
            "system_identifier",
            "generating_software",
            "creation_date",
        ):
            if hasattr(source_header, attr):
                try:
                    setattr(header, attr, getattr(source_header, attr))
                except Exception:
                    pass
        source_vlrs = source_header.vlrs
    else:
        header = source_header.copy()
        source_vlrs = header.vlrs

    header.offsets = source_header.offsets
    header.scales = source_header.scales
    header.vlrs = VLRList([
        vlr for vlr in (source_vlrs or [])
        if not _is_stale_copc_vlr(vlr)
        and (not dimension_reduction or not _is_extra_bytes_vlr(vlr))
    ])

    source_evlrs = getattr(source_header, "evlrs", None)
    if source_evlrs is not None and not dimension_reduction:
        header.evlrs = VLRList([vlr for vlr in source_evlrs if not _is_stale_copc_vlr(vlr)])

    return header


def _copy_non_xyz_dimensions(source_las: laspy.LasData, output_las: laspy.LasData, indices: np.ndarray) -> None:
    """Copy all output-supported non-XYZ dimensions from selected source points."""
    for dim_name in output_las.point_format.dimension_names:
        if dim_name in {"X", "Y", "Z"}:
            continue
        if not hasattr(source_las, dim_name):
            continue
        try:
            setattr(output_las, dim_name, np.asarray(getattr(source_las, dim_name))[indices])
        except Exception:
            pass


def _point_record_to_xyz(points) -> np.ndarray:
    coords = np.empty((len(points), 3), dtype=np.float64)
    coords[:, 0] = points.x
    coords[:, 1] = points.y
    coords[:, 2] = points.z
    return coords


def _aggregate_center_of_mass_sparse(coords: np.ndarray, resolution: float) -> np.ndarray:
    voxel_keys = np.floor(coords / resolution).astype(np.int64)
    _, inverse, counts = np.unique(
        voxel_keys,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    sums = np.zeros((len(counts), 3), dtype=np.float64)
    np.add.at(sums, inverse, coords)
    return sums / counts[:, None]


def _aggregate_center_of_mass_bincount(coords: np.ndarray, resolution: float) -> Optional[np.ndarray]:
    keys = np.floor(coords / resolution).astype(np.int64)
    key_min = keys.min(axis=0)
    key_max = keys.max(axis=0)
    shape = (key_max - key_min + 1).astype(np.int64)
    bins = int(shape[0] * shape[1] * shape[2])
    if bins <= 0 or bins > COPC_COM_DENSE_BIN_LIMIT:
        return None

    yz = int(shape[1] * shape[2])
    z = int(shape[2])
    linear = (
        (keys[:, 0] - key_min[0]) * yz
        + (keys[:, 1] - key_min[1]) * z
        + (keys[:, 2] - key_min[2])
    )
    counts = np.bincount(linear, minlength=bins)
    occupied = np.flatnonzero(counts)
    sums_x = np.bincount(linear, weights=coords[:, 0], minlength=bins)[occupied]
    sums_y = np.bincount(linear, weights=coords[:, 1], minlength=bins)[occupied]
    sums_z = np.bincount(linear, weights=coords[:, 2], minlength=bins)[occupied]
    occupied_counts = counts[occupied].astype(np.float64)

    centers = np.empty((len(occupied), 3), dtype=np.float64)
    centers[:, 0] = sums_x / occupied_counts
    centers[:, 1] = sums_y / occupied_counts
    centers[:, 2] = sums_z / occupied_counts
    return centers


def _aggregate_center_of_mass_xyz(points, resolution: float) -> np.ndarray:
    coords = _point_record_to_xyz(points)
    if len(coords) == 0:
        return np.empty((0, 3), dtype=np.float64)
    centers = _aggregate_center_of_mass_bincount(coords, resolution)
    if centers is not None:
        return centers
    return _aggregate_center_of_mass_sparse(coords, resolution)


def _aligned_edges(start: float, stop: float, step: float, align: float) -> List[Tuple[float, float]]:
    first = math.floor(start / align) * align
    final = math.ceil(stop / align) * align
    edges = []
    cur = first
    while cur < stop:
        nxt = min(cur + step, final)
        if nxt > start and cur < stop:
            edges.append((max(cur, start), min(nxt, stop)))
        cur = nxt
    return edges


def _copc_com_window_size(resolution: float) -> float:
    return max(resolution, min(COPC_COM_MAX_WINDOW_SIZE, resolution * COPC_COM_TARGET_WINDOW_CELLS))


def _iter_copc_center_of_mass_windows(header: laspy.LasHeader, resolution: float) -> Iterable[object]:
    """Yield voxel-aligned, half-open XY bounds for COPC center-of-mass queries."""
    from laspy.copc import Bounds

    step = _copc_com_window_size(resolution)
    x_edges = _aligned_edges(header.x_min, header.x_max, step, resolution)
    y_edges = _aligned_edges(header.y_min, header.y_max, step, resolution)
    eps_x = float(header.scales[0]) * 0.5
    eps_y = float(header.scales[1]) * 0.5

    for xi, (xmin, xmax) in enumerate(x_edges):
        for yi, (ymin, ymax) in enumerate(y_edges):
            qmaxx = xmax if xi == len(x_edges) - 1 else xmax - eps_x
            qmaxy = ymax if yi == len(y_edges) - 1 else ymax - eps_y
            if qmaxx < xmin or qmaxy < ymin:
                continue
            yield Bounds(
                np.array([xmin, ymin, header.z_min], dtype=np.float64),
                np.array([qmaxx, qmaxy, header.z_max], dtype=np.float64),
            )


def _copc_center_of_mass_window_worker(
    args: Tuple[int, str, Tuple[float, float, float], Tuple[float, float, float], float]
) -> Tuple[int, int, np.ndarray]:
    window_idx, input_file, mins, maxs, resolution = args
    from laspy.copc import Bounds, CopcReader

    with CopcReader.open(input_file) as reader:
        points = reader.query(
            Bounds(
                np.array(mins, dtype=np.float64),
                np.array(maxs, dtype=np.float64),
            )
        )
        point_count = len(points)
        if point_count == 0:
            return (window_idx, 0, np.empty((0, 3), dtype=np.float64))
        return (window_idx, point_count, _aggregate_center_of_mass_xyz(points, resolution))


def _write_center_of_mass_points(writer, header: laspy.LasHeader, centers: np.ndarray) -> int:
    """Write one batch of XYZ center points to an open LAS/LAZ writer."""
    if len(centers) == 0:
        return 0
    points = laspy.ScaleAwarePointRecord.zeros(len(centers), header=header)
    points.x = centers[:, 0]
    points.y = centers[:, 1]
    points.z = centers[:, 2]
    writer.write_points(points)
    return len(centers)


def center_of_mass_subsample_copc(
    input_file: Path,
    output_file: Path,
    resolution: float,
    num_workers: int = 1,
) -> int:
    """Subsample one COPC file by querying voxel-aligned windows and averaging XYZ."""
    from laspy.copc import CopcReader

    if resolution <= 0:
        raise ValueError("resolution must be positive")
    num_workers = max(1, int(num_workers or 1))

    pending_centers = {}
    total_input_points = 0
    total_output_points = 0

    with CopcReader.open(input_file) as reader:
        header = reader.header
        windows = list(_iter_copc_center_of_mass_windows(header, resolution))

    if not windows:
        raise ValueError(f"No COPC query windows available for {input_file}")

    output_header = _make_center_of_mass_header(header, dimension_reduction=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    def make_task(window_idx: int):
        bounds = windows[window_idx]
        return (
            window_idx,
            str(input_file),
            tuple(float(value) for value in bounds.mins),
            tuple(float(value) for value in bounds.maxs),
            resolution,
        )

    with laspy.open(
        str(output_file),
        mode="w",
        header=output_header,
        do_compress=output_file.suffix.lower() == ".laz",
    ) as writer:
        if num_workers == 1 or len(windows) <= 1:
            with CopcReader.open(input_file) as reader:
                for bounds in windows:
                    points = reader.query(bounds)
                    if len(points) == 0:
                        continue
                    total_input_points += len(points)
                    centers = _aggregate_center_of_mass_xyz(points, resolution)
                    total_output_points += _write_center_of_mass_points(writer, output_header, centers)
        else:
            worker_count = min(num_workers, len(windows))
            max_in_flight = min(len(windows), worker_count * 2)
            print(f"      → COPC COM window parallelism: {worker_count} workers")

            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                futures = {}
                next_submit = 0
                next_write = 0

                def submit_until_capacity():
                    nonlocal next_submit
                    while (
                        next_submit < len(windows)
                        and len(futures) + len(pending_centers) < max_in_flight
                    ):
                        future = executor.submit(_copc_center_of_mass_window_worker, make_task(next_submit))
                        futures[future] = next_submit
                        next_submit += 1

                submit_until_capacity()

                while futures or next_submit < len(windows):
                    if not futures:
                        submit_until_capacity()
                    for future in as_completed(futures):
                        futures.pop(future)
                        window_idx, point_count, centers = future.result()
                        total_input_points += point_count
                        pending_centers[window_idx] = centers

                        while next_write in pending_centers:
                            centers_to_write = pending_centers.pop(next_write)
                            total_output_points += _write_center_of_mass_points(
                                writer,
                                output_header,
                                centers_to_write,
                            )
                            next_write += 1
                        submit_until_capacity()
                        break

                while next_write in pending_centers:
                    centers_to_write = pending_centers.pop(next_write)
                    total_output_points += _write_center_of_mass_points(
                        writer,
                        output_header,
                        centers_to_write,
                    )
                    next_write += 1

    if total_output_points == 0:
        raise ValueError(f"No points available in {input_file}")

    print(
        f"      COPC COM windows: {len(windows):,}; "
        f"input points read: {total_input_points:,}; output points: {total_output_points:,}"
    )
    return total_output_points


def center_of_mass_subsample_las(
    input_file: Path,
    output_file: Path,
    resolution: float,
    dimension_reduction: bool = True,
) -> int:
    """Subsample one LAS/LAZ file by averaging XYZ per voxel.

    Non-coordinate attributes are copied from the real point nearest to the averaged XYZ
    within each voxel. They are never averaged.
    """
    if resolution <= 0:
        raise ValueError("resolution must be positive")

    source_las = laspy.read(str(input_file), laz_backend=laspy.LazBackend.LazrsParallel)
    point_count = len(source_las.points)
    if point_count == 0:
        raise ValueError(f"No points available in {input_file}")

    coords = _point_record_to_xyz(source_las)
    voxel_keys = np.floor(coords / resolution).astype(np.int64)
    _, inverse, counts = np.unique(
        voxel_keys,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )

    sums = np.zeros((len(counts), 3), dtype=np.float64)
    np.add.at(sums, inverse, coords)
    means = sums / counts[:, None]

    dist2 = np.sum((coords - means[inverse]) ** 2, axis=1)
    order = np.lexsort((dist2, inverse))
    sorted_inverse = inverse[order]
    first_in_voxel = np.empty(len(order), dtype=bool)
    first_in_voxel[0] = True
    first_in_voxel[1:] = sorted_inverse[1:] != sorted_inverse[:-1]
    selected_indices = order[first_in_voxel]
    selected_voxels = inverse[selected_indices]
    mean_coords = means[selected_voxels]

    header = _make_center_of_mass_header(source_las.header, dimension_reduction)
    output_las = laspy.LasData(header)
    output_las.points = laspy.ScaleAwarePointRecord.zeros(len(selected_indices), header=header)
    _copy_non_xyz_dimensions(source_las, output_las, selected_indices)
    output_las.x = mean_coords[:, 0]
    output_las.y = mean_coords[:, 1]
    output_las.z = mean_coords[:, 2]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_las.write(str(output_file), do_compress=output_file.suffix.lower() == ".laz")
    return len(selected_indices)


def _crop_input_to_laz(
    input_file: Path,
    crop_file: Path,
    bounds_str: str,
    dimension_reduction: bool,
) -> bool:
    """Crop a chunk with PDAL before SmartTile center-of-mass aggregation."""
    is_copc = _is_copc_file(input_file)
    reader_type = "readers.copc" if is_copc else "readers.las"
    writer_opts = {
        "type": "writers.las",
        "filename": str(crop_file),
        "compression": True,
        "forward": "all",
    }
    if dimension_reduction:
        writer_opts["minor_version"] = 2
        writer_opts["dataformat_id"] = 0
    else:
        writer_opts["minor_version"] = 4
        writer_opts["extra_dims"] = "all"

    if is_copc:
        stages = [{"type": reader_type, "filename": str(input_file), "bounds": bounds_str}]
    else:
        stages = [
            {"type": reader_type, "filename": str(input_file)},
            {"type": "filters.crop", "bounds": bounds_str},
        ]
    stages.append(writer_opts)
    pipeline = {"pipeline": stages}

    pipeline_file = crop_file.parent / f"_{crop_file.stem}_crop.json"
    with open(pipeline_file, "w") as f:
        json.dump(pipeline, f, indent=2)
    try:
        pdal_cmd = get_pdal_path()
        result = subprocess.run(
            [pdal_cmd, "pipeline", str(pipeline_file)],
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        if pipeline_file.exists():
            pipeline_file.unlink()

    return result.returncode == 0 and crop_file.exists() and crop_file.stat().st_size > 0


def subsample_tile_chunk(args: Tuple[Path, str, float, Path, int, int, bool, str]) -> Tuple[Path, int]:
    """
    Subsample a spatial chunk of a tile using PDAL with COPC-optimized bounds filter.
    
    COPC optimizations:
    - Uses bounds parameter directly in readers.copc (more efficient than filters.crop)
    - Output is always LAZ format (compressed LAS)
    
    Args:
        args: Tuple of (input_file, bounds_str, resolution, output_dir, chunk_idx, total_chunks, dimension_reduction, subsampling_method)
              dimension_reduction: If True, write only standard dimensions (no extra_dims); minimal output.
    
    Returns:
        Tuple of (output_file, point_count)
    """
    input_file, bounds_str, resolution, output_dir, chunk_idx, total_chunks, dimension_reduction, subsampling_method = args
    subsampling_method = normalize_subsampling_method(subsampling_method)
    
    try:
        # Determine reader type - can read COPC or LAS
        is_copc = _is_copc_file(input_file)
        reader_type = "readers.copc" if is_copc else "readers.las"
        
        # Always output as LAZ format (compressed LAS)
        chunk_file = output_dir / f"{input_file.stem}_chunk{chunk_idx}.laz"

        if subsampling_method == SUBSAMPLING_METHOD_CENTER_OF_MASS:
            crop_file = output_dir / f"{input_file.stem}_chunk{chunk_idx}_crop.laz"
            try:
                if not _crop_input_to_laz(input_file, crop_file, bounds_str, dimension_reduction):
                    print(f"      ⚠ Chunk {chunk_idx}/{total_chunks}: crop failed for center-of-mass")
                    return (None, 0)
                point_count = center_of_mass_subsample_las(
                    crop_file,
                    chunk_file,
                    resolution,
                    dimension_reduction=dimension_reduction,
                )
                print(f"      ✓ Chunk {chunk_idx}/{total_chunks}: {point_count:,} points")
                return (chunk_file, point_count)
            finally:
                if crop_file.exists():
                    try:
                        crop_file.unlink()
                    except Exception:
                        pass
        
        # Build pipeline - use COPC bounds filtering if available
        # When keeping all dims, do not set dataformat_id=0 (format 0 has no extra bytes); use LAS 1.4 for format 6/7.
        writer_opts = {
            "type": "writers.las",
            "filename": str(chunk_file),
            "compression": True,
            "forward": "all",
        }
        if dimension_reduction:
            writer_opts["minor_version"] = 2
            writer_opts["dataformat_id"] = 0  # Minimal: 20 bytes only (LAS 1.2)
        else:
            writer_opts["minor_version"] = 4  # LAS 1.4 required for point format 6/7 (extra dims)
            writer_opts["extra_dims"] = "all"

        if is_copc:
            # COPC: Use bounds parameter directly in reader (most efficient)
            pipeline = {
                "pipeline": [
                    {
                        "type": reader_type,
                        "filename": str(input_file),
                        "bounds": bounds_str  # COPC native bounds filtering - very efficient
                    },
                    _voxel_subsampling_filter(resolution, subsampling_method),
                    writer_opts,
                ]
            }
        else:
            # LAS: Use filters.crop as fallback (LAS readers don't support bounds parameter)
            pipeline = {
                "pipeline": [
                    {"type": reader_type, "filename": str(input_file)},
                    {"type": "filters.crop", "bounds": bounds_str},
                    _voxel_subsampling_filter(resolution, subsampling_method),
                    writer_opts,
                ]
            }

        # Write and execute pipeline
        pipeline_file = output_dir / f"_pipeline_chunk{chunk_idx}.json"
        with open(pipeline_file, 'w') as f:
            json.dump(pipeline, f, indent=2)
        
        pdal_cmd = get_pdal_path()
        result = subprocess.run(
            [pdal_cmd, "pipeline", str(pipeline_file)],
            capture_output=True,
            text=True,
            check=False
        )
        
        # Clean up pipeline
        if pipeline_file.exists():
            pipeline_file.unlink()
        
        if result.returncode != 0:
            # Fallback: if COPC reader failed, retry with readers.las + filters.crop
            if is_copc and ("copc" in result.stderr.lower() or "vlr" in result.stderr.lower()):
                print(f"      ⚠ Chunk {chunk_idx}/{total_chunks}: COPC reader failed, falling back to readers.las")
                pipeline = {
                    "pipeline": [
                        {"type": "readers.las", "filename": str(input_file)},
                        {"type": "filters.crop", "bounds": bounds_str},
                        _voxel_subsampling_filter(resolution, subsampling_method),
                        writer_opts,
                    ]
                }
                pipeline_file = output_dir / f"_pipeline_chunk{chunk_idx}_fallback.json"
                with open(pipeline_file, 'w') as f:
                    json.dump(pipeline, f, indent=2)
                try:
                    result = subprocess.run(
                        [pdal_cmd, "pipeline", str(pipeline_file)],
                        capture_output=True, text=True, check=False,
                    )
                finally:
                    if pipeline_file.exists():
                        pipeline_file.unlink()
                if result.returncode != 0:
                    print(f"      ⚠ Chunk {chunk_idx}/{total_chunks} fallback error: {result.stderr[:100]}")
                    return (None, 0)
            else:
                print(f"      ⚠ Chunk {chunk_idx}/{total_chunks} error: {result.stderr[:100]}")
                return (None, 0)

        if not chunk_file.exists() or chunk_file.stat().st_size == 0:
            return (None, 0)
        
        # Get point count
        point_count = 0
        try:
            info_result = subprocess.run(
                [pdal_cmd, "info", "--metadata", str(chunk_file)],
                capture_output=True,
                text=True,
                check=True
            )
            import re
            match = re.search(r'"count":\s*(\d+)', info_result.stdout)
            if match:
                point_count = int(match.group(1))
        except Exception:
            pass
        
        print(f"      ✓ Chunk {chunk_idx}/{total_chunks}: {point_count:,} points")
        return (chunk_file, point_count)
        
    except Exception as e:
        print(f"      ✗ Chunk {chunk_idx}/{total_chunks} failed: {e}")
        return (None, 0)


def subsample_single_file(args: Tuple[Path, Path, float, Path, int, bool, str]) -> Tuple[str, bool, str, int]:
    """
    Subsample a single file by splitting it into subtiles along X-axis and processing in parallel.
    
    Process:
    1. Split tile into num_threads subtiles along X-axis only
    2. Subsample each subtile in parallel using ProcessPoolExecutor (true CPU parallelism)
    3. Merge all subsampled subtiles back together
    
    Args:
        args: Tuple of (input_file, output_file, resolution, pipeline_dir, num_threads, dimension_reduction, subsampling_method)
    
    Returns:
        Tuple of (filename, success, message, point_count)
    """
    input_file, output_file, resolution, pipeline_dir, num_threads, dimension_reduction, subsampling_method = args
    subsampling_method = normalize_subsampling_method(subsampling_method)
    try:
        print(f"    → Processing {input_file.name}...")

        if (
            subsampling_method == SUBSAMPLING_METHOD_CENTER_OF_MASS
            and dimension_reduction
            and _is_copc_file(input_file)
        ):
            print("      → Using COPC voxel-aligned center-of-mass windows")
            point_count = center_of_mass_subsample_copc(
                input_file,
                output_file,
                resolution,
                num_workers=num_threads,
            )
            print(f"    ✓ {input_file.name}: {point_count:,} points")
            return (input_file.name, True, "Success", point_count)
        
        # Get file bounds
        bounds = get_file_bounds(input_file)
        if not bounds:
            # Fall back to simple single-pass subsampling when bounds cannot be determined
            return subsample_simple(
                input_file,
                output_file,
                resolution,
                pipeline_dir,
                dimension_reduction,
                subsampling_method,
            )
        
        minx, maxx, miny, maxy = bounds

        # If the file has zero extent in X (or Y), splitting into X-chunks
        # with degenerate bounds (minx == maxx) can result in empty outputs
        # from the COPC reader. In that case, fall back to the simple
        # single-pass subsampling without spatial chunking.
        if maxx - minx == 0 or maxy - miny == 0:
            return subsample_simple(
                input_file,
                output_file,
                resolution,
                pipeline_dir,
                dimension_reduction,
                subsampling_method,
            )

        # Split into num_threads subtiles along X-axis only.
        # Align boundaries to the voxel grid and make non-final chunks half-open
        # so one output voxel is never computed in two chunks.
        grid_x = num_threads
        grid_y = 1
        
        # Calculate step size for X-axis
        raw_x_step = (maxx - minx) / grid_x
        x_step = max(resolution, math.ceil(raw_x_step / resolution) * resolution)
        scale_x, _ = get_file_xy_scales(input_file)
        
        print(f"      Splitting into {num_threads} subtiles along X-axis only ({grid_x}x{grid_y} grid)")
        
        # Create chunk tasks - exactly num_threads chunks
        chunk_dir = pipeline_dir / f"{input_file.stem}_chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        
        chunk_tasks = []
        chunk_edges = _aligned_edges(minx, maxx, x_step, resolution)
        for chunk_idx, (chunk_minx, chunk_maxx) in enumerate(chunk_edges):
            # Keep full Y range for each chunk
            chunk_miny = miny
            chunk_maxy = maxy
            query_maxx = chunk_maxx if chunk_idx == len(chunk_edges) - 1 else chunk_maxx - scale_x * 0.5
            if query_maxx < chunk_minx:
                continue
            
            bounds_str = f"([{chunk_minx},{query_maxx}],[{chunk_miny},{chunk_maxy}])"
            chunk_tasks.append((
                input_file,
                bounds_str,
                resolution,
                chunk_dir,
                chunk_idx,
                len(chunk_edges),
                dimension_reduction,
                subsampling_method,
            ))
        
        # Process chunks in parallel using ProcessPoolExecutor for true CPU parallelism
        chunk_files = []
        total_points = 0
        
        print(f"      → Subsampling {len(chunk_tasks)} subtiles in parallel...")
        with ProcessPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(subsample_tile_chunk, task) for task in chunk_tasks]
            for future in as_completed(futures):
                chunk_file, point_count = future.result()
                if chunk_file and chunk_file.exists():
                    chunk_files.append(chunk_file)
                    total_points += point_count
        
        if not chunk_files:
            return (input_file.name, False, "No chunks produced", 0)
        
        print(f"      → Merging {len(chunk_files)} subsampled subtiles...")
        
        # Merge chunks using PDAL - always write as LAZ format
        # Chunk files are already LAZ from subsample_tile_chunk
        reader_type = "readers.las"  # Chunks are LAZ files
        
        # Ensure output is LAZ format
        if not output_file.name.endswith('.laz'):
            output_file = output_file.parent / (output_file.stem + '.laz')
        
        merge_writer = {
            "type": "writers.las",
            "filename": str(output_file),
            "compression": True,
            "forward": "all",
        }
        if dimension_reduction:
            merge_writer["minor_version"] = 2
            merge_writer["dataformat_id"] = 0
        else:
            merge_writer["minor_version"] = 4
            merge_writer["extra_dims"] = "all"
        merge_pipeline = {
            "pipeline": [
                *[{"type": reader_type, "filename": str(f)} for f in chunk_files],
                {"type": "filters.merge"},
                merge_writer,
            ]
        }
        
        merge_pipeline_file = chunk_dir / "merge.json"
        with open(merge_pipeline_file, 'w') as f:
            json.dump(merge_pipeline, f, indent=2)
        
        pdal_cmd = get_pdal_path()
        result = subprocess.run(
            [pdal_cmd, "pipeline", str(merge_pipeline_file)],
            capture_output=True,
            text=True,
            check=False
        )
        
        # Clean up chunks and temporary files
        for chunk_file in chunk_files:
            if chunk_file.exists():
                try:
                    chunk_file.unlink()
                except Exception:
                    pass
        
        # Clean up merge pipeline
        if merge_pipeline_file.exists():
            try:
                merge_pipeline_file.unlink()
            except Exception:
                pass
        
        # Remove chunk directory
        if chunk_dir.exists():
            try:
                import shutil
                shutil.rmtree(chunk_dir)
            except Exception:
                pass
        
        if result.returncode != 0:
            return (input_file.name, False, f"Merge failed: {result.stderr[:100]}", 0)
        
        if not output_file.exists() or output_file.stat().st_size == 0:
            return (input_file.name, False, "Output file empty", 0)
        
        print(f"    ✓ {input_file.name}: {total_points:,} points")
        return (input_file.name, True, "Success", total_points)
        
    except Exception as e:
        print(f"    ✗ {input_file.name}: {e}")
        return (input_file.name, False, str(e), 0)


def subsample_simple(
    input_file: Path,
    output_file: Path,
    resolution: float,
    pipeline_dir: Path,
    dimension_reduction: bool = True,
    subsampling_method: str = SUBSAMPLING_METHOD_CENTER_OF_MASS,
) -> Tuple[str, bool, str, int]:
    """
    Simple single-pass subsampling (fallback method) with COPC reader optimization.
    
    Uses COPC reader for efficient reading, but always outputs LAZ format.
    When dimension_reduction is True, only standard dimensions are written (minimal output).
    
    Args:
        input_file: Input file path
        output_file: Output file path
        resolution: Voxel resolution
        pipeline_dir: Directory for pipeline files
        dimension_reduction: If True, write only standard dimensions (no extra_dims).
        subsampling_method: SmartTile subsampling method.
    
    Returns:
        Tuple of (filename, success, message, point_count)
    """
    subsampling_method = normalize_subsampling_method(subsampling_method)
    try:
        is_copc = _is_copc_file(input_file)
        reader_type = "readers.copc" if is_copc else "readers.las"
        
        # Always output as LAZ format (compressed LAS)
        if not output_file.name.endswith('.laz'):
            output_file = output_file.parent / (output_file.stem + '.laz')
        
        writer_opts = {
            "type": "writers.las",
            "filename": str(output_file),
            "compression": True,
            "forward": "all",
        }
        if dimension_reduction:
            writer_opts["minor_version"] = 2
            writer_opts["dataformat_id"] = 0
        else:
            writer_opts["minor_version"] = 4
            writer_opts["extra_dims"] = "all"

        if subsampling_method == SUBSAMPLING_METHOD_CENTER_OF_MASS:
            if is_copc and dimension_reduction:
                point_count = center_of_mass_subsample_copc(input_file, output_file, resolution)
            else:
                point_count = center_of_mass_subsample_las(
                    input_file,
                    output_file,
                    resolution,
                    dimension_reduction=dimension_reduction,
                )
            return (input_file.name, True, "Success", point_count)

        pipeline = {
            "pipeline": [
                {"type": reader_type, "filename": str(input_file)},
                _voxel_subsampling_filter(resolution, subsampling_method),
                writer_opts,
            ]
        }
        
        pipeline_file = pipeline_dir / f"{input_file.stem}_simple.json"
        with open(pipeline_file, 'w') as f:
            json.dump(pipeline, f, indent=2)
        
        pdal_cmd = get_pdal_path()
        result = subprocess.run(
            [pdal_cmd, "pipeline", str(pipeline_file)],
            capture_output=True,
            text=True,
            check=False
        )
        
        if pipeline_file.exists():
            pipeline_file.unlink()
        
        if result.returncode != 0:
            # Fallback: if COPC reader failed, retry with readers.las
            if is_copc and ("copc" in result.stderr.lower() or "vlr" in result.stderr.lower()):
                print(f"    ⚠ {input_file.name}: COPC reader failed, falling back to readers.las")
                pipeline = {
                    "pipeline": [
                        {"type": "readers.las", "filename": str(input_file)},
                        _voxel_subsampling_filter(resolution, subsampling_method),
                        writer_opts,
                    ]
                }
                pipeline_file = pipeline_dir / f"{input_file.stem}_simple_fallback.json"
                with open(pipeline_file, 'w') as f:
                    json.dump(pipeline, f, indent=2)
                try:
                    result = subprocess.run(
                        [pdal_cmd, "pipeline", str(pipeline_file)],
                        capture_output=True, text=True, check=False,
                    )
                finally:
                    if pipeline_file.exists():
                        pipeline_file.unlink()
                if result.returncode != 0:
                    return (input_file.name, False, result.stderr[:200], 0)
            else:
                return (input_file.name, False, result.stderr[:200], 0)

        if not output_file.exists() or output_file.stat().st_size == 0:
            return (input_file.name, False, "Output file empty", 0)

        # Get point count
        point_count = 0
        try:
            info_result = subprocess.run(
                [pdal_cmd, "info", "--metadata", str(output_file)],
                capture_output=True,
                text=True,
                check=True
            )
            import re
            match = re.search(r'"count":\s*(\d+)', info_result.stdout)
            if match:
                point_count = int(match.group(1))
        except Exception:
            pass
        
        return (input_file.name, True, "Success", point_count)
        
    except Exception as e:
        return (input_file.name, False, str(e), 0)


def subsample_parallel(
    input_dir: Path,
    output_dir: Path,
    resolution: float,
    num_cores: int,
    num_threads: int,
    output_prefix: Optional[str] = None,
    dimension_reduction: bool = True,
    subsampling_method: str = SUBSAMPLING_METHOD_CENTER_OF_MASS,
) -> List[Path]:
    """
    Subsample all files in directory using parallel chunk processing.
    
    Files are processed sequentially (one at a time), but each file is split
    spatially into chunks along X-axis and processed in parallel.
    Uses the selected SmartTile subsampling method.
    When dimension_reduction is True, only standard LAS dimensions are written (minimal output).
    
    Args:
        input_dir: Directory containing input files
        output_dir: Directory for output files
        resolution: Voxel resolution in meters
        num_cores: Not used (kept for compatibility)
        num_threads: Number of spatial chunks per file (from TILE_PARAMS['threads'])
        output_prefix: Optional prefix for output filenames
        dimension_reduction: If True, write only standard dimensions (no extra_dims); default True = minimal.
        subsampling_method: "center-of-mass" or "nearest-to-centroid".
    
    Returns:
        List of created output file paths
    """
    # Create output directory
    subsampling_method = normalize_subsampling_method(subsampling_method)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create pipeline directory
    pipeline_dir = output_dir / "pipelines"
    pipeline_dir.mkdir(exist_ok=True)
    
    # Find input files
    # Get all LAZ files (both .laz and .copc.laz)
    # Note: *.laz will match both .laz and .copc.laz, so we use set to deduplicate
    input_files = sorted(set(list(input_dir.glob("*.laz")) + list(input_dir.glob("*.copc.laz"))))
    
    if not input_files:
        print(f"    No input files found in {input_dir}")
        return []
    
    # Convert resolution to cm for filename
    res_cm = int(resolution * 100)
    
    # Prepare tasks
    tasks = []
    for input_file in sorted(input_files):
        # Generate output filename
        stem = input_file.stem
        # Remove .copc suffix if present
        if stem.endswith('.copc'):
            stem = stem[:-5]
        
        # Extract original base filename by removing prefixes and resolution suffixes
        import re
        base_name = stem
        
        # Remove resolution suffix patterns (e.g., "_1cm", "_10cm", "_subsampled0.01m", "_subsampled0.1m")
        base_name = re.sub(r'_subsampled[\d.]+m$', '', base_name)
        base_name = re.sub(r'_\d+cm$', '', base_name)
        
        # Remove output_prefix if present at the start (e.g., "output_dir_100m_")
        if output_prefix and base_name.startswith(output_prefix + '_'):
            base_name = base_name[len(output_prefix) + 1:]
        
        # Remove any remaining prefix patterns that look like "something_100m_" or "output_dir_100m_"
        base_name = re.sub(r'^[^_]+_\d+m_', '', base_name)
        
        # For tiled files, try to extract tile ID (c##_r##) pattern
        tile_match = re.search(r'(c\d+_r\d+)', base_name)
        if tile_match:
            # Keep tile ID for tiled files
            tile_id = tile_match.group(1)
            # Extract base name before tile ID if there's a prefix
            base_before_tile = base_name[:tile_match.start()]
            if base_before_tile and base_before_tile.endswith('_'):
                base_before_tile = base_before_tile[:-1]
            # Remove any remaining prefix from base_before_tile
            if base_before_tile:
                base_before_tile = re.sub(r'^[^_]+_\d+m_', '', base_before_tile)
                if base_before_tile:
                    output_name = f"{base_before_tile}_{tile_id}_subsampled_{res_cm}cm.laz"
                else:
                    output_name = f"{tile_id}_subsampled_{res_cm}cm.laz"
            else:
                output_name = f"{tile_id}_subsampled_{res_cm}cm.laz"
        else:
            # Single file or no tile ID - use clean base name
            # Remove any remaining prefix patterns
            base_name = re.sub(r'^[^_]+_\d+m_', '', base_name)
            output_name = f"{base_name}_subsampled_{res_cm}cm.laz"
        
        output_file = output_dir / output_name
        
        # Skip if already exists
        if output_file.exists() and output_file.stat().st_size > 0:
            print(f"    ⊙ Skipping {input_file.name} (already exists)")
            continue
        
        tasks.append((
            input_file,
            output_file,
            resolution,
            pipeline_dir,
            num_threads,
            dimension_reduction,
            subsampling_method,
        ))
    
    if not tasks:
        print(f"    ✓ All files already subsampled")
        return list(output_dir.glob("*.laz"))
    
    print(f"    Files to process: {len(tasks)}")
    print(f"    Processing mode: Sequential (one file at a time)")
    print(f"    Chunk parallelism: {num_threads} chunks per file (parallel)")
    print(f"    Subsampling method: {subsampling_method}")
    print()
    
    # Process files sequentially, but chunks within each file in parallel
    successful = 0
    failed = 0
    total_points = 0
    
    for task in tasks:
        filename, success, message, point_count = subsample_single_file(task)
        if success:
            successful += 1
            total_points += point_count
        else:
            failed += 1
            print(f"    ✗ {filename}: {message}")
    
    # Clean up pipeline directory
    if pipeline_dir.exists() and not any(pipeline_dir.iterdir()):
        pipeline_dir.rmdir()
    
    print()
    print(f"    ═══ Summary ═══")
    print(f"    Complete: {successful} successful, {failed} failed")
    print(f"    Total points: {total_points:,}")
    
    return list(output_dir.glob("*.laz"))


def run_subsample_pipeline(
    tiles_dir: Path,
    res1: float = 0.01,
    res2: float = 0.1,
    num_cores: Optional[int] = None,
    num_threads: Optional[int] = None,
    output_prefix: Optional[str] = None,
    output_base_dir: Optional[Path] = None,
    dimension_reduction: bool = True,
    subsampling_method: str = SUBSAMPLING_METHOD_CENTER_OF_MASS,
) -> Tuple[Path, Path]:
    """
    Run the complete subsampling pipeline.
    
    Steps:
    1. Subsample tiles to resolution 1 (default: 1cm)
    2. Subsample resolution 1 files to resolution 2 (default: 10cm)
    
    Files are processed sequentially (one at a time), but each file is split
    spatially into num_threads chunks along X-axis and processed in parallel.
    When dimension_reduction is True (default), only standard LAS dimensions are written (minimal output).
    
    Args:
        tiles_dir: Directory containing tile COPC files (input)
        res1: First resolution in meters (default: 0.01 = 1cm)
        res2: Second resolution in meters (default: 0.1 = 10cm)
        num_cores: Not used (kept for compatibility)
        num_threads: Number of parallel chunks per file (default: from TILE_PARAMS['threads'])
        output_prefix: Optional prefix for output filenames
        output_base_dir: Base directory for output (default: parent of tiles_dir)
        dimension_reduction: If True, write only standard dimensions (minimal); if False, keep extra_dims (e.g. PredInstance).
        subsampling_method: "center-of-mass" or "nearest-to-centroid".
    
    Returns:
        Tuple of (subsampled_res1_dir, subsampled_res2_dir)
    """
    subsampling_method = normalize_subsampling_method(subsampling_method)
    # Auto-detect CPU count
    if num_cores is None:
        num_cores = get_cpu_count()
    
    # Get num_threads from TILE_PARAMS
    if num_threads is None:
        num_threads = TILE_PARAMS.get('threads', 10)
    
    # Convert to cm for display/filenames (but use simple directory names)
    res1_cm = int(res1 * 100)
    res2_cm = int(res2 * 100)
    
    # Define output directories - use output_base_dir if provided, otherwise use tiles_dir's parent
    if output_base_dir is None:
        output_base_dir = tiles_dir.parent
    
    # Create output directories directly under output_base_dir
    subsampled_res1_dir = output_base_dir / "subsampled_res1"
    subsampled_res2_dir = output_base_dir / "subsampled_res2"
    
    print("=" * 60)
    print("3DTrees Subsampling Pipeline")
    print("=" * 60)
    print(f"Input: {tiles_dir}")
    print(f"Resolution 1: {res1}m ({res1_cm}cm)")
    print(f"Resolution 2: {res2}m ({res2_cm}cm)")
    print(f"CPU cores: {num_cores}")
    print(f"Threads (chunks per file): {num_threads}")
    print(f"Dimension reduction: {dimension_reduction} ({'minimal (standard dims only)' if dimension_reduction else 'keep all (extra_dims preserved)'})")
    print(f"Subsampling method: {subsampling_method}")
    print()
    
    # Step 1: Subsample to resolution 1
    print("=" * 60)
    print(f"Step 1: Subsampling to {res1_cm}cm ({res1}m)")
    print("=" * 60)
    
    res1_files = subsample_parallel(
        input_dir=tiles_dir,
        output_dir=subsampled_res1_dir,
        resolution=res1,
        num_cores=num_cores,
        num_threads=num_threads,
        output_prefix=output_prefix,
        dimension_reduction=dimension_reduction,
        subsampling_method=subsampling_method,
    )
    
    if not res1_files:
        raise ValueError(f"No files created in {subsampled_res1_dir}")
    
    print(f"\n  ✓ {res1_cm}cm subsampling complete: {len(res1_files)} files")
    print(f"  Output: {subsampled_res1_dir}")
    
    # Step 2: Subsample resolution 1 to resolution 2
    print()
    print("=" * 60)
    print(f"Step 2: Subsampling to {res2_cm}cm ({res2}m)")
    print("=" * 60)
    
    res2_files = subsample_parallel(
        input_dir=subsampled_res1_dir,
        output_dir=subsampled_res2_dir,
        resolution=res2,
        num_cores=num_cores,
        num_threads=num_threads,
        output_prefix=output_prefix,
        dimension_reduction=dimension_reduction,
        subsampling_method=subsampling_method,
    )
    
    if not res2_files:
        raise ValueError(f"No files created in {subsampled_res2_dir}")
    
    print(f"\n  ✓ {res2_cm}cm subsampling complete: {len(res2_files)} files")
    print(f"  Output: {subsampled_res2_dir}")
    
    # Summary
    print()
    print("=" * 60)
    print("Subsampling Pipeline Complete")
    print("=" * 60)
    print(f"  Resolution 1 ({res1_cm}cm): {len(res1_files)} files in {subsampled_res1_dir}")
    print(f"  Resolution 2 ({res2_cm}cm): {len(res2_files)} files in {subsampled_res2_dir}")
    
    return subsampled_res1_dir, subsampled_res2_dir


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="3DTrees Subsampling Pipeline - Parallel subsampling to multiple resolutions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--tiles_dir", "-i",
        type=Path,
        required=True,
        help="Directory containing tile COPC files"
    )
    
    parser.add_argument(
        "--res1",
        type=float,
        default=TILE_PARAMS.get('resolution_1', 0.01),
        help=f"First resolution in meters (default: {TILE_PARAMS.get('resolution_1', 0.01)})"
    )
    
    parser.add_argument(
        "--res2",
        type=float,
        default=TILE_PARAMS.get('resolution_2', 0.1),
        help=f"Second resolution in meters (default: {TILE_PARAMS.get('resolution_2', 0.1)})"
    )
    
    parser.add_argument(
        "--num_cores",
        type=int,
        default=None,
        help="Number of CPU cores (default: auto-detect, not used for chunking)"
    )
    
    parser.add_argument(
        "--num_threads",
        type=int,
        default=None,
        help=f"Number of spatial chunks per file for parallel processing (default: {TILE_PARAMS.get('threads', 5)})"
    )

    parser.add_argument(
        "--subsampling-method",
        "--subsampling_method",
        choices=sorted(SUBSAMPLING_METHODS),
        default=TILE_PARAMS.get("subsampling_method", SUBSAMPLING_METHOD_CENTER_OF_MASS),
        help=(
            "Subsampling method: center-of-mass averages XYZ per voxel; "
            "nearest-to-centroid preserves the previous PDAL voxel centroid nearest-neighbor behavior "
            f"(default: {TILE_PARAMS.get('subsampling_method', SUBSAMPLING_METHOD_CENTER_OF_MASS)})"
        ),
    )
    
    parser.add_argument(
        "--output_prefix",
        type=str,
        default=None,
        help="Optional prefix for output filenames"
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not args.tiles_dir.exists():
        print(f"Error: Tiles directory does not exist: {args.tiles_dir}")
        sys.exit(1)
    
    # Run pipeline
    try:
        res1_dir, res2_dir = run_subsample_pipeline(
            tiles_dir=args.tiles_dir,
            res1=args.res1,
            res2=args.res2,
            num_cores=args.num_cores,
            num_threads=args.num_threads,
            output_prefix=args.output_prefix,
            subsampling_method=args.subsampling_method,
        )
        print(f"\nSubsampled files ready:")
        print(f"  Resolution 1: {res1_dir}")
        print(f"  Resolution 2: {res2_dir}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
