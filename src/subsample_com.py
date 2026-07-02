#!/usr/bin/env python3
"""Center-of-mass subsampling strategies for SmartTile."""

from __future__ import annotations

import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import laspy
import numpy as np


COPC_COM_TARGET_WINDOW_CELLS = 500
COPC_COM_MAX_WINDOW_SIZE = 20.0
COPC_COM_DENSE_BIN_LIMIT = 5_000_000


def process_pool_kwargs() -> dict:
    """Use a clean worker start method to avoid inherited COPC/lazrs thread state."""
    return {"mp_context": multiprocessing.get_context("spawn")}


def _is_stale_copc_vlr(vlr) -> bool:
    return getattr(vlr, "user_id", "") == "copc" and getattr(vlr, "record_id", None) in (1, 2)


def _is_extra_bytes_vlr(vlr) -> bool:
    return getattr(vlr, "user_id", "") == "LASF_Spec" and getattr(vlr, "record_id", None) == 4


def make_center_of_mass_header(source_header: laspy.LasHeader, dimension_reduction: bool) -> laspy.LasHeader:
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


def copy_non_xyz_dimensions(source_las: laspy.LasData, output_las: laspy.LasData, indices: np.ndarray) -> None:
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


def point_record_to_xyz(points) -> np.ndarray:
    coords = np.empty((len(points), 3), dtype=np.float64)
    coords[:, 0] = points.x
    coords[:, 1] = points.y
    coords[:, 2] = points.z
    return coords


def aggregate_center_of_mass_sparse(coords: np.ndarray, resolution: float) -> np.ndarray:
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


def aggregate_center_of_mass_bincount(coords: np.ndarray, resolution: float) -> Optional[np.ndarray]:
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


def aggregate_center_of_mass_xyz(points, resolution: float) -> np.ndarray:
    coords = point_record_to_xyz(points)
    if len(coords) == 0:
        return np.empty((0, 3), dtype=np.float64)
    centers = aggregate_center_of_mass_bincount(coords, resolution)
    if centers is not None:
        return centers
    return aggregate_center_of_mass_sparse(coords, resolution)


def aligned_edges(start: float, stop: float, step: float, align: float) -> List[Tuple[float, float]]:
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


def iter_copc_center_of_mass_windows(header: laspy.LasHeader, resolution: float) -> Iterable[object]:
    """Yield voxel-aligned, half-open XY bounds for COPC center-of-mass queries."""
    from laspy.copc import Bounds

    step = _copc_com_window_size(resolution)
    x_edges = aligned_edges(header.x_min, header.x_max, step, resolution)
    y_edges = aligned_edges(header.y_min, header.y_max, step, resolution)
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
        return (window_idx, point_count, aggregate_center_of_mass_xyz(points, resolution))


def write_center_of_mass_points(writer, header: laspy.LasHeader, centers: np.ndarray) -> int:
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
        windows = list(iter_copc_center_of_mass_windows(header, resolution))

    if not windows:
        raise ValueError(f"No COPC query windows available for {input_file}")

    output_header = make_center_of_mass_header(header, dimension_reduction=True)
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
                    centers = aggregate_center_of_mass_xyz(points, resolution)
                    total_output_points += write_center_of_mass_points(writer, output_header, centers)
        else:
            worker_count = min(num_workers, len(windows))
            max_in_flight = min(len(windows), worker_count * 2)
            print(f"      → COPC COM window parallelism: {worker_count} workers")

            with ProcessPoolExecutor(max_workers=worker_count, **process_pool_kwargs()) as executor:
                futures = {}
                next_submit = 0
                next_write = 0

                def submit_until_capacity():
                    nonlocal next_submit
                    while next_submit < len(windows) and len(futures) < max_in_flight:
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
                            total_output_points += write_center_of_mass_points(
                                writer,
                                output_header,
                                centers_to_write,
                            )
                            next_write += 1
                        submit_until_capacity()
                        break

                while next_write in pending_centers:
                    centers_to_write = pending_centers.pop(next_write)
                    total_output_points += write_center_of_mass_points(
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

    coords = point_record_to_xyz(source_las)
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

    header = make_center_of_mass_header(source_las.header, dimension_reduction)
    output_las = laspy.LasData(header)
    output_las.points = laspy.ScaleAwarePointRecord.zeros(len(selected_indices), header=header)
    copy_non_xyz_dimensions(source_las, output_las, selected_indices)
    output_las.x = mean_coords[:, 0]
    output_las.y = mean_coords[:, 1]
    output_las.z = mean_coords[:, 2]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_las.write(str(output_file), do_compress=output_file.suffix.lower() == ".laz")
    return len(selected_indices)
