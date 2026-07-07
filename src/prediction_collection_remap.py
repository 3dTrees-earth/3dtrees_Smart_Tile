#!/usr/bin/env python3
"""Stream finalized prediction collections onto original point-cloud files.

Prediction collections are expected to be finalized independently before this
step. Each collection keeps its model-specific dimension names; duplicate names
across collections fail early instead of being renamed at product time.
"""

from __future__ import annotations

import gc
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import laspy
import numpy as np
from scipy.spatial import cKDTree

from point_cloud_metadata import (
    bounds_overlap_xy,
    copy_single_source_header,
    extra_bytes_params_from_dimension_info,
    extra_bytes_params_from_params,
    point_cloud_files,
    raw_point_cloud_files,
)
from dimension_transfer import next_available_suffix
from worker_budget import kdtree_query_workers


def prediction_collection_files(path: Path) -> List[Path]:
    """Return LAZ/LAS files for a prediction collection folder or single file."""
    path = Path(path)
    if path.is_file():
        if path.suffix.lower() == ".las" or path.name.lower().endswith(".laz"):
            return [path]
        return []
    return point_cloud_files(path)


def scan_prediction_collection_metadata(
    collections: List[Path],
    target_dims: Optional[Set[str]] = None,
) -> List[Dict[str, object]]:
    """Scan prediction collections and fail on duplicate output extra-dimension names."""
    collection_meta: List[Dict[str, object]] = []
    seen_dims: Dict[str, Path] = {}

    for collection in collections:
        files = prediction_collection_files(collection)
        if not files:
            raise ValueError(f"No LAZ/LAS files found in prediction collection: {collection}")

        file_meta = []
        collection_dims: Dict[str, laspy.ExtraBytesParams] = {}
        for file_path in files:
            with laspy.open(str(file_path), laz_backend=laspy.LazBackend.LazrsParallel) as reader:
                header = reader.header
                bounds = (header.x_min, header.x_max, header.y_min, header.y_max)
                z_bounds = (header.z_min, header.z_max)
                extra_names = {dim.name for dim in header.point_format.extra_dimensions}
                for dim in header.point_format.extra_dimensions:
                    if target_dims is not None and dim.name not in target_dims:
                        continue
                    if dim.name not in collection_dims:
                        collection_dims[dim.name] = extra_bytes_params_from_dimension_info(dim)
                file_meta.append({
                    "path": file_path,
                    "bounds": bounds,
                    "z_bounds": z_bounds,
                    "extra_names": extra_names,
                })

        selected_names = sorted(collection_dims.keys())
        if not selected_names:
            wanted = ", ".join(sorted(target_dims)) if target_dims else "any extra dimensions"
            raise ValueError(f"Prediction collection {collection} has no selected dimensions ({wanted})")

        for file_info in file_meta:
            missing = set(selected_names) - set(file_info["extra_names"])
            if missing:
                raise ValueError(
                    f"Prediction collection {collection} has inconsistent schema: "
                    f"{Path(file_info['path']).name} is missing {', '.join(sorted(missing))}"
                )

        for dim_name in selected_names:
            if dim_name in seen_dims:
                raise ValueError(
                    "Duplicate prediction dimension name across collections: "
                    f"{dim_name} appears in {seen_dims[dim_name]} and {collection}. "
                    "Model outputs must be uniquely named before final remap."
                )
            seen_dims[dim_name] = collection

        collection_meta.append(
            {
                "path": collection,
                "files": file_meta,
                "dims": selected_names,
                "extra_params": collection_dims,
            }
        )

    return collection_meta


def _prediction_output_names_for_used_dims(
    dim_names: List[str],
    used_names: Set[str],
) -> Dict[str, str]:
    """Resolve prediction output names against dimensions already on the original."""
    resolved: Dict[str, str] = {}
    for dim_name in dim_names:
        output_name = dim_name if dim_name not in used_names else next_available_suffix(dim_name, used_names)
        resolved[dim_name] = output_name
        used_names.add(output_name)
    return resolved


def _prediction_output_names_for_header(
    header,
    collection_meta: List[Dict[str, object]],
) -> Dict[str, str]:
    used_names = set(header.point_format.dimension_names)
    used_names.update(dim.name for dim in header.point_format.extra_dimensions)
    dim_names = [
        dim_name
        for coll_meta in collection_meta
        for dim_name in coll_meta["dims"]
    ]
    return _prediction_output_names_for_used_dims(dim_names, used_names)


def _log_prediction_name_collisions(output_names: Dict[str, str], input_name: str) -> None:
    renamed = [f"{src}->{dst}" for src, dst in output_names.items() if src != dst]
    if renamed:
        print(
            f"  Prediction dimensions renamed for {input_name}: {', '.join(renamed)}",
            flush=True,
        )


def load_collection_subset_for_bounds(
    coll_meta: Dict[str, object],
    bounds: Tuple[float, float, float, float],
    spatial_buffer: float,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Load prediction points/dim arrays inside a small XY bounds window."""
    min_x, max_x, min_y, max_y = bounds
    min_x -= spatial_buffer
    max_x += spatial_buffer
    min_y -= spatial_buffer
    max_y += spatial_buffer

    point_chunks = []
    dim_chunks: Dict[str, List[np.ndarray]] = {name: [] for name in coll_meta["dims"]}

    for file_info in coll_meta["files"]:
        if not bounds_overlap_xy(file_info["bounds"], (min_x, max_x, min_y, max_y), 0.0):
            continue
        source_path = Path(file_info["path"])
        if source_path.name.lower().endswith(".copc.laz"):
            z_min, z_max = file_info.get("z_bounds", (None, None))
            if z_min is None or z_max is None:
                z_min, z_max = -np.inf, np.inf
            query_bounds = laspy.copc.Bounds(
                mins=np.array([min_x, min_y, float(z_min)], dtype=np.float64),
                maxs=np.array([max_x, max_y, float(z_max)], dtype=np.float64),
            )
            with laspy.CopcReader.open(str(source_path)) as copc_reader:
                chunk = copc_reader.spatial_query(query_bounds)
                if len(chunk) == 0:
                    continue
                xs = np.asarray(chunk.x)
                ys = np.asarray(chunk.y)
                mask = (xs >= min_x) & (xs <= max_x) & (ys >= min_y) & (ys <= max_y)
                if not np.any(mask):
                    continue
                point_chunks.append(np.column_stack([xs[mask], ys[mask], np.asarray(chunk.z)[mask]]))
                for dim_name in coll_meta["dims"]:
                    dim_chunks[dim_name].append(np.asarray(chunk[dim_name])[mask])
        else:
            with laspy.open(str(source_path), laz_backend=laspy.LazBackend.LazrsParallel) as reader:
                for chunk in reader.chunk_iterator(5_000_000):
                    xs = np.asarray(chunk.x)
                    ys = np.asarray(chunk.y)
                    mask = (xs >= min_x) & (xs <= max_x) & (ys >= min_y) & (ys <= max_y)
                    if not np.any(mask):
                        continue
                    point_chunks.append(np.column_stack([xs[mask], ys[mask], np.asarray(chunk.z)[mask]]))
                    for dim_name in coll_meta["dims"]:
                        dim_chunks[dim_name].append(np.asarray(chunk[dim_name])[mask])

    if not point_chunks:
        raise ValueError(f"No prediction points found in {coll_meta['path']} for bounds {bounds}")

    points = np.concatenate(point_chunks, axis=0)
    dims = {name: np.concatenate(chunks) for name, chunks in dim_chunks.items()}
    return points, dims


def stream_add_collections_to_file(
    input_file: Path,
    output_file: Path,
    collection_meta: List[Dict[str, object]],
    spatial_buffer: float,
    tolerance: float,
    chunk_size: int = 5_000_000,
    kdtree_workers: int = 1,
    chunk_parallel_workers: int = 1,
) -> Tuple[int, int]:
    """Stream all prediction collections onto one original file in one write pass."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    chunk_spatial_buffer = max(spatial_buffer, tolerance * 2.0, 0.25)
    requested_chunk_workers = max(1, int(chunk_parallel_workers or 1))
    read_chunk_size, chunk_parallel_workers = _parallel_raw_chunk_plan(
        chunk_size,
        requested_chunk_workers,
    )
    if requested_chunk_workers > 1:
        print(
            f"    Raw chunk parallelism: {chunk_parallel_workers} worker(s), "
            f"{read_chunk_size:,} points/chunk/worker "
            f"(requested {requested_chunk_workers} worker(s), "
            f"chunk budget {chunk_size:,})",
            flush=True,
        )

    with laspy.open(str(input_file), laz_backend=laspy.LazBackend.LazrsParallel) as reader:
        header = copy_single_source_header(reader.header, preserve_extra_dimensions=True)
        output_names = _prediction_output_names_for_header(header, collection_meta)
        _log_prediction_name_collisions(output_names, input_file.name)
        extra_dims_to_add = []
        for coll_meta in collection_meta:
            for dim_name in coll_meta["dims"]:
                params = coll_meta["extra_params"][dim_name]
                extra_dims_to_add.append(
                    extra_bytes_params_from_params(params, name=output_names[dim_name])
                )
        if extra_dims_to_add:
            header.add_extra_dims(extra_dims_to_add)

        n_points = 0
        matched_total = 0
        with laspy.open(
            str(output_file),
            mode="w",
            header=header,
            laz_backend=laspy.LazBackend.LazrsParallel,
        ) as writer:
            if chunk_parallel_workers <= 1:
                for chunk in reader.chunk_iterator(read_chunk_size):
                    out_chunk, chunk_matches = _enrich_original_chunk(
                        chunk,
                        header,
                        collection_meta,
                        chunk_spatial_buffer,
                        tolerance,
                        kdtree_workers,
                        input_file.name,
                        output_names,
                    )
                    writer.write_points(out_chunk)
                    n_points += len(chunk)
                    matched_total += chunk_matches
                    del out_chunk
                    if n_points % 25_000_000 < len(chunk):
                        print(
                            f"    prediction collections -> {output_file.name}: "
                            f"{n_points:,} points",
                            flush=True,
                        )
            else:
                ready_chunks: Dict[int, Tuple[object, int, int]] = {}
                future_to_idx = {}
                next_submit_idx = 0
                next_write_idx = 0
                max_in_flight = chunk_parallel_workers

                with ThreadPoolExecutor(max_workers=chunk_parallel_workers) as executor:
                    for chunk in reader.chunk_iterator(read_chunk_size):
                        while len(future_to_idx) >= max_in_flight:
                            done, _ = wait(future_to_idx, return_when=FIRST_COMPLETED)
                            for future in done:
                                idx = future_to_idx.pop(future)
                                out_chunk, chunk_matches = future.result()
                                ready_chunks[idx] = (out_chunk, chunk_matches, len(out_chunk))
                            next_write_idx, n_points, matched_total = _write_ready_chunks_in_order(
                                writer,
                                ready_chunks,
                                next_write_idx,
                                output_file.name,
                                n_points,
                                matched_total,
                            )

                        future = executor.submit(
                            _enrich_original_chunk,
                            chunk,
                            header,
                            collection_meta,
                            chunk_spatial_buffer,
                            tolerance,
                            kdtree_workers,
                            input_file.name,
                            output_names,
                        )
                        future_to_idx[future] = next_submit_idx
                        next_submit_idx += 1

                    while future_to_idx:
                        done, _ = wait(future_to_idx, return_when=FIRST_COMPLETED)
                        for future in done:
                            idx = future_to_idx.pop(future)
                            out_chunk, chunk_matches = future.result()
                            ready_chunks[idx] = (out_chunk, chunk_matches, len(out_chunk))
                        next_write_idx, n_points, matched_total = _write_ready_chunks_in_order(
                            writer,
                            ready_chunks,
                            next_write_idx,
                            output_file.name,
                            n_points,
                            matched_total,
                        )

    gc.collect()
    return n_points, matched_total


def _parallel_raw_chunk_plan(chunk_size: int, requested_workers: int) -> Tuple[int, int]:
    """Plan raw chunk reads so total in-flight points stay near the chunk budget."""
    chunk_size = max(1, int(chunk_size or 1))
    requested_workers = max(1, int(requested_workers or 1))
    if requested_workers <= 1:
        return chunk_size, 1

    min_parallel_chunk_size = min(chunk_size, 5_000_000)
    read_chunk_size = min(
        chunk_size,
        max(min_parallel_chunk_size, int(np.ceil(chunk_size / requested_workers))),
    )
    active_workers = min(requested_workers, int(np.ceil(chunk_size / read_chunk_size)))
    return read_chunk_size, max(1, active_workers)


def _add_prediction_dims_to_header(
    header,
    collection_meta: List[Dict[str, object]],
    input_name: str,
) -> Dict[str, str]:
    """Add selected prediction dimensions to an output header with collision-safe names."""
    output_names = _prediction_output_names_for_header(header, collection_meta)
    _log_prediction_name_collisions(output_names, input_name)
    extra_dims_to_add = []
    for coll_meta in collection_meta:
        for dim_name in coll_meta["dims"]:
            params = coll_meta["extra_params"][dim_name]
            extra_dims_to_add.append(
                extra_bytes_params_from_params(params, name=output_names[dim_name])
            )
    if extra_dims_to_add:
        header.add_extra_dims(extra_dims_to_add)
    return output_names


def _copy_source_record_dimensions(source_points, out_record) -> None:
    """Copy dimensions present on a source point record into an output record."""
    output_dim_names = set(out_record.point_format.dimension_names)
    output_dim_names.update(dim.name for dim in out_record.point_format.extra_dimensions)
    for dim_name in source_points.point_format.dimension_names:
        if dim_name in output_dim_names:
            out_record[dim_name] = source_points[dim_name]
    for dim in source_points.point_format.extra_dimensions:
        if dim.name in output_dim_names:
            out_record[dim.name] = source_points[dim.name]


def _enrich_original_chunk(
    chunk,
    header,
    collection_meta: List[Dict[str, object]],
    chunk_spatial_buffer: float,
    tolerance: float,
    kdtree_workers: int,
    input_name: str,
    output_names: Dict[str, str],
) -> Tuple[object, int]:
    """Copy one original chunk and add prediction collection dimensions."""
    chunk_points = np.column_stack([chunk.x, chunk.y, chunk.z])
    chunk_bounds = (
        float(np.min(chunk_points[:, 0])),
        float(np.max(chunk_points[:, 0])),
        float(np.min(chunk_points[:, 1])),
        float(np.max(chunk_points[:, 1])),
    )
    out_chunk = laspy.ScaleAwarePointRecord.zeros(len(chunk), header=header)
    for dim_name in chunk.point_format.dimension_names:
        if dim_name in out_chunk.point_format.dimension_names:
            out_chunk[dim_name] = chunk[dim_name]

    matched_total = 0
    for coll_meta in collection_meta:
        source_points, source_dims = load_collection_subset_for_bounds(
            coll_meta,
            chunk_bounds,
            chunk_spatial_buffer,
        )
        tree = cKDTree(source_points)
        distances, indices = tree.query(chunk_points, workers=kdtree_workers)
        matched = distances <= tolerance
        matched_count = int(np.count_nonzero(matched))
        if matched_count != len(chunk):
            raise ValueError(
                f"Prediction collection {coll_meta['path']} matched "
                f"{matched_count:,}/{len(chunk):,} points in chunk from {input_name} "
                f"within tolerance {tolerance} m"
            )
        for dim_name, values in source_dims.items():
            out_chunk[output_names[dim_name]] = values[indices]
        matched_total += matched_count
        del source_points, source_dims, tree, distances, indices

    del chunk_points
    return out_chunk, matched_total


def _write_ready_chunks_in_order(
    writer,
    ready_chunks: Dict[int, Tuple[object, int, int]],
    next_write_idx: int,
    output_name: str,
    n_points: int,
    matched_total: int,
) -> Tuple[int, int, int]:
    """Write completed chunk results while preserving source point order."""
    while next_write_idx in ready_chunks:
        out_chunk, chunk_matches, chunk_len = ready_chunks.pop(next_write_idx)
        writer.write_points(out_chunk)
        n_points += chunk_len
        matched_total += chunk_matches
        del out_chunk
        if n_points % 25_000_000 < chunk_len:
            print(
                f"    prediction collections -> {output_name}: "
                f"{n_points:,} points",
                flush=True,
            )
        next_write_idx += 1
    return next_write_idx, n_points, matched_total


def _copc_spatial_windows(header, num_spatial_chunks: int) -> List[Tuple[float, float, bool]]:
    """Return X windows for COPC spatial-query enrichment."""
    min_x = float(header.x_min)
    max_x = float(header.x_max)
    if max_x <= min_x:
        return [(min_x, max_x, True)]

    chunks = max(1, int(num_spatial_chunks or 1))
    step = (max_x - min_x) / chunks
    windows = []
    for idx in range(chunks):
        start = min_x + idx * step
        stop = max_x if idx == chunks - 1 else min_x + (idx + 1) * step
        windows.append((start, stop, idx == chunks - 1))
    return windows


def _query_copc_window(copc_reader, header, min_x: float, max_x: float):
    """Load one COPC X window with native COPC spatial query."""
    bounds = laspy.copc.Bounds(
        mins=np.array([min_x, float(header.y_min), float(header.z_min)], dtype=np.float64),
        maxs=np.array([max_x, float(header.y_max), float(header.z_max)], dtype=np.float64),
    )
    return copc_reader.spatial_query(bounds)


def stream_add_collections_to_copc_file_spatial(
    input_file: Path,
    output_file: Path,
    collection_meta: List[Dict[str, object]],
    spatial_buffer: float,
    tolerance: float,
    num_spatial_chunks: int = 4,
    kdtree_workers: int = 1,
) -> Tuple[int, int]:
    """Spatial-query a COPC original and write one enriched LAZ output."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    chunk_spatial_buffer = max(spatial_buffer, tolerance * 2.0, 0.25)

    with laspy.CopcReader.open(str(input_file)) as copc_reader:
        source_header = copc_reader.header
        header = copy_single_source_header(source_header, preserve_extra_dimensions=True)
        output_names = _add_prediction_dims_to_header(header, collection_meta, input_file.name)
        windows = _copc_spatial_windows(source_header, num_spatial_chunks)

        n_points = 0
        matched_total = 0
        with laspy.open(
            str(output_file),
            mode="w",
            header=header,
            laz_backend=laspy.LazBackend.LazrsParallel,
        ) as writer:
            for window_idx, (min_x, max_x, include_upper) in enumerate(windows, start=1):
                source_points = _query_copc_window(copc_reader, source_header, min_x, max_x)
                if len(source_points) == 0:
                    continue

                xs = np.asarray(source_points.x)
                if include_upper:
                    mask = (xs >= min_x) & (xs <= max_x)
                else:
                    mask = (xs >= min_x) & (xs < max_x)
                if not np.any(mask):
                    continue
                if not np.all(mask):
                    source_points = source_points[mask]

                chunk_points = np.column_stack([source_points.x, source_points.y, source_points.z])
                chunk_bounds = (
                    float(np.min(chunk_points[:, 0])),
                    float(np.max(chunk_points[:, 0])),
                    float(np.min(chunk_points[:, 1])),
                    float(np.max(chunk_points[:, 1])),
                )
                out_chunk = laspy.ScaleAwarePointRecord.zeros(len(source_points), header=header)
                _copy_source_record_dimensions(source_points, out_chunk)

                for coll_meta in collection_meta:
                    prediction_points, prediction_dims = load_collection_subset_for_bounds(
                        coll_meta,
                        chunk_bounds,
                        chunk_spatial_buffer,
                    )
                    tree = cKDTree(prediction_points)
                    distances, indices = tree.query(chunk_points, workers=kdtree_workers)
                    matched = distances <= tolerance
                    matched_count = int(np.count_nonzero(matched))
                    if matched_count != len(source_points):
                        raise ValueError(
                            f"Prediction collection {coll_meta['path']} matched "
                            f"{matched_count:,}/{len(source_points):,} points in COPC window "
                            f"{window_idx}/{len(windows)} from {input_file.name} "
                            f"within tolerance {tolerance} m"
                        )
                    for dim_name, values in prediction_dims.items():
                        out_chunk[output_names[dim_name]] = values[indices]
                    matched_total += matched_count
                    del prediction_points, prediction_dims, tree, distances, indices

                writer.write_points(out_chunk)
                n_points += len(source_points)
                del source_points, chunk_points, out_chunk
                print(
                    f"    COPC spatial remap {input_file.name}: "
                    f"window {window_idx}/{len(windows)}, {n_points:,} points",
                    flush=True,
                )

    gc.collect()
    return n_points, matched_total


def _existing_output_is_reusable(
    input_file: Path,
    output_file: Path,
    expected_prediction_dims: List[str],
) -> bool:
    """Return True when an existing enriched original matches this remap request."""
    if not output_file.exists() or output_file.stat().st_size == 0:
        return False
    try:
        with laspy.open(str(input_file), laz_backend=laspy.LazBackend.LazrsParallel) as input_reader:
            expected_count = int(input_reader.header.point_count)
            input_dims = set(input_reader.header.point_format.dimension_names)
            input_dims.update(dim.name for dim in input_reader.header.point_format.extra_dimensions)
            expected_output_dims = set(
                _prediction_output_names_for_used_dims(
                    expected_prediction_dims,
                    set(input_dims),
                ).values()
            )
        with laspy.open(str(output_file), laz_backend=laspy.LazBackend.LazrsParallel) as output_reader:
            if int(output_reader.header.point_count) != expected_count:
                return False
            output_dims = set(output_reader.header.point_format.dimension_names)
            output_dims.update(dim.name for dim in output_reader.header.point_format.extra_dimensions)
        return expected_output_dims.issubset(output_dims)
    except Exception as exc:
        print(f"  Existing output is not reusable ({output_file.name}): {exc}", flush=True)
        return False


def remap_prediction_collections_to_original_files(
    collections: List[Path],
    original_input_dir: Path,
    output_dir: Path,
    tolerance: float = 0.1,
    num_threads: int = 4,
    retile_buffer: float = 2.0,
    target_dims: Optional[Set[str]] = None,
    chunk_size: int = 5_000_000,
    num_spatial_chunks: Optional[int] = None,
) -> None:
    """Remap finalized prediction collections onto original files."""
    print(f"\n{'=' * 60}", flush=True)
    print("Remapping prediction collections to original input files", flush=True)
    print(f"{'=' * 60}", flush=True)

    if not collections:
        raise ValueError("At least one prediction collection is required")
    original_files = raw_point_cloud_files(original_input_dir)
    if not original_files:
        raise ValueError(f"No LAZ/LAS files found in original input dir: {original_input_dir}")
    print("  Raw-original mode: ignoring COPC twins in original input dir", flush=True)

    collection_meta = scan_prediction_collection_metadata(collections, target_dims=target_dims)
    for coll_meta in collection_meta:
        print(
            f"  Collection {coll_meta['path']}: {', '.join(coll_meta['dims'])}",
            flush=True,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    spatial_buffer = max(tolerance * 2, 1.0) + retile_buffer
    expected_prediction_dims = [
        dim_name
        for coll_meta in collection_meta
        for dim_name in coll_meta["dims"]
    ]
    files_to_process = []
    skipped = 0
    stale = 0
    for input_file in original_files:
        output_name = input_file.name.replace(".copc.laz", ".laz")
        output_file = output_dir / output_name
        if output_file.exists():
            if _existing_output_is_reusable(input_file, output_file, expected_prediction_dims):
                skipped += 1
                continue
            stale += 1
            try:
                output_file.unlink()
            except OSError as exc:
                raise RuntimeError(f"Could not replace stale output {output_file}: {exc}") from exc
        files_to_process.append((input_file, output_file))

    if skipped:
        print(f"  Skipping {skipped} already processed files", flush=True)
    if stale:
        print(f"  Reprocessing {stale} stale existing output file(s)", flush=True)
    if not files_to_process:
        print("  All original files already processed", flush=True)
        return

    def process_one(args):
        input_file, output_file = args
        try:
            with laspy.open(str(input_file), laz_backend=laspy.LazBackend.LazrsParallel) as reader:
                n_points = reader.header.point_count

            if input_file.name.lower().endswith(".copc.laz"):
                spatial_chunks = num_spatial_chunks or max(1, num_threads)
                print(
                    f"  COPC original fast path: {input_file.name} "
                    f"with {spatial_chunks} spatial query window(s)",
                    flush=True,
                )
                stage_points, matched_total = stream_add_collections_to_copc_file_spatial(
                    input_file,
                    output_file,
                    collection_meta,
                    spatial_buffer,
                    tolerance,
                    num_spatial_chunks=spatial_chunks,
                    kdtree_workers=query_workers,
                )
            else:
                stage_points, matched_total = stream_add_collections_to_file(
                    input_file,
                    output_file,
                    collection_meta,
                    spatial_buffer,
                    tolerance,
                    chunk_size=chunk_size,
                    kdtree_workers=query_workers,
                    chunk_parallel_workers=active_raw_chunk_workers,
                )
            if stage_points != n_points:
                raise ValueError(
                    f"Streaming remap wrote {stage_points:,}/{n_points:,} points "
                    f"for {input_file.name}"
                )

            gc.collect()
            return (input_file.name, n_points, matched_total, True, "Success")
        except Exception as exc:
            return (input_file.name, 0, 0, False, str(exc))

    total_points = 0
    total_matches = 0
    parallel_workers = min(max(1, num_threads), len(files_to_process))
    if parallel_workers == 1:
        raw_chunk_workers = max(1, min(num_threads, num_spatial_chunks or num_threads))
        _, active_raw_chunk_workers = _parallel_raw_chunk_plan(chunk_size, raw_chunk_workers)
        query_workers = kdtree_query_workers(num_threads, active_raw_chunk_workers)
    else:
        raw_chunk_workers = 1
        active_raw_chunk_workers = 1
        query_workers = kdtree_query_workers(num_threads, parallel_workers)
    print(
        f"  Processing {len(files_to_process)} original files with {parallel_workers} worker(s); "
        f"{active_raw_chunk_workers} raw chunk worker(s)"
        f"{f' (requested {raw_chunk_workers})' if raw_chunk_workers != active_raw_chunk_workers else ''}; "
        f"{query_workers} KDTree query worker(s) each",
        flush=True,
    )
    if parallel_workers > 1:
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            results = list(executor.map(process_one, files_to_process))
    else:
        results = [process_one(args) for args in files_to_process]

    failures = []
    for idx, (filename, n_points, matched_total, success, message) in enumerate(results, start=1):
        if success:
            expected = n_points * len(collection_meta)
            match_pct = (matched_total / expected * 100) if expected else 0.0
            print(
                f"  [{idx}/{len(results)}] {filename}: "
                f"{matched_total:,}/{expected:,} collection-point matches ({match_pct:.1f}%)",
                flush=True,
            )
            total_points += expected
            total_matches += matched_total
        else:
            failures.append(f"{filename}: {message}")
            print(f"  [{idx}/{len(results)}] FAILED {filename}: {message}", flush=True)

    if failures:
        raise RuntimeError("Multi-collection remap failed:\n  " + "\n  ".join(failures))

    overall = (total_matches / total_points * 100) if total_points else 0.0
    print(
        f"\n  ✓ Multi-collection remap complete: "
        f"{total_matches:,}/{total_points:,} matches ({overall:.1f}%)",
        flush=True,
    )
