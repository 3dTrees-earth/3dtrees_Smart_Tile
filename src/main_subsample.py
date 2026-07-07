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
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import laspy

# Import parameters
from parameters import TILE_PARAMS
from point_cloud_metadata import point_cloud_files
from subsample_chunk_worker import subsample_tile_chunk
from subsample_com import (
    COPC_COM_DENSE_BIN_LIMIT,
    COPC_COM_MAX_WINDOW_SIZE,
    COPC_COM_TARGET_WINDOW_CELLS,
    aggregate_center_of_mass_xyz as _aggregate_center_of_mass_xyz,
    aligned_edges as _aligned_edges,
    center_of_mass_subsample_copc,
    center_of_mass_subsample_las,
    iter_copc_center_of_mass_windows as _iter_copc_center_of_mass_windows,
    process_pool_kwargs as _process_pool_kwargs,
    write_center_of_mass_points as _write_center_of_mass_points,
)
from subsample_methods import (
    SUBSAMPLING_METHOD_CENTER_OF_MASS,
    SUBSAMPLING_METHOD_NEAREST_TO_CENTROID,
    SUBSAMPLING_METHODS,
    is_copc_file as _is_copc_file,
    normalize_subsampling_method,
    voxel_subsampling_filter as _voxel_subsampling_filter,
)
from subsample_outputs import (
    convert_laz_output_to_copc,
    copc_output_path,
    laz_output_path,
    subsample_output_files,
    temporary_laz_path,
)
from subsample_encoding import encoding_summary, safe_las_writer_encoding_options


SUBSAMPLE_MANIFEST_FILENAME = ".smarttile_subsample_manifest.json"


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


def _subsample_input_files(input_dir: Path) -> List[Path]:
    """Return point-cloud files for subsampling, preferring COPC twins when present."""
    return point_cloud_files(input_dir)


def _subsample_manifest_path(output_dir: Path) -> Path:
    return output_dir / SUBSAMPLE_MANIFEST_FILENAME


def _read_subsample_manifest(output_dir: Path) -> dict:
    manifest_file = _subsample_manifest_path(output_dir)
    if not manifest_file.exists():
        return {"version": 1, "outputs": {}}
    try:
        with manifest_file.open() as handle:
            manifest = json.load(handle)
    except Exception:
        return {"version": 1, "outputs": {}}
    if not isinstance(manifest, dict):
        return {"version": 1, "outputs": {}}
    manifest.setdefault("version", 1)
    manifest.setdefault("outputs", {})
    return manifest


def _write_subsample_manifest(output_dir: Path, manifest: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with _subsample_manifest_path(output_dir).open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)


def _subsample_source_fingerprint(input_file: Path) -> dict:
    stat = input_file.stat()
    return {
        "name": input_file.name,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _subsample_request_signature(
    input_file: Path,
    resolution: float,
    dimension_reduction: bool,
    subsampling_method: str,
    output_copc: bool,
) -> dict:
    return {
        "source": _subsample_source_fingerprint(input_file),
        "resolution": float(resolution),
        "dimension_reduction": bool(dimension_reduction),
        "subsampling_method": normalize_subsampling_method(subsampling_method),
        "output_copc": bool(output_copc),
    }


def _existing_subsample_is_reusable(
    output_file: Path,
    manifest: dict,
    expected_signature: dict,
) -> bool:
    """Return True when an existing subsample matches the current request."""
    if not output_file.exists() or output_file.stat().st_size == 0:
        return False
    entry = manifest.get("outputs", {}).get(output_file.name)
    if not isinstance(entry, dict):
        return False
    return entry.get("signature") == expected_signature


def _record_subsample_output(
    output_dir: Path,
    output_file: Path,
    signature: dict,
    point_count: int,
) -> None:
    manifest = _read_subsample_manifest(output_dir)
    outputs = manifest.setdefault("outputs", {})
    outputs[output_file.name] = {
        "signature": signature,
        "point_count": int(point_count),
    }
    _write_subsample_manifest(output_dir, manifest)


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


def _subsample_las_writer_options(
    input_file: Path,
    output_file: Path,
    dimension_reduction: bool,
    xy_bounds: Optional[Tuple[float, float, float, float]] = None,
) -> dict:
    """Return a LAS writer stage with explicit safe scale/offset encoding."""
    writer_opts = {
        "type": "writers.las",
        "filename": str(output_file),
        "compression": True,
        "forward": "all",
    }
    writer_opts.update(safe_las_writer_encoding_options(input_file, xy_bounds=xy_bounds))
    if dimension_reduction:
        writer_opts["minor_version"] = 2
        writer_opts["dataformat_id"] = 0
    else:
        writer_opts["minor_version"] = 4
        writer_opts["extra_dims"] = "all"
    return writer_opts


def subsample_single_file(args: Tuple[Path, Path, float, Path, int, bool, str, bool]) -> Tuple[str, bool, str, int]:
    """
    Subsample a single file by splitting it into subtiles along X-axis and processing in parallel.

    Process:
    1. Split tile into num_threads subtiles along X-axis only
    2. Subsample each subtile in parallel using ProcessPoolExecutor (true CPU parallelism)
    3. Merge all subsampled subtiles back together

    Args:
        args: Tuple of (input_file, output_file, resolution, pipeline_dir, num_threads, dimension_reduction, subsampling_method, output_copc)

    Returns:
        Tuple of (filename, success, message, point_count)
    """
    if len(args) == 7:
        input_file, output_file, resolution, pipeline_dir, num_threads, dimension_reduction, subsampling_method = args
        output_copc = False
    else:
        input_file, output_file, resolution, pipeline_dir, num_threads, dimension_reduction, subsampling_method, output_copc = args
    subsampling_method = normalize_subsampling_method(subsampling_method)
    target_output_file = copc_output_path(output_file) if output_copc else laz_output_path(output_file)
    try:
        print(f"    → Processing {input_file.name}...")

        if (
            subsampling_method == SUBSAMPLING_METHOD_CENTER_OF_MASS
            and dimension_reduction
            and _is_copc_file(input_file)
        ):
            print("      → Using COPC voxel-aligned center-of-mass windows")
            work_output_file = (
                temporary_laz_path(target_output_file, pipeline_dir)
                if output_copc
                else target_output_file
            )
            point_count = center_of_mass_subsample_copc(
                input_file,
                work_output_file,
                resolution,
                num_workers=num_threads,
            )
            if output_copc:
                print(f"      → Converting {work_output_file.name} to COPC")
                if not convert_laz_output_to_copc(
                    work_output_file,
                    target_output_file,
                    source_metadata_file=input_file,
                ):
                    return (input_file.name, False, "COPC conversion failed", 0)
                try:
                    work_output_file.unlink()
                except OSError:
                    pass
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
                output_copc,
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
                output_copc,
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
        with ProcessPoolExecutor(max_workers=num_threads, **_process_pool_kwargs()) as executor:
            futures = [executor.submit(subsample_tile_chunk, task) for task in chunk_tasks]
            for future in as_completed(futures):
                chunk_file, point_count = future.result()
                if chunk_file and chunk_file.exists():
                    chunk_files.append(chunk_file)
                    total_points += point_count

        if not chunk_files:
            return (input_file.name, False, "No chunks produced", 0)

        print(f"      → Merging {len(chunk_files)} subsampled subtiles...")

        # Merge chunks using PDAL to regular LAZ, then optionally convert the merge to COPC.
        # Chunk files are already LAZ from subsample_tile_chunk
        reader_type = "readers.las"  # Chunks are LAZ files
        merge_output_file = (
            temporary_laz_path(target_output_file, pipeline_dir)
            if output_copc
            else target_output_file
        )

        merge_writer = _subsample_las_writer_options(
            input_file,
            merge_output_file,
            dimension_reduction,
            xy_bounds=(minx, maxx, miny, maxy),
        )
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
            return (
                input_file.name,
                False,
                f"Merge failed with {encoding_summary(merge_writer)}: {result.stderr[:200]}",
                0,
            )

        if not merge_output_file.exists() or merge_output_file.stat().st_size == 0:
            return (input_file.name, False, "Output file empty", 0)

        if output_copc:
            print(f"      → Converting {merge_output_file.name} to COPC")
            if not convert_laz_output_to_copc(
                merge_output_file,
                target_output_file,
                source_metadata_file=input_file,
            ):
                return (input_file.name, False, "COPC conversion failed", 0)
            try:
                merge_output_file.unlink()
            except OSError:
                pass
            if not target_output_file.exists() or target_output_file.stat().st_size == 0:
                return (input_file.name, False, "COPC output file empty", 0)

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
    output_copc: bool = False,
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
        output_copc: If True, write final output as COPC LAZ.

    Returns:
        Tuple of (filename, success, message, point_count)
    """
    subsampling_method = normalize_subsampling_method(subsampling_method)
    try:
        is_copc = _is_copc_file(input_file)
        reader_type = "readers.copc" if is_copc else "readers.las"

        target_output_file = copc_output_path(output_file) if output_copc else laz_output_path(output_file)
        work_output_file = (
            temporary_laz_path(target_output_file, pipeline_dir)
            if output_copc
            else target_output_file
        )

        writer_opts = _subsample_las_writer_options(
            input_file,
            work_output_file,
            dimension_reduction,
        )

        if subsampling_method == SUBSAMPLING_METHOD_CENTER_OF_MASS:
            if is_copc and dimension_reduction:
                point_count = center_of_mass_subsample_copc(input_file, work_output_file, resolution)
            else:
                point_count = center_of_mass_subsample_las(
                    input_file,
                    work_output_file,
                    resolution,
                    dimension_reduction=dimension_reduction,
                )
            if output_copc:
                print(f"    → Converting {work_output_file.name} to COPC")
                if not convert_laz_output_to_copc(
                    work_output_file,
                    target_output_file,
                    source_metadata_file=input_file,
                ):
                    return (input_file.name, False, "COPC conversion failed", 0)
                try:
                    work_output_file.unlink()
                except OSError:
                    pass
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
                    return (
                        input_file.name,
                        False,
                        f"{result.stderr[:200]} ({encoding_summary(writer_opts)})",
                        0,
                    )
            else:
                return (
                    input_file.name,
                    False,
                    f"{result.stderr[:200]} ({encoding_summary(writer_opts)})",
                    0,
                )

        if not work_output_file.exists() or work_output_file.stat().st_size == 0:
            return (input_file.name, False, "Output file empty", 0)

        # Get point count
        point_count = 0
        try:
            info_result = subprocess.run(
                [pdal_cmd, "info", "--metadata", str(work_output_file)],
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

        if output_copc:
            print(f"    → Converting {work_output_file.name} to COPC")
            if not convert_laz_output_to_copc(
                work_output_file,
                target_output_file,
                source_metadata_file=input_file,
            ):
                return (input_file.name, False, "COPC conversion failed", 0)
            try:
                work_output_file.unlink()
            except OSError:
                pass
            if not target_output_file.exists() or target_output_file.stat().st_size == 0:
                return (input_file.name, False, "COPC output file empty", 0)

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
    output_copc: bool = False,
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
        output_copc: If True, write final outputs as COPC LAZ.

    Returns:
        List of created output file paths
    """
    # Create output directory
    subsampling_method = normalize_subsampling_method(subsampling_method)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create pipeline directory
    pipeline_dir = output_dir / "pipelines"
    pipeline_dir.mkdir(exist_ok=True)

    input_files = _subsample_input_files(input_dir)

    if not input_files:
        print(f"    No input files found in {input_dir}")
        return []

    # Convert resolution to cm for filename
    res_cm = int(resolution * 100)
    output_ext = ".copc.laz" if output_copc else ".laz"

    # Prepare tasks
    tasks = []
    manifest = _read_subsample_manifest(output_dir)
    for input_file in sorted(input_files):
        # Generate output filename
        stem = input_file.stem
        # Remove .copc suffix if present
        if stem.endswith('.copc'):
            stem = stem[:-5]

        # Extract original base filename by removing prefixes and resolution suffixes
        import re
        base_name = stem

        # Remove resolution suffix patterns from previous subsampling stages.
        base_name = re.sub(r'_subsampled[\d.]+m$', '', base_name)
        base_name = re.sub(r'_subsampled_\d+(?:\.\d+)?cm$', '', base_name)
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
                    output_name = f"{base_before_tile}_{tile_id}_subsampled_{res_cm}cm{output_ext}"
                else:
                    output_name = f"{tile_id}_subsampled_{res_cm}cm{output_ext}"
            else:
                output_name = f"{tile_id}_subsampled_{res_cm}cm{output_ext}"
        else:
            # Single file or no tile ID - use clean base name
            # Remove any remaining prefix patterns
            base_name = re.sub(r'^[^_]+_\d+m_', '', base_name)
            output_name = f"{base_name}_subsampled_{res_cm}cm{output_ext}"

        output_file = output_dir / output_name
        target_output_file = copc_output_path(output_file) if output_copc else laz_output_path(output_file)
        signature = _subsample_request_signature(
            input_file,
            resolution,
            dimension_reduction,
            subsampling_method,
            output_copc,
        )

        if _existing_subsample_is_reusable(target_output_file, manifest, signature):
            print(f"    ⊙ Skipping {input_file.name} (existing output matches request)")
            continue
        if target_output_file.exists() and target_output_file.stat().st_size > 0:
            print(f"    ↻ Rebuilding stale subsample for {input_file.name}")
            try:
                target_output_file.unlink()
            except OSError as exc:
                raise RuntimeError(f"Could not replace stale subsample {target_output_file}: {exc}") from exc

        tasks.append((
            input_file,
            output_file,
            resolution,
            pipeline_dir,
            num_threads,
            dimension_reduction,
            subsampling_method,
            output_copc,
            signature,
        ))

    if not tasks:
        print(f"    ✓ All files already subsampled")
        return subsample_output_files(output_dir, output_copc)

    print(f"    Files to process: {len(tasks)}")
    print(f"    Processing mode: Sequential (one file at a time)")
    print(f"    Chunk parallelism: {num_threads} chunks per file (parallel)")
    print(f"    Subsampling method: {subsampling_method}")
    print(f"    Output format: {'COPC LAZ' if output_copc else 'LAZ'}")
    print()

    # Process files sequentially, but chunks within each file in parallel
    successful = 0
    failed = 0
    total_points = 0

    for task in tasks:
        worker_task = task[:-1]
        signature = task[-1]
        filename, success, message, point_count = subsample_single_file(worker_task)
        if success:
            successful += 1
            total_points += point_count
            target_output_file = copc_output_path(task[1]) if output_copc else laz_output_path(task[1])
            _record_subsample_output(output_dir, target_output_file, signature, point_count)
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

    return subsample_output_files(output_dir, output_copc)


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
    output_copc_res1: bool = True,
    output_copc_res2: bool = False,
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
        num_threads: Number of parallel chunks per file (default: available CPU count)
        output_prefix: Optional prefix for output filenames
        output_base_dir: Base directory for output (default: parent of tiles_dir)
        dimension_reduction: If True, write only standard dimensions (minimal); if False, keep extra_dims (e.g. PredInstance).
        subsampling_method: "center-of-mass" or "nearest-to-centroid".
        output_copc_res1: Write first-resolution outputs as COPC LAZ.
        output_copc_res2: Write second-resolution outputs as COPC LAZ.

    Returns:
        Tuple of (subsampled_res1_dir, subsampled_res2_dir)
    """
    subsampling_method = normalize_subsampling_method(subsampling_method)
    # Auto-detect CPU count
    if num_cores is None:
        num_cores = get_cpu_count()

    if num_threads is None:
        num_threads = num_cores

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
    print(f"Subsampling method: {subsampling_method}")
    print(f"Resolution 1 output: {'COPC LAZ' if output_copc_res1 else 'LAZ'}")
    print(f"Resolution 2 output: {'COPC LAZ' if output_copc_res2 else 'LAZ'}")
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
        output_copc=output_copc_res1,
    )

    if not res1_files:
        raise ValueError(f"No files created in {subsampled_res1_dir}")

    print(f"\n  ✓ {res1_cm}cm subsampling complete: {len(res1_files)} files")
    print(f"  Output: {subsampled_res1_dir}")

    if abs(float(res1) - float(res2)) < 1e-12 and output_copc_res1 == output_copc_res2:
        print()
        print("=" * 60)
        print(f"Step 2: Reusing {res1_cm}cm output for matching resolution 2")
        print("=" * 60)
        if subsampled_res2_dir.exists():
            shutil.rmtree(subsampled_res2_dir)
        shutil.copytree(subsampled_res1_dir, subsampled_res2_dir)
        res2_files = subsample_output_files(subsampled_res2_dir, output_copc_res2)
        print(f"  ✓ Resolution 2 equals resolution 1; copied {len(res2_files)} file(s)")
        print(f"  Output: {subsampled_res2_dir}")
        print()
        print("=" * 60)
        print("Subsampling Pipeline Complete")
        print("=" * 60)
        print(f"  Resolution 1 ({res1_cm}cm): {len(res1_files)} files in {subsampled_res1_dir}")
        print(f"  Resolution 2 ({res2_cm}cm): {len(res2_files)} files in {subsampled_res2_dir}")
        return subsampled_res1_dir, subsampled_res2_dir

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
        output_copc=output_copc_res2,
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
        "--output-copc-res1",
        "--output_copc_res1",
        action=argparse.BooleanOptionalAction,
        default=TILE_PARAMS.get("output_copc_res1", True),
        help=(
            "Write resolution 1 outputs as COPC LAZ "
            f"(default: {TILE_PARAMS.get('output_copc_res1', True)})"
        ),
    )

    parser.add_argument(
        "--output-copc-res2",
        "--output_copc_res2",
        action=argparse.BooleanOptionalAction,
        default=TILE_PARAMS.get("output_copc_res2", False),
        help=(
            "Write resolution 2 outputs as COPC LAZ "
            f"(default: {TILE_PARAMS.get('output_copc_res2', False)})"
        ),
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
        help="Number of spatial chunks per file for parallel processing (default: auto-detected CPU count)"
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
            output_copc_res1=args.output_copc_res1,
            output_copc_res2=args.output_copc_res2,
        )
        print(f"\nSubsampled files ready:")
        print(f"  Resolution 1: {res1_dir}")
        print(f"  Resolution 2: {res2_dir}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
