#!/usr/bin/env python3
"""PDAL-backed spatial chunk worker for SmartTile subsampling."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

from subsample_com import center_of_mass_subsample_las
from subsample_methods import (
    SUBSAMPLING_METHOD_CENTER_OF_MASS,
    is_copc_file,
    normalize_subsampling_method,
    voxel_subsampling_filter,
)
from subsample_encoding import (
    encoding_summary,
    safe_las_writer_encoding_options_for_pdal_bounds,
)


def get_pdal_path() -> str:
    """Return the PDAL executable path."""
    pdal_path = shutil.which("pdal")
    return pdal_path if pdal_path else "pdal"


def crop_input_to_laz(
    input_file: Path,
    crop_file: Path,
    bounds_str: str,
    dimension_reduction: bool,
) -> bool:
    """Crop a chunk with PDAL before SmartTile center-of-mass aggregation."""
    is_copc = is_copc_file(input_file)
    reader_type = "readers.copc" if is_copc else "readers.las"
    writer_opts = {
        "type": "writers.las",
        "filename": str(crop_file),
        "compression": True,
        "forward": "all",
    }
    writer_opts.update(safe_las_writer_encoding_options_for_pdal_bounds(input_file, bounds_str))
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

    pipeline_file = crop_file.parent / f"_{crop_file.stem}_crop.json"
    with open(pipeline_file, "w") as f:
        json.dump({"pipeline": stages}, f, indent=2)
    try:
        result = subprocess.run(
            [get_pdal_path(), "pipeline", str(pipeline_file)],
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        if pipeline_file.exists():
            pipeline_file.unlink()

    success = result.returncode == 0 and crop_file.exists() and crop_file.stat().st_size > 0
    if not success:
        print(
            f"      Crop writer failed with {encoding_summary(writer_opts)}: "
            f"{result.stderr[:200]}"
        )
    return success


def _chunk_writer_options(
    input_file: Path,
    chunk_file: Path,
    bounds_str: str,
    dimension_reduction: bool,
) -> dict:
    writer_opts = {
        "type": "writers.las",
        "filename": str(chunk_file),
        "compression": True,
        "forward": "all",
    }
    writer_opts.update(safe_las_writer_encoding_options_for_pdal_bounds(input_file, bounds_str))
    if dimension_reduction:
        writer_opts["minor_version"] = 2
        writer_opts["dataformat_id"] = 0
    else:
        writer_opts["minor_version"] = 4
        writer_opts["extra_dims"] = "all"
    return writer_opts


def _voxel_chunk_pipeline(
    input_file: Path,
    bounds_str: str,
    resolution: float,
    subsampling_method: str,
    writer_opts: dict,
    use_copc_bounds: bool,
) -> dict:
    reader_type = "readers.copc" if use_copc_bounds else "readers.las"
    if use_copc_bounds:
        stages = [
            {"type": reader_type, "filename": str(input_file), "bounds": bounds_str},
            voxel_subsampling_filter(resolution, subsampling_method),
            writer_opts,
        ]
    else:
        stages = [
            {"type": reader_type, "filename": str(input_file)},
            {"type": "filters.crop", "bounds": bounds_str},
            voxel_subsampling_filter(resolution, subsampling_method),
            writer_opts,
        ]
    return {"pipeline": stages}


def _run_pdal_pipeline(pipeline: dict, pipeline_file: Path) -> subprocess.CompletedProcess:
    with open(pipeline_file, "w") as f:
        json.dump(pipeline, f, indent=2)
    try:
        return subprocess.run(
            [get_pdal_path(), "pipeline", str(pipeline_file)],
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        if pipeline_file.exists():
            pipeline_file.unlink()


def _point_count(path: Path) -> int:
    try:
        info_result = subprocess.run(
            [get_pdal_path(), "info", "--metadata", str(path)],
            capture_output=True,
            text=True,
            check=True,
        )
        match = re.search(r'"count":\s*(\d+)', info_result.stdout)
        return int(match.group(1)) if match else 0
    except Exception:
        return 0


def subsample_tile_chunk(
    args: Tuple[Path, str, float, Path, int, int, bool, str]
) -> Tuple[Optional[Path], int]:
    """Subsample one spatial chunk using COPC bounds where possible."""
    input_file, bounds_str, resolution, output_dir, chunk_idx, total_chunks, dimension_reduction, method = args
    subsampling_method = normalize_subsampling_method(method)

    try:
        chunk_file = output_dir / f"{input_file.stem}_chunk{chunk_idx}.laz"

        if subsampling_method == SUBSAMPLING_METHOD_CENTER_OF_MASS:
            crop_file = output_dir / f"{input_file.stem}_chunk{chunk_idx}_crop.laz"
            try:
                if not crop_input_to_laz(input_file, crop_file, bounds_str, dimension_reduction):
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

        pdal_cmd = get_pdal_path()
        writer_opts = _chunk_writer_options(input_file, chunk_file, bounds_str, dimension_reduction)
        input_is_copc = is_copc_file(input_file)
        pipeline = _voxel_chunk_pipeline(
            input_file,
            bounds_str,
            resolution,
            subsampling_method,
            writer_opts,
            use_copc_bounds=input_is_copc,
        )
        result = _run_pdal_pipeline(pipeline, output_dir / f"_pipeline_chunk{chunk_idx}.json")

        if result.returncode != 0 and input_is_copc and ("copc" in result.stderr.lower() or "vlr" in result.stderr.lower()):
            print(f"      ⚠ Chunk {chunk_idx}/{total_chunks}: COPC reader failed, falling back to readers.las")
            pipeline = _voxel_chunk_pipeline(
                input_file,
                bounds_str,
                resolution,
                subsampling_method,
                writer_opts,
                use_copc_bounds=False,
            )
            result = _run_pdal_pipeline(pipeline, output_dir / f"_pipeline_chunk{chunk_idx}_fallback.json")
            if result.returncode != 0:
                print(
                    f"      ⚠ Chunk {chunk_idx}/{total_chunks} fallback error "
                    f"with {encoding_summary(writer_opts)}: {result.stderr[:100]}"
                )
                return (None, 0)
        elif result.returncode != 0:
            print(
                f"      ⚠ Chunk {chunk_idx}/{total_chunks} error "
                f"with {encoding_summary(writer_opts)}: {result.stderr[:100]}"
            )
            return (None, 0)

        if not chunk_file.exists() or chunk_file.stat().st_size == 0:
            return (None, 0)

        point_count = _point_count(chunk_file)
        print(f"      ✓ Chunk {chunk_idx}/{total_chunks}: {point_count:,} points")
        return (chunk_file, point_count)
    except Exception as e:
        print(f"      ✗ Chunk {chunk_idx}/{total_chunks} failed: {e}")
        return (None, 0)
