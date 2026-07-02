#!/usr/bin/env python3
"""
Create prod-merged point-cloud products from Original-with-predictions files.

Prod-merged products use original uploaded points as geometry and select real
points with PDAL's nearest-to-centroid voxel subsampling when a lower product
resolution is requested.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from point_cloud_metadata import (
    copc_files,
    load_standardization_dims,
    point_cloud_dimension_names,
    point_cloud_source_key as source_key,
    raw_point_cloud_files as point_cloud_files,
)
from copc_staging import (
    staged_copc_matches_source,
    write_copc_stage_manifest_entry,
)

SUPPORTED_OUTPUT_FORMATS = {"laz", "copc.laz", "ply"}


def get_pdal_path() -> str:
    import shutil

    pdal_path = shutil.which("pdal")
    if pdal_path is None:
        raise RuntimeError("PDAL executable not found on PATH; create_merged_file requires pdal")
    return pdal_path


def resolution_label(resolution: float) -> str:
    centimeters = resolution * 100.0
    if math.isclose(centimeters, round(centimeters), rel_tol=0.0, abs_tol=1e-9):
        return f"{int(round(centimeters))}cm"
    return f"{centimeters:g}cm"


def parse_merged_resolutions(
    value: str,
    res1: float,
    res2: float,
) -> List[Tuple[str, float]]:
    """Parse create_merged_file resolution selector into unique label/resolution pairs."""
    aliases = {
        "res1": ("res1", res1),
        "resolution1": ("res1", res1),
        "resolution_1": ("res1", res1),
        "resolution-1": ("res1", res1),
        "res2": ("res2", res2),
        "resolution2": ("res2", res2),
        "resolution_2": ("res2", res2),
        "resolution-2": ("res2", res2),
    }
    parsed = []
    seen = set()

    for raw_token in (value or "res1,res2").split(","):
        token = raw_token.strip().lower()
        if not token:
            continue
        if token in aliases:
            base_label, resolution = aliases[token]
            label = resolution_label(resolution)
        elif token.endswith("cm"):
            number = token[:-2].strip()
            resolution = float(number) / 100.0
            label = resolution_label(resolution)
        else:
            resolution = float(token)
            label = resolution_label(resolution)

        if resolution <= 0:
            raise ValueError("merged resolutions must be positive")
        key = round(resolution, 9)
        if key in seen:
            continue
        seen.add(key)
        parsed.append((label, resolution))

    if not parsed:
        raise ValueError("No merged resolutions selected")
    return parsed


def _selector_tokens(value) -> Iterable[str]:
    """Yield selector tokens from CLI strings, Galaxy lists, or list-like strings."""
    if isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _selector_tokens(item)
        return
    text = str(value or "")
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    for raw_token in text.split(","):
        yield raw_token.strip().strip("'\"")


def parse_merged_output_formats(value: str) -> List[str]:
    """Parse prod-merged output format selector into unique normalized formats."""
    aliases = {
        "las": "laz",
        "laz": "laz",
        ".laz": "laz",
        "copc": "copc.laz",
        "copc_laz": "copc.laz",
        "copc-laz": "copc.laz",
        "copc.laz": "copc.laz",
        ".copc.laz": "copc.laz",
        "ply": "ply",
        ".ply": "ply",
    }
    parsed = []
    seen = set()
    for raw_token in _selector_tokens(value or "copc.laz"):
        token = raw_token.strip().lower()
        if not token:
            continue
        output_format = aliases.get(token)
        if output_format is None:
            raise ValueError(
                f"Unsupported merged output format '{raw_token}'. "
                "Use laz, copc.laz, or ply."
            )
        if output_format in seen:
            continue
        seen.add(output_format)
        parsed.append(output_format)
    if not parsed:
        raise ValueError("No merged output formats selected")
    return parsed


def prod_merged_output_path(output_dir: Path, label: str, output_format: str = "copc.laz") -> Path:
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_")
    normalized_format = parse_merged_output_formats(output_format)[0]
    return output_dir / f"prod_merged_{safe_label}.{normalized_format}"


def expensive_prod_merged_warning(label: str, resolution: float, output_format: str) -> Optional[str]:
    """Return a warning for product requests known to be scratch-disk heavy."""
    normalized_format = parse_merged_output_formats(output_format)[0]
    if normalized_format == "copc.laz" and resolution <= 0.010000001:
        return (
            f"{label} COPC prod-merged output can be very slow and scratch-disk heavy. "
            "For large datasets, prefer 10cm COPC for routine downloads or request "
            "1cm as LAZ unless a full-resolution COPC is explicitly needed."
        )
    return None


def _keep_failed_chunk_work_dir() -> bool:
    return os.environ.get("SMARTTILE_KEEP_FAILED_CHUNKS", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _cleanup_chunk_work_dir(work_dir: Path, success: bool) -> None:
    """Remove scratch chunks by default so failed large jobs do not fill disk."""
    if not work_dir.exists():
        return
    if success or not _keep_failed_chunk_work_dir():
        shutil.rmtree(work_dir, ignore_errors=True)


def _remove_existing_output(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        raise RuntimeError(f"Could not remove existing output before rewrite: {path}: {exc}") from exc


def _is_reusable_copc(path: Path) -> bool:
    """Return True when a staged COPC exists and has a readable LAS/COPC header."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        import laspy

        with laspy.open(str(path), laz_backend=laspy.LazBackend.LazrsParallel) as reader:
            return int(reader.header.point_count) > 0
    except Exception as exc:
        print(f"    Ignoring unreadable staged COPC {path.name}: {exc}")
        return False


def _copc_by_source(
    directory: Optional[Path],
    source_by_key: Optional[dict[str, Path]] = None,
) -> dict[str, Path]:
    if directory is None or not directory.exists():
        return {}
    by_source = {}
    for path in copc_files(directory):
        key = source_key(path)
        source_file = source_by_key.get(key) if source_by_key else None
        if _is_reusable_copc(path) and staged_copc_matches_source(path, source_file):
            by_source[key] = path
        elif source_file is not None:
            print(
                f"    Ignoring stale staged COPC {path.name}: "
                f"no fresh source manifest for {source_file.name}"
            )
    return by_source


def prepare_copc_inputs(
    original_with_predictions_dir: Path,
    output_dir: Path,
    staged_copc_dir: Optional[Path] = None,
) -> List[Path]:
    """Convert Original-with-predictions LAZ/LAS files to COPC using SmartTile's tiling converter."""
    if not original_with_predictions_dir.exists():
        raise FileNotFoundError(f"Original-with-predictions directory not found: {original_with_predictions_dir}")

    existing_copc = [path for path in copc_files(original_with_predictions_dir) if _is_reusable_copc(path)]
    raw_files = point_cloud_files(original_with_predictions_dir)
    if not raw_files and not existing_copc:
        raise FileNotFoundError(f"No LAZ/LAS files found in {original_with_predictions_dir}")

    copc_dir = output_dir / "original_with_predictions_copc"
    copc_dir.mkdir(parents=True, exist_ok=True)
    all_input_files = [*raw_files, *existing_copc]
    expected_keys = sorted({source_key(path) for path in all_input_files})
    raw_by_source = {source_key(path): path for path in raw_files}
    selected_by_source = {
        source_key(path): path
        for path in existing_copc
        if source_key(path) not in raw_by_source
        or staged_copc_matches_source(path, raw_by_source[source_key(path)])
    }
    selected_by_source.update({
        key: path
        for key, path in _copc_by_source(staged_copc_dir, raw_by_source).items()
        if key in expected_keys and key not in selected_by_source
    })
    selected_by_source.update({
        key: path
        for key, path in _copc_by_source(copc_dir, raw_by_source).items()
        if key in expected_keys and key not in selected_by_source
    })
    converter = None

    for input_file in raw_files:
        key = source_key(input_file)
        if key in selected_by_source:
            print(f"    Reusing existing COPC for {input_file.name}: {selected_by_source[key].name}")
            continue

        output_copc = copc_dir / f"{key}.copc.laz"
        print(f"    Converting {input_file.name} -> {output_copc.name}")
        if converter is None:
            from main_tile import _convert_laz_to_copc

            converter = _convert_laz_to_copc
        if not converter(input_file, output_copc, preserve_extra_dims=True):
            raise RuntimeError(f"LAZ/LAS -> COPC conversion failed: {input_file}")
        write_copc_stage_manifest_entry(input_file, output_copc)
        selected_by_source[key] = output_copc

    return [selected_by_source[key] for key in expected_keys]


def prod_merged_writer_stage(output_file: Path, output_format: str) -> dict:
    """Return the PDAL writer stage for a prod-merged output format."""
    normalized_format = parse_merged_output_formats(output_format)[0]
    if normalized_format == "laz":
        return {
            "type": "writers.las",
            "filename": str(output_file),
            "compression": True,
            "forward": "all",
            "extra_dims": "all",
        }
    if normalized_format == "copc.laz":
        return {
            "type": "writers.copc",
            "filename": str(output_file),
            "forward": "all",
            "extra_dims": "all",
        }
    if normalized_format == "ply":
        return {
            "type": "writers.ply",
            "filename": str(output_file),
            "storage_mode": "little endian",
        }
    raise ValueError(f"Unsupported merged output format '{output_format}'")


def _scale_offset_options(source_file: Path) -> dict:
    import laspy

    with laspy.open(str(source_file)) as reader:
        header = reader.header
        return {
            "scale_x": float(header.scales[0]),
            "scale_y": float(header.scales[1]),
            "scale_z": float(header.scales[2]),
            "offset_x": float(header.offsets[0]),
            "offset_y": float(header.offsets[1]),
            "offset_z": float(header.offsets[2]),
        }


def prod_merged_pipeline(
    input_files: List[Path],
    output_file: Path,
    resolution: float,
    output_format: str = "copc.laz",
) -> dict:
    """Build the PDAL pipeline for one prod-merged output."""
    readers = [
        {
            "type": "readers.copc" if path.name.lower().endswith(".copc.laz") else "readers.las",
            "filename": str(path),
        }
        for path in input_files
    ]
    pipeline = [*readers]
    if len(readers) > 1:
        pipeline.append({"type": "filters.merge"})
    pipeline.extend([
        {"type": "filters.voxelcentroidnearestneighbor", "cell": resolution},
        prod_merged_writer_stage(output_file, output_format),
    ])
    return {"pipeline": pipeline}


def _run_pdal_pipeline(pipeline: dict, pipeline_file: Path) -> subprocess.CompletedProcess:
    pipeline_file.parent.mkdir(parents=True, exist_ok=True)
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


def _pdal_error(result: subprocess.CompletedProcess, limit: int = 2000) -> str:
    details = "\n".join(part for part in (result.stderr, result.stdout) if part)
    return details[:limit]


def _point_cloud_point_count(path: Path) -> int:
    """Return the LAS/LAZ/COPC point count from the header."""
    import laspy

    from copc_metadata import laspy_laz_backend

    with laspy.open(str(path), laz_backend=laspy_laz_backend()) as reader:
        return int(reader.header.point_count)


def _copc_union_bounds(input_files: List[Path]) -> Tuple[float, float, float, float]:
    from main_subsample import get_file_bounds

    bounds = []
    for input_file in input_files:
        file_bounds = get_file_bounds(input_file)
        if file_bounds is None:
            raise RuntimeError(f"Could not determine bounds for {input_file}")
        bounds.append(file_bounds)

    return (
        min(bound[0] for bound in bounds),
        max(bound[1] for bound in bounds),
        min(bound[2] for bound in bounds),
        max(bound[3] for bound in bounds),
    )


def _prod_merged_chunk_bounds(
    input_files: List[Path],
    resolution: float,
    num_spatial_chunks: int,
) -> List[str]:
    from main_subsample import _aligned_edges, get_file_xy_scales

    minx, maxx, miny, maxy = _copc_union_bounds(input_files)
    if maxx - minx == 0 or maxy - miny == 0:
        return [f"([{minx},{maxx}],[{miny},{maxy}])"]

    raw_x_step = (maxx - minx) / max(1, num_spatial_chunks)
    x_step = max(resolution, math.ceil(raw_x_step / resolution) * resolution)
    scale_x = min(get_file_xy_scales(input_file)[0] for input_file in input_files)
    edges = _aligned_edges(minx, maxx, x_step, resolution)

    bounds = []
    for chunk_idx, (chunk_minx, chunk_maxx) in enumerate(edges):
        query_maxx = chunk_maxx if chunk_idx == len(edges) - 1 else chunk_maxx - scale_x * 0.5
        if query_maxx < chunk_minx:
            continue
        bounds.append(f"([{chunk_minx},{query_maxx}],[{miny},{maxy}])")
    return bounds


def _chunked_prod_merged_pipeline(
    input_files: List[Path],
    output_file: Path,
    bounds_str: str,
    resolution: float,
    scale_offset_options: dict,
) -> dict:
    readers = [
        {
            "type": "readers.copc" if path.name.lower().endswith(".copc.laz") else "readers.las",
            "filename": str(path),
            "bounds": bounds_str,
        }
        for path in input_files
    ]
    stages = [*readers]
    if len(readers) > 1:
        stages.append({"type": "filters.merge"})
    stages.extend([
        {"type": "filters.voxelcentroidnearestneighbor", "cell": resolution},
        {
            "type": "writers.las",
            "filename": str(output_file),
            "compression": True,
            "minor_version": 4,
            "forward": "all",
            "extra_dims": "all",
            **scale_offset_options,
        },
    ])
    return {"pipeline": stages}


def _merge_chunk_files_pipeline(
    chunk_files: List[Path],
    output_file: Path,
    output_format: str,
    scale_offset_options: Optional[dict] = None,
) -> dict:
    stages = [{"type": "readers.las", "filename": str(path)} for path in chunk_files]
    if len(stages) > 1:
        stages.append({"type": "filters.merge"})
    normalized_format = parse_merged_output_formats(output_format)[0]
    writer = prod_merged_writer_stage(output_file, normalized_format)
    if scale_offset_options and normalized_format in {"laz", "copc.laz"}:
        writer.update(scale_offset_options)
    stages.append(writer)
    return {"pipeline": stages}


def _validate_expected_dims(
    files: List[Path],
    expected_dims: Optional[set[str]],
    context: str,
) -> None:
    """Fail when LAS/COPC files do not expose expected standardized dimensions."""
    if not expected_dims:
        return

    available = set()
    unreadable = []
    for path in files:
        try:
            available.update(point_cloud_dimension_names(path))
        except Exception as exc:
            unreadable.append(f"{path.name}: {exc}")

    if unreadable:
        raise RuntimeError(
            f"Could not validate standardized dimensions for {context}: "
            + "; ".join(unreadable)
        )

    missing = sorted(expected_dims - available)
    if missing:
        raise RuntimeError(
            f"{context} is missing {len(missing)} standardized dimension(s): {missing}"
        )

    print(
        f"  Standardization JSON: {context} contains "
        f"{len(expected_dims)} expected dimension(s)",
        flush=True,
    )


def _validate_preserved_product_dims(
    source_files: List[Path],
    output_file: Path,
) -> Tuple[bool, str]:
    """Validate that LAS/COPC product writing did not drop point dimensions."""
    expected = set()
    unreadable = []
    for path in source_files:
        try:
            expected.update(point_cloud_dimension_names(path))
        except Exception as exc:
            unreadable.append(f"{path.name}: {exc}")

    if unreadable:
        return (
            False,
            "could not inspect source product dimensions: " + "; ".join(unreadable),
        )

    try:
        output_dims = point_cloud_dimension_names(output_file)
    except Exception as exc:
        return (False, f"could not inspect output product dimensions: {exc}")

    missing = sorted(expected - output_dims)
    if missing:
        return (
            False,
            f"prod-merged output dropped {len(missing)} point dimension(s): {missing}",
        )
    return (True, "product dimensions preserved")


def _preserve_and_validate_las_metadata(source_metadata_file: Path, output_file: Path) -> Tuple[bool, str]:
    """Ensure a LAS/LAZ/COPC product carries source CRS/GeoTIFF projection metadata."""
    from copc_metadata import (
        append_source_geotiff_projection_evlrs,
        copc_preserves_source_crs,
    )

    preserved_geotiff, message = append_source_geotiff_projection_evlrs(
        source_metadata_file,
        output_file,
    )
    if not preserved_geotiff:
        return (False, f"GeoTIFF projection preservation failed: {message}")

    valid_crs, message = copc_preserves_source_crs(source_metadata_file, output_file)
    if not valid_crs:
        return (False, f"CRS validation failed: {message}")
    return (True, "LAS metadata preserved")


def _preserve_and_validate_copc_metadata(source_metadata_file: Path, output_file: Path) -> Tuple[bool, str]:
    """Backward-compatible alias for older tests/callers."""
    return _preserve_and_validate_las_metadata(source_metadata_file, output_file)


def _untwine_chunk_files_to_copc(
    chunk_files: List[Path],
    output_file: Path,
    source_metadata_file: Path,
    temp_dir: Optional[Path] = None,
) -> Tuple[bool, str]:
    """Write final COPC directly from chunk files, avoiding a giant merged LAZ."""
    untwine_cmd = shutil.which("untwine")
    if not untwine_cmd:
        return (False, "untwine not available")

    try:
        if output_file.exists():
            output_file.unlink()

        from copc_metadata import srs_assignment_from_file

        input_args = []
        for chunk_file in chunk_files:
            input_args.extend(["-i", str(chunk_file)])
        srs_arg = srs_assignment_from_file(source_metadata_file)
        srs_args = ["--a_srs", srs_arg] if srs_arg else []
        temp_args = []
        if temp_dir is not None:
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_args = ["--temp_dir", str(temp_dir)]

        result = subprocess.run(
            [untwine_cmd] + input_args + ["-o", str(output_file)] + srs_args + temp_args,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return (False, f"untwine failed: {_pdal_error(result)}")
        if not output_file.exists() or output_file.stat().st_size == 0:
            return (False, "untwine produced no output")

        expected_points = sum(_point_cloud_point_count(chunk_file) for chunk_file in chunk_files)
        actual_points = _point_cloud_point_count(output_file)
        if actual_points != expected_points:
            try:
                output_file.unlink()
            except OSError:
                pass
            return (
                False,
                "untwine point-count mismatch: "
                f"expected {expected_points}, got {actual_points}",
            )

        valid_metadata, message = _preserve_and_validate_las_metadata(
            source_metadata_file,
            output_file,
        )
        if not valid_metadata:
            return (False, message)
        valid_dims, message = _validate_preserved_product_dims(chunk_files, output_file)
        if not valid_dims:
            try:
                output_file.unlink()
            except OSError:
                pass
            return (False, message)
        return (True, "untwine")
    except Exception as exc:
        return (False, f"untwine error: {exc}")


def _merge_prod_chunks(
    chunk_files: List[Path],
    output_file: Path,
    output_format: str,
    work_dir: Path,
    source_metadata_file: Path,
    scale_offset_options: dict,
) -> None:
    normalized_format = parse_merged_output_formats(output_format)[0]
    if normalized_format == "copc.laz":
        success, message = _untwine_chunk_files_to_copc(
            chunk_files,
            output_file,
            source_metadata_file,
            temp_dir=work_dir / "_untwine_tmp",
        )
        if success:
            print("    Direct untwine chunk->COPC complete")
            return
        print(f"    Direct untwine chunk->COPC failed; falling back to PDAL merge: {message}")

    pipeline = _merge_chunk_files_pipeline(
        chunk_files,
        output_file,
        output_format,
        scale_offset_options,
    )
    result = _run_pdal_pipeline(pipeline, work_dir / f"_{output_file.stem}_merge_chunks.json")
    if result.returncode == 0 and output_file.exists() and output_file.stat().st_size > 0:
        if normalized_format in {"laz", "copc.laz"}:
            valid_dims, message = _validate_preserved_product_dims(chunk_files, output_file)
            if not valid_dims:
                raise RuntimeError(message)
            valid_metadata, message = _preserve_and_validate_las_metadata(
                source_metadata_file,
                output_file,
            )
            if not valid_metadata:
                raise RuntimeError(message)
        return

    if normalized_format != "copc.laz":
        raise RuntimeError(f"PDAL prod-merged chunk merge failed: {_pdal_error(result)}")

    temp_laz = work_dir / f"_{output_file.stem}_for_copc.laz"
    temp_pipeline = _merge_chunk_files_pipeline(chunk_files, temp_laz, "laz", scale_offset_options)
    temp_result = _run_pdal_pipeline(temp_pipeline, work_dir / f"_{output_file.stem}_merge_chunks_laz.json")
    if temp_result.returncode != 0 or not temp_laz.exists() or temp_laz.stat().st_size == 0:
        raise RuntimeError(
            "PDAL prod-merged chunk merge failed and LAZ fallback failed: "
            f"{_pdal_error(result)}\n{_pdal_error(temp_result)}"
        )

    from subsample_outputs import convert_laz_output_to_copc

    if not convert_laz_output_to_copc(
        temp_laz,
        output_file,
        source_metadata_file=source_metadata_file,
        preserve_extra_dims=True,
    ):
        raise RuntimeError(f"Prod-merged COPC conversion failed: {output_file}")
    try:
        temp_laz.unlink()
    except OSError:
        pass


def create_chunked_prod_merged_file(
    copc_input_files: List[Path],
    output_file: Path,
    resolution: float,
    output_format: str,
    num_spatial_chunks: int,
) -> Path:
    """Create one prod-merged product using bounded COPC reads per spatial chunk."""
    chunk_bounds = _prod_merged_chunk_bounds(copc_input_files, resolution, num_spatial_chunks)
    if not chunk_bounds:
        raise RuntimeError("No spatial chunks available for prod-merged output")

    _remove_existing_output(output_file)
    work_dir = output_file.parent / f"_{output_file.stem}_chunks"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    scale_offset_options = _scale_offset_options(copc_input_files[0])

    chunk_files: List[Path] = []
    success = False
    try:
        print(f"    Spatial chunks: {len(chunk_bounds)}")
        for chunk_idx, bounds_str in enumerate(chunk_bounds):
            chunk_file = work_dir / f"{output_file.stem}_chunk{chunk_idx:04d}.laz"
            pipeline = _chunked_prod_merged_pipeline(
                copc_input_files,
                chunk_file,
                bounds_str,
                resolution,
                scale_offset_options,
            )
            result = _run_pdal_pipeline(pipeline, work_dir / f"_chunk{chunk_idx:04d}.json")
            if result.returncode != 0:
                raise RuntimeError(f"PDAL prod-merged chunk {chunk_idx + 1} failed: {_pdal_error(result)}")
            if not chunk_file.exists() or chunk_file.stat().st_size == 0:
                print(f"    Chunk {chunk_idx + 1}/{len(chunk_bounds)}: empty")
                continue
            chunk_files.append(chunk_file)
            print(f"    Chunk {chunk_idx + 1}/{len(chunk_bounds)} complete: {chunk_file.name}")

        if not chunk_files:
            raise RuntimeError("No prod-merged chunks were created")

        _merge_prod_chunks(
            chunk_files,
            output_file,
            output_format,
            work_dir,
            source_metadata_file=copc_input_files[0],
            scale_offset_options=scale_offset_options,
        )
        success = output_file.exists() and output_file.stat().st_size > 0
    finally:
        _cleanup_chunk_work_dir(work_dir, success)

    if not output_file.exists() or output_file.stat().st_size == 0:
        raise RuntimeError(f"Prod-merged output was not created: {output_file}")
    return output_file


def create_chunked_prod_merged_files_for_resolution(
    copc_input_files: List[Path],
    outputs: List[Tuple[Path, str]],
    resolution: float,
    num_spatial_chunks: int,
) -> List[Path]:
    """Create multiple output formats from one canonical set of chunk files."""
    if not copc_input_files:
        raise ValueError("No COPC input files provided")
    if not outputs:
        return []

    chunk_bounds = _prod_merged_chunk_bounds(copc_input_files, resolution, num_spatial_chunks)
    if not chunk_bounds:
        raise RuntimeError("No spatial chunks available for prod-merged output")

    for output_file, _output_format in outputs:
        _remove_existing_output(output_file)

    first_output = outputs[0][0]
    work_dir = first_output.parent / f"_{first_output.stem}_shared_chunks"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    scale_offset_options = _scale_offset_options(copc_input_files[0])

    chunk_files: List[Path] = []
    created: List[Path] = []
    success = False
    try:
        print(f"    Spatial chunks: {len(chunk_bounds)}")
        for chunk_idx, bounds_str in enumerate(chunk_bounds):
            chunk_file = work_dir / f"prod_merged_chunk{chunk_idx:04d}.laz"
            pipeline = _chunked_prod_merged_pipeline(
                copc_input_files,
                chunk_file,
                bounds_str,
                resolution,
                scale_offset_options,
            )
            result = _run_pdal_pipeline(pipeline, work_dir / f"_chunk{chunk_idx:04d}.json")
            if result.returncode != 0:
                raise RuntimeError(f"PDAL prod-merged chunk {chunk_idx + 1} failed: {_pdal_error(result)}")
            if not chunk_file.exists() or chunk_file.stat().st_size == 0:
                print(f"    Chunk {chunk_idx + 1}/{len(chunk_bounds)}: empty")
                continue
            chunk_files.append(chunk_file)
            print(f"    Chunk {chunk_idx + 1}/{len(chunk_bounds)} complete: {chunk_file.name}")

        if not chunk_files:
            raise RuntimeError("No prod-merged chunks were created")

        for output_file, output_format in outputs:
            _merge_prod_chunks(
                chunk_files,
                output_file,
                output_format,
                work_dir,
                source_metadata_file=copc_input_files[0],
                scale_offset_options=scale_offset_options,
            )
            if not output_file.exists() or output_file.stat().st_size == 0:
                raise RuntimeError(f"Prod-merged output was not created: {output_file}")
            created.append(output_file)
        success = len(created) == len(outputs) and all(
            path.exists() and path.stat().st_size > 0 for path in created
        )
    finally:
        _cleanup_chunk_work_dir(work_dir, success)

    return created


def create_prod_merged_file(
    copc_input_files: List[Path],
    output_file: Path,
    resolution: float,
    output_format: str = "copc.laz",
    num_spatial_chunks: Optional[int] = None,
) -> Path:
    """Create one prod-merged product at the requested resolution and format."""
    if not copc_input_files:
        raise ValueError("No COPC input files provided")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    _remove_existing_output(output_file)
    if num_spatial_chunks and num_spatial_chunks > 1:
        return create_chunked_prod_merged_file(
            copc_input_files,
            output_file,
            resolution,
            output_format,
            num_spatial_chunks,
        )

    pipeline = prod_merged_pipeline(copc_input_files, output_file, resolution, output_format)
    pipeline_file = output_file.parent / f"_{output_file.stem}_pipeline.json"
    result = _run_pdal_pipeline(pipeline, pipeline_file)

    if result.returncode != 0:
        raise RuntimeError(f"PDAL prod-merged pipeline failed: {_pdal_error(result, 500)}")
    if not output_file.exists() or output_file.stat().st_size == 0:
        raise RuntimeError(f"Prod-merged output was not created: {output_file}")
    if parse_merged_output_formats(output_format)[0] in {"laz", "copc.laz"}:
        valid_dims, message = _validate_preserved_product_dims(copc_input_files, output_file)
        if not valid_dims:
            raise RuntimeError(message)
        valid_metadata, message = _preserve_and_validate_las_metadata(
            copc_input_files[0],
            output_file,
        )
        if not valid_metadata:
            raise RuntimeError(message)
    return output_file


def create_prod_merged_files(
    original_with_predictions_dir: Path,
    output_dir: Path,
    resolution_selector: str,
    output_format_selector: str,
    res1: float,
    res2: float,
    num_spatial_chunks: Optional[int] = None,
    staged_copc_dir: Optional[Path] = None,
    standardization_json: Optional[Path] = None,
) -> List[Path]:
    """Create all selected prod-merged products."""
    outputs = []
    expected_dims = None
    if standardization_json:
        expected_dims = load_standardization_dims(Path(standardization_json))
        print(
            f"  Standardization JSON: validating expected dimensions from {standardization_json}",
            flush=True,
        )

    print("  Preparing COPC inputs with SmartTile untwine/PDAL fallback conversion")
    copc_input_files = prepare_copc_inputs(
        original_with_predictions_dir,
        output_dir,
        staged_copc_dir=staged_copc_dir,
    )
    _validate_expected_dims(copc_input_files, expected_dims, "staged Original-with-predictions COPCs")
    output_formats = parse_merged_output_formats(output_format_selector)
    for label, resolution in parse_merged_resolutions(resolution_selector, res1, res2):
        selected_outputs = [
            (prod_merged_output_path(output_dir, label, output_format), output_format)
            for output_format in output_formats
        ]
        for output_file, output_format in selected_outputs:
            warning = expensive_prod_merged_warning(label, resolution, output_format)
            if warning:
                print(f"  Warning: {warning}")
            print(
                f"  Creating {output_file.name} at {resolution:g}m "
                f"with nearest-to-centroid"
            )

        if num_spatial_chunks and num_spatial_chunks > 1:
            created_outputs = create_chunked_prod_merged_files_for_resolution(
                copc_input_files,
                selected_outputs,
                resolution,
                num_spatial_chunks,
            )
        else:
            created_outputs = [
                create_prod_merged_file(
                    copc_input_files,
                    output_file,
                    resolution,
                    output_format,
                    num_spatial_chunks=num_spatial_chunks,
                )
                for output_file, output_format in selected_outputs
            ]

        for created, output_format in zip(created_outputs, output_formats):
            if parse_merged_output_formats(output_format)[0] in {"laz", "copc.laz"}:
                _validate_expected_dims([created], expected_dims, created.name)
            outputs.append(created)
    return outputs
