#!/usr/bin/env python3
"""COPC finalization and conversion helpers for SmartTile tiling."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

from copc_metadata import (
    append_source_geotiff_projection_evlrs,
    copc_preserves_source_crs,
    first_crs_source,
    first_srs_assignment,
    srs_assignment_from_file,
)

UNTWINE_STRIP_EXTRA_DIMS_ARG = "Classification"


def get_pdal_path() -> str:
    """Return the PDAL executable path."""
    pdal_path = shutil.which("pdal")
    return pdal_path if pdal_path else "pdal"


def _run_untwine(
    inputs: List[Path],
    output_copc: Path,
    srs_arg: Optional[str],
    strip_extra_dims: bool = False,
) -> Tuple[bool, str]:
    """Run untwine for one or more inputs."""
    untwine_cmd = shutil.which("untwine")
    if not untwine_cmd:
        return (False, "untwine not available")

    input_args = []
    for path in inputs:
        input_args.extend(["-i", str(path)])
    # Untwine treats --dims as an extra-dimension keep-list while X/Y/Z and the
    # standard LAS fields remain loaded. An empty keep-list is rejected by
    # Untwine 1.5.1, so Classification is used as a stable standard LAS
    # dimension to activate Untwine's dimension-limiting path. Callers inspect
    # the output and fall back to PDAL when extra bytes remain.
    dims_args = ["--dims", UNTWINE_STRIP_EXTRA_DIMS_ARG] if strip_extra_dims else []
    srs_args = ["--a_srs", srs_arg] if srs_arg else []
    try:
        result = subprocess.run(
            [untwine_cmd] + input_args + ["-o", str(output_copc)] + dims_args + srs_args,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:
        return (False, f"untwine error: {exc}")
    if result.returncode != 0:
        return (False, f"untwine failed: {result.stderr[:200]}")
    if not output_copc.exists() or output_copc.stat().st_size == 0:
        return (False, "untwine produced no output")
    return (True, "untwine")


def _has_extra_dimensions(path: Path) -> bool:
    """Return True when a LAS/LAZ/COPC file exposes extra byte dimensions."""
    import laspy

    from copc_metadata import laspy_laz_backend

    with laspy.open(str(path), laz_backend=laspy_laz_backend()) as reader:
        return bool(list(reader.header.point_format.extra_dimensions))


def _output_has_no_extra_dimensions(path: Path) -> bool:
    """Validate that dimension-reduced output does not retain extra bytes."""
    try:
        return not _has_extra_dimensions(path)
    except Exception:
        return False


def finalize_tile_to_copc(args: Tuple) -> Tuple[str, bool, str]:
    """Merge a tile's LAS part files into one COPC tile."""
    label, tiles_dir, log_dir, tile_bounds = args
    final_tile = tiles_dir / f"{label}.copc.laz"

    if final_tile.exists() and final_tile.stat().st_size > 0:
        return (label, True, "Already exists")

    tile_dir = tiles_dir / label
    if not tile_dir.exists():
        return (label, True, "No data in bounds")

    parts = sorted(tile_dir.glob("part_*.las"))
    if not parts:
        if not any(tile_dir.iterdir()):
            tile_dir.rmdir()
        return (label, True, "No data in bounds")

    try:
        success, message = finalize_tile_to_copc_untwine(
            parts,
            final_tile,
            log_dir,
            label,
            tile_bounds=tile_bounds,
        )
        if not success:
            return (label, False, message)

        crs_source = first_crs_source(parts)
        if crs_source is not None:
            preserved_geotiff, geotiff_message = append_source_geotiff_projection_evlrs(
                crs_source,
                final_tile,
            )
            if not preserved_geotiff:
                return (label, False, geotiff_message)
            valid_crs, crs_message = copc_preserves_source_crs(crs_source, final_tile)
            if not valid_crs and message.startswith("untwine"):
                try:
                    final_tile.unlink(missing_ok=True)
                except OSError:
                    pass
                success, message = finalize_tile_to_copc_pdal(
                    parts,
                    final_tile,
                    log_dir,
                    label,
                    tile_bounds,
                )
                if not success:
                    return (label, False, f"{message}; after untwine CRS validation failed: {crs_message}")
                preserved_geotiff, geotiff_message = append_source_geotiff_projection_evlrs(
                    crs_source,
                    final_tile,
                )
                if not preserved_geotiff:
                    return (label, False, geotiff_message)
                valid_crs, crs_message = copc_preserves_source_crs(crs_source, final_tile)
            if not valid_crs:
                return (label, False, f"COPC CRS validation failed: {crs_message}")

        for part in parts:
            if part.exists():
                part.unlink()
        if tile_dir.exists() and not any(tile_dir.iterdir()):
            tile_dir.rmdir()

        return (label, True, f"{len(parts)} parts merged ({message})")
    except Exception as exc:
        return (label, False, str(exc))


def finalize_tile_to_copc_pdal(
    parts: List[Path],
    final_tile: Path,
    log_dir: Path,
    label: str,
    tile_bounds: Optional[Tuple[float, float, float, float]] = None,
    preserve_extra_dims: bool = False,
) -> Tuple[bool, str]:
    """Finalize a tile with PDAL writers.copc."""
    writer_opts = {
        "type": "writers.copc",
        "filename": str(final_tile),
        "forward": "all",
    }
    if preserve_extra_dims:
        writer_opts["extra_dims"] = "all"
    if tile_bounds is not None:
        bxmin, bymin, bxmax, bymax = tile_bounds
        writer_opts["offset_x"] = (bxmin + bxmax) / 2.0
        writer_opts["offset_y"] = (bymin + bymax) / 2.0

    if len(parts) == 1:
        pipeline = {
            "pipeline": [
                {"type": "readers.las", "filename": str(parts[0])},
                writer_opts,
            ]
        }
    else:
        readers = [{"type": "readers.las", "filename": str(path)} for path in parts]
        pipeline = {"pipeline": readers + [{"type": "filters.merge"}, writer_opts]}

    pipeline_file = log_dir / f"{label}_pipeline.json"
    with open(pipeline_file, "w") as handle:
        json.dump(pipeline, handle)

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

    if result.returncode != 0:
        return (False, f"COPC conversion failed: {result.stderr[:200]}")
    return (True, "OK")


def finalize_tile_to_copc_untwine(
    parts: List[Path],
    final_tile: Path,
    log_dir: Path,
    label: str,
    tile_bounds: Optional[Tuple[float, float, float, float]] = None,
    preserve_extra_dims: bool = False,
) -> Tuple[bool, str]:
    """Finalize a tile using untwine, falling back to PDAL when unavailable."""
    if not preserve_extra_dims:
        untwine_cmd = shutil.which("untwine")
        if untwine_cmd:
            success, message = _run_untwine(
                parts,
                final_tile,
                first_srs_assignment(parts),
                strip_extra_dims=True,
            )
            if success and _output_has_no_extra_dimensions(final_tile):
                return (True, "untwine-dims-classification")
            if success:
                message = "untwine --dims Classification retained extra dimensions"
            try:
                final_tile.unlink(missing_ok=True)
            except OSError:
                pass
        success, message = finalize_tile_to_copc_pdal(
            parts,
            final_tile,
            log_dir,
            label,
            tile_bounds=tile_bounds,
            preserve_extra_dims=False,
        )
        if success:
            return (True, "pdal-stripped")
        return (False, message)

    untwine_cmd = shutil.which("untwine")
    if not untwine_cmd:
        success, message = finalize_tile_to_copc_pdal(
            parts,
            final_tile,
            log_dir,
            label,
            tile_bounds=tile_bounds,
            preserve_extra_dims=True,
        )
        if success:
            return (True, "pdal")
        return (False, message)

    try:
        input_args = []
        for part in parts:
            input_args.extend(["-i", str(part)])
        srs_arg = first_srs_assignment(parts)
        srs_args = ["--a_srs", srs_arg] if srs_arg else []

        result = subprocess.run(
            [untwine_cmd] + input_args + ["-o", str(final_tile)] + srs_args,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return (False, f"untwine failed: {result.stderr[:200]}")
        if not final_tile.exists() or final_tile.stat().st_size == 0:
            return (False, "untwine produced no output")
        return (True, "untwine")
    except Exception as exc:
        return (False, f"untwine error: {exc}")


def convert_laz_to_copc(
    input_laz: Path,
    output_copc: Path,
    preserve_extra_dims: bool = False,
) -> bool:
    """Convert a single LAZ/LAS file to COPC with CRS/GeoTIFF validation.

    The default SmartTile COPC conversion first tries Untwine's
    ``--dims Classification`` path while forwarding header metadata/CRS, then
    validates that no extra byte dimensions remain. If Untwine retains extras,
    SmartTile falls back to PDAL's standard-dimension writer path. Prod-merged creation passes
    preserve_extra_dims=True because those products must retain enriched
    prediction and source attributes.
    """
    if not preserve_extra_dims:
        untwine_cmd = shutil.which("untwine")
        if untwine_cmd:
            success, _ = _run_untwine(
                [input_laz],
                output_copc,
                srs_assignment_from_file(input_laz),
                strip_extra_dims=True,
            )
            if success:
                if _output_has_no_extra_dimensions(output_copc):
                    append_source_geotiff_projection_evlrs(input_laz, output_copc)
                    valid_crs, message = copc_preserves_source_crs(input_laz, output_copc)
                    if valid_crs:
                        return True
                    print(f"  Warning: untwine COPC CRS validation failed for {output_copc.name}: {message}; retrying with PDAL")
                else:
                    print(f"  Warning: untwine dimension limiting kept extra dimensions for {output_copc.name}; retrying with PDAL")
                try:
                    output_copc.unlink(missing_ok=True)
                except OSError:
                    pass
        if not convert_laz_to_copc_pdal(input_laz, output_copc, preserve_extra_dims=False):
            return False
        preserved_geotiff, message = append_source_geotiff_projection_evlrs(input_laz, output_copc)
        if not preserved_geotiff:
            print(f"  Warning: COPC GeoTIFF projection preservation failed for {output_copc.name}: {message}")
            return False
        valid_crs, message = copc_preserves_source_crs(input_laz, output_copc)
        if not valid_crs:
            print(f"  Warning: COPC CRS validation failed for {output_copc.name}: {message}")
        return valid_crs

    untwine_cmd = shutil.which("untwine")
    if untwine_cmd:
        success, _ = _run_untwine(
            [input_laz],
            output_copc,
            srs_assignment_from_file(input_laz),
            strip_extra_dims=False,
        )
        if success:
            append_source_geotiff_projection_evlrs(input_laz, output_copc)
            valid_crs, _ = copc_preserves_source_crs(input_laz, output_copc)
            if valid_crs:
                return True
            try:
                output_copc.unlink(missing_ok=True)
            except OSError:
                pass

    if not convert_laz_to_copc_pdal(input_laz, output_copc, preserve_extra_dims=True):
        return False
    preserved_geotiff, message = append_source_geotiff_projection_evlrs(input_laz, output_copc)
    if not preserved_geotiff:
        print(f"  Warning: COPC GeoTIFF projection preservation failed for {output_copc.name}: {message}")
        return False
    valid_crs, message = copc_preserves_source_crs(input_laz, output_copc)
    if not valid_crs:
        print(f"  Warning: COPC CRS validation failed for {output_copc.name}: {message}")
    return valid_crs


def convert_laz_to_copc_pdal(
    input_laz: Path,
    output_copc: Path,
    preserve_extra_dims: bool = False,
) -> bool:
    """Convert a single LAZ/LAS file to COPC using PDAL writers.copc."""
    writer_opts = {
        "type": "writers.copc",
        "filename": str(output_copc),
        "forward": "all",
    }
    if preserve_extra_dims:
        writer_opts["extra_dims"] = "all"
    pipeline = {
        "pipeline": [
            {"type": "readers.las", "filename": str(input_laz)},
            writer_opts,
        ]
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as handle:
        json.dump(pipeline, handle)
        pipeline_file = Path(handle.name)
    try:
        result = subprocess.run(
            [get_pdal_path(), "pipeline", str(pipeline_file)],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0 and output_copc.exists() and output_copc.stat().st_size > 0
    finally:
        if pipeline_file.exists():
            pipeline_file.unlink()
