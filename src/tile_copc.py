#!/usr/bin/env python3
"""LAZ/LAS to COPC conversion (Untwine, PDAL fallback) with CRS validation."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

from copc_metadata import srs_assignment_from_file
from crs_records import validate_single_crs_record


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

    try:
        output_copc.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".smarttile-standard-", dir=output_copc.parent) as tmpdir:
            input_args = []
            for i, path in enumerate(inputs):
                if strip_extra_dims and _has_extra_dimensions(path):
                    staged = Path(tmpdir) / f"{i}.las"
                    _stage_standard_dimensions(path, staged)
                    path = staged
                input_args.extend(["-i", str(path)])
            # Inputs are already reduced when requested. Untwine's --dims may
            # retain extras, and can crash on an already reduced PF0/LAS 1.4 file.
            srs_args = ["--a_srs", srs_arg] if srs_arg else []
            result = subprocess.run(
                [untwine_cmd] + input_args + ["-o", str(output_copc)] + srs_args,
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


def _stage_standard_dimensions(source: Path, output: Path) -> None:
    """Stream native LAS fields into an uncompressed input for bounded Untwine.

    The original remains authoritative and untouched. This is used only for
    processing tiles whose contract already drops ExtraBytes, never enriched
    original or prod-merged products.
    """
    import laspy
    from point_cloud_metadata import copy_single_source_header, write_retained_evlrs

    with laspy.open(source) as reader:
        header = copy_single_source_header(reader.header)
        header.remove_extra_dims(list(header.point_format.extra_dimension_names))
        with laspy.open(output, mode="w", header=header) as writer:
            for record in reader.chunk_iterator(32_768):
                standard = laspy.ScaleAwarePointRecord.zeros(len(record), header=header)
                for name in standard.array.dtype.names:
                    standard.array[name] = record.array[name]
                writer.write_points(standard)
            write_retained_evlrs(writer, header)


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


def convert_laz_to_copc(
    input_laz: Path,
    output_copc: Path,
    preserve_extra_dims: bool = False,
) -> bool:
    """Convert to COPC, automatically scalarizing vector ExtraBytes when kept."""
    if not preserve_extra_dims:
        return _convert_laz_to_copc_impl(input_laz, output_copc, preserve_extra_dims=False)

    from vector_extra_bytes import (
        preserve_vector_schema,
        scalarize_vector_extra_bytes,
        validate_vector_conversion,
        vector_schema_sidecar_path,
    )

    output_copc.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".smarttile-vector-extra-", dir=output_copc.parent) as tmpdir:
        scalar_input = Path(tmpdir) / "scalarized.laz"
        working_input, schema = scalarize_vector_extra_bytes(input_laz, scalar_input)
        # Keep every enriched conversion private until metadata validation passes.
        candidate_dir = Path(tmpdir) / "candidate"
        candidate_dir.mkdir()
        candidate = candidate_dir / output_copc.name
        if not _convert_laz_to_copc_impl(working_input, candidate, preserve_extra_dims=True):
            return False
        from point_cloud_metadata import preserve_extra_bytes_missingness
        try:
            preserve_extra_bytes_missingness(working_input, candidate)
        except ValueError as exc:
            print(f"  Warning: ExtraBytes missingness validation failed: {exc}")
            return False
        if schema is None:
            candidate.replace(output_copc)
            return True
        valid, message = validate_vector_conversion(working_input, candidate)
        if not valid:
            print(f"  Warning: vector ExtraBytes validation failed: {message}")
            return False
        preserved, message = preserve_vector_schema(working_input, candidate)
        if not preserved:
            print(f"  Warning: {message}")
            return False
        # Install the cloud last; restore the previous sidecar if that rename fails.
        sidecar = vector_schema_sidecar_path(output_copc)
        previous_sidecar = Path(tmpdir) / "previous-sidecar.json"
        if sidecar.exists():
            sidecar.replace(previous_sidecar)
        try:
            vector_schema_sidecar_path(candidate).replace(sidecar)
            candidate.replace(output_copc)
        except OSError:
            if previous_sidecar.exists():
                previous_sidecar.replace(sidecar)
            else:
                sidecar.unlink(missing_ok=True)
            raise
        return True


def _convert_laz_to_copc_impl(
    input_laz: Path,
    output_copc: Path,
    preserve_extra_dims: bool = False,
) -> bool:
    """Convert a single LAZ/LAS file to COPC with CRS/GeoTIFF validation.

    The default SmartTile COPC conversion streams standard dimensions into a
    temporary LAS when needed, then runs Untwine and validates the schema/CRS.
    This avoids a full enriched conversion followed by a memory-heavy PDAL retry.
    Prod-merged creation passes
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
                    valid_crs, message = validate_single_crs_record(input_laz, output_copc)
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
        valid_crs, message = validate_single_crs_record(input_laz, output_copc)
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
            valid_crs, _ = validate_single_crs_record(input_laz, output_copc)
            if valid_crs:
                return True
            try:
                output_copc.unlink(missing_ok=True)
            except OSError:
                pass

    if not convert_laz_to_copc_pdal(input_laz, output_copc, preserve_extra_dims=True):
        return False
    valid_crs, message = validate_single_crs_record(input_laz, output_copc)
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
            ["pdal", "pipeline", str(pipeline_file)],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0 and output_copc.exists() and output_copc.stat().st_size > 0
    finally:
        if pipeline_file.exists():
            pipeline_file.unlink()
