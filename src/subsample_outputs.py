#!/usr/bin/env python3
"""Output-path and COPC-conversion helpers for SmartTile subsampling products."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

from subsample_encoding import encoding_summary, safe_las_writer_encoding_options


def get_pdal_path() -> str:
    """Return the PDAL executable path."""
    pdal_path = shutil.which("pdal")
    return pdal_path if pdal_path else "pdal"


def laz_output_path(output_file: Path) -> Path:
    """Normalize an output path to regular compressed LAZ."""
    name = output_file.name
    if name.endswith(".copc.laz"):
        name = name[: -len(".copc.laz")] + ".laz"
    elif not name.endswith(".laz"):
        name = output_file.stem + ".laz"
    return output_file.parent / name


def copc_output_path(output_file: Path) -> Path:
    """Normalize an output path to COPC LAZ."""
    name = output_file.name
    if name.endswith(".copc.laz"):
        return output_file
    if name.endswith(".laz"):
        name = name[: -len(".laz")] + ".copc.laz"
    return output_file.parent / f"{output_file.stem}.copc.laz"


def temporary_laz_path(target_output: Path, pipeline_dir: Path) -> Path:
    """Return a temporary LAZ path outside chunk directories for COPC conversion."""
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    stem = target_output.name
    if stem.endswith(".copc.laz"):
        stem = stem[: -len(".copc.laz")]
    elif stem.endswith(".laz"):
        stem = stem[: -len(".laz")]
    return pipeline_dir / f"_{stem}_for_copc.laz"


def subsample_output_files(output_dir: Path, output_copc: bool) -> List[Path]:
    """List subsampling outputs for the selected output format."""
    if output_copc:
        return sorted(output_dir.glob("*.copc.laz"))
    return sorted(path for path in output_dir.glob("*.laz") if not path.name.endswith(".copc.laz"))


def convert_laz_output_to_copc(
    input_laz: Path,
    output_copc: Path,
    source_metadata_file: Optional[Path] = None,
    preserve_extra_dims: bool = False,
) -> bool:
    """Convert a reduced LAZ to COPC using the tiling converter and CRS checks."""
    output_copc.parent.mkdir(parents=True, exist_ok=True)
    if output_copc.exists():
        try:
            output_copc.unlink()
        except OSError:
            pass

    try:
        from copc_metadata import (
            append_source_geotiff_projection_evlrs,
            copc_preserves_source_crs,
        )
        from main_tile import _convert_laz_to_copc

        converted = _convert_laz_to_copc(
            input_laz,
            output_copc,
            preserve_extra_dims=preserve_extra_dims,
        )
        if not converted:
            return False
        if source_metadata_file is not None:
            preserved_geotiff, message = append_source_geotiff_projection_evlrs(
                source_metadata_file,
                output_copc,
            )
            if not preserved_geotiff:
                print(f"      COPC GeoTIFF projection preservation failed: {message}")
                return False
            valid_crs, message = copc_preserves_source_crs(source_metadata_file, output_copc)
            if not valid_crs:
                print(f"      COPC CRS validation failed: {message}")
                return False
        return True
    except Exception as exc:
        print(f"      Warning: shared COPC converter unavailable ({exc}); falling back to PDAL")

    writer_opts = {
        "type": "writers.copc",
        "filename": str(output_copc),
        "forward": "all",
    }
    encoding_source = source_metadata_file if source_metadata_file is not None else input_laz
    writer_opts.update(safe_las_writer_encoding_options(encoding_source))
    if preserve_extra_dims:
        writer_opts["extra_dims"] = "all"
    pipeline = {
        "pipeline": [
            {"type": "readers.las", "filename": str(input_laz)},
            writer_opts,
        ]
    }
    pipeline_file = output_copc.parent / f"_{output_copc.stem}_convert_copc.json"
    with open(pipeline_file, "w") as f:
        json.dump(pipeline, f, indent=2)
    try:
        result = subprocess.run(
            [get_pdal_path(), "pipeline", str(pipeline_file)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            print(
                f"      COPC conversion failed with {encoding_summary(writer_opts)}: "
                f"{result.stderr[:200]}"
            )
            return False
        converted = output_copc.exists() and output_copc.stat().st_size > 0
        if converted and source_metadata_file is not None:
            try:
                from copc_metadata import (
                    append_source_geotiff_projection_evlrs,
                    copc_preserves_source_crs,
                )

                preserved_geotiff, message = append_source_geotiff_projection_evlrs(
                    source_metadata_file,
                    output_copc,
                )
                if not preserved_geotiff:
                    print(f"      COPC GeoTIFF projection preservation failed: {message}")
                    return False
                valid_crs, message = copc_preserves_source_crs(source_metadata_file, output_copc)
                if not valid_crs:
                    print(f"      COPC CRS validation failed: {message}")
                    return False
            except Exception as exc:
                print(f"      COPC CRS validation unavailable after PDAL conversion: {exc}")
                return False
        return converted
    finally:
        if pipeline_file.exists():
            pipeline_file.unlink()
