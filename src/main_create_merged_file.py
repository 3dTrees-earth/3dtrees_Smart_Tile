#!/usr/bin/env python3
"""
Create prod-merged point-cloud products from Original-with-predictions files.

Prod-merged products use original uploaded points as geometry and select real
points with PDAL's nearest-to-centroid voxel subsampling when a lower product
resolution is requested.
"""

from __future__ import annotations

import math
import re
import subprocess
from pathlib import Path
from typing import List, Tuple


SUPPORTED_OUTPUT_FORMATS = {"laz", "copc.laz", "ply"}


def get_pdal_path() -> str:
    import shutil

    pdal_path = shutil.which("pdal")
    if pdal_path is None:
        raise RuntimeError("PDAL executable not found on PATH; create_merged_file requires pdal")
    return pdal_path


def point_cloud_files(directory: Path, include_copc: bool = False) -> List[Path]:
    """Return source LAZ/LAS files."""
    files = sorted(directory.glob("*.laz")) + sorted(directory.glob("*.las"))
    if include_copc:
        return files
    return [path for path in files if not path.name.lower().endswith(".copc.laz")]


def copc_files(directory: Path) -> List[Path]:
    return sorted(directory.glob("*.copc.laz"))


def source_key(path: Path) -> str:
    """Return a stable source key for raw LAZ/LAS and its COPC derivative."""
    name = path.name.lower()
    if name.endswith(".copc.laz"):
        return path.name[:-9]
    if name.endswith(".laz") or name.endswith(".las"):
        return path.stem
    return path.stem


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
    for raw_token in (value or "copc.laz").split(","):
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


def prepare_copc_inputs(original_with_predictions_dir: Path, output_dir: Path) -> List[Path]:
    """Convert Original-with-predictions LAZ/LAS files to COPC using SmartTile's tiling converter."""
    if not original_with_predictions_dir.exists():
        raise FileNotFoundError(f"Original-with-predictions directory not found: {original_with_predictions_dir}")

    existing_copc = copc_files(original_with_predictions_dir)
    raw_files = point_cloud_files(original_with_predictions_dir)
    if not raw_files and not existing_copc:
        raise FileNotFoundError(f"No LAZ/LAS files found in {original_with_predictions_dir}")

    copc_dir = output_dir / "original_with_predictions_copc"
    copc_dir.mkdir(parents=True, exist_ok=True)
    selected_by_source = {source_key(path): path for path in existing_copc}
    converter = None

    for input_file in raw_files:
        key = source_key(input_file)
        if key in selected_by_source:
            print(f"    Reusing existing COPC for {input_file.name}: {selected_by_source[key].name}")
            continue

        output_copc = copc_dir / f"{key}.copc.laz"
        if output_copc.exists() and output_copc.stat().st_size > 0:
            print(f"    Reusing COPC: {output_copc.name}")
            selected_by_source[key] = output_copc
            continue

        print(f"    Converting {input_file.name} -> {output_copc.name}")
        if converter is None:
            from main_tile import _convert_laz_to_copc

            converter = _convert_laz_to_copc
        if not converter(input_file, output_copc):
            raise RuntimeError(f"LAZ/LAS -> COPC conversion failed: {input_file}")
        selected_by_source[key] = output_copc

    return [selected_by_source[key] for key in sorted(selected_by_source)]


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


def create_prod_merged_file(
    copc_input_files: List[Path],
    output_file: Path,
    resolution: float,
    output_format: str = "copc.laz",
) -> Path:
    """Create one prod-merged product at the requested resolution and format."""
    if not copc_input_files:
        raise ValueError("No COPC input files provided")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    pipeline = prod_merged_pipeline(copc_input_files, output_file, resolution, output_format)
    pipeline_file = output_file.parent / f"_{output_file.stem}_pipeline.json"
    with open(pipeline_file, "w") as f:
        import json

        json.dump(pipeline, f, indent=2)

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
        raise RuntimeError(f"PDAL prod-merged pipeline failed: {result.stderr[:500]}")
    if not output_file.exists() or output_file.stat().st_size == 0:
        raise RuntimeError(f"Prod-merged output was not created: {output_file}")
    return output_file


def create_prod_merged_files(
    original_with_predictions_dir: Path,
    output_dir: Path,
    resolution_selector: str,
    output_format_selector: str,
    res1: float,
    res2: float,
) -> List[Path]:
    """Create all selected prod-merged products."""
    outputs = []
    print("  Preparing COPC inputs with SmartTile untwine/PDAL fallback conversion")
    copc_input_files = prepare_copc_inputs(original_with_predictions_dir, output_dir)
    output_formats = parse_merged_output_formats(output_format_selector)
    for label, resolution in parse_merged_resolutions(resolution_selector, res1, res2):
        for output_format in output_formats:
            output_file = prod_merged_output_path(output_dir, label, output_format)
            print(
                f"  Creating {output_file.name} at {resolution:g}m "
                f"with nearest-to-centroid"
            )
            outputs.append(
                create_prod_merged_file(
                    copc_input_files,
                    output_file,
                    resolution,
                    output_format,
                )
            )
    return outputs
