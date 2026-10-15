#!/usr/bin/env python3
"""Tindex and tile-bounds helpers for SmartTile tiling."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from point_cloud_metadata import point_cloud_files


def _common_header_srs(paths: List[Path]) -> Optional[str]:
    """Only assign a missing PDAL CRS when every source header agrees."""
    import laspy
    from copc_metadata import crs_equivalent, parse_crs

    common = None
    for path in paths:
        with laspy.open(path) as reader:
            crs = parse_crs(reader.header)
        if crs is None or (common is not None and not crs_equivalent(common, crs)):
            return None
        common = crs
    return common.to_wkt() if common is not None else None


def build_tindex(input_dir: Path, output_gpkg: Path) -> Path:
    """Build a GeoPackage tindex from LAZ/LAS/COPC source files."""
    print()
    print("=" * 60)
    print("Step 1: Building spatial index (tindex)")
    print("=" * 60)

    if output_gpkg.exists():
        if output_gpkg.stat().st_size > 0:
            print(f"  Using existing tindex: {output_gpkg}")
            return output_gpkg
        print(f"  Removing empty tindex from previous failed run: {output_gpkg}")
        output_gpkg.unlink()

    output_gpkg.parent.mkdir(parents=True, exist_ok=True)
    source_files = point_cloud_files(input_dir)
    if not source_files:
        raise ValueError(f"No LAZ/LAS files found in {input_dir}")

    common_header_srs = _common_header_srs(source_files)
    tindex_srs = common_header_srs
    try:
        info_result = subprocess.run(
            ["pdal", "info", "--metadata", str(source_files[0])],
            capture_output=True,
            text=True,
            check=False,
        )
        if info_result.returncode == 0:
            meta = json.loads(info_result.stdout)
            tindex_srs = (
                meta.get("metadata", {}).get("srs", {}).get("compoundwkt")
                or meta.get("metadata", {}).get("spatialreference")
                or common_header_srs
            )
    except Exception as exc:
        print(f"  Warning: Could not extract SRS for tindex: {exc}")

    print(f"  Found {len(source_files)} source files")
    print(f"  Output: {output_gpkg}")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as handle:
        for source_file in source_files:
            handle.write(f"{source_file.absolute()}\n")
        file_list_path = Path(handle.name)

    try:
        with tempfile.TemporaryDirectory(prefix="smarttile_tindex_") as tmp_dir:
            tmp_gpkg = Path(tmp_dir) / output_gpkg.name
            tmp_cmd = [
                "pdal",
                "tindex",
                "create",
                str(tmp_gpkg),
                "--filelist",
                str(file_list_path),
                "--tindex_name=Location",
                "--ogrdriver=GPKG",
                "--fast_boundary",
                "--write_absolute_path",
            ]
            if tindex_srs:
                tmp_cmd.append(f"--t_srs={tindex_srs}")

            if common_header_srs:
                tmp_cmd.append(f"--a_srs={common_header_srs}")

            result = subprocess.run(tmp_cmd, capture_output=True, text=True, check=False)
            if result.returncode != 0 and "Unexpected argument 'filelist'" in (result.stderr or result.stdout):
                stdin_cmd = [
                    "pdal",
                    "tindex",
                    "create",
                    str(tmp_gpkg),
                    "--stdin",
                    "--tindex_name=Location",
                    "--ogrdriver=GPKG",
                    "--fast_boundary",
                    "--write_absolute_path",
                ]
                if tindex_srs:
                    stdin_cmd.append(f"--t_srs={tindex_srs}")
                if common_header_srs:
                    stdin_cmd.append(f"--a_srs={common_header_srs}")
                result = subprocess.run(
                    stdin_cmd,
                    input=file_list_path.read_text(),
                    capture_output=True,
                    text=True,
                    check=False,
                )

            if result.returncode != 0:
                raise RuntimeError(f"pdal tindex failed: {result.stderr or result.stdout or 'unknown error'}")
            if not tmp_gpkg.exists() or tmp_gpkg.stat().st_size == 0:
                raise RuntimeError(f"pdal tindex produced an empty GeoPackage: {tmp_gpkg}")

            shutil.copy2(tmp_gpkg, output_gpkg)

        if not output_gpkg.exists() or output_gpkg.stat().st_size == 0:
            raise RuntimeError(f"Copied tindex is missing or empty: {output_gpkg}")

        print(f"  ✓ Tindex created: {output_gpkg}")
    finally:
        if file_list_path.exists():
            file_list_path.unlink()

    return output_gpkg


def calculate_tile_bounds(
    tindex_file: Path,
    tile_length: float,
    tile_buffer: float,
    output_dir: Path,
    grid_origin=None,
) -> Tuple[Path, Path, dict]:
    """Calculate tile jobs and bounds JSON from a tindex."""
    from get_bounds_from_tindex import write_tile_bounds
    from prepare_tile_jobs import write_job_list

    print()
    print("=" * 60)
    print("Step 2: Calculating tile bounds")
    print("=" * 60)
    jobs_file = output_dir / f"tile_jobs_{int(tile_length)}m.txt"
    bounds_json = output_dir / "tile_bounds_tindex.json"
    print(f"  Tile length: {tile_length}m")
    print(f"  Tile buffer: {tile_buffer}m")
    # Floats, as the former script CLI parsed them: the JSON records 300.0, not 300.
    tile_count = write_tile_bounds(Path(tindex_file), float(tile_length), float(tile_buffer), bounds_json,
                                   grid_origin=grid_origin)
    write_job_list(bounds_json, jobs_file)
    print(f"  ✓ Calculated {tile_count} tiles")
    print(f"  Jobs file: {jobs_file}")
    print(f"  Bounds file: {bounds_json}")
    return jobs_file, bounds_json, {"tile_count": str(tile_count)}


def write_single_cloud_bounds(tile_bounds_json: Path, source_file: Path) -> None:
    """Describe the actual single cloud when the size threshold bypasses tiling."""
    import laspy

    with laspy.open(source_file) as reader:
        h = reader.header
        x0, y0, x1, y1 = float(h.x_min), float(h.y_min), float(h.x_max), float(h.y_max)
    data = json.loads(tile_bounds_json.read_text())
    from tile_bounds_graph import single_cloud_layout
    data = single_cloud_layout(data, (x0, x1, y0, y1))
    tile_bounds_json.write_text(json.dumps(data, indent=2) + "\n")


def update_tile_bounds_json_from_files(
    tile_bounds_json: Path,
    files_dirs,
    file_glob: str = "*.laz",
) -> int:
    """Set each tile's bounds to the union of its output file headers in ``files_dirs``.

    Merge finds overlaps from these bounds, so they must hold the points of
    every resolution, not only the first output's (a coarse first output can
    be up to one voxel tighter than the 1 cm tiles merged later).
    """
    from tile_file_matching import get_file_bounds

    files_dirs = [Path(files_dirs)] if isinstance(files_dirs, (str, Path)) else [Path(d) for d in files_dirs]
    if not tile_bounds_json.exists() or not files_dirs:
        return 0
    with tile_bounds_json.open() as handle:
        data = json.load(handle)
    tiles = data.get("tiles", [])
    if not tiles:
        return 0

    if data.get("tiling_skipped"):
        files = point_cloud_files(files_dirs[0])
        if len(tiles) != 1 or len(files) != 1:
            raise ValueError("Skipped tiling requires exactly one cloud and one layout entry")
        write_single_cloud_bounds(tile_bounds_json, files[0])
        return 1

    extents: Dict[str, Tuple[float, float, float, float]] = {}
    for files_dir in files_dirs:
        for path in sorted(files_dir.glob(file_glob)):
            match = re.search(r"c\d+_r\d+", path.name)
            bounds = get_file_bounds(path) if match else None
            if bounds is None:
                continue
            label = match.group(0)
            seen = extents.get(label)
            extents[label] = bounds if seen is None else (
                min(seen[0], bounds[0]), max(seen[1], bounds[1]), min(seen[2], bounds[2]), max(seen[3], bounds[3]))

    updated = 0
    for tile in tiles:
        bounds = extents.get(f"c{tile['col']:02d}_r{tile['row']:02d}")
        if bounds is None:
            continue
        minx, maxx, miny, maxy = bounds
        tile["bounds"] = [[minx, maxx], [miny, maxy]]
        updated += 1

    if updated > 0:
        with tile_bounds_json.open("w") as handle:
            json.dump(data, handle, indent=2)
    return updated


def parse_proj_bounds(proj_bounds: str) -> Optional[Tuple[float, float, float, float]]:
    """Parse '([xmin,xmax],[ymin,ymax])' into (xmin, ymin, xmax, ymax)."""
    try:
        string_value = proj_bounds.strip().strip("()")
        parts = string_value.split("],[")
        xpart = parts[0].strip("([])").split(",")
        ypart = parts[1].strip("([])").split(",")
        xmin, xmax = float(xpart[0]), float(xpart[1])
        ymin, ymax = float(ypart[0]), float(ypart[1])
        return (xmin, ymin, xmax, ymax)
    except (ValueError, IndexError):
        return None

