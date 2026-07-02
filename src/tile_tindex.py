#!/usr/bin/env python3
"""Tindex and tile-bounds helpers for SmartTile tiling."""

from __future__ import annotations

import json
import shutil
import sqlite3
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from point_cloud_metadata import point_cloud_files


def get_pdal_path() -> str:
    """Return the PDAL executable path."""
    pdal_path = shutil.which("pdal")
    return pdal_path if pdal_path else "pdal"


def get_pdal_wrench_path() -> str:
    """Return the pdal_wrench executable path."""
    wrench_path = shutil.which("pdal_wrench")
    return wrench_path if wrench_path else "pdal_wrench"


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

    tindex_srs = None
    try:
        info_result = subprocess.run(
            [get_pdal_path(), "info", "--metadata", str(source_files[0])],
            capture_output=True,
            text=True,
            check=False,
        )
        if info_result.returncode == 0:
            meta = json.loads(info_result.stdout)
            tindex_srs = (
                meta.get("metadata", {}).get("srs", {}).get("compoundwkt")
                or meta.get("metadata", {}).get("spatialreference")
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
                get_pdal_path(),
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

            result = subprocess.run(tmp_cmd, capture_output=True, text=True, check=False)
            if result.returncode != 0 and "Unexpected argument 'filelist'" in (result.stderr or result.stdout):
                stdin_cmd = [
                    get_pdal_path(),
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
    grid_offset: float = 1.0,
) -> Tuple[Path, Path, dict]:
    """Calculate tile jobs and bounds JSON from a tindex."""
    print()
    print("=" * 60)
    print("Step 2: Calculating tile bounds")
    print("=" * 60)

    prepare_jobs_script = Path(__file__).parent / "prepare_tile_jobs.py"
    jobs_file = output_dir / f"tile_jobs_{int(tile_length)}m.txt"
    bounds_json = output_dir / "tile_bounds_tindex.json"
    cmd = [
        sys.executable,
        str(prepare_jobs_script),
        str(tindex_file),
        f"--tile-length={tile_length}",
        f"--tile-buffer={tile_buffer}",
        f"--jobs-out={jobs_file}",
        f"--bounds-out={bounds_json}",
        f"--grid-offset={grid_offset}",
    ]

    print(f"  Tile length: {tile_length}m")
    print(f"  Tile buffer: {tile_buffer}m")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"prepare_tile_jobs.py failed: {result.stderr}")

    env = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip().strip('"')

    print(f"  ✓ Calculated {env.get('tile_count', 'unknown')} tiles")
    print(f"  Jobs file: {jobs_file}")
    print(f"  Bounds file: {bounds_json}")
    return jobs_file, bounds_json, env


def update_tile_bounds_json_from_files(
    tile_bounds_json: Path,
    files_dir: Path,
    file_glob: str = "*.laz",
) -> int:
    """Update tile_bounds_tindex.json from created tile file headers."""
    from tile_spatial import get_tile_bounds_from_header

    if not tile_bounds_json.exists():
        return 0
    with tile_bounds_json.open() as handle:
        data = json.load(handle)
    tiles = data.get("tiles", [])
    if not tiles:
        return 0

    label_to_path: Dict[str, Path] = {}
    for path in files_dir.glob(file_glob):
        stem = path.stem
        for sep in ("_subsampled", "_chunk", "."):
            if sep in stem:
                stem = stem.split(sep)[0]
                break
        if stem and stem not in label_to_path:
            label_to_path[stem] = path

    updated = 0
    for tile in tiles:
        label = f"c{tile['col']:02d}_r{tile['row']:02d}"
        path = label_to_path.get(label)
        if path is None:
            continue
        bounds = get_tile_bounds_from_header(path)
        if bounds is None:
            continue
        minx, maxx, miny, maxy = bounds
        tile["bounds"] = [[minx, maxx], [miny, maxy]]
        updated += 1

    if updated > 0:
        with tile_bounds_json.open("w") as handle:
            json.dump(data, handle, indent=2)
    return updated


def get_source_files_from_tindex(tindex_file: Path) -> List[str]:
    """Return source point-cloud paths from a tindex database."""
    conn = sqlite3.connect(str(tindex_file))
    cursor = conn.cursor()
    cursor.execute('SELECT table_name FROM gpkg_contents WHERE data_type = "features" LIMIT 1')
    result = cursor.fetchone()
    if not result:
        conn.close()
        return []

    table_name = result[0]
    cursor.execute(f'SELECT DISTINCT Location FROM "{table_name}"')
    files = [row[0] for row in cursor.fetchall()]
    conn.close()
    return files


def get_source_bounds_from_tindex(tindex_file: Path) -> Dict[str, Tuple[float, float, float, float]]:
    """Return source-file bounds from tindex GeoPackage geometries."""
    conn = sqlite3.connect(str(tindex_file))
    cursor = conn.cursor()
    cursor.execute("SELECT table_name, column_name FROM gpkg_geometry_columns LIMIT 1")
    row = cursor.fetchone()
    if not row:
        conn.close()
        return {}
    table_name, geom_col = row

    cursor.execute(f'SELECT Location, "{geom_col}" FROM "{table_name}"')
    bounds_map = {}
    for filepath, geom_blob in cursor.fetchall():
        if not geom_blob or not filepath:
            continue
        try:
            flags = geom_blob[3]
            envelope_type = (flags >> 1) & 0x07
            header_size = 8
            if envelope_type in (1, 2):
                minx, maxx, miny, maxy = struct.unpack_from("<dddd", geom_blob, header_size)
                bounds_map[filepath] = (minx, miny, maxx, maxy)
        except (struct.error, IndexError):
            continue

    conn.close()
    return bounds_map


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


def bounds_overlap(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> bool:
    """Return whether two (minx, miny, maxx, maxy) boxes overlap."""
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def get_bounds(
    filepath: str,
    source_bounds: Dict[str, Tuple[float, float, float, float]],
    bounds_by_basename: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
) -> Optional[Tuple[float, float, float, float]]:
    """Look up source bounds by exact path, with basename fallback."""
    file_bounds = source_bounds.get(filepath)
    if file_bounds is not None:
        return file_bounds
    if bounds_by_basename is not None:
        return bounds_by_basename.get(Path(filepath).name)
    return None


def filter_source_files_for_tile(
    source_files: List[str],
    source_bounds: Dict[str, Tuple[float, float, float, float]],
    tile_bounds: Tuple[float, float, float, float],
    bounds_by_basename: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
) -> List[str]:
    """Return source files whose bounds overlap tile bounds."""
    result = []
    for source_file in source_files:
        file_bounds = get_bounds(source_file, source_bounds, bounds_by_basename)
        if file_bounds is None or bounds_overlap(file_bounds, tile_bounds):
            result.append(source_file)
    return result
