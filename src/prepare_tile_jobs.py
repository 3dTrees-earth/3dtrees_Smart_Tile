"""Write the per-tile job list (label, projected and geographic bounds) from the bounds JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Tuple


def _transformer_from_crs(srs: str):
    """Create a geographic transformer when pyproj is available."""
    try:
        from pyproj import Transformer
    except ImportError:
        print(
            "[prepare_tile_jobs] Warning: pyproj unavailable; "
            "writing projected bounds as geographic fallback",
            file=sys.stderr,
        )
        return None
    return Transformer.from_crs(srs, "EPSG:4326", always_xy=True)


def write_job_list(bounds_json: Path, job_file: Path) -> None:
    with bounds_json.open() as f:
        data = json.load(f)

    # Get SRS from tile bounds data
    # get_bounds_from_tindex.py uses 'proj_srs' for the working projection
    srs = data.get("proj_srs", data.get("tindex_srs", "missing"))

    transformer = None
    if srs != "missing":
        try:
            transformer = _transformer_from_crs(srs)
        except Exception as e:
            print(f"[prepare_tile_jobs] Warning: Could not create transformer from {srs} to EPSG:4326: {e}", file=sys.stderr)

    def to_geo_bounds(bx: List[float], by: List[float]) -> Tuple[List[float], List[float]]:
        if transformer is None:
            # Fallback: if no transformer, return original bounds (planar units)
            return bx, by

        corners = [
            transformer.transform(bx[0], by[0]),
            transformer.transform(bx[0], by[1]),
            transformer.transform(bx[1], by[0]),
            transformer.transform(bx[1], by[1]),
        ]
        lons, lats = zip(*corners)
        return [min(lons), max(lons)], [min(lats), max(lats)]

    lines: List[str] = []
    for tile in data["tiles"]:
        label = f"c{tile['col']:02d}_r{tile['row']:02d}"
        bx, by = tile["bounds"]
        proj_bounds = f"([{bx[0]},{bx[1]}],[{by[0]},{by[1]}])"
        geo_x, geo_y = to_geo_bounds(bx, by)
        geo_bounds = f"([{geo_x[0]},{geo_x[1]}],[{geo_y[0]},{geo_y[1]}])"
        lines.append(f"{label}|{proj_bounds}|{geo_bounds}")

    # Ensure the file ends with a newline so shell read loops don't drop the last tile.
    job_file.write_text("\n".join(lines) + "\n")
    print(f"[prepare_tile_jobs] wrote {len(lines)} jobs to {job_file}", file=sys.stderr)
