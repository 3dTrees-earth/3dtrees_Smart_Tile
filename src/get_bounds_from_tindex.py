"""Get the extent of a tile index and compute the tile bounds JSON.

Always treats coordinates as planar/metric units (input units are checked by tile_crs).
"""

import json
import math
import sys
from pathlib import Path

import fiona
from pyproj import CRS

PROJECTED_CRS_FALLBACK = "EPSG:32630"


def load_extent_from_tindex(tindex_path: Path):
    """Load extent from tindex shapefile.

    Returns:
        Tuple of (minx, miny, maxx, maxy), crs_string

    The coordinates are returned in the native units of the data (assumed metric).
    """
    with fiona.open(tindex_path) as src:
        # Get CRS from file
        srs_info = "missing"
        if src.crs:
            try:
                crs = CRS.from_user_input(src.crs)
                srs_info = crs.to_string()
            except Exception:
                srs_info = str(src.crs)
            # Units are checked on the input headers (tile_crs). The index CRS
            # is no evidence: PDAL labels an index of CRS-less, local-metre
            # inputs EPSG:4326 by default without reprojecting them.

        print(f"  Detected CRS: {srs_info} (Treating as Planar/Metric)", file=sys.stderr)

        if srs_info == "missing":
             print(f"  ⚠ Warning: CRS missing; tiling in dataset-local planar coordinates.", file=sys.stderr)

        # Get bounds of all features
        minx = miny = math.inf
        maxx = maxy = -math.inf

        feature_count = 0
        for feature in src:
            feature_count += 1
            geom = feature['geometry']
            if geom['type'] == 'Polygon':
                coords = geom['coordinates'][0]
            elif geom['type'] == 'MultiPolygon':
                coords = [c for poly in geom['coordinates'] for c in poly[0]]
            else:
                continue

            xs, ys = zip(*coords)
            minx = min(minx, min(xs))
            miny = min(miny, min(ys))
            maxx = max(maxx, max(xs))
            maxy = max(maxy, max(ys))

        if feature_count == 0:
            bounds = src.bounds
            if bounds and bounds != (0.0, 0.0, 0.0, 0.0):
                minx, miny, maxx, maxy = bounds
            else:
                raise ValueError(f"No features found in tindex: {tindex_path}")

        return (minx, miny, maxx, maxy), srs_info


def build_tiles(minx, miny, maxx, maxy, length, buffer, align_to_grid=False, grid_origin=None):
    """Build tile grid.

    Args:
        minx, miny, maxx, maxy: Data extent bounds
        length: Tile size in units
        buffer: Buffer size in units
        align_to_grid: If True, snap to grid.
    """
    # Validate inputs - check for infinity or NaN
    if not all(math.isfinite(v) for v in [minx, miny, maxx, maxy]):
        raise ValueError(
            f"Invalid bounds detected (infinity or NaN): "
            f"minx={minx}, miny={miny}, maxx={maxx}, maxy={maxy}."
        )

    if length <= 0 or not math.isfinite(length) or buffer < 0 or not math.isfinite(buffer):
        raise ValueError("Tile length must be positive and buffer nonnegative")
    if grid_origin is not None:
        if len(grid_origin) != 2 or not all(math.isfinite(v) for v in grid_origin):
            raise ValueError("Grid origin must have two finite coordinates")
        start_x, start_y = grid_origin
        if start_x > minx or start_y > miny:
            raise ValueError("Grid origin must not exclude source minimum bounds")
        end_x = start_x + math.ceil((maxx - start_x) / length) * length
        end_y = start_y + math.ceil((maxy - start_y) / length) * length
    elif align_to_grid:
        start_x = math.floor(minx / length) * length
        start_y = math.floor(miny / length) * length
        end_x = math.ceil(maxx / length) * length
        end_y = math.ceil(maxy / length) * length
    else:
        start_x = minx
        start_y = miny
        x_range = maxx - minx
        y_range = maxy - miny
        if not math.isfinite(x_range) or not math.isfinite(y_range):
            raise ValueError(
                f"Invalid range calculated: x_range={x_range}, y_range={y_range}. "
                f"Bounds: minx={minx}, miny={miny}, maxx={maxx}, maxy={maxy}"
            )
        end_x = math.ceil(x_range / length) * length + start_x
        end_y = math.ceil(y_range / length) * length + start_y

    # Estimate number of tiles and warn if excessive
    num_tiles_x = int(math.ceil((end_x - start_x) / length))
    num_tiles_y = int(math.ceil((end_y - start_y) / length))
    total_tiles = num_tiles_x * num_tiles_y

    # Warn if creating too many tiles (more than 1 million)
    MAX_TILES = 1000000
    if total_tiles > MAX_TILES:
        raise ValueError(
            f"Would create {total_tiles:,} tiles ({num_tiles_x} x {num_tiles_y}), "
            f"which exceeds the maximum of {MAX_TILES:,}. "
            f"This usually indicates the data extent is too large or the tile size is too small. "
            f"Bounds: minx={minx:.2f}, miny={miny:.2f}, maxx={maxx:.2f}, maxy={maxy:.2f}, "
            f"tile_length={length}. "
            f"Consider using a larger tile size or splitting the data into smaller regions."
        )

    tiles = []
    col = 0
    x = start_x
    while x < end_x:
        row = 0
        y = start_y
        while y < end_y:
            core_x = [x, x + length]
            core_y = [y, y + length]
            buffered_x = [core_x[0] - buffer, core_x[1] + buffer]
            buffered_y = [core_y[0] - buffer, core_y[1] + buffer]
            tiles.append(
                {
                    "col": col,
                    "row": row,
                    "core": [core_x, core_y],
                    "planned_bounds": [buffered_x, buffered_y],
                    "bounds": [buffered_x, buffered_y],
                }
            )
            row += 1
            y += length
        col += 1
        x += length

    grid_bounds = (
        start_x - buffer,
        end_x + buffer,
        start_y - buffer,
        end_y + buffer,
    )
    tiles.sort(key=lambda t: (t["col"], t["row"]))
    return tiles, grid_bounds


def write_tile_bounds(tindex_path: Path, tile_length: float, tile_buffer: float, out: Path,
                      grid_origin=None) -> int:
    """Write the tile bounds JSON for the tindex extent; returns the tile count."""
    (minx, miny, maxx, maxy), srs = load_extent_from_tindex(tindex_path)
    proj_crs = srs if srs != "missing" else PROJECTED_CRS_FALLBACK
    # Data-aligned tiling; the geographic extent equals the planar one (metric input).
    tiles, grid_bounds = build_tiles(minx, miny, maxx, maxy, tile_length, tile_buffer,
                                     align_to_grid=False, grid_origin=grid_origin)
    extent = {"minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy}
    summary = {
        "tindex": str(tindex_path),
        "tindex_srs": srs,
        "proj_srs": proj_crs,
        "proj_extent": extent,
        "geo_extent": dict(extent),
        "tile_length": tile_length,
        "tile_buffer": tile_buffer,
        "grid_bounds": {"xmin": grid_bounds[0], "xmax": grid_bounds[1],
                        "ymin": grid_bounds[2], "ymax": grid_bounds[3]},
        "tiles": tiles,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(summary, f, indent=2)
    return len(tiles)
