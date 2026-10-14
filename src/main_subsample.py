"""
Subsampled-output naming and point-cloud bounds helpers.

`subsample_output_name`/`resolution_label` name the tile task's subsampled
outputs (tile_centroids); `get_file_bounds`/`get_file_xy_scales` serve
create_merged_file's chunk bounds.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import laspy


def get_pdal_path() -> str:
    """Get the path to pdal executable."""
    import shutil
    # Use shutil.which to find pdal in PATH
    pdal_path = shutil.which("pdal")
    return pdal_path if pdal_path else "pdal"


def _parse_bounds_metadata(metadata) -> Optional[Tuple[float, float, float, float]]:
    """Extract finite XY bounds from PDAL metadata."""
    if not isinstance(metadata, dict):
        return None

    keys = ("minx", "maxx", "miny", "maxy")
    if all(key in metadata for key in keys):
        try:
            minx, maxx, miny, maxy = (float(metadata[key]) for key in keys)
        except (TypeError, ValueError):
            return None
        if (
            all(math.isfinite(value) for value in (minx, maxx, miny, maxy))
            and maxx >= minx
            and maxy >= miny
        ):
            return (minx, maxx, miny, maxy)
        return None

    for value in metadata.values():
        if isinstance(value, dict):
            bounds = _parse_bounds_metadata(value)
            if bounds is not None:
                return bounds
        elif isinstance(value, list):
            for item in value:
                bounds = _parse_bounds_metadata(item)
                if bounds is not None:
                    return bounds
    return None


def get_file_bounds(filepath: Path) -> Optional[Tuple[float, float, float, float]]:
    """
    Get spatial bounds of a point cloud file using pdal info.

    Returns:
        Tuple of (minx, maxx, miny, maxy) or None on error
    """
    try:
        pdal_cmd = get_pdal_path()
        result = subprocess.run(
            [pdal_cmd, "info", "--metadata", str(filepath)],
            capture_output=True,
            text=True,
            check=True
        )

        payload = json.loads(result.stdout)
        return _parse_bounds_metadata(payload.get("metadata", payload))
    except Exception:
        return None


def get_file_xy_scales(filepath: Path) -> Tuple[float, float]:
    """Return LAS/COPC XY scales for half-open chunk bounds."""
    try:
        with laspy.open(str(filepath), laz_backend=laspy.LazBackend.LazrsParallel) as reader:
            scales = reader.header.scales
            return (float(scales[0]), float(scales[1]))
    except Exception:
        return (0.001, 0.001)


def resolution_label(resolution: float) -> str:
    """Original resolution, whole centimetres ("29cm"), or millimetres ("5mm")."""
    if resolution == 0:
        return "original"
    centimetres = resolution * 100
    if abs(centimetres - round(centimetres)) < 1e-6 and round(centimetres) >= 1:
        return f"{round(centimetres)}cm"
    return f"{round(resolution * 1000)}mm"


def subsample_output_name(input_file: Path, resolution: float, output_prefix: Optional[str],
                          output_copc: bool) -> str:
    """Output file name for one subsampled input, shared by every subsampling path."""
    res_label = resolution_label(resolution)
    output_ext = ".copc.laz" if output_copc else ".laz"
    # Generate output filename
    stem = input_file.stem
    # Remove .copc suffix if present
    if stem.endswith('.copc'):
        stem = stem[:-5]

    # Extract original base filename by removing prefixes and resolution suffixes
    base_name = stem

    # Remove resolution suffix patterns from previous subsampling stages.
    base_name = re.sub(r'_subsampled[\d.]+m$', '', base_name)
    base_name = re.sub(r'_subsampled_(?:original|\d+(?:\.\d+)?(?:cm|mm))$', '', base_name)
    base_name = re.sub(r'_\d+cm$', '', base_name)

    # Remove output_prefix if present at the start (e.g., "output_dir_100m_")
    if output_prefix and base_name.startswith(output_prefix + '_'):
        base_name = base_name[len(output_prefix) + 1:]

    # Remove any remaining prefix patterns that look like "something_100m_" or "output_dir_100m_"
    base_name = re.sub(r'^[^_]+_\d+m_', '', base_name)

    # For tiled files, try to extract tile ID (c##_r##) pattern
    tile_match = re.search(r'(c\d+_r\d+)', base_name)
    if tile_match:
        # Keep tile ID for tiled files
        tile_id = tile_match.group(1)
        # Extract base name before tile ID if there's a prefix
        base_before_tile = base_name[:tile_match.start()]
        if base_before_tile and base_before_tile.endswith('_'):
            base_before_tile = base_before_tile[:-1]
        # Remove any remaining prefix from base_before_tile
        if base_before_tile:
            base_before_tile = re.sub(r'^[^_]+_\d+m_', '', base_before_tile)
            if base_before_tile:
                output_name = f"{base_before_tile}_{tile_id}_subsampled_{res_label}{output_ext}"
            else:
                output_name = f"{tile_id}_subsampled_{res_label}{output_ext}"
        else:
            output_name = f"{tile_id}_subsampled_{res_label}{output_ext}"
    else:
        # Single file or no tile ID - use clean base name
        # Remove any remaining prefix patterns
        base_name = re.sub(r'^[^_]+_\d+m_', '', base_name)
        output_name = f"{base_name}_subsampled_{res_label}{output_ext}"
    return output_name
