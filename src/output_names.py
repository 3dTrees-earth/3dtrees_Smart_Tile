"""File naming: `file_stem`, `subsample_output_name` and `resolution_label`."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

KNOWN_EXTENSIONS = (".laz", ".las", ".copc", ".txt", ".ply")


def split_known_extensions(path) -> tuple[str, list[str]]:
    """Split a file name into its stem and trailing known extensions (lowercase, last first).

    Extensions are stripped repeatedly and case-insensitively, so names that a
    caller such as Galaxy built as `<name>.<ext>` from a name that already had
    an extension still pair: `plot_A.laz.laz` and `plot_A.LAZ` -> `plot_A`,
    `c00_r00.copc.laz` -> `c00_r00`. A name is never stripped to nothing.
    """
    name, extensions = Path(path).name, []
    while True:
        extension = next((e for e in KNOWN_EXTENSIONS
                          if name.lower().endswith(e) and len(name) > len(e)), None)
        if extension is None:
            return name, extensions
        name = name[:-len(extension)]
        extensions.append(extension)


def file_stem(path) -> str:
    """The name that pairs and names a file: its name without known extensions."""
    return split_known_extensions(path)[0]


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
    stem = file_stem(input_file)

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
