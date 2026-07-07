from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional, Tuple

import laspy


_TREE_FILE_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"^input_trees(?:_[^.]+)?\.txt$", re.IGNORECASE),
    re.compile(r"^input_trees_info(?:_[^.]+)?\.txt$", re.IGNORECASE),
    re.compile(r"^trees(?:_[^.]+)?\.txt$", re.IGNORECASE),
    re.compile(r"^trees_info(?:_[^.]+)?\.txt$", re.IGNORECASE),
    re.compile(r"^.+_trees(?:_[^.]+)?\.txt$", re.IGNORECASE),
    re.compile(r"^.+_trees_info(?:_[^.]+)?\.txt$", re.IGNORECASE),
)


def is_tree_sidecar_file(path: Path) -> bool:
    """Return True for tree sidecar text files."""
    path = Path(path)
    name = path.name
    if any(pattern.match(name) for pattern in _TREE_FILE_PATTERNS):
        return True
    return path.suffix.lower() == ".txt"


def classify_tree_sidecar_file(path: Path) -> Optional[str]:
    """Return the canonical output suffix for a supported tree sidecar text file."""
    if not is_tree_sidecar_file(path):
        return None
    lower_name = Path(path).name.lower()
    if "trees_info" in lower_name:
        return "_trees_info.txt"
    return "_trees.txt"


def derive_tile_buffer_from_json(tile_bounds_json: Path) -> float:
    """
    Derive the tile buffer width from tile_bounds_tindex.json.

    Preference order:
    1. root-level ``tile_buffer``
    2. per-tile ``bounds`` vs ``core`` difference
    """
    tile_bounds_json = Path(tile_bounds_json)
    with tile_bounds_json.open() as f:
        data = json.load(f)

    if isinstance(data, dict):
        root_buffer = data.get("tile_buffer")
        if root_buffer is not None:
            return float(root_buffer)

        tiles = data.get("tiles", [])
        for tile in tiles:
            bounds = tile.get("bounds")
            core = tile.get("core")
            if not bounds or not core:
                continue
            try:
                x_pad = (
                    float(bounds[0][1])
                    - float(core[0][1])
                    + float(core[0][0])
                    - float(bounds[0][0])
                ) / 2.0
                y_pad = (
                    float(bounds[1][1])
                    - float(core[1][1])
                    + float(core[1][0])
                    - float(bounds[1][0])
                ) / 2.0
                pad = max(x_pad, y_pad)
            except (TypeError, ValueError, IndexError):
                continue
            if pad > 0:
                return float(pad)

    raise ValueError(f"Could not derive tile buffer width from {tile_bounds_json}")


def derive_border_zone_width_from_json(tile_bounds_json: Path) -> float:
    """Derive the default border-zone width from tile buffer metadata."""
    return derive_tile_buffer_from_json(tile_bounds_json)


def validate_pointcloud_header(path: Path) -> Tuple[bool, Optional[str]]:
    """Return whether a LAS/LAZ/COPC header can be opened by laspy."""
    try:
        with laspy.open(str(path), laz_backend=laspy.LazBackend.LazrsParallel) as reader:
            _ = reader.header.point_count
        return True, None
    except Exception as exc:
        return False, str(exc)
