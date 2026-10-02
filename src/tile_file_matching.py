"""Match prediction tiles to their target tiles by the tile task's bounds JSON.

Both sides match JSON tiles by grid position (bounds, then centroid).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import laspy

from point_cloud_metadata import point_cloud_files
from tile_bounds_graph import build_neighbor_graph_from_bounds_json, match_tiles_to_json_bounds

LAZ_BACKEND = getattr(laspy.LazBackend, "Lazrs", laspy.LazBackend.LazrsParallel)


def get_file_bounds(filepath: Path) -> Optional[Tuple[float, float, float, float]]:
    """
    Get spatial bounds of a point cloud file using laspy header only (no point loading).

    Args:
        filepath: Path to LAZ file

    Returns:
        Tuple of (minx, maxx, miny, maxy) or None on error
    """
    try:
        # Use laspy.open() to read only header, not all points
        with laspy.open(str(filepath), laz_backend=LAZ_BACKEND) as las:
            return (las.header.x_min, las.header.x_max, las.header.y_min, las.header.y_max)
    except Exception:
        return None


def _point_cloud_files(directory: Path) -> List[Path]:
    """Return LAS/LAZ inputs for remap matching, preferring COPC twins."""
    return point_cloud_files(directory)


def _match_files_via_json(
    tile_bounds_json: Path,
    source_folder: Path,
    target_folder: Path,
) -> List[Tuple[Path, Path, str]]:
    """
    Match source and target files using tile_bounds_tindex.json.
    Both source and target files are matched to JSON entries (stepwise bounds/centroid);
    pairs are formed by shared JSON index. Uses the same stable matching as merge.
    """
    source_files = _point_cloud_files(source_folder)
    target_files = _point_cloud_files(target_folder)
    if not source_files or not target_files:
        return []

    # Single-file shortcut: 1 source and 1 target -> pair directly
    if len(source_files) == 1 and len(target_files) == 1:
        src = source_files[0]
        tgt = target_files[0]
        tile_id = re.sub(r"_segmented$|_results$|_subsampled[\d.]+m$", "", src.stem)
        if not tile_id:
            tile_id = src.stem
        print(f"  Single file pair: {src.name} <-> {tgt.name}")
        return [(src, tgt, tile_id)]

    json_bounds, centers, _ = build_neighbor_graph_from_bounds_json(tile_bounds_json)

    source_boundaries: Dict[str, Tuple[float, float, float, float]] = {}
    stem_to_source_path: Dict[str, Path] = {}
    for f in source_files:
        b = get_file_bounds(f)
        if b is not None:
            stem = f.stem
            source_boundaries[stem] = b
            stem_to_source_path[stem] = f

    target_boundaries: Dict[str, Tuple[float, float, float, float]] = {}
    stem_to_target_path: Dict[str, Path] = {}
    for f in target_files:
        b = get_file_bounds(f)
        if b is not None:
            stem = f.stem
            target_boundaries[stem] = b
            stem_to_target_path[stem] = f

    if not source_boundaries:
        raise ValueError("Could not read bounds from any source file")
    if not target_boundaries:
        raise ValueError("Could not read bounds from any target file")

    source_to_json, json_to_source = match_tiles_to_json_bounds(
        source_boundaries, json_bounds, centers
    )
    target_to_json, json_to_target = match_tiles_to_json_bounds(
        target_boundaries, json_bounds, centers
    )

    matches: List[Tuple[Path, Path, str]] = []
    for j in range(len(json_bounds)):
        src_stem = json_to_source.get(j)
        tgt_stem = json_to_target.get(j)
        if src_stem is None or tgt_stem is None:
            if src_stem is not None:
                raise ValueError(
                    f"Source file {stem_to_source_path[src_stem].name} matched JSON tile index {j} "
                    "but no target file matched that tile. Cannot remap."
                )
            continue
        src_path = stem_to_source_path[src_stem]
        tgt_path = stem_to_target_path[tgt_stem]
        tile_id = re.sub(r"_segmented$|_results$|_subsampled[\d.]+m$", "", src_stem)
        if not tile_id:
            tile_id = src_stem
        matches.append((src_path, tgt_path, tile_id))

    return matches


def find_matching_files(
    source_folder: Path,
    target_folder: Path,
    *,
    tile_bounds_json: Path,
) -> List[Tuple[Path, Path, str]]:
    """Match prediction tiles to target tiles by the tile task's bounds JSON.

    Returns (source_file, target_file, tile_id) tuples. The JSON is required:
    merge needs the same layout for tile cores and buffers.
    """
    if tile_bounds_json is None or not Path(tile_bounds_json).exists():
        raise ValueError(f"Tile bounds JSON is required for tile matching: {tile_bounds_json}")
    print(f"  Using tile_bounds_tindex.json for matching: {tile_bounds_json}")
    return _match_files_via_json(Path(tile_bounds_json), source_folder, target_folder)
