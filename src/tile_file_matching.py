"""Match prediction tiles to their target tiles by tile bounds or file extents.

With a tile bounds JSON, tiles match by grid position (bounds, then centroid);
without one, by a strict 1 m extent tolerance, then a 30% IoU fallback.
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


def calculate_bounds_overlap(
    bounds1: Tuple[float, float, float, float],
    bounds2: Tuple[float, float, float, float]
) -> float:
    """
    Calculate IoU (Intersection over Union) between two bounding boxes.

    IoU is more robust than "overlap % of smaller file" because:
    - It's symmetric
    - Penalizes size mismatches
    - Prevents nested files from all getting 100%

    Args:
        bounds1: Tuple of (minx, maxx, miny, maxy) for first file
        bounds2: Tuple of (minx, maxx, miny, maxy) for second file

    Returns:
        IoU as percentage (0-100), where 100% = perfect overlap
    """
    if bounds1 is None or bounds2 is None:
        return 0.0

    minx1, maxx1, miny1, maxy1 = bounds1
    minx2, maxx2, miny2, maxy2 = bounds2

    # Calculate intersection region
    overlap_minx = max(minx1, minx2)
    overlap_maxx = min(maxx1, maxx2)
    overlap_miny = max(miny1, miny2)
    overlap_maxy = min(maxy1, maxy2)

    # Check if there's actual overlap
    if overlap_minx >= overlap_maxx or overlap_miny >= overlap_maxy:
        return 0.0

    # Calculate intersection area
    intersection = (overlap_maxx - overlap_minx) * (overlap_maxy - overlap_miny)

    # Calculate union area
    area1 = (maxx1 - minx1) * (maxy1 - miny1)
    area2 = (maxx2 - minx2) * (maxy2 - miny2)
    union = area1 + area2 - intersection

    # Calculate IoU as percentage
    iou = (intersection / union) * 100 if union > 0 else 0.0

    return iou


def bounds_match_tolerance(
    bounds1: Tuple[float, float, float, float],
    bounds2: Tuple[float, float, float, float],
    tolerance: float = 1.0
) -> bool:
    """
    Check if two bounds match within a strict tolerance.

    Args:
        bounds1: Tuple of (minx, maxx, miny, maxy) for first file
        bounds2: Tuple of (minx, maxx, miny, maxy) for second file
        tolerance: Maximum difference in meters for each bound component (default: 1.0m)

    Returns:
        True if all bound components match within tolerance
    """
    if bounds1 is None or bounds2 is None:
        return False

    minx1, maxx1, miny1, maxy1 = bounds1
    minx2, maxx2, miny2, maxy2 = bounds2

    return (abs(minx1 - minx2) <= tolerance and
            abs(maxx1 - maxx2) <= tolerance and
            abs(miny1 - miny2) <= tolerance and
            abs(maxy1 - maxy2) <= tolerance)


def _match_files_via_json(
    tile_bounds_json: Path,
    source_folder: Path,
    target_folder: Path,
    verbose: bool = False,
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
        if verbose:
            print(f"  ✓ Matched (JSON): {src_path.name} <-> {tgt_path.name} ({tile_id})")

    return matches


def find_matching_files(
    source_folder: Path,
    target_folder: Path,
    overlap_threshold: float = 99.0,
    verbose: bool = False,
    tile_bounds_json: Optional[Path] = None,
) -> List[Tuple[Path, Path, str]]:
    """
    Find matching files between source and target folders.
    If tile_bounds_json is provided and exists, uses JSON-based matching (stepwise
    bounds/centroid, same as merge). Otherwise uses two-stage matching:
    1. Strict tolerance matching (1m)
    2. IoU matching (30%) fallback

    Args:
        source_folder: Directory containing source LAZ files (e.g., segmented files)
        target_folder: Directory containing target LAZ files (e.g., resolution-1 subsampled files)
        overlap_threshold: DEPRECATED - not used (kept for compatibility)
        verbose: If True, print detailed matching diagnostics
        tile_bounds_json: Optional path to tile_bounds_tindex.json for grid-based matching

    Returns:
        List of (source_file, target_file, tile_id) tuples
    """
    if tile_bounds_json is not None and tile_bounds_json.exists():
        print(f"  Using tile_bounds_tindex.json for matching: {tile_bounds_json}")
        return _match_files_via_json(tile_bounds_json, source_folder, target_folder, verbose)

    matches = []

    # Get all LAZ/LAS files from both folders (flat structure)
    source_files = _point_cloud_files(source_folder)
    target_files = _point_cloud_files(target_folder)

    if not source_files:
        print(f"  Warning: No LAZ/LAS files found in source folder: {source_folder}")
        return matches

    if not target_files:
        print(f"  Warning: No LAZ/LAS files found in target folder: {target_folder}")
        return matches

    print(f"  Found {len(source_files)} source files and {len(target_files)} target files")
    print(f"  Two-stage matching: 1m tolerance → 30% IoU fallback")

    # Extract bounds for all target files once
    target_bounds_map = {}
    for target_file in target_files:
        bounds = get_file_bounds(target_file)
        if bounds:
            target_bounds_map[target_file] = bounds

    # Match each source file to target files using two-stage approach
    unmatched_count = 0
    tolerance_matches = 0
    iou_matches = 0

    for source_file in source_files:
        source_bounds = get_file_bounds(source_file)
        if source_bounds is None:
            print(f"  Warning: Could not extract bounds from {source_file.name}")
            continue

        # Find matching target file(s) and track overlap for each
        matched_targets = []

        for target_file, target_bounds in target_bounds_map.items():
            # Check both matching methods
            tolerance_match = bounds_match_tolerance(source_bounds, target_bounds, tolerance=1.0)
            iou = calculate_bounds_overlap(source_bounds, target_bounds)
            iou_match = iou >= 30.0

            # Determine which method succeeded (priority: tolerance > iou)
            if tolerance_match:
                match_method = 'tolerance'
            elif iou_match:
                match_method = 'iou'
            else:
                continue  # No match

            matched_targets.append((target_file, match_method, iou))

        if len(matched_targets) == 0:
            unmatched_count += 1
            print(f"  ⚠ Warning: No matching target file found for {source_file.name}")

            # Provide helpful diagnostics for best candidate
            if verbose:
                best_iou = 0.0
                best_target = None
                for target_file, target_bounds in target_bounds_map.items():
                    iou = calculate_bounds_overlap(source_bounds, target_bounds)
                    if iou > best_iou:
                        best_iou = iou
                        best_target = target_file
                if best_target:
                    print(f"    Best candidate: {best_target.name} (IoU: {best_iou:.2f}%)")

            continue

        # Sort by match quality (method priority, then IoU)
        def match_score(match_tuple):
            target_file, match_method, iou = match_tuple
            method_priority = {'tolerance': 2, 'iou': 1}
            return (method_priority[match_method], iou)

        matched_targets.sort(key=match_score, reverse=True)

        # Check for ambiguous matches (same method and IoU)
        if len(matched_targets) > 1:
            best_method = matched_targets[0][1]
            best_iou = matched_targets[0][2]

            # Count how many have the same score as the best
            ambiguous_matches = [
                m for m in matched_targets
                if m[1] == best_method and abs(m[2] - best_iou) < 0.01
            ]

            if len(ambiguous_matches) > 1:
                # Ambiguous match - cannot determine correct target
                ambiguous_names = [m[0].name for m in ambiguous_matches]
                raise ValueError(
                    f"Ambiguous match for {source_file.name}: "
                    f"Multiple targets with identical bounds and IoU ({best_iou:.2f}%): "
                    f"{', '.join(ambiguous_names)}. "
                    f"Cannot determine correct target file."
                )

        target_file, match_method, best_iou = matched_targets[0]

        # Update counters
        if match_method == 'tolerance':
            tolerance_matches += 1
        else:
            iou_matches += 1

        # Extract tile_id from filename if possible, otherwise use stem
        tile_id_match = re.search(r'(c\d+_r\d+)', source_file.stem)
        if tile_id_match:
            tile_id = tile_id_match.group(1)
        else:
            # Fallback: use filename stem without extension
            tile_id = source_file.stem.replace('_segmented', '').replace('_results', '')

        matches.append((source_file, target_file, tile_id))

        if verbose:
            if match_method == 'tolerance':
                method_str = "1m tolerance"
            else:
                method_str = f"IoU {best_iou:.1f}%"
            print(f"  ✓ Matched: {source_file.name} <-> {target_file.name} ({method_str})")
        else:
            print(f"  Matched: {source_file.name} <-> {target_file.name}")

    # Summary
    print()
    if tolerance_matches > 0:
        print(f"  ✓ {tolerance_matches} file(s) matched by tolerance (1m)")
    if iou_matches > 0:
        print(f"  ✓ {iou_matches} file(s) matched by IoU fallback (30%)")
    if unmatched_count > 0:
        print(f"  ⚠ {unmatched_count} file(s) could not be matched")

    return matches

