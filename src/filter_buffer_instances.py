#!/usr/bin/env python3
"""
Buffer Filter Preprocessing Script

Removes instances (and all their points) whose centroids are in buffer zones
facing neighboring tiles. This is a preprocessing step that should run before merging.

Usage:
    python filter_buffer_instances.py \
        --input-dir /path/to/segmented_remapped \
        --output-dir /path/to/filtered \
        --buffer 10.0
"""

import argparse
import shutil
import numpy as np
import laspy
from laspy.vlrs.vlrlist import VLRList
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional

from filter_task_support import classify_tree_sidecar_file, is_tree_sidecar_file
from instance_labels import validate_prediction_instance_labels


def get_tile_neighbors(tile_name: str, all_tile_names: List[str]) -> Dict[str, bool]:
    """
    Determine which edges of a tile have neighbors.
    Returns dict with 'east', 'west', 'north', 'south' boolean values.
    """
    # Parse tile name format: c{col}_r{row}. Non-grid filenames can appear
    # when the filter task is used on a single tile; treat them as edge tiles.
    try:
        parts = tile_name.split('_')
        col_str = parts[0][1:]  # Extract number after 'c'
        row_str = parts[1][1:]  # Extract number after 'r'
        col = int(col_str)
        row = int(row_str)
    except (IndexError, ValueError):
        return {'east': False, 'west': False, 'north': False, 'south': False}

    col_padding = len(col_str)
    row_padding = len(row_str)

    def format_tile_name(c, r):
        return f"c{str(c).zfill(col_padding)}_r{str(r).zfill(row_padding)}"

    return {
        'east': format_tile_name(col+1, row) in all_tile_names,
        'west': col > 0 and format_tile_name(col-1, row) in all_tile_names,
        'north': format_tile_name(col, row+1) in all_tile_names,
        'south': row > 0 and format_tile_name(col, row-1) in all_tile_names,
    }


def _tile_base_name(path: Path) -> str:
    """Return the tile grid id from a LAZ/LAS filename."""
    name = path.stem
    if name.endswith(".copc"):
        name = name[:-5]
    for suffix in ['_segmented_remapped', '_segmented', '_remapped', '_filtered']:
        name = name.replace(suffix, '')
    return name


def _input_point_clouds(input_dir: Path) -> List[Path]:
    """Return all filterable point-cloud files in stable order."""
    files = []
    seen = set()
    for pattern in ("*.laz", "*.las", "*.LAZ", "*.LAS"):
        for path in sorted(input_dir.glob(pattern)):
            key = path.resolve()
            if key in seen:
                continue
            seen.add(key)
            files.append(path)
    return sorted(files, key=lambda p: p.name.lower())


def _input_tree_sidecars(input_dir: Path) -> List[Path]:
    """Return recognized tree sidecar text files in stable order."""
    return sorted(
        [
            path
            for path in input_dir.glob("*.txt")
            if is_tree_sidecar_file(path)
        ],
        key=lambda p: p.name.lower(),
    )


def _filtered_tree_sidecar_name(path: Path, suffix: str) -> str:
    """Return the filtered sidecar filename matching the point-cloud suffix policy."""
    sidecar_suffix = classify_tree_sidecar_file(path) or "_trees.txt"
    stem = path.name[:-len(sidecar_suffix)] if path.name.lower().endswith(sidecar_suffix) else path.stem
    if suffix and not stem.endswith(suffix):
        stem = f"{stem}{suffix}"
    return f"{stem}{sidecar_suffix}"


def _copy_filtered_output_header(source_header: laspy.LasHeader) -> laspy.LasHeader:
    """Copy source metadata for a filtered LAZ output."""
    header = source_header.copy()
    header.point_count = 0

    def is_stale_copc_vlr(vlr) -> bool:
        return getattr(vlr, "user_id", "") == "copc" and getattr(vlr, "record_id", None) in (1, 2)

    header.vlrs = VLRList([vlr for vlr in header.vlrs if not is_stale_copc_vlr(vlr)])
    if getattr(header, "evlrs", None) is not None:
        header.evlrs = VLRList([vlr for vlr in header.evlrs if not is_stale_copc_vlr(vlr)])
    return header


def _write_las_points(
    source_las: laspy.LasData,
    output_file: Path,
    keep_mask: Optional[np.ndarray] = None,
) -> None:
    """Write selected source points with stale COPC VLRs stripped."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    header = _copy_filtered_output_header(source_las.header)
    output_las = laspy.LasData(header)
    if keep_mask is None:
        output_las.points = source_las.points.copy()
    else:
        output_las.points = source_las.points[keep_mask].copy()
    output_las.write(str(output_file), do_compress=True, laz_backend=laspy.LazBackend.LazrsParallel)


def compute_tile_bounds(points: np.ndarray) -> Tuple[float, float, float, float]:
    """Get the XY bounding box of a point cloud."""
    return (
        points[:, 0].min(),
        points[:, 0].max(),
        points[:, 1].min(),
        points[:, 1].max()
    )


def get_instances_to_remove(
    points: np.ndarray,
    instances: np.ndarray,
    boundary: Tuple[float, float, float, float],
    tile_name: str,
    all_tile_names: List[str],
    buffer: float = 10.0,
) -> Set[int]:
    """
    Find instances whose centroid is in the buffer zone on inner edges.

    Args:
        points: Nx3 array of point coordinates
        instances: Array of instance IDs
        boundary: (min_x, max_x, min_y, max_y) of the tile
        tile_name: Name of the tile (e.g., "c00_r00")
        all_tile_names: List of all tile names to determine neighbors
        buffer: Buffer distance from inner edges

    Returns:
        Set of instance IDs to REMOVE (centroid in buffer zone)
    """
    min_x, max_x, min_y, max_y = boundary

    # Determine which edges have neighbors
    neighbors = get_tile_neighbors(tile_name, all_tile_names)

    # Calculate tile dimensions and cap buffer
    tile_width = max_x - min_x
    tile_height = max_y - min_y
    min_dimension = min(tile_width, tile_height)
    actual_buffer = min(buffer, min_dimension * 0.4)
    actual_buffer = max(actual_buffer, 2.0)

    # Define buffer zone boundaries (only on inner edges)
    buf_min_x = min_x + (actual_buffer if neighbors['west'] else 0)
    buf_max_x = max_x - (actual_buffer if neighbors['east'] else 0)
    buf_min_y = min_y + (actual_buffer if neighbors['south'] else 0)
    buf_max_y = max_y - (actual_buffer if neighbors['north'] else 0)

    # Find instances to remove
    instances_to_remove = set()
    unique_ids = np.unique(instances)

    for inst_id in unique_ids:
        if inst_id <= 0:
            continue

        mask = instances == inst_id
        inst_points = points[mask]

        # Calculate centroid (XYZ)
        centroid = np.mean(inst_points, axis=0)
        cx, cy = centroid[0], centroid[1]

        # Check if centroid is in buffer zone
        in_west_buffer = neighbors['west'] and cx < buf_min_x
        in_east_buffer = neighbors['east'] and cx > buf_max_x
        in_south_buffer = neighbors['south'] and cy < buf_min_y
        in_north_buffer = neighbors['north'] and cy > buf_max_y

        if in_west_buffer or in_east_buffer or in_south_buffer or in_north_buffer:
            instances_to_remove.add(inst_id)

    return instances_to_remove


def process_tile(
    input_file: Path,
    output_file: Path,
    all_tile_names: List[str],
    buffer: float = 10.0,
    instance_dimension: str = "PredInstance",
) -> Tuple[int, int, int]:
    """
    Process a single tile: load, filter buffer instances, save.

    Returns:
        Tuple of (original_points, removed_points, removed_instances)
    """
    print(f"Processing {input_file.name}...")

    try:
        las = laspy.read(str(input_file), laz_backend=laspy.LazBackend.LazrsParallel)
    except Exception as e:
        print(f"  Error loading {input_file}: {e}")
        return 0, 0, 0

    points = np.vstack((
        np.array(las.x),
        np.array(las.y),
        np.array(las.z)
    )).T

    if hasattr(las, instance_dimension):
        instances = np.array(getattr(las, instance_dimension))
        validate_prediction_instance_labels(instances, instance_dimension, input_file)
    elif hasattr(las, 'treeID'):
        instances = np.array(las.treeID)
        validate_prediction_instance_labels(instances, "treeID", input_file)
    else:
        print(f"  Warning: No instance attribute ({instance_dimension}/treeID) found in {input_file}")
        _write_las_points(las, output_file)
        return len(points), 0, 0

    original_point_count = len(points)

    # Extract tile name from filename
    tile_name = _tile_base_name(input_file)

    # Compute tile boundary
    boundary = compute_tile_bounds(points)

    # Find instances to remove
    instances_to_remove = get_instances_to_remove(
        points, instances, boundary, tile_name, all_tile_names, buffer
    )

    # Create boolean mask: True = keep point, False = remove point
    keep_mask = np.ones(len(points), dtype=bool)
    for inst_id in instances_to_remove:
        keep_mask[instances == inst_id] = False

    kept_point_count = int(np.count_nonzero(keep_mask))
    removed_point_count = original_point_count - kept_point_count
    removed_instance_count = len(instances_to_remove)

    if removed_instance_count == 0:
        print(f"  {original_point_count:,} points, 0 instances removed")
        _write_las_points(las, output_file)
        return original_point_count, 0, 0

    _write_las_points(las, output_file, keep_mask=keep_mask)

    print(f"  {original_point_count:,} -> {kept_point_count:,} points "
          f"({removed_point_count:,} removed, {removed_instance_count} instances)")

    return original_point_count, removed_point_count, removed_instance_count


def filter_buffer_instances_dir(
    input_dir: Path,
    output_dir: Path,
    buffer: float = 10.0,
    suffix: str = "_filtered",
    instance_dimension: str = "PredInstance",
    output_extension: Optional[str] = None,
) -> Dict[str, object]:
    """Filter buffer instances for every LAZ/LAS file in a directory."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    point_clouds = _input_point_clouds(input_dir)
    tree_sidecars = _input_tree_sidecars(input_dir)
    if output_extension:
        output_extension = output_extension if output_extension.startswith(".") else f".{output_extension}"
        if output_extension.lower() == ".copc.laz":
            raise ValueError("filter outputs are rewritten LAS/LAZ files; use .laz instead of .copc.laz")

    if len(point_clouds) == 0:
        tree_output_files = []
        if tree_sidecars:
            output_dir.mkdir(parents=True, exist_ok=True)
            for tree_file in tree_sidecars:
                tree_output_file = output_dir / _filtered_tree_sidecar_name(tree_file, suffix)
                shutil.copy2(tree_file, tree_output_file)
                tree_output_files.append(tree_output_file)
                print(f"Copied tree sidecar: {tree_file.name} -> {tree_output_file.name}")
        print(f"No LAZ/LAS files found in {input_dir}")
        return {
            "input_files": 0,
            "output_files": [],
            "tree_files": tree_sidecars,
            "tree_output_files": tree_output_files,
            "total_original": 0,
            "total_removed": 0,
            "total_instances_removed": 0,
        }

    print("=" * 60)
    print("Buffer Filter Preprocessing")
    print("=" * 60)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Buffer: {buffer}m")
    print(f"Instance dimension: {instance_dimension}")
    print(f"Found {len(point_clouds)} tiles to process")
    if tree_sidecars:
        print(f"Found {len(tree_sidecars)} tree sidecar file(s) to copy")
    print("=" * 60)

    all_tile_names = [_tile_base_name(path) for path in point_clouds]
    total_original = 0
    total_removed = 0
    total_instances_removed = 0
    output_files = []
    tree_output_files = []

    for input_file in point_clouds:
        extension = output_extension or input_file.suffix
        output_file = output_dir / f"{input_file.stem}{suffix}{extension}"
        output_files.append(output_file)

        orig, removed, inst_removed = process_tile(
            input_file,
            output_file,
            all_tile_names,
            buffer,
            instance_dimension=instance_dimension,
        )

        total_original += orig
        total_removed += removed
        total_instances_removed += inst_removed

    if tree_sidecars:
        output_dir.mkdir(parents=True, exist_ok=True)
        for tree_file in tree_sidecars:
            tree_output_file = output_dir / _filtered_tree_sidecar_name(tree_file, suffix)
            shutil.copy2(tree_file, tree_output_file)
            tree_output_files.append(tree_output_file)
            print(f"Copied tree sidecar: {tree_file.name} -> {tree_output_file.name}")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total points: {total_original:,} -> {total_original - total_removed:,} "
          f"({total_removed:,} removed, {100*total_removed/max(total_original,1):.1f}%)")
    print(f"Total instances removed: {total_instances_removed}")
    print("=" * 60)

    return {
        "input_files": len(point_clouds),
        "output_files": output_files,
        "tree_files": tree_sidecars,
        "tree_output_files": tree_output_files,
        "total_original": total_original,
        "total_removed": total_removed,
        "total_instances_removed": total_instances_removed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Filter buffer zone instances - Remove instances whose centroids are in buffer zones",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input-dir", "-i",
        type=Path,
        required=True,
        help="Directory containing input LAZ tiles"
    )

    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        required=True,
        help="Output directory for filtered LAZ files"
    )

    parser.add_argument(
        "--buffer",
        type=float,
        default=10.0,
        help="Buffer zone distance in meters (default: 10.0)"
    )

    parser.add_argument(
        "--suffix",
        type=str,
        default="_filtered",
        help="Suffix to add to output filenames (default: '_filtered')"
    )

    parser.add_argument(
        "--instance-dimension",
        type=str,
        default="PredInstance",
        help="Instance dimension to filter (default: PredInstance, fallback: treeID)"
    )

    parser.add_argument(
        "--output-extension",
        type=str,
        default=None,
        help="Optional output extension override, e.g. .laz for Galaxy collection discovery"
    )

    args = parser.parse_args()
    filter_buffer_instances_dir(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        buffer=args.buffer,
        suffix=args.suffix,
        instance_dimension=args.instance_dimension,
        output_extension=args.output_extension,
    )


if __name__ == "__main__":
    main()
