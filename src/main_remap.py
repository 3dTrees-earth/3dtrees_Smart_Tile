#!/usr/bin/env python3
"""
Main remap script: Remap predictions from 10cm to target resolution (default: 2cm).

This script handles remapping of segmented predictions from coarse resolution
back to finer resolution using KDTree nearest neighbor lookup.

Usage:
    python main_remap.py --subsampled_10cm_folder /path/to/10cm --target_resolution 2
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import laspy
    from scipy.spatial import KDTree
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install laspy scipy numpy")
    sys.exit(1)

# Import parameters
try:
    from parameters import REMAP_PARAMS
except ImportError:
    REMAP_PARAMS = {
        'target_resolution_cm': 2,
        'workers': 4,
    }


def remap_single_tile(
    segmented_file: Path,
    target_file: Path,
    output_file: Path,
    workers: int = -1
) -> Tuple[str, bool, str, int]:
    """
    Remap predictions from segmented file to target resolution file.
    
    Uses KDTree nearest neighbor search to transfer attributes from
    the segmented (coarse) file to the target (fine) file.
    
    Args:
        segmented_file: Path to segmented LAZ file (e.g., 10cm with predictions)
        target_file: Path to target resolution LAZ file (e.g., 2cm)
        output_file: Path for output LAZ file
        workers: Number of workers for KDTree queries (-1 = all CPUs)
    
    Returns:
        Tuple of (tile_id, success, message, point_count)
    """
    tile_id = segmented_file.stem.replace('_segmented', '').replace('_results', '')
    
    try:
        # Load segmented point cloud (source of predictions)
        segmented_las = laspy.read(
            str(segmented_file), 
            laz_backend=laspy.LazBackend.LazrsParallel
        )
        segmented_points = np.vstack((
            segmented_las.x, 
            segmented_las.y, 
            segmented_las.z
        )).T
        
        # Load target resolution point cloud
        target_las = laspy.read(
            str(target_file), 
            laz_backend=laspy.LazBackend.LazrsParallel
        )
        target_points = np.vstack((
            target_las.x, 
            target_las.y, 
            target_las.z
        )).T
        
        # Check for required attributes in segmented file
        extra_dims = {dim.name for dim in segmented_las.point_format.extra_dimensions}
        has_pred_instance = 'PredInstance' in extra_dims
        has_pred_semantic = 'PredSemantic' in extra_dims
        has_species_id = 'species_id' in extra_dims
        
        if not has_pred_instance:
            return (tile_id, False, "No PredInstance attribute in segmented file", 0)
        
        # Create KDTree from segmented points
        tree = KDTree(segmented_points)
        
        # Query nearest neighbors
        distances, indices = tree.query(target_points, workers=-1)
        
        # Create output with target resolution points
        # Add extra dimensions if they don't exist
        target_extra_dims = {dim.name for dim in target_las.point_format.extra_dimensions}
        
        if "PredInstance" not in target_extra_dims:
            target_las.add_extra_dim(
                laspy.ExtraBytesParams(name="PredInstance", type=np.int32)
            )
        
        if has_pred_semantic and "PredSemantic" not in target_extra_dims:
            target_las.add_extra_dim(
                laspy.ExtraBytesParams(name="PredSemantic", type=np.int32)
            )
        
        if has_species_id and "species_id" not in target_extra_dims:
            target_las.add_extra_dim(
                laspy.ExtraBytesParams(name="species_id", type=np.int32)
            )
        
        # Transfer attributes using nearest neighbor indices
        target_las.PredInstance = segmented_las.PredInstance[indices]
        
        if has_pred_semantic:
            target_las.PredSemantic = segmented_las.PredSemantic[indices]
        
        if has_species_id:
            target_las.species_id = segmented_las.species_id[indices]
        
        # Create output directory if needed
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save output
        with open(str(output_file), "wb") as f:
            target_las.write(
                f, 
                do_compress=True, 
                laz_backend=laspy.LazBackend.LazrsParallel
            )
            f.flush()
            os.fsync(f.fileno())
        
        return (tile_id, True, "Success", len(target_points))
        
    except Exception as e:
        return (tile_id, False, str(e), 0)


def calculate_spatial_overlap(
    source_file: Path,
    target_file: Path,
    tolerance: float = 5.0
) -> Dict[str, float]:
    """
    Calculate spatial overlap between two LAZ files.
    
    This function computes detailed spatial overlap metrics including:
    - Bounds differences
    - Center distance
    - Overlap area
    - Overlap percentage
    
    Args:
        source_file: Path to source LAZ file (e.g., segmented file)
        target_file: Path to target LAZ file (e.g., target resolution file)
        tolerance: Tolerance for bounds matching in meters (default: 5.0)
    
    Returns:
        Dictionary with overlap metrics:
        - 'x_min_diff': Difference in X minimum bounds (meters)
        - 'x_max_diff': Difference in X maximum bounds (meters)
        - 'y_min_diff': Difference in Y minimum bounds (meters)
        - 'y_max_diff': Difference in Y maximum bounds (meters)
        - 'center_distance': Distance between centers (meters)
        - 'x_overlap': X-axis overlap distance (meters)
        - 'y_overlap': Y-axis overlap distance (meters)
        - 'overlap_area': Overlapping area (square meters)
        - 'source_area': Source file area (square meters)
        - 'target_area': Target file area (square meters)
        - 'overlap_percentage': Percentage of overlap (0-100)
        - 'has_overlap': Boolean indicating if there's any overlap
        - 'max_bounds_diff': Maximum bounds difference (meters)
        - 'within_tolerance': Boolean indicating if within tolerance
    """
    try:
        # Read headers only (fast, no point loading)
        with laspy.open(str(source_file), laz_backend=laspy.LazBackend.LazrsParallel) as source_las:
            source_bounds = (
                source_las.header.x_min,
                source_las.header.x_max,
                source_las.header.y_min,
                source_las.header.y_max
            )
        
        with laspy.open(str(target_file), laz_backend=laspy.LazBackend.LazrsParallel) as target_las:
            target_bounds = (
                target_las.header.x_min,
                target_las.header.x_max,
                target_las.header.y_min,
                target_las.header.y_max
            )
        
        # Calculate bounds differences
        x_min_diff = abs(source_bounds[0] - target_bounds[0])
        x_max_diff = abs(source_bounds[1] - target_bounds[1])
        y_min_diff = abs(source_bounds[2] - target_bounds[2])
        y_max_diff = abs(source_bounds[3] - target_bounds[3])
        max_bounds_diff = max(x_min_diff, x_max_diff, y_min_diff, y_max_diff)
        
        # Calculate center points
        source_center_x = (source_bounds[0] + source_bounds[1]) / 2
        source_center_y = (source_bounds[2] + source_bounds[3]) / 2
        target_center_x = (target_bounds[0] + target_bounds[1]) / 2
        target_center_y = (target_bounds[2] + target_bounds[3]) / 2
        
        # Calculate center distance
        center_distance = math.sqrt(
            (source_center_x - target_center_x)**2 + 
            (source_center_y - target_center_y)**2
        )
        
        # Calculate overlap
        x_overlap = max(0, min(source_bounds[1], target_bounds[1]) - max(source_bounds[0], target_bounds[0]))
        y_overlap = max(0, min(source_bounds[3], target_bounds[3]) - max(source_bounds[2], target_bounds[2]))
        overlap_area = x_overlap * y_overlap if (x_overlap > 0 and y_overlap > 0) else 0.0
        
        # Calculate areas
        source_area = (source_bounds[1] - source_bounds[0]) * (source_bounds[3] - source_bounds[2])
        target_area = (target_bounds[1] - target_bounds[0]) * (target_bounds[3] - target_bounds[2])
        
        # Calculate overlap percentage (relative to smaller area)
        min_area = min(source_area, target_area)
        overlap_percentage = (overlap_area / min_area * 100) if min_area > 0 else 0.0
        
        has_overlap = overlap_area > 0
        within_tolerance = max_bounds_diff <= tolerance
        
        return {
            'x_min_diff': x_min_diff,
            'x_max_diff': x_max_diff,
            'y_min_diff': y_min_diff,
            'y_max_diff': y_max_diff,
            'center_distance': center_distance,
            'x_overlap': x_overlap,
            'y_overlap': y_overlap,
            'overlap_area': overlap_area,
            'source_area': source_area,
            'target_area': target_area,
            'overlap_percentage': overlap_percentage,
            'has_overlap': has_overlap,
            'max_bounds_diff': max_bounds_diff,
            'within_tolerance': within_tolerance,
            'source_bounds': source_bounds,
            'target_bounds': target_bounds
        }
    except Exception as e:
        return {
            'error': str(e),
            'has_overlap': False,
            'within_tolerance': False
        }


def find_matching_files(
    results_dir: Path,
    target_folder: Path,
    target_resolution_m: float,
    tolerance: float = 5.0
) -> List[Tuple[Path, Path, str]]:
    """
    Find matching segmented and target resolution files.
    
    Args:
        results_dir: Directory containing *_results folders with segmented_pc.laz
        target_folder: Directory containing target resolution files
        target_resolution_m: Target resolution in meters
        tolerance: Spatial bounds matching tolerance in meters (default: 5.0)
        use_spatial_matching: If True, verify spatial bounds match within tolerance
    
    Returns:
        List of (segmented_file, target_file, tile_id) tuples
    """
    matches = []
    failed_matches = []
    
    # Find all results directories
    results_dirs = sorted(results_dir.glob("*_results"))
    
    for result_dir in results_dirs:
        # Get tile ID from directory name (e.g., c00_r00)
        dir_name = result_dir.name
        match = re.search(r'(c\d+_r\d+)', dir_name)
        if not match:
            continue
        
        tile_id = match.group(1)
        
        # Path to segmented file
        segmented_file = result_dir / "segmented_pc.laz"
        if not segmented_file.exists():
            continue
        
        # Find matching target resolution file
        # Try multiple naming patterns
        patterns = [
            f"{tile_id}.copc_subsampled{target_resolution_m}m.laz",
            f"{tile_id}*_subsampled{target_resolution_m}m.laz",
            f"*{tile_id}*.laz",
        ]
        
        target_file = None
        for pattern in patterns:
            matches_found = list(target_folder.glob(pattern))
            if matches_found:
                target_file = matches_found[0]
                break
        
        if target_file and target_file.exists():
            # Always verify spatial bounds match
            overlap_info = calculate_spatial_overlap(segmented_file, target_file, tolerance)
            
            if overlap_info.get('error'):
                print(f"  Warning: Could not calculate overlap for {tile_id}: {overlap_info['error']}")
                matches.append((segmented_file, target_file, tile_id))
            elif not overlap_info.get('within_tolerance', False):
                # Matching failed - store for detailed reporting
                failed_matches.append((segmented_file, target_file, tile_id, overlap_info))
                print(f"  Warning: Spatial bounds mismatch for {tile_id}")
                print(f"    Max bounds difference: {overlap_info['max_bounds_diff']:.3f}m (tolerance: {tolerance}m)")
                print(f"    Center distance: {overlap_info['center_distance']:.3f}m")
                print(f"    Overlap: {overlap_info['overlap_area']:.2f}m² ({overlap_info['overlap_percentage']:.1f}%)")
            else:
                matches.append((segmented_file, target_file, tile_id))
    
    # If we have failed matches, print detailed report
    if failed_matches:
        print("\n" + "=" * 60)
        print("Spatial Matching Failures (Detailed Analysis)")
        print("=" * 60)
        for seg_file, tgt_file, tile_id, overlap_info in failed_matches:
            print(f"\nTile: {tile_id}")
            print(f"  Source: {seg_file.name}")
            print(f"  Target: {tgt_file.name}")
            print(f"  Bounds differences:")
            print(f"    X_min: {overlap_info['x_min_diff']:.3f}m")
            print(f"    X_max: {overlap_info['x_max_diff']:.3f}m")
            print(f"    Y_min: {overlap_info['y_min_diff']:.3f}m")
            print(f"    Y_max: {overlap_info['y_max_diff']:.3f}m")
            print(f"  Max difference: {overlap_info['max_bounds_diff']:.3f}m (tolerance: {tolerance}m)")
            print(f"  Center distance: {overlap_info['center_distance']:.3f}m")
            print(f"  Spatial overlap:")
            print(f"    X overlap: {overlap_info['x_overlap']:.3f}m")
            print(f"    Y overlap: {overlap_info['y_overlap']:.3f}m")
            print(f"    Overlap area: {overlap_info['overlap_area']:.2f}m²")
            print(f"    Overlap percentage: {overlap_info['overlap_percentage']:.1f}%")
            print(f"  Source bounds: X[{overlap_info['source_bounds'][0]:.3f}, {overlap_info['source_bounds'][1]:.3f}], "
                  f"Y[{overlap_info['source_bounds'][2]:.3f}, {overlap_info['source_bounds'][3]:.3f}]")
            print(f"  Target bounds: X[{overlap_info['target_bounds'][0]:.3f}, {overlap_info['target_bounds'][1]:.3f}], "
                  f"Y[{overlap_info['target_bounds'][2]:.3f}, {overlap_info['target_bounds'][3]:.3f}]")
        print("\n" + "=" * 60)
    
    return matches


def remap_all_tiles(
    subsampled_10cm_dir: Path,
    target_resolution_cm: int = 2,
    subsampled_target_folder: Optional[Path] = None,
    output_folder: Optional[Path] = None,
    num_threads: int = 4
) -> Path:
    """
    Remap predictions from 10cm to target resolution for all tiles.
    
    Args:
        subsampled_10cm_dir: Path to folder containing *_results directories
        target_resolution_cm: Target resolution in cm (default: 2)
        subsampled_target_folder: Path to target resolution folder (auto-derived if None)
        output_folder: Output folder for remapped files (auto-derived if None)
        num_threads: Number of workers for KDTree queries
    
    Returns:
        Path to output folder
    """
    print("=" * 60)
    print("3DTrees Remap Pipeline")
    print("=" * 60)
    
    # Auto-derive paths if not provided
    if subsampled_target_folder is None:
        # Replace "subsampled_10cm" with "subsampled_{target}cm"
        folder_name = subsampled_10cm_dir.name
        new_name = folder_name.replace("10cm", f"{target_resolution_cm}cm")
        subsampled_target_folder = subsampled_10cm_dir.parent / new_name
    
    if output_folder is None:
        # Create segmented_remapped folder at same level
        output_folder = subsampled_10cm_dir.parent / "segmented_remapped"
    
    print(f"Input (10cm): {subsampled_10cm_dir}")
    print(f"Target ({target_resolution_cm}cm): {subsampled_target_folder}")
    print(f"Output: {output_folder}")
    print(f"Workers: {num_threads}")
    print()
    
    # Validate directories exist
    if not subsampled_10cm_dir.exists():
        raise ValueError(f"Input directory not found: {subsampled_10cm_dir}")
    
    if not subsampled_target_folder.exists():
        raise ValueError(f"Target directory not found: {subsampled_target_folder}")
    
    # Create output directory
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Convert resolution to meters
    target_resolution_m = target_resolution_cm / 100.0
    
    # Find matching files
    # Internal tolerance for bounds matching (not exposed as parameter)
    bounds_tolerance = 5.0
    print("Matching files by spatial bounds...")
    matches = find_matching_files(
        subsampled_10cm_dir, 
        subsampled_target_folder, 
        target_resolution_m,
        tolerance=bounds_tolerance
    )
    
    if not matches:
        # Try alternative: direct LAZ files in the 10cm folder
        print("  Looking for direct LAZ files...")
        segmented_files = list(subsampled_10cm_dir.glob("*_segmented*.laz"))
        
        for seg_file in segmented_files:
            tile_id_match = re.search(r'(c\d+_r\d+)', seg_file.name)
            if not tile_id_match:
                continue
            tile_id = tile_id_match.group(1)
            
            # Find target file
            target_patterns = [
                f"*{tile_id}*.laz",
            ]
            
            for pattern in target_patterns:
                target_files = list(subsampled_target_folder.glob(pattern))
                if target_files:
                    target_file = target_files[0]
                    
                    # Always verify spatial bounds - if matching fails, show overlap analysis
                    overlap_info = calculate_spatial_overlap(seg_file, target_file, bounds_tolerance)
                    if overlap_info.get('within_tolerance', False):
                        matches.append((seg_file, target_file, tile_id))
                    else:
                        # Bounds mismatch - show warning but continue to fallback analysis
                        print(f"  Warning: Spatial bounds mismatch for {tile_id}")
                        print(f"    Max bounds difference: {overlap_info['max_bounds_diff']:.3f}m (tolerance: {bounds_tolerance}m)")
                        print(f"    Overlap: {overlap_info['overlap_area']:.2f}m² ({overlap_info['overlap_percentage']:.1f}%)")
                    break
    
    if not matches:
        # Calculate spatial overlap for all potential pairs before failing
        print("\n" + "=" * 60)
        print("No matching files found. Analyzing spatial overlap for all potential pairs...")
        print("=" * 60)
        
        segmented_files = list(subsampled_10cm_dir.glob("*_segmented*.laz"))
        if not segmented_files:
            segmented_files = [f for d in subsampled_10cm_dir.glob("*_results") for f in [(d / "segmented_pc.laz")] if f.exists()]
        
        target_files = list(subsampled_target_folder.glob("*.laz"))
        
        print(f"  Found {len(segmented_files)} segmented files and {len(target_files)} target files")
        
        if segmented_files and target_files:
            # Try to match by filename first, then check spatial overlap
            for seg_file in segmented_files:
                tile_id_match = re.search(r'(c\d+_r\d+)', seg_file.name)
                if not tile_id_match:
                    continue
                tile_id = tile_id_match.group(1)
                
                # Find potential target files
                potential_targets = [f for f in target_files if tile_id in f.name]
                
                if potential_targets:
                    print(f"\n  Analyzing tile {tile_id}:")
                    for tgt_file in potential_targets:
                        overlap_info = calculate_spatial_overlap(seg_file, tgt_file, bounds_tolerance)
                        if 'error' not in overlap_info:
                            print(f"    {tgt_file.name}:")
                            print(f"      Max bounds diff: {overlap_info['max_bounds_diff']:.3f}m (tolerance: {bounds_tolerance}m)")
                            print(f"      Center distance: {overlap_info['center_distance']:.3f}m")
                            print(f"      Overlap: {overlap_info['overlap_area']:.2f}m² ({overlap_info['overlap_percentage']:.1f}%)")
                            print(f"      Status: {'✓ Within tolerance' if overlap_info['within_tolerance'] else '✗ Exceeds tolerance'}")
                        else:
                            print(f"    {tgt_file.name}: Error - {overlap_info['error']}")
        
        raise ValueError(f"No matching source/target file pairs found (tolerance: {bounds_tolerance}m)")
    
    print(f"Found {len(matches)} tiles to remap")
    print()
    
    # Process each tile
    successful = 0
    failed = 0
    total_points = 0
    
    for i, (segmented_file, target_file, tile_id) in enumerate(matches, 1):
        print(f"[{i}/{len(matches)}] Processing tile {tile_id}...")
        
        output_file = output_folder / f"{tile_id}_segmented_remapped.laz"
        
        # Skip if already exists
        if output_file.exists() and output_file.stat().st_size > 0:
            print(f"  Skipping (already exists)")
            successful += 1
            continue
        
        tile_id_result, success, message, point_count = remap_single_tile(
            segmented_file,
            target_file,
            output_file,
            workers=num_threads
        )
        
        if success:
            successful += 1
            total_points += point_count
            print(f"  ✓ {point_count:,} points")
        else:
            failed += 1
            print(f"  ✗ {message}")
    
    # Summary
    print()
    print("=" * 60)
    print("Remap Pipeline Complete")
    print("=" * 60)
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total points: {total_points:,}")
    print(f"  Output: {output_folder}")
    
    return output_folder


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="3DTrees Remap Pipeline - Remap predictions to target resolution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--subsampled_10cm_folder",
        type=Path,
        required=True,
        help="Path to folder containing *_results directories or segmented LAZ files"
    )
    
    parser.add_argument(
        "--target_resolution",
        type=int,
        default=REMAP_PARAMS.get('target_resolution_cm', 2),
        help=f"Target resolution in cm (default: {REMAP_PARAMS.get('target_resolution_cm', 2)})"
    )
    
    parser.add_argument(
        "--subsampled_target_folder",
        type=Path,
        default=None,
        help="Path to target resolution folder (auto-derived if not specified)"
    )
    
    parser.add_argument(
        "--output_folder",
        type=Path,
        default=None,
        help="Output folder for remapped files (auto-derived if not specified)"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=REMAP_PARAMS.get('workers', 4),
        help=f"Number of workers for KDTree queries (default: {REMAP_PARAMS.get('workers', 4)})"
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    try:
        output_folder = remap_all_tiles(
            subsampled_10cm_dir=args.subsampled_10cm_folder,
            target_resolution_cm=args.target_resolution,
            subsampled_target_folder=args.subsampled_target_folder,
            output_folder=args.output_folder,
            num_threads=args.workers
        )
        print(f"\nRemapped files ready: {output_folder}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

