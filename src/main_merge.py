#!/usr/bin/env python3
"""Compatibility CLI for SmartTile's strict remap-first merge pipeline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from parameters import MERGE_PARAMS
from point_cloud_metadata import point_cloud_files as _point_cloud_files
from worker_budget import DEFAULT_MEMORY_GB


def run_merge(
    segmented_dir: Path,
    output_tiles_dir: Path,
    original_tiles_dir: Path,
    tile_bounds_json: Path,
    original_input_dir: Optional[Path] = None,
    output_merged: Optional[Path] = None,
    overlap_threshold: float = 0.3,
    max_centroid_distance: float = 3.0,
    correspondence_tolerance: float = 0.05,
    max_volume_for_merge: float = 4.0,
    border_zone_width: float = 10.0,
    min_cluster_size: int = 300,
    num_threads: int = 4,
    enable_matching: bool = True,
    require_overlap: bool = True,
    enable_volume_merge: bool = True,
    skip_merged_file: bool = False,
    verbose: bool = True,
    retile_buffer: float = 2.0,  # Fixed to 2.0m
    retile_max_radius: float = 0.1,
    instance_dimension: str = "PredInstance",
    transfer_original_dims_to_merged: bool = True,
    threedtrees_dims: Optional[List[str]] = None,
    threedtrees_suffix: str = "SAT",
    chunk_size: int = 1_000_000,
    memory_gb: float = DEFAULT_MEMORY_GB,
    filter_anchor: str = "centroid",
) -> Path:
    """Run the strict remap-first pipeline (3DT-2101).

    Legacy tuning arguments remain accepted by this Python/CLI adapter. Dense
    geometry uses core ownership filtering before reconciliation; small-cluster
    reassignment stays disabled. Final original coverage is 100% within the first-stage voxel diagonal.
    """
    from strict_prediction_pipeline import merge_collections

    if memory_gb <= 0:
        raise ValueError("memory_gb must be greater than 0")
    if tile_bounds_json is None or not Path(tile_bounds_json).is_file():
        raise ValueError("tile_bounds_json is required")
    source = Path(segmented_dir)
    target = Path(original_tiles_dir) if original_tiles_dir else None
    ready = target is None or not _point_cloud_files(target)
    merged = Path(output_merged or Path(output_tiles_dir).parent / "merged.laz")
    merge_collections(
        collections=[source], target_dir=None if ready else target,
        output_tiles=Path(output_tiles_dir), tile_bounds_json=Path(tile_bounds_json),
        originals=Path(original_input_dir) if original_input_dir else None,
        merged_output=None if skip_merged_file else merged,
        filter_anchor=filter_anchor, workers=num_threads,
        overlap_threshold=overlap_threshold, correspondence_radius=correspondence_tolerance,
        ready=ready, matching=enable_matching, instance_dimension=instance_dimension,
        target_dims=set(threedtrees_dims) if threedtrees_dims else None,
    )
    return merged


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="3DTrees Merge Pipeline - Merge segmented tiles with instance matching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--segmented_dir", "--segmented_folder", "-i",
        type=Path,
        required=True,
        dest="segmented_dir",
        help="Directory containing segmented LAZ tiles"
    )

    parser.add_argument(
        "--output_merged", "-o",
        type=Path,
        default=None,
        help="Output path for merged LAZ file (auto-derived if not specified)"
    )

    parser.add_argument(
        "--output_tiles_dir",
        type=Path,
        required=True,
        help="Output directory for retiled files (required)"
    )

    parser.add_argument(
        "--original_tiles_dir",
        type=Path,
        required=True,
        help="Directory with original tile files for retiling (required)"
    )

    parser.add_argument(
        "--tile_bounds_json",
        type=Path,
        required=True,
        help="Path to tile_bounds_tindex.json (required; used for neighbor graph)"
    )

    parser.add_argument(
        "--original_input_dir",
        type=Path,
        default=None,
        help="Directory with original input LAZ files for final remap (optional, enables Stage 7)"
    )

    parser.add_argument(
        "--overlap_threshold",
        type=float,
        default=MERGE_PARAMS.get('overlap_threshold', 0.3),
        help=f"Overlap ratio threshold (default: {MERGE_PARAMS.get('overlap_threshold', 0.3)})"
    )

    parser.add_argument(
        "--max_centroid_distance",
        type=float,
        default=MERGE_PARAMS.get('max_centroid_distance', 3.0),
        help=f"Max centroid distance (default: {MERGE_PARAMS.get('max_centroid_distance', 3.0)})"
    )

    parser.add_argument(
        "--correspondence_tolerance",
        type=float,
        default=0.05,
        help="Max distance for point correspondence during merge (default: 0.05m)"
    )

    parser.add_argument(
        "--max_volume_for_merge",
        type=float,
        default=MERGE_PARAMS.get('max_volume_for_merge', 4.0),
        help=f"Max volume for small instance merge (default: {MERGE_PARAMS.get('max_volume_for_merge', 4.0)})"
    )

    parser.add_argument(
        "--min_cluster_size",
        type=int,
        default=MERGE_PARAMS.get('min_cluster_size', 300),
        help=f"Minimum cluster size in points for reassignment (default: {MERGE_PARAMS.get('min_cluster_size', 300)})"
    )

    parser.add_argument(
        "--num_threads", "--workers",
        type=int,
        default=MERGE_PARAMS.get('workers', 4),
        dest="num_threads",
        help=f"Number of workers (default: {MERGE_PARAMS.get('workers', 4)})"
    )

    parser.add_argument(
        "--border_zone_width",
        type=float,
        default=MERGE_PARAMS.get('border_zone_width', 10.0),
        help=f"Width of border zone beyond buffer for instance matching (default: {MERGE_PARAMS.get('border_zone_width', 10.0)})"
    )

    parser.add_argument(
        "--retile_buffer",
        type=float,
        default=2.0,
        help="Spatial buffer expansion in meters for retiling (fixed: 2.0m)"
    )

    parser.add_argument(
        "--retile_max_radius",
        type=float,
        default=MERGE_PARAMS.get('retile_max_radius', 0.1),
        help=f"Max distance for nearest neighbor matching during retiling (default: {MERGE_PARAMS.get('retile_max_radius', 0.1)})"
    )

    parser.add_argument(
        "--disable_matching",
        action="store_true",
        help="Disable cross-tile instance matching"
    )

    parser.add_argument(
        "--disable_overlap_check",
        action="store_true",
        help="Disable overlap ratio check (centroid distance only)"
    )

    parser.add_argument(
        "--disable_volume_merge",
        action="store_true",
        help="Disable small volume instance merging"
    )

    parser.add_argument(
        "--skip_merged_file",
        action="store_true",
        help="Skip creating merged LAZ file (only create retiled outputs)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed merge decisions"
    )

    parser.add_argument(
        "--memory-gb",
        type=float,
        default=DEFAULT_MEMORY_GB,
        help=f"Memory cap for merge worker pools in GiB (default: {DEFAULT_MEMORY_GB})",
    )

    parser.add_argument("--filter-anchor", "--filter_anchor",
                        choices=("centroid", "highest_point", "lowest_point"), default="centroid",
                        help="Representative point for dense instance core ownership")

    args = parser.parse_args()

    # Run pipeline
    try:
        output_file = run_merge(
            segmented_dir=args.segmented_dir,
            output_tiles_dir=args.output_tiles_dir,
            original_tiles_dir=args.original_tiles_dir,
            tile_bounds_json=args.tile_bounds_json,
            original_input_dir=args.original_input_dir,
            output_merged=args.output_merged,
            overlap_threshold=args.overlap_threshold,
            max_centroid_distance=args.max_centroid_distance,
            correspondence_tolerance=args.correspondence_tolerance,
            max_volume_for_merge=args.max_volume_for_merge,
            min_cluster_size=args.min_cluster_size,
            num_threads=args.num_threads,
            enable_matching=not args.disable_matching,
            require_overlap=not args.disable_overlap_check,
            enable_volume_merge=not args.disable_volume_merge,
            skip_merged_file=args.skip_merged_file,
            verbose=args.verbose,
            border_zone_width=args.border_zone_width,
            retile_buffer=args.retile_buffer,
            retile_max_radius=args.retile_max_radius,
            memory_gb=args.memory_gb,
            filter_anchor=args.filter_anchor,
        )
        if not args.skip_merged_file:
            print(f"\nMerged output: {output_file}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
