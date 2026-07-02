#!/usr/bin/env python3
"""Command-line parser for the SmartTile merge task."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(merge_tiles_func):
    parser = argparse.ArgumentParser(
        description="Tile Merger - Merge segmented point cloud tiles with species ID preservation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input-dir",
        "-i",
        type=Path,
        required=True,
        help="Directory containing segmented LAZ tiles",
    )

    parser.add_argument(
        "--original-tiles-dir",
        type=Path,
        default=None,
        help="Directory containing original tile files for retiling",
    )

    parser.add_argument(
        "--output-merged",
        "-o",
        type=Path,
        required=True,
        help="Output path for merged LAZ file",
    )

    parser.add_argument(
        "--output-tiles-dir",
        type=Path,
        default=None,
        help="Output directory for retiled files",
    )

    parser.add_argument(
        "--original-input-dir",
        type=Path,
        default=None,
        help="Directory with original input LAZ files for final remap (optional, enables Stage 7)",
    )

    parser.add_argument(
        "--buffer",
        type=float,
        default=10.0,
        help="Buffer zone distance in meters (default: 10.0)",
    )

    parser.add_argument(
        "--overlap-threshold",
        "--ff3d-threshold",
        type=float,
        default=0.3,
        dest="overlap_threshold",
        help="Overlap ratio threshold for instance matching (default: 0.3 = 30%%)",
    )

    parser.add_argument(
        "--correspondence-tolerance",
        type=float,
        default=0.1,
        help="Max distance for point correspondence in meters (default: 0.1). "
        "Should be small (~10cm) to only match actual duplicate points from overlapping tiles.",
    )

    parser.add_argument(
        "--max-volume-for-merge",
        type=float,
        default=4.0,
        help="Max convex hull volume (m³) for small instance merging (default: 4.0)",
    )

    parser.add_argument(
        "--border-zone-width",
        type=float,
        default=10.0,
        help="Width of border zone beyond buffer for instance matching (default: 10.0m)",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        dest="num_threads",
        help="Number of workers for parallel processing (default: 4)",
    )

    parser.add_argument(
        "--disable-matching",
        "--disable-ff3d",
        action="store_true",
        dest="disable_matching",
        help="Disable cross-tile instance matching",
    )

    parser.add_argument(
        "--disable-volume-merge",
        action="store_true",
        help="Disable small volume instance merging",
    )

    parser.add_argument(
        "--skip-merged-file",
        action="store_true",
        help="Skip creating merged LAZ file (only create retiled outputs)",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print detailed merge decisions"
    )

    parser.add_argument(
        "--debug-instances",
        type=str,
        default=None,
        help="Comma-separated list of instance IDs to debug (e.g., '485,73'). Enables detailed logging for these instances in Stage 3.",
    )

    parser.add_argument(
        "--match-all-instances",
        action="store_true",
        dest="match_all_instances",
        help="Match all instances between neighbor tiles, not just border region instances. "
        "When enabled, Stage 3 will check all instances in overlapping tiles for matching, "
        "not just those in border regions. Default: False (only border region instances are matched).",
    )

    parser.add_argument(
        "--retile-buffer",
        type=float,
        default=2.0,
        help="Spatial buffer expansion in meters for filtering merged points during retiling (fixed: 2.0m)",
    )

    parser.add_argument(
        "--retile-max-radius",
        type=float,
        default=0.2,
        help="Maximum distance threshold in meters for cKDTree nearest neighbor matching during retiling (default: 2.0m)",
    )

    parser.add_argument(
        "--instance-dimension",
        type=str,
        default="PredInstance",
        help="Name of the instance ID dimension in input files (default: PredInstance, fallback: treeID)",
    )

    parser.add_argument(
        "--tile-bounds-json",
        type=Path,
        required=True,
        help="Path to tile_bounds_tindex.json (required; used for neighbor graph)",
    )

    args = parser.parse_args()

    # Parse debug instance IDs
    debug_instance_ids = None
    if args.debug_instances:
        try:
            debug_instance_ids = set(int(x.strip()) for x in args.debug_instances.split(','))
        except ValueError:
            print(f"ERROR: Invalid --debug-instances format: {args.debug_instances}")
            print("Expected format: comma-separated integers (e.g., '485,73')")
            sys.exit(1)

    merge_tiles_func(
        input_dir=args.input_dir,
        original_tiles_dir=args.original_tiles_dir,
        output_merged=args.output_merged,
        output_tiles_dir=args.output_tiles_dir,
        tile_bounds_json=args.tile_bounds_json,
        original_input_dir=args.original_input_dir,
        buffer=args.buffer,
        overlap_threshold=args.overlap_threshold,
        correspondence_tolerance=args.correspondence_tolerance,
        max_volume_for_merge=args.max_volume_for_merge,
        border_zone_width=args.border_zone_width,
        num_threads=args.num_threads,
        enable_matching=not args.disable_matching,
        enable_volume_merge=not args.disable_volume_merge,
        skip_merged_file=args.skip_merged_file,
        verbose=args.verbose,
        retile_buffer=args.retile_buffer,
        retile_max_radius=args.retile_max_radius,
        debug_instance_ids=debug_instance_ids,
        match_all_instances=args.match_all_instances,
        instance_dimension=args.instance_dimension,
    )


