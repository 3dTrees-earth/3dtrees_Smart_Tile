#!/usr/bin/env python3
"""Instance ID assignment, cross-tile matching, and orphan recovery for SmartTile merge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from merge_instance_ids import global_id
from merge_overlap import compute_ff3d_overlap_ratios
from merge_orphan_recovery import recover_orphaned_instances
from merge_tile_loading import TileData
from tile_spatial import (
    compute_centroids_vectorized,
    find_overlap_region,
    find_spatial_neighbors,
    get_border_region_mask,
)
from union_find import UnionFind


@dataclass
class InstanceMergeResult:
    """Outputs needed by downstream merge stages after instance matching."""

    global_to_merged: Dict[int, int]
    merged_instance_sources: Dict[int, List[int]]
    tile_idx_to_name: Dict[int, str]


def assign_and_match_instances(
    tiles: List[TileData],
    tile_boundaries: Dict[str, Tuple[float, float, float, float]],
    neighbors_by_tile: Dict[str, Dict[str, Optional[str]]],
    kept_instances_per_tile: Dict[str, Set[int]],
    filtered_instances_per_tile: Dict[str, Set[int]],
    buffer_direction_per_tile: Dict[str, Dict[int, Optional[str]]],
    buffer: float,
    border_zone_width: float,
    overlap_threshold: float,
    correspondence_tolerance: float,
    num_threads: int,
    debug_instance_ids: Optional[Set[int]] = None,
    match_all_instances: bool = False,
    verbose: bool = False,
) -> InstanceMergeResult:
    """Assign global instance IDs, match instances across tiles, and recover orphans."""
    # Stage 2: Assign Global Instance IDs
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("Stage 2: Assigning global instance IDs")
    print(f"{'=' * 60}")

    # Initialize Union-Find and track instance sizes
    uf = UnionFind()
    instance_sizes = {}  # global_id -> point count

    for tile_idx, tile in enumerate(tiles):
        print(f"  Processing tile {tile_idx + 1}/{len(tiles)}: {tile.name} ({len(tile.points):,} points)...")
        kept_instances = kept_instances_per_tile[tile.name]

        unique_inst, inst_counts = np.unique(tile.instances, return_counts=True)

        for i, local_inst in enumerate(unique_inst):
            if local_inst <= 0 or local_inst not in kept_instances:
                continue

            gid = global_id(tile_idx, local_inst)
            size = int(inst_counts[i])
            uf.make_set(gid, size)
            instance_sizes[gid] = size

    print(f"  Total global instances: {len(instance_sizes)}")
    print(f"  ✓ Stage 2 completed: Assigned global IDs to {len(instance_sizes)} instances")

    # Helper functions for border matching
    def get_opposite_direction(direction: str) -> str:
        """Get opposite direction."""
        opposites = {"east": "west", "west": "east", "north": "south", "south": "north"}
        return opposites.get(direction, direction)

    def log_instance_pair_analysis(
        inst_id_a: int,
        inst_id_b: int,
        tile_a_name: str,
        tile_b_name: str,
        direction: str,
        bbox_a: Tuple[float, float, float, float],
        bbox_b: Tuple[float, float, float, float],
        overlap_ratio: float,
        overlap_threshold: float,
        bbox_overlaps: bool,
        centroid_a: np.ndarray,
        centroid_b: np.ndarray,
        size_a: int,
        size_b: int,
        matched: bool,
    ):
        """Log detailed analysis of an instance pair for debugging."""
        print(f"\n{'='*60}")
        print(f"DEBUG: Instance Pair Analysis")
        print(f"{'='*60}")
        print(f"Instance {inst_id_a} ({tile_a_name}) <-> Instance {inst_id_b} ({tile_b_name})")
        print(f"Direction: {tile_a_name} ({direction}) <-> {tile_b_name} ({get_opposite_direction(direction)})")
        print(f"\nInstance {inst_id_a}:")
        print(f"  Tile: {tile_a_name}")
        print(f"  Point count: {size_a:,}")
        print(f"  Centroid: ({centroid_a[0]:.2f}, {centroid_a[1]:.2f}, {centroid_a[2]:.2f})")
        print(f"  BBox: ({bbox_a[0]:.2f}, {bbox_a[1]:.2f}) x ({bbox_a[2]:.2f}, {bbox_a[3]:.2f})")
        print(f"\nInstance {inst_id_b}:")
        print(f"  Tile: {tile_b_name}")
        print(f"  Point count: {size_b:,}")
        print(f"  Centroid: ({centroid_b[0]:.2f}, {centroid_b[1]:.2f}, {centroid_b[2]:.2f})")
        print(f"  BBox: ({bbox_b[0]:.2f}, {bbox_b[1]:.2f}) x ({bbox_b[2]:.2f}, {bbox_b[3]:.2f})")
        centroid_dist = np.linalg.norm(centroid_a - centroid_b)
        print(f"\nCentroid distance: {centroid_dist:.2f}m")
        print(f"BBox overlaps (10cm tolerance): {'YES' if bbox_overlaps else 'NO'}")
        print(f"FF3D overlap ratio: {overlap_ratio:.4f}")
        print(f"Overlap threshold: {overlap_threshold:.4f}")
        print(f"Match result: {'MATCHED' if matched else 'NOT MATCHED'}")
        if not matched:
            reasons = []
            if not bbox_overlaps:
                reasons.append("BBox doesn't overlap (within 10cm)")
            if overlap_ratio < overlap_threshold:
                reasons.append(f"Overlap ratio {overlap_ratio:.4f} < threshold {overlap_threshold:.4f}")
            if reasons:
                print(f"  Reasons: {', '.join(reasons)}")
        print(f"{'='*60}\n")

    def bboxes_overlap(bbox_a: Tuple[float, float, float, float], bbox_b: Tuple[float, float, float, float], tolerance: float = 0.1) -> bool:
        """
        Check if two bounding boxes overlap or are within tolerance distance.

        Args:
            bbox_a: (minx, maxx, miny, maxy) of first bounding box
            bbox_b: (minx, maxx, miny, maxy) of second bounding box
            tolerance: Maximum distance between boxes to still consider them (default: 0.1m = 10cm)

        Returns:
            True if boxes overlap or are within tolerance distance
        """
        minx_a, maxx_a, miny_a, maxy_a = bbox_a
        minx_b, maxx_b, miny_b, maxy_b = bbox_b

        # Check if boxes overlap (original check)
        if not (maxx_a < minx_b or minx_a > maxx_b or maxy_a < miny_b or miny_a > maxy_b):
            return True

        # Check if boxes are within tolerance distance (almost touching)
        # Compute gaps in X and Y dimensions
        # If boxes don't overlap, find the minimum separation
        x_gap = 0.0
        if maxx_a < minx_b:
            x_gap = minx_b - maxx_a  # A is to the left of B
        elif maxx_b < minx_a:
            x_gap = minx_a - maxx_b  # B is to the left of A
        # else: they overlap in X, x_gap = 0

        y_gap = 0.0
        if maxy_a < miny_b:
            y_gap = miny_b - maxy_a  # A is below B
        elif maxy_b < miny_a:
            y_gap = miny_a - maxy_b  # B is below A
        # else: they overlap in Y, y_gap = 0

        # Minimum separation is the diagonal distance between closest corners
        # For non-overlapping boxes: √(x_gap² + y_gap²)
        # But if boxes overlap in one dimension, we use the gap in the other dimension
        separation = np.sqrt(x_gap * x_gap + y_gap * y_gap)

        return separation <= tolerance

    # =========================================================================
    # Stage 3: Border Region Instance Matching (or All Instance Matching)
    # =========================================================================
    # Note: Cross-tile matching is optimized - each tile pair is checked exactly once
    # using `for j in range(i + 1, len(tiles))`, avoiding duplicate A->B and B->A checks.
    stage_name = "All Instance Matching" if match_all_instances else "Border Region Instance Matching"
    print(f"\n{'=' * 60}")
    print(f"Stage 3: {stage_name}")
    print(f"{'=' * 60}")

    # Instance tracking for debugging
    instance_tracking = {}  # (tile_name, local_inst_id) -> tracking info
    if debug_instance_ids:
        print(f"  Debug mode enabled for instances: {sorted(debug_instance_ids)}")
        # Initialize tracking for all instances in all tiles
        for tile_idx, tile in enumerate(tiles):
            unique_instances = np.unique(tile.instances[tile.instances > 0])
            for local_inst in unique_instances:
                gid = global_id(tile_idx, local_inst)
                if local_inst in debug_instance_ids:
                    instance_tracking[(tile.name, local_inst)] = {
                        "tile_name": tile.name,
                        "local_id": local_inst,
                        "global_id": gid,
                        "filtered_in_stage1": local_inst not in kept_instances_per_tile[tile.name],
                        "in_border_region": False,
                        "border_direction": None,
                        "compared_with": [],
                        "matched_with": None
                    }

    if match_all_instances:
        print(f"  Finding all instances (matching all instances, not just border region)...")
    else:
        print(f"  Finding border region instances (centroids in buffer to buffer+{border_zone_width}m zone)...")

    # Find instances to match (border region or all instances)
    border_instances = {}  # tile_name -> {instance_id: {'centroid': [...], 'points': [...], 'boundary': [...]}}

    # Build tile name to index mapping
    tile_name_to_idx = {tile.name: idx for idx, tile in enumerate(tiles)}

    for tile_idx, tile in enumerate(tiles):
        print(f"    Processing tile {tile_idx + 1}/{len(tiles)}: {tile.name} ({len(tile.points):,} points)...")
        tile_name = tile.name
        # Use neighbors from JSON graph when available; fall back to spatial neighbors only
        # if this tile was somehow not present in the JSON mapping (should not happen).
        if tile_name in neighbors_by_tile:
            neighbors = neighbors_by_tile[tile_name]
        else:
            neighbors = find_spatial_neighbors(tile.boundary, tile_name, tile_boundaries, tolerance=buffer)
        kept_instances = kept_instances_per_tile[tile_name]

        neighbor_names = [n for n in neighbors.values() if n is not None]
        print(f"      Neighbors: {', '.join(neighbor_names) if neighbor_names else 'none'}")
        if verbose:
            for direction, neighbor_name in neighbors.items():
                if neighbor_name is not None:
                    neighbor_boundary = tile_boundaries.get(neighbor_name)
                    if neighbor_boundary:
                        overlap = find_overlap_region(tile.boundary, neighbor_boundary)
                        if overlap:
                            ov_minx, ov_maxx, ov_miny, ov_maxy = overlap
                            ov_width = ov_maxx - ov_minx
                            ov_height = ov_maxy - ov_miny
                            print(f"        {direction.upper()} {neighbor_name}: overlap {ov_width:.1f}m x {ov_height:.1f}m")

        min_x, max_x, min_y, max_y = tile.boundary
        border_zone_end = buffer + border_zone_width  # border_zone_width beyond buffer

        # Define border region boundaries (buffer to buffer+border_zone_width from edges with neighbors)
        # Inner edge of border region (end of buffer zone)
        border_inner_min_x = min_x + (buffer if neighbors["west"] is not None else 0)
        border_inner_max_x = max_x - (buffer if neighbors["east"] is not None else 0)
        border_inner_min_y = min_y + (buffer if neighbors["south"] is not None else 0)
        border_inner_max_y = max_y - (buffer if neighbors["north"] is not None else 0)

        # Outer edge of border region (buffer+border_zone_width from tile edge)
        border_outer_min_x = min_x + (border_zone_end if neighbors["west"] is not None else 0)
        border_outer_max_x = max_x - (border_zone_end if neighbors["east"] is not None else 0)
        border_outer_min_y = min_y + (border_zone_end if neighbors["south"] is not None else 0)
        border_outer_max_y = max_y - (border_zone_end if neighbors["north"] is not None else 0)

        border_instances[tile_name] = {}

        if match_all_instances:
            # Collect ALL kept instances (not just border region)
            all_unique_insts = kept_instances - {0}  # All kept instances except ground

            if len(all_unique_insts) == 0:
                print(f"      No instances to match in {tile.name}")
                continue

            # Compute centroids for all instances
            print(f"      Computing centroids for {len(all_unique_insts)} instances (all instances)...")
            all_centroids = compute_centroids_vectorized(tile.points, tile.instances)
            instance_centroids = {
                inst_id: all_centroids[inst_id]
                for inst_id in all_unique_insts
                if inst_id in all_centroids
            }

            instance_count = 0

            # For each instance, extract full points (no direction filtering)
            for inst_id in all_unique_insts:
                if inst_id not in instance_centroids:
                    continue

                centroid = instance_centroids[inst_id]

                # Extract full instance points
                inst_mask = tile.instances == inst_id
                inst_points = tile.points[inst_mask]

                # Compute instance bounding box
                inst_minx = inst_points[:, 0].min()
                inst_maxx = inst_points[:, 0].max()
                inst_miny = inst_points[:, 1].min()
                inst_maxy = inst_points[:, 1].max()

                # Use "all" as direction to indicate this is not border-specific
                border_instances[tile_name][inst_id] = {
                    'centroid': centroid,
                    'points': inst_points,
                    'boundary': (inst_minx, inst_maxx, inst_miny, inst_maxy),
                    'direction': 'all',  # Special direction for all-instance matching
                    'tile_idx': tile_idx
                }
                instance_count += 1

                # Update tracking for debug instances
                if debug_instance_ids and inst_id in debug_instance_ids:
                    key = (tile_name, inst_id)
                    if key in instance_tracking:
                        instance_tracking[key]["in_border_region"] = True
                        instance_tracking[key]["border_direction"] = 'all'
                        print(f"      DEBUG: Instance {inst_id} included in all-instance matching")

            print(f"      Found {instance_count} instances in {tile.name} (all instances)")
        else:
            pass

        border_mask = get_border_region_mask(
                tile.points, tile.boundary, buffer, border_zone_end, neighbors
        )
        border_points = tile.points[border_mask]
        border_inst_ids = tile.instances[border_mask]

        # Get unique instances in border region (much smaller set than all instances)
        border_unique_insts = set(np.unique(border_inst_ids)) - {0}
        border_unique_insts &= kept_instances  # Only kept instances

        if len(border_unique_insts) == 0:
            continue

        if verbose:
            print(f"      Computing centroids for {len(border_unique_insts)} border instances...")
        all_centroids = compute_centroids_vectorized(tile.points, tile.instances)
        border_centroids = {
            inst_id: all_centroids[inst_id]
            for inst_id in border_unique_insts
            if inst_id in all_centroids
        }

        border_count = 0

        # For each border instance, determine direction and extract full points
        for inst_id in border_unique_insts:
            if inst_id not in border_centroids:
                continue

            centroid = border_centroids[inst_id]
            cx, cy = centroid[0], centroid[1]

            # Determine border direction based on centroid position
            border_direction = None
            if neighbors["west"] is not None and cx < min_x + border_zone_end:
                border_direction = "west"
            elif neighbors["east"] is not None and cx > max_x - border_zone_end:
                border_direction = "east"
            elif neighbors["south"] is not None and cy < min_y + border_zone_end:
                border_direction = "south"
            elif neighbors["north"] is not None and cy > max_y - border_zone_end:
                border_direction = "north"

            if border_direction is None:
                continue

            # Extract full instance points (from original tile, not just border region)
            inst_mask = tile.instances == inst_id
            inst_points = tile.points[inst_mask]

            # Compute instance bounding box
            inst_minx = inst_points[:, 0].min()
            inst_maxx = inst_points[:, 0].max()
            inst_miny = inst_points[:, 1].min()
            inst_maxy = inst_points[:, 1].max()

            border_instances[tile_name][inst_id] = {
                'centroid': centroid,
                'points': inst_points,
                'boundary': (inst_minx, inst_maxx, inst_miny, inst_maxy),
                'direction': border_direction,
                'tile_idx': tile_idx
            }
            border_count += 1

            # Update tracking for debug instances
            if debug_instance_ids and inst_id in debug_instance_ids:
                    key = (tile_name, inst_id)
                    if key in instance_tracking:
                        instance_tracking[key]["in_border_region"] = True
                        instance_tracking[key]["border_direction"] = border_direction
                        print(f"      DEBUG: Instance {inst_id} in border region ({border_direction})")

        if border_count > 0:
            print(f"      {tile.name}: {border_count} border instances")

    # Match instances between neighbor tiles
    total_border_insts = sum(len(insts) for insts in border_instances.values())
    tiles_with_border = len([t for t in border_instances if border_instances[t]])
    if match_all_instances:
        print(f"  Found {total_border_insts} instances across {tiles_with_border} tiles (all instances)")
    else:
        print(f"  Found {total_border_insts} border region instances across {tiles_with_border} tiles")
    print(f"  Processing tile pairs...")

    # Track which global IDs have already been matched to avoid duplicate checks
    matched_gids = set()

    border_matches = 0
    total_bbox_checks = 0
    total_ff3d_computations = 0
    tiles_processed = 0

    for i in range(len(tiles)):
        tile_a = tiles[i]
        # Use JSON-based neighbors when available
        if tile_a.name in neighbors_by_tile:
            neighbors_a = neighbors_by_tile[tile_a.name]
        else:
            neighbors_a = find_spatial_neighbors(tile_a.boundary, tile_a.name, tile_boundaries)

        for direction, neighbor_name in neighbors_a.items():
            if neighbor_name is None:
                continue

            # Find neighbor tile index
            tile_b_idx = tile_name_to_idx.get(neighbor_name)
            if tile_b_idx is None:
                continue

            tile_b = tiles[tile_b_idx]

            # Get instances from both tiles
            if match_all_instances:
                # Match ALL instances between neighbor tiles (no direction filtering)
                border_insts_a = border_instances.get(tile_a.name, {})
                border_insts_b = border_instances.get(tile_b.name, {})
            else:
                # Original logic: only match border instances in specific directions
                border_insts_a = {
                    inst_id: data for inst_id, data in border_instances.get(tile_a.name, {}).items()
                    if data['direction'] == direction
                }
                border_insts_b = {
                    inst_id: data for inst_id, data in border_instances.get(tile_b.name, {}).items()
                    if data['direction'] == get_opposite_direction(direction)
                }

            if not border_insts_a or not border_insts_b:
                continue

            # Progress: Show which tile pair is being processed
            matches_before = border_matches
            if match_all_instances:
                print(f"    Checking {tile_a.name} <-> {tile_b.name} ({direction} neighbors): "
                      f"{len(border_insts_a)} vs {len(border_insts_b)} instances", end=" ... ")
            else:
                print(f"    Checking {tile_a.name} ({direction}) <-> {tile_b.name} ({get_opposite_direction(direction)}): "
                      f"{len(border_insts_a)} vs {len(border_insts_b)} border instances", end=" ... ")

            # Build list of candidate instances from tile B (not already matched)
            candidates_b = []
            for inst_id_b, data_b in border_insts_b.items():
                gid_b = global_id(tile_b_idx, inst_id_b)
                if gid_b not in matched_gids:
                    candidates_b.append((inst_id_b, gid_b, data_b))

            if not candidates_b:
                continue

            # For each instance in tile A, check overlap with all candidate instances in tile B
            for inst_id_a, data_a in border_insts_a.items():
                gid_a = global_id(i, inst_id_a)

                # Skip if already matched
                if gid_a in matched_gids:
                    continue

                bbox_a = data_a['boundary']

                # Check each candidate in tile B
                for inst_id_b, gid_b, data_b in candidates_b:
                    # Skip if already matched
                    if gid_b in matched_gids:
                        continue

                    bbox_b = data_b['boundary']

                    # Quick bounding box overlap/nearby check (within 10cm tolerance)
                    total_bbox_checks += 1
                    bbox_overlaps = bboxes_overlap(bbox_a, bbox_b, tolerance=0.1)

                    # Check if we should debug this pair
                    should_debug = (
                        debug_instance_ids is not None and
                        (inst_id_a in debug_instance_ids or inst_id_b in debug_instance_ids)
                    )

                    if should_debug:
                        print(f"\n  DEBUG: Checking pair {inst_id_a} <-> {inst_id_b}")
                        print(f"    BBox overlap check: {'PASS' if bbox_overlaps else 'FAIL'}")

                    if not bbox_overlaps:
                        if should_debug:
                            print(f"    Skipping: BBox doesn't overlap (within 10cm tolerance)")
                        continue

                    # Now compute expensive FF3D overlap ratio
                    total_ff3d_computations += 1
                    points_a = data_a['points']
                    points_b = data_b['points']
                    instances_a = np.full(len(points_a), inst_id_a, dtype=np.int32)
                    instances_b = np.full(len(points_b), inst_id_b, dtype=np.int32)

                    overlap_ratios_dict, size_a, size_b = compute_ff3d_overlap_ratios(
                        instances_a, instances_b, points_a, points_b, correspondence_tolerance
                    )

                    overlap_ratio = overlap_ratios_dict.get((inst_id_a, inst_id_b), 0.0)

                    # Debug logging for instance pairs
                    if should_debug:
                        centroid_a = data_a['centroid']
                        centroid_b = data_b['centroid']
                        size_a_val = size_a.get(inst_id_a, 0)
                        size_b_val = size_b.get(inst_id_b, 0)

                        # Update tracking
                        key_a = (tile_a.name, inst_id_a)
                        key_b = (tile_b.name, inst_id_b)
                        if key_a in instance_tracking:
                            instance_tracking[key_a]["compared_with"].append({
                                "tile": tile_b.name,
                                "instance": inst_id_b,
                                "overlap_ratio": overlap_ratio,
                                "matched": overlap_ratio >= overlap_threshold
                            })
                        if key_b in instance_tracking:
                            instance_tracking[key_b]["compared_with"].append({
                                "tile": tile_a.name,
                                "instance": inst_id_a,
                                "overlap_ratio": overlap_ratio,
                                "matched": overlap_ratio >= overlap_threshold
                            })

                        log_instance_pair_analysis(
                            inst_id_a, inst_id_b,
                            tile_a.name, tile_b.name, direction,
                            bbox_a, bbox_b,
                            overlap_ratio, overlap_threshold,
                            bbox_overlaps,
                            centroid_a, centroid_b,
                            size_a_val, size_b_val,
                            overlap_ratio >= overlap_threshold
                        )

                    if overlap_ratio >= overlap_threshold:
                        # Merge via Union-Find
                        root = uf.union(gid_a, gid_b)
                        matched_gids.add(gid_a)
                        matched_gids.add(gid_b)
                        border_matches += 1

                        # Update tracking for matched instances
                        if debug_instance_ids:
                            key_a = (tile_a.name, inst_id_a)
                            key_b = (tile_b.name, inst_id_b)
                            if key_a in instance_tracking:
                                instance_tracking[key_a]["matched_with"] = (tile_b.name, inst_id_b)
                            if key_b in instance_tracking:
                                instance_tracking[key_b]["matched_with"] = (tile_a.name, inst_id_a)

                        if verbose:
                            print(f"      ✓ Match: {tile_a.name}:{inst_id_a} <-> {tile_b.name}:{inst_id_b} (overlap: {overlap_ratio:.3f})")

            # Progress: Show results for this tile pair
            matches_this_pair = border_matches - matches_before
            if matches_this_pair > 0:
                print(f"{matches_this_pair} match(es) found")
            else:
                print("no matches")
            tiles_processed += 1

            # Periodic progress update every 10 tile pairs
            if tiles_processed % 10 == 0:
                print(f"  Progress: {tiles_processed} tile pairs processed, {border_matches} total matches so far...")

    if match_all_instances:
        print(f"  Matched {border_matches} instance pairs (all instances)")
    else:
        print(f"  Matched {border_matches} border region instance pairs")
    print(f"  Performance: {total_bbox_checks} bbox checks, {total_ff3d_computations} FF3D computations")
    print(f"  ✓ Stage 3 completed: {stage_name} done")

    # Print instance tracking summary for debug instances
    if debug_instance_ids and instance_tracking:
        print(f"\n{'='*60}")
        print("Instance Tracking Summary")
        print(f"{'='*60}")
        for (tile_name, local_id), info in sorted(instance_tracking.items()):
            print(f"\nInstance {local_id} (Tile: {tile_name}):")
            print(f"  Global ID: {info['global_id']}")
            print(f"  Filtered in Stage 1: {'YES' if info['filtered_in_stage1'] else 'NO'}")
            print(f"  In border region: {'YES' if info['in_border_region'] else 'NO'}")
            if info['in_border_region']:
                print(f"  Border direction: {info['border_direction']}")
            print(f"  Compared with {len(info['compared_with'])} instance(s):")
            for comp in info['compared_with']:
                print(f"    - {comp['tile']}:{comp['instance']} (overlap: {comp['overlap_ratio']:.4f}, matched: {comp['matched']})")
            if info['matched_with']:
                print(f"  Matched with: {info['matched_with'][0]}:{info['matched_with'][1]}")
            else:
                print(f"  Matched with: NONE")
        print(f"{'='*60}\n")

    # Get connected components
    components = uf.get_components()
    print(f"  Connected components: {len(components)}")
    print(f"  ✓ Instance matching completed: {len(components)} merged instance groups")

    # Create mapping from global ID to final merged ID
    global_to_merged = {}
    merged_instance_sources = {}  # merged_id -> list of source global IDs (for CSV tracking)

    for merged_id, (root, members) in enumerate(components.items(), start=1):
        merged_instance_sources[merged_id] = list(members)

        if len(members) > 1 and verbose:
            print(f"  Merged ID {merged_id} created from {len(members)} global IDs: {sorted(members)}")

        for gid in members:
            global_to_merged[gid] = merged_id

    tile_idx_to_name = recover_orphaned_instances(
        tiles=tiles,
        kept_instances_per_tile=kept_instances_per_tile,
        filtered_instances_per_tile=filtered_instances_per_tile,
        buffer_direction_per_tile=buffer_direction_per_tile,
        neighbors_by_tile=neighbors_by_tile,
        global_to_merged=global_to_merged,
        merged_instance_sources=merged_instance_sources,
        buffer=buffer,
        border_zone_width=border_zone_width,
        num_threads=num_threads,
        verbose=verbose,
    )

    return InstanceMergeResult(
        global_to_merged=global_to_merged,
        merged_instance_sources=merged_instance_sources,
        tile_idx_to_name=tile_idx_to_name,
    )
