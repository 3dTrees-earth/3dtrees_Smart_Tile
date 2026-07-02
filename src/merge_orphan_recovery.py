#!/usr/bin/env python3
"""Recover filtered tile instances that are not covered by neighboring tiles."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.spatial import cKDTree

from merge_instance_ids import global_id
from merge_tile_loading import TileData
from tile_spatial import compute_centroids_vectorized


def _instance_is_in_border(
    centroid: np.ndarray,
    boundary: Tuple[float, float, float, float],
    neighbors: Dict[str, Optional[str]],
    buffer: float,
    border_zone_end: float,
) -> bool:
    min_x, max_x, min_y, max_y = boundary
    cx, cy = centroid[0], centroid[1]
    bi_min_x = min_x + (buffer if neighbors.get("west") is not None else 0)
    bi_max_x = max_x - (buffer if neighbors.get("east") is not None else 0)
    bi_min_y = min_y + (buffer if neighbors.get("south") is not None else 0)
    bi_max_y = max_y - (buffer if neighbors.get("north") is not None else 0)
    bo_min_x = min_x + (border_zone_end if neighbors.get("west") is not None else 0)
    bo_max_x = max_x - (border_zone_end if neighbors.get("east") is not None else 0)
    bo_min_y = min_y + (border_zone_end if neighbors.get("south") is not None else 0)
    bo_max_y = max_y - (border_zone_end if neighbors.get("north") is not None else 0)

    return (
        (neighbors.get("west") is not None and bi_min_x <= cx <= bo_min_x)
        or (neighbors.get("east") is not None and bo_max_x <= cx <= bi_max_x)
        or (neighbors.get("south") is not None and bi_min_y <= cy <= bo_min_y)
        or (neighbors.get("north") is not None and bo_max_y <= cy <= bi_max_y)
    )


def _build_instance_bboxes(
    tiles: List[TileData],
    kept_instances_per_tile: Dict[str, Set[int]],
    filtered_instances_per_tile: Dict[str, Set[int]],
    neighbors_by_tile: Dict[str, Dict[str, Optional[str]]],
    buffer: float,
    border_zone_width: float,
) -> Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]]:
    """Compute bboxes only for filtered instances and kept border instances."""
    border_zone_end = buffer + border_zone_width
    instance_bboxes: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]] = {}

    for tile in tiles:
        centroids = compute_centroids_vectorized(tile.points, tile.instances)
        kept_instances = kept_instances_per_tile[tile.name]
        neighbors = neighbors_by_tile.get(tile.name) or {
            "east": None,
            "west": None,
            "north": None,
            "south": None,
        }
        kept_in_border = {
            inst_id
            for inst_id in kept_instances
            if inst_id in centroids
            and _instance_is_in_border(
                centroids[inst_id],
                tile.boundary,
                neighbors,
                buffer,
                border_zone_end,
            )
        }
        need_bbox = filtered_instances_per_tile[tile.name] | kept_in_border

        sort_idx = np.argsort(tile.instances)
        sorted_inst = tile.instances[sort_idx]
        sorted_points = tile.points[sort_idx]
        unique_inst, first_idx, counts = np.unique(
            sorted_inst,
            return_index=True,
            return_counts=True,
        )
        bboxes = {}
        for i, inst_id in enumerate(unique_inst):
            if inst_id <= 0 or inst_id not in need_bbox:
                continue
            start = first_idx[i]
            end = start + counts[i]
            pts = sorted_points[start:end]
            bboxes[inst_id] = (pts.min(axis=0), pts.max(axis=0))
        instance_bboxes[tile.name] = bboxes

    return instance_bboxes


def recover_orphaned_instances(
    tiles: List[TileData],
    kept_instances_per_tile: Dict[str, Set[int]],
    filtered_instances_per_tile: Dict[str, Set[int]],
    buffer_direction_per_tile: Dict[str, Dict[int, Optional[str]]],
    neighbors_by_tile: Dict[str, Dict[str, Optional[str]]],
    global_to_merged: Dict[int, int],
    merged_instance_sources: Dict[int, List[int]],
    buffer: float,
    border_zone_width: float,
    num_threads: int,
    verbose: bool = False,
) -> Dict[int, str]:
    """Recover filtered instances that are not covered by neighboring kept instances."""
    print("\n  Checking for orphaned filtered instances...")

    tile_idx_to_name = {idx: tile.name for idx, tile in enumerate(tiles)}
    instance_bboxes = _build_instance_bboxes(
        tiles,
        kept_instances_per_tile,
        filtered_instances_per_tile,
        neighbors_by_tile,
        buffer,
        border_zone_width,
    )

    print("  Building spatial index of kept instances (border region only)...")
    kept_instance_data = []
    for tile_idx, tile in enumerate(tiles):
        tile_bboxes = instance_bboxes[tile.name]
        kept_instances = kept_instances_per_tile[tile.name]
        for inst_id in kept_instances:
            if inst_id not in tile_bboxes:
                continue
            bbox_min, bbox_max = tile_bboxes[inst_id]
            center_x = (bbox_min[0] + bbox_max[0]) / 2.0
            center_y = (bbox_min[1] + bbox_max[1]) / 2.0
            kept_instance_data.append((center_x, center_y, tile_idx, inst_id, bbox_min, bbox_max))

    search_radius = 5.0
    if kept_instance_data:
        centers = np.array([(x, y) for x, y, _, _, _, _ in kept_instance_data])
        kept_tree = cKDTree(centers)
    else:
        kept_tree = None
        print("  Warning: No kept instances found for spatial indexing")

    neighbor_trees: Dict[Tuple[int, int], cKDTree] = {}
    for _, _, check_tile_idx, neighbor_inst, _, _ in kept_instance_data:
        cache_key = (check_tile_idx, neighbor_inst)
        if cache_key in neighbor_trees:
            continue
        check_tile = tiles[check_tile_idx]
        neighbor_mask = check_tile.instances == neighbor_inst
        neighbor_points = check_tile.points[neighbor_mask]
        if len(neighbor_points) > 0:
            neighbor_trees[cache_key] = cKDTree(neighbor_points[:, :2])

    overlap_tolerance = 1.0

    def check_one_orphan_covered(item: Tuple[int, int]) -> Tuple[int, int, bool]:
        tile_idx, local_inst = item
        tile = tiles[tile_idx]
        tile_name = tile.name
        tile_bboxes = instance_bboxes[tile_name]
        if local_inst <= 0 or local_inst not in tile_bboxes:
            return (tile_idx, local_inst, True)
        if buffer_direction_per_tile[tile_name].get(local_inst) is None:
            return (tile_idx, local_inst, True)
        fmin, fmax = tile_bboxes[local_inst]
        orphan_center = ((fmin[0] + fmax[0]) / 2.0, (fmin[1] + fmax[1]) / 2.0)
        inst_mask = tile.instances == local_inst
        filtered_points = tile.points[inst_mask]
        if len(filtered_points) == 0:
            return (tile_idx, local_inst, True)
        neighbor_has_tree = False
        if kept_tree is not None:
            nearby_indices = kept_tree.query_ball_point(orphan_center, r=search_radius)
            for idx in nearby_indices:
                _, _, check_tile_idx, neighbor_inst, nmin, nmax = kept_instance_data[idx]
                if check_tile_idx == tile_idx:
                    continue
                if (
                    fmax[0] < nmin[0] - overlap_tolerance
                    or fmin[0] > nmax[0] + overlap_tolerance
                    or fmax[1] < nmin[1] - overlap_tolerance
                    or fmin[1] > nmax[1] + overlap_tolerance
                ):
                    continue
                neighbor_tree = neighbor_trees.get((check_tile_idx, neighbor_inst))
                if neighbor_tree is None:
                    continue
                distances, _ = neighbor_tree.query(filtered_points[:, :2], k=1)
                fraction_within = np.sum(distances <= overlap_tolerance) / len(filtered_points)
                if fraction_within > 0.50:
                    neighbor_has_tree = True
                    break
        return (tile_idx, local_inst, neighbor_has_tree)

    orphan_candidates: List[Tuple[int, int]] = []
    for tile_idx, tile in enumerate(tiles):
        filtered_instances = filtered_instances_per_tile[tile.name]
        buffer_directions = buffer_direction_per_tile[tile.name]
        tile_bboxes = instance_bboxes[tile.name]
        for local_inst in filtered_instances:
            if local_inst <= 0 or local_inst not in tile_bboxes:
                continue
            if buffer_directions.get(local_inst) is None:
                continue
            if np.sum(tile.instances == local_inst) == 0:
                continue
            orphan_candidates.append((tile_idx, local_inst))

    next_merged_id = max(global_to_merged.values()) + 1 if global_to_merged else 1
    recovered_count = 0
    skipped_covered = 0
    orphan_parallel_workers = max(1, min(num_threads, len(orphan_candidates) or 1))

    if orphan_candidates:
        print(
            f"  Checking {len(orphan_candidates)} orphan candidates "
            f"with {orphan_parallel_workers} workers...",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=orphan_parallel_workers) as executor:
            orphan_results = list(executor.map(check_one_orphan_covered, orphan_candidates))
        for tile_idx, local_inst, covered in orphan_results:
            if covered:
                skipped_covered += 1
                continue
            tile = tiles[tile_idx]
            gid = global_id(tile_idx, local_inst)
            global_to_merged[gid] = next_merged_id
            merged_instance_sources[next_merged_id] = [gid]
            if verbose:
                print(
                    "  Recovered orphan - "
                    f"global_id={gid} (tile={tile.name}, local={local_inst}) "
                    f"-> merged_id={next_merged_id}"
                )
            kept_instances_per_tile[tile.name].add(local_inst)
            next_merged_id += 1
            recovered_count += 1

    if recovered_count > 0 or skipped_covered > 0:
        print(f"  Recovered {recovered_count} orphaned instances")
        print(f"  Skipped {skipped_covered} instances (neighbor has overlapping tree)")
    else:
        print("  No orphaned instances found")

    return tile_idx_to_name
