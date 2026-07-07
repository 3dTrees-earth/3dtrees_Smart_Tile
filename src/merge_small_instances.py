#!/usr/bin/env python3
"""Small-instance reassignment for SmartTile merged point clouds."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree


def _compute_hull_wrapper(args):
    """Compute one convex-hull volume for process-pool execution."""
    from scipy.spatial import ConvexHull

    points, bbox_volume = args
    try:
        hull = ConvexHull(points)
        return (hull.volume, True)
    except Exception:
        return (bbox_volume, False)


def merge_small_volume_instances(
    points: np.ndarray,
    instances: np.ndarray,
    min_points_for_hull_check: int = 1000,
    min_cluster_size: int = 300,
    max_volume_for_merge: float = 4.0,
    max_search_radius: float = 5.0,
    num_threads: int = 1,
    verbose: bool = True,
    presorted_points: Optional[np.ndarray] = None,
    presorted_instances: Optional[np.ndarray] = None,
    presorted_unique_inst: Optional[np.ndarray] = None,
    presorted_first_idx: Optional[np.ndarray] = None,
    presorted_inst_counts: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int]:
    """Reassign small/noisy instances to the nearest large instance.

    Only the instance ID array is modified. Extra dimensions remain attached to
    their original points.
    """
    use_presorted = (
        presorted_points is not None
        and presorted_instances is not None
        and presorted_unique_inst is not None
        and presorted_first_idx is not None
        and presorted_inst_counts is not None
    )

    summary_mode = "presorted" if use_presorted else "unsorted"
    centroid_by_inst = None
    bbox_volume_by_inst = None

    if use_presorted:
        sorted_points = presorted_points
        sorted_instances = presorted_instances
        unique_inst = presorted_unique_inst
        first_idx = presorted_first_idx
        inst_counts = presorted_inst_counts
        print(f"  {len(unique_inst):,} unique instances (pre-sorted).", flush=True)
    else:
        total_points = len(instances)
        nonzero_mask = instances > 0
        nonzero_count = nonzero_mask.sum()

        print(
            f"  Summarizing {nonzero_count:,} instance points (of {total_points:,} total)...",
            flush=True,
        )

        if nonzero_count == 0:
            print("  Found 0 unique instances.", flush=True)
            return (instances, 0)

        positive_instances = instances[nonzero_mask].astype(np.int64, copy=False)
        max_positive_id = int(positive_instances.max())
        counts = np.bincount(positive_instances, minlength=max_positive_id + 1)
        unique_inst = np.flatnonzero(counts)
        unique_inst = unique_inst[unique_inst > 0]
        inst_counts = counts[unique_inst].astype(np.int64, copy=False)
        first_idx = np.zeros(len(unique_inst), dtype=np.int64)

        pos_x = points[nonzero_mask, 0]
        pos_y = points[nonzero_mask, 1]
        pos_z = points[nonzero_mask, 2]

        sum_x = np.bincount(positive_instances, weights=pos_x, minlength=max_positive_id + 1)
        sum_y = np.bincount(positive_instances, weights=pos_y, minlength=max_positive_id + 1)
        sum_z = np.bincount(positive_instances, weights=pos_z, minlength=max_positive_id + 1)

        centroid_by_inst = np.zeros((max_positive_id + 1, 3), dtype=np.float64)
        centroid_by_inst[unique_inst, 0] = sum_x[unique_inst] / counts[unique_inst]
        centroid_by_inst[unique_inst, 1] = sum_y[unique_inst] / counts[unique_inst]
        centroid_by_inst[unique_inst, 2] = sum_z[unique_inst] / counts[unique_inst]

        min_x = np.full(max_positive_id + 1, np.inf, dtype=np.float64)
        max_x = np.full(max_positive_id + 1, -np.inf, dtype=np.float64)
        min_y = np.full(max_positive_id + 1, np.inf, dtype=np.float64)
        max_y = np.full(max_positive_id + 1, -np.inf, dtype=np.float64)
        min_z = np.full(max_positive_id + 1, np.inf, dtype=np.float64)
        max_z = np.full(max_positive_id + 1, -np.inf, dtype=np.float64)
        np.minimum.at(min_x, positive_instances, pos_x)
        np.maximum.at(max_x, positive_instances, pos_x)
        np.minimum.at(min_y, positive_instances, pos_y)
        np.maximum.at(max_y, positive_instances, pos_y)
        np.minimum.at(min_z, positive_instances, pos_z)
        np.maximum.at(max_z, positive_instances, pos_z)
        bbox_volume_by_inst = (max_x - min_x) * (max_y - min_y) * (max_z - min_z)

        sorted_points = None
        sorted_instances = None
        print(f"  Found {len(unique_inst):,} unique instances.", flush=True)

    hull_candidates = []
    small_volume_instances = []
    small_point_count_instances = []
    large_instances = []
    bbox_skipped_count = 0

    total_instances = len(unique_inst)
    print(f"  Categorizing {total_instances:,} instances...", flush=True)

    for idx, (inst_id, start, count) in enumerate(zip(unique_inst, first_idx, inst_counts)):
        count = int(count)

        if total_instances >= 1000 and idx % 1000 == 0 and idx > 0:
            print(f"    {idx:,}/{total_instances:,} instances processed...", flush=True)

        end = start + count
        if count >= min_points_for_hull_check:
            centroid = (
                sorted_points[start:end].mean(axis=0)
                if summary_mode == "presorted"
                else centroid_by_inst[int(inst_id)]
            )
            large_instances.append((inst_id, count, centroid))
            continue

        if summary_mode == "presorted":
            bbox_volume = np.prod(
                sorted_points[start:end].max(axis=0) - sorted_points[start:end].min(axis=0)
            )
            centroid = sorted_points[start:end].mean(axis=0)
        else:
            bbox_volume = float(bbox_volume_by_inst[int(inst_id)])
            centroid = centroid_by_inst[int(inst_id)]

        if bbox_volume >= max_volume_for_merge * 4.0:
            if count < min_cluster_size:
                small_point_count_instances.append((inst_id, count, centroid))
                bbox_skipped_count += 1
                if verbose:
                    print(
                        f"    Instance {inst_id}: {count} pts, bbox {bbox_volume:.2f} m3 - "
                        f"REDISTRIBUTE (sparse, < {min_cluster_size} pts)"
                    )
            else:
                large_instances.append((inst_id, count, centroid))
                bbox_skipped_count += 1
                if verbose:
                    print(
                        f"    Instance {inst_id}: {count} pts, bbox {bbox_volume:.2f} m3 - "
                        "keeping (bbox too large, enough points)"
                    )
            continue

        hull_candidates.append((inst_id, count, start, end, bbox_volume))

    if verbose:
        print(
            f"  Categorized instances: {len(large_instances):,} large, "
            f"{bbox_skipped_count:,} skipped (large bbox), "
            f"{len(hull_candidates):,} need hull computation",
            flush=True,
        )

    if len(hull_candidates) > 0:
        if verbose:
            print(
                f"  Computing centroids and convex hulls for {len(hull_candidates):,} "
                f"instances (< {min_points_for_hull_check} points)...",
                flush=True,
            )
            if num_threads > 1:
                print(
                    f"    Using {num_threads} workers (--workers={num_threads}) "
                    "for parallel processing...",
                    flush=True,
                )

        hull_args = []
        hull_centroids = []
        for _inst_id, _count, start, end, bbox_volume in hull_candidates:
            if summary_mode == "presorted":
                pts = sorted_points[start:end]
            else:
                pts = points[instances == _inst_id]
            hull_args.append((pts, bbox_volume))
            if summary_mode == "presorted":
                hull_centroids.append(pts.mean(axis=0))
            else:
                hull_centroids.append(centroid_by_inst[int(_inst_id)])

        use_parallel = num_threads > 1 and len(hull_candidates) > 10
        if use_parallel:
            batch_size = max(100, len(hull_args) // 20)
            hull_results = []

            with ProcessPoolExecutor(max_workers=num_threads) as executor:
                for batch_idx in range(0, len(hull_args), batch_size):
                    batch = hull_args[batch_idx:batch_idx + batch_size]
                    batch_results = list(executor.map(_compute_hull_wrapper, batch))
                    hull_results.extend(batch_results)

                    if verbose:
                        progress = min(100.0, len(hull_results) * 100.0 / len(hull_candidates))
                        print(
                            f"    Hull progress: {len(hull_results):,}/"
                            f"{len(hull_candidates):,} ({progress:.1f}%)...",
                            flush=True,
                        )
        else:
            if verbose and num_threads == 1:
                print("    Using sequential computation (--workers=1 or <10 candidates)...", flush=True)
            hull_results = []
            for idx, args in enumerate(hull_args):
                hull_results.append(_compute_hull_wrapper(args))

                if verbose and (idx % 100 == 0 or idx == len(hull_args) - 1):
                    progress = (idx + 1) * 100.0 / len(hull_candidates)
                    print(
                        f"    Hull progress: {idx + 1:,}/"
                        f"{len(hull_candidates):,} ({progress:.1f}%)...",
                        flush=True,
                    )

        if verbose:
            print("  Processing hull results and categorizing instances...", flush=True)

        for (inst_id, count, _start, _end, _bbox_volume), (volume, hull_success), centroid in zip(
            hull_candidates,
            hull_results,
            hull_centroids,
        ):
            if verbose and not hull_success:
                print(f"    Instance {inst_id}: hull computation failed, using bbox volume")

            if volume < max_volume_for_merge:
                small_volume_instances.append((inst_id, count, volume, centroid))
                if verbose:
                    print(
                        f"    Instance {inst_id}: {count} pts, {volume:.2f} m3 - "
                        f"SMALL (< {max_volume_for_merge} m3) - merge"
                    )
            elif count < min_cluster_size:
                small_point_count_instances.append((inst_id, count, centroid))
                if verbose:
                    print(
                        f"    Instance {inst_id}: {count} pts, {volume:.2f} m3 - "
                        f"REDISTRIBUTE (< {min_cluster_size} pts)"
                    )
            else:
                large_instances.append((inst_id, count, centroid))
                if verbose:
                    print(
                        f"    Instance {inst_id}: {count} pts, {volume:.2f} m3 - "
                        "keeping (volume ok, enough points)"
                    )

    all_small_instances = small_volume_instances + [
        (inst_id, count, 0.0, centroid)
        for inst_id, count, centroid in small_point_count_instances
    ]

    if len(all_small_instances) == 0:
        print("  No small instances to merge/redistribute", flush=True)
        if bbox_skipped_count > 0:
            print(
                f"  Skipped {bbox_skipped_count} instances using bounding box filter "
                f"(bbox >= {max_volume_for_merge * 4.0:.1f} m3)",
                flush=True,
            )
        return (instances, bbox_skipped_count)

    if len(large_instances) == 0:
        print("  No large instances to merge into", flush=True)
        if bbox_skipped_count > 0:
            print(
                f"  Skipped {bbox_skipped_count} instances using bounding box filter "
                f"(bbox >= {max_volume_for_merge * 4.0:.1f} m3)",
                flush=True,
            )
        return (instances, bbox_skipped_count)

    print(
        f"  Found {len(small_volume_instances)} small-volume instances "
        f"(< {max_volume_for_merge} m3) to merge",
        flush=True,
    )
    if len(small_point_count_instances) > 0:
        print(
            f"  Found {len(small_point_count_instances)} small point-count instances "
            f"(< {min_cluster_size} pts, volume >= {max_volume_for_merge} m3) to redistribute",
            flush=True,
        )
    if bbox_skipped_count > 0:
        print(
            f"  Skipped {bbox_skipped_count} instances using bounding box filter "
            f"(bbox >= {max_volume_for_merge * 4.0:.1f} m3) - saved convex hull computation",
            flush=True,
        )

    large_ids = [x[0] for x in large_instances]
    large_coords = np.array([x[2] for x in large_instances])
    tree = cKDTree(large_coords)

    max_inst = instances.max() + 1
    inst_to_target = np.arange(max_inst, dtype=np.int32)

    small_centroids = np.array([centroid for _, _, _, centroid in all_small_instances])
    distances, indices = tree.query(small_centroids)

    total_merged = 0
    for i, (inst_id, count, _volume, _centroid) in enumerate(all_small_instances):
        distance = distances[i]
        idx = indices[i]

        if distance > max_search_radius:
            if verbose:
                print(
                    f"    x Cluster {inst_id} ({count} pts) - "
                    f"no target within {max_search_radius}m"
                )
            continue

        target_inst = large_ids[idx]
        inst_to_target[inst_id] = target_inst
        total_merged += count
        if verbose:
            print(f"    + Cluster {inst_id} ({count} pts) -> Instance {target_inst} (dist: {distance:.1f}m)")

    valid_mask = (instances > 0) & (instances < max_inst)
    instances[valid_mask] = inst_to_target[instances[valid_mask]]

    print(
        f"  Merged/redistributed {total_merged:,} points from "
        f"{len(all_small_instances)} small instances "
        f"({len(small_volume_instances)} small-volume + "
        f"{len(small_point_count_instances)} small point-count)",
        flush=True,
    )

    return (instances, bbox_skipped_count)
