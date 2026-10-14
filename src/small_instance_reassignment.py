"""Reassign small final instances to the nearest remaining instance.

Runs on reconciled, deduplicated 1 cm tiles, so every instance is counted once
across tiles. An instance is small when it has fewer than ``max_cluster_size``
points and its axis-aligned bounding box is below ``max_volume_m3``. Small
instances take the ID of the non-small instance with the nearest XY centroid
(horizontal distance, height ignored) within ``SEARCH_RADIUS_M``; otherwise
they keep their ID. Only instance IDs change: geometry, semantics and scores
stay attached to their points.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import laspy
import numpy as np
from scipy.spatial import cKDTree

from bounded_point_index import MAX_BATCH_POINTS
from instance_statistics import InstanceStatistics
from point_cloud_metadata import copy_single_source_header, write_retained_evlrs

SEARCH_RADIUS_M = 5.0
REPORTED_DECISIONS = 1000


@dataclass(frozen=True)
class SmallInstancePolicy:
    max_cluster_size: int = 3000
    max_volume_m3: float = 4.0

    def __post_init__(self):
        if self.max_cluster_size < 1:
            raise ValueError("max_cluster_size must be at least 1")
        if not (np.isfinite(self.max_volume_m3) and self.max_volume_m3 > 0):
            raise ValueError("max_volume_for_merge must be a finite positive volume")


def plan_reassignment(stats: InstanceStatistics, policy: SmallInstancePolicy):
    """Return ``{small_id: target_id}`` and the report of every decision."""
    volumes = stats.bbox_volumes
    small = (stats.counts < policy.max_cluster_size) & (volumes < policy.max_volume_m3)
    mapping, reassigned, kept = {}, [], []
    targets = np.flatnonzero(~small)
    if np.any(small) and len(targets):
        centroids = stats.xy_centroids
        distances, nearest = cKDTree(centroids[targets]).query(centroids[small])
        for row, distance, target in zip(np.flatnonzero(small), distances, targets[nearest]):
            decision = {"instance": int(stats.ids[row]), "points": int(stats.counts[row]),
                        "bbox_volume_m3": float(volumes[row]), "xy_distance_m": float(distance)}
            if distance <= SEARCH_RADIUS_M:
                mapping[decision["instance"]] = int(stats.ids[target])
                reassigned.append(dict(decision, target=int(stats.ids[target])))
            else:
                kept.append(dict(decision, reason=f"no non-small instance within {SEARCH_RADIUS_M} m"))
    else:
        kept = [{"instance": int(i), "reason": "no non-small instance"} for i in stats.ids[small]]
    moved = np.isin(stats.ids, list(mapping)) if mapping else np.zeros(len(stats.ids), dtype=bool)
    report = {"policy": "fewer than max_cluster_size points and bounding box below max_volume_m3; "
                        "nearest non-small instance XY centroid (horizontal distance) within search_radius_m",
              "max_cluster_size": policy.max_cluster_size, "max_volume_m3": policy.max_volume_m3,
              "search_radius_m": SEARCH_RADIUS_M, "instances": int(len(stats.ids)), "small": int(small.sum()),
              "reassigned_count": len(mapping), "reassigned_points": int(stats.counts[moved].sum()),
              "reassigned": reassigned[:REPORTED_DECISIONS], "kept": kept[:REPORTED_DECISIONS]}
    return mapping, report


def relabel_tiles(files, instance_dimension, mapping):
    """Rewrite instance IDs in place; each file is replaced only after a complete write."""
    source = np.fromiter(sorted(mapping), dtype=np.int64)
    target = np.fromiter((mapping[s] for s in sorted(mapping)), dtype=np.int64)
    changed = []
    for file in files:
        file = Path(file)
        temporary = file.with_name(f".{file.name}.relabel")
        points = 0
        with laspy.open(file) as reader:
            header = copy_single_source_header(reader.header)
            with laspy.open(temporary, mode="w", header=header) as writer:
                for record in reader.chunk_iterator(MAX_BATCH_POINTS):
                    labels = np.asarray(record[instance_dimension])
                    hit = np.isin(labels, source)
                    if np.any(hit):
                        relabeled = labels.copy()
                        relabeled[hit] = target[np.searchsorted(source, labels[hit])]
                        record[instance_dimension] = relabeled.astype(labels.dtype)
                        points += int(hit.sum())
                    writer.write_points(record)
                write_retained_evlrs(writer, header)
        os.replace(temporary, file)
        changed.append(points)
    return changed


def reassign_small_instances(stats, policy, files, instance_dimension, report):
    """Plan, record and apply the reassignment; returns ``{source: target}``."""
    mapping, report["small_instance_reassignment"] = plan_reassignment(stats, policy)
    if mapping:
        report["small_instance_reassignment"]["relabeled_points_per_tile"] = relabel_tiles(
            files, instance_dimension, mapping)
    return mapping
