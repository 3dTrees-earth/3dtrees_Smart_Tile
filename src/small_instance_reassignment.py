"""Reassign small final instances to the nearest remaining instance.

Runs on reconciled, deduplicated 1 cm tiles, so every instance is counted once
across tiles. An instance is small when it has fewer than ``max_cluster_size``
points and its axis-aligned bounding box is below ``max_volume_m3``. Small
instances take the ID of the non-small instance with the nearest XY centroid
(horizontal distance, height ignored) within ``SEARCH_RADIUS_M``; otherwise
they keep their ID. Only instance IDs
change: geometry, semantics and scores stay attached to their points.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import laspy
import numpy as np
from scipy.spatial import cKDTree

from bounded_point_index import MAX_BATCH_POINTS
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


class InstanceStatistics:
    """Stream per-instance point count, XYZ bounds and coordinate sums."""

    def __init__(self):
        self.ids = np.empty(0, dtype=np.int64)
        self.counts = np.empty(0, dtype=np.int64)
        self.lo = np.empty((0, 3))
        self.hi = np.empty((0, 3))
        self.sums = np.empty((0, 3))

    def add(self, ids, xyz):
        positive = np.asarray(ids) > 0
        if not np.any(positive):
            return
        ids, xyz = np.asarray(ids)[positive].astype(np.int64), np.asarray(xyz, dtype=np.float64)[positive]
        batch, inverse = np.unique(ids, return_inverse=True)
        new = np.setdiff1d(batch, self.ids, assume_unique=True)
        if len(new):
            self.ids = np.concatenate([self.ids, new])
            self.counts = np.concatenate([self.counts, np.zeros(len(new), dtype=np.int64)])
            self.lo = np.vstack([self.lo, np.full((len(new), 3), np.inf)])
            self.hi = np.vstack([self.hi, np.full((len(new), 3), -np.inf)])
            self.sums = np.vstack([self.sums, np.zeros((len(new), 3))])
            order = np.argsort(self.ids, kind="stable")
            self.ids, self.counts = self.ids[order], self.counts[order]
            self.lo, self.hi, self.sums = self.lo[order], self.hi[order], self.sums[order]
        rows = np.searchsorted(self.ids, batch)[inverse]
        np.add.at(self.counts, rows, 1)
        np.minimum.at(self.lo, rows, xyz)
        np.maximum.at(self.hi, rows, xyz)
        np.add.at(self.sums, rows, xyz)


def plan_reassignment(stats: InstanceStatistics, policy: SmallInstancePolicy, *, origin=None):
    """Return ``{small_id: target_id}`` and the report of every decision."""
    volumes = np.prod(stats.hi - stats.lo, axis=1) if len(stats.ids) else np.empty(0)
    small = (stats.counts < policy.max_cluster_size) & (volumes < policy.max_volume_m3)
    report = {"policy": "fewer than max_cluster_size points and bounding box below max_volume_m3; "
                        "nearest non-small instance XY centroid (horizontal distance) within search_radius_m",
              "max_cluster_size": policy.max_cluster_size, "max_volume_m3": policy.max_volume_m3,
              "search_radius_m": SEARCH_RADIUS_M, "instances": int(len(stats.ids)),
              "small": int(small.sum()), "reassigned": [], "kept": []}
    if not np.any(small) or np.all(small):
        report["kept"] = [{"instance": int(i), "reason": "no non-small instance"}
                          for i in stats.ids[small][:REPORTED_DECISIONS]] if np.any(small) else []
        return {}, report
    centroids = stats.sums[:, :2] / stats.counts[:, None]
    targets = np.flatnonzero(~small)
    distances, nearest = cKDTree(centroids[targets]).query(centroids[small])
    mapping = {}
    for row, distance, target in zip(np.flatnonzero(small), distances, targets[nearest]):
        decision = {"instance": int(stats.ids[row]), "points": int(stats.counts[row]),
                    "bbox_volume_m3": float(volumes[row]), "xy_distance_m": float(distance)}
        if distance <= SEARCH_RADIUS_M:
            mapping[int(stats.ids[row])] = int(stats.ids[target])
            decision["target"] = int(stats.ids[target])
            bucket = report["reassigned"]
        else:
            decision["reason"] = f"no non-small instance within {SEARCH_RADIUS_M} m"
            bucket = report["kept"]
        if len(bucket) < REPORTED_DECISIONS:
            bucket.append(decision)
    report["reassigned_count"] = len(mapping)
    report["reassigned_points"] = int(sum(stats.counts[np.searchsorted(stats.ids, list(mapping))])) if mapping else 0
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


def instance_summary(stats: InstanceStatistics, mapping=None, *, origin=None):
    """Final per-instance summary after an optional ``{source: target}`` relabel."""
    mapping = mapping or {}
    offset = np.zeros(3) if origin is None else np.asarray(origin, dtype=np.float64)[:3]
    final = np.array([mapping.get(int(i), int(i)) for i in stats.ids], dtype=np.int64)
    instances = []
    for label in np.unique(final):
        rows = final == label
        count = int(stats.counts[rows].sum())
        lo, hi = stats.lo[rows].min(axis=0), stats.hi[rows].max(axis=0)
        instances.append({
            "id": int(label), "points": count,
            "bbox_min": (lo + offset).tolist(), "bbox_max": (hi + offset).tolist(),
            "bbox_volume_m3": float(np.prod(hi - lo)),
            "centroid": (stats.sums[rows].sum(axis=0) / count + offset).tolist(),
            "centroid_xy": (stats.sums[rows].sum(axis=0)[:2] / count + offset[:2]).tolist(),
            "reassigned_from": sorted(int(i) for i in stats.ids[rows] if int(i) != label),
        })
    return instances
