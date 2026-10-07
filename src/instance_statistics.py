"""Per-instance statistics of final tiles and the ``<instance dimension>_summary.json``.

Statistics are streamed from the records a writer already emits (deduplication,
RCT namespacing), so no extra pass over the tiles is needed.
"""
from __future__ import annotations


import numpy as np


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

    @property
    def bbox_volumes(self):
        return np.prod(self.hi - self.lo, axis=1) if len(self.ids) else np.empty(0)

    @property
    def xy_centroids(self):
        return self.sums[:, :2] / self.counts[:, None]


def instance_summary(stats: InstanceStatistics, aliases=None, *, origin=None):
    """Final per-instance rows after folding ``{source: target}`` aliases into their targets."""
    aliases = aliases or {}
    offset = np.zeros(3) if origin is None else np.asarray(origin, dtype=np.float64)[:3]
    final = np.fromiter((aliases.get(int(i), int(i)) for i in stats.ids), dtype=np.int64, count=len(stats.ids))
    labels, group = np.unique(final, return_inverse=True)
    counts = np.zeros(len(labels), dtype=np.int64)
    lo, hi = np.full((len(labels), 3), np.inf), np.full((len(labels), 3), -np.inf)
    sums = np.zeros((len(labels), 3))
    np.add.at(counts, group, stats.counts)
    np.minimum.at(lo, group, stats.lo)
    np.maximum.at(hi, group, stats.hi)
    np.add.at(sums, group, stats.sums)
    merged = {}
    for source, target in aliases.items():
        merged.setdefault(int(target), []).append(int(source))
    return [{"id": int(label), "points": int(count),
             "bbox_min": (low + offset).tolist(), "bbox_max": (high + offset).tolist(),
             "bbox_volume_m3": float(np.prod(high - low)),
             "centroid": (total / count + offset).tolist(),
             "centroid_xy": (total[:2] / count + offset[:2]).tolist(),
             "reassigned_from": sorted(merged.get(int(label), []))}
            for label, count, low, high, total in zip(labels, counts, lo, hi, sums)]


def summary_file_name(instance_dimension: str) -> str:
    return f"{instance_dimension}_summary.json"


def summary_document(model, stats, aliases, *, origin, resolution_1, id_encoding, reassignment_enabled):
    return {"model": model.name, "instance_dimension": model.instance,
            "resolution_1_m": resolution_1, "coordinates": "source CRS", "id_encoding": id_encoding,
            "small_instance_reassignment": reassignment_enabled,
            "instances": instance_summary(stats, aliases, origin=origin)}
