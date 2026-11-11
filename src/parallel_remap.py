"""Bounded, ordered remap queries against immutable disk-backed indexes.

Workers keep a cache of 4 m region search trees. Consecutive batches of an
original file revisit the same regions, so each batch is split by region and a
region always goes to the same worker: its tree is built once, not once per
worker, and every batch still spreads over all workers. Regions are assigned by
their indexed point count, largest first onto the least-loaded worker, since
dense terrestrial scans hold millions of points in a few regions. Regions contain whole
2 m query cells, so each point's query group, and with it the result, equals a
one-process run.
"""
from collections import deque
import heapq
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import numpy as np

from bounded_point_index import PointIndex
from spatial_query_cache import REGION_SIZE, SUBCELL_SIZE


_worker_indices = None
_worker_baselines = None
_worker_radius = None
_worker_matches = None
_worker_prefer_rct_trees = True


def _query(xyz, indices, baselines, radius, matches=None, labels=None, prefer_rct_trees=True):
    """Per model: (baseline distances, distances, values).

    ``matches[i]`` names an instance dimension: model i then only matches
    prediction points with the query's own label (``labels[name]``).
    """
    results = []
    matches = matches or [None] * len(indices)
    radii = [radius] * len(indices) if np.isscalar(radius) else radius
    for index, baseline, match, radius in zip(indices, baselines, matches, radii):
        base_distances = None if baseline is None else baseline.nearest_distances(xyz, radius)
        if match is not None:
            distances, values, _ = index.nearest_matching(xyz, radius, match, labels[match])
        elif prefer_rct_trees and 'PredInstance_RCT' in index.dimensions:
            # A retained/recovered tree wins over neighboring background. This
            # affects original enrichment only; tile memberships stay intact.
            distances, values, _ = index.nearest(xyz, radius, positive_dimension='PredInstance_RCT')
            missing = ~np.isfinite(distances)
            if np.any(missing):
                background_distances, background_values, _ = index.nearest(xyz[missing], radius)
                distances[missing] = background_distances
                for name in values:
                    values[name][missing] = background_values[name]
        else:
            distances, values, _ = index.nearest(xyz, radius)
        results.append((base_distances, distances, values))
    return results


def _initialize_worker(index_specs, baseline_specs, radius, matches=None, prefer_rct_trees=True):
    global _worker_indices, _worker_baselines, _worker_radius, _worker_matches, _worker_prefer_rct_trees
    # Spawned processes open their own connections; no SQLite handle is shared.
    _worker_indices = [PointIndex(path, dims, query_workers=1, read_only=True)
                       for path, dims in index_specs]
    _worker_baselines = [None if spec is None else PointIndex(*spec, query_workers=1, read_only=True)
                         for spec in baseline_specs]
    _worker_radius = radius
    _worker_matches = matches
    _worker_prefer_rct_trees = prefer_rct_trees


def _worker_stats():
    from spatial_query_cache import shared_cache
    return shared_cache().stats()


def _worker_query(xyz, labels=None):
    return _query(xyz, _worker_indices, _worker_baselines, _worker_radius, _worker_matches, labels,
                  _worker_prefer_rct_trees)


def _keys(cells):
    """One sortable int64 per XY cell."""
    return cells[:, 0] * (1 << 32) + (cells[:, 1] + (1 << 31))


def _lookup(keys, table):
    """Worker of each key in (sorted keys, workers), -1 where absent."""
    sorted_keys, workers = table
    if not len(sorted_keys):
        return np.full(len(keys), -1, dtype=np.int64)
    at = np.minimum(np.searchsorted(sorted_keys, keys), len(sorted_keys) - 1)
    return np.where(sorted_keys[at] == keys, workers[at], -1)


def region_owners(xyz, workers, assigned=None):
    """Deterministic worker per 4 m cache region (each 2 m query cell lies in one region).

    ``assigned`` ({"regions": table, "cells": table} of balanced_regions)
    places indexed regions and the 2 m query cells of split dense regions;
    others are hashed.
    """
    regions = np.floor(xyz[:, :2] / REGION_SIZE).astype(np.int64)
    owners = ((regions[:, 0] * 73856093) ^ (regions[:, 1] * 19349663)) % workers
    if assigned:
        lookup = _lookup(_keys(regions), assigned["regions"])
        cells = _lookup(_keys(np.floor(xyz[:, :2] / SUBCELL_SIZE).astype(np.int64)), assigned["cells"])
        lookup = np.where(cells >= 0, cells, lookup)
        owners = np.where(lookup >= 0, lookup, owners)
    return owners


def balanced_regions(indexes, workers):
    """{4 m region: worker} for the regions of ``indexes``: largest first onto the least-loaded worker.

    A region's weight is its indexed point count over all indexes; ties go by
    region, so the assignment is deterministic. A region above a quarter of an
    even worker share is placed as its four 2 m query cells instead, so one
    dense spot near a scanner does not hold one worker while the others wait.
    Any fixed assignment keeps results identical: a query cell is still
    answered by one process.
    """
    weights, cells = {}, {}
    for index in indexes:
        if index is None:
            continue
        counts = index.cell_points
        if counts is None:
            # Built from shards: index blobs are single 2 m cells, so each
            # blob's lower corner names its cell.
            size = index._storage_dtype().itemsize
            rows = np.array(index.db.execute(
                'SELECT r.x0, r.y0, length(b.data) FROM bounds r JOIN batches b ON b.id=r.id').fetchall(),
                dtype=np.float64).reshape(-1, 3)
            keys, inverse = np.unique(np.floor(rows[:, :2] / SUBCELL_SIZE).astype(np.int64), axis=0, return_inverse=True)
            counts = dict(zip(map(tuple, keys.tolist()), np.bincount(inverse.ravel(), weights=rows[:, 2] / size).tolist()))
        for (cx, cy), n in counts.items():
            cells[(cx, cy)] = cells.get((cx, cy), 0) + n
            region = (cx // 2, cy // 2)  # REGION_SIZE is two query cells
            weights[region] = weights.get(region, 0) + n
    limit = sum(weights.values()) / (4 * workers)
    for region in [r for r, weight in weights.items() if weight > limit]:
        del weights[region]
        for dx in (0, 1):
            for dy in (0, 1):
                cell = (2 * region[0] + dx, 2 * region[1] + dy)
                if cell in cells:
                    weights[("cell", *cell)] = cells[cell]
    load = [(0.0, w) for w in range(workers)]  # heap of (load, worker)
    placed = {"regions": {}, "cells": {}}
    for unit, weight in sorted(weights.items(), key=lambda item: (-item[1], repr(item[0]))):
        total, worker = heapq.heappop(load)
        if unit[0] == "cell":
            placed["cells"][unit[1:]] = worker
        else:
            placed["regions"][unit] = worker
        heapq.heappush(load, (total + weight, worker))
    assigned = {}
    for kind, table in placed.items():
        keys = _keys(np.array(sorted(table), dtype=np.int64).reshape(-1, 2))
        order = np.argsort(keys)
        assigned[kind] = (keys[order], np.array([table[k] for k in sorted(table)], dtype=np.int64)[order])
    return assigned


def _merge(parts, length):
    """Reassemble per-model (base distances, distances, values) from region parts."""
    merged = []
    first = parts[0][1]
    for model, (base, dist, values) in enumerate(first):
        base_all = None if base is None else np.empty(length, dtype=base.dtype)
        dist_all = np.empty(length, dtype=dist.dtype)
        values_all = {name: np.empty((length, *data.shape[1:]), dtype=data.dtype) for name, data in values.items()}
        for positions, result in parts:
            part_base, part_dist, part_values = result[model]
            if base_all is not None:
                base_all[positions] = part_base
            dist_all[positions] = part_dist
            for name, data in part_values.items():
                values_all[name][positions] = data
        merged.append((base_all, dist_all, values_all))
    return merged


class RemapBatchQueries:
    """Keep records in the parent; send only XYZ and receive prediction arrays.

    Indexes must be complete before entry and unchanged until exit. A baseline
    may be None, which skips that model's baseline distances. At most eight
    batches per worker are admitted, including completed out-of-order results.
    """

    def __init__(self, indices, baselines, *, workers, radius, matches=None, prefer_rct_trees=True):
        if workers < 1 or len(indices) != len(baselines):
            raise ValueError("Positive workers and paired remap indexes are required")
        if not np.isscalar(radius) and len(radius) != len(indices):
            raise ValueError("Provide one original remap radius per model")
        self.indices, self.baselines = indices, baselines
        self.matches = matches
        self.prefer_rct_trees = prefer_rct_trees
        self.workers, self.radius = workers, radius
        # Batches follow the scan; a few dense cells can hold one worker for
        # several batches, so admit enough that the others keep working.
        self.max_pending = 8 * workers
        self.pools = []
        self.assigned = None
        self.cache_stats = None
        self.points_per_worker = [0] * workers

    def __enter__(self):
        if self.workers > 1:
            for index in [*self.indices, *self.baselines]:
                if index is not None:
                    index.flush()
            specs = lambda indexes: [None if index is None else (index.path, index.dimensions) for index in indexes]
            self.assigned = balanced_regions([*self.indices, *self.baselines], self.workers)
            context = multiprocessing.get_context("spawn")
            # One single-process pool per worker: a region is always answered by
            # the same process, which keeps that region's trees cached.
            self.pools = [ProcessPoolExecutor(max_workers=1, mp_context=context, initializer=_initialize_worker,
                                              initargs=(specs(self.indices), specs(self.baselines), self.radius,
                                                        self.matches, self.prefer_rct_trees))
                          for _ in range(self.workers)]
        return self

    def __exit__(self, exc_type, *args):
        if exc_type is None and self.pools:
            stats = [pool.submit(_worker_stats).result() for pool in self.pools]
            self.cache_stats = {name: sum(s[name] for s in stats) for name in stats[0]}
            self.cache_stats.update(max_bytes=stats[0]["max_bytes"], points_per_worker=self.points_per_worker)
        # Join workers before the caller deletes the temporary index files.
        for pool in self.pools:
            pool.shutdown(wait=True, cancel_futures=True)

    def map(self, batches):
        """Yield (record, XYZ, per-model results) in original input order.

        Batches are (record, XYZ) or, with ``matches``, (record, XYZ, labels).
        """
        if not self.pools:
            for record, xyz, *labels in batches:
                yield record, xyz, _query(xyz, self.indices, self.baselines, self.radius, self.matches,
                                          labels[0] if labels else None, self.prefer_rct_trees)
            return
        pending = deque()
        iterator = iter(batches)
        exhausted = False
        try:
            while pending or not exhausted:
                while not exhausted and len(pending) < self.max_pending:
                    try:
                        record, xyz, *labels = next(iterator)
                    except StopIteration:
                        exhausted = True
                        break
                    labels = labels[0] if labels else None
                    owners = region_owners(xyz, self.workers, self.assigned)
                    self.points_per_worker = [n + int(c) for n, c in
                                              zip(self.points_per_worker, np.bincount(owners, minlength=self.workers))]
                    parts = [(positions, self.pools[worker].submit(
                                 _worker_query, xyz[positions],
                                 None if labels is None else {n: v[positions] for n, v in labels.items()}))
                             for worker in np.unique(owners)
                             for positions in [np.flatnonzero(owners == worker)]]
                    pending.append((record, xyz, parts))
                if pending:
                    record, xyz, parts = pending.popleft()
                    results = [(positions, future.result()) for positions, future in parts]
                    yield record, xyz, (_merge(results, len(xyz)) if len(results) > 1 else results[0][1])
        finally:
            for _, _, parts in pending:
                for _, future in parts:
                    future.cancel()
