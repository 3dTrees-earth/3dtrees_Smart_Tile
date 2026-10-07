"""Bounded, ordered remap queries against immutable disk-backed indexes.

Workers keep a cache of 4 m region search trees. Consecutive batches of an
original file revisit the same regions, so each batch is split by region and a
region always goes to the same worker: its tree is built once, not once per
worker, and every batch still spreads over all workers. Regions contain whole
2 m query cells, so each point's query group, and with it the result, equals a
one-process run.
"""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import numpy as np

from bounded_point_index import PointIndex
from spatial_query_cache import REGION_SIZE


_worker_indices = None
_worker_baselines = None
_worker_radius = None


def _query(xyz, indices, baselines, radius):
    results = []
    for index, baseline in zip(indices, baselines):
        base_distances = baseline.nearest_distances(xyz, radius)
        if 'PredInstance_RCT' in index.dimensions:
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


def _initialize_worker(index_specs, baseline_specs, radius):
    global _worker_indices, _worker_baselines, _worker_radius
    # Spawned processes open their own connections; no SQLite handle is shared.
    _worker_indices = [PointIndex(path, dims, query_workers=1, read_only=True)
                       for path, dims in index_specs]
    _worker_baselines = [PointIndex(path, dims, query_workers=1, read_only=True)
                         for path, dims in baseline_specs]
    _worker_radius = radius


def _worker_query(xyz):
    return _query(xyz, _worker_indices, _worker_baselines, _worker_radius)


def region_owners(xyz, workers):
    """Deterministic worker per 4 m cache region (each 2 m query cell lies in one region)."""
    regions = np.floor(xyz[:, :2] / REGION_SIZE).astype(np.int64)
    return ((regions[:, 0] * 73856093) ^ (regions[:, 1] * 19349663)) % workers


def _merge(parts, length):
    """Reassemble per-model (base distances, distances, values) from region parts."""
    merged = []
    first = parts[0][1]
    for model, (base, dist, values) in enumerate(first):
        base_all = np.empty(length, dtype=base.dtype)
        dist_all = np.empty(length, dtype=dist.dtype)
        values_all = {name: np.empty((length, *data.shape[1:]), dtype=data.dtype) for name, data in values.items()}
        for positions, result in parts:
            part_base, part_dist, part_values = result[model]
            base_all[positions], dist_all[positions] = part_base, part_dist
            for name, data in part_values.items():
                values_all[name][positions] = data
        merged.append((base_all, dist_all, values_all))
    return merged


class RemapBatchQueries:
    """Keep records in the parent; send only XYZ and receive prediction arrays.

    Indexes must be complete before entry and unchanged until exit. At most two
    batches per worker are admitted, including completed out-of-order results.
    """

    def __init__(self, indices, baselines, *, workers, radius):
        if workers < 1 or len(indices) != len(baselines):
            raise ValueError("Positive workers and paired remap indexes are required")
        self.indices, self.baselines = indices, baselines
        self.workers, self.radius = workers, radius
        self.max_pending = 2 * workers
        self.pools = []

    def __enter__(self):
        if self.workers > 1:
            for index in [*self.indices, *self.baselines]:
                index.flush()
            specs = lambda indexes: [(index.path, index.dimensions) for index in indexes]
            context = multiprocessing.get_context("spawn")
            # One single-process pool per worker: a region is always answered by
            # the same process, which keeps that region's trees cached.
            self.pools = [ProcessPoolExecutor(max_workers=1, mp_context=context, initializer=_initialize_worker,
                                              initargs=(specs(self.indices), specs(self.baselines), self.radius))
                          for _ in range(self.workers)]
        return self

    def __exit__(self, *args):
        # Join workers before the caller deletes the temporary index files.
        for pool in self.pools:
            pool.shutdown(wait=True, cancel_futures=True)

    def map(self, batches):
        """Yield (record, XYZ, per-model results) in original input order."""
        if not self.pools:
            for record, xyz in batches:
                yield record, xyz, _query(xyz, self.indices, self.baselines, self.radius)
            return
        pending = deque()
        iterator = iter(batches)
        exhausted = False
        try:
            while pending or not exhausted:
                while not exhausted and len(pending) < self.max_pending:
                    try:
                        record, xyz = next(iterator)
                    except StopIteration:
                        exhausted = True
                        break
                    owners = region_owners(xyz, self.workers)
                    parts = [(positions, self.pools[worker].submit(_worker_query, xyz[positions]))
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
