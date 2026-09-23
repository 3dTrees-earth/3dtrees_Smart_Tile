"""Bounded, ordered remap queries against immutable disk-backed indexes."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

from bounded_point_index import PointIndex


_worker_indices = None
_worker_baselines = None
_worker_radius = None


def _query(xyz, indices, baselines, radius):
    results = []
    for index, baseline in zip(indices, baselines):
        base_distances, _, _ = baseline.nearest(xyz, radius)
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
        self.pool = None

    def __enter__(self):
        if self.workers > 1:
            for index in [*self.indices, *self.baselines]:
                index.flush()
            specs = lambda indexes: [(index.path, index.dimensions) for index in indexes]
            self.pool = ProcessPoolExecutor(
                max_workers=self.workers, mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize_worker,
                initargs=(specs(self.indices), specs(self.baselines), self.radius))
        return self

    def __exit__(self, *args):
        if self.pool is not None:
            # Join workers before the caller deletes the temporary index files.
            self.pool.shutdown(wait=True, cancel_futures=True)

    def map(self, batches):
        """Yield (record, XYZ, per-model results) in original input order."""
        if self.pool is None:
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
                    pending.append((record, xyz, self.pool.submit(_worker_query, xyz)))
                if pending:
                    record, xyz, future = pending.popleft()
                    yield record, xyz, future.result()
        finally:
            for _, _, future in pending:
                future.cancel()
