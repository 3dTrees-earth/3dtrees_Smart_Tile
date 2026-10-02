"""Answer index queries (nearest or covered) for a stream of batches in worker processes.

The per-cell searches are Python and hold the GIL, so threads cannot
speed it up. Workers open the finished index read-only; the parent keeps
reading, writing and indexing, and receives results in input order, so output
equals a one-process run. At most two batches per worker are in flight.
"""
from __future__ import annotations

import multiprocessing
from collections import deque
from concurrent.futures import ProcessPoolExecutor

from bounded_point_index import PointIndex

_query = None


def _initialize(path, dimensions, method, radius, options):
    global _query
    # Spawned processes open their own connection; no SQLite handle is shared.
    index = PointIndex(path, dimensions, query_workers=1, read_only=True)
    method = getattr(index, method)
    _query = lambda xyz: method(xyz, radius, **options)


def _run(xyz):
    return _query(xyz)


class IndexQueries:
    """``index.<method>(xyz, radius, **options)`` per batch; the index must be complete."""

    def __init__(self, index, radius, *, workers, method="nearest", **options):
        if workers < 1:
            raise ValueError("Index queries need at least one worker")
        if method not in ("nearest", "covered"):
            raise ValueError(f"Unsupported index query: {method}")
        self.index, self.radius, self.workers = index, radius, workers
        self.method, self.options = method, options
        self.pool = None

    def __enter__(self):
        if self.workers > 1:
            self.index.flush()
            self.pool = ProcessPoolExecutor(
                max_workers=self.workers, mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize,
                initargs=(self.index.path, self.index.dimensions, self.method, self.radius, self.options))
        return self

    def __exit__(self, *args):
        if self.pool is not None:
            # Join workers before the caller deletes the index file.
            self.pool.shutdown(wait=True, cancel_futures=True)

    def map(self, batches):
        """Yield ``(item, xyz, result)`` in input order."""
        if self.pool is None:
            query = getattr(self.index, self.method)
            for item, xyz in batches:
                yield item, xyz, query(xyz, self.radius, **self.options)
            return
        pending = deque()
        iterator = iter(batches)
        exhausted = False
        try:
            while pending or not exhausted:
                while not exhausted and len(pending) < 2 * self.workers:
                    try:
                        item, xyz = next(iterator)
                    except StopIteration:
                        exhausted = True
                        break
                    pending.append((item, xyz, self.pool.submit(_run, xyz)))
                if pending:
                    item, xyz, future = pending.popleft()
                    yield item, xyz, future.result()
        finally:
            for _, _, future in pending:
                future.cancel()
