"""Run independent per-tile work in processes and return results in tile order.

Per-tile stages are Python-bound (the GIL), so threads cannot overlap them.
Callers merge each result into shared state in tile order, which keeps every
output identical to a one-process run. Larger tiles start first.
"""
from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor


def map_tiles(function, jobs, *, processes, sizes=None):
    """Yield ``function(*job)`` for each job, in job order."""
    jobs = list(jobs)
    processes = min(int(processes), len(jobs))
    if processes <= 1:
        for job in jobs:
            yield function(*job)
        return
    with ProcessPoolExecutor(max_workers=processes, mp_context=multiprocessing.get_context("spawn")) as pool:
        order = sorted(range(len(jobs)), key=lambda t: (-(sizes[t] if sizes else 0), t))
        submitted = {tile: pool.submit(function, *jobs[tile]) for tile in order}
        try:
            for tile in range(len(jobs)):
                yield submitted[tile].result()
        finally:
            for future in submitted.values():
                future.cancel()
