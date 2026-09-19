#!/usr/bin/env python3
"""Shared worker-budget helpers for nested SmartTile parallelism."""

from __future__ import annotations

import os


DEFAULT_FILE_WORKERS = 2
DEFAULT_MEMORY_GB = 4.0
BYTES_PER_GB = 1024**3


def available_cpu_count() -> int:
    """Return the available CPU count, falling back to one."""
    return max(1, os.cpu_count() or 1)


def kdtree_query_workers(total_workers: int, outer_workers: int) -> int:
    """Return per-task cKDTree query workers without oversubscribing CPUs."""
    total = max(1, int(total_workers or 1))
    outer = max(1, int(outer_workers or 1))
    return max(1, total // outer)


def file_worker_count(requested_workers: int | None, item_count: int | None = None) -> int:
    """Return file-level workers, defaulting to two concurrent files."""
    workers = max(1, int(requested_workers or DEFAULT_FILE_WORKERS))
    if item_count is not None:
        workers = min(workers, max(1, int(item_count)))
    return workers


def memory_limited_worker_count(
    requested_workers: int | None,
    item_count: int | None = None,
    *,
    bytes_per_worker: int | None = None,
    memory_gb: float | None = DEFAULT_MEMORY_GB,
) -> int:
    """Return a worker count capped by explicit memory in GiB."""
    workers = file_worker_count(requested_workers, item_count)
    if not bytes_per_worker or bytes_per_worker <= 0 or memory_gb is None:
        return workers
    budget = max(1, int(float(memory_gb) * BYTES_PER_GB))
    memory_workers = max(1, budget // int(bytes_per_worker))
    return max(1, min(workers, memory_workers))
