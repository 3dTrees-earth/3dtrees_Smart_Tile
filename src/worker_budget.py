#!/usr/bin/env python3
"""Shared worker-budget helpers for nested SmartTile parallelism."""

from __future__ import annotations

import os
from pathlib import Path


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


def spatial_query_worker_count(requested_workers: int) -> int:
    """Cap explicit query threads to scheduler slots, CPU affinity and quota.

    Separate from file-worker defaults. Merge/filter use this for native query
    threads; standalone final remap uses it for processes with one query thread.
    """
    requested = int(requested_workers)
    if requested < 1:
        raise ValueError("Query workers must be positive")
    limits = [requested, available_cpu_count()]
    try:
        limits.append(len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        pass
    if os.environ.get("GALAXY_SLOTS"):
        slots = int(os.environ["GALAXY_SLOTS"])
        if slots < 1:
            raise ValueError("GALAXY_SLOTS must be positive")
        limits.append(slots)
    # Docker's cgroup namespace exposes its quota at the root. On a host,
    # include the process cgroup and its ancestors as well.
    root = Path("/sys/fs/cgroup")
    groups = {root}
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            if line.startswith("0::"):
                relative = Path(line[3:].lstrip("/"))
                if ".." not in relative.parts:
                    group = root / relative
                    groups.update([group, *[p for p in group.parents if p == root or root in p.parents]])
    except OSError:
        pass
    for group in groups:
        try:
            quota, period = (group / "cpu.max").read_text().split()
            if quota != "max":
                limits.append(max(1, int(quota) // int(period)))
        except (OSError, ValueError, ZeroDivisionError):
            pass
    # Compatibility with hosts using the cgroup v1 CPU controller.
    try:
        cpu = root / "cpu"
        quota = int((cpu / "cpu.cfs_quota_us").read_text())
        period = int((cpu / "cpu.cfs_period_us").read_text())
        if quota > 0 and period > 0:
            limits.append(max(1, quota // period))
    except (OSError, ValueError):
        pass
    return max(1, min(limits))
