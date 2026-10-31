#!/usr/bin/env python3
"""Shared worker-budget helpers for nested SmartTile parallelism."""

from __future__ import annotations

import os
from pathlib import Path


# BLAS/OpenMP pools that otherwise start one thread per host core on import.
NATIVE_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
WORKERS_FLAGS = ("--workers", "--number-of-threads", "--number_of_threads")


def requested_workers(argv) -> int | None:
    """Read --workers from the command line before numpy or laspy load; None if absent or invalid."""
    for index, arg in enumerate(argv):
        name, _, value = arg.partition("=")
        if name in WORKERS_FLAGS:
            if not value and index + 1 < len(argv):
                value = argv[index + 1]
            try:
                return int(value) if int(value) > 0 else None
            except ValueError:
                return None
    return None


def _cap(name: str, limit: int) -> None:
    try:
        configured = int(os.environ.get(name, limit))
    except ValueError:
        configured = limit
    os.environ[name] = str(min(limit, configured) if configured > 0 else limit)


def limit_native_threads(workers: int, *, worker_process: bool) -> None:
    """Keep native thread pools within --workers; call before numpy, laspy or lazrs load.

    SmartTile parallelises with processes, so BLAS/OpenMP pools get one thread in
    every process: on dataset 2024 their per-process pools otherwise started
    3,114 threads at --workers 16, 1,423 of them running at once. LAZ
    compression (lazrs, rayon) may use ``workers`` threads in the main process and
    one in each worker process. Smaller caps already set in the environment stay.
    """
    for name in NATIVE_THREAD_VARIABLES:
        _cap(name, 1)
    _cap("RAYON_NUM_THREADS", 1 if worker_process else max(1, int(workers)))


def available_cpu_count() -> int:
    """Return the available CPU count, falling back to one."""
    return max(1, os.cpu_count() or 1)


def allocated_cpu_count() -> int:
    """CPUs this process may use: scheduler slots, affinity and cgroup quota."""
    return spatial_query_worker_count(available_cpu_count())


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
