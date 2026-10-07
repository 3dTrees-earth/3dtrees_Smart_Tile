#!/usr/bin/env python3
"""Shared worker-budget helpers for nested SmartTile parallelism."""

from __future__ import annotations

import os
from pathlib import Path


DEFAULT_FILE_WORKERS = 2
BYTES_PER_GB = 1024**3


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
