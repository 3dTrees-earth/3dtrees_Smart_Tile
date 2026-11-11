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
    for group in _cgroup_groups(root):
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


# Peak memory of each task (cgroup anonymous memory, GB; page cache is
# reclaimable): a fixed part plus each worker's share, measured on 8 Nov 2026
# with v2.5 and about 40 % margin. Tile keeps one block of up to 4 M points per
# worker (Brandenburg 1129/1134 at 10 workers: 4.3/4.7 GB);
# merge sizes its query caches to the limit, and its block-query and index
# processes each hold about 0.3 GB of libraries and one block (11 Nov, 10
# workers, no limit: 3753 3.8, 2779 3.0, 2585 2-tile 5.3 GB); remap
# streams 32 k-point batches (3753: fails in 2 GB, passes in 4 GB);
# create_merged_file picks up to 2 M points per worker (3808: 9.7 GB).
TASK_MEMORY_GB = {
    "tile": (0.5, 0.6),
    "merge": (0.5, 0.35),
    "remap": (0.5, 0.35),
    "create_merged_file": (0.5, 0.95),
}
# Exit code of a run that stopped for lack of memory, as for a kernel OOM kill;
# the Galaxy wrapper maps it to fatal_oom so the job is resubmitted with more.
OUT_OF_MEMORY_EXIT = 137


class MemoryBudgetError(RuntimeError):
    """Not even one worker fits the memory limit."""


def _cgroup_groups(root: Path) -> set[Path]:
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
    return groups


def memory_limit_bytes() -> int | None:
    """Smallest of GALAXY_MEMORY_MB, the cgroup memory limits and physical memory; None if none is known."""
    limits = []
    if os.environ.get("GALAXY_MEMORY_MB"):
        try:
            megabytes = float(os.environ["GALAXY_MEMORY_MB"])
            if megabytes > 0:
                limits.append(int(megabytes * 1024**2))
        except ValueError:
            pass
    root = Path("/sys/fs/cgroup")
    for path in [*(group / "memory.max" for group in _cgroup_groups(root)), root / "memory/memory.limit_in_bytes"]:
        try:
            value = int(path.read_text())
            if value > 0:
                limits.append(value)
        except (OSError, ValueError):
            pass
    try:
        limits.append(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError, AttributeError):
        pass
    return min(limits) if limits else None


def memory_capped_workers(tasks, workers: int, limit: int | None = None) -> int:
    """Workers that fit the memory limit when every task in ``tasks`` runs with them.

    A task's need is its fixed part plus one share per worker (TASK_MEMORY_GB).
    Raises MemoryBudgetError when one worker does not fit.
    """
    limit = memory_limit_bytes() if limit is None else limit
    workers = max(1, int(workers))
    if limit is None:
        return workers
    gb = limit / 1024**3
    fixed = max(TASK_MEMORY_GB[task][0] for task in tasks)
    share = max(TASK_MEMORY_GB[task][1] for task in tasks)
    fitting = int((gb - fixed) // share)
    if fitting < 1:
        raise MemoryBudgetError(
            f"Out of memory: {'/'.join(tasks)} needs about {fixed + share:.1f} GB for one worker, "
            f"{gb:.1f} GB available (GALAXY_MEMORY_MB, cgroup memory limit)")
    return min(workers, fitting)


def oom_kill_count() -> int:
    """Processes the kernel killed for memory in this cgroup so far (cgroup v2; 0 when unknown)."""
    for group in _cgroup_groups(Path("/sys/fs/cgroup")):
        try:
            for line in (group / "memory.events").read_text().splitlines():
                name, _, value = line.partition(" ")
                if name == "oom_kill" and int(value):
                    return int(value)
        except (OSError, ValueError):
            pass
    return 0


def failure_exit_code(exc: BaseException, oom_kills_before: int = 0) -> int:
    """OUT_OF_MEMORY_EXIT, with a message, when ``exc`` came from a lack of memory; 1 otherwise."""
    from concurrent.futures.process import BrokenProcessPool

    killed = oom_kill_count() - oom_kills_before
    if isinstance(exc, (MemoryError, MemoryBudgetError)) or (isinstance(exc, BrokenProcessPool) and killed > 0):
        limit = memory_limit_bytes()
        available = f"{limit / 1024**3:.1f} GB" if limit else "unknown"
        detail = f"; the kernel killed {killed} process(es) for memory" if killed > 0 else ""
        print(f"Error: out of memory (limit {available}{detail}). Rerun with more memory.")
        return OUT_OF_MEMORY_EXIT
    return 1
