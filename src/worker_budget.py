#!/usr/bin/env python3
"""Shared worker-budget helpers for nested SmartTile parallelism."""

from __future__ import annotations


def kdtree_query_workers(total_workers: int, outer_workers: int) -> int:
    """Return per-task cKDTree query workers without oversubscribing CPUs."""
    total = max(1, int(total_workers or 1))
    outer = max(1, int(outer_workers or 1))
    return max(1, total // outer)
