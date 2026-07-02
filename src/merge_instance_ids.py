#!/usr/bin/env python3
"""Global instance ID helpers for SmartTile merge stages."""

from __future__ import annotations

from typing import Tuple


TILE_OFFSET = 100000


def global_id(tile_idx: int, local_id: int) -> int:
    """Return the unique global ID for one local tile instance."""
    return tile_idx * TILE_OFFSET + local_id


def local_id(gid: int) -> Tuple[int, int]:
    """Return tile index and local instance ID from a global instance ID."""
    return gid // TILE_OFFSET, gid % TILE_OFFSET
