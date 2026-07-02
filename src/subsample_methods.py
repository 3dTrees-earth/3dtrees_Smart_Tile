#!/usr/bin/env python3
"""Shared SmartTile subsampling method policy."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


SUBSAMPLING_METHOD_CENTER_OF_MASS = "center-of-mass"
SUBSAMPLING_METHOD_NEAREST_TO_CENTROID = "nearest-to-centroid"
SUBSAMPLING_METHODS = {
    SUBSAMPLING_METHOD_CENTER_OF_MASS,
    SUBSAMPLING_METHOD_NEAREST_TO_CENTROID,
}


def normalize_subsampling_method(method: Optional[str]) -> str:
    """Return a canonical SmartTile subsampling method name."""
    normalized = (method or SUBSAMPLING_METHOD_CENTER_OF_MASS).strip().lower()
    if normalized in {"centroid", "voxelcentroidnearestneighbor", "voxel-centroid-nearest-neighbor"}:
        return SUBSAMPLING_METHOD_NEAREST_TO_CENTROID
    if normalized in {"com", "center_of_mass", "center-of-mass"}:
        return SUBSAMPLING_METHOD_CENTER_OF_MASS
    if normalized not in SUBSAMPLING_METHODS:
        allowed = ", ".join(sorted(SUBSAMPLING_METHODS))
        raise ValueError(f"Unknown subsampling method {method!r}; expected one of: {allowed}")
    return normalized


def voxel_subsampling_filter(resolution: float, method: str) -> dict:
    """Return the PDAL filter stage for methods implemented by PDAL."""
    method = normalize_subsampling_method(method)
    if method == SUBSAMPLING_METHOD_NEAREST_TO_CENTROID:
        return {"type": "filters.voxelcentroidnearestneighbor", "cell": resolution}
    raise ValueError(f"{method} is implemented by SmartTile, not a PDAL filter stage")


def is_copc_file(path: Path) -> bool:
    return path.name.lower().endswith(".copc.laz")
