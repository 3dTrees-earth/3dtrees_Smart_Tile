#!/usr/bin/env python3
"""Safe LAS/COPC writer encoding helpers for SmartTile subsampling."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Sequence, Tuple

import laspy
import numpy as np


INT32_MIN = np.iinfo(np.int32).min
INT32_MAX = np.iinfo(np.int32).max


def _header_bounds(header) -> Tuple[float, float, float, float, float, float]:
    return (
        float(header.x_min),
        float(header.x_max),
        float(header.y_min),
        float(header.y_max),
        float(header.z_min),
        float(header.z_max),
    )


def xy_bounds_from_pdal_bounds(bounds_str: str) -> Tuple[float, float, float, float]:
    """Parse a PDAL XY bounds string like ``([minx,maxx],[miny,maxy])``."""
    values = [float(value) for value in re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", bounds_str)]
    if len(values) < 4:
        raise ValueError(f"Could not parse XY bounds string: {bounds_str!r}")
    min_x, max_x, min_y, max_y = values[:4]
    return min_x, max_x, min_y, max_y


def _xyz_bounds_from_xy(
    xy_bounds: Optional[Sequence[float]],
    header,
) -> Tuple[float, float, float, float, float, float]:
    if xy_bounds is None:
        return _header_bounds(header)
    if len(xy_bounds) != 4:
        raise ValueError("xy_bounds must contain minx, maxx, miny, maxy")
    min_x, max_x, min_y, max_y = (float(value) for value in xy_bounds)
    return min_x, max_x, min_y, max_y, float(header.z_min), float(header.z_max)


def _fits_int32(
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    scales: np.ndarray,
    offsets: np.ndarray,
) -> bool:
    if np.any(scales <= 0) or not np.all(np.isfinite(scales)):
        return False
    scaled_min = np.round((bounds_min - offsets) / scales)
    scaled_max = np.round((bounds_max - offsets) / scales)
    scaled = np.concatenate([scaled_min, scaled_max])
    return bool(np.all(scaled >= INT32_MIN) and np.all(scaled <= INT32_MAX))


def _encoding_options(scales: np.ndarray, offsets: np.ndarray) -> dict:
    return {
        "scale_x": float(scales[0]),
        "scale_y": float(scales[1]),
        "scale_z": float(scales[2]),
        "offset_x": float(offsets[0]),
        "offset_y": float(offsets[1]),
        "offset_z": float(offsets[2]),
    }


def safe_las_writer_encoding_options(
    source_file: Path,
    xy_bounds: Optional[Sequence[float]] = None,
) -> dict:
    """
    Return explicit LAS/COPC writer scale/offset options safe for int32 storage.

    Source scales and offsets are preserved when they can encode the requested
    bounds. If not, the source scales are kept and offsets are moved to the
    requested bounds minimum.
    """
    with laspy.open(str(source_file), laz_backend=laspy.LazBackend.LazrsParallel) as reader:
        header = reader.header
        scales = np.asarray(header.scales, dtype=np.float64)
        source_offsets = np.asarray(header.offsets, dtype=np.float64)
        min_x, max_x, min_y, max_y, min_z, max_z = _xyz_bounds_from_xy(xy_bounds, header)

    bounds_min = np.array([min_x, min_y, min_z], dtype=np.float64)
    bounds_max = np.array([max_x, max_y, max_z], dtype=np.float64)

    if _fits_int32(bounds_min, bounds_max, scales, source_offsets):
        return _encoding_options(scales, source_offsets)

    fallback_offsets = bounds_min.copy()
    if _fits_int32(bounds_min, bounds_max, scales, fallback_offsets):
        return _encoding_options(scales, fallback_offsets)

    raise ValueError(
        "Could not choose safe LAS writer encoding for "
        f"{source_file}: scales={scales.tolist()}, "
        f"bounds_min={bounds_min.tolist()}, bounds_max={bounds_max.tolist()}"
    )


def safe_las_writer_encoding_options_for_pdal_bounds(source_file: Path, bounds_str: str) -> dict:
    """Return safe writer options for one PDAL XY bounds string."""
    return safe_las_writer_encoding_options(
        source_file,
        xy_bounds=xy_bounds_from_pdal_bounds(bounds_str),
    )


def encoding_summary(options: dict) -> str:
    """Return a compact diagnostic string for selected writer encoding."""
    return (
        "scale="
        f"({options.get('scale_x')}, {options.get('scale_y')}, {options.get('scale_z')}), "
        "offset="
        f"({options.get('offset_x')}, {options.get('offset_y')}, {options.get('offset_z')})"
    )
