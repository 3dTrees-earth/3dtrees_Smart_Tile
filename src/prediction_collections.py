"""Prediction collection discovery and lossless prediction-dimension handling."""
from __future__ import annotations

from pathlib import Path
from typing import List

import laspy
import numpy as np

from point_cloud_metadata import point_cloud_files


def prediction_collection_files(path: Path) -> List[Path]:
    """Return LAZ/LAS files for a prediction collection folder or single file."""
    path = Path(path)
    if path.is_file():
        if path.suffix.lower() == ".las" or path.name.lower().endswith(".laz"):
            return [path]
        return []
    return point_cloud_files(path)


def original_prediction_radius(collection: Path) -> float:
    """Bound the coordinate rounding introduced by encoding retained points.

    Original-resolution predictions contain source points, not voxel means.
    Re-encoding those coordinates in LAS can move each axis by half its scale
    step. Use the coarsest XYZ steps in this collection and their half diagonal;
    the spatial index supplies its usual numerical roundoff guard separately.
    This is a file-precision allowance, never an inferred subsampling radius.
    """
    files = prediction_collection_files(collection)
    if not files:
        raise ValueError(f"No prediction files in {collection}")
    scales = np.zeros(3, dtype=np.float64)
    coordinate_magnitude = 1.0
    for path in files:
        with laspy.open(path, read_evlrs=False) as reader:
            incoming = np.asarray(reader.header.scales, dtype=np.float64)
            if not np.all(np.isfinite(incoming)) or np.any(incoming <= 0):
                raise ValueError(f"{path}: original-resolution predictions need positive finite XYZ scales")
            scales = np.maximum(scales, incoming)
            coordinate_magnitude = max(coordinate_magnitude,
                                       float(np.max(np.abs(reader.header.mins))),
                                       float(np.max(np.abs(reader.header.maxs))))
    # LAS writers may round global float64 XYZ before integer encoding.
    # Account for that conversion even though the query index uses local XYZ.
    return float(np.linalg.norm(scales / 2) + 8 * np.spacing(coordinate_magnitude))


def promote_collection_extra_dim(
    current: laspy.ExtraBytesParams,
    incoming,
) -> laspy.ExtraBytesParams:
    """Return a lossless collection-wide schema for one prediction dimension."""
    current_dtype = np.dtype(current.type)
    incoming_dtype = np.dtype(incoming.dtype)
    promoted_dtype = np.promote_types(current_dtype, incoming_dtype)
    if promoted_dtype == current_dtype:
        return current
    return laspy.ExtraBytesParams(
        name=current.name,
        type=promoted_dtype,
        description=getattr(current, "description", "") or "",
        offsets=getattr(current, "offsets", None),
        scales=getattr(current, "scales", None),
        no_data=getattr(current, "no_data", None),
    )


def assign_prediction_values(out_record, name: str, values: np.ndarray, *, mask=None, raw=False) -> None:
    """Assign one prediction dimension, rejecting lossy integer downcasts."""
    values = np.asarray(values)
    target = out_record.array[name] if raw else out_record[name]
    target_dtype = np.asarray(target).dtype
    if np.issubdtype(values.dtype, np.integer) and np.issubdtype(
        target_dtype, np.integer
    ):
        limits = np.iinfo(target_dtype)
        minimum = int(values.min()) if values.size else 0
        maximum = int(values.max()) if values.size else 0
        if minimum < limits.min or maximum > limits.max:
            raise OverflowError(
                f"Prediction dimension {name} has values [{minimum}, {maximum}] "
                f"that do not fit output dtype {target_dtype}"
            )
    if raw:
        if mask is None:
            target[:] = values
        else:
            target[mask] = values
    elif mask is None:
        out_record[name] = values
    else:
        out_record[name][mask] = values
