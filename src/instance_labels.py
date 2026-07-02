#!/usr/bin/env python3
"""SmartTile prediction instance ID output contract.

SmartTile persists prediction instance dimensions as unsigned integer extra
bytes. The semantic contract is intentionally simple:

- `0` means background/no tree.
- Positive values are tree instance IDs.
- Negative values are invalid at input/output boundaries.
- `uint32` is used only when an instance ID exceeds the `uint16` range.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import laspy
import numpy as np


INSTANCE_UINT32_THRESHOLD = np.iinfo(np.uint16).max
INSTANCE_DEFAULT_OUTPUT_DTYPE = np.uint16
INSTANCE_LARGE_OUTPUT_DTYPE = np.uint32
MERGED_OUTPUT_SCALES = np.array([0.01, 0.01, 0.01], dtype=np.float64)


def validate_prediction_instance_labels(
    instances: np.ndarray,
    name: str = "PredInstance",
    source: Optional[Path | str] = None,
) -> None:
    """Fail fast when SmartTile receives invalid negative prediction labels."""
    arr = np.asarray(instances)
    if arr.size == 0 or not np.any(arr < 0):
        return

    negative_count = int(np.count_nonzero(arr < 0))
    min_value = int(np.min(arr))
    source_prefix = f"{source} " if source is not None else ""
    raise ValueError(
        f"{source_prefix}contains {negative_count} negative {name} values "
        f"(minimum {min_value}). SmartTile expects {name}=0 for background/no tree "
        "and positive values for tree instances."
    )


def instance_output_dtype(instances: Optional[np.ndarray] = None) -> np.dtype:
    """Return uint32 only when any instance ID exceeds the uint16 range."""
    if instances is None:
        return np.dtype(INSTANCE_DEFAULT_OUTPUT_DTYPE)

    arr = np.asarray(instances)
    if arr.size == 0:
        return np.dtype(INSTANCE_DEFAULT_OUTPUT_DTYPE)
    validate_prediction_instance_labels(arr, "instance IDs")

    max_instance = int(np.max(arr))
    if max_instance > INSTANCE_UINT32_THRESHOLD:
        return np.dtype(INSTANCE_LARGE_OUTPUT_DTYPE)
    return np.dtype(INSTANCE_DEFAULT_OUTPUT_DTYPE)


def instance_extra_bytes_params(name: str, instances: Optional[np.ndarray] = None) -> laspy.ExtraBytesParams:
    """Return the persisted SmartTile instance dimension schema."""
    return laspy.ExtraBytesParams(name=name, type=instance_output_dtype(instances))


def cast_instances_for_output(instances: np.ndarray, name: str = "instance IDs") -> np.ndarray:
    """Cast non-negative instance IDs to the persisted unsigned output dtype."""
    arr = np.asarray(instances)
    validate_prediction_instance_labels(arr, name)
    return arr.astype(instance_output_dtype(arr), copy=False)


def validate_merged_output_contract(
    merged_laz: Path,
    instance_dimension: str = "PredInstance",
) -> None:
    """Verify persisted merged LAZ follows the SmartTile output contract."""
    with laspy.open(str(merged_laz), laz_backend=laspy.LazBackend.LazrsParallel) as f:
        scales = np.asarray(f.header.scales, dtype=np.float64)
        if not np.allclose(scales, MERGED_OUTPUT_SCALES):
            raise ValueError(
                f"{merged_laz} has XYZ scales {scales.tolist()}, expected "
                f"{MERGED_OUTPUT_SCALES.tolist()} for 1cm merged output"
            )

        extra_dims = {dim.name: dim for dim in f.header.point_format.extra_dimensions}
        if instance_dimension not in extra_dims:
            raise ValueError(f"{merged_laz} is missing required {instance_dimension} extra dimension")
        dtype = np.dtype(extra_dims[instance_dimension].dtype)
        if dtype not in {
            np.dtype(INSTANCE_DEFAULT_OUTPUT_DTYPE),
            np.dtype(INSTANCE_LARGE_OUTPUT_DTYPE),
        }:
            raise ValueError(
                f"{merged_laz} has {instance_dimension} dtype {dtype}, expected uint16 or uint32"
            )

        point_data = f.read()
        instances = np.asarray(getattr(point_data, instance_dimension))
        expected_dtype = instance_output_dtype(instances)
        if dtype != expected_dtype:
            raise ValueError(
                f"{merged_laz} has {instance_dimension} dtype {dtype}, expected "
                f"{expected_dtype} for max instance ID {int(np.max(instances)) if instances.size else 0}"
            )
