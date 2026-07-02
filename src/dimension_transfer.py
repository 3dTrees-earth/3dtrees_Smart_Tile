#!/usr/bin/env python3
"""Dimension collision and transfer policy for SmartTile point-cloud products."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Set, Tuple

import numpy as np


CORE_COORDINATE_DIMS = {"X", "Y", "Z"}


@dataclass(frozen=True)
class ResolvedDimensionTransferPlan:
    """Fully resolved dimension transfer plan with output dtypes."""

    add_new: Dict[str, np.dtype]
    overwrite: Dict[str, np.dtype]
    renamed: Dict[str, str]
    output_dtypes: Dict[str, np.dtype]
    output_to_source: Dict[str, str]

    @property
    def has_transfers(self) -> bool:
        return bool(self.output_dtypes)


def next_available_suffix(base: str, used: Set[str]) -> str:
    """Return base_1, base_2, ... first not in used."""
    for i in range(1, 10000):
        candidate = f"{base}_{i}"
        if candidate not in used:
            return candidate
    return f"{base}_9999"


def suffixes_for_collision(base: str, used: Set[str]) -> Tuple[str, str]:
    """Return collision-safe source/target suffixes for one base dimension name."""
    first = f"{base}_1"
    second = f"{base}_2"
    out_first = first if first not in used else next_available_suffix(base, used)
    used.add(out_first)
    out_second = second if second not in used else next_available_suffix(base, used)
    used.add(out_second)
    return out_first, out_second


def _target_dimension_is_empty_or_constant(values) -> bool:
    arr = np.asarray(values)
    if arr.size == 0:
        return True
    return np.min(arr) == np.max(arr) or np.count_nonzero(arr) == 0


def plan_dimension_transfer(
    source_dims: Dict[str, np.dtype],
    target_dim_names: Set[str],
    get_target_array: Callable[[str], object],
    skip: Optional[Set[str]] = None,
) -> ResolvedDimensionTransferPlan:
    """Plan how source dimensions should be transferred into a target cloud.

    Policy:
    - skip core coordinate dimensions by default
    - add source-only dimensions under their original name
    - overwrite target dimensions only when the target array is empty/constant
    - preserve non-empty target/source collisions by adding the source as name_1
    """
    skipped = skip or CORE_COORDINATE_DIMS
    source_names = set(source_dims.keys()) - skipped
    target_names = set(target_dim_names) - skipped

    add_new = {name: source_dims[name] for name in source_names - target_names}
    overwrite: Dict[str, np.dtype] = {}
    for name in target_names & source_names:
        arr = get_target_array(name)
        if arr is None:
            continue
        if _target_dimension_is_empty_or_constant(arr):
            overwrite[name] = source_dims[name]

    used_names = set(target_dim_names) | set(add_new.keys()) | set(overwrite.keys())
    collision = (source_names & target_names) - set(overwrite.keys())
    renamed: Dict[str, str] = {}
    for name in sorted(collision):
        candidate = f"{name}_1"
        output_name = candidate if candidate not in used_names else next_available_suffix(name, used_names)
        renamed[name] = output_name
        used_names.add(output_name)

    output_dtypes = {**add_new, **overwrite}
    output_dtypes.update({out_name: source_dims[source_name] for source_name, out_name in renamed.items()})
    output_to_source = {name: name for name in add_new}
    output_to_source.update({name: name for name in overwrite})
    output_to_source.update({out_name: source_name for source_name, out_name in renamed.items()})

    return ResolvedDimensionTransferPlan(
        add_new=add_new,
        overwrite=overwrite,
        renamed=renamed,
        output_dtypes=output_dtypes,
        output_to_source=output_to_source,
    )
