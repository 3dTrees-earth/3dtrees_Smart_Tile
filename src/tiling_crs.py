"""Reject input CRSs that SmartTile cannot tile in metres (3DT-1898, criterion 1).

Tile length, buffer, both subsampling resolutions and every remap radius are
metres in the coordinate units of the input. Geographic CRSs (degrees),
geocentric CRSs (ECEF X/Y/Z), non-metre linear units and uploads mixing
different CRSs therefore produce nonsense extents ("would create millions of
tiles", 3DT-1709/1710/1777) or silently wrong voxel sizes. The tile task checks
every input header before indexing and fails with the file and CRS named.
Files without readable CRS metadata are local coordinates and are accepted.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import laspy

from copc_metadata import crs_authority_from_crs, crs_equivalent, parse_crs

REPROJECT_HINT = ("SmartTile tile size, buffer and subsampling resolutions are metres; "
                  "reproject the point clouds to a projected CRS in metres (e.g. the local UTM zone) "
                  "and upload them again.")
SHOWN_FILES = 5


def _label(crs) -> str:
    authority = crs_authority_from_crs(crs)
    return f"{authority} ({crs.name})" if authority else repr(crs.name)


def _unbound(crs):
    return crs.source_crs if getattr(crs, "is_bound", False) and crs.source_crs is not None else crs


def horizontal_crs(crs):
    """The CRS that defines X/Y: source of a bound CRS, first part of a compound CRS."""
    crs = _unbound(crs)
    if crs.is_compound and crs.sub_crs_list:
        crs = _unbound(crs.sub_crs_list[0])
    return crs


def unit_problem(crs) -> Optional[str]:
    """Why ``crs`` cannot be tiled in metres, or None when it can."""
    horizontal = horizontal_crs(crs)
    if horizontal.is_geocentric:
        return "is geocentric (Earth-centred X/Y/Z), not a projected map CRS"
    if horizontal.is_geographic:
        return "is geographic: X/Y are degrees of longitude/latitude, not metres"
    for axis in horizontal.axis_info[:2]:
        if axis.unit_conversion_factor and abs(axis.unit_conversion_factor - 1.0) > 1e-12:
            return f"uses linear unit '{axis.unit_name}', not metres"
    return None


def require_metric_tiling_crs(files: Iterable[Path]):
    """Return the common CRS of ``files`` (None if none declares one); raise ValueError otherwise."""
    declared: List[Tuple[Path, object]] = []
    for path in files:
        with laspy.open(path, read_evlrs=False) as reader:
            crs = parse_crs(reader.header)
        if crs is not None:
            declared.append((Path(path), crs))
    problems = [(path, crs, why) for path, crs in declared if (why := unit_problem(crs))]
    if problems:
        shown = "; ".join(f"{path.name}: {_label(crs)} {why}" for path, crs, why in problems[:SHOWN_FILES])
        more = f" (and {len(problems) - SHOWN_FILES} more files)" if len(problems) > SHOWN_FILES else ""
        raise ValueError(f"Unsupported input CRS for tiling: {shown}{more}. {REPROJECT_HINT}")
    distinct: List[Tuple[Path, object]] = []
    for path, crs in declared:
        if not any(crs_equivalent(crs, seen) for _, seen in distinct):
            distinct.append((path, crs))
    if len(distinct) > 1:
        shown = "; ".join(f"{_label(crs)}, e.g. {path.name}" for path, crs in distinct[:SHOWN_FILES])
        raise ValueError(f"Input files use {len(distinct)} different CRSs: {shown}. "
                         "All point clouds of one dataset must share one projected CRS in metres; "
                         "reproject them to a common CRS and upload them again.")
    return distinct[0][1] if distinct else None
