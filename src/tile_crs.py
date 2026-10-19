"""Reject input CRSs that SmartTile cannot tile in metres (3DT-1898, criterion 1).

Tile length, buffer, both subsampling resolutions and every remap radius are
metres in the coordinate units of the input. Geographic CRSs (degrees),
geocentric CRSs (ECEF X/Y/Z), non-metre linear units and uploads mixing
different CRSs therefore produce nonsense extents ("would create millions of
tiles", 3DT-1709/1710/1777) or silently wrong voxel sizes. The tile task checks
every input header before indexing and fails with the file and CRS named.
Files without readable CRS metadata are local coordinates and are accepted
when no input declares a CRS; a mix of files with and without a CRS is
rejected, because their coordinates need not share one frame.
3Dtrees standardization moves data without a usable CRS to 0,0,0 and labels it
EPSG:4978; a geocentric CRS whose points lie far inside the Earth is such a
stand-in and is tiled as local metres.
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
# Real Earth-centred coordinates lie about 6,350-6,390 km from the centre.
MAX_STAND_IN_RADIUS = 6_000_000.0


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


def is_local_stand_in(crs, header) -> bool:
    """True for a geocentric CRS whose points cannot be Earth-centred (standardization's local EPSG:4978)."""
    if not horizontal_crs(crs).is_geocentric:
        return False
    corners = [(x, y, z) for x in (header.mins[0], header.maxs[0])
               for y in (header.mins[1], header.maxs[1]) for z in (header.mins[2], header.maxs[2])]
    return max((x * x + y * y + z * z) ** 0.5 for x, y, z in corners) < MAX_STAND_IN_RADIUS


def unit_problem(crs) -> Optional[str]:
    """Why ``crs`` cannot be tiled in metres, or None when it can."""
    horizontal = horizontal_crs(crs)
    if horizontal.is_geocentric:
        return "is geocentric (Earth-centred X/Y/Z), not a projected map CRS"
    if horizontal.is_geographic:
        return "is geographic: X/Y are degrees of longitude/latitude, not metres"
    # Z voxels are metres too, so a vertical part in feet is as wrong as X/Y in feet.
    crs = _unbound(crs)
    vertical = ([axis for sub in crs.sub_crs_list[1:] for axis in _unbound(sub).axis_info]
                if crs.is_compound else horizontal.axis_info[2:])
    for axis in [*horizontal.axis_info[:2], *vertical]:
        if axis.unit_conversion_factor and abs(axis.unit_conversion_factor - 1.0) > 1e-12:
            return f"uses linear unit '{axis.unit_name}', not metres"
    return None


def _has_projection_records(header) -> bool:
    return any(getattr(vlr, "user_id", "") == "LASF_Projection"
               for vlr in [*getattr(header, "vlrs", []), *(getattr(header, "evlrs", None) or [])])


def require_metric_tiling_crs(files: Iterable[Path]):
    """Return the common CRS of ``files`` (None if none declares one); raise ValueError otherwise."""
    declared: List[Tuple[Path, object]] = []
    undeclared: List[Path] = []
    unreadable: List[Path] = []
    stand_ins: List[Path] = []
    for path in files:
        # LAS 1.4 may keep the WKT in an EVLR.
        with laspy.open(path) as reader:
            crs = parse_crs(reader.header)
            if crs is None and _has_projection_records(reader.header):
                unreadable.append(Path(path))
            if crs is not None and is_local_stand_in(crs, reader.header):
                stand_ins.append(Path(path))
        if crs is not None:
            declared.append((Path(path), crs))
        else:
            undeclared.append(Path(path))
    if unreadable:
        shown = ", ".join(path.name for path in unreadable[:SHOWN_FILES])
        raise ValueError(f"Unreadable CRS records in {shown}: their units cannot be checked. {REPROJECT_HINT}")
    if declared and undeclared:
        shown = ", ".join(path.name for path in undeclared[:SHOWN_FILES])
        more = f" (and {len(undeclared) - SHOWN_FILES} more files)" if len(undeclared) > SHOWN_FILES else ""
        raise ValueError(f"{len(undeclared)} input file(s) have no CRS ({shown}{more}) while "
                         f"{declared[0][0].name} declares {_label(declared[0][1])}. All point clouds of one "
                         "dataset must share one projected CRS in metres; assign or reproject and upload again.")
    if stand_ins:
        shown = ", ".join(path.name for path in stand_ins[:SHOWN_FILES])
        print(f"Warning: {shown}: geocentric CRS with points far inside the Earth; "
              "tiling as local metres (3Dtrees standardization stand-in).")
    problems = [(path, crs, why) for path, crs in declared
                if path not in stand_ins and (why := unit_problem(crs))]
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
