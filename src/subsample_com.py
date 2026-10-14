#!/usr/bin/env python3
"""Center-of-mass output header and aligned chunk edges for SmartTile."""

from __future__ import annotations

from fractions import Fraction
import math
from typing import List, Tuple

import laspy

from crs_records import standardize_projection_records


def _is_stale_copc_vlr(vlr) -> bool:
    return getattr(vlr, "user_id", "") == "copc"


def _is_extra_bytes_vlr(vlr) -> bool:
    return getattr(vlr, "user_id", "") == "LASF_Spec" and getattr(vlr, "record_id", None) == 4


def make_center_of_mass_header(source_header: laspy.LasHeader, dimension_reduction: bool) -> laspy.LasHeader:
    """Create an output header for center-of-mass subsampling."""
    from laspy.vlrs.vlrlist import VLRList

    if dimension_reduction:
        header = laspy.LasHeader(point_format=0, version=str(source_header.version))
        for attr in (
            "file_source_id",
            "global_encoding",
            "uuid",
            "system_identifier",
            "generating_software",
            "creation_date",
        ):
            if hasattr(source_header, attr):
                try:
                    setattr(header, attr, getattr(source_header, attr))
                except Exception:
                    pass
        source_vlrs = source_header.vlrs
    else:
        header = source_header.copy()
        source_vlrs = header.vlrs

    header.offsets = source_header.offsets
    header.scales = source_header.scales
    header.vlrs = VLRList([
        vlr for vlr in (source_vlrs or [])
        if not _is_stale_copc_vlr(vlr)
        and (not dimension_reduction or not _is_extra_bytes_vlr(vlr))
    ])

    source_evlrs = getattr(source_header, "evlrs", None)
    if source_evlrs is not None:
        header.evlrs = VLRList([vlr for vlr in source_evlrs if not _is_stale_copc_vlr(vlr)])

    return standardize_projection_records(header)


def aligned_edges(start: float, stop: float, step: float, align: float) -> List[Tuple[float, float]]:
    """Chunk edges on the absolute decimal voxel grid, clipped to LAS bounds."""
    if (not all(math.isfinite(value) for value in (start, stop, step, align)) or
            step <= 0 or align <= 0):
        raise ValueError("Aligned edges need finite bounds and positive step/alignment")
    grid = Fraction(str(align))
    start_exact, stop_exact = Fraction(str(start)), Fraction(str(stop))
    # The caller forms step as a whole number of voxels. Recover that count
    # before floating multiplication can turn e.g. 3 * .07 into .21000000000000002.
    cells = max(1, round(step / align))
    position = (start_exact // grid) * grid
    stride = cells * grid
    edges = []
    while position < stop_exact:
        end = min(position + stride, stop_exact)
        if float(end) <= float(position):
            raise ValueError("Aligned edge step cannot advance at this coordinate magnitude")
        if end > start_exact:
            edges.append((float(max(position, start_exact)), float(end)))
        position = end
    return edges
