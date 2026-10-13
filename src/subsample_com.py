#!/usr/bin/env python3
"""Center-of-mass output header and aligned chunk edges for SmartTile."""

from __future__ import annotations

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
    first = math.floor(start / align) * align
    # The rounded grid boundary can fall one ULP below the actual LAS bound.
    # Reach the true bound so the final window cannot repeat forever.
    final = max(stop, math.ceil(stop / align) * align)
    edges = []
    cur = first
    while cur < stop:
        nxt = min(cur + step, final)
        if nxt <= cur:
            raise ValueError("Aligned edge step cannot advance at this coordinate magnitude")
        if nxt > start and cur < stop:
            edges.append((max(cur, start), min(nxt, stop)))
        cur = nxt
    return edges
