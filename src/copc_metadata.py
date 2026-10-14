#!/usr/bin/env python3
"""COPC CRS and GeoTIFF metadata preservation helpers.

These helpers are shared by tiling, subsampling conversion, and prod-merged
creation. They keep CRS/GeoTIFF handling out of task orchestration modules and
make the metadata contract explicit in one place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECTION_VLR_USER_ID = "LASF_Projection"
PROJECTION_VLR_RECORD_IDS = {2111, 2112, 2113, 34735, 34736, 34737}


def laspy_laz_backend():
    """Return the best available laspy LAZ backend."""
    try:
        import laspy

        if hasattr(laspy.LazBackend, "LazrsParallel"):
            return laspy.LazBackend.LazrsParallel
        if hasattr(laspy.LazBackend, "Lazrs"):
            return laspy.LazBackend.Lazrs
    except Exception:
        pass
    return None


def projection_vlr_keys(header) -> set[tuple[str, int]]:
    """Return CRS/projection VLR keys that should survive format conversion."""
    keys = set()
    for vlr in getattr(header, "vlrs", []):
        user_id = getattr(vlr, "user_id", "")
        record_id = getattr(vlr, "record_id", None)
        if user_id == PROJECTION_VLR_USER_ID or record_id in PROJECTION_VLR_RECORD_IDS:
            keys.add((user_id, record_id))
    return keys


def vlr_record_bytes(vlr) -> bytes:
    """Return raw record bytes for regular laspy VLR objects."""
    if hasattr(vlr, "record_data_bytes"):
        try:
            return bytes(vlr.record_data_bytes())
        except Exception:
            pass
    record_data = getattr(vlr, "record_data", None)
    if record_data is None:
        return b""
    try:
        return bytes(record_data)
    except Exception:
        return str(record_data).encode("utf-8", errors="replace")


def projection_records(header, record_ids: set[int]) -> Dict[tuple[str, int], bytes]:
    """Return projection record payloads from header VLRs/EVLRs."""
    records = {}
    collections = [getattr(header, "vlrs", [])]
    evlrs = getattr(header, "evlrs", None)
    if evlrs:
        collections.append(evlrs)

    for collection in collections:
        for vlr in collection:
            user_id = getattr(vlr, "user_id", "")
            record_id = getattr(vlr, "record_id", None)
            if user_id == PROJECTION_VLR_USER_ID and record_id in record_ids:
                records[(user_id, record_id)] = vlr_record_bytes(vlr)
    return records


def projection_vlr_fingerprints(header) -> Dict[tuple[str, int], bytes]:
    """Return CRS/projection VLR record bytes keyed by LAS VLR identity."""
    return projection_records(header, PROJECTION_VLR_RECORD_IDS)


def parse_crs(header) -> Optional[Any]:
    try:
        return header.parse_crs()
    except Exception:
        return None


def crs_text(header) -> Optional[str]:
    """Best-effort parseable CRS text; returns None when CRS metadata is unavailable."""
    crs = parse_crs(header)
    if crs is None:
        return None
    try:
        return crs.to_wkt()
    except Exception:
        return str(crs)


def crs_authority_from_crs(crs) -> Optional[str]:
    if crs is None:
        return None

    try:
        authority = crs.to_authority()
    except Exception:
        authority = None
    if authority and authority[0] and authority[1]:
        return f"{authority[0].upper()}:{authority[1]}"

    try:
        epsg = crs.to_epsg()
    except Exception:
        epsg = None
    if epsg:
        return f"EPSG:{epsg}"
    return None


def crs_authority_string(header) -> Optional[str]:
    """Return a compact CRS authority string such as EPSG:32632 when available."""
    return crs_authority_from_crs(parse_crs(header))


def crs_equivalent(source_crs, output_crs) -> bool:
    """Return True when two parsed CRS objects describe the same CRS."""
    if source_crs is None or output_crs is None:
        return False

    source_authority = crs_authority_from_crs(source_crs)
    output_authority = crs_authority_from_crs(output_crs)
    if source_authority and source_authority == output_authority:
        return True

    try:
        if source_crs.equals(output_crs, ignore_axis_order=True):
            return True
    except Exception:
        pass

    try:
        return source_crs.to_wkt() == output_crs.to_wkt()
    except Exception:
        return str(source_crs) == str(output_crs)


def srs_assignment_from_file(path: Path) -> Optional[str]:
    """Read an input header and return an Untwine --a_srs value when possible."""
    import laspy

    try:
        with laspy.open(str(path), laz_backend=laspy_laz_backend()) as src:
            return crs_authority_string(src.header)
    except Exception:
        return None


def first_srs_assignment(paths: List[Path]) -> Optional[str]:
    crs_source = first_crs_source(paths)
    if crs_source is None:
        return None
    return srs_assignment_from_file(crs_source)


def crs_metadata_present(header) -> bool:
    return bool(projection_vlr_keys(header) or crs_text(header))


def first_crs_source(paths: List[Path]) -> Optional[Path]:
    """Return the first source file with CRS metadata, else the first source file."""
    if not paths:
        return None
    import laspy

    fallback = paths[0]
    for path in paths:
        try:
            with laspy.open(str(path), laz_backend=laspy_laz_backend()) as src:
                if crs_metadata_present(src.header):
                    return path
        except Exception:
            continue
    return fallback
