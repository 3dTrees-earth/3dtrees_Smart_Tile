#!/usr/bin/env python3
"""COPC CRS and GeoTIFF metadata preservation helpers.

These helpers are shared by tiling, subsampling conversion, and prod-merged
creation. They keep CRS/GeoTIFF handling out of task orchestration modules and
make the metadata contract explicit in one place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


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


def _without_bound(crs):
    """``crs`` with every BoundCRS (TOWGS84, geoid grid) replaced by its source CRS, also inside a compound CRS.

    A WKT record may attach a datum shift or a geoid grid that GeoTIFF keys of the same CRS cannot carry;
    both describe the same coordinates.
    """
    from pyproj.crs import CompoundCRS
    if getattr(crs, "is_bound", False) and crs.source_crs is not None:
        crs = crs.source_crs
    if crs.is_compound:
        parts = [sub.source_crs if getattr(sub, "is_bound", False) and sub.source_crs is not None else sub
                 for sub in crs.sub_crs_list]
        return CompoundCRS(crs.name, parts)
    return crs


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
        if _without_bound(source_crs).equals(_without_bound(output_crs), ignore_axis_order=True):
            return True
    except Exception:
        pass

    try:
        return source_crs.to_wkt() == output_crs.to_wkt()
    except Exception:
        return str(source_crs) == str(output_crs)


def wkt_only_srs(header) -> Optional[str]:
    """The WKT record's text when untwine and PDAL would not read it, None otherwise.

    They take the CRS of a file below LAS 1.4, or without the global-encoding WKT bit,
    from the GeoTIFF keys. Without GeoTIFF keys such a file's WKT record would be lost,
    so it is passed as --a_srs. A file that has GeoTIFF keys keeps them as the writers'
    source, so the CRS check still sees a WKT record that disagrees with them.
    """
    records = projection_records(header, {2112, 34735})
    wkt = records.get((PROJECTION_VLR_USER_ID, 2112), b"").rstrip(b"\0").strip()
    if not wkt or (PROJECTION_VLR_USER_ID, 34735) in records:
        return None
    if str(header.version) >= "1.4" and header.global_encoding.wkt:
        return None
    try:
        text = wkt.decode("utf-8")
        from pyproj import CRS

        CRS.from_wkt(text)
    except Exception:
        return None
    return text


def srs_assignment_from_file(path: Path) -> Optional[str]:
    """Read an input header and return an Untwine --a_srs value when possible.

    The EPSG authority when the CRS has one, else the WKT record of a file whose
    CRS only the WKT record holds and the writers would not read (``wkt_only_srs``).
    """
    import laspy

    try:
        with laspy.open(str(path), laz_backend=laspy_laz_backend()) as src:
            return crs_authority_string(src.header) or wkt_only_srs(src.header)
    except Exception:
        return None
