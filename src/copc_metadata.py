#!/usr/bin/env python3
"""COPC CRS and GeoTIFF metadata preservation helpers.

These helpers are shared by tiling, subsampling conversion, and prod-merged
creation. They keep CRS/GeoTIFF handling out of task orchestration modules and
make the metadata contract explicit in one place.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROJECTION_VLR_USER_ID = "LASF_Projection"
PROJECTION_VLR_RECORD_IDS = {2111, 2112, 2113, 34735, 34736, 34737}
GEOTIFF_PROJECTION_RECORD_IDS = {34735, 34736, 34737}
LAS14_START_OF_FIRST_EVLR_OFFSET = 235
LAS14_NUMBER_OF_EVLRS_OFFSET = 243


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


def geotiff_projection_records(header) -> Dict[tuple[str, int], bytes]:
    """Return GeoTIFF CRS projection records, including GeoKeyDirectoryVlr."""
    return projection_records(header, GEOTIFF_PROJECTION_RECORD_IDS)


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


def evlr_record_bytes(user_id: str, record_id: int, description: str, data: bytes) -> bytes:
    user = user_id.encode("ascii", errors="replace")[:16].ljust(16, b"\0")
    desc = description.encode("ascii", errors="replace")[:32].ljust(32, b"\0")
    return struct.pack("<H16sHQ32s", 0, user, int(record_id), len(data), desc) + data


def source_geotiff_projection_vlrs(source_file: Path) -> List[Tuple[str, int, str, bytes]]:
    """Return source projection VLR payloads for preservation.

    The function name is kept for compatibility with existing callers. It now
    preserves all LASF_Projection CRS records, including WKT and GeoTIFF keys,
    because COPC writers can normalize WKT while still carrying an equivalent
    CRS. Appending the source records as EVLRs keeps the original metadata
    available without rewriting COPC hierarchy bytes.
    """
    import laspy

    records = []
    with laspy.open(str(source_file), laz_backend=laspy_laz_backend()) as src:
        collections = [getattr(src.header, "vlrs", [])]
        evlrs = getattr(src.header, "evlrs", None)
        if evlrs:
            collections.append(evlrs)
        for collection in collections:
            for vlr in collection:
                user_id = getattr(vlr, "user_id", "")
                record_id = getattr(vlr, "record_id", None)
                if user_id == PROJECTION_VLR_USER_ID and record_id in PROJECTION_VLR_RECORD_IDS:
                    records.append(
                        (
                            user_id,
                            int(record_id),
                            getattr(vlr, "description", "") or "",
                            vlr_record_bytes(vlr),
                        )
                    )
    return records


def append_source_geotiff_projection_evlrs(source_file: Path, copc_file: Path) -> Tuple[bool, str]:
    """Append original projection VLRs as EVLRs without moving COPC chunks."""
    import laspy

    try:
        source_records = source_geotiff_projection_vlrs(source_file)
        if not source_records:
            return (True, "source has no projection VLRs")

        with laspy.open(str(copc_file), laz_backend=laspy_laz_backend()) as out:
            output_header = out.header
            output_records = projection_vlr_fingerprints(output_header)
            existing_count = int(getattr(output_header, "number_of_evlrs", 0) or 0)
            existing_start = int(getattr(output_header, "start_of_first_evlr", 0) or 0)
            if str(output_header.version) != "1.4":
                return (False, "GeoTIFF EVLR preservation requires LAS 1.4 output")

        missing = [
            (user_id, record_id, description, data)
            for user_id, record_id, description, data in source_records
            if output_records.get((user_id, record_id)) != data
        ]
        if not missing:
            return (True, "projection VLRs already preserved")

        with open(copc_file, "r+b") as f:
            f.seek(0, os.SEEK_END)
            append_start = f.tell()
            for user_id, record_id, description, data in missing:
                f.write(evlr_record_bytes(user_id, record_id, description, data))
            f.seek(LAS14_START_OF_FIRST_EVLR_OFFSET)
            f.write(struct.pack("<Q", existing_start or append_start))
            f.seek(LAS14_NUMBER_OF_EVLRS_OFFSET)
            f.write(struct.pack("<I", existing_count + len(missing)))

        return (True, f"appended {len(missing)} projection EVLR(s)")
    except Exception as e:
        return (False, f"could not preserve projection VLRs: {e}")


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


def copc_preserves_source_crs(source_file: Path, copc_file: Path) -> Tuple[bool, str]:
    """Validate that a COPC output still carries the source CRS/projection metadata."""
    import laspy

    try:
        with laspy.open(str(source_file), laz_backend=laspy_laz_backend()) as src:
            source_header = src.header
            source_crs = parse_crs(source_header)
            source_projection = projection_vlr_fingerprints(source_header)
            source_geotiff_projection = geotiff_projection_records(source_header)

        if not source_crs and not source_projection:
            return (True, "source has no CRS metadata")

        with laspy.open(str(copc_file), laz_backend=laspy_laz_backend()) as out:
            output_header = out.header
            output_crs = parse_crs(output_header)
            output_projection = projection_vlr_fingerprints(output_header)
            output_geotiff_projection = geotiff_projection_records(output_header)
    except Exception as e:
        return (False, f"could not validate CRS metadata: {e}")

    if source_geotiff_projection and not all(
        output_geotiff_projection.get(key) == value
        for key, value in source_geotiff_projection.items()
    ):
        return (
            False,
            "source GeoTIFF projection VLRs missing or changed in COPC output",
        )
    changed_projection_keys = [
        key
        for key, value in source_projection.items()
        if key in output_projection and output_projection[key] != value
    ]
    if changed_projection_keys:
        return (
            False,
            "source CRS/projection metadata missing or changed in COPC output",
        )
    if crs_equivalent(source_crs, output_crs):
        return (True, "CRS metadata preserved")
    if source_projection and all(
        output_projection.get(key) == value for key, value in source_projection.items()
    ):
        return (True, "projection VLRs preserved")
    if source_crs or source_projection:
        return (
            False,
            "source CRS/projection metadata missing or changed in COPC output",
        )
    return (True, "source has no CRS metadata")
