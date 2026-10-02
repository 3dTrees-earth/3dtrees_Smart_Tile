"""One standardized CRS record per output file.

Point-cloud writers (PDAL/GDAL, untwine) re-serialize CRS text. Keeping the
writer's text next to the source's original text left two different records
for one CRS: readers disagreed (PDAL reads the first, laspy the last) and byte
validation failed for valid uploads (3DT-2200 class). SmartTile therefore
writes exactly one WKT record with the *standardized* text — the EPSG
definition as WKT1_GDAL when the CRS is identified, as PDAL does — and
validates outputs semantically.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from copc_metadata import (
    PROJECTION_VLR_USER_ID,
    crs_equivalent,
    laspy_laz_backend,
    parse_crs,
    projection_vlr_fingerprints,
    vlr_record_bytes,
)

WKT_RECORD_ID = 2112
STANDARD_EPSG_CONFIDENCE = 70
PLY_COMMENT_MAX_BYTES = 8192


def standardized_wkt_bytes(wkt: bytes | str) -> bytes:
    """Return the one CRS text SmartTile writes, matching the PDAL/GDAL writer.

    A CRS identified as an EPSG code (confidence >= 70) is written as that code's
    WKT1_GDAL definition; an unidentified CRS (e.g. a compound CRS with a local
    vertical datum) keeps its original text. Always NUL-terminated.
    """
    from pyproj import CRS

    text = wkt.decode("utf-8") if isinstance(wkt, (bytes, bytearray)) else str(wkt)
    text = text.rstrip("\0").strip()
    crs = CRS.from_wkt(text)
    epsg = crs.to_epsg(min_confidence=STANDARD_EPSG_CONFIDENCE)
    if epsg:
        text = CRS.from_epsg(epsg).to_wkt("WKT1_GDAL")
    return text.encode("utf-8") + b"\0"


def standardized_crs_record(header) -> Optional[bytes]:
    """Standardized WKT for a header's effective CRS, or None without a readable CRS.

    Uses the effective (last) non-empty WKT record, else the CRS laspy derives
    (e.g. from GeoTIFF keys). Unreadable CRS metadata returns None so callers
    leave it untouched and validation reports it.
    """
    for _, record_id, data in reversed(projection_record_list(header)):
        if record_id == WKT_RECORD_ID and data.rstrip(b"\0").strip():
            try:
                return standardized_wkt_bytes(data)
            except Exception:
                break
    crs = parse_crs(header)
    if crs is None:
        return None
    try:
        return standardized_wkt_bytes(crs.to_wkt("WKT1_GDAL"))
    except Exception:
        return None


def projection_record_list(header) -> List[Tuple[str, int, bytes]]:
    """Every LASF_Projection record as (vlr|evlr, record_id, payload)."""
    rows = []
    for kind, collection in (("vlr", getattr(header, "vlrs", []) or []),
                             ("evlr", getattr(header, "evlrs", None) or [])):
        for vlr in collection:
            if getattr(vlr, "user_id", "") == PROJECTION_VLR_USER_ID:
                rows.append((kind, int(getattr(vlr, "record_id", -1)), vlr_record_bytes(vlr)))
    return rows


def standardize_projection_records(header):
    """Replace every CRS record of a LAS 1.4 header by one standardized WKT VLR.

    Keeping the writer's and the source's serializations side by side made
    readers disagree (PDAL reads the first, laspy the last) and broke byte
    validation (3DT-2200 class). Headers without CRS, or older than LAS 1.4
    (GeoTIFF keys only), are left unchanged.
    """
    from laspy.vlrs.known import WktCoordinateSystemVlr

    if str(header.version) != "1.4":
        return header
    data = standardized_crs_record(header)
    if data is None:
        return header
    # The typed VLR keeps parse_crs() working before the header is written.
    header.vlrs[:] = [v for v in header.vlrs if getattr(v, "user_id", "") != PROJECTION_VLR_USER_ID] + [
        WktCoordinateSystemVlr(data.rstrip(b"\0").decode("utf-8"))]
    evlrs = getattr(header, "evlrs", None)
    if evlrs:
        evlrs[:] = [v for v in evlrs if getattr(v, "user_id", "") != PROJECTION_VLR_USER_ID]
    try:
        header.global_encoding.wkt = True
    except AttributeError:
        pass
    return header


def validate_single_crs_record(source_file: Path, copc_file: Path) -> Tuple[bool, str]:
    """Validate that an output carries exactly one CRS record, equal to the source CRS.

    The comparison is semantic (same authority code or equal CRS definition),
    not byte-wise: writers standardize CRS text. Duplicate or GeoTIFF+WKT
    records in a LAS 1.4 output fail, since readers would disagree about them.
    """
    import laspy

    try:
        with laspy.open(str(source_file), laz_backend=laspy_laz_backend()) as src:
            source_header = src.header
            source_crs = parse_crs(source_header)
            source_projection = projection_vlr_fingerprints(source_header)
        # Writers' empty placeholder records (untwine writes a 1-byte NUL WKT
        # VLR for sources without CRS) are not CRS metadata.
        source_content = {key: data for key, data in source_projection.items()
                          if data.rstrip(b"\0").strip()}
        if not source_crs and not source_content:
            return (True, "source has no CRS metadata" if not source_projection
                    else "source has no readable CRS metadata")
        with laspy.open(str(copc_file), laz_backend=laspy_laz_backend()) as out:
            output_header = out.header
            output_crs = parse_crs(output_header)
            records = projection_record_list(output_header)
            version = str(output_header.version)
    except Exception as e:
        return (False, f"could not validate CRS metadata: {e}")

    if not records:
        return (False, "source CRS missing in output")
    if source_crs is None:
        # Unreadable but present source CRS: only an unchanged copy is provable.
        output_payloads = {data for _, _, data in records}
        if all(data in output_payloads for data in source_content.values()):
            return (True, "unreadable source CRS records kept unchanged")
        return (False, "unreadable source CRS records missing or changed in output")
    if output_crs is None:
        return (False, "source CRS missing in output")
    if version == "1.4" and len(records) != 1:
        found = ", ".join(f"{kind} {record_id}" for kind, record_id, _ in records)
        return (False, f"output carries {len(records)} CRS records ({found}); exactly one standardized WKT record is required")
    if not crs_equivalent(source_crs, output_crs):
        return (False, f"output CRS {output_crs.name!r} differs from source CRS {source_crs.name!r}")
    return (True, "single CRS record, equal to the source CRS")


def crs_comment_value(header) -> Optional[str]:
    """``EPSG:<code>`` or single-line standardized WKT, None without a readable CRS."""
    from pyproj import CRS

    data = standardized_crs_record(header)
    if data is None:
        return None
    text = data.rstrip(b"\0").decode("utf-8")
    epsg = CRS.from_wkt(text).to_epsg(min_confidence=STANDARD_EPSG_CONFIDENCE)
    value = f"EPSG:{epsg}" if epsg else " ".join(text.split())
    if not value.isascii() or len(value) + 14 > PLY_COMMENT_MAX_BYTES:
        # PLY headers are ASCII lines; the LAS/COPC products keep the full CRS.
        print(f"  Warning: CRS cannot be written as a PLY comment ({len(value)} characters)")
        return None
    return value
