"""CRS records of output files.

SmartTile's own writers copy the source's CRS records byte for byte, in every
LAS/LAZ output (3DT-2718): a rewritten record is a changed record for readers
that compare CRSs strictly, and the EPSG text depends on the PROJ build.
Point-cloud writers (PDAL/GDAL, untwine) re-serialize CRS text, so COPC and PDAL
products cannot keep the source's bytes; they carry one WKT record and are
validated semantically. The standardized text below only labels PLY comments.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from laspy.vlrs import known

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


def _keep_source_bytes(vlr_class) -> None:
    """Make laspy write a CRS record as it read it.

    laspy parses these records and serializes them anew, which drops NUL padding
    and repairs key counts. A record changed after reading is written as changed.
    """
    parse, serialize = vlr_class.parse_record_data, vlr_class.record_data_bytes

    def parse_record_data(self, record_data):
        parse(self, record_data)
        self.source_bytes = bytes(record_data), serialize(self)

    def record_data_bytes(self):
        current = serialize(self)
        source, parsed = getattr(self, "source_bytes", (current, current))
        return source if current == parsed else current

    vlr_class.parse_record_data, vlr_class.record_data_bytes = parse_record_data, record_data_bytes


for _record in (known.GeoKeyDirectoryVlr, known.GeoDoubleParamsVlr, known.GeoAsciiParamsVlr,
                known.WktMathTransformVlr, known.WktCoordinateSystemVlr):
    _keep_source_bytes(_record)


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


def validate_crs_records_unchanged(source_file: Path, output_file: Path) -> None:
    """Raise unless an output written by SmartTile carries the source's CRS records byte for byte."""
    import laspy

    with laspy.open(str(source_file), laz_backend=laspy_laz_backend()) as src, \
            laspy.open(str(output_file), laz_backend=laspy_laz_backend()) as out:
        if projection_record_list(src.header) != projection_record_list(out.header):
            raise ValueError(f"{Path(output_file).name}: CRS records differ from {Path(source_file).name}")
        if src.header.global_encoding.wkt != out.header.global_encoding.wkt:
            raise ValueError(f"{Path(output_file).name}: WKT flag differs from {Path(source_file).name}")


UNNAMED_DATUMS = {"", "unknown", "unnamed"}


def _split_crs(crs):
    """Horizontal CRS and vertical datum names of ``crs`` (source of a bound CRS, parts of a compound CRS)."""
    if getattr(crs, "is_bound", False) and crs.source_crs is not None:
        crs = crs.source_crs
    if not crs.is_compound:
        return crs, []
    parts = [sub.source_crs if getattr(sub, "is_bound", False) and sub.source_crs is not None else sub
             for sub in crs.sub_crs_list]
    return parts[0], [sub.datum.name if sub.datum else "" for sub in parts[1:]]


def same_horizontal_crs_unknown_vertical(source_crs, output_crs) -> bool:
    """True when two CRSs share the projection and neither names a vertical datum.

    COPC writers turn a source whose GeoTIFF keys carry an unknown vertical
    datum (dataset 1368: SIRGAS 2000 / UTM 23S) into a compound CRS with an
    "unknown" vertical part and an "unnamed" horizontal datum; nothing about
    the coordinates changed.
    """
    import warnings

    source_horizontal, source_vertical = _split_crs(source_crs)
    output_horizontal, output_vertical = _split_crs(output_crs)
    if any(name.strip().lower() not in UNNAMED_DATUMS for name in [*source_vertical, *output_vertical]):
        return False
    if not (source_horizontal.is_projected and output_horizontal.is_projected):
        return False
    datums = {(crs.datum.name if crs.datum else "").strip().lower() for crs in (source_horizontal, output_horizontal)}
    if len(datums - UNNAMED_DATUMS) > 1:
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return source_horizontal.to_2d().to_proj4() == output_horizontal.to_2d().to_proj4()


def validate_single_crs_record(source_file: Path, copc_file: Path) -> Tuple[bool, str]:
    """Validate that an output keeps the source's CRS records, or carries one equal CRS record.

    The second comparison is semantic (same authority code or equal CRS
    definition): PDAL and untwine rewrite CRS text. Duplicate or GeoTIFF+WKT
    records that a writer added to a LAS 1.4 output fail, since readers would
    disagree about them.
    """
    import laspy

    try:
        with laspy.open(str(source_file), laz_backend=laspy_laz_backend()) as src:
            source_header = src.header
            source_crs = parse_crs(source_header)
            source_projection = projection_vlr_fingerprints(source_header)
            source_records = projection_record_list(source_header)
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
    if records == source_records:
        return (True, "source CRS records kept unchanged")
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
        return (False, f"output carries {len(records)} CRS records ({found}); a rewritten CRS must be exactly one WKT record")
    if not crs_equivalent(source_crs, output_crs) and not same_horizontal_crs_unknown_vertical(source_crs, output_crs):
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
