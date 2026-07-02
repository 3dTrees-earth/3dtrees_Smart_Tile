#!/usr/bin/env python3
"""Shared LAS/LAZ metadata helpers for SmartTile products."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Set, Tuple

import laspy
from laspy.vlrs.vlrlist import VLRList


# R/lidR, LAS, and laspy names can all appear in standardization summaries.
# Extra-byte names are preserved as-is via the .get() default.
DIMENSION_NAME_ALIASES = {
    "Intensity": "intensity",
    "intensity": "intensity",
    "ReturnNumber": "return_number",
    "return_number": "return_number",
    "NumberOfReturns": "number_of_returns",
    "number_of_returns": "number_of_returns",
    "ScanDirectionFlag": "scan_direction_flag",
    "scan_direction_flag": "scan_direction_flag",
    "EdgeOfFlightline": "edge_of_flight_line",
    "edge_of_flight_line": "edge_of_flight_line",
    "Classification": "classification",
    "classification": "classification",
    "ScannerChannel": "scanner_channel",
    "scanner_channel": "scanner_channel",
    "Synthetic_flag": "synthetic",
    "synthetic": "synthetic",
    "Keypoint_flag": "key_point",
    "key_point": "key_point",
    "Withheld_flag": "withheld",
    "withheld": "withheld",
    "Overlap_flag": "overlap",
    "overlap": "overlap",
    "ScanAngle": "scan_angle",
    "scan_angle": "scan_angle",
    "ScanAngleRank": "scan_angle",
    "scan_angle_rank": "scan_angle",
    "UserData": "user_data",
    "user_data": "user_data",
    "PointSourceID": "point_source_id",
    "point_source_id": "point_source_id",
    "gpstime": "gps_time",
    "gps_time": "gps_time",
    "R": "red",
    "red": "red",
    "G": "green",
    "green": "green",
    "B": "blue",
    "blue": "blue",
}


def extra_bytes_params_from_dimension_info(
    dim_info,
    name: Optional[str] = None,
) -> laspy.ExtraBytesParams:
    """Build ExtraBytesParams from laspy DimensionInfo while preserving metadata."""
    return laspy.ExtraBytesParams(
        name=name or dim_info.name,
        type=dim_info.dtype,
        description=getattr(dim_info, "description", "") or "",
        offsets=getattr(dim_info, "offsets", None),
        scales=getattr(dim_info, "scales", None),
        no_data=getattr(dim_info, "no_data", None),
    )


def extra_bytes_params_from_params(
    params: laspy.ExtraBytesParams,
    name: Optional[str] = None,
) -> laspy.ExtraBytesParams:
    """Clone ExtraBytesParams while optionally renaming the dimension."""
    return laspy.ExtraBytesParams(
        name=name or params.name,
        type=params.type,
        description=getattr(params, "description", "") or "",
        offsets=getattr(params, "offsets", None),
        scales=getattr(params, "scales", None),
        no_data=getattr(params, "no_data", None),
    )


def is_stale_copc_vlr(vlr) -> bool:
    """Return True for COPC index VLRs that cannot be copied to a new container."""
    return getattr(vlr, "user_id", "") == "copc" and getattr(vlr, "record_id", None) in (1, 2)


def is_extra_bytes_vlr(vlr) -> bool:
    """Return True for ExtraBytes metadata that laspy regenerates from dimensions."""
    return getattr(vlr, "user_id", "") == "LASF_Spec" and getattr(vlr, "record_id", None) == 4


def copy_single_source_header(
    source_header: laspy.LasHeader,
    offsets=None,
    scales=None,
    preserve_extra_dimensions: bool = True,
) -> laspy.LasHeader:
    """Copy a source header for outputs that still represent exactly that source file."""
    if preserve_extra_dimensions:
        header = source_header.copy()
    else:
        fmt_id = getattr(source_header.point_format, "id", source_header.point_format)
        if hasattr(fmt_id, "id"):
            fmt_id = fmt_id.id
        header = laspy.LasHeader(point_format=int(fmt_id), version=source_header.version)
        for attr in (
            "file_source_id",
            "global_encoding",
            "uuid",
            "system_identifier",
            "generating_software",
            "creation_date",
        ):
            if hasattr(source_header, attr):
                setattr(header, attr, getattr(source_header, attr))

    header.offsets = offsets if offsets is not None else source_header.offsets
    header.scales = scales if scales is not None else source_header.scales

    source_vlrs = header.vlrs if preserve_extra_dimensions else source_header.vlrs
    header.vlrs = VLRList([
        vlr for vlr in source_vlrs
        if not is_stale_copc_vlr(vlr)
        and (preserve_extra_dimensions or not is_extra_bytes_vlr(vlr))
    ])
    source_evlrs = (
        getattr(header, "evlrs", None)
        if preserve_extra_dimensions
        else getattr(source_header, "evlrs", None)
    )
    if source_evlrs is not None:
        header.evlrs = VLRList([
            vlr for vlr in source_evlrs
            if not is_stale_copc_vlr(vlr)
            and (preserve_extra_dimensions or not is_extra_bytes_vlr(vlr))
        ])

    return header


def projection_metadata_vlrs(vlrs) -> VLRList:
    """Keep CRS/projection records that remain true across a CRS-consistent run."""
    return VLRList([
        vlr for vlr in (vlrs or [])
        if getattr(vlr, "user_id", "") == "LASF_Projection"
    ])


def _point_cloud_paths(directory: Optional[Path]) -> List[Path]:
    """Return LAS/LAZ-like paths case-insensitively."""
    if directory is None or not directory.exists():
        return []
    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file()
            and path.name.lower().endswith((".laz", ".las"))
        ),
        key=lambda path: path.name.lower(),
    )


def point_cloud_files(directory: Optional[Path]) -> List[Path]:
    """Return point-cloud files, preferring COPC over matching plain LAZ/LAS."""
    files = _point_cloud_paths(directory)
    by_source = {}
    for path in files:
        key = point_cloud_source_key(path)
        existing = by_source.get(key)
        if existing is None or path.name.lower().endswith(".copc.laz"):
            by_source[key] = path
    return [by_source[key] for key in sorted(by_source)]


def raw_point_cloud_files(directory: Optional[Path]) -> List[Path]:
    """Return only non-COPC LAS/LAZ point-cloud files."""
    files = _point_cloud_paths(directory)
    return [path for path in files if not path.name.lower().endswith(".copc.laz")]


def copc_files(directory: Optional[Path]) -> List[Path]:
    """Return COPC LAZ point-cloud files."""
    return [path for path in _point_cloud_paths(directory) if path.name.lower().endswith(".copc.laz")]


def point_cloud_source_key(path: Path) -> str:
    """Return a stable source key shared by raw LAS/LAZ and derived COPC files."""
    name = path.name.lower()
    if name.endswith(".copc.laz"):
        return name[:-9]
    if name.endswith(".laz") or name.endswith(".las"):
        return name.rsplit(".", 1)[0]
    return path.stem.lower()


def point_cloud_dimension_names(path: Path) -> Set[str]:
    """Return all standard and extra dimension names in a point-cloud header."""
    with laspy.open(str(path), laz_backend=laspy.LazBackend.LazrsParallel) as reader:
        names = set(str(name) for name in reader.header.point_format.dimension_names)
        names.update(dim.name for dim in reader.header.point_format.extra_dimensions)
        return names


def load_standardization_dims(json_path: Path) -> Set[str]:
    """Load expected source dimensions from a tool_standard collection summary.

    The v2.1 SmartTile contract accepted `collection_summary.json` and used its
    `collection.reference_attribute_names` as the canonical source-attribute
    schema. Constant/all-zero dimensions are ignored when
    `global_attribute_stats` is present, matching the old behavior.
    """
    with Path(json_path).open() as handle:
        data = json.load(handle)

    collection = data.get("collection", data)
    ref_names = collection.get("reference_attribute_names")
    if not isinstance(ref_names, list):
        raise ValueError(
            f"standardization JSON does not contain collection.reference_attribute_names: {json_path}"
        )

    global_stats = collection.get("global_attribute_stats", [])
    has_variation = set()
    if isinstance(global_stats, list):
        for stat in global_stats:
            if not isinstance(stat, dict):
                continue
            name = stat.get("name", "")
            variance = stat.get("variance", 0)
            try:
                if float(variance) > 0:
                    has_variation.add(name)
            except (TypeError, ValueError):
                pass

    expected = set()
    skipped = []
    for name in ref_names:
        if name in ("X", "Y", "Z"):
            continue
        if has_variation and name not in has_variation:
            skipped.append(name)
            continue
        expected.add(DIMENSION_NAME_ALIASES.get(name, name))

    if skipped:
        print(
            f"  Standardization JSON: skipping {len(skipped)} constant/zero dims: {skipped}",
            flush=True,
        )
    return expected


def bounds_overlap_xy(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
    buffer: float = 0.0,
) -> bool:
    """Return whether two XY bounds overlap, optionally expanding both by buffer."""
    return not (
        a[1] < b[0] - buffer
        or a[0] > b[1] + buffer
        or a[3] < b[2] - buffer
        or a[2] > b[3] + buffer
    )
