#!/usr/bin/env python3
"""Shared LAS/LAZ metadata helpers for SmartTile products."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Set

import laspy
import numpy as np
from laspy.vlrs.vlrlist import VLRList

from crs_records import standardize_projection_records
from output_names import file_stem


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


def extra_bytes_attribute_equal(left, right) -> bool:
    """Compare descriptor attributes, including declared NaN no-data values."""
    if left is None or right is None:
        return left is right
    left, right = np.asarray(left), np.asarray(right)
    numeric_nan = left.dtype.kind in "fc" and right.dtype.kind in "fc"
    return np.array_equal(left, right, equal_nan=numeric_nan)


def extra_bytes_metadata_equal(left, right) -> bool:
    """Compare the scaling and missing-value semantics of two ExtraBytesParams."""
    return all(extra_bytes_attribute_equal(getattr(left, name), getattr(right, name))
               for name in ("scales", "offsets", "no_data"))


def extra_bytes_params_compatible(left, right) -> bool:
    """Whether a shared field can retain one storage type and interpretation."""
    return np.dtype(left.type) == np.dtype(right.type) and extra_bytes_metadata_equal(left, right)


def extra_bytes_params_from_dimension_info(
    dim_info,
    name: Optional[str] = None,
    header: Optional[laspy.LasHeader] = None,
) -> laspy.ExtraBytesParams:
    """Build ExtraBytesParams from laspy DimensionInfo while preserving metadata."""
    no_data = getattr(dim_info, "no_data", None)
    # laspy currently omits ExtraBytes no-data values when it turns the VLR
    # structs back into DimensionInfo objects. Recover the value from the VLR
    # so remap background values and rewritten metadata remain exact.
    if no_data is None and header is not None:
        for vlr in header.vlrs:
            for struct in getattr(vlr, "extra_bytes_structs", ()):
                if struct.format_name() == dim_info.name:
                    no_data = struct.no_data
                    break
            if no_data is not None:
                break
    return laspy.ExtraBytesParams(
        name=name or dim_info.name,
        type=dim_info.dtype,
        description=getattr(dim_info, "description", "") or "",
        offsets=getattr(dim_info, "offsets", None),
        scales=getattr(dim_info, "scales", None),
        no_data=no_data,
    )


def validate_collection_extra_bytes(paths: List[Path]) -> None:
    """Require one ExtraBytes interpretation before combining source records.

    A single product cannot declare different sentinels/scales for different
    input files. Check originals before conversion and staged inputs before
    selection so the first file's metadata cannot redefine another's values.
    """
    if len(paths) < 2:
        return
    expected = None
    for path in paths:
        with laspy.open(path) as reader:
            actual = {dim.name: extra_bytes_params_from_dimension_info(dim, header=reader.header)
                      for dim in reader.header.point_format.extra_dimensions}
        if expected is None:
            expected = actual
        elif (actual.keys() != expected.keys()
              or any(not extra_bytes_params_compatible(expected[name], actual[name]) for name in expected)):
            raise ValueError(f"{path.name}: ExtraBytes encoding or missingness differs from {paths[0].name}")


def _physical_extra_values(values, params):
    """Interpret stored values with the same float64 scaling used by LAS readers."""
    values = np.asarray(values)
    if params.scales is not None:
        values = values * np.asarray(params.scales, dtype=np.float64)
    if params.offsets is not None:
        values = values + np.asarray(params.offsets, dtype=np.float64)
    return values


def _converted_no_data(source, target):
    """Translate a missing-value declaration without rounding away its meaning."""
    if source.no_data is None:
        return None
    source_dtype, target_dtype = np.dtype(source.type), np.dtype(target.type)
    source_base, target_base = source_dtype.base, target_dtype.base
    source_count = source_dtype.itemsize // source_base.itemsize
    target_count = target_dtype.itemsize // target_base.itemsize
    if source_count != target_count:
        raise ValueError(f"{source.name}: ExtraBytes component count changed")
    if (source_dtype == target_dtype
            and extra_bytes_attribute_equal(source.scales, target.scales)
            and extra_bytes_attribute_equal(source.offsets, target.offsets)):
        return np.asarray(source.no_data, dtype=target_base)
    if (source_base.kind in 'iu' and target_base.kind in 'iu'
            and target_base.itemsize < source_base.itemsize):
        # Checking the sentinel and its neighbours cannot detect distant values
        # wrapping/clamping onto it during a narrowing integer cast. Without a
        # complete value-range proof, keep the source integer width instead.
        raise ValueError(f"{source.name}: no_data is ambiguous when narrowing integer storage")

    physical = _physical_extra_values(np.asarray(source.no_data, dtype=source_base), source)
    encoded = np.asarray(physical, dtype=np.longdouble)
    if target.offsets is not None:
        encoded = encoded - np.asarray(target.offsets, dtype=np.longdouble)
    if target.scales is not None:
        scales = np.asarray(target.scales, dtype=np.longdouble)
        if np.any(scales == 0) or not np.isfinite(scales).all():
            raise ValueError(f"{source.name}: invalid output ExtraBytes scale")
        encoded = encoded / scales
    if target_base.kind in 'iu':
        limits = np.iinfo(target_base)
        if (not np.isfinite(encoded).all() or np.any(encoded != np.rint(encoded))
                or np.any(encoded < limits.min) or np.any(encoded > limits.max)):
            raise ValueError(f"{source.name}: no_data cannot be represented by {target_dtype}")
    with np.errstate(over='ignore', invalid='ignore'):
        converted = encoded.astype(target_base)
    restored = _physical_extra_values(converted, target)
    # longdouble keeps integer sentinels above 2**53 exact for this comparison.
    # NaN declarations retain NaN missingness; a finite value may not become NaN/Inf.
    if not np.array_equal(np.asarray(physical, dtype=np.longdouble),
                          np.asarray(restored, dtype=np.longdouble), equal_nan=True):
        raise ValueError(f"{source.name}: no_data loses precision in {target_dtype}")
    # A representable sentinel is insufficient: a neighbouring valid raw
    # value must not round onto it when scaling or narrowing storage. Check
    # neighbours in the original dtype before scaling loses integer bits.
    # Same-encoding copies return above; NaN retains its missing-value class.
    raw = np.asarray(source.no_data, dtype=source_base)
    for step in (-1, 1):
        if source_base.kind in 'iu':
            bounds = np.iinfo(source_base)
            neighbours = raw.copy()
            eligible = np.array([bounds.min <= int(value) + step <= bounds.max for value in raw])
            for index in np.flatnonzero(eligible):
                neighbours[index] = int(raw[index]) + step
        else:
            neighbours = np.nextafter(raw, np.full(raw.shape, step * np.inf, dtype=source_base))
            eligible = ~np.isnan(raw) & (neighbours != raw)
        values = np.asarray(_physical_extra_values(neighbours, source), dtype=np.longdouble)
        if source_base.kind == 'f':
            # Scaling can underflow a float neighbour onto no_data in the
            # source's own reading; that neighbour gains no new alias.
            eligible &= values != np.asarray(physical, dtype=np.longdouble)
        if target.offsets is not None:
            values = values - np.asarray(target.offsets, dtype=np.longdouble)
        if target.scales is not None:
            values = values / np.asarray(target.scales, dtype=np.longdouble)
        if target_base.kind in 'iu':
            limits = np.iinfo(target_base)
            eligible &= np.isfinite(values) & (values >= limits.min) & (values <= limits.max)
        with np.errstate(over='ignore', invalid='ignore'):
            aliases = values.astype(target_base) == converted
        if np.any(eligible & aliases):
            raise ValueError(f"{source.name}: no_data becomes indistinguishable from a valid source value")
    return converted


def _expected_extra_no_data(source_header, output_header):
    targets = {dim.name: extra_bytes_params_from_dimension_info(dim, header=output_header)
               for dim in output_header.point_format.extra_dimensions}
    expected = {}
    for dim in source_header.point_format.extra_dimensions:
        source = extra_bytes_params_from_dimension_info(dim, header=source_header)
        if dim.name not in targets:
            # Vector transport validates its renamed scalar components separately.
            if np.dtype(dim.dtype).subdtype is not None:
                continue
            raise ValueError(f"{dim.name}: output dropped an ExtraBytes dimension")
        expected[dim.name] = _converted_no_data(source, targets[dim.name])
    return expected


def validate_extra_bytes_missingness(source: Path, output: Path) -> None:
    """Raise when converted declarations do not identify the source missing values."""
    with laspy.open(source) as reader:
        source_header = reader.header
    with laspy.open(output) as reader:
        output_header = reader.header
    expected = _expected_extra_no_data(source_header, output_header)
    for dim in output_header.point_format.extra_dimensions:
        if dim.name in expected:
            actual = extra_bytes_params_from_dimension_info(dim, header=output_header).no_data
            if not extra_bytes_attribute_equal(actual, expected[dim.name]):
                raise ValueError(f"{dim.name}: output no_data declaration differs from source missingness")


def preserve_extra_bytes_missingness(source: Path, output: Path) -> None:
    """Repair no_data in a private LAS/LAZ/COPC output, then validate its meaning.

    Converters may materialize scaled integer fields as unscaled floating values.
    Keep that output encoding and translate the sentinel into it. Only the existing
    fixed-size ExtraBytes VLR payload changes: point bytes, COPC hierarchy offsets,
    and all other VLR/EVLR records remain untouched. Call before output publication.
    """
    import struct

    with laspy.open(source) as reader:
        source_header = reader.header
    with laspy.open(output) as reader:
        output_header = reader.header
    expected = _expected_extra_no_data(source_header, output_header)
    if not expected:
        return
    vlrs = [vlr for vlr in output_header.vlrs if is_extra_bytes_vlr(vlr)]
    if len(vlrs) != 1:
        raise ValueError("Output needs one ExtraBytes VLR to preserve missingness")
    original = vlrs[0].record_data_bytes()
    updated = deepcopy(vlrs[0])
    for entry in updated.extra_bytes_structs:
        name = entry.format_name()
        if name in expected:
            entry.no_data = expected[name]
    replacement = updated.record_data_bytes()
    if replacement == original:
        validate_extra_bytes_missingness(source, output)
        return
    if len(replacement) != len(original):
        raise ValueError("ExtraBytes repair must not move the point-data offset")

    # Locate the regular VLR directly instead of rewriting a COPC header/index.
    vlr_header = struct.Struct('<H16sHH32s')
    with Path(output).open('r+b') as handle:
        handle.seek(94)
        header_size, point_offset, count = struct.unpack('<HII', handle.read(10))
        handle.seek(header_size)
        for _ in range(count):
            _, user_id, record_id, length, _ = vlr_header.unpack(handle.read(vlr_header.size))
            position = handle.tell()
            if position + length > point_offset:
                raise ValueError("ExtraBytes VLR extends beyond the point-data offset")
            if user_id.rstrip(b'\0 ') == b'LASF_Spec' and record_id == 4:
                if handle.read(length) != original:
                    raise ValueError("Output ExtraBytes metadata changed before repair")
                break
            handle.seek(length, 1)
        else:
            raise ValueError("Output ExtraBytes VLR was not found")
        try:
            handle.seek(position)
            if handle.write(replacement) != len(replacement):
                raise OSError("Incomplete ExtraBytes metadata write")
            handle.flush()
            validate_extra_bytes_missingness(source, output)
        except BaseException:
            handle.seek(position)
            handle.write(original)
            handle.flush()
            raise


def is_stale_copc_vlr(vlr) -> bool:
    """Return True for COPC index VLRs that cannot be copied to a new container."""
    return getattr(vlr, "user_id", "") == "copc" and getattr(vlr, "record_id", None) in (1, 2, 1000)


def is_extra_bytes_vlr(vlr) -> bool:
    """Return True for ExtraBytes metadata that laspy regenerates from dimensions."""
    return getattr(vlr, "user_id", "") == "LASF_Spec" and getattr(vlr, "record_id", None) == 4


def update_extra_dimensions(header, params, *, replace=False):
    """Augment a header without regenerating untouched ExtraBytes descriptors.

    laspy rebuilds the entire ExtraBytes VLR when dimensions change, but its
    reader does not retain every descriptor field in DimensionInfo. Preserve
    the original descriptors (including no-data and option bits) verbatim.
    Replaced fields deliberately take their new schema from ``params``.
    """
    params = list(params)
    changed = {param.name for param in params}
    retained = {struct.format_name(): deepcopy(struct)
                for vlr in header.vlrs
                for struct in getattr(vlr, "extra_bytes_structs", ())
                if struct.format_name() not in changed}
    if replace:
        existing = set(header.point_format.extra_dimension_names)
        for name in sorted(changed.intersection(existing)):
            header.remove_extra_dim(name)
    header.add_extra_dims(params)
    for vlr in header.vlrs:
        if hasattr(vlr, "extra_bytes_structs"):
            vlr.extra_bytes_structs = [retained.get(struct.format_name(), struct)
                                      for struct in vlr.extra_bytes_structs]


def write_retained_evlrs(writer, header):
    """Finish a streaming writer with retained extended metadata after points."""
    if header.evlrs:
        writer.write_evlrs(header.evlrs)


def copy_single_source_header(
    source_header: laspy.LasHeader,
    offsets=None,
    scales=None,
    preserve_extra_dimensions: bool = True,
) -> laspy.LasHeader:
    """Copy source metadata, promoting LAS 1.0 to a writable output version."""
    # laspy can read historical LAS 1.0 uploads but cannot write that version.
    # Only promote this legacy version; retain point format, encoding and VLRs.
    output_version = (
        laspy.header.Version(1, 4)
        if str(source_header.version) == "1.0" else source_header.version
    )
    if preserve_extra_dimensions:
        header = source_header.copy()
        if header.version != output_version:
            header.version = output_version
    else:
        fmt_id = getattr(source_header.point_format, "id", source_header.point_format)
        if hasattr(fmt_id, "id"):
            fmt_id = fmt_id.id
        header = laspy.LasHeader(point_format=int(fmt_id), version=output_version)
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
    # Assigning header.vlrs invokes laspy's setter and regenerates ExtraBytes,
    # losing descriptor fields that its reader omitted from DimensionInfo.
    header.vlrs[:] = [
        deepcopy(vlr) for vlr in source_vlrs
        if not is_stale_copc_vlr(vlr)
        and (preserve_extra_dimensions or not is_extra_bytes_vlr(vlr))
    ]
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

    return standardize_projection_records(header)


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
        is_copc = path.name.lower().endswith(".copc.laz")
        if existing is not None and existing.name.lower().endswith(".copc.laz") == is_copc:
            # e.g. Plot1.las and plot1.laz: two sources, one key; never drop one silently.
            raise ValueError(f"{existing.name} and {path.name} name the same source; rename one of them")
        if existing is None or is_copc:
            by_source[key] = path
    return [by_source[key] for key in sorted(by_source)]


def raw_point_cloud_files(directory: Optional[Path]) -> List[Path]:
    """Return only non-COPC LAS/LAZ point-cloud files."""
    files = _point_cloud_paths(directory)
    raw = [path for path in files if not path.name.lower().endswith(".copc.laz")]
    keys = {}
    for path in raw:
        key = point_cloud_source_key(path)
        if key in keys:
            raise ValueError(f"{keys[key].name} and {path.name} name the same source; rename one of them")
        keys[key] = path
    return raw


def copc_only_files(directory: Optional[Path]) -> List[Path]:
    """Return COPC files whose source has no LAS/LAZ twin in ``directory``."""
    raw_keys = {point_cloud_source_key(path) for path in raw_point_cloud_files(directory)}
    return [path for path in copc_files(directory) if point_cloud_source_key(path) not in raw_keys]


def copc_files(directory: Optional[Path]) -> List[Path]:
    """Return COPC LAZ point-cloud files."""
    return [path for path in _point_cloud_paths(directory) if path.name.lower().endswith(".copc.laz")]


def point_cloud_source_key(path: Path) -> str:
    """Return a stable source key shared by raw LAS/LAZ and derived COPC files."""
    return file_stem(path).lower()


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


