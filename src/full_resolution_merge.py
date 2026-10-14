"""Full-resolution prod-merged product: every input point, in input record order.

Inputs are streamed file by file (sorted file names) and record by record, so
the optional ``OriginalPointIndex`` is each point's 0-based position in that
concatenation. It is the key ForestStructure's ordered-COPC builder adds to
restore source record order after COPC reorders points (Julia's script breaks
ties by record order), with the same layout: a uint64 first ExtraByte, then the
PredInstance dimensions ForestStructure reads selectively, then all others.
"""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import List

import laspy
import numpy as np

from point_cloud_metadata import (
    copy_single_source_header,
    extra_bytes_params_compatible,
    extra_bytes_params_from_dimension_info,
    write_retained_evlrs,
)

ORIGINAL_POINT_INDEX = "OriginalPointIndex"
# Addressable through lidR selectors 2.. after the order key (ForestStructure).
SELECTIVE_DIMENSIONS = ("PredInstance", "PredInstance_SAT", "PredInstance_FM")
READ_POINTS = 1_000_000
INT32 = np.iinfo(np.int32)
UINT32 = np.iinfo(np.uint32)


def _schema(header):
    return {dimension.name: extra_bytes_params_from_dimension_info(dimension, header=header)
            for dimension in header.point_format.extra_dimensions}


def merged_header(inputs: List[Path], original_point_index: bool) -> laspy.LasHeader:
    """One header for every input: same schema and CRS, finest scale, int32-safe offsets."""
    headers = []
    for path in inputs:
        with laspy.open(path) as reader:
            headers.append(reader.header)
    first = headers[0]
    extras = _schema(first)
    for path, header in zip(inputs[1:], headers[1:]):
        actual = _schema(header)
        if (header.point_format.id != first.point_format.id or list(actual) != list(extras)
                or any(not extra_bytes_params_compatible(extras[name], actual[name]) for name in extras)):
            raise ValueError(f"{path.name}: point format or ExtraBytes differ from {inputs[0].name}; "
                             "a full-resolution merge needs one schema")
        if header.parse_crs() != first.parse_crs():
            raise ValueError(f"{path.name}: CRS differs from {inputs[0].name}")
    names = list(extras)
    if original_point_index and ORIGINAL_POINT_INDEX in names:
        raise ValueError(f"Inputs already contain {ORIGINAL_POINT_INDEX}; disable --original-point-index")
    scales = np.min([h.scales for h in headers], axis=0)
    mins = np.min([h.mins for h in headers], axis=0)
    maxs = np.max([h.maxs for h in headers], axis=0)
    offsets = np.asarray(first.offsets, dtype=float)
    if np.any((mins - offsets) / scales < INT32.min) or np.any((maxs - offsets) / scales > INT32.max):
        offsets = np.floor(mins)
    header = copy_single_source_header(first, offsets=offsets, scales=scales)
    header.remove_extra_dims(names)
    if original_point_index:
        header.add_extra_dim(laspy.ExtraBytesParams(
            name=ORIGINAL_POINT_INDEX, type=np.uint64,
            description="Record index in input order"))
    ordered = [n for n in SELECTIVE_DIMENSIONS if n in extras] + [n for n in names if n not in SELECTIVE_DIMENSIONS]
    header.add_extra_dims([extras[name] for name in ordered])
    return header


def _ply_header(header: laspy.LasHeader) -> laspy.LasHeader:
    """Use PLY's uint32 order key; reject other unsupported integer storage."""
    extras = _schema(header)
    for name, params in extras.items():
        dtype = np.dtype(params.type)
        if name == ORIGINAL_POINT_INDEX:
            if (dtype.kind not in "ui" or dtype.subdtype is not None
                    or params.scales is not None and np.any(np.asarray(params.scales) != 1)
                    or params.offsets is not None and np.any(np.asarray(params.offsets) != 0)):
                raise ValueError(f"PLY {name} must be an unscaled scalar integer")
            extras[name] = laspy.ExtraBytesParams(
                name=name, type=np.uint32, description=params.description)
        elif dtype.base.kind in "ui" and dtype.base.itemsize > 4:
            raise ValueError(f"PLY cannot represent 64-bit integer attribute {name}; "
                             "use LAZ or COPC to retain its values")
    header.remove_extra_dims(list(extras))
    header.add_extra_dims(list(extras.values()))
    return header


def _validate_ply_index_range(minimum: int, maximum: int) -> None:
    if minimum < 0 or maximum > UINT32.max:
        raise ValueError(f"PLY {ORIGINAL_POINT_INDEX} must be in uint32 range "
                         f"0..{UINT32.max}; found {minimum}..{maximum}. Use LAZ or COPC.")


def write_full_resolution(inputs: List[Path], output: Path, *, compress: bool,
                          original_point_index: bool, ply: bool = False) -> int:
    """Concatenate every input record in order into one LAS/LAZ; return the point count."""
    header = merged_header(inputs, original_point_index)
    if ply:
        header = _ply_header(header)
        total = 0
        for path in inputs:
            with laspy.open(path) as reader:
                total += int(reader.header.point_count)
        if total > UINT32.max:
            raise ValueError(f"PLY point count {total} exceeds the writer limit {UINT32.max}; "
                             "use LAZ or COPC")
    standard = [n for n in header.point_format.standard_dimension_names if n not in ("X", "Y", "Z")]
    extras = [d.name for d in header.point_format.extra_dimensions
              if not original_point_index or d.name != ORIGINAL_POINT_INDEX]
    written = expected = 0
    with laspy.open(output, mode="w", header=header, do_compress=compress) as writer:
        for path in inputs:
            with laspy.open(path) as reader:
                expected += int(reader.header.point_count)
                same_grid = (np.array_equal(reader.header.scales, header.scales)
                             and np.array_equal(reader.header.offsets, header.offsets))
                for chunk in reader.chunk_iterator(READ_POINTS):
                    if ply and len(chunk):
                        if original_point_index:
                            _validate_ply_index_range(written, written + len(chunk) - 1)
                        elif ORIGINAL_POINT_INDEX in extras:
                            values = chunk.array[ORIGINAL_POINT_INDEX]
                            _validate_ply_index_range(int(values.min()), int(values.max()))
                    record = laspy.ScaleAwarePointRecord.zeros(len(chunk), header=header)
                    if same_grid:
                        for axis in ("X", "Y", "Z"):
                            record[axis] = chunk[axis]
                    else:
                        record.x, record.y, record.z = chunk.x, chunk.y, chunk.z
                    for name in standard:
                        record[name] = chunk[name]
                    for name in extras:
                        record.array[name] = chunk.array[name]
                    if original_point_index:
                        record[ORIGINAL_POINT_INDEX] = np.arange(written, written + len(chunk), dtype=np.uint64)
                    writer.write_points(record)
                    written += len(chunk)
        write_retained_evlrs(writer, header)
    if written != expected:
        raise RuntimeError(f"Full-resolution merge wrote {written:,} of {expected:,} points")
    return written


def create_full_resolution_product(inputs: List[Path], output_file: Path, output_format: str, *,
                                   original_point_index: bool) -> Path:
    """Write the full-resolution ``merged.<format>`` (LAZ, COPC LAZ or PLY) from the inputs in record order."""
    from product_formats import parse_merged_output_formats

    if not inputs:
        raise ValueError("A full-resolution merge needs LAS/LAZ inputs (COPC inputs have lost record order)")
    output_format = parse_merged_output_formats(output_format)[0]
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "laz":
        write_full_resolution(inputs, output_file, compress=True, original_point_index=original_point_index)
        return output_file
    with tempfile.TemporaryDirectory(prefix=".full-merge-", dir=output_file.parent) as work:
        merged = Path(work) / "merged.las"
        count = write_full_resolution(inputs, merged, compress=False,
                                      original_point_index=original_point_index, ply=output_format == "ply")
        if output_format == "copc.laz":
            from tile_copc import convert_laz_to_copc

            if not convert_laz_to_copc(merged, output_file, preserve_extra_dims=True):
                raise RuntimeError(f"COPC conversion failed for {output_file.name}")
            with laspy.open(output_file) as reader:
                names = [d.name for d in reader.header.point_format.extra_dimensions]
                if reader.header.point_count != count:
                    raise RuntimeError(f"{output_file.name}: COPC holds {reader.header.point_count:,} of {count:,} points")
                if original_point_index and names[:1] != [ORIGINAL_POINT_INDEX]:
                    raise RuntimeError(f"{output_file.name}: {ORIGINAL_POINT_INDEX} is not the first ExtraByte ({names})")
        else:
            from output_publication import publish_files

            candidate = Path(work) / "merged.ply"
            pdal = shutil.which("pdal") or "pdal"
            # LAS 1.2/1.3 can carry ExtraBytes VLRs, which PDAL ignores by default.
            result = subprocess.run([pdal, "translate", str(merged), str(candidate),
                                     "--readers.las.use_eb_vlr=true",
                                     "--writers.ply.storage_mode=little endian"],
                                    capture_output=True, text=True, check=False)
            if result.returncode != 0:
                raise RuntimeError(f"PLY conversion failed: {result.stderr[-500:]}")
            publish_files([(candidate, output_file)], inputs=inputs)
    return output_file
