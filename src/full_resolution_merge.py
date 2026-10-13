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

from point_cloud_metadata import copy_single_source_header, write_retained_evlrs

ORIGINAL_POINT_INDEX = "OriginalPointIndex"
# Addressable through lidR selectors 2.. after the order key (ForestStructure).
SELECTIVE_DIMENSIONS = ("PredInstance", "PredInstance_SAT", "PredInstance_FM")
READ_POINTS = 1_000_000
INT32 = np.iinfo(np.int32)


def _extra_bytes_params(dimension):
    """Clone one ExtraByte definition without changing its storage semantics."""
    return laspy.ExtraBytesParams(name=dimension.name, type=dimension.dtype, description=dimension.description,
                                  offsets=dimension.offsets, scales=dimension.scales, no_data=dimension.no_data)


def _schema(header):
    return (header.point_format.id,
            [(d.name, str(d.dtype)) for d in header.point_format.extra_dimensions])


def merged_header(inputs: List[Path], original_point_index: bool) -> laspy.LasHeader:
    """One header for every input: same schema and CRS, finest scale, int32-safe offsets."""
    headers = []
    for path in inputs:
        with laspy.open(path) as reader:
            headers.append(reader.header)
    first = headers[0]
    for path, header in zip(inputs[1:], headers[1:]):
        if _schema(header) != _schema(first):
            raise ValueError(f"{path.name}: point format or ExtraBytes differ from {inputs[0].name}; "
                             "a full-resolution merge needs one schema")
        if header.parse_crs() != first.parse_crs():
            raise ValueError(f"{path.name}: CRS differs from {inputs[0].name}")
    names = [d.name for d in first.point_format.extra_dimensions]
    if original_point_index and ORIGINAL_POINT_INDEX in names:
        raise ValueError(f"Inputs already contain {ORIGINAL_POINT_INDEX}; disable --original-point-index")
    scales = np.min([h.scales for h in headers], axis=0)
    mins = np.min([h.mins for h in headers], axis=0)
    maxs = np.max([h.maxs for h in headers], axis=0)
    offsets = np.asarray(first.offsets, dtype=float)
    if np.any((mins - offsets) / scales < INT32.min) or np.any((maxs - offsets) / scales > INT32.max):
        offsets = np.floor(mins)
    header = copy_single_source_header(first, offsets=offsets, scales=scales)
    extras = {d.name: d for d in first.point_format.extra_dimensions}
    header.remove_extra_dims(names)
    if original_point_index:
        header.add_extra_dim(laspy.ExtraBytesParams(
            name=ORIGINAL_POINT_INDEX, type=np.uint64,
            description="Record index in input order"))
    ordered = [n for n in SELECTIVE_DIMENSIONS if n in extras] + [n for n in names if n not in SELECTIVE_DIMENSIONS]
    for name in ordered:
        header.add_extra_dim(_extra_bytes_params(extras[name]))
    return header


def write_full_resolution(inputs: List[Path], output: Path, *, compress: bool, original_point_index: bool) -> int:
    """Concatenate every input record in order into one LAS/LAZ; return the point count."""
    header = merged_header(inputs, original_point_index)
    standard = [n for n in header.point_format.standard_dimension_names if n not in ("X", "Y", "Z")]
    extras = [d.name for d in header.point_format.extra_dimensions if d.name != ORIGINAL_POINT_INDEX]
    written = expected = 0
    with laspy.open(output, mode="w", header=header, do_compress=compress) as writer:
        for path in inputs:
            with laspy.open(path) as reader:
                expected += int(reader.header.point_count)
                same_grid = (np.array_equal(reader.header.scales, header.scales)
                             and np.array_equal(reader.header.offsets, header.offsets))
                for chunk in reader.chunk_iterator(READ_POINTS):
                    record = laspy.ScaleAwarePointRecord.zeros(len(chunk), header=header)
                    if same_grid:
                        for axis in ("X", "Y", "Z"):
                            record[axis] = chunk[axis]
                    else:
                        record.x, record.y, record.z = chunk.x, chunk.y, chunk.z
                    for name in standard + extras:
                        record[name] = chunk[name]
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
        count = write_full_resolution(inputs, merged, compress=False, original_point_index=original_point_index)
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
            pdal = shutil.which("pdal") or "pdal"
            result = subprocess.run([pdal, "translate", str(merged), str(output_file),
                                     "--writers.ply.storage_mode=little endian"],
                                    capture_output=True, text=True, check=False)
            if result.returncode != 0:
                raise RuntimeError(f"PLY conversion failed: {result.stderr[-500:]}")
    return output_file
