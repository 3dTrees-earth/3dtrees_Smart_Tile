"""Convert staged instance tiles without dropping their application dimensions."""
from __future__ import annotations

from pathlib import Path

import laspy
import numpy as np

from crs_records import crs_comment_value

PLY_TYPES = {"i1": "char", "u1": "uchar", "i2": "short", "u2": "ushort",
             "i4": "int", "u4": "uint", "f4": "float", "f8": "double"}


def _ply_fields(header, dimensions):
    fields = [("x", np.dtype("<f8")), ("y", np.dtype("<f8")), ("z", np.dtype("<f8"))]
    for name in dimensions:
        if not name.isascii() or not name or any(c.isspace() or ord(c) < 32 for c in name):
            raise ValueError(f"PLY needs an ASCII property name without whitespace: {name!r}; use LAZ or COPC")
        if name in {field[0] for field in fields}:
            raise ValueError(f"PLY has a duplicate property {name!r}")
        info = header.point_format.dimension_by_name(name)
        dtype = np.dtype(info.dtype or np.uint8)
        if dtype.subdtype:
            raise ValueError(f"PLY cannot retain vector attribute {name}; use LAZ or COPC")
        scaled = ((info.scales is not None and np.any(np.asarray(info.scales) != 1))
                  or (info.offsets is not None and np.any(np.asarray(info.offsets) != 0)))
        if scaled:
            if dtype.kind in "iu" and dtype.itemsize > 4:
                raise ValueError(f"PLY cannot retain scaled 64-bit integer attribute {name}; use LAZ or COPC")
            dtype = np.dtype("<f8")  # PLY stores the interpreted measurement value.
        elif dtype.kind in "iu" and dtype.itemsize > 4:
            dtype = np.dtype(f"<{dtype.kind}4")  # Each chunk is range-checked before casting.
        else:
            dtype = dtype.newbyteorder("<")
        if dtype.kind + str(dtype.itemsize) not in PLY_TYPES:
            raise ValueError(f"PLY cannot retain attribute {name} with type {dtype}; use LAZ or COPC")
        fields.append((name, dtype))
    return fields


def write_instance_ply(source: Path, target: Path, dimensions):
    """Stream binary PLY with exact scalar attributes and checked 32-bit integers.

    PLY has no LAS ExtraBytes descriptors. Scaled measurements are stored as
    physical doubles; callers needing LAS no-data declarations must use LAZ/COPC.
    """
    with laspy.open(source) as reader:
        fields = _ply_fields(reader.header, dimensions)
        dtype = np.dtype(fields)
        lines = ["ply", "format binary_little_endian 1.0"]
        crs = crs_comment_value(reader.header)
        if crs:
            lines.append(f"comment crs: {crs}")
        lines.append(f"element vertex {reader.header.point_count}")
        lines.extend(f"property {PLY_TYPES[field.kind + str(field.itemsize)]} {name}" for name, field in fields)
        lines.append("end_header")
        count = 0
        with target.open("wb") as output:
            output.write(("\n".join(lines) + "\n").encode("ascii"))
            for chunk in reader.chunk_iterator(1_000_000):
                records = np.empty(len(chunk), dtype=dtype)
                for name, field in fields:
                    values = np.asarray(chunk.array[name] if field.kind in "iu" and name in chunk.array.dtype.names
                                        else chunk[name])
                    if len(values) and field.kind in "iu":
                        limits = np.iinfo(field)
                        if int(values.min()) < limits.min or int(values.max()) > limits.max:
                            raise ValueError(f"PLY attribute {name} exceeds {field.name} range "
                                             f"{limits.min}..{limits.max}; use LAZ or COPC")
                    records[name] = values
                records.tofile(output)
                count += len(records)
        if count != reader.header.point_count:
            raise RuntimeError(f"{target.name}: wrote {count} of {reader.header.point_count} points")


def convert_instance_tile(source: Path, output_format: str, dimensions) -> Path:
    """Convert a private staged LAZ; publication belongs to the collection caller."""
    if output_format == "laz":
        return source
    target = source.with_suffix("." + output_format)
    if output_format == "ply":
        write_instance_ply(source, target, dimensions)
    elif output_format == "copc.laz":
        from subsample_outputs import convert_laz_output_to_copc

        if not convert_laz_output_to_copc(source, target, source_metadata_file=source, preserve_extra_dims=True):
            raise RuntimeError(f"COPC conversion failed for instance tile {target.name}")
        with laspy.open(source) as before, laspy.open(target) as after:
            if before.header.point_count != after.header.point_count:
                raise RuntimeError(f"COPC point count differs for instance tile {target.name}")
    else:
        raise ValueError(f"Unsupported instance tile format {output_format}")
    source.unlink()
    return target
