"""Reversible scalar transport for LAS vector ExtraBytes dimensions.

Some COPC toolchains cannot carry LAS ExtraBytes entries whose data type has
two or three elements. SmartTile converts those records to scalar components,
keeps the raw values byte-for-byte, and records the original schema both in a
3Dtrees VLR/EVLR and a mandatory JSON sidecar.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import struct
import tempfile
import zlib
from collections import Counter
from contextlib import closing
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import laspy
import numpy as np
from laspy.vlrs.vlr import VLR

from point_cloud_metadata import (
    copy_single_source_header,
    extra_bytes_params_from_dimension_info,
    extra_bytes_attribute_equal,
    write_retained_evlrs,
)


VECTOR_SCHEMA_USER_ID = "3DTrees"
VECTOR_SCHEMA_RECORD_ID = 24001
VECTOR_SCHEMA_DESCRIPTION = "SmartTile vector ExtraBytes v1"
VECTOR_SCHEMA_MAGIC = b"STV1"
VECTOR_SCHEMA_SIDECAR_SUFFIX = ".vector-extra-bytes.json"
LAS14_START_OF_FIRST_EVLR_OFFSET = 235
LAS14_NUMBER_OF_EVLRS_OFFSET = 243


def _json_value(value):
    if value is None:
        return None
    return np.asarray(value).tolist()


def _dimension_layout(dim) -> Tuple[np.dtype, int]:
    dtype = np.dtype(dim.dtype)
    if dtype.subdtype is not None:
        base, shape = dtype.subdtype
        return np.dtype(base), int(np.prod(shape))
    count = int(getattr(dim, "num_elements", 1) or 1)
    return dtype, count


def vector_extra_dimensions(header) -> List[object]:
    return [
        dim
        for dim in header.point_format.extra_dimensions
        if _dimension_layout(dim)[1] > 1
    ]


def _component_name(original_name: str, index: int, used: set[str]) -> str:
    suffix = f"__{index}"
    candidate = f"{original_name}{suffix}"
    if len(candidate.encode("utf-8")) <= 32 and candidate not in used:
        used.add(candidate)
        return candidate

    digest = hashlib.sha256(f"{original_name}\0{index}".encode("utf-8")).hexdigest()[:8]
    suffix = f"_{index}_{digest}"
    max_prefix_bytes = 32 - len(suffix)
    prefix = original_name.encode("utf-8")[:max_prefix_bytes].decode("utf-8", errors="ignore")
    candidate = f"{prefix}{suffix}"
    if candidate in used:
        raise ValueError(f"Could not create unique scalar component name for {original_name}[{index}]")
    used.add(candidate)
    return candidate


def _component_description(description: str, index: int) -> str:
    suffix = f"[{index}]"
    prefix = description.encode("utf-8")[:32 - len(suffix)].decode("utf-8", errors="ignore")
    return prefix + suffix


def _schema_payload(schema: Dict[str, object]) -> bytes:
    raw = json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload = VECTOR_SCHEMA_MAGIC + zlib.compress(raw, level=9)
    if len(payload) > 65_535:
        raise ValueError(
            "Vector ExtraBytes schema is too large for an embedded LAS VLR; "
            "refusing a non-reversible conversion"
        )
    return payload


def _decode_schema_payload(payload: bytes) -> Dict[str, object]:
    if not payload.startswith(VECTOR_SCHEMA_MAGIC):
        raise ValueError("Unsupported SmartTile vector ExtraBytes schema payload")
    schema = json.loads(zlib.decompress(payload[len(VECTOR_SCHEMA_MAGIC) :]))
    if schema.get("schema") != "3dtrees.smarttile.vector-extra-bytes" or schema.get("version") != 1:
        raise ValueError("Unsupported SmartTile vector ExtraBytes schema version")
    return schema


def vector_schema_sidecar_path(point_cloud: Path) -> Path:
    return point_cloud.with_name(point_cloud.name + VECTOR_SCHEMA_SIDECAR_SUFFIX)


def write_vector_schema_sidecar(point_cloud: Path, schema: Dict[str, object]) -> Path:
    sidecar = vector_schema_sidecar_path(point_cloud)
    sidecar.write_text(
        json.dumps(schema, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return sidecar


def read_vector_schema(point_cloud: Path) -> Optional[Dict[str, object]]:
    with laspy.open(str(point_cloud), laz_backend=laspy.LazBackend.LazrsParallel) as reader:
        collections = [reader.header.vlrs]
        if reader.header.evlrs:
            collections.append(reader.header.evlrs)
        for collection in collections:
            for vlr in collection:
                if (
                    getattr(vlr, "user_id", "") == VECTOR_SCHEMA_USER_ID
                    and getattr(vlr, "record_id", None) == VECTOR_SCHEMA_RECORD_ID
                ):
                    if hasattr(vlr, "record_data_bytes"):
                        payload = bytes(vlr.record_data_bytes())
                    else:
                        payload = bytes(vlr.record_data)
                    return _decode_schema_payload(payload)
    return None


def _copy_common_record_fields(source, target) -> None:
    source_fields = getattr(source.array.dtype, "names", ()) or ()
    target_fields = set(getattr(target.array.dtype, "names", ()) or ())
    for field in source_fields:
        if field in target_fields:
            target.array[field] = source.array[field]


def scalarize_vector_extra_bytes(
    source: Path,
    output: Path,
    *,
    chunk_size: int = 5_000_000,
) -> Tuple[Path, Optional[Dict[str, object]]]:
    """Write a scalar-only LAZ transport file and its reversible schema."""
    source = Path(source)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with laspy.open(str(source), laz_backend=laspy.LazBackend.LazrsParallel) as reader:
        vector_dims = vector_extra_dimensions(reader.header)
        existing_schema = read_vector_schema(source) if not vector_dims else None
        if not vector_dims:
            return source, existing_schema

        source_header = reader.header
        output_header = copy_single_source_header(
            source_header,
            preserve_extra_dimensions=False,
        )
        # Reserve scalar names regardless of descriptor order.
        used = set(output_header.point_format.dimension_names)
        used.update(dim.name for dim in source_header.point_format.extra_dimensions
                    if _dimension_layout(dim)[1] == 1)
        schema_dimensions = []
        output_params = []
        component_names_by_original: Dict[str, List[str]] = {}

        for ordinal, dim in enumerate(source_header.point_format.extra_dimensions):
            params = extra_bytes_params_from_dimension_info(dim, header=source_header)
            base_dtype, count = _dimension_layout(dim)
            if count == 1:
                output_params.append(params)
                used.add(dim.name)
                continue

            component_names = [
                _component_name(dim.name, index, used)
                for index in range(count)
            ]
            component_names_by_original[dim.name] = component_names
            scales = getattr(params, "scales", None)
            offsets = getattr(params, "offsets", None)
            no_data = getattr(params, "no_data", None)
            for index, component_name in enumerate(component_names):
                output_params.append(
                    laspy.ExtraBytesParams(
                        name=component_name,
                        type=base_dtype,
                        description=_component_description(params.description or dim.name, index),
                        scales=None if scales is None else [np.asarray(scales).reshape(-1)[index]],
                        offsets=None if offsets is None else [np.asarray(offsets).reshape(-1)[index]],
                        no_data=None if no_data is None else [np.asarray(no_data).reshape(-1)[index]],
                    )
                )
            schema_dimensions.append(
                {
                    "ordinal": ordinal,
                    "original_name": dim.name,
                    "component_names": component_names,
                    "component_count": count,
                    "base_dtype": base_dtype.str,
                    "original_dtype": str(np.dtype(dim.dtype)),
                    "description": getattr(params, "description", "") or "",
                    "scales": _json_value(scales),
                    "offsets": _json_value(offsets),
                    "no_data": _json_value(no_data),
                }
            )

        schema = {
            "schema": "3dtrees.smarttile.vector-extra-bytes",
            "version": 1,
            "provenance": {
                "tool": "3dtrees_Smart_Tile",
                "operation": "automatic-vector-extra-bytes-scalarization",
                "source_name": source.name,
            },
            "dimensions": schema_dimensions,
        }
        output_header.add_extra_dims(output_params)
        output_header.vlrs.append(
            VLR(
                user_id=VECTOR_SCHEMA_USER_ID,
                record_id=VECTOR_SCHEMA_RECORD_ID,
                description=VECTOR_SCHEMA_DESCRIPTION,
                record_data=_schema_payload(schema),
            )
        )

        temp_output = output.with_name(f".{output.name}.smarttile-vector-part")
        if temp_output.exists():
            temp_output.unlink()
        try:
            with laspy.open(
                str(temp_output),
                mode="w",
                header=output_header,
                do_compress=output.name.lower().endswith(".laz"),
                laz_backend=laspy.LazBackend.LazrsParallel,
            ) as writer:
                for chunk in reader.chunk_iterator(chunk_size):
                    out_chunk = laspy.ScaleAwarePointRecord.zeros(len(chunk), header=output_header)
                    _copy_common_record_fields(chunk, out_chunk)
                    for original_name, component_names in component_names_by_original.items():
                        raw_values = chunk.array[original_name]
                        for index, component_name in enumerate(component_names):
                            out_chunk.array[component_name] = raw_values[:, index]
                    writer.write_points(out_chunk)
                write_retained_evlrs(writer, output_header)
            temp_output.replace(output)
            write_vector_schema_sidecar(output, schema)
            return output, schema
        except Exception:
            if temp_output.exists():
                temp_output.unlink()
            raise


def _evlr_record_bytes(payload: bytes) -> bytes:
    user = VECTOR_SCHEMA_USER_ID.encode("ascii")[:16].ljust(16, b"\0")
    desc = VECTOR_SCHEMA_DESCRIPTION.encode("ascii")[:32].ljust(32, b"\0")
    return struct.pack(
        "<H16sHQ32s",
        0,
        user,
        VECTOR_SCHEMA_RECORD_ID,
        len(payload),
        desc,
    ) + payload


def preserve_vector_schema(source: Path, output: Path) -> Tuple[bool, str]:
    """Copy embedded vector schema to a LAS 1.4 output EVLR and sidecar."""
    schema = read_vector_schema(source)
    if schema is None:
        return (True, "source has no vector ExtraBytes schema")
    existing = read_vector_schema(output)
    if existing is not None and _schema_payload(existing) != _schema_payload(schema):
        return (False, "output contains a different vector ExtraBytes schema")

    if existing is None:
        payload = _schema_payload(schema)
        try:
            with laspy.open(str(output), laz_backend=laspy.LazBackend.LazrsParallel) as reader:
                header = reader.header
                if str(header.version) != "1.4":
                    return (False, "embedded vector schema preservation requires LAS 1.4 output")
                existing_count = int(getattr(header, "number_of_evlrs", 0) or 0)
                existing_start = int(getattr(header, "start_of_first_evlr", 0) or 0)
            with open(output, "r+b") as handle:
                handle.seek(0, os.SEEK_END)
                append_start = handle.tell()
                handle.write(_evlr_record_bytes(payload))
                handle.seek(LAS14_START_OF_FIRST_EVLR_OFFSET)
                handle.write(struct.pack("<Q", existing_start or append_start))
                handle.seek(LAS14_NUMBER_OF_EVLRS_OFFSET)
                handle.write(struct.pack("<I", existing_count + 1))
        except Exception as exc:
            return (False, f"could not embed vector ExtraBytes schema: {exc}")

    try:
        if _schema_payload(read_vector_schema(output)) != _schema_payload(schema):
            return (False, "embedded vector ExtraBytes schema validation failed")
        write_vector_schema_sidecar(output, schema)
    except Exception as exc:
        return (False, f"could not validate vector ExtraBytes schema: {exc}")
    return (True, "vector ExtraBytes schema and sidecar preserved")


def _validated_vector_components(schema, headers):
    """Check the transport fields against the embedded logical descriptors."""
    components = []
    for vector in schema["dimensions"]:
        names = vector["component_names"]
        if len(names) != vector["component_count"] or len(names) not in (2, 3):
            raise ValueError(f"Invalid vector component mapping for {vector['original_name']}")
        for index, name in enumerate(names):
            if name in components:
                raise ValueError(f"Duplicate vector component mapping: {name}")
            components.append(name)
            for header in headers:
                if name not in set(header.point_format.extra_dimension_names):
                    raise ValueError(f"Missing vector component: {name}")
                dim = header.point_format.dimension_by_name(name)
                params = extra_bytes_params_from_dimension_info(dim, header=header)
                if np.dtype(params.type) != np.dtype(vector["base_dtype"]):
                    raise ValueError(f"Changed vector component dtype: {name}")
                for attribute in ("scales", "offsets", "no_data"):
                    declared = vector[attribute]
                    expected = None if declared is None else [declared[index]]
                    if not extra_bytes_attribute_equal(expected, getattr(params, attribute)):
                        raise ValueError(f"Changed vector component {attribute}: {name}")
    if not components:
        raise ValueError("Empty vector component mapping")
    return components


def _vector_record_keys(chunk, header, source_header, components):
    """Encode XYZ and raw vector components together, without point-order assumptions.

    COPC may reorder points or change coordinate offsets. Normalize coordinates
    to the source integer lattice; accept only floating-point representation
    roundoff, never a spatial matching tolerance. Component bytes remain exact.
    """
    dtype = np.dtype([(axis, "<i8") for axis in ("x", "y", "z")] +
                     [(f"v{i}", chunk.array.dtype.fields[name][0])
                      for i, name in enumerate(components)])
    rows = np.empty(len(chunk), dtype=dtype)
    for i, axis in enumerate(("X", "Y", "Z")):
        local = (np.asarray(chunk[axis], dtype=np.longdouble) * np.longdouble(header.scales[i]) +
                 (np.longdouble(header.offsets[i]) - np.longdouble(source_header.offsets[i])))
        scale = np.longdouble(source_header.scales[i])
        encoded = np.rint(local / scale)
        magnitude = max(1.0, abs(float(header.offsets[i])), abs(float(source_header.offsets[i])),
                        float(np.max(np.abs(local))) if len(local) else 0.0)
        roundoff = 8 * np.spacing(magnitude)
        if (np.any(~np.isfinite(encoded)) or
            np.any(np.abs(local - encoded * scale) > roundoff) or
            np.any(encoded < np.iinfo(np.int32).min) or
            np.any(encoded > np.iinfo(np.int32).max)):
            raise ValueError("Output coordinates do not match the source coordinate lattice")
        rows[axis.lower()] = encoded.astype(np.int64)
    for i, name in enumerate(components):
        rows[f"v{i}"] = chunk.array[name]
    return Counter(row.tobytes() for row in rows)


def validate_vector_conversion(source: Path, output: Path, *, chunk_size: int = 32_768) -> Tuple[bool, str]:
    """Compare every vector record exactly using a bounded, disk-backed multiset.

    The source is the scalar transport file. A row includes XYZ and all raw
    components, so swapped labels, missing duplicates and value changes fail,
    while COPC point reordering succeeds. SQLite stores full rows, not hashes.
    """
    try:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        schema = read_vector_schema(source)
        if schema is None:
            raise ValueError("Vector reconstruction schema is missing from the source")
        with laspy.open(source) as before, laspy.open(output) as after:
            if before.header.point_count != after.header.point_count:
                raise ValueError("Vector conversion changed the point count")
            components = _validated_vector_components(schema, (before.header, after.header))
            with tempfile.TemporaryDirectory(prefix="smarttile-vector-check-", dir=output.parent) as tmp:
                with closing(sqlite3.connect(Path(tmp) / "records.sqlite")) as db:
                    db.execute("PRAGMA cache_size=-16384")
                    db.execute("PRAGMA temp_store=FILE")
                    db.execute("CREATE TABLE records (value BLOB PRIMARY KEY, amount INTEGER NOT NULL) WITHOUT ROWID")
                    for reader, sign in ((before, 1), (after, -1)):
                        for chunk in reader.chunk_iterator(min(chunk_size, 32_768)):
                            keys = _vector_record_keys(chunk, reader.header, before.header, components)
                            db.executemany(
                                "INSERT INTO records VALUES (?, ?) ON CONFLICT(value) "
                                "DO UPDATE SET amount=amount+excluded.amount",
                                ((key, sign * count) for key, count in keys.items()),
                            )
                            db.commit()
                    if db.execute("SELECT 1 FROM records WHERE amount != 0 LIMIT 1").fetchone():
                        raise ValueError("Vector component values or their XYZ associations changed")
    except Exception as exc:
        return False, str(exc)
    return True, "Vector descriptors and all component values validated"


def reconstruct_vector_values(point_cloud: Path) -> Dict[str, np.ndarray]:
    """Reconstruct logical vector values from scalar components for validation."""
    schema = read_vector_schema(point_cloud)
    if schema is None:
        return {}
    las = laspy.read(point_cloud)
    reconstructed = {}
    for dim in schema["dimensions"]:
        reconstructed[dim["original_name"]] = np.column_stack(
            [np.asarray(las[name]) for name in dim["component_names"]]
        )
    return reconstructed
