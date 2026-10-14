"""Record the CRS of PLY outputs as a ``comment crs: <value>`` header line.

PLY has no CRS field. ``comment crs:`` is the one convention a GIS tool reads
automatically (QGIS via MDAL); other readers ignore comments. The value is
``EPSG:<code>`` when the CRS is identified, otherwise the standardized WKT on
one line. PDAL's PLY writer cannot write custom comments, so product PLYs are
annotated after writing; only the ASCII header changes, the payload is copied
byte for byte.

PDAL's PLY writer also lowercases every property name (``treeID`` becomes
``treeid``), which breaks exact lookups such as ForestMamba's
``pcd["treeID"]`` (3DT-2352). The same header rewrite restores the names of
the given application dimensions; property names do not affect the binary
layout. ``x``, ``y``, ``z`` and ``red``, ``green``, ``blue`` stay lowercase.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

from copc_metadata import laspy_laz_backend
from crs_records import crs_comment_value

MAX_HEADER_LINES = 200
MAX_LINE_BYTES = 8192
CONVENTIONAL_LOWERCASE = {"x", "y", "z", "red", "green", "blue"}


def property_case_map(names) -> dict:
    """``{lowercase: original}`` for dimension names whose case PDAL loses."""
    names_by_key = {}
    for name in names:
        key = name.lower()
        if key in names_by_key and names_by_key[key] != name:
            raise ValueError(f"PLY property names differ only in case: {names_by_key[key]}, {name}")
        names_by_key[key] = name
    return {key: name for key, name in names_by_key.items()
            if key not in CONVENTIONAL_LOWERCASE and key != name}


def _restore_property_case(line: bytes, mapping: dict) -> bytes:
    if not line.startswith(b"property "):
        return line
    body = line.rstrip(b"\r\n")
    head, _, name = body.rpartition(b" ")
    original = mapping.get(name.decode("ascii"))
    return line if original is None else head + b" " + original.encode("ascii") + line[len(body):]


def crs_comment_from_file(path: Path) -> Optional[str]:
    import laspy

    with laspy.open(str(path), laz_backend=laspy_laz_backend()) as reader:
        return crs_comment_value(reader.header)


def add_crs_comment_to_ply(path: Path, crs_value: Optional[str], restore_names=()) -> bool:
    """Insert or replace ``comment crs:`` after the ``format`` line and restore the
    case of ``restore_names`` properties; atomic rewrite of the header only."""
    mapping = property_case_map(restore_names)
    if not crs_value and not mapping:
        return False
    path = Path(path)
    temporary = path.with_name(f".{path.name}.crs")
    try:
        with path.open("rb") as source:
            lines = []
            while True:
                line = source.readline(MAX_LINE_BYTES)
                if not line or len(lines) >= MAX_HEADER_LINES or not line.endswith(b"\n"):
                    raise ValueError(f"{path.name}: invalid or oversized PLY header")
                lines.append(line)
                if line.rstrip(b"\r\n") == b"end_header":
                    break
            if lines[0].rstrip(b"\r\n") != b"ply" or not lines[1].startswith(b"format "):
                raise ValueError(f"{path.name}: not a PLY file")
            newline = b"\r\n" if lines[0].endswith(b"\r\n") else b"\n"
            kept = [_restore_property_case(line, mapping) for line in lines]
            if crs_value:
                kept = [line for line in kept if not line.lower().startswith(b"comment crs:")]
                kept.insert(2, b"comment crs: " + crs_value.encode("ascii") + newline)
            with temporary.open("wb") as output:
                output.writelines(kept)
                shutil.copyfileobj(source, output, 16 << 20)
        os.replace(temporary, path)
        return True
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
