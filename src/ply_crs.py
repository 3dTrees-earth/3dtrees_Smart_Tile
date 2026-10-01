"""Record the CRS of PLY outputs as a ``comment crs: <value>`` header line.

PLY has no CRS field. ``comment crs:`` is the one convention a GIS tool reads
automatically (QGIS via MDAL); other readers ignore comments. The value is
``EPSG:<code>`` when the CRS is identified, otherwise the standardized WKT on
one line. PDAL's PLY writer cannot write custom comments, so product PLYs are
annotated after writing; only the ASCII header changes, the payload is copied
byte for byte.
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


def crs_comment_from_file(path: Path) -> Optional[str]:
    import laspy

    with laspy.open(str(path), laz_backend=laspy_laz_backend()) as reader:
        return crs_comment_value(reader.header)


def add_crs_comment_to_ply(path: Path, crs_value: Optional[str]) -> bool:
    """Insert or replace ``comment crs:`` after the ``format`` line; atomic rewrite."""
    if not crs_value:
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
            kept = [line for line in lines if not line.lower().startswith(b"comment crs:")]
            kept.insert(2, b"comment crs: " + crs_value.encode("ascii") + newline)
            with temporary.open("wb") as output:
                output.writelines(kept)
                shutil.copyfileobj(source, output, 16 << 20)
        os.replace(temporary, path)
        return True
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
