"""Whether a point range of a LAS/LAZ file can be read with ``reader.seek``.

laspy 2.7 with lazrs 0.8.2 (both backends) misplaces ``seek(n)`` in LAZ files
with variable-size chunks, the layout of COPC and of PDAL/untwine output: the
records read after it are not records n, n + 1, ... Fixed-size chunks, as
LASzip and laspy write them, and uncompressed LAS seek exactly. Readers that
split a file into ranges read a variable-chunk file in one sequential range
instead.

The cause is in laz-rs 0.12/0.13, which lazrs 0.8.2 builds on: its seek
takes the point's position in its chunk as ``n % chunk point count`` (and
``n % chunk_size`` in the parallel decompressor), true only for fixed-size
chunks. laz-rs 0.14.0 ("Fix seeking in files using variable-size chunks")
uses ``n - first point of the chunk``. Once a lazrs release built on laz
>= 0.14 is the minimum, this guard can go.
"""
from __future__ import annotations

import laspy


def seekable(path) -> bool:
    """True when ``reader.seek(n)`` lands on record n of ``path``."""
    with laspy.open(path) as reader:
        for vlr in reader.header.vlrs:
            if type(vlr).__name__ == "LasZipVlr":
                import lazrs
                return not lazrs.LazVlr(vlr.record_data_bytes()).uses_variable_size_chunks()
    return True
