"""Spatially ordered access to an original file for enrichment queries.

Region search trees (spatial_query_cache) are reused only while queries stay in
one area. Originals are often stored in acquisition order: on dataset 645 each
32k-point batch spans about 120 m x 130 m and touches ~450 4 m regions, so the
trees were rebuilt for a few dozen points each. Here an original is first
spilled into XY blocks of whole cache regions, in parallel point ranges, and
then queried block by block. Each point keeps its index so results can be
written back in the original point order.

Blocks are sized from the header extent, so stray far points or dense scanner
positions can concentrate most points in one block (483: one 4 m region with
about 99 M points). A block above BLOCK_POINT_CAP is therefore streamed into
its 2 m query cells, ordered by cache region, and each cell is streamed in
batches without loading it whole: memory no longer depends on density.
Results do not depend on the layout: queries group points by 2 m cell, and
every point is written back by its index.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import math
import multiprocessing
import os
from pathlib import Path

import laspy
import numpy as np

from bounded_point_index import CELL_SIZE, MAX_BATCH_POINTS, coordinates
from spatial_query_cache import REGION_SIZE

# Point index within the file plus the raw LAS integer coordinates.
SPILL = np.dtype([("i", "<u4"), ("X", "<i4"), ("Y", "<i4"), ("Z", "<i4")])
READ_POINTS = 1_000_000
BLOCK_POINTS = 4_000_000
# The parent holds one block at a time: about 60 B per point while sorting.
BLOCK_POINT_CAP = 8_000_000
MAX_BLOCK = 64.0


def block_size(point_count, extent):
    """Block edge holding about BLOCK_POINTS points, in whole cache regions."""
    area = max((extent[1] - extent[0]) * (extent[3] - extent[2]), 1e-9)
    edge = math.sqrt(BLOCK_POINTS * area / max(point_count, 1))
    return float(min(MAX_BLOCK, max(REGION_SIZE, math.floor(edge / REGION_SIZE) * REGION_SIZE)))


def _spill(task):
    task_id, path, start, count, origin, size, directory, label_stores = task
    os.environ["RAYON_NUM_THREADS"] = "1"
    stores = {name: np.load(store, mmap_mode="r+") for name, store in label_stores.items()}
    with laspy.open(path) as reader:
        header = reader.header
        reader.seek(start)
        done = 0
        while done < count:
            points = reader.read_points(min(READ_POINTS, count - done))
            if not len(points):
                raise ValueError(f"{path}: expected {count} points from {start}, read {done}")
            records = np.empty(len(points), dtype=SPILL)
            records["i"] = np.arange(start + done, start + done + len(points), dtype=np.uint32)
            for name, store in stores.items():
                store[start + done:start + done + len(points)] = np.asarray(points[name])
            for axis in ("X", "Y", "Z"):
                records[axis] = np.asarray(points[axis])
            done += len(points)
            blocks = np.floor(coordinates(points, header, origin)[:, :2] / size).astype(np.int64)
            lo = blocks.min(axis=0)
            span = int(blocks[:, 1].max() - lo[1]) + 1
            linear = (blocks[:, 0] - lo[0]) * span + (blocks[:, 1] - lo[1])
            order = np.argsort(linear, kind="stable")
            linear = linear[order]
            starts = np.r_[0, np.flatnonzero(np.diff(linear)) + 1]
            for first, end in zip(starts, np.r_[starts[1:], len(order)]):
                bx, by = blocks[order[first]]
                block = Path(directory) / f"{bx}_{by}"
                block.mkdir(parents=True, exist_ok=True)
                with open(block / f"{task_id:06d}.bin", "ab") as handle:
                    handle.write(records[order[first:end]].tobytes())


def spill_original(path, origin, directory, workers, labels=(), label_dir=None):
    """Write every point of `path` to its XY block under `directory`; return the block edge.

    ``labels`` names dimensions of the original whose values are also stored,
    by point index, as ``label_dir/<name>.npy`` (read with ``np.load(..., mmap_mode="r")``).
    """
    with laspy.open(path) as reader:
        header = reader.header
        count = int(header.point_count)
        extent = (header.x_min, header.x_max, header.y_min, header.y_max)
        label_types = {name: header.point_format.dimension_by_name(name).dtype for name in labels}
    if count >= 2**32:
        raise ValueError(f"{path}: more than 2^32 points per original are not supported")
    size = block_size(count, extent)
    stores = {}
    for name, dtype in label_types.items():
        stores[name] = str(Path(label_dir) / f"{name}.npy")
        np.lib.format.open_memmap(stores[name], mode="w+", dtype=dtype, shape=(count,)).flush()
    if not count:
        return size
    chunk = max(READ_POINTS, math.ceil(count / max(1, 2 * workers)))
    tasks = [(index, str(path), start, min(chunk, count - start), tuple(origin), size, str(directory), stores)
             for index, start in enumerate(range(0, count, chunk))]
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(tasks))),
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        list(pool.map(_spill, tasks))
    return size


def _block_key(path):
    bx, by = path.name.split("_")
    return int(bx), int(by)


def _block_point_count(files):
    return sum(path.stat().st_size for path in files) // SPILL.itemsize


def _split_into_cells(block, files, header, origin):
    """Stream an oversized block into one file per 2 m query cell.

    Spill files are in ascending point-index order and are read in that
    order, so each cell file is too. Cells are returned grouped by 4 m cache
    region, so consecutive cells reuse the region's search trees.
    """
    cells = block / "cells"
    cells.mkdir()
    for path in files:
        with open(path, "rb") as handle:
            while len(records := np.fromfile(handle, dtype=SPILL, count=READ_POINTS)):
                keys = np.floor(coordinates(records, header, origin)[:, :2] / CELL_SIZE).astype(np.int64)
                order = np.lexsort((records["i"], keys[:, 1], keys[:, 0]))
                keys, records = keys[order], records[order]
                starts = np.flatnonzero(np.r_[True, np.any(keys[1:] != keys[:-1], axis=1)])
                for first, end in zip(starts, np.r_[starts[1:], len(records)]):
                    with open(cells / f"{keys[first, 0]}_{keys[first, 1]}.bin", "ab") as out:
                        out.write(records[first:end].tobytes())
        path.unlink()
    per_region = int(round(REGION_SIZE / CELL_SIZE))

    def key(path):
        cx, cy = (int(v) for v in path.stem.split("_"))
        return cx // per_region, cy // per_region, cx, cy
    return sorted(cells.iterdir(), key=key)


def _sorted_batches(files, header, origin):
    records = np.concatenate([np.fromfile(path, dtype=SPILL) for path in files])
    xyz = coordinates(records, header, origin)
    cells = np.floor(xyz[:, :2] / CELL_SIZE).astype(np.int64)
    order = np.lexsort((records["i"], cells[:, 1], cells[:, 0]))
    records, xyz = records[order], xyz[order]
    for start in range(0, len(records), MAX_BATCH_POINTS):
        end = start + MAX_BATCH_POINTS
        yield records["i"][start:end].astype(np.int64), xyz[start:end]


def block_batches(directory, header, origin, point_cap=None):
    """Yield (point indices, local XYZ) batches block by block.

    Within a block, points are ordered by 2 m query cell and then by point
    index, so batches are spatially compact and their order is deterministic.
    A block above ``point_cap`` (default BLOCK_POINT_CAP) is streamed cell by
    cell in the same order (cell, then point index), never loaded whole.
    """
    point_cap = BLOCK_POINT_CAP if point_cap is None else point_cap
    if not Path(directory).exists():  # an original without points spills nothing
        return
    for block in sorted(Path(directory).iterdir(), key=_block_key):
        files = sorted(path for path in block.iterdir() if path.is_file())
        if _block_point_count(files) <= point_cap:
            yield from _sorted_batches(files, header, origin)
        else:
            for cell in _split_into_cells(block, files, header, origin):
                with open(cell, "rb") as handle:
                    while len(records := np.fromfile(handle, dtype=SPILL, count=MAX_BATCH_POINTS)):
                        yield records["i"].astype(np.int64), coordinates(records, header, origin)
                cell.unlink()
            (block / "cells").rmdir()
        for path in files:
            path.unlink(missing_ok=True)
        block.rmdir()
