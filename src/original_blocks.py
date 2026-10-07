"""Spatially ordered access to an original file for enrichment queries.

Region search trees (spatial_query_cache) are reused only while queries stay in
one area. Originals are often stored in acquisition order: on dataset 645 each
32k-point batch spans about 120 m x 130 m and touches ~450 4 m regions, so the
trees were rebuilt for a few dozen points each. Here an original is first
spilled into XY blocks of whole cache regions, in parallel point ranges, and
then queried block by block. Each point keeps its index so results can be
written back in the original point order.
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
MAX_BLOCK = 64.0


def block_size(point_count, extent):
    """Block edge holding about BLOCK_POINTS points, in whole cache regions."""
    area = max((extent[1] - extent[0]) * (extent[3] - extent[2]), 1e-9)
    edge = math.sqrt(BLOCK_POINTS * area / max(point_count, 1))
    return float(min(MAX_BLOCK, max(REGION_SIZE, math.floor(edge / REGION_SIZE) * REGION_SIZE)))


def _spill(task):
    task_id, path, start, count, origin, size, directory = task
    os.environ["RAYON_NUM_THREADS"] = "1"
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


def spill_original(path, origin, directory, workers):
    """Write every point of `path` to its XY block under `directory`; return the block edge."""
    with laspy.open(path) as reader:
        header = reader.header
        count = int(header.point_count)
        extent = (header.x_min, header.x_max, header.y_min, header.y_max)
    if count >= 2**32:
        raise ValueError(f"{path}: more than 2^32 points per original are not supported")
    size = block_size(count, extent)
    if not count:
        return size
    chunk = max(READ_POINTS, math.ceil(count / max(1, 2 * workers)))
    tasks = [(index, str(path), start, min(chunk, count - start), tuple(origin), size, str(directory))
             for index, start in enumerate(range(0, count, chunk))]
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(tasks))),
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        list(pool.map(_spill, tasks))
    return size


def _block_key(path):
    bx, by = path.name.split("_")
    return int(bx), int(by)


def block_batches(directory, header, origin):
    """Yield (point indices, local XYZ) batches block by block.

    Within a block, points are ordered by 2 m query cell and then by point
    index, so batches are spatially compact and their order is deterministic.
    """
    if not Path(directory).exists():  # an original without points spills nothing
        return
    for block in sorted(Path(directory).iterdir(), key=_block_key):
        files = sorted(block.iterdir())
        records = np.concatenate([np.fromfile(path, dtype=SPILL) for path in files])
        xyz = coordinates(records, header, origin)
        cells = np.floor(xyz[:, :2] / CELL_SIZE).astype(np.int64)
        order = np.lexsort((records["i"], cells[:, 1], cells[:, 0]))
        records, xyz = records[order], xyz[order]
        for start in range(0, len(records), MAX_BATCH_POINTS):
            end = start + MAX_BATCH_POINTS
            yield records["i"][start:end].astype(np.int64), xyz[start:end]
        for path in files:
            path.unlink()
        block.rmdir()
