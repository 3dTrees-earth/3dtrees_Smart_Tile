"""Positive-tree coverage answered block by block instead of per 2 m query group.

``PointIndex.covered`` answers one 2 m XY group at a time from a region tree
it reads from SQLite. On dense TLS the regions do not fit the per-process
cache, so every group re-reads its region and its neighbours' blobs (orphan
claims on 2585: 4,500 of 4,660 CPU seconds). Here the query points are first
spilled to disk in XY strips, then each strip is answered in ``BLOCK_SIZE``
blocks: one index read and one tree per block for all of its points.

The answer is the same as ``covered``: a point is covered when some positive
indexed point lies within the distance bound that ``covered`` would use for
it, ``radius`` plus 8 ULPs of its original 2 m group (``point_limits``).
Distances come from the same coordinates, and coverage is an OR over points,
so neither the tree nor the grouping can change it.
"""
from __future__ import annotations

import multiprocessing
import sqlite3
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from bounded_point_index import PointIndex, distance_limit, spatial_batches

BLOCK_SIZE = 8.0
STRIP_SIZE = 32.0
SPILL_DTYPE = np.dtype([('xyz', np.float64, (3,)), ('limit', np.float64), ('tile', np.int64), ('label', np.int64)])


def point_limits(xyz, radius):
    """The distance bound ``PointIndex.covered(xyz, radius)`` applies to each point."""
    limits = np.empty(len(xyz), dtype=np.float64)
    for group in spatial_batches(xyz):
        limits[group] = distance_limit(xyz[group], radius)
    return limits


class CoverageSpill:
    """Query points of one pass (with their tile and label), spilled to one file per XY strip."""

    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.strips = {}

    def adopt(self, strips):
        """Take over strip files written by another spill (for example in a worker process)."""
        for strip, path in strips.items():
            self.strips[(len(self.strips), strip)] = path

    def add(self, tile, labels, xyz, limits):
        records = np.empty(len(xyz), dtype=SPILL_DTYPE)
        records['xyz'], records['limit'], records['tile'], records['label'] = xyz, limits, tile, labels
        strips = np.floor(xyz[:, 0] / STRIP_SIZE).astype(np.int64)
        for strip in np.unique(strips):
            path = self.folder / f'strip_{strip}.bin'
            with open(path, 'ab') as stream:
                stream.write(records[strips == strip].tobytes())
            self.strips[int(strip)] = path

    def uncovered(self, index_path, dimensions, positive_dimension, *, workers=1):
        """Yield ``(tiles, labels, xyz)`` of the spilled points without positive support, strip by strip."""
        jobs = [(str(path), str(index_path), dimensions, positive_dimension)
                for _, path in sorted(self.strips.items())]
        if workers > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(max_workers=min(workers, len(jobs)),
                                     mp_context=multiprocessing.get_context('spawn')) as pool:
                for path, hits in zip((job[0] for job in jobs), pool.map(_covered_strip, *zip(*jobs))):
                    yield _missing(path, hits)
        else:
            for job in jobs:
                yield _missing(job[0], _covered_strip(*job))


def _missing(path, hits):
    records = np.fromfile(path, dtype=SPILL_DTYPE)[~hits]
    Path(path).unlink()
    return records['tile'], records['label'], records['xyz']


def _covered_strip(spill_path, index_path, dimensions, positive_dimension):
    records = np.fromfile(spill_path, dtype=SPILL_DTYPE)
    xyz = records['xyz']
    blocks = np.floor(xyz[:, :2] / BLOCK_SIZE).astype(np.int64)
    order = np.lexsort((blocks[:, 1], blocks[:, 0]))
    boundaries = np.flatnonzero(np.any(np.diff(blocks[order], axis=0), axis=1)) + 1
    hits = np.zeros(len(records), dtype=bool)
    with PointIndex(index_path, dimensions, query_workers=1, read_only=True) as index:
        value = f'value_{list(dimensions).index(positive_dimension)}'
        dtype = index._storage_dtype()
        for group in np.split(order, boundaries):
            if not len(group):
                continue
            points, limits = xyz[group], records['limit'][group]
            halo = limits.max()
            lo, hi = points.min(axis=0) - halo, points.max(axis=0) + halo
            data = _positive_points(index.db, dtype, value, lo, hi)
            if not len(data):
                continue
            distances, _ = cKDTree(data).query(points, distance_upper_bound=np.nextafter(halo, np.inf))
            hits[group] = distances <= limits
    return hits


def _positive_points(db, dtype, value, lo, hi):
    """XYZ of positive indexed points inside the box [lo, hi] (all tiles)."""
    rows = db.execute('SELECT b.data FROM bounds r JOIN batches b ON b.id=r.id '
                      'WHERE r.x1>=? AND r.x0<=? AND r.y1>=? AND r.y0<=? AND r.z1>=? AND r.z0<=?',
                      (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])).fetchall()
    if not rows:
        return np.empty((0, 3))
    data = np.frombuffer(b''.join(row[0] for row in rows), dtype=dtype)
    xyz = data['xyz']
    keep = (data[value] > 0) & np.all((xyz >= lo) & (xyz <= hi), axis=1)
    return xyz[keep]
