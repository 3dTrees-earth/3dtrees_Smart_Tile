"""Bounded scratch ordering and conservative tree occupancy for ownership queries."""
from collections import defaultdict
import sqlite3

import numpy as np

from bounded_point_index import CELL_SIZE, distance_limit
from spatial_query_cache import REGION_SIZE


class TreeOccupancy:
    """Conservative positive-tree voxel presence; budget exhaustion disables pruning.

    This summarizes the immutable, retained source index. Background never marks
    a voxel occupied. Core eligibility is checked later by the existing matcher.
    No absent-voxel answer is used if the complete index could not be summarized.
    """
    MAX_VOXELS = 200_000

    def __init__(self, index, dimension):
        occupied = set()
        self.columns = None
        self.skipped_groups = 0
        self.skipped_points = 0
        field = f'value_{list(index.dimensions).index(dimension)}'
        dtype = index._storage_dtype()
        for tile, blob in index.db.execute('SELECT tile,data FROM batches ORDER BY id'):
            records = np.frombuffer(blob, dtype=dtype)
            cells = np.unique(np.floor(records['xyz'][records[field] > 0] / CELL_SIZE).astype(np.int64), axis=0)
            for x, y, z in cells:
                occupied.add((tile, int(x), int(y), int(z)))
                if len(occupied) > self.MAX_VOXELS:
                    self.voxels = len(occupied)
                    return
        self.voxels = len(occupied)
        columns = defaultdict(list)
        for tile, x, y, z in occupied:
            columns[tile, x, y].append(z)
        self.columns = {key: np.sort(values) for key, values in columns.items()}

    def may_contain(self, tile, xyz, radius):
        if self.columns is None:
            return True
        # Include both sides of exact voxel boundaries and the caller's original
        # numerical allowance. False positives only cost work; false negatives
        # would change ownership and are forbidden.
        lo = np.floor(np.nextafter(xyz.min(axis=0) - radius, -np.inf) / CELL_SIZE).astype(np.int64)
        hi = np.floor(np.nextafter(xyz.max(axis=0) + radius, np.inf) / CELL_SIZE).astype(np.int64)
        for x in range(int(lo[0]), int(hi[0]) + 1):
            for y in range(int(lo[1]), int(hi[1]) + 1):
                zs = self.columns.get((tile, x, y))
                if zs is not None:
                    position = np.searchsorted(zs, lo[2])
                    if position < len(zs) and zs[position] <= hi[2]:
                        return True
        self.skipped_groups += 1
        self.skipped_points += len(xyz)
        return False

    def report(self):
        return dict(enabled=self.columns is not None, voxels=self.voxels,
                    max_voxels=self.MAX_VOXELS, skipped_groups=self.skipped_groups,
                    skipped_points=self.skipped_points)


class OwnershipQuerySpool:
    """Order original bounded query groups by region without changing each group.

    XYZ, source row positions and numerical allowances survive ordering. The
    spool contains only eligible query points; prediction attributes stay in
    the original files. All SQL sort/cache storage is bounded or disk-backed.
    """
    dtype = np.dtype([('xyz', np.float64, (3,)), ('position', np.int64)])

    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute('PRAGMA cache_size=-16384')
        self.db.execute('PRAGMA temp_store=FILE')
        self.db.execute('PRAGMA journal_mode=OFF')
        self.db.execute('PRAGMA synchronous=OFF')
        self.db.execute('CREATE TABLE queries (id INTEGER PRIMARY KEY, x INTEGER, y INTEGER, radius REAL, data BLOB)')
        self.points = 0
        self.blocks = 0

    def add(self, xyz, positions, radius):
        cell = np.floor(xyz[0, :2] / REGION_SIZE).astype(np.int64)
        values = np.empty(len(xyz), self.dtype)
        values['xyz'], values['position'] = xyz, positions
        self.db.execute('INSERT INTO queries(x,y,radius,data) VALUES(?,?,?,?)',
                        (int(cell[0]), int(cell[1]), distance_limit(xyz, radius), values.tobytes()))
        self.points += len(xyz)

    def groups(self):
        self.db.execute('CREATE INDEX query_region ON queries(x,y,id)')
        self.db.commit()
        for x, y, radius in self.db.execute('SELECT x,y,MAX(radius) FROM queries GROUP BY x,y ORDER BY x,y'):
            self.blocks += 1
            for (blob,) in self.db.execute('SELECT data FROM queries WHERE x=? AND y=? ORDER BY id', (x, y)):
                values = np.frombuffer(blob, self.dtype)
                yield values['position'], values['xyz'], radius

    def close(self):
        self.db.close()
