"""Process-wide, memory-bounded cache of immutable XY-region search trees.

SQLite remains authoritative. A region too large for the cache is cached per
2 m query cell instead; only a cell too large as well uses the original bounded
query path. Inserts invalidate an index's cached regions. No labels are
reconciled here.
"""
from collections import OrderedDict
import os

import numpy as np
from scipy.spatial import cKDTree

from worker_budget import memory_limit_bytes, spatial_query_worker_count

REGION_SIZE = 4.0
# Query groups are 2 m XY cells (bounded_point_index.CELL_SIZE); a region holds 2 x 2 of them.
SUBCELL_SIZE = 2.0
# Upper bound for one cKDTree point: contiguous XYZ copy, index and nodes.
TREE_BYTES_PER_POINT = 64


def default_cache_bytes():
    requested = float(os.environ.get('SMARTTILE_SPATIAL_CACHE_MB', '512'))
    if not np.isfinite(requested) or requested < 0:
        raise ValueError('SMARTTILE_SPATIAL_CACHE_MB must be finite and nonnegative')
    limit = memory_limit_bytes()
    # Reserve at most a third of the allocation across parent + remap workers
    # (1919 at its 2.5 GB grant: a quarter sent the densest 2 m cells to disk).
    # SMARTTILE_WORKERS: the run's worker count after its memory cap (run.py).
    workers = int(os.environ.get('SMARTTILE_WORKERS') or spatial_query_worker_count(os.cpu_count() or 1))
    processes = 1 + workers
    safe = limit // (3 * processes) if limit else 64 * 1024**2
    return min(int(requested * 1024**2), safe)


class SpatialQueryCache:
    def __init__(self, max_bytes):
        self.max_bytes = max(0, int(max_bytes))
        self.bytes_used = 0
        self.hits = self.builds = self.cell_builds = self.fallbacks = 0
        self.peak_bytes = 0
        self.entries = OrderedDict()

    def discard(self, owner):
        for key in list(self.entries):
            if key[0] is owner:
                _, size = self.entries.pop(key)
                self.bytes_used -= size

    def get(self, key):
        if key not in self.entries:
            return None
        self.hits += 1
        value, _ = self.entries[key]
        self.entries.move_to_end(key)
        return value

    def stats(self):
        return {"max_bytes": self.max_bytes, "peak_bytes": self.peak_bytes, "hits": self.hits,
                "region_builds": self.builds, "cell_builds": self.cell_builds, "disk_fallbacks": self.fallbacks}

    def reserve(self, size, *, transient=False):
        # A kept entry may take half the cache: the other half holds the entry
        # of another index queried for the same group, and the active arrays.
        # A transient read may use the whole budget; it is freed before the tree.
        if size > (self.max_bytes if transient else self.max_bytes // 2):
            return False
        while self.bytes_used + size > self.max_bytes:
            _, (_, removed) = self.entries.popitem(last=False)
            self.bytes_used -= removed
        return True

    def put(self, key, value, size):
        self.entries[key] = (value, size)
        self.bytes_used += size
        self.peak_bytes = max(self.peak_bytes, self.bytes_used)


_shared_cache = None


def shared_cache():
    global _shared_cache
    if _shared_cache is None:
        _shared_cache = SpatialQueryCache(default_cache_bytes())
    return _shared_cache


def query_cells(xyz):
    """(4 m region, 2 m query cell) of a query group, from its first point."""
    return (tuple(np.floor(xyz[0, :2] / REGION_SIZE).astype(np.int64)),
            tuple(np.floor(xyz[0, :2] / SUBCELL_SIZE).astype(np.int64)))


def region_entry(index, tile, cell, radius, positive_dimension, bounds, *,
                 selection_key=None, select_points=None, matching=None, subcell=None):
    """The search tree and points of ``tile`` in one 4 m region with a ``radius`` halo.

    When the region does not fit the cache, the 2 m query cell ``subcell``
    with the same halo is cached instead: it holds every candidate of that
    cell's queries, and ties go to the lowest source index either way, so
    results are identical. None requests the bounded disk path.
    """
    options = dict(selection_key=selection_key, select_points=select_points, matching=matching)
    entry = _area_entry(index, tile, cell, REGION_SIZE, radius, positive_dimension, bounds, **options)
    if entry is None and subcell is not None:
        entry = _area_entry(index, tile, subcell, SUBCELL_SIZE, radius, positive_dimension, bounds, **options)
    if entry is None:
        index.query_cache.fallbacks += 1
    return entry


def _area_entry(index, tile, cell, size, radius, positive_dimension, bounds, *,
                selection_key=None, select_points=None, matching=None):
    bound_key = None if bounds is None else tuple(np.asarray(bounds).ravel())
    key = (index._cache_owner, tile, size, *cell, radius, positive_dimension, bound_key,
           repr(index._storage_dtype().descr), selection_key, matching)
    cached = index.query_cache.get(key)
    if cached is not None:
        return cached
    lo = np.asarray(cell) * size - radius
    hi = (np.asarray(cell) + 1) * size + radius
    clause = (' FROM bounds r JOIN batches b ON b.id=r.id '
              'WHERE b.tile=? AND r.x1>=? AND r.x0<=? AND r.y1>=? AND r.y0<=?')
    args = (int(tile), lo[0], hi[0], lo[1], hi[1])
    total = index.db.execute('SELECT COALESCE(SUM(length(b.data)),0)' + clause, args).fetchone()[0]
    dtype = index._storage_dtype()
    # Reserve the read first: the joined blobs, the selected copy and one blob
    # in flight. Blobs touching the halo hold several times the points the
    # area keeps (a 2 m cell reads its 8 neighbors' blobs), so the tree is
    # reserved separately, at the kept size.
    if not index.query_cache.reserve(total * 2 + 1024**2, transient=True):
        return None
    # Index batches are small (one 2 m cell of one insert), so a region spans
    # hundreds of blobs: copy them into one buffer and filter once, in id order.
    buffer = bytearray(total)
    offset = 0
    for (blob,) in index.db.execute('SELECT b.data' + clause + ' ORDER BY b.id', args):
        buffer[offset:offset + len(blob)] = blob
        offset += len(blob)
    assert offset == total, 'immutable index changed during a region read'
    data = np.frombuffer(buffer, dtype=dtype)
    x, y = data['xyz'][:, 0], data['xyz'][:, 1]
    mask = (x >= lo[0]) & (x <= hi[0]) & (y >= lo[1]) & (y <= hi[1])
    if positive_dimension is not None:
        mask &= data[f'value_{list(index.dimensions).index(positive_dimension)}'] > 0
    if matching is not None:
        dimension, label = matching
        mask &= data[f'value_{list(index.dimensions).index(dimension)}'] == label
    if bounds is not None:
        from bounded_point_index import inside_xy
        mask &= inside_xy(data['xyz'], bounds)
    if select_points is not None:
        mask &= select_points(data['xyz'])
    payload = data[mask]
    del buffer, data
    if not index.query_cache.reserve(payload.nbytes + len(payload) * TREE_BYTES_PER_POINT + 1024**2):
        return None
    entry = (cKDTree(payload['xyz']) if len(payload) else None, payload)
    if size == REGION_SIZE:
        index.query_cache.builds += 1
    else:
        index.query_cache.cell_builds += 1
    # Never above the reserved transient, so the cache stays within its budget.
    index.query_cache.put(key, entry, payload.nbytes + len(payload) * TREE_BYTES_PER_POINT + 1024**2)
    return entry


def covered_region(index, xyz, radius, positive_dimension):
    """Positive-tree support for one XY group, without labels or tie resolution.

    Reuse the same bounded region trees as nearest queries. None requests the
    bounded disk fallback when a region exceeds the cache budget.
    """
    cell, subcell = query_cells(xyz)
    from bounded_point_index import distance_limit
    limit = distance_limit(xyz, radius)
    found = np.zeros(len(xyz), dtype=bool)
    for tile in sorted(index._cache_tiles):
        entry = region_entry(index, tile, cell, limit, positive_dimension, None, subcell=subcell)
        if entry is None:
            return None
        tree, _ = entry
        if tree is None:
            continue
        remaining = np.flatnonzero(~found)
        distances, _ = index.query_tree(
            tree, xyz[remaining], distance_upper_bound=np.nextafter(limit, np.inf))
        found[remaining] = distances <= limit
        if np.all(found):
            break
    return found


def nearest_region(index, xyz, radius, *, tile=None, before_tile=None, overlaps=None,
                   positive_dimension=None, matching=None):
    """Return exact nearest results for one XY region, or request disk fallback."""
    from bounded_point_index import distance_limit, distance_roundoff, inside_xy
    cell, subcell = query_cells(xyz)
    limit = distance_limit(xyz, radius)
    dists = np.full(len(xyz), np.inf)
    refs = np.full((len(xyz), 2), -1, dtype=np.int64)
    values = {name: np.zeros(len(xyz), dtype=dtype) for name, dtype in index.dimensions.items()}
    numerical_ties = np.zeros(len(xyz), dtype=bool)
    tiles = [tile] if tile is not None else sorted(index._cache_tiles)
    for source in tiles:
        if before_tile is not None and source >= before_tile:
            continue
        if overlaps is not None and source not in overlaps:
            continue
        bounds = None if overlaps is None else overlaps[source]
        allowed = np.ones(len(xyz), dtype=bool) if bounds is None else inside_xy(xyz, bounds)
        if not np.any(allowed):
            continue
        entry = region_entry(index, source, cell, limit, positive_dimension, bounds,
                             matching=matching, subcell=subcell)
        if entry is None:
            return None
        tree, data = entry
        if tree is None:
            continue
        d, nearest, numerical = index.nearest_in_tree(tree, xyz, data['indices'], limit)
        found = np.isfinite(d) & (d <= limit) & allowed
        safe = np.minimum(nearest, len(data) - 1)
        better_ref = ((refs[:, 0] < 0) | (source < refs[:, 0]) |
                      ((source == refs[:, 0]) & (data['indices'][safe] < refs[:, 1])))
        roundoff = distance_roundoff(xyz)
        tied = np.abs(np.subtract(d, dists, out=np.full_like(d, np.inf), where=np.isfinite(d) & np.isfinite(dists))) <= roundoff
        numerical_ties |= numerical | (tied & (d != dists))
        take = found & ((d < dists - roundoff) | (tied & better_ref))
        dists[take] = d[take]
        refs[take, 0] = source
        refs[take, 1] = data['indices'][nearest[take]]
        for i, name in enumerate(values):
            values[name][take] = data[f'value_{i}'][nearest[take]]
    index.resolve_numerical_ties(xyz, radius, numerical_ties, dists, values, refs,
                                 tile=tile, before_tile=before_tile, overlaps=overlaps,
                                 positive_dimension=positive_dimension, matching=matching)
    return dists, values, refs
