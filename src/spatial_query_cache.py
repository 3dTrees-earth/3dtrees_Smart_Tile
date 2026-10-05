"""Process-wide, memory-bounded cache of immutable XY-region search trees.

SQLite remains authoritative. Oversized regions use the original bounded query
path; inserts invalidate an index's cached regions. No labels are reconciled here.
"""
from collections import OrderedDict
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from worker_budget import spatial_query_worker_count

REGION_SIZE = 4.0
# Upper bound for one cKDTree point: contiguous XYZ copy, index and nodes.
TREE_BYTES_PER_POINT = 64


def default_cache_bytes():
    requested = float(os.environ.get('SMARTTILE_SPATIAL_CACHE_MB', '512'))
    if not np.isfinite(requested) or requested < 0:
        raise ValueError('SMARTTILE_SPATIAL_CACHE_MB must be finite and nonnegative')
    limits = []
    root = Path('/sys/fs/cgroup')
    groups = {root}
    try:
        for line in Path('/proc/self/cgroup').read_text().splitlines():
            if line.startswith('0::') and '..' not in Path(line[3:]).parts:
                group = root / line[3:].lstrip('/')
                groups.update([group, *[p for p in group.parents if p == root or root in p.parents]])
    except OSError:
        pass
    for path in [*(p / 'memory.max' for p in groups), root / 'memory/memory.limit_in_bytes']:
        try:
            value = int(path.read_text())
            if value > 0:
                limits.append(value)
        except (OSError, ValueError):
            pass
    try:
        limits.append(os.sysconf('SC_PHYS_PAGES') * os.sysconf('SC_PAGE_SIZE'))
    except (ValueError, OSError):
        pass
    # Reserve at most a quarter of the allocation across parent + remap workers.
    processes = 1 + spatial_query_worker_count(os.cpu_count() or 1)
    safe = min(limits) // (4 * processes) if limits else 64 * 1024**2
    return min(int(requested * 1024**2), safe)


class SpatialQueryCache:
    def __init__(self, max_bytes):
        self.max_bytes = max(0, int(max_bytes))
        self.bytes_used = 0
        self.hits = 0
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

    def reserve(self, size):
        # Leave room for other simultaneously queried indexes and active arrays.
        if size > self.max_bytes // 4:
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


def region_entry(index, tile, cell, radius, positive_dimension, bounds, *,
                 selection_key=None, select_points=None):
    bound_key = None if bounds is None else tuple(np.asarray(bounds).ravel())
    key = (index._cache_owner, tile, *cell, radius, positive_dimension, bound_key,
           repr(index._storage_dtype().descr), selection_key)
    cached = index.query_cache.get(key)
    if cached is not None:
        return cached
    lo = np.asarray(cell) * REGION_SIZE - radius
    hi = (np.asarray(cell) + 1) * REGION_SIZE + radius
    clause = (' FROM bounds r JOIN batches b ON b.id=r.id '
              'WHERE b.tile=? AND r.x1>=? AND r.x0<=? AND r.y1>=? AND r.y0<=?')
    args = (int(tile), lo[0], hi[0], lo[1], hi[1])
    total = index.db.execute('SELECT COALESCE(SUM(length(b.data)),0)' + clause, args).fetchone()[0]
    dtype = index._storage_dtype()
    count = total // dtype.itemsize
    # Reserve the transient peak: the joined blobs, the selected copy, one blob
    # in flight and tree construction. Blobs touching the halo hold several
    # times the region's points, so the kept entry is charged afterwards at its
    # real size; charging the transient peak evicted entries far too early.
    transient = total * 2 + count * TREE_BYTES_PER_POINT + 1024**2
    if not index.query_cache.reserve(transient):
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
    if bounds is not None:
        from bounded_point_index import inside_xy
        mask &= inside_xy(data['xyz'], bounds)
    if select_points is not None:
        mask &= select_points(data['xyz'])
    payload = data[mask]
    del buffer, data
    entry = (cKDTree(payload['xyz']) if len(payload) else None, payload)
    # Never above the reserved transient, so the cache stays within its budget.
    index.query_cache.put(key, entry, payload.nbytes + len(payload) * TREE_BYTES_PER_POINT + 1024**2)
    return entry


def covered_region(index, xyz, radius, positive_dimension):
    """Positive-tree support for one XY group, without labels or tie resolution.

    Reuse the same bounded region trees as nearest queries. None requests the
    bounded disk fallback when a region exceeds the cache budget.
    """
    cell = tuple(np.floor(xyz[0, :2] / REGION_SIZE).astype(np.int64))
    from bounded_point_index import distance_limit
    limit = distance_limit(xyz, radius)
    found = np.zeros(len(xyz), dtype=bool)
    for tile in sorted(index._cache_tiles):
        entry = region_entry(index, tile, cell, limit, positive_dimension, None)
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


def nearest_region(index, xyz, radius, *, tile=None, before_tile=None, overlaps=None, positive_dimension=None):
    """Return exact nearest results for one XY region, or request disk fallback."""
    from bounded_point_index import distance_limit, distance_roundoff, inside_xy
    cell = tuple(np.floor(xyz[0, :2] / REGION_SIZE).astype(np.int64))
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
        entry = region_entry(index, source, cell, limit, positive_dimension, bounds)
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
                                 positive_dimension=positive_dimension)
    return dists, values, refs
