"""Disk-backed spatial batches for dense prediction geometry.

A bounded shared cache accelerates immutable regions; oversized regions fall
back to one query batch and one stored batch at a time. SQLite's
RTree indexes batch bounds, never individual points; XYZ/attributes stay in
bounded binary blobs. All coordinates use a common local origin.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from worker_budget import spatial_query_worker_count


MAX_BATCH_POINTS = 32_768
CELL_SIZE = 2.0


def coordinates(points, header, origin):
    # Subtract offsets before converting to float64 so large projected offsets
    # do not consume the precision needed for a one-centimetre boundary.
    return np.column_stack([
        (np.asarray(points[axis], dtype=np.longdouble) * np.longdouble(scale)
         + (np.longdouble(offset) - np.longdouble(base))).astype(np.float64)
        for axis, scale, offset, base in zip(
            ("X", "Y", "Z"), header.scales, header.offsets, origin
        )
    ])


def distance_roundoff(xyz):
    """Float64 coordinate error budget shared by boundaries and nearest ties."""
    magnitude = max(1.0, float(np.max(np.abs(xyz))) if len(xyz) else 1.0)
    return 8 * np.spacing(magnitude)


def distance_limit(xyz, radius):
    """Inclusive boundary with an explicit float64 roundoff allowance (8 ULPs).

    This is a numerical guard in local coordinates, not a spatial tolerance
    setting. It is recorded in the pipeline report and shared by every check.
    """
    return radius + distance_roundoff(xyz)


def spatial_batches(xyz):
    """Yield stable XY-cell index groups; input size is already capped."""
    cells = np.floor(xyz[:, :2] / CELL_SIZE).astype(np.int64)
    order = np.lexsort((cells[:, 1], cells[:, 0]))
    sorted_cells = cells[order]
    boundaries = np.flatnonzero(np.any(np.diff(sorted_cells, axis=0), axis=1)) + 1
    for group in np.split(order, boundaries):
        if len(group):
            yield group


class PointIndex:
    def __init__(self, path: Path, dimensions: dict[str, np.dtype], *, query_workers: int = 1,
                 read_only: bool = False, query_cache=None):
        from spatial_query_cache import shared_cache
        self.query_cache = shared_cache() if query_cache is None else query_cache
        self._cache_owner = object()
        self._cache_ready = read_only
        self._cache_tiles = set()
        self.query_workers = spatial_query_worker_count(query_workers)
        self.dimensions = dimensions
        self.path = Path(path).resolve()
        self.db = (sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)
                   if read_only else sqlite3.connect(self.path))
        self.db.execute("PRAGMA cache_size=-16384")
        self.db.execute("PRAGMA temp_store=FILE")
        if read_only:
            self.db.execute("PRAGMA query_only=ON")
            self._cache_tiles = {row[0] for row in self.db.execute("SELECT DISTINCT tile FROM batches")}
        else:
            self.db.execute("CREATE TABLE batches (id INTEGER PRIMARY KEY, tile INTEGER, data BLOB)")
            self.db.execute("CREATE VIRTUAL TABLE bounds USING rtree(id,x0,x1,y0,y1,z0,z1)")

    def query_tree(self, tree, xyz, **kwargs):
        """Parallelize large native queries; avoid thread overhead on tiny cells."""
        workers = min(self.query_workers, max(1, len(xyz) // 1024))
        return tree.query(xyz, workers=workers, **kwargs)

    def nearest_in_tree(self, tree, xyz, indices, limit):
        """Resolve numerical nearest ties by stable source point order.

        Header offsets can change float64 distances by a few ULPs. Use the
        existing coordinate error budget, while retaining the radius bound.
        """
        ds, ns = self.query_tree(tree, xyz, k=2,
                                 distance_upper_bound=np.nextafter(limit, np.inf))
        d, nearest = ds[:, 0].copy(), ns[:, 0].copy()
        roundoff = distance_roundoff(xyz)
        ambiguous = np.isfinite(ds[:, 1]) & (ds[:, 1] <= d + roundoff)
        for pos in np.flatnonzero(ambiguous):
            choices = np.asarray(tree.query_ball_point(
                xyz[pos], np.nextafter(min(limit, d[pos] + roundoff), np.inf)), dtype=np.int64)
            exact = np.linalg.norm(tree.data[choices] - xyz[pos], axis=1)
            eligible = (exact <= limit) & (exact <= exact.min() + roundoff)
            choices = choices[eligible]
            if len(choices):
                nearest[pos] = choices[np.argmin(indices[choices])]
                d[pos] = np.linalg.norm(tree.data[nearest[pos]] - xyz[pos])
        numerical = (np.isfinite(ds[:, 1]) & (ds[:, 1] > ds[:, 0])
                     & (ds[:, 1] <= ds[:, 0] + roundoff))
        numerical |= np.isfinite(d) & (d != ds[:, 0])
        return d, nearest, numerical

    def close(self):
        self.query_cache.discard(self._cache_owner)
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _storage_dtype(self):
        # This database is private scratch created for one run. A fixed record
        # layout avoids ZIP/NPY parsing and array copies for every spatial query.
        # Derive it from current dimensions: survivor schemas may be narrowed
        # before their first insertion.
        return np.dtype([("xyz", np.float64, (3,)), ("indices", np.int64)] +
                        [(f"value_{i}", dtype) for i, dtype in enumerate(self.dimensions.values())])

    def add(self, tile, xyz, values, point_indices):
        if len(xyz) > MAX_BATCH_POINTS:
            raise ValueError("Spatial index batch exceeds the fixed memory bound")
        if self._cache_ready:
            self.query_cache.discard(self._cache_owner)
        self._cache_ready = False
        self._cache_tiles.add(int(tile))
        for group in spatial_batches(xyz):
            pts = xyz[group]
            payload = np.empty(len(group), dtype=self._storage_dtype())
            payload["xyz"] = pts
            payload["indices"] = point_indices[group]
            for i, name in enumerate(self.dimensions):
                payload[f"value_{i}"] = values[name][group]
            row = self.db.execute("INSERT INTO batches(tile,data) VALUES (?,?)",
                                  (int(tile), payload.tobytes())).lastrowid
            lo, hi = pts.min(axis=0), pts.max(axis=0)
            self.db.execute("INSERT INTO bounds VALUES (?,?,?,?,?,?,?)",
                            (row, lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]))

    def flush(self):
        self.db.commit()
        self._cache_ready = True

    def candidates(self, xyz, radius, *, tile=None, before_tile=None):
        if not len(xyz):
            return
        radius = distance_limit(xyz, radius)
        lo, hi = xyz.min(axis=0) - radius, xyz.max(axis=0) + radius
        sql = ("SELECT b.tile,b.data FROM bounds r JOIN batches b ON b.id=r.id "
               "WHERE r.x1>=? AND r.x0<=? AND r.y1>=? AND r.y0<=? "
               "AND r.z1>=? AND r.z0<=?")
        args = [lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]]
        if tile is not None:
            sql += " AND b.tile=?"
            args.append(int(tile))
        if before_tile is not None:
            sql += " AND b.tile<?"
            args.append(int(before_tile))
        sql += " ORDER BY b.id"
        dtype = self._storage_dtype()
        for tile_id, data in self.db.execute(sql, args):
            arrays = np.frombuffer(data, dtype=dtype)
            yield tile_id, {"xyz": arrays["xyz"], "indices": arrays["indices"],
                            "values": {name: arrays[f"value_{i}"] for i, name in enumerate(self.dimensions)}}

    def nearest_distances(self, xyz, radius):
        """Minimum distances for baseline coverage, without ownership or attributes.

        Baseline validation needs only geometric coverage. Avoid resolving source
        identity for coincident unfiltered copies from neighboring tiles.
        """
        from spatial_query_cache import REGION_SIZE, region_entry
        distances = np.full(len(xyz), np.inf)
        for group in spatial_batches(xyz):
            points = xyz[group]
            limit = distance_limit(points, radius)
            cell = tuple(np.floor(points[0, :2] / REGION_SIZE).astype(np.int64))
            best = np.full(len(points), np.inf)
            for tile in sorted(self._cache_tiles):
                entry = None
                if self._cache_ready and self.query_cache.max_bytes:
                    entry = region_entry(self, tile, cell, limit, None, None)
                if entry is not None:
                    tree, _ = entry
                    trees = () if tree is None else (tree,)
                else:
                    trees = (cKDTree(data['xyz']) for _, data in self.candidates(points, radius, tile=tile))
                for tree in trees:
                    d, _ = self.query_tree(tree, points, distance_upper_bound=np.nextafter(limit, np.inf))
                    best = np.minimum(best, np.where(d <= limit, d, np.inf))
            distances[group] = best
        return distances

    def covered(self, xyz, radius, *, positive_dimension):
        """Whether any positive tree lies within radius of each query point.

        Coverage needs neither winning attributes nor nearest-point tie breaks.
        Preserve each original XY group's inclusive numerical distance bound,
        including when previously supported queries are skipped.
        """
        from spatial_query_cache import covered_region
        found = np.zeros(len(xyz), dtype=bool)
        for group in spatial_batches(xyz):
            points = xyz[group]
            result = None
            if self._cache_ready and self.query_cache.max_bytes:
                result = covered_region(self, points, radius, positive_dimension)
            if result is None:
                result = self._covered_batch(points, radius, positive_dimension)
            found[group] = result
        return found

    def _covered_batch(self, xyz, radius, positive_dimension):
        """Bounded fallback for one XY group; skip already supported queries."""
        found = np.zeros(len(xyz), dtype=bool)
        limit = distance_limit(xyz, radius)
        for _, data in self.candidates(xyz, radius):
            positive = data['values'][positive_dimension] > 0
            if not np.any(positive):
                continue
            remaining = np.flatnonzero(~found)
            distances, _ = self.query_tree(
                cKDTree(data['xyz'][positive]), xyz[remaining],
                distance_upper_bound=np.nextafter(limit, np.inf))
            found[remaining] = distances <= limit
            if np.all(found):
                break
        return found

    def nearest(self, xyz, radius, *, tile=None, before_tile=None, overlaps=None, positive_dimension=None):
        from spatial_query_cache import nearest_region
        options = dict(tile=tile, before_tile=before_tile, overlaps=overlaps,
                       positive_dimension=positive_dimension)
        if not self._cache_ready or not self.query_cache.max_bytes:
            return self._nearest_batches(xyz, radius, **options)
        distances = np.full(len(xyz), np.inf)
        values = {name: np.zeros(len(xyz), dtype=dtype) for name, dtype in self.dimensions.items()}
        refs = np.full((len(xyz), 2), -1, dtype=np.int64)
        # Preserve the original query grouping and numerical boundary allowance.
        for group in spatial_batches(xyz):
            result = nearest_region(self, xyz[group], radius, **options)
            if result is None:
                result = self._nearest_batches(xyz[group], radius, **options)
            distances[group], found_values, refs[group] = result
            for name in values:
                values[name][group] = found_values[name]
        return distances, values, refs

    def _nearest_batches(self, xyz, radius, *, tile=None, before_tile=None, overlaps=None, positive_dimension=None):
        """Nearest within radius; ties resolve by tile insertion/point order."""
        distances = np.full(len(xyz), np.inf)
        values = {name: np.zeros(len(xyz), dtype=dtype)
                  for name, dtype in self.dimensions.items()}
        refs = np.full((len(xyz), 2), -1, dtype=np.int64)
        numerical_ties = np.zeros(len(xyz), dtype=bool)
        for group in spatial_batches(xyz):
            pts = xyz[group]
            limit = distance_limit(pts, radius)
            for tile_id, data in self.candidates(pts, radius, tile=tile, before_tile=before_tile):
                if positive_dimension is not None:
                    data = select_data(data, data['values'][positive_dimension] > 0)
                    if not len(data['xyz']):
                        continue
                query_allowed = np.ones(len(pts), dtype=bool)
                if overlaps is not None:
                    if tile_id not in overlaps:
                        continue
                    query_allowed = inside_xy(pts, overlaps[tile_id])
                    selected = inside_xy(data["xyz"], overlaps[tile_id])
                    data = select_data(data, selected)
                    if not np.any(query_allowed) or not len(data["xyz"]):
                        continue
                tree = cKDTree(data["xyz"])
                d, nearest, numerical = self.nearest_in_tree(tree, pts, data['indices'], limit)
                found = np.isfinite(d) & (d <= limit) & query_allowed
                old = distances[group]
                old_refs = refs[group]
                better_ref = ((old_refs[:, 0] < 0) | (tile_id < old_refs[:, 0]) |
                              ((tile_id == old_refs[:, 0]) &
                               (data["indices"][np.minimum(nearest, len(data["indices"]) - 1)] < old_refs[:, 1])))
                roundoff = distance_roundoff(pts)
                tied = np.abs(np.subtract(d, old, out=np.full_like(d, np.inf), where=np.isfinite(d) & np.isfinite(old))) <= roundoff
                numerical_ties[group] |= numerical | (tied & (d != old))
                better = found & ((d < old - roundoff) | (tied & better_ref))
                chosen = group[better]
                distances[chosen] = d[better]
                refs[chosen, 0] = tile_id
                refs[chosen, 1] = data["indices"][nearest[better]]
                for name in values:
                    values[name][chosen] = data["values"][name][nearest[better]]
        self.resolve_numerical_ties(xyz, radius, numerical_ties, distances, values, refs,
                                    tile=tile, before_tile=before_tile, overlaps=overlaps,
                                    positive_dimension=positive_dimension)
        return distances, values, refs

    def resolve_numerical_ties(self, xyz, radius, ambiguous, distances, values, refs,
                               *, tile=None, before_tile=None, overlaps=None,
                               positive_dimension=None):
        """Resolve rare non-exact ties against the global minimum in two passes.

        A chain of pairwise-close distances is not an equivalence relation.
        Anchor the tie set to the true minimum, independently of cache regions
        and SQLite batch boundaries. Both passes hold one stored batch at a time.
        """
        if not np.any(ambiguous):
            return
        for group in spatial_batches(xyz):
            positions = group[ambiguous[group]]
            if not len(positions):
                continue
            points = xyz[positions]
            limit = distance_limit(xyz[group], radius)
            roundoff = distance_roundoff(xyz[group])

            def candidates():
                from spatial_query_cache import REGION_SIZE, region_entry
                cell = tuple(np.floor(points[0, :2] / REGION_SIZE).astype(np.int64))
                sources = [tile] if tile is not None else sorted(self._cache_tiles)
                for source in sources:
                    if before_tile is not None and source >= before_tile:
                        continue
                    if overlaps is not None and source not in overlaps:
                        continue
                    bounds = None if overlaps is None else overlaps[source]
                    allowed = np.ones(len(points), dtype=bool) if bounds is None else inside_xy(points, bounds)
                    if not np.any(allowed):
                        continue
                    entry = None
                    if self._cache_ready and self.query_cache.max_bytes:
                        entry = region_entry(self, source, cell, limit, positive_dimension, bounds)
                    if entry is not None:
                        tree, payload = entry
                        if tree is not None:
                            data = {'xyz': payload['xyz'], 'indices': payload['indices'],
                                    'values': {name: payload[f'value_{i}'] for i, name in enumerate(self.dimensions)}}
                            yield source, data, allowed, tree
                        continue
                    for _, data in self.candidates(points, radius, tile=source):
                        selected = np.ones(len(data['xyz']), dtype=bool)
                        if positive_dimension is not None:
                            selected &= data['values'][positive_dimension] > 0
                        if bounds is not None:
                            selected &= inside_xy(data['xyz'], bounds)
                        if np.any(selected):
                            data = select_data(data, selected)
                            yield source, data, allowed, cKDTree(data['xyz'])

            minimum = np.full(len(points), np.inf)
            for _, data, allowed, tree in candidates():
                d, _ = self.query_tree(tree, points,
                                       distance_upper_bound=np.nextafter(limit, np.inf))
                minimum[allowed] = np.minimum(minimum[allowed], d[allowed])
            refs[positions] = -1
            for source, data, allowed, tree in candidates():
                for i in np.flatnonzero(allowed & np.isfinite(minimum)):
                    ceiling = min(limit, minimum[i] + roundoff)
                    choices = np.asarray(tree.query_ball_point(points[i], np.nextafter(ceiling, np.inf)), dtype=np.int64)
                    if not len(choices):
                        continue
                    exact = np.linalg.norm(data['xyz'][choices] - points[i], axis=1)
                    choices = choices[exact <= ceiling]
                    if not len(choices):
                        continue
                    chosen = choices[np.argmin(data['indices'][choices])]
                    ref = (source, int(data['indices'][chosen]))
                    pos = positions[i]
                    if refs[pos, 0] < 0 or ref < tuple(refs[pos]):
                        refs[pos] = ref
                        distances[pos] = np.linalg.norm(data['xyz'][chosen] - points[i])
                        for name in values:
                            values[name][pos] = data['values'][name][chosen]

    def conflicting_match(self, xyz, labels, radius, *, before_tile, map_labels, overlaps=None,
                          background_semantics_owned=False, core_preferred=None):
        """Find any conflicting neighbour, including a non-nearest neighbour.

        Group stored points by label tuple and query once per group. This avoids
        materializing quadratic neighbour lists for coincident dense points.
        """
        for group in spatial_batches(xyz):
            pts = xyz[group]
            limit = distance_limit(pts, radius)
            for tile, data in self.candidates(pts, radius, before_tile=before_tile):
                query_allowed = np.ones(len(pts), dtype=bool)
                if overlaps is not None:
                    if tile not in overlaps:
                        continue
                    query_allowed = inside_xy(pts, overlaps[tile])
                    selected = inside_xy(data["xyz"], overlaps[tile])
                    data = select_data(data, selected)
                    if not np.any(query_allowed) or not len(data["xyz"]):
                        continue
                other = map_labels(tile, data["values"])
                unique, inverse = np.unique(other, axis=0, return_inverse=True)
                for label_id, label in enumerate(unique):
                    subset = np.flatnonzero(inverse == label_id)
                    d, nearest = self.query_tree(cKDTree(data["xyz"][subset]), pts)
                    disagrees = np.any(labels[group] != label, axis=1)
                    if background_semantics_owned and label[0] == 0:
                        # Adjacent spatial owners may legitimately classify nearby
                        # background differently; each keeps its own semantic value.
                        disagrees &= labels[group, 0] != 0
                    if core_preferred is not None:
                        other_pts = data["xyz"][subset[nearest]]
                        separate_owners = (core_preferred(None, tile, pts) &
                                           core_preferred(tile, None, other_pts))
                        # Adjacent records can keep different per-point semantics
                        # even within one merged tree when each owns its location.
                        disagrees &= ~((labels[group, 0] > 0) & (label[0] > 0) & separate_owners)
                    conflict = (d <= limit) & query_allowed & disagrees
                    if np.any(conflict):
                        pos = int(np.flatnonzero(conflict)[0])
                        ref = subset[nearest[pos]]
                        return {"tile": int(tile), "point": int(data["indices"][ref]),
                                "query_index": int(group[pos]), "xyz": data["xyz"][ref].tolist(),
                                "labels": label.tolist(), "distance_m": float(d[pos])}
        return None


def inside_xy(xyz, bounds):
    """Inclusive overlap membership with a fixed per-axis roundoff allowance.

    Equivalent LAS coordinates can straddle a declared bound by a few ULPs
    after offset conversion. Derive the allowance only from the local bounds,
    so cache contents, query batch composition and Z cannot change membership.
    This numerical guard does not change the point-matching radius or cores.
    """
    lo, hi = np.asarray(bounds, dtype=np.float64)
    magnitude = np.maximum(1., np.maximum(np.abs(lo), np.abs(hi)))
    allowance = 8 * np.spacing(magnitude)
    return np.all((xyz[:, :2] >= lo - allowance) &
                  (xyz[:, :2] <= hi + allowance), axis=1)


def select_data(data, mask):
    return {"xyz": data["xyz"][mask], "indices": data["indices"][mask],
            "values": {name: values[mask] for name, values in data["values"].items()}}
