"""Disk-backed spatial batches for dense prediction geometry.

Only one query batch and one stored batch are resident during a search. SQLite's
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


def distance_limit(xyz, radius):
    """Inclusive boundary with an explicit float64 roundoff allowance (8 ULPs).

    This is a numerical guard in local coordinates, not a spatial tolerance
    setting. It is recorded in the pipeline report and shared by every check.
    """
    magnitude = max(1.0, float(np.max(np.abs(xyz))) if len(xyz) else 1.0)
    return radius + 8 * np.spacing(magnitude)


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
                 read_only: bool = False):
        self.query_workers = spatial_query_worker_count(query_workers)
        self.dimensions = dimensions
        self.path = Path(path).resolve()
        self.db = (sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)
                   if read_only else sqlite3.connect(self.path))
        self.db.execute("PRAGMA cache_size=-16384")
        self.db.execute("PRAGMA temp_store=FILE")
        if read_only:
            self.db.execute("PRAGMA query_only=ON")
        else:
            self.db.execute("CREATE TABLE batches (id INTEGER PRIMARY KEY, tile INTEGER, data BLOB)")
            self.db.execute("CREATE VIRTUAL TABLE bounds USING rtree(id,x0,x1,y0,y1,z0,z1)")

    def query_tree(self, tree, xyz, **kwargs):
        """Parallelize large native queries; avoid thread overhead on tiny cells."""
        workers = min(self.query_workers, max(1, len(xyz) // 1024))
        return tree.query(xyz, workers=workers, **kwargs)

    def close(self):
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

    def nearest(self, xyz, radius, *, tile=None, before_tile=None, overlaps=None):
        """Nearest within radius; ties resolve by tile insertion/point order."""
        distances = np.full(len(xyz), np.inf)
        values = {name: np.zeros(len(xyz), dtype=dtype)
                  for name, dtype in self.dimensions.items()}
        refs = np.full((len(xyz), 2), -1, dtype=np.int64)
        for group in spatial_batches(xyz):
            pts = xyz[group]
            limit = distance_limit(pts, radius)
            for tile_id, data in self.candidates(pts, radius, tile=tile, before_tile=before_tile):
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
                ds, ns = self.query_tree(tree, pts, k=2, distance_upper_bound=np.nextafter(limit, np.inf))
                d, nearest = ds[:, 0], ns[:, 0]
                found = np.isfinite(d) & (d <= limit) & query_allowed
                # cKDTree does not specify equal-distance tie ordering. Resolve
                # only ambiguous nearest ties, one point at a time (bounded).
                for pos in np.flatnonzero(found & (ds[:, 0] == ds[:, 1])):
                    choices = tree.query_ball_point(pts[pos], np.nextafter(d[pos], np.inf))
                    if len(choices) > 1:
                        exact = np.linalg.norm(data["xyz"][choices] - pts[pos], axis=1)
                        best = np.min(exact)
                        choices = np.asarray(choices)[exact == best]
                        nearest[pos] = choices[np.argmin(data["indices"][choices])]
                        d[pos] = best
                old = distances[group]
                old_refs = refs[group]
                better_ref = ((old_refs[:, 0] < 0) | (tile_id < old_refs[:, 0]) |
                              ((tile_id == old_refs[:, 0]) &
                               (data["indices"][np.minimum(nearest, len(data["indices"]) - 1)] < old_refs[:, 1])))
                better = found & ((d < old) | ((d == old) & better_ref))
                chosen = group[better]
                distances[chosen] = d[better]
                refs[chosen, 0] = tile_id
                refs[chosen, 1] = data["indices"][nearest[better]]
                for name in values:
                    values[name][chosen] = data["values"][name][nearest[better]]
        return distances, values, refs

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
                        disagrees &= ~((labels[group, 0] > 0) & (label[0] > 0) &
                                       (labels[group, 0] != label[0]) & separate_owners)
                    conflict = (d <= limit) & query_allowed & disagrees
                    if np.any(conflict):
                        pos = int(np.flatnonzero(conflict)[0])
                        ref = subset[nearest[pos]]
                        return {"tile": int(tile), "point": int(data["indices"][ref]),
                                "query_index": int(group[pos]), "xyz": data["xyz"][ref].tolist(),
                                "labels": label.tolist(), "distance_m": float(d[pos])}
        return None


def inside_xy(xyz, bounds):
    lo, hi = bounds
    return np.all((xyz[:, :2] >= lo) & (xyz[:, :2] <= hi), axis=1)


def select_data(data, mask):
    return {"xyz": data["xyz"][mask], "indices": data["indices"][mask],
            "values": {name: values[mask] for name, values in data["values"].items()}}
