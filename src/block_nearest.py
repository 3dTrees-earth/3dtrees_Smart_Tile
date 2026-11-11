"""Nearest-point queries answered block by block instead of per 2 m query group.

``PointIndex.nearest`` answers one 2 m XY group at a time from a region tree
read from SQLite. Dense transfer streams a tile's 1 cm targets in record
order, and on dense TLS every 8 m block is visited by about 24 of a tile's
1 M-point chunks, so each region is read again and again (2-tile 2585 block:
12,000 of 15,800 worker CPU seconds in the region reads).

Here the queries are spilled to disk in XY cells, each point with the
distance bound and roundoff ``nearest`` would use for its original 2 m group,
and answered per ``BLOCK_SIZE`` block with one read and one tree:

* a point whose second-nearest candidate lies beyond its bound, or farther
  than its nearest by more than the roundoff, has an unambiguous nearest
  point; ``nearest_in_tree`` returns exactly that point (distances come from
  the same coordinates, and a candidate within the bound is in every tree);
* any original group with an ambiguous point is answered again, whole, by
  ``PointIndex.nearest``, so ties are resolved by the existing code.
"""
from __future__ import annotations

import multiprocessing
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from bounded_point_index import PointIndex, distance_limit, distance_roundoff, inside_xy, spatial_batches
from point_ranges import seekable

BLOCK_SIZE = 8.0
# One spill file per block: dense TLS piles most of a tile into a few metres
# (2585: 570 M of 706 M points in one 16 m strip), so a worker holds one block.
CELL_SIZE = BLOCK_SIZE
SPILL_DTYPE = np.dtype([('xyz', np.float64, (3,)), ('limit', np.float64), ('roundoff', np.float64),
                        ('row', np.int64), ('group', np.int64)])


class NearestSpill:
    """Query points of one pass, each with its original group's bound, spilled per XY cell."""

    def __init__(self, folder, *, first_row=0):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.cells = {}
        self.rows = first_row
        # Group numbers only need to be unique; a part starting at row r uses r, r + 1, ...
        self.groups = first_row

    def add(self, xyz, radius):
        """Spill one query batch (``nearest(xyz, radius)``); rows number points in ``add`` order."""
        records = np.empty(len(xyz), dtype=SPILL_DTYPE)
        records['xyz'] = xyz
        records['row'] = np.arange(self.rows, self.rows + len(xyz))
        for group in spatial_batches(xyz):
            records['limit'][group] = distance_limit(xyz[group], radius)
            records['roundoff'][group] = distance_roundoff(xyz[group])
            records['group'][group] = self.groups
            self.groups += 1
        self.rows += len(xyz)
        cells = np.floor(xyz[:, :2] / CELL_SIZE).astype(np.int64)
        order = np.lexsort((cells[:, 1], cells[:, 0]))
        boundaries = np.flatnonzero(np.any(np.diff(cells[order], axis=0), axis=1)) + 1
        for members in np.split(order, boundaries):
            if not len(members):
                continue
            cell = tuple(int(c) for c in cells[members[0]])
            path = self.folder / f'cell_{cell[0]}_{cell[1]}.bin'
            with open(path, 'ab') as stream:
                stream.write(records[members].tobytes())
            self.cells[cell] = path

    def answer(self, index_path, dimensions, radius, results, *, workers=1, tile=None, overlaps=None,
               before_tile=None):
        """Fill ``results`` (a NearestResults over ``self.rows``) with ``nearest(..., tile, before_tile, overlaps)``."""
        jobs = [(paths, str(index_path), dimensions, radius, str(results.folder), self.rows, tile, overlaps,
                 before_tile) for paths in cell_jobs(self.cells, workers)]
        if workers > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(max_workers=min(workers, len(jobs)),
                                     mp_context=multiprocessing.get_context('spawn')) as pool:
                list(pool.map(_answer_cells, *zip(*jobs)))
        else:
            for job in jobs:
                _answer_cells(*job)
        for path in self.cells.values():
            Path(path).unlink()
        self.folder.rmdir()


def spill_file(path, origin, radius, folder, *, processes=1, chunk_points):
    """Spill every point of a LAS/LAZ file, as ``nearest`` would be asked chunk by chunk.

    Chunk ranges are read in up to ``processes`` processes (LAZ seek, the same
    chunk boundaries) and their cells joined, so rows are the file's record
    numbers and each point keeps the bound of its group in its chunk.
    """
    import laspy
    with laspy.open(path) as reader:
        total = int(reader.header.point_count)
    chunks = max(1, -(-total // chunk_points))
    # Parts after the first seek: a variable-chunk file is spilled in one part.
    share = max(1, -(-chunks // max(1, int(processes)))) if seekable(path) else chunks
    jobs = [(str(path), origin, radius, str(Path(folder) / f'part_{first:06d}'), first, min(first + share, chunks),
             chunk_points) for first in range(0, chunks, share)]
    if len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=len(jobs), mp_context=multiprocessing.get_context('spawn')) as pool:
            parts = list(pool.map(_spill_part, *zip(*jobs)))
    else:
        parts = [_spill_part(*job) for job in jobs]
    spill = NearestSpill(folder)
    for cell in sorted({cell for part in parts for cell in part}):
        target = spill.folder / f'cell_{cell[0]}_{cell[1]}.bin'
        with open(target, 'wb') as stream:
            for part in parts:
                if cell in part:
                    with open(part[cell], 'rb') as source:
                        shutil.copyfileobj(source, stream, 16 * 1024**2)
                    Path(part[cell]).unlink()
        spill.cells[cell] = target
    for job in jobs:
        Path(job[3]).rmdir()
    spill.rows = total
    return spill


def _spill_part(path, origin, radius, folder, first, stop, chunk_points):
    import laspy
    from bounded_point_index import coordinates, read_chunks
    spill = NearestSpill(folder, first_row=first * chunk_points)
    with laspy.open(path) as reader:
        if first:
            reader.seek(first * chunk_points)
        for number, record in enumerate(read_chunks(reader, chunk_points), start=first):
            if number >= stop:
                break
            spill.add(coordinates(record, reader.header, origin), radius)
    return spill.cells


class NearestResults:
    """Disk arrays of ``nearest`` answers by row: whether a point matched and its values."""

    def __init__(self, folder, dimensions, rows):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.dimensions, self.rows = dict(dimensions), rows
        for name, dtype in self.arrays():
            np.lib.format.open_memmap(self.folder / name, mode='w+', dtype=dtype, shape=(rows,)).flush()

    def arrays(self):
        yield 'matched.npy', np.bool_
        for i, dtype in enumerate(self.dimensions.values()):
            yield f'value_{i}.npy', np.dtype(dtype)

    def open(self, mode='r'):
        """``(matched, {dimension: values})`` memory maps."""
        maps = [np.load(self.folder / name, mmap_mode=mode) for name, _ in self.arrays()]
        return maps[0], dict(zip(self.dimensions, maps[1:]))

    def remove(self):
        for name, _ in self.arrays():
            (self.folder / name).unlink(missing_ok=True)
        self.folder.rmdir()


def cell_jobs(cells, workers):
    """Spill files as lists of about equal bytes, a few per worker, each list in cell order.

    A job opens the index once for all of its cells.
    """
    paths = sorted(cells.items())
    count = max(1, min(len(paths), 4 * max(1, int(workers))))
    loads, jobs = [0] * count, [[] for _ in range(count)]
    sizes = [Path(path).stat().st_size for _, path in paths]
    for number in sorted(range(len(paths)), key=lambda n: (-sizes[n], n)):
        lightest = loads.index(min(loads))
        loads[lightest] += sizes[number]
        jobs[lightest].append(number)
    return [[str(paths[number][1]) for number in sorted(job)] for job in jobs if job]


def _answer_cells(spill_paths, index_path, dimensions, radius, results_folder, rows, tile=None, overlaps=None,
                  before_tile=None):
    results = NearestResults.__new__(NearestResults)
    results.folder, results.dimensions, results.rows = Path(results_folder), dict(dimensions), rows
    matched, values = results.open('r+')
    with PointIndex(index_path, dimensions, query_workers=1, read_only=True) as index:
        dtype = index._storage_dtype()
        for spill_path in spill_paths:
            records = np.fromfile(spill_path, dtype=SPILL_DTYPE)
            for block in cell_blocks(records['xyz']):
                _answer_block(index, dtype, records[block], radius, matched, values, tile, overlaps, before_tile)
    matched.flush()
    for array in values.values():
        array.flush()


def cell_blocks(xyz):
    """Index arrays of the points of each ``BLOCK_SIZE`` block, in block order."""
    blocks = np.floor(xyz[:, :2] / BLOCK_SIZE).astype(np.int64)
    order = np.lexsort((blocks[:, 1], blocks[:, 0]))
    boundaries = np.flatnonzero(np.any(np.diff(blocks[order], axis=0), axis=1)) + 1
    return [block for block in np.split(order, boundaries) if len(block)]


def _answer_block(index, dtype, records, radius, matched, values, tile=None, overlaps=None, before_tile=None):
    """``nearest(xyz, radius, tile=, before_tile=, overlaps=)`` for one block of spilled points.

    Sources are searched in tile order with one tree each, combined as in
    ``nearest_region``: a lower tile keeps an exact tie, a closer point wins.
    With ``overlaps`` ({source tile: XY bounds}) a source offers only its
    points inside its bounds, to queries inside them. Any original group with
    a near tie, inside a source or between sources, is answered again by
    ``PointIndex.nearest`` itself.
    """
    xyz, limits, roundoff = records['xyz'], records['limit'], records['roundoff']
    halo = limits.max()
    lo, hi = xyz.min(axis=0) - halo, xyz.max(axis=0) + halo
    data, tiles = _indexed_points(index.db, dtype, lo, hi, tile, before_tile)
    rows = records['row']
    count = len(xyz)
    best = np.full(count, np.inf)
    best_tile = np.full(count, -1, dtype=np.int64)
    best_index = np.full(count, -1, dtype=np.int64)
    best_row = np.full(count, -1, dtype=np.int64)
    redo = np.zeros(count, dtype=bool)
    sources = [tile] if tile is not None else sorted(set(tiles.tolist()))
    for source in sources:
        if before_tile is not None and source >= before_tile:
            continue
        if overlaps is not None and source not in overlaps:
            continue
        selected = np.flatnonzero(tiles == source)
        allowed = np.ones(count, dtype=bool)
        if overlaps is not None:
            selected = selected[inside_xy(data['xyz'][selected], overlaps[source])]
            allowed = inside_xy(xyz, overlaps[source])
        if not len(selected):
            continue
        ds, ns = cKDTree(data['xyz'][selected]).query(xyz, k=2, distance_upper_bound=np.nextafter(halo, np.inf))
        # A candidate beyond the point's own bound does not exist for nearest().
        ds[ds > limits[:, None]] = np.inf
        redo |= np.isfinite(ds[:, 1]) & (ds[:, 1] <= ds[:, 0] + roundoff)
        d = ds[:, 0]
        rows_in_data = selected[np.minimum(ns[:, 0], len(selected) - 1)]
        indices = data['indices'][rows_in_data]
        found = np.isfinite(d) & allowed
        both = np.isfinite(d) & np.isfinite(best)
        tied = np.zeros(count, dtype=bool)
        tied[both] = np.abs(d[both] - best[both]) <= roundoff[both]
        redo |= tied & (d != best)
        better_ref = (best_tile < 0) | (source < best_tile) | ((source == best_tile) & (indices < best_index))
        take = found & ((d < best - roundoff) | (tied & better_ref))
        best[take], best_tile[take], best_index[take], best_row[take] = d[take], source, indices[take], rows_in_data[take]
    groups = np.unique(records['group'][redo])
    redo = np.isin(records['group'], groups)
    found = np.isfinite(best) & ~redo
    matched[rows[found]] = True
    for i, name in enumerate(values):
        values[name][rows[found]] = data[f'value_{i}'][best_row[found]]
    # Groups with a near tie: answer them as nearest() does, whole.
    for group in groups:
        members = np.flatnonzero(records['group'] == group)
        distances, found_values, _ = index.nearest(xyz[members], radius, tile=tile, before_tile=before_tile,
                                                   overlaps=overlaps)
        matched[rows[members]] = np.isfinite(distances)
        for name in values:
            values[name][rows[members]] = found_values[name]


def _indexed_points(db, dtype, lo, hi, tile=None, before_tile=None):
    """Indexed records inside the box [lo, hi] and their tiles (of ``tile``, or tiles before ``before_tile``)."""
    sql = ('SELECT b.tile, b.data FROM bounds r JOIN batches b ON b.id=r.id '
           'WHERE r.x1>=? AND r.x0<=? AND r.y1>=? AND r.y0<=? AND r.z1>=? AND r.z0<=?')
    args = [lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]]
    if tile is not None:
        sql += ' AND b.tile=?'
        args.append(int(tile))
    if before_tile is not None:
        sql += ' AND b.tile<?'
        args.append(int(before_tile))
    rows = db.execute(sql + ' ORDER BY b.id', args).fetchall()
    if not rows:
        return np.empty(0, dtype=dtype), np.empty(0, dtype=np.int64)
    data = np.frombuffer(b''.join(row[1] for row in rows), dtype=dtype)
    tiles = np.repeat(np.array([row[0] for row in rows], dtype=np.int64),
                      [len(row[1]) // dtype.itemsize for row in rows])
    inside = np.all((data['xyz'] >= lo) & (data['xyz'] <= hi), axis=1)
    return data[inside], tiles[inside]
