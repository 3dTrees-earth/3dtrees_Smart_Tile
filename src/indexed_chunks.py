"""A tile's records chunk by chunk, read back from the index built while writing it.

Every stage that writes a tile also indexes exactly the written records
(``index_written_record``): local coordinates, the prediction values and the
record number. The next stage's passes that need only those (query staging,
the shard of its own output) can read them from the index instead of decoding
the LAZ file again; on dense TLS a decode is most of such a pass.

Batches of one tile are stored in write order. One ``add`` call covers
consecutive records, at most ``MAX_BATCH_POINTS`` of them, in 2 m groups, so a
read chunk gathers its records from at most two adds. Every record must come
back exactly once, or ``indexed_chunks`` raises: checking that up front would
read the whole tile.
"""
from __future__ import annotations

import numpy as np

from bounded_point_index import MAX_BATCH_POINTS


class IndexMismatch(ValueError):
    """The index does not hold exactly the tile's records."""


def tile_rows(db, tile):
    """``(first id, last id)`` of the tile's batches, or None, found by bisection on the row id.

    Shards join the index in tile order, so tiles do not decrease with the id
    and a tile's batches are one id range. ``indexed_chunks`` checks every
    batch it reads, so an index built otherwise raises instead of misreading.
    """
    last = db.execute('SELECT MAX(id) FROM batches').fetchone()[0]
    if last is None:
        return None

    def first_at_least(value):
        lo, hi = 1, last + 1
        while lo < hi:
            middle = (lo + hi) // 2
            row = db.execute('SELECT id, tile FROM batches WHERE id>=? ORDER BY id LIMIT 1', (middle,)).fetchone()
            if row is None or row[1] >= value:
                hi = middle
            else:
                lo = middle + 1
        return lo

    first, stop = first_at_least(int(tile)), first_at_least(int(tile) + 1)
    return (first, stop - 1) if first < stop else None


def indexed_chunks(db, dtype, names, tile, point_count, chunk_points=MAX_BATCH_POINTS):
    """Yield ``(xyz, values)`` of records ``[c * chunk_points, (c + 1) * chunk_points)`` for c = 0, 1, ...

    ``xyz`` and ``values`` ({name: array}) equal ``coordinates`` and
    ``prediction_values`` of ``chunk_iterator(chunk_points)`` on the file.
    """
    rows = tile_rows(db, tile)
    chunks = -(-point_count // chunk_points) if point_count else 0
    if rows is None:
        if point_count:
            raise IndexMismatch(f'tile {tile}: no indexed records')
        return
    pending = {}
    following = 0
    cursor = db.execute('SELECT tile, data FROM batches WHERE id BETWEEN ? AND ? ORDER BY id', rows)
    for other, blob in cursor:
        if other != tile:
            raise IndexMismatch(f'tile {tile}: batches of tile {other} interleaved')
        data = np.frombuffer(blob, dtype=dtype)
        numbers = data['indices'] // chunk_points
        for number in np.unique(numbers):
            number = int(number)
            if not following <= number < chunks:
                raise IndexMismatch(f'tile {tile}: record outside chunk {following}..{chunks - 1}')
            members = data[numbers == number]
            chunk = pending.get(number)
            if chunk is None:
                size = min(chunk_points, point_count - number * chunk_points)
                chunk = pending[number] = [np.empty(size, dtype=dtype), np.zeros(size, dtype=bool), 0]
            slots = members['indices'] - number * chunk_points
            if np.any(chunk[1][slots]):
                raise IndexMismatch(f'tile {tile}: record indexed twice')
            chunk[0][slots], chunk[1][slots] = members, True
            chunk[2] += len(members)
        while following in pending and pending[following][2] == len(pending[following][0]):
            yield _split(pending.pop(following)[0], names)
            following += 1
    if pending or following != chunks:
        raise IndexMismatch(f'tile {tile}: {following} of {chunks} chunks complete')


def _split(records, names):
    xyz = np.ascontiguousarray(records['xyz'])
    return xyz, {name: np.ascontiguousarray(records[f'value_{i}']) for i, name in enumerate(names)}
