"""Disk-backed, incremental selection of uncovered orphan claims."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from bounded_point_index import MAX_BATCH_POINTS, distance_limit, spatial_batches
from merge_stages import DUPLICATE_RADIUS


def _coordinates(rows):
    return np.array([np.frombuffer(row[0], dtype='<f8') for row in rows])


def _prepare_selection(db):
    # Keep exact float64 coordinates in loc; the RTree's outward-rounded float32
    # bounds only prune candidates, never decide whether a sample is covered.
    db.execute('CREATE TABLE locations (id INTEGER PRIMARY KEY, loc BLOB UNIQUE)')
    db.execute('INSERT INTO locations(loc) SELECT DISTINCT loc FROM claims')
    db.execute('CREATE VIRTUAL TABLE location_bounds USING rtree(id,x0,x1,y0,y1,z0,z1)')
    cursor = db.execute('SELECT loc,id FROM locations')
    padding = DUPLICATE_RADIUS
    while rows := cursor.fetchmany(MAX_BATCH_POINTS):
        xyz = _coordinates(rows)
        padding = max(padding, distance_limit(xyz, DUPLICATE_RADIUS))
        db.executemany('INSERT INTO location_bounds VALUES (?,?,?,?,?,?,?)',
                       ((row[1], x, x, y, y, z, z) for row, (x, y, z) in zip(rows, xyz)))
    db.execute('CREATE TABLE scores (tile INTEGER, instance INTEGER, score INTEGER, '
               'PRIMARY KEY(tile,instance)) WITHOUT ROWID')
    db.execute('INSERT INTO scores SELECT tile,instance,COUNT(*) FROM claims GROUP BY tile,instance')
    db.execute('CREATE INDEX scores_rank ON scores(score DESC,tile,instance) WHERE score>0')
    db.execute('CREATE TEMP TABLE newly_covered (loc BLOB PRIMARY KEY) WITHOUT ROWID')
    return padding


def _cover_nearby(db, geometry, padding, query_tree, metric):
    for group in spatial_batches(geometry):
        points = geometry[group]
        tree = cKDTree(points)
        lo, hi = points.min(axis=0) - padding, points.max(axis=0) + padding
        cursor = db.execute(
            'SELECT l.loc FROM location_bounds r JOIN locations l ON l.id=r.id '
            'WHERE r.x1>=? AND r.x0<=? AND r.y1>=? AND r.y0<=? AND r.z1>=? AND r.z0<=? '
            'AND NOT EXISTS (SELECT 1 FROM covered c WHERE c.loc=l.loc)',
            (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]))
        while rows := cursor.fetchmany(MAX_BATCH_POINTS):
            xyz = _coordinates(rows)
            metric['selection_checked_locations'] += len(xyz)
            supported = np.zeros(len(xyz), dtype=bool)
            for batch in spatial_batches(xyz):
                distances, _ = query_tree(tree, xyz[batch])
                supported[batch] = distances <= distance_limit(xyz[batch], DUPLICATE_RADIUS)
            if not np.any(supported):
                continue
            db.execute('DELETE FROM newly_covered')
            db.executemany('INSERT INTO newly_covered VALUES (?)',
                           (rows[pos] for pos in np.flatnonzero(supported)))
            # Each newly covered location decrements every claimant exactly once.
            # Indexed claimant updates avoid rescoring unrelated instances.
            changes = db.execute(
                'SELECT COUNT(*),c.tile,c.instance FROM newly_covered n '
                'CROSS JOIN claims c ON c.loc=n.loc GROUP BY c.tile,c.instance')
            db.executemany('UPDATE scores SET score=score-? WHERE tile=? AND instance=?', changes)
            db.execute('INSERT INTO covered SELECT loc FROM newly_covered')


def select_claims(db, query_tree, metric, *, admit_candidate=None):
    """Yield greedy winners with stable ties, updating only nearby coverage."""
    padding = _prepare_selection(db)
    metric['selection_checked_locations'] = 0
    while True:
        best = db.execute('SELECT tile,instance,score FROM scores WHERE score>0 '
                          'ORDER BY score DESC,tile,instance LIMIT 1').fetchone()
        if best is None:
            return
        tile, uid, score = map(int, best)
        if admit_candidate is not None and not admit_candidate(tile, uid):
            db.execute('DELETE FROM scores WHERE tile=? AND instance=?', (tile, uid))
            continue
        cursor = db.execute('SELECT loc FROM claims WHERE tile=? AND instance=?', (tile, uid))
        # Include already covered samples of the winner: they can cover a
        # different claimant's nearby samples when this instance is admitted.
        while rows := cursor.fetchmany(MAX_BATCH_POINTS):
            _cover_nearby(db, _coordinates(rows), padding, query_tree, metric)
        yield tile, uid, score
