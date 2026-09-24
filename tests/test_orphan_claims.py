"""Selection equivalence and work bounds for spatial orphan recovery."""
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from bounded_point_index import distance_limit
from dense_tile_merge import DUPLICATE_RADIUS
from orphan_claims import select_claims


def select(claimants):
    metric = {}
    with sqlite3.connect(':memory:') as db:
        db.execute('CREATE TABLE claims (tile INTEGER, instance INTEGER, loc BLOB, '
                   'PRIMARY KEY(tile,instance,loc)) WITHOUT ROWID')
        db.execute('CREATE INDEX claims_loc ON claims(loc)')
        db.execute('CREATE TABLE covered (loc BLOB PRIMARY KEY) WITHOUT ROWID')
        for (tile, uid), points in claimants.items():
            db.executemany('INSERT OR IGNORE INTO claims VALUES (?,?,?)',
                           ((tile, uid, np.asarray(p, dtype='<f8').tobytes()) for p in points))
        winners = list(select_claims(db, lambda tree, xyz: tree.query(xyz), metric))
        scores = db.execute('SELECT score FROM scores').fetchall()
        assert all(score == 0 for score, in scores)
    return winners, metric['selection_checked_locations']


def reference(claimants):
    """Original greedy policy: recompute all scores from admitted geometry."""
    remaining = {key: np.unique(np.asarray(points), axis=0) for key, points in claimants.items()}
    admitted, winners = [], []
    while True:
        scores = {}
        for key, points in remaining.items():
            if not admitted:
                scores[key] = len(points)
            else:
                distance = np.linalg.norm(points[:, None, :] - np.array(admitted)[None, :, :], axis=2).min(axis=1)
                scores[key] = int(np.count_nonzero(distance > distance_limit(points, DUPLICATE_RADIUS)))
        best = min(scores, key=lambda key: (-scores[key], *key))
        if scores[best] == 0:
            return winners
        winners.append((*best, scores[best]))
        admitted.extend(remaining[best])


def test_independent_claims_require_linear_support_checks():
    for n in (8, 16, 64):
        winners, checked = select({(0, i): [[1.1, 2.0 * i, 0.]] for i in range(n)})
        assert winners == [(0, i, 1) for i in range(n)]
        assert checked == n


def test_matches_full_rescoring_for_overlapping_claims(monkeypatch):
    # Exercise boundaries between SQL reads and geometry chunks as well as
    # exact duplicate locations shared by multiple claimants.
    import orphan_claims
    monkeypatch.setattr(orphan_claims, 'MAX_BATCH_POINTS', 3)
    for seed in range(20):
        rng = np.random.default_rng(seed)
        cloud = rng.integers(-5, 6, size=(30, 3)) * .006
        claims = {(i % 3, i): cloud[rng.choice(len(cloud), 8, replace=False)] for i in range(9)}
        winners, _ = select(claims)
        assert winners == reference(claims), seed


def test_already_covered_geometry_of_new_winner_still_adds_support():
    claims = {
        (0, 1): [[0., 0., 0.], [0., 1., 0.], [0., 2., 0.]],
        (1, 1): [[.009, 0., 0.], [1., 0., 0.]],
        (2, 1): [[.018, 0., 0.]],
    }
    assert select(claims)[0] == [(0, 1, 3), (1, 1, 1)]


def test_spatial_pruning_keeps_radius_and_float32_boundaries():
    for offset in (0., 2., 1000000.):
        claims = {(0, 1): [[offset, 0., 0.]],
                  (1, 1): [[offset + .01, 0., 0.]],
                  (2, 1): [[offset + .0101, 0., 0.]]}
        assert select(claims)[0] == reference(claims)


def test_no_claims():
    assert select({}) == ([], 0)
