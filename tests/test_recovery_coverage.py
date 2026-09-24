"""Recovery coverage stays identical to the uncached positive-tree search."""
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree

from bounded_point_index import PointIndex, distance_limit, spatial_batches
from orphan_instance_recovery import _positive_support
from spatial_query_cache import SpatialQueryCache


def legacy_support(index, xyz, dimension):
    found = np.zeros(len(xyz), dtype=bool)
    for group in spatial_batches(xyz):
        points = xyz[group]
        for _, data in index.candidates(points, .01):
            positive = data['values'][dimension] > 0
            if not np.any(positive):
                continue
            distances, _ = index.query_tree(cKDTree(data['xyz'][positive]), points)
            found[group] |= distances <= distance_limit(points, .01)
            if np.all(found[group]):
                break
    return found


def test_coverage_boundaries_background_and_cache_reuse(tmp_path):
    pool = SpatialQueryCache(32 * 1024**2)
    with PointIndex(tmp_path/'index.db', {'tree': np.uint32}, query_cache=pool) as index:
        xyz = np.array([[0., 0, 0], [.01, 0, 0], [4., 0, .01],
                        [-4., 0, 0], [8.01, 0, 0], [12., 0, .010001]])
        index.add(0, xyz, {'tree': np.array([0, 1, 2, 3, 4, 5])}, np.arange(len(xyz)))
        index.flush()
        queries = np.array([[0., 0, 0], [4., 0, 0], [-4.01, 0, 0],
                            [8., 0, 0], [12., 0, 0], [99., 0, 0]])
        expected = legacy_support(index, queries, 'tree')
        np.testing.assert_array_equal(expected, [True, True, True, True, False, False])
        # Coverage must not invoke nearest's attribute gathering or tie handling.
        with patch.object(index, 'nearest', side_effect=AssertionError('nearest called')):
            np.testing.assert_array_equal(index.covered(queries, .01, positive_dimension='tree'), expected)
            hits = pool.hits
            with patch('spatial_query_cache.cKDTree', side_effect=AssertionError('rebuilt warm tree')):
                np.testing.assert_array_equal(_positive_support(index, queries, 'tree'), expected)
            assert pool.hits > hits
        assert index.covered(np.empty((0, 3)), .01, positive_dimension='tree').shape == (0,)


def test_coverage_cache_fallback_eviction_and_models_match_legacy(tmp_path):
    rng = np.random.default_rng(2183)
    for budget in (0, 1, 8 * 1024**2):
        pool = SpatialQueryCache(budget)
        with PointIndex(tmp_path/f'{budget}.db', {'SAT': np.uint32, 'FM': np.uint32}, query_cache=pool) as index:
            queries = []
            for tile in (2, 0, 1):
                xyz = rng.uniform([-12, -8, -2], [12, 8, 2], (160, 3))
                xyz[:3] = [[0, 0, 0], [0, 0, 0], [.01, 0, 0]]
                values = {key: rng.integers(0, 3, len(xyz), dtype=np.uint32) for key in index.dimensions}
                for start in range(0, len(xyz), 20):
                    stop = start + 20
                    index.add(tile, xyz[start:stop], {k: v[start:stop] for k, v in values.items()}, np.arange(start, stop))
                queries.append(xyz + rng.uniform(-.012, .012, xyz.shape))
            index.flush()
            queries = np.concatenate(queries)
            for key in index.dimensions:
                expected = legacy_support(index, queries, key)
                np.testing.assert_array_equal(index.covered(queries, .01, positive_dimension=key), expected)
                np.testing.assert_array_equal(_positive_support(index, queries, key), expected)
            assert pool.bytes_used <= pool.max_bytes
        assert pool.bytes_used == 0


def test_coverage_invalidates_on_append_and_read_only_reopen(tmp_path):
    path = tmp_path/'index.db'
    pool = SpatialQueryCache(8 * 1024**2)
    query = np.zeros((1, 3))
    with PointIndex(path, {'tree': np.uint32}, query_cache=pool) as index:
        index.add(0, query, {'tree': np.array([0])}, np.array([0]))
        index.flush()
        assert not _positive_support(index, query, 'tree')[0]
        index.add(1, query, {'tree': np.array([42])}, np.array([0]))
        # A mutable index uses the disk fallback, including uncommitted inserts.
        assert index.covered(query, .01, positive_dimension='tree')[0]
        index.flush()
        assert _positive_support(index, query, 'tree')[0]
    with PointIndex(path, {'tree': np.uint32}, query_cache=pool, read_only=True) as index:
        assert _positive_support(index, query, 'tree')[0]


def test_coverage_uses_original_group_roundoff_allowance(tmp_path):
    pool = SpatialQueryCache(16 * 1024**2)
    with PointIndex(tmp_path/'index.db', {'tree': np.uint32}, query_cache=pool) as index:
        # The high-Z query defines the original XY group's numerical allowance,
        # even after it is covered and only the low-Z query remains to search.
        query = np.array([[0., 0, 1e6], [0., 0, 0]])
        index.add(0, query[:1], {'tree': np.array([1])}, np.array([0]))
        index.add(1, np.array([[.01 + 4*np.spacing(1e6), 0, 0]]), {'tree': np.array([2])}, np.array([0]))
        index.flush()
        expected = legacy_support(index, query, 'tree')
        np.testing.assert_array_equal(expected, [True, True])
        np.testing.assert_array_equal(index.covered(query, .01, positive_dimension='tree'), expected)
