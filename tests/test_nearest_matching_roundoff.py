"""Same-instance remapping must share ordinary nearest's numerical tie policy."""
import numpy as np
import pytest

from bounded_point_index import PointIndex
from parallel_remap import RemapBatchQueries
from spatial_query_cache import SpatialQueryCache


DIMS = {'instance': np.uint32, 'score': np.int32}


@pytest.mark.parametrize('count,expected', [(2, 0), (3, 1)])
@pytest.mark.parametrize('split', [False, True])
@pytest.mark.parametrize('separate_tiles', [False, True])
@pytest.mark.parametrize('budget', [0, 1, 32 * 1024**2])
def test_matching_ties_anchor_to_global_minimum_across_storage_batches(
        tmp_path, count, expected, split, separate_tiles, budget):
    epsilon = 8 * np.spacing(1.)
    xyz = np.c_[.125 + np.arange(count - 1, -1, -1) * epsilon, np.zeros((count, 2))]
    with PointIndex(tmp_path / 'index.db', DIMS, query_cache=SpatialQueryCache(budget)) as index:
        # An exact point from another instance must not enter either tie pass.
        index.add(0, np.zeros((1, 3)), {'instance': np.array([99]), 'score': np.array([999])}, np.array([99]))
        groups = [np.array([i]) for i in range(count)] if split or separate_tiles else [np.arange(count)]
        for group in groups:
            index.add(int(group[0]) if separate_tiles else 0, xyz[group],
                      {'instance': np.full(len(group), 7), 'score': 100 + group}, group)
        index.flush()
        distances, values, refs = index.nearest_matching(np.zeros((1, 3)), .2, 'instance', np.array([7]))
        assert refs.tolist() == [[expected if separate_tiles else 0, expected]]
        assert values['instance'].tolist() == [7]
        assert values['score'].tolist() == [100 + expected]
        assert distances[0] == xyz[expected, 0]
        assert index.query_cache.bytes_used <= budget


@pytest.mark.parametrize('budget', [0, 1, 32 * 1024**2])
def test_matching_exact_controls_missing_labels_and_cache_isolation(tmp_path, budget):
    with PointIndex(tmp_path / 'index.db', DIMS, query_cache=SpatialQueryCache(budget)) as index:
        # Insert later source first; exact ties still prefer earlier tile/point.
        index.add(1, np.array([[.125, 0., 0.]]),
                  {'instance': np.array([7]), 'score': np.array([71])}, np.array([0]))
        index.add(0, np.array([[.125, 0., 0.], [.125, 0., 0.], [0., 0., 0.], [.03, 0., 0.]]),
                  {'instance': np.array([7, 7, 8, 0]), 'score': np.array([79, 73, 80, 0])},
                  np.array([9, 3, 0, 1]))
        index.flush()
        labels = np.array([7, 8, 0, 123])
        queries = np.zeros((4, 3))
        distances, values, refs = index.nearest_matching(queries, .2, 'instance', labels)
        assert refs.tolist() == [[0, 3], [0, 0], [0, 1], [-1, -1]]
        assert values['score'].tolist() == [73, 80, 0, 0]
        np.testing.assert_array_equal(distances[:3], [.125, 0., .03])
        assert np.isinf(distances[-1])
        # Warm cache queries in reverse label order cannot borrow other trees.
        again = index.nearest_matching(queries, .2, 'instance', labels[::-1])
        np.testing.assert_array_equal(again[2], refs[::-1])
        hits = index.query_cache.hits
        index.nearest_matching(queries, .2, 'instance', labels)
        if budget > 1:
            assert index.query_cache.hits > hits
        assert index.query_cache.bytes_used <= budget
        assert np.isinf(index.nearest_matching(queries[:1], .01, 'instance', labels[:1])[0][0])
        assert index.nearest_matching(np.empty((0, 3)), .2, 'instance', np.array([], dtype=int))[0].size == 0


@pytest.mark.parametrize('workers', [1, 4])
@pytest.mark.parametrize('budget_mb', [0, 32])
def test_instance_remap_workers_share_global_minimum_ties(tmp_path, monkeypatch, workers, budget_mb):
    monkeypatch.setenv('SMARTTILE_SPATIAL_CACHE_MB', str(budget_mb))
    with PointIndex(tmp_path / 'index.db', DIMS,
                    query_cache=SpatialQueryCache(budget_mb * 1024**2)) as index:
        queries, labels = [], []
        for base in (0., 4., 8., 12.):
            epsilon = 8 * np.spacing(max(1., base))
            for point in range(3):
                index.add(0, np.array([[base + .125 + (2 - point) * epsilon, 0., 0.]]),
                          {'instance': np.array([7]), 'score': np.array([100 + point])}, np.array([point]))
            index.add(0, np.array([[base, 0., 0.]]),
                      {'instance': np.array([42]), 'score': np.array([999])}, np.array([99]))
            queries.extend([[base, 0., 0.]] * 3)
            labels.extend([7, 42, 123])
        index.flush()
        queries, labels = np.asarray(queries), np.asarray(labels)
        with RemapBatchQueries([index], [None], workers=workers, radius=.2, matches=['instance']) as pool:
            for number, xyz, models in pool.map((i, queries, {'instance': labels}) for i in range(3)):
                baseline, distances, values = models[0]
                assert baseline is None
                assert values['score'].tolist() == [101, 999, 0] * 4
                assert np.isfinite(distances[labels != 123]).all()
                assert np.isinf(distances[labels == 123]).all()
