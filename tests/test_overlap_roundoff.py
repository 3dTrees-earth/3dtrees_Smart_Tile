"""3DT-2209: real GFZ boundary duplicates must keep the core owner's score."""
from contextlib import ExitStack
from pathlib import Path

import laspy
import numpy as np
import pytest

from bounded_point_index import PointIndex, coordinates, inside_xy
from dense_instance_ownership import assign_shared_points, preferred_core
from dense_tile_merge import describe_model, index_file, deduplicate
from spatial_query_cache import SpatialQueryCache


def cloud(offsets, raw, score):
    header = laspy.LasHeader(point_format=3, version='1.2')
    header.scales = np.full(3, .001)
    header.offsets = np.asarray(offsets)
    for name, dtype in [('PredInstance_FM', np.uint32), ('PredSemantic_FM', np.int32),
                        ('PredScore_FM', np.float32)]:
        header.add_extra_dim(laspy.ExtraBytesParams(name=name, type=dtype))
    result = laspy.LasData(header)
    for axis, value in zip(('X', 'Y', 'Z'), raw):
        result[axis] = [value]
    result.PredInstance_FM = [65]
    result.PredSemantic_FM = [2]
    result.PredScore_FM = [score]
    return result


def test_overlap_roundoff_is_xy_only_and_independent_of_query_batch():
    bounds = (np.array([10., -50.]), np.array([50., -10.]))
    # Include the immediately adjacent representable floats on every edge.
    points = np.array([[np.nextafter(10., -np.inf), -30., 0.],
                       [np.nextafter(50., np.inf), -30., 0.],
                       [30., np.nextafter(-50., -np.inf), 0.],
                       [30., np.nextafter(-10., np.inf), 0.],
                       [10. - .0001, -30., 1e12],
                       [50. + .0001, -30., 0.]])
    expected = [True, True, True, True, False, False]
    np.testing.assert_array_equal(inside_xy(points, bounds), expected)
    assert [inside_xy(p[None, :], bounds)[0] for p in points] == expected
    assert inside_xy(np.empty((0, 3)), bounds).size == 0


@pytest.mark.parametrize('budget', [0, 32 * 1024**2])
def test_cached_and_disk_overlap_include_rounded_source_edge(tmp_path, budget):
    bounds = (np.array([0., 10.]), np.array([50., 50.]))
    with PointIndex(tmp_path/'source.db', {'score': np.float32},
                    query_cache=SpatialQueryCache(budget)) as index:
        index.add(0, np.array([[41.893, np.nextafter(10., -np.inf), 42.763]]),
                  {'score': np.array([.8], np.float32)}, np.array([0]))
        index.flush()
        distances, values, refs = index.nearest(np.array([[41.893, 10., 42.763]]),
                                               .01, overlaps={0: bounds})
        assert np.isfinite(distances).all()
        assert refs.tolist() == [[0, 0]]
        assert values['score'][0] == np.float32(.8)


@pytest.mark.parametrize('budget', [0, 32 * 1024**2])
@pytest.mark.parametrize('workers', [1, 4])
def test_gfz_boundary_ownership_dedup_and_remap_keep_core_score(tmp_path, budget, workers):
    source = tmp_path/'source'
    source.mkdir()
    origin = np.array([398693.428, 5646177.674, 368.34000000000003])
    inputs = [cloud([398693.428, 5646237.674, 368.34000000000003],
                    [41893, -50000, 42763], .6418741345405579),
              cloud([398753.428, 5646177.674, 368.34000000000003],
                    [-18107, 10000, 42763], .8754913806915283)]
    files = [source/'tile_00001.las', source/'tile_00002.las']
    for data, path in zip(inputs, files):
        data.write(path)
    model = describe_model(source, 'PredInstance_FM')
    dims = {name: param.type for name, param in model.dimensions.items()}
    regions = [{'core': [[398663.428, 398723.428], [5646207.674, 5646267.674]]},
               {'core': [[398723.428, 398783.428], [5646147.674, 5646207.674]]}]
    overlaps = [{}, {0: (np.array([14.15700000000652, 10.]),
                         np.array([46.570999999996275, 50.]))}]
    mapping = {(0, 65): 65, (1, 65): 65}
    report = {}
    # Both are already retained members of the same accepted group; the real
    # bug occurs after core filtering/reconciliation, before original remap.
    with ExitStack() as stack:
        def index(name):
            return stack.enter_context(PointIndex(tmp_path/(name+'.db'), dims,
                query_workers=workers, query_cache=SpatialQueryCache(budget)))
        original, owned, survivors = index('source'), index('owned'), index('survivors')
        for tile, path in enumerate(files):
            index_file(original, path, tile, origin, model)
        retained = assign_shared_points(model, files, original, regions, mapping,
            tmp_path/'owned', owned, origin, report, overlaps=overlaps)
        assert [len(laspy.read(p).points) for p in retained] == [0, 1]
        final = deduplicate(model, retained, owned, survivors, origin, mapping,
            tmp_path/'final', report, overlaps=overlaps, background_semantics_owned=True,
            core_preferred=lambda a, b, pts: preferred_core(pts, a, b, regions, origin))
        assert sum(len(laspy.read(p).points) for p in final) == 1
        query = cloud([399157.564, 5646649.995, 368.34000000000003],
                      [-422243, -462321, 42763], 0.)
        distances, values, refs = survivors.nearest(
            coordinates(query.points, query.header, origin), np.sqrt(3)*.01)
        assert np.isfinite(distances).all()
        assert refs[0, 0] == 1
        assert values['PredInstance_FM'][0] == 65
        assert values['PredSemantic_FM'][0] == 2
        assert values['PredScore_FM'][0] == np.float32(.8754913806915283)
