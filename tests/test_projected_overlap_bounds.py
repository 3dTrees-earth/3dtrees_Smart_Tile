"""Projected GFZ bounds must retain their precision when converted to local XY."""
from contextlib import ExitStack
import json

import laspy
import numpy as np
import pytest

from bounded_point_index import PointIndex, coordinates, inside_xy
from dense_instance_ownership import assign_shared_points, preferred_core
from merge_stages import deduplicate, describe_model, index_file
from remap_first_pipeline import tile_overlaps
from spatial_query_cache import SpatialQueryCache
from test_overlap_roundoff import cloud


ORIGIN = np.array([398693.428, 5646177.674, 368.34000000000003])
BOUNDS = (398706.857, 398739.999, 5646175.349, 5646237.673)


def declared_overlaps(root, boxes, origin=ORIGIN):
    """Real LAS extent headers exercise the normal layout-matching path."""
    root.mkdir()
    pairs, tiles = [], []
    for i, (x0, x1, y0, y1) in enumerate(boxes):
        h = laspy.LasHeader(point_format=3, version='1.2')
        h.scales, h.offsets = np.full(3, .001), origin
        extent = laspy.LasData(h)
        extent.x, extent.y, extent.z = [x0, x1], [y0, y1], [0., 0.]
        path = root / f'extent_{i}.las'
        extent.write(path)
        pairs.append((path, path, path.stem))
        tiles.append({'col': i, 'row': 0, 'bounds': [[x0, x1], [y0, y1]]})
    layout = root / 'bounds.json'
    layout.write_text(json.dumps({'tiles': tiles}))
    return tile_overlaps(pairs, layout, origin)


@pytest.mark.parametrize('axis,side,raw', [(0, -1, [13429, 10000, 0]),
                                        (0, 1, [46571, 10000, 0]),
                                        (1, -1, [30000, -2325, 0]),
                                        (1, 1, [30000, 59999, 0])])
def test_projected_lower_upper_xy_bounds_and_true_outside(tmp_path, axis, side, raw):
    bounds = declared_overlaps(tmp_path / 'layout', [BOUNDS, BOUNDS])[1][0]
    point = cloud(ORIGIN, raw, .5)
    xyz = coordinates(point.points, point.header, ORIGIN)
    queries = np.repeat(xyz, 3, axis=0)
    queries[1, 2] = 1e12  # Z and query composition cannot change overlap membership.
    queries[2, axis] += side * .0001  # 0.1 mm is truly outside, not encoding roundoff.
    np.testing.assert_array_equal(inside_xy(queries, bounds), [True, True, False])
    assert [inside_xy(p[None, :], bounds)[0] for p in queries] == [True, True, False]


@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('gap', [0., .001])
def test_localization_does_not_connect_disjoint_or_edge_only_tiles(tmp_path, axis, gap):
    first = list(BOUNDS)
    second = list(BOUNDS)
    second[2 * axis] = first[2 * axis + 1] + gap
    second[2 * axis + 1] = second[2 * axis] + 1.
    assert declared_overlaps(tmp_path / 'layout', [first, second]) == [{}, {}]


@pytest.mark.parametrize('budget', [0, 32 * 1024**2])
@pytest.mark.parametrize('workers', [1, 4])
def test_gfz_global_xmax_ownership_conflicts_cache_disk_and_workers(tmp_path, budget, workers):
    source = tmp_path / 'source'
    source.mkdir()
    # Actual GFZ FM headers/coordinates: final IDs 64 and 63 wrongly survived
    # at this identical XYZ before overlap localization preserved global precision.
    inputs = [cloud(ORIGIN, [46571, -277, 31141], .25),
              cloud([398753.428, 5646177.674, ORIGIN[2]], [-13429, -277, 31141], .875)]
    inputs[0].PredInstance_FM = [78]
    files = [source / 'tile_00000.las', source / 'tile_00002.las']
    for data, path in zip(inputs, files):
        data.write(path)
    model = describe_model(source, 'PredInstance_FM')
    dims = {name: param.type for name, param in model.dimensions.items()}
    regions = [{'core': [[398663.428, 398723.428], [5646147.674, 5646207.674]]},
               {'core': [[398723.428, 398783.428], [5646147.674, 5646207.674]]}]
    overlaps = declared_overlaps(tmp_path / 'layout', [BOUNDS, BOUNDS])
    mapping = {(0, 78): 64, (1, 65): 63}
    query = coordinates(inputs[1].points, inputs[1].header, ORIGIN)
    assert preferred_core(query, 1, 0, regions, ORIGIN).all()
    report = {}
    with ExitStack() as stack:
        def index(name):
            return stack.enter_context(PointIndex(tmp_path / (name + '.db'), dims,
                query_workers=workers, query_cache=SpatialQueryCache(budget)))
        original, owned, survivors = index('source'), index('owned'), index('survivors')
        for tile, path in enumerate(files):
            index_file(original, path, tile, ORIGIN, model)
        # Both cache and disk must admit the source and query at the true edge.
        for nearest in (original.nearest, original._nearest_batches):
            distances, _, refs = nearest(query, .01, before_tile=1, overlaps=overlaps[1])
            assert np.isfinite(distances).all()
            assert refs.tolist() == [[0, 0]]
            distant = query + [0., 0., .02]
            assert not np.isfinite(nearest(distant, .01, before_tile=1, overlaps=overlaps[1])[0]).any()
        if budget:
            hits = original.query_cache.hits
            original.nearest(query, .01, before_tile=1, overlaps=overlaps[1])
            assert original.query_cache.hits > hits
        conflict = original.conflicting_match(query, np.array([[63, 2]]), .01, before_tile=1,
            map_labels=lambda tile, values: np.column_stack([
                np.full(len(values['PredInstance_FM']), 64), values['PredSemantic_FM']]),
            overlaps=overlaps[1])
        assert conflict is not None
        retained = assign_shared_points(model, files, original, regions, mapping,
            tmp_path / 'owned', owned, ORIGIN, report, overlaps=overlaps, workers=workers)
        assert [len(laspy.read(p).points) for p in retained] == [0, 1]
        final = deduplicate(model, retained, owned, survivors, ORIGIN, mapping,
            tmp_path / 'final', report, overlaps=overlaps, background_semantics_owned=True,
            core_regions=regions, workers=workers)
        assert sum(len(laspy.read(p).points) for p in final) == 1
        distances, values, refs = survivors.nearest(query, np.sqrt(3) * .01)
        assert np.isfinite(distances).all()
        assert refs.tolist() == [[1, 0]]
        assert values['PredInstance_FM'].tolist() == [63]
        assert values['PredSemantic_FM'].tolist() == [2]
        assert values['PredScore_FM'].tolist() == [.875]
