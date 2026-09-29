import json
import sys
from pathlib import Path

import laspy
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from instance_finalization import compact_originals
from get_bounds_from_tindex import build_tiles
from strict_prediction_pipeline import strict_remap
from test_dense_tile_merge import write_cloud


def test_compaction_across_files_models_and_background(tmp_path):
    originals = tmp_path / 'originals'
    originals.mkdir()
    write_cloud(originals / 'a.las', [0, 1])
    write_cloud(originals / 'b.las', [2, 1])
    models = []
    for name, ids in [('SAT', [91, 300, 0]), ('FM', [0, 91, 700]), ('THIRD', [0, 0, 0])]:
        folder = tmp_path / name
        folder.mkdir()
        write_cloud(folder / 'tile.las', [0, 1, 2], ids, [3, 4, 5], instance=f'PredInstance_{name}')
        models.append(folder)
    report = strict_remap(collections=models, baseline_collections=[models[0]],
                          originals=originals, output=tmp_path / 'final')
    expected = {'a.las': [[1, 2], [0, 1], [0, 0]], 'b.las': [[0, 2], [2, 1], [0, 0]]}
    for filename, labels in expected.items():
        cloud = laspy.read(tmp_path / 'final' / filename)
        source = laspy.read(originals / filename)
        for name in source.point_format.dimension_names:
            np.testing.assert_array_equal(cloud[name], source[name])
        for model, ids in zip(['SAT', 'FM', 'THIRD'], labels):
            assert cloud[f'PredInstance_{model}'].tolist() == ids
            assert cloud[f'PredInstance_{model}'].dtype == np.uint32
            assert cloud[f'PredSemantic_{model}'].tolist() == ([3, 4] if filename == 'a.las' else [5, 4])
    mapping = json.loads((tmp_path / 'final/instance_mapping.json').read_text())
    assert mapping['models']['PredInstance_SAT']['instances'] == [
        {'source_instance_id': 91, 'instance_id': 1}, {'source_instance_id': 300, 'instance_id': 2}]
    assert report['instance_finalization']['counts']['PredInstance_THIRD'] == 0


def test_compaction_rejects_unexpected_labels(tmp_path):
    write_cloud(tmp_path / 'a.las', [0], [99], [3])
    before = (tmp_path / 'a.las').read_bytes()
    with pytest.raises(ValueError, match='IDs changed'):
        compact_originals(tmp_path, {'PredInstance': {'a.las': {7}}})
    assert (tmp_path / 'a.las').read_bytes() == before
    assert not (tmp_path / '.compact-a.las').exists()


def test_explicit_grid_has_four_cores_and_complete_coverage():
    extent = (398706.857, 5646175.349, 398739.999, 5646239.999)
    origin = (398663.428, 5646147.674)
    tiles, bounds = build_tiles(*extent, 60, 20, grid_origin=origin)
    assert len(tiles) == 4
    for tile in tiles:
        (x0, x1), (y0, y1) = tile['core']
        assert x0 < extent[2] and x1 > extent[0]
        assert y0 < extent[3] and y1 > extent[1]
        assert tile['bounds'] == [[x0 - 20, x1 + 20], [y0 - 20, y1 + 20]]
    with pytest.raises(ValueError, match='exclude'):
        build_tiles(*extent, 60, 20, grid_origin=(extent[0] + 1, extent[1]))
