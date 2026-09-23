"""Regressions for historical whole-cloud predictions with an unused tile plan."""
import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import laspy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from dense_instance_ownership import ownership_regions, owned_anchor
from strict_prediction_pipeline import merge_collections


# Actual header extents from the failed 2590 / 2594 / 2595 replays.
EXTENTS = [(610094.22, 610687.57, 4494701.75, 4495359.23),
           (602749.91, 603320.19, 4501639., 4502167.21),
           (610764.09, 611273.39, 4502346.37, 4502875.6)]


def legacy_layout(bounds):
    x0, x1, y0, y1 = bounds
    tiles = []
    for col in range(math.ceil((x1 - x0) / 300)):
        for row in range(math.ceil((y1 - y0) / 300)):
            x, y = x0 + 300 * col, y0 + 300 * row
            planned = [[x - 20, x + 320], [y - 20, y + 320]]
            tiles.append(dict(col=col, row=row, core=[[x, x + 300], [y, y + 300]],
                              bounds=planned, planned_bounds=planned))
    return dict(tile_buffer=20, proj_extent=dict(minx=x0, maxx=x1, miny=y0, maxy=y1), tiles=tiles)


def cloud(path, bounds, predictions=False):
    x0, x1, y0, y1 = bounds
    header = laspy.LasHeader(point_format=3, version='1.2')
    header.scales = np.array([.01] * 3)
    header.offsets = np.array([x0, y0, 0])
    data = laspy.LasData(header)
    data.x, data.y, data.z = [x0, (x0 + x1) / 2, x1], [y0, (y0 + y1) / 2, y1], [0, 1, 2]
    if predictions:
        data.add_extra_dim(laspy.ExtraBytesParams(name='PredInstance', type=np.uint32))
        data.add_extra_dim(laspy.ExtraBytesParams(name='PredSemantic', type=np.uint8))
        data.PredInstance, data.PredSemantic = [0, 7, 7], [1, 2, 3]
    data.write(path)
    return path


class LegacyBypassLayoutTests(unittest.TestCase):
    def test_three_failed_layouts_recover_whole_cloud_core(self):
        for bounds in EXTENTS:
            with self.subTest(bounds=bounds), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                layout = root / 'bounds.json'; layout.write_text(json.dumps(legacy_layout(bounds)))
                original = layout.read_bytes()
                target = cloud(root / 'target.las', bounds)
                region = ownership_regions([(target, target, 'single')], layout)[0]
                x0, x1, y0, y1 = bounds
                np.testing.assert_allclose(region['core'], [[x0, x1], [y0, y1]], rtol=0, atol=1e-8)
                self.assertTrue(all(v is None for v in region['neighbors'].values()))
                self.assertEqual(region['layout_recovery']['reason'], 'legacy_single_cloud_bypass')
                self.assertEqual(region['layout_recovery']['planned_tile_count'], len(legacy_layout(bounds)['tiles']))
                self.assertEqual(layout.read_bytes(), original)

    def test_merge_retains_geometry_and_semantics_across_unused_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / 'source'; target = root / 'target'
            source.mkdir(); target.mkdir()
            bounds = EXTENTS[0]
            cloud(source / '8645.las', bounds, predictions=True)
            cloud(target / '8645.las', bounds)
            layout = root / 'bounds.json'; layout.write_text(json.dumps(legacy_layout(bounds)))
            report = merge_collections(collections=[source], target_dir=target,
                output_tiles=root / 'out', tile_bounds_json=layout, originals=target)
            self.assertEqual(report['state'], 'validated')
            result = laspy.read(root / 'out/tile_00000.laz')
            np.testing.assert_array_equal(result.PredSemantic, [1, 2, 3])
            np.testing.assert_array_equal(result.X, laspy.read(target / '8645.las').X)
            np.testing.assert_array_equal(result.Y, laspy.read(target / '8645.las').Y)
            self.assertEqual(report['models'][0]['instance_ownership']['tiles'][0]['removed'], 0)
            effective = json.loads((root / 'out/tile_bounds.json').read_text())
            self.assertEqual(len(effective['tiles']), 1)
            self.assertEqual(effective['tile_buffer'], 0)
            self.assertTrue(effective['tiling_skipped'])
            self.assertEqual(len(json.loads(layout.read_text())['tiles']), 6)
            from hashlib import sha256
            manifest = json.loads((root / 'out/smarttile_merge.json').read_text())
            self.assertEqual(manifest['tile_bounds_sha256'], sha256((root / 'out/tile_bounds.json').read_bytes()).hexdigest())
            self.assertEqual(effective['layout_recovery']['source_sha256'], sha256(layout.read_bytes()).hexdigest())


    def test_partial_collection_keeps_declared_missing_neighbors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); data = legacy_layout(EXTENTS[0])
            layout = root / 'bounds.json'; layout.write_text(json.dumps(data))
            (x0, x1), (y0, y1) = data['tiles'][0]['bounds']
            target = cloud(root / 'target.las', (x0, x1, y0, y1))
            region = ownership_regions([(target, target, 'one_tile')], layout)[0]
            self.assertNotIn('layout_recovery', region)
            self.assertIsNotNone(region['neighbors']['east'])
            self.assertFalse(owned_anchor([x1, y0 + 30], region))

    def test_unrelated_or_updated_metadata_is_not_recovered(self):
        for change in ['wrong_extent', 'actual_tiles', 'no_extent', 'nan_extent']:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); data = copy.deepcopy(legacy_layout(EXTENTS[0]))
                if change == 'wrong_extent': data['proj_extent']['maxx'] += 1
                if change == 'actual_tiles': data['tiles'][0]['bounds'][0][0] += 1; data['tiles'][0]['planned_bounds'] = [[0, 1], [0, 1]]
                if change == 'no_extent': del data['proj_extent']
                if change == 'nan_extent': data['proj_extent']['maxx'] = float('nan')
                layout = root / 'bounds.json'; layout.write_text(json.dumps(data))
                target = cloud(root / 'target.las', EXTENTS[0])
                with self.assertRaisesRegex(ValueError, 'Failed to match all tiles'):
                    ownership_regions([(target, target, 'single')], layout)
