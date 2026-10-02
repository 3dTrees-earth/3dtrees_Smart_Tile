"""3DT-2209: accepted groups preserve member geometry through final remap."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import laspy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from remap_first_pipeline import merge_collections
from test_merge_stages import write_cloud


class RecoveredMergePipelineTests(unittest.TestCase):
    def test_independent_recovered_tips_survive_transitive_merge(self):
        for workers in (1, 4):
            with self.subTest(workers=workers), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); source = root / 'source'; source.mkdir()
                write_cloud(source / 'a.las', [.9], [1], [2])
                write_cloud(source / 'b.las', [.9, 1.05], [2, 2], [3, 3])
                write_cloud(source / 'c.las', [.9, 1.15], [3, 3], [4, 4])
                originals = root / 'originals'; originals.mkdir()
                write_cloud(originals / 'raw.las', [.9, 1.05, 1.15])
                regions = [dict(core=[[lo, hi], [-1, 1]], layout_tile=i,
                    neighbors=dict(west=i-1 if i else None, east=i+1 if i<2 else None,
                                   north=None, south=None))
                    for i, (lo, hi) in enumerate([(-1, 1), (1, 3), (3, 5)])]
                overlap = (np.array([-10., -10.]), np.array([10., 10.]))
                overlaps = [{j: overlap for j in range(i)} for i in range(3)]
                bounds = root / 'bounds.json'; bounds.write_text('{}')
                # Explicit metadata isolates the real recovery -> reconciliation ->
                # group-union -> shared-point -> deduplication chain from layout matching.
                with patch('remap_first_pipeline.ownership_regions', return_value=regions), \
                     patch('remap_first_pipeline.tile_overlaps', return_value=overlaps):
                    report = merge_collections(collections=[source], target_dir=None,
                        output_tiles=root / 'out', tile_bounds_json=bounds, ready=True,
                        workers=workers, originals=originals)
                self.assertEqual(report['state'], 'validated')
                model = report['models'][0]
                self.assertEqual(model['orphan_recovery']['final_support']['missing_samples'], 0)
                self.assertEqual(len(model['orphan_recovery']['admitted']), 2)
                self.assertEqual(model['reconciliation']['groups'], 1)
                clouds = [laspy.read(p) for p in sorted((root / 'out').glob('*.laz'))]
                points = [(float(x), int(uid), int(sem)) for c in clouds
                          for x, uid, sem in zip(c.x, c.PredInstance, c.PredSemantic)]
                tips = {round(x, 2): (uid, sem) for x, uid, sem in points}
                self.assertEqual(set(tips), {.9, 1.05, 1.15})
                self.assertEqual(len(points), 3)
                self.assertEqual(len({uid for uid, _ in tips.values()}), 1)
                self.assertEqual(tips[.9][1], 2)
                self.assertEqual((tips[1.05][1], tips[1.15][1]), (3, 4))
                # The recovered tips must remain queryable in the published
                # product, not merely appear in intermediate instance metadata.
                enriched = laspy.read(root/'original_with_predictions/raw.las')
                np.testing.assert_array_equal(enriched.PredInstance, [1, 1, 1])
                np.testing.assert_array_equal(enriched.PredSemantic, [2, 3, 4])
                np.testing.assert_array_equal(enriched.X, laspy.read(originals/'raw.las').X)

    def test_transitive_normal_group_keeps_all_unique_geometry(self):
        for workers in (1, 4):
            with self.subTest(workers=workers), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); source = root / 'source'; source.mkdir()
                write_cloud(source / 'a.las', [.5, 1.0], [1, 1], [2, 2])
                write_cloud(source / 'b.las', [1.0, 1.5], [2, 2], [3, 3])
                write_cloud(source / 'c.las', [1.5, 2.0], [3, 3], [4, 4])
                regions = [dict(core=[[lo, hi], [-1, 1]], layout_tile=i,
                    neighbors=dict(west=i-1 if i else None, east=i+1 if i<2 else None,
                                   north=None, south=None))
                    for i, (lo, hi) in enumerate([(-1, 1), (1, 1.6), (1.6, 3)])]
                overlap = (np.array([-10., -10.]), np.array([10., 10.]))
                overlaps = [{j: overlap for j in range(i)} for i in range(3)]
                originals = root / 'originals'; originals.mkdir()
                write_cloud(originals / 'raw.las', [.5, 1, 1.5, 2])
                bounds = root / 'bounds.json'; bounds.write_text('{}')
                with patch('remap_first_pipeline.ownership_regions', return_value=regions), \
                     patch('remap_first_pipeline.tile_overlaps', return_value=overlaps):
                    report = merge_collections(collections=[source], target_dir=None,
                        output_tiles=root / 'out', tile_bounds_json=bounds, ready=True,
                        workers=workers, originals=originals)
                model = report['models'][0]
                self.assertEqual(model['reconciliation']['groups'], 1)
                self.assertEqual({tuple(p['tiles']) for p in model['reconciliation']['accepted_pairs']},
                                 {(1, 0), (2, 1)})  # A and C have no direct correspondence.
                clouds = [laspy.read(p) for p in sorted((root/'out').glob('*.laz'))]
                self.assertEqual(sum(len(c.points) for c in clouds), 4)
                enriched = laspy.read(root/'original_with_predictions/raw.las')
                np.testing.assert_array_equal(enriched.PredInstance, [1, 1, 1, 1])
                np.testing.assert_array_equal(enriched.PredSemantic, [2, 2, 3, 4])

    def test_merged_tree_keeps_semantics_of_distinct_points_across_core_boundary(self):
        from test_dense_instance_ownership import core_layout
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / 'source'; source.mkdir()
            originals = root / 'originals'; originals.mkdir()
            write_cloud(source/'a.las', [.5, .999], [1, 1], [2, 3])
            write_cloud(source/'b.las', [1.001, 1.5], [2, 2], [8, 9])
            write_cloud(originals/'raw.las', [.5, .999, 1.001, 1.5])
            report = merge_collections(collections=[source], target_dir=None,
                output_tiles=root/'out', tile_bounds_json=core_layout(root), ready=True,
                originals=originals)
            self.assertEqual(report['models'][0]['reconciliation']['groups'], 1)
            cloud = laspy.read(root/'original_with_predictions/raw.las')
            np.testing.assert_array_equal(cloud.PredInstance, [1, 1, 1, 1])
            np.testing.assert_array_equal(cloud.PredSemantic, [2, 3, 8, 9])
