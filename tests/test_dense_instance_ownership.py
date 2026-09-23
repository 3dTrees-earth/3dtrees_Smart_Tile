import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import laspy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from strict_prediction_pipeline import merge_collections
from dense_instance_ownership import (owned_anchor, owned_background, ownership_regions,
                                      core_distance, preferred_core)
from test_dense_tile_merge import write_cloud


def core_layout(root):
    path = root / 'bounds.json'
    path.write_text(json.dumps({'tile_buffer': .2, 'tiles': [
        {'bounds': [[-1, 1.2], [-1, 1]], 'core': [[-1, 1], [-1, 1]], 'col': 0, 'row': 0},
        {'bounds': [[.8, 3], [-1, 1]], 'core': [[1, 3], [-1, 1]], 'col': 1, 'row': 0}]}))
    return path


class CoreOwnershipTests(unittest.TestCase):
    def test_3111_conflict_coordinate_uses_xy_core_distance(self):
        origin = np.array([342246.6488, 5724317.152, 0.])
        point = np.array([[342398.9624, 5724467.1791, 184.1956]]) - origin
        regions = [
            {'core': [[342096.6488, 342396.6488], [5724167.152, 5724467.152]]},
            {'core': [[342096.6488, 342396.6488], [5724467.152, 5724767.152]]},
            {'core': [[342396.6488, 342696.6488], [5724167.152, 5724467.152]]},
            {'core': [[342396.6488, 342696.6488], [5724467.152, 5724767.152]]},
        ]
        # Tile 3 contains the point but has no retained tree claim; compare the
        # actual positive claimants (0 and 2), not all layout tiles.
        self.assertEqual(core_distance(point, regions[3], origin)[0], 0)
        self.assertAlmostEqual(core_distance(point, regions[2], origin)[0], .0271)
        self.assertTrue(preferred_core(point, 2, 0, regions, origin)[0])
        # XY core distance ignores tree height and is not distance to a tile centre.
        regions = [{'core': [[-1, 1], [-1, 1]]}, {'core': [[2, 10], [-1, 1]]}]
        self.assertTrue(preferred_core(np.array([[1.8, 0, 1000.]]), 1, 0, regions, np.zeros(3))[0])

    def test_four_tile_zero_owner_recovers_one_whole_instance(self):
        for workers in (1, 4):
            from contextlib import ExitStack
            from bounded_point_index import PointIndex
            from dense_tile_merge import describe_model
            from dense_instance_ownership import filter_owned_instances
            from orphan_instance_recovery import recover_orphaned_instances, validate_recovered_geometry
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); source = root / 'source'; source.mkdir()
                tails = [(3, 3), (-1, -1), (-1, -1), (-1, -1)]
                files = [write_cloud(source / f'tile_{tile:05d}.las', [1.505 if tile == 1 else 1.5, tail[0]],
                        [tile + 1] * 2, [tile + 2] * 2, ys=[1.5, tail[1]])
                        for tile, tail in enumerate(tails)]
                cores = [([0, 1], [0, 1]), ([1, 2], [0, 1]),
                         ([0, 1], [1, 2]), ([1, 2], [1, 2])]
                neighbors = [dict(west=None, east=1, south=None, north=2),
                             dict(west=0, east=None, south=None, north=3),
                             dict(west=None, east=3, south=0, north=None),
                             dict(west=2, east=None, south=1, north=None)]
                regions = [dict(core=[list(x), list(y)], neighbors=n)
                           for (x, y), n in zip(cores, neighbors)]
                overlap = (np.array([-5., -5.]), np.array([5., 5.]))
                overlaps = [{j: overlap for j in range(tile)} for tile in range(4)]
                model = describe_model(source, 'PredInstance')
                dims = {name: p.type for name, p in model.dimensions.items()}
                report = {'tile_sources': [{'prediction': str(p)} for p in files]}
                with ExitStack() as stack:
                    owned = stack.enter_context(PointIndex(root / 'owned.sqlite', dims, query_workers=workers))
                    recovered = stack.enter_context(PointIndex(root / 'recovered.sqlite', dims, query_workers=workers))
                    filtered, counts = filter_owned_instances(model, files, regions,
                        root / 'filtered', owned, np.zeros(3), report)
                    self.assertEqual(sum(counts.values()), 0)
                    outputs, counts, selected, claims_path = recover_orphaned_instances(
                        model, files, filtered, owned, regions, overlaps, root / 'restored',
                        recovered, np.zeros(3), counts, report)
                    self.assertEqual(selected, [(0, 1)])
                    self.assertEqual(report['orphan_recovery']['uncovered_locations'], 2)
                    self.assertEqual(report['orphan_recovery']['admitted'][0]['new_locations'], 1)
                    self.assertTrue(report['instance_ownership']['tiles'][0]['instances'][0]['recovered'])
                    self.assertEqual(counts[0, 1], 2)
                    self.assertEqual(sum(np.count_nonzero(laspy.read(p).PredInstance > 0)
                                         for p in outputs), 2)
                    validate_recovered_geometry(recovered, model, selected, claims_path,
                                                report, np.zeros(3))
                    with PointIndex(root / 'empty.sqlite', dims) as empty:
                        with self.assertRaisesRegex(ValueError, 'recovered tree geometry lost'):
                            validate_recovered_geometry(empty, model, selected, claims_path,
                                                        report, np.zeros(3))

    def test_anchor_options_filter_whole_instances_before_conflict_checks(self):
        for anchor, expected in [('centroid', [2, 0]), ('highest_point', [0, 2]), ('lowest_point', [2, 0])]:
            with self.subTest(anchor=anchor), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source, originals = root / 'source', root / 'originals'
                source.mkdir(); originals.mkdir()
                # Opposing semantics must not conflict after one instance is removed.
                write_cloud(source / 'a.las', [.5, 1.1], [7, 7], [2, 2], zs=[0, 2])
                write_cloud(source / 'b.las', [.5, 1.1], [38, 38], [3, 3], zs=[0, 2])
                write_cloud(originals / 'raw.las', [.5, 1.1], zs=[0, 2])
                # Statistics must span chunks, including extreme-point selection.
                with patch('dense_instance_ownership.MAX_BATCH_POINTS', 1):
                    report = merge_collections(collections=[source], target_dir=None,
                        output_tiles=root / 'out', tile_bounds_json=core_layout(root), ready=True,
                        originals=originals, filter_anchor=anchor)
                self.assertEqual(report['state'], 'validated')
                clouds = [laspy.read(root / f'out/tile_{i:05d}.laz') for i in range(2)]
                self.assertEqual([len(c.points) for c in clouds], expected)
                owned = clouds[expected.index(2)]
                np.testing.assert_allclose(owned.x, [.5, 1.1])  # No clipping of the kept crown.
                np.testing.assert_array_equal(owned.intensity, [5, 6])
                expected_semantic = 3 if anchor == 'highest_point' else 2
                np.testing.assert_array_equal(owned.PredSemantic, [expected_semantic] * 2)
                enriched = laspy.read(root / 'original_with_predictions/raw.las')
                np.testing.assert_array_equal(enriched.PredSemantic, [expected_semantic] * 2)
                self.assertEqual([m['matched'] for m in report['original_coverage']], [2, 2])
                self.assertEqual(sum(t['removed'] for t in report['models'][0]['instance_ownership']['tiles']), 2)
                self.assertEqual([len(laspy.read(p).points) for p in sorted((root / 'out_unfiltered_1cm').glob('*.laz'))], [2, 2])

    def test_centroid_comes_from_dense_targets_not_coarse_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, targets = root / 'source', root / 'targets'
            source.mkdir(); targets.mkdir()
            for name in ['a', 'b']:
                write_cloud(source / f'{name}.las', [.8, 1.1], [7, 7])  # Coarse mean .95.
                write_cloud(targets / f'{name}.las', [.8, 1.09, 1.1, 1.11])  # Dense mean 1.025.
            report = merge_collections(collections=[source], target_dir=targets,
                output_tiles=root / 'out', tile_bounds_json=core_layout(root))
            ownership = report['models'][0]['instance_ownership']['tiles']
            self.assertEqual([t['surviving'] for t in ownership], [0, 4])
            self.assertAlmostEqual(ownership[0]['instances'][0]['anchor_xyz'][0], 1.025)

    def test_recovered_alias_with_extra_geometry_becomes_owner(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, originals = root / 'source', root / 'originals'
            source.mkdir(); originals.mkdir()
            write_cloud(source / 'a.las', [.9], [1])
            write_cloud(source / 'b.las', [.9, 1.05], [2, 2])
            write_cloud(originals / 'raw.las', [1.05])
            report = merge_collections(collections=[source], target_dir=None, output_tiles=root / 'out',
                tile_bounds_json=core_layout(root), ready=True, originals=originals)
            self.assertEqual(report['state'], 'validated')
            self.assertEqual(report['models'][0]['orphan_recovery']['admitted'][0]['local_instance'], 2)
            self.assertEqual(report['models'][0]['semantic_ownership']['instances'][0]['tile'], 1)
            enriched = laspy.read(root / 'original_with_predictions/raw.las')
            self.assertGreater(enriched.PredInstance[0], 0)

    def test_background_and_shared_tree_points_follow_core_ownership(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / 'source'; source.mkdir()
            write_cloud(source / 'a.las', [.5, 1.1], [0, 0])
            write_cloud(source / 'b.las', [.5, 1.1], [0, 0])
            report = merge_collections(collections=[source], target_dir=None,
                output_tiles=root / 'out', tile_bounds_json=core_layout(root), ready=True)
            self.assertEqual(sum(t['background_removed'] for t in report['models'][0]['instance_ownership']['tiles']), 2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / 'source'; source.mkdir()
            write_cloud(source / 'a.las', [.5, 1], [1, 1], [2, 2])
            write_cloud(source / 'b.las', [1, 1.5], [2, 2], [3, 3])
            report = merge_collections(collections=[source], target_dir=None,
                output_tiles=root / 'out', tile_bounds_json=core_layout(root), ready=True, matching=False)
            a, b = [laspy.read(root / f'out/tile_{i:05d}.laz') for i in range(2)]
            np.testing.assert_allclose(a.x, [.5, 1])
            np.testing.assert_allclose(b.x, [1.5])
            np.testing.assert_array_equal(a.PredSemantic, [2, 2])
            np.testing.assert_array_equal(b.PredSemantic, [3])

    def test_shared_tree_points_use_spatial_owner_preserving_unique_buffer_tails(self):
        for reverse in [False, True]:
            with self.subTest(reverse=reverse), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); source = root / 'source'; source.mkdir()
                originals = root / 'originals'; originals.mkdir()
                shared = [.9, .998, 1., 1.002, 1.1]
                left = [-1, -.8, *shared, 1.19]
                right = [.81, *shared, 2.8, 3.]
                names = ['b', 'a'] if reverse else ['a', 'b']
                write_cloud(source / f'{names[0]}.las', left, [7] * len(left), [2] * len(left))
                write_cloud(source / f'{names[1]}.las', right, [38] * len(right), [3] * len(right))
                write_cloud(originals / 'raw.las', shared + [.81, 1.19])
                with patch('dense_instance_ownership.MAX_BATCH_POINTS', 2):
                    report = merge_collections(collections=[source], target_dir=None,
                        output_tiles=root / 'out', tile_bounds_json=core_layout(root),
                        ready=True, matching=False, originals=originals)
                self.assertEqual(report['state'], 'validated')
                clouds = [laspy.read(root / f'out/tile_{i:05d}.laz') for i in range(2)]
                a, b = clouds[::-1] if reverse else clouds
                np.testing.assert_allclose(a.x, [-1, -.8, .9, .998, 1.19] if reverse else [-1, -.8, .9, .998, 1., 1.19])
                np.testing.assert_allclose(b.x, [.81, 1., 1.002, 1.1, 2.8, 3.] if reverse else [.81, 1.002, 1.1, 2.8, 3.])
                np.testing.assert_array_equal(a.PredSemantic, [2] * (5 if reverse else 6))
                np.testing.assert_array_equal(b.PredSemantic, [3] * (6 if reverse else 5))
                # Retain the complete source record of each surviving point.
                np.testing.assert_array_equal(a.intensity, [5, 6, 7, 8, 12] if reverse else [5, 6, 7, 8, 9, 12])
                enriched = laspy.read(root / 'original_with_predictions/raw.las')
                np.testing.assert_array_equal(enriched.PredSemantic, [2, 2, 3 if reverse else 2, 3, 3, 3, 2])
                self.assertNotEqual(int(a.PredInstance[0]), int(b.PredInstance[0]))
                self.assertEqual(sum(t['removed'] for t in report['models'][0]['shared_point_ownership']['tiles']), 5)

    def test_3111_three_tile_claims_choose_nearest_retained_tree_core(self):
        # Two retained trees extend into a third tile that predicts background.
        # Neither tree point is inside its own core (the 3111 failure pattern).
        for reverse in [False, True]:
            with self.subTest(reverse=reverse), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); source = root / 'source'; source.mkdir()
                originals = root / 'originals'; originals.mkdir()
                bounds = root / 'bounds.json'
                bounds.write_text(json.dumps({'tile_buffer': 1., 'tiles': [
                    {'bounds': [[-1, 2], [-1, 1]], 'core': [[-1, 1], [-1, 1]], 'col': 0, 'row': 0},
                    {'bounds': [[0, 3], [-1, 1]], 'core': [[1, 2], [-1, 1]], 'col': 1, 'row': 0},
                    {'bounds': [[1, 4], [-1, 1]], 'core': [[2, 4], [-1, 1]], 'col': 2, 'row': 0}]}))
                left_name, right_name = ('c', 'a') if reverse else ('a', 'c')
                shared = [1.2, 1.5, 1.8]
                write_cloud(source / f'{left_name}.las', [-1, -.8, *shared, 1.35],
                            [7] * 6, [2, 2, 3, 4, 5, 6])
                write_cloud(source / 'b.las', [0, *shared, 3], [0] * 5, [9] * 5)
                write_cloud(source / f'{right_name}.las', [*shared, 3.8, 4],
                            [38] * 5, [11, 12, 13, 14, 14])
                write_cloud(originals / 'raw.las', shared + [1.35])
                with patch('dense_instance_ownership.MAX_BATCH_POINTS', 2):
                    report = merge_collections(collections=[source], target_dir=None,
                        output_tiles=root / 'out', tile_bounds_json=bounds,
                        ready=True, matching=False, originals=originals)
                self.assertEqual(report['state'], 'validated')
                clouds = [laspy.read(p) for p in sorted((root / 'out').glob('*.laz'))]
                left, right = (clouds[2], clouds[0]) if reverse else (clouds[0], clouds[2])
                enriched = laspy.read(root / 'original_with_predictions/raw.las')
                np.testing.assert_array_equal(enriched.PredSemantic, [3, 12 if reverse else 4, 13, 6])
                left_id, right_id = int(left.PredInstance[0]), int(right.PredInstance[-1])
                self.assertNotEqual(left_id, right_id)
                np.testing.assert_array_equal(enriched.PredInstance,
                    [left_id, right_id if reverse else left_id, right_id, left_id])
                for x in shared:
                    self.assertEqual(sum(np.count_nonzero(np.isclose(c.x, x)) for c in clouds), 1)
                np.testing.assert_allclose(left.x, [-1, -.8, 1.2, 1.35] if reverse else [-1, -.8, 1.2, 1.5, 1.35])
                np.testing.assert_array_equal(left.intensity, [5, 6, 7, 10] if reverse else [5, 6, 7, 8, 10])
                self.assertEqual(sum(t['removed'] for t in report['models'][0]['shared_point_ownership']['tiles']), 3)
                self.assertEqual(sum(t['removed'] for t in report['models'][0]['tree_background_ownership']['tiles']), 3)
                self.assertEqual([m['matched'] for m in report['original_coverage']], [4, 4])

    def test_nearby_buffer_points_across_core_bisector_keep_separate_owners(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / 'source'; source.mkdir()
            originals = root / 'originals'; originals.mkdir()
            bounds = root / 'bounds.json'
            bounds.write_text(json.dumps({'tile_buffer': 1., 'tiles': [
                {'bounds': [[-1, 2], [-1, 1]], 'core': [[-1, 1], [-1, 1]], 'col': 0, 'row': 0},
                {'bounds': [[0, 3], [-1, 1]], 'core': [[1, 2], [-1, 1]], 'col': 1, 'row': 0},
                {'bounds': [[1, 4], [-1, 1]], 'core': [[2, 4], [-1, 1]], 'col': 2, 'row': 0}]}))
            write_cloud(source / 'a.las', [-1, -.8, 1.498], [7] * 3, [2] * 3)
            write_cloud(source / 'b.las', [0, 1.498, 1.502, 3], [0] * 4, [9] * 4)
            write_cloud(source / 'c.las', [1.502, 3.8, 4], [38] * 3, [3] * 3)
            write_cloud(originals / 'raw.las', [1.498, 1.502])
            report = merge_collections(collections=[source], target_dir=None,
                output_tiles=root / 'out', tile_bounds_json=bounds,
                ready=True, matching=False, originals=originals)
            self.assertEqual(report['state'], 'validated')
            enriched = laspy.read(root / 'original_with_predictions/raw.las')
            np.testing.assert_array_equal(enriched.PredSemantic, [2, 3])
            self.assertEqual(sum(t['removed'] for t in report['models'][0]['shared_point_ownership']['tiles']), 0)

    def test_retained_tree_beats_neighbor_background_with_its_own_semantics(self):
        for reverse in [False, True]:
            with self.subTest(reverse=reverse), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); source = root / 'source'; source.mkdir()
                originals = root / 'originals'; originals.mkdir()
                names = ['b', 'a'] if reverse else ['a', 'b']
                write_cloud(source / f'{names[0]}.las', [-1, .5, 1.1, 1.19], [7] * 4, [2, 3, 4, 5])
                write_cloud(source / f'{names[1]}.las', [.8, 1.1, 1.115, 3], [0] * 4, [9] * 4)
                write_cloud(originals / 'raw.las', [1.1, 1.115, 1.19, 3])
                with patch('dense_instance_ownership.MAX_BATCH_POINTS', 1):
                    report = merge_collections(collections=[source], target_dir=None,
                        output_tiles=root / 'out', tile_bounds_json=core_layout(root),
                        ready=True, matching=False, originals=originals)
                self.assertEqual(report['state'], 'validated')
                enriched = laspy.read(root / 'original_with_predictions/raw.las')
                np.testing.assert_array_equal(enriched.PredSemantic, [4, 9, 5, 9])
                self.assertGreater(enriched.PredInstance[0], 0)
                self.assertEqual(enriched.PredInstance[1], 0)
                clouds = [laspy.read(p) for p in sorted((root / 'out').glob('*.laz'))]
                self.assertEqual(sum(np.count_nonzero(np.isclose(c.x, 1.1)) for c in clouds), 1)
                tree = next(c for c in clouds if np.any(c.PredInstance > 0))
                np.testing.assert_array_equal(tree.intensity, [5, 6, 7, 8])
                self.assertEqual(sum(t['removed'] for t in report['models'][0]['tree_background_ownership']['tiles']), 1)

    def test_orphaned_tree_beats_neighbor_background(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / 'source'; source.mkdir()
            originals = root / 'originals'; originals.mkdir()
            write_cloud(source / 'a.las', [1.1, 1.15], [7, 7], [4, 4])
            write_cloud(source / 'b.las', [.8, 1.1, 3], [0, 0, 0], [9, 9, 9])
            write_cloud(originals / 'raw.las', [1.1])
            report = merge_collections(collections=[source], target_dir=None,
                output_tiles=root / 'out', tile_bounds_json=core_layout(root),
                ready=True, matching=False, originals=originals)
            enriched = laspy.read(root / 'original_with_predictions/raw.las')
            self.assertGreater(enriched.PredInstance[0], 0)
            np.testing.assert_array_equal(enriched.PredSemantic, [4])
            self.assertEqual(report['models'][0]['orphan_recovery']['admitted'][0]['tile'], 0)

    def test_declared_neighbors_apply_even_when_only_one_tile_is_processed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = write_cloud(root / 'a.las', [-1, 1.2], [1, 1])
            region = ownership_regions([(source, source, 'a')], core_layout(root))[0]
            self.assertFalse(owned_anchor([1.1, 0], region))
            self.assertTrue(owned_anchor([1, 0], region))  # Legacy inclusive boundary.
            self.assertTrue(owned_anchor([-2, 0], region))  # No neighbor on dataset exterior.

    def test_background_semantics_follow_spatial_owner_including_shared_edge(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / 'source'; source.mkdir()
            originals = root / 'originals'; originals.mkdir()
            # Nearby points on opposite sides keep different classes; the exact
            # shared-edge point is assigned only to the right-hand tile.
            xs = [.998, 1., 1.002]
            write_cloud(source / 'a.las', [-1, *xs, 1.2], [0] * 5, [1] * 5)
            write_cloud(source / 'b.las', [.8, *xs, 3], [0] * 5, [0] * 5)
            write_cloud(originals / 'raw.las', xs)
            report = merge_collections(collections=[source], target_dir=None,
                output_tiles=root / 'out', tile_bounds_json=core_layout(root),
                ready=True, originals=originals)
            self.assertEqual(report['state'], 'validated')
            a, b = [laspy.read(root / f'out/tile_{i:05d}.laz') for i in range(2)]
            np.testing.assert_allclose(a.x, [-1, .998])
            np.testing.assert_array_equal(a.PredSemantic, [1, 1])
            np.testing.assert_allclose(b.x, [1., 1.002, 3])
            np.testing.assert_array_equal(b.PredSemantic, [0, 0, 0])
            enriched = laspy.read(root / 'original_with_predictions/raw.las')
            np.testing.assert_array_equal(enriched.PredSemantic, [1, 0, 0])
            self.assertEqual(sum(t['background_removed'] for t in report['models'][0]['instance_ownership']['tiles']), 5)

    def test_reconciled_instance_keeps_one_core_owners_semantics(self):
        for include_tail in [False, True]:
            with self.subTest(include_tail=include_tail), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); source = root / 'source'; source.mkdir()
                originals = root / 'originals'; originals.mkdir()
                write_cloud(source / 'a.las', [.5, 1], [1, 1], [2, 3])
                write_cloud(source / 'b.las', [1, 1.5], [7, 7], [9, 9])
                write_cloud(originals / 'raw.las', [.5, 1, 1.5] if include_tail else [.5, 1])
                kwargs = dict(collections=[source], target_dir=None, output_tiles=root / 'out',
                              tile_bounds_json=core_layout(root), ready=True, originals=originals)
                if include_tail:
                    with self.assertRaisesRegex(ValueError, '100% original coverage'):
                        merge_collections(**kwargs)
                    self.assertFalse((root / 'out').exists())
                else:
                    report = merge_collections(**kwargs)
                    enriched = laspy.read(root / 'original_with_predictions/raw.las')
                    np.testing.assert_array_equal(enriched.PredSemantic, [2, 3])
                    self.assertEqual(len(laspy.read(root / 'out/tile_00001.laz').points), 0)
                    self.assertEqual(report['models'][0]['semantic_ownership']['instances'][0]['tile'], 0)

    def test_background_subset_does_not_take_missing_neighbors_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = write_cloud(root / 'a.las', [-1, 1.2], [0, 0])
            region = ownership_regions([(source, source, 'a')], core_layout(root))[0]
            np.testing.assert_array_equal(owned_background(np.array([[.9, 0], [1, 0], [1.1, 0], [-2, 0]]), region),
                                          [True, False, False, True])

    def test_cli_passes_anchor_for_both_merge_and_filter(self):
        from parameters import Parameters
        import run
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / 'source'; source.mkdir()
            write_cloud(source / 'a.las', [.9], [1])
            for task in ['merge', 'filter']:
                params = Parameters(task=task, input_dir=source, output_dir=root / 'out',
                    segmented_remapped_folder=source, output_tiles_folder=root / 'out',
                    tile_bounds_json=core_layout(root), filter_anchor='highest_point',
                    skip_merged_file=True, _cli_parse_args=False)
                with patch('strict_prediction_pipeline.merge_collections', return_value={'state': 'test'}) as merge:
                    getattr(run, f'run_{task}_task')(params)
                self.assertEqual(merge.call_args.kwargs['filter_anchor'], 'highest_point')


if __name__ == '__main__':
    unittest.main()
