import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import laspy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from bounded_point_index import PointIndex
from strict_prediction_pipeline import merge_collections, strict_remap
from test_dense_tile_merge import write_cloud
from test_dense_instance_ownership import core_layout
from worker_budget import spatial_query_worker_count


class QueryBudgetTests(unittest.TestCase):
    def budget(self, requested, *, slots=None, affinity=8, quota='max 100000', parent_quota=None):
        contents = {'/proc/self/cgroup': '0::/jobs/test\n',
                    '/sys/fs/cgroup/cpu.max': 'max 100000',
                    '/sys/fs/cgroup/jobs/test/cpu.max': quota}
        if parent_quota is not None:
            contents['/sys/fs/cgroup/jobs/cpu.max'] = parent_quota
        def read(path, *args, **kwargs):
            if str(path) not in contents:
                raise FileNotFoundError(path)
            return contents[str(path)]
        with patch('worker_budget.os.cpu_count', return_value=64), \
             patch('worker_budget.os.sched_getaffinity', return_value=set(range(affinity))), \
             patch.dict('os.environ', {} if slots is None else {'GALAXY_SLOTS': str(slots)}, clear=True), \
             patch('worker_budget.Path.read_text', autospec=True, side_effect=read):
            return spatial_query_worker_count(requested)

    def test_request_scheduler_affinity_and_quota_all_bound_threads(self):
        self.assertEqual(self.budget(4), 4)
        self.assertEqual(self.budget(16, slots=6), 6)
        self.assertEqual(self.budget(16, slots=6, affinity=3), 3)
        self.assertEqual(self.budget(16, slots=6, quota='250000 100000'), 2)
        self.assertEqual(self.budget(16, quota='800000 100000', parent_quota='100000 100000'), 1)
        self.assertEqual(self.budget(4, quota='50000 100000'), 1)

    def test_invalid_explicit_budget_is_rejected(self):
        with self.assertRaises(ValueError):
            self.budget(0)
        with self.assertRaises(ValueError):
            self.budget(4, slots=0)


class ParallelQueryTests(unittest.TestCase):
    def test_nearest_ties_radius_and_non_nearest_conflicts_match_serial(self):
        rng = np.random.default_rng(512)
        xyz = rng.uniform(.1, .8, (8192, 3))
        xyz[1] = xyz[0]  # Equal-distance tie must choose the lowest source index.
        queries = np.vstack((xyz, [1.9, 1.9, 1.9]))
        outcomes = []
        with tempfile.TemporaryDirectory() as tmp, \
             patch('bounded_point_index.spatial_query_worker_count', side_effect=lambda n: n):
            for workers in [1, 4]:
                with PointIndex(Path(tmp) / f'{workers}.sqlite', {'id': np.uint32}, query_workers=workers) as index:
                    index.add(0, xyz, {'id': np.arange(len(xyz), dtype=np.uint32)}, np.arange(len(xyz)))
                    index.flush()
                    calls = []
                    native = index.query_tree
                    def tracked(tree, points, **kwargs):
                        class Tree:
                            def query(self, points, **options):
                                calls.append(options['workers'])
                                return tree.query(points, **options)
                        return native(Tree(), points, **kwargs)
                    with patch.object(index, 'query_tree', side_effect=tracked):
                        result = index.nearest(queries, .01)
                        # One other label lies within the radius of the shared point.
                        conflict = index.conflicting_match(
                            queries[:4096], np.zeros((4096, 1), dtype=np.uint32), .01,
                            before_tile=1, map_labels=lambda tile, values: (values['id'] > 0)[:, None])
                    self.assertIn(workers, calls)
                    self.assertEqual(result[2][0].tolist(), [0, 0])
                    self.assertTrue(np.isinf(result[0][-1]))
                    self.assertIsNotNone(conflict)
                    outcomes.append((result, conflict))
        for a, b in zip(outcomes[0][0], outcomes[1][0]):
            if isinstance(a, dict):
                np.testing.assert_array_equal(a['id'], b['id'])
            else:
                np.testing.assert_array_equal(a, b)
        self.assertEqual(outcomes[0][1], outcomes[1][1])

    def test_tiny_queries_do_not_spawn_extra_threads(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch('bounded_point_index.spatial_query_worker_count', return_value=8):
            with PointIndex(Path(tmp) / 'i.sqlite', {}, query_workers=8) as index:
                class Tree:
                    def query(self, xyz, **kwargs):
                        return kwargs['workers']
                self.assertEqual(index.query_tree(Tree(), np.zeros((8, 3))), 1)
                self.assertEqual(index.query_tree(Tree(), np.zeros((32768, 3))), 8)

    def test_merge_and_original_remap_match_with_four_query_workers(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch('bounded_point_index.spatial_query_worker_count', side_effect=lambda n: n), \
             patch('strict_prediction_pipeline.spatial_query_worker_count', side_effect=lambda n: n):
            root = Path(tmp)
            source, originals, targets = root / 'source', root / 'originals', root / 'targets'
            source.mkdir(); originals.mkdir(); targets.mkdir()
            shared = np.linspace(.9, 1.1, 8192)
            left = np.concatenate((np.linspace(.4, .6, 8192), shared))
            right = np.concatenate((shared, np.linspace(1.4, 1.6, 8192)))
            write_cloud(source / 'a.las', left, np.full(len(left), 7), np.full(len(left), 2))
            write_cloud(source / 'b.las', right, np.full(len(right), 38), np.full(len(right), 3))
            write_cloud(originals / 'raw.las', shared)
            write_cloud(targets / 'a.las', left)
            write_cloud(targets / 'b.las', right)
            results = []
            for workers in [1, 4]:
                dest = root / str(workers);dest.mkdir()
                report = merge_collections(collections=[source], target_dir=targets,
                    output_tiles=dest / 'out', tile_bounds_json=core_layout(root),
                    matching=False, workers=workers)
                self.assertEqual(report['parallelism']['query_workers'], workers)
                remap = strict_remap(collections=[dest / 'out'], originals=originals,
                    output=dest / 'enriched', workers=workers)
                self.assertEqual(remap['state'], 'validated')
                results.append([laspy.read(p).points.array for p in
                    [dest / 'out/tile_00000.laz', dest / 'out/tile_00001.laz', dest / 'enriched/raw.las']])
            for a, b in zip(*results):
                np.testing.assert_array_equal(a, b)

    def test_cli_and_compatibility_merge_forward_worker_budget(self):
        from parameters import Parameters
        import run
        from main_merge import run_merge
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);source = root / 'source';source.mkdir()
            write_cloud(source / 'a.las', [.5], [7])
            layout = core_layout(root)
            for task in ['merge', 'filter']:
                params = Parameters(task=task, input_dir=source, output_dir=root / 'out',
                    segmented_remapped_folder=source,
                    output_tiles_folder=root / 'out', tile_bounds_json=layout,
                    skip_merged_file=True, workers=4, _cli_parse_args=False)
                with patch('strict_prediction_pipeline.merge_collections', return_value={'state':'test'}) as merge:
                    getattr(run, f'run_{task}_task')(params)
                self.assertEqual(merge.call_args.kwargs['workers'], 4)
            with patch('strict_prediction_pipeline.merge_collections') as merge:
                run_merge(source, root / 'out', None, layout, num_threads=4, skip_merged_file=True)
            self.assertEqual(merge.call_args.kwargs['workers'], 4)


if __name__ == '__main__':
    unittest.main()
