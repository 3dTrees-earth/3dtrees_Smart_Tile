import json
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import laspy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from strict_prediction_pipeline import merge_collections, strict_remap
from test_dense_tile_merge import write_cloud
from test_strict_prediction_pipeline import add_extended_metadata, assert_extended_metadata
from raycloud_instance_ids import encode_instance_ids, read_tile_namespace
from raycloud_tree_files import filter_tree_sidecars


class RayCloudFilterOnlyTests(unittest.TestCase):
    def _fixture(self, root):
        source = root / "rct"
        source.mkdir()
        write_cloud(source / "c00_r00_segmented.las", [1, 9.5, 10.5, 10.6], [1, 1, 2, 2],
                    instance="PredInstance_RCT")
        write_cloud(source / "c01_r00_segmented.las", [10.5, 11, 9.5, 9.6], [1, 1, 2, 2],
                    instance="PredInstance_RCT")
        for tile in ("c00_r00", "c01_r00"):
            for suffix in ("trees", "trees_info"):
                (source / f"{tile}_{suffix}.txt").write_text(
                    "# RCT tree table\nheader\ntree-1\ntree-2\n", encoding="utf-8")
        layout = root / "bounds.json"
        layout.write_text(json.dumps({"tile_buffer": 1, "tiles": [
            {"bounds": [[-1, 11], [-1, 1]], "core": [[-1, 10], [-1, 1]], "col": 0, "row": 0},
            {"bounds": [[9, 21], [-1, 1]], "core": [[10, 21], [-1, 1]], "col": 1, "row": 0},
        ]}))
        return source, layout

    def test_encodes_tile_ids_and_filters_both_tree_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, layout = self._fixture(root)
            report = merge_collections(
                collections=[source], target_dir=None, output_tiles=root / "output_tiles",
                tile_bounds_json=layout, ready=True, instance_dimension="PredInstance_RCT")
            self.assertIn("filter-only", report["contract"])
            self.assertEqual(report["models"][0]["reconciliation"]["accepted_pairs"], [])
            self.assertEqual((root / "output_tiles" / "instance_metadata.csv").read_text().splitlines(),
                             ["tile,tile_id,local_instance_id,PredInstance_RCT,has_added_clusters",
                              "0,1,1,100001,0", "1,2,1,200001,0"])
            summary = json.loads((root / "output_tiles" / "PredInstance_RCT_summary.json").read_text())
            self.assertEqual(summary["id_encoding"], "tile_id * 100000 + local_id")
            self.assertEqual([(i["id"], i["points"]) for i in summary["instances"]], [(100001, 2), (200001, 2)])
            self.assertEqual(json.loads((root / "output_tiles" / "smarttile_merge.json").read_text())
                             ["instance_summary"], "PredInstance_RCT_summary.json")
            for tile in range(2):
                output = laspy.read(root / "output_tiles" / f"tile_{tile:05d}.laz")
                self.assertEqual(output.PredInstance_RCT.tolist(), [(tile + 1) * 100000 + 1] * 2)
                self.assertEqual(output.PredInstance_RCT.dtype, np.dtype("uint32"))
                for suffix in ("trees", "trees_info"):
                    sidecar = root / "segmented_filtered" / f"c{tile:02d}_r00_filtered_{suffix}.txt"
                    self.assertEqual(sidecar.read_text(),
                                     f"# RCT tree table\npredinstance,header\n{(tile + 1) * 100000 + 1},tree-1\n")

    def test_supplied_tree_files_bypass_merging_and_point_ownership(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source, layout = self._fixture(root)
            for tile in range(2):
                write_cloud(source / f'c{tile:02d}_r00_segmented.las', [9.5, 10, 10.5],
                    [1, 1, 1], [2 + tile] * 3, instance='PredInstance_RCT')
            # Both centroids lie on the inclusive core boundary. Their geometry
            # overlaps completely and would qualify for merging without sidecars.
            with patch('strict_prediction_pipeline.reconcile_instances', side_effect=AssertionError('must not merge')), \
                 patch('strict_prediction_pipeline.assign_shared_points', side_effect=AssertionError('must not reassign')), \
                 patch('strict_prediction_pipeline.deduplicate', side_effect=AssertionError('must not thin')):
                report = merge_collections(collections=[source], target_dir=None,
                    output_tiles=root/'output', tile_bounds_json=layout, ready=True,
                    instance_dimension='PredInstance_RCT', matching=True)
            self.assertEqual(report['models'][0]['reconciliation']['accepted_pairs'], [])
            for tile in range(2):
                cloud = laspy.read(root/f'output/tile_{tile:05d}.laz')
                np.testing.assert_allclose(cloud.x, [9.5, 10, 10.5])
                np.testing.assert_array_equal(cloud.PredInstance_RCT, [(tile+1)*100000+1]*3)
                np.testing.assert_array_equal(cloud.PredSemantic_RCT, [2+tile]*3)
                for kind in ('trees', 'trees_info'):
                    lines=(root/f'segmented_filtered/c{tile:02d}_r00_filtered_{kind}.txt').read_text().splitlines()
                    self.assertEqual(lines[2:], [f'{(tile+1)*100000+1},tree-1'])

    def test_rejects_missing_tree_file_before_publishing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, layout = self._fixture(root)
            (source / "c01_r00_trees.txt").unlink()
            (source / "c01_r00_trees_info.txt").unlink()
            with self.assertRaisesRegex(ValueError, "Missing RayCloudTools tree or treeinfo file"):
                merge_collections(collections=[source], target_dir=None,
                                  output_tiles=root / "output_tiles", tile_bounds_json=layout,
                                  ready=True, instance_dimension="PredInstance_RCT")
            self.assertFalse((root / "output_tiles").exists())

    def test_recovers_uncontested_tree_over_neighbor_background(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, layout = self._fixture(root)
            write_cloud(source / 'c00_r00_segmented.las', [1, 2, 9.5, 9.6], [1, 1, 0, 0],
                        [5, 5, 0, 0], instance='PredInstance_RCT')
            write_cloud(source / 'c01_r00_segmented.las', [11, 12, 9.5, 9.6], [1, 1, 2, 2],
                        [7, 7, 9, 9], instance='PredInstance_RCT')
            originals = root / 'originals'
            originals.mkdir()
            write_cloud(originals / 'raw.las', [9.5, 9.6])
            report = merge_collections(collections=[source], target_dir=None, output_tiles=root / 'output',
                tile_bounds_json=layout, ready=True, originals=originals, instance_dimension='PredInstance_RCT')
            recovery = report['models'][0]['orphan_recovery']
            self.assertEqual([(r['tile'], r['local_instance']) for r in recovery['admitted']], [(1, 2)])
            self.assertEqual(recovery['blocked'], [])
            self.assertEqual(recovery['final_support']['missing_samples'], 0)
            cloud = laspy.read(root / 'output/tile_00001.laz')
            self.assertEqual(cloud.PredInstance_RCT.tolist(), [200001, 200001, 200002, 200002])
            self.assertEqual(cloud.PredSemantic_RCT.tolist(), [7, 7, 9, 9])
            np.testing.assert_allclose(cloud.x, [11, 12, 9.5, 9.6])
            self.assertEqual(cloud.intensity.tolist(), [5, 6, 7, 8])
            self.assertEqual(report['models'][0]['reconciliation']['accepted_pairs'], [])
            for suffix in ('trees', 'trees_info'):
                table = root / f'segmented_filtered/c01_r00_filtered_{suffix}.txt'
                self.assertEqual(table.read_text().splitlines()[2:], ['200001,tree-1', '200002,tree-2'])
            enriched = laspy.read(root / 'original_with_predictions/raw.las')
            self.assertEqual(enriched.PredInstance_RCT.tolist(), [1, 1])
            for suffix in ("trees", "trees_info"):
                table = root / "original_with_predictions" / f"raw_{suffix}.txt"
                self.assertEqual(table.read_text().splitlines()[2:], ["1,tree-2"])
            self.assertEqual(enriched.PredSemantic_RCT.tolist(), [9, 9])
            # Separate original remap must make the same ownership decision.
            strict_remap(collections=[root / 'output'], originals=originals, output=root / 'separate',
                         instance_dimension='PredInstance_RCT')
            self.assertEqual(laspy.read(root / 'separate/raw.las').PredInstance_RCT.tolist(), [1, 1])
            # Re-filtering keeps the recovered ID and matching tree rows.
            for table in (root / 'segmented_filtered').glob('*.txt'):
                shutil.copy2(table, root / 'output' / table.name)
            second = root / 'repeat/output'
            merge_collections(collections=[root / 'output'], target_dir=None, output_tiles=second,
                tile_bounds_json=layout, ready=True, instance_dimension='PredInstance_RCT')
            self.assertEqual(laspy.read(second / 'tile_00001.laz').PredInstance_RCT.tolist(),
                             [200001, 200001, 200002, 200002])

    def test_one_conflicting_buffer_point_blocks_entire_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, layout = self._fixture(root)
            # The candidate has an unsupported core point at 9.5. Its buffer
            # tail at 12 conflicts with a different tile's tree of the same ID.
            write_cloud(source / 'c00_r00_segmented.las', [1, 9.5, 10.5, 12], [1, 2, 2, 2],
                        instance='PredInstance_RCT')
            write_cloud(source / 'c01_r00_segmented.las', [11, 12], [2, 2], instance='PredInstance_RCT')
            with patch('dense_tile_merge.MAX_BATCH_POINTS', 1):
                report = merge_collections(collections=[source], target_dir=None, output_tiles=root / 'output',
                    tile_bounds_json=layout, ready=True, instance_dimension='PredInstance_RCT')
            recovery = report['models'][0]['orphan_recovery']
            self.assertEqual(recovery['admitted'], [])
            self.assertEqual([(r['tile'], r['local_instance'], r['conflicting_tile'],
                               r['conflicting_local_instance']) for r in recovery['blocked']], [(0, 2, 1, 2)])
            self.assertEqual(laspy.read(root / 'output/tile_00000.laz').PredInstance_RCT.tolist(), [100001])
            for suffix in ('trees', 'trees_info'):
                table = root / f'segmented_filtered/c00_r00_filtered_{suffix}.txt'
                self.assertEqual(table.read_text().splitlines()[2:], ['100001,tree-1'])

    def test_recovered_tree_blocks_competing_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, layout = self._fixture(root)
            write_cloud(source / 'c00_r00_segmented.las', [1, 2, 10.1, 10.2], [1, 1, 2, 2],
                        instance='PredInstance_RCT')
            write_cloud(source / 'c01_r00_segmented.las', [11, 12, 9.6, 10.1], [1, 1, 2, 2],
                        instance='PredInstance_RCT')
            report = merge_collections(collections=[source], target_dir=None, output_tiles=root / 'output',
                tile_bounds_json=layout, ready=True, instance_dimension='PredInstance_RCT')
            recovery = report['models'][0]['orphan_recovery']
            self.assertEqual([(r['tile'], r['local_instance']) for r in recovery['admitted']], [(0, 2)])
            self.assertEqual([(r['tile'], r['local_instance'], r['conflicting_tile'])
                              for r in recovery['blocked']], [(1, 2, 0)])
            self.assertEqual(laspy.read(root / 'output/tile_00000.laz').PredInstance_RCT.tolist(),
                             [100001, 100001, 100002, 100002])
            self.assertEqual(laspy.read(root / 'output/tile_00001.laz').PredInstance_RCT.tolist(),
                             [200001, 200001])

    def test_recovery_conflict_uses_one_centimeter_xyz_radius(self):
        for separation, admitted in ((.01, False), (.0101, True)):
            with self.subTest(separation=separation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source, layout = self._fixture(root)
                # A second, unsupported point keeps the candidate eligible for
                # recovery even when its first point touches the other tree.
                write_cloud(source / 'c00_r00_segmented.las', [1, 10.5, 10.8], [1, 2, 2],
                            instance='PredInstance_RCT')
                write_cloud(source / 'c01_r00_segmented.las', [10.5 + separation, 12], [1, 1],
                            instance='PredInstance_RCT')
                report = merge_collections(collections=[source], target_dir=None, output_tiles=root / 'output',
                    tile_bounds_json=layout, ready=True, instance_dimension='PredInstance_RCT')
                self.assertEqual(bool(report['models'][0]['orphan_recovery']['admitted']), admitted)

    def test_rejects_rct_dimension_without_sidecars(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, layout = self._fixture(root)
            for sidecar in source.glob("*.txt"):
                sidecar.unlink()
            with self.assertRaisesRegex(ValueError, "require matching _trees.txt and _trees_info.txt"):
                merge_collections(collections=[source], target_dir=None,
                                  output_tiles=root / "output_tiles", tile_bounds_json=layout,
                                  ready=True, instance_dimension="PredInstance_RCT")
            self.assertFalse((root / "output_tiles").exists())

    def test_rejects_missing_treeinfo(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, layout = self._fixture(root)
            (source / "c00_r00_trees_info.txt").unlink()
            with self.assertRaisesRegex(ValueError, "Missing RayCloudTools tree or treeinfo file"):
                merge_collections(collections=[source], target_dir=None,
                                  output_tiles=root / "output_tiles", tile_bounds_json=layout,
                                  ready=True, instance_dimension="PredInstance_RCT")

    def test_rejects_treeinfo_row_count_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, layout = self._fixture(root)
            with (source / "c00_r00_trees_info.txt").open("a") as stream:
                stream.write("orphan-row\n")
            with self.assertRaisesRegex(ValueError, "row counts differ"):
                merge_collections(collections=[source], target_dir=None,
                                  output_tiles=root / "output_tiles", tile_bounds_json=layout,
                                  ready=True, instance_dimension="PredInstance_RCT")
            self.assertFalse((root / "output_tiles").exists())

    def test_merge_with_originals_publishes_rct_background_and_tree_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, layout = self._fixture(root)
            originals = root / "originals"
            originals.mkdir()
            write_cloud(originals / "raw.las", [10.6])
            report = merge_collections(
                collections=[source], target_dir=None, output_tiles=root / "output_tiles",
                tile_bounds_json=layout, ready=True, originals=originals,
                instance_dimension="PredInstance_RCT")
            self.assertEqual(report["state"], "validated")
            self.assertEqual(report["background_assigned_points"], 1)
            self.assertEqual(laspy.read(root / "original_with_predictions" / "raw.las").PredInstance_RCT.tolist(), [0])
            self.assertTrue((root / "segmented_filtered" / "c00_r00_filtered_trees.txt").exists())

    def test_shared_point_rejected_by_both_tiles_becomes_background(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, layout = self._fixture(root)
            # Both tiles contain every coordinate. The shared point at 10.1
            # belongs to ID 2 in both, with opposite-side buffer centroids.
            write_cloud(source / "c00_r00_segmented.las", [9.6, 10.1, 10.4],
                        [1, 2, 2], [5, 6, 6], instance="PredInstance_RCT")
            write_cloud(source / "c01_r00_segmented.las", [9.6, 10.1, 10.4],
                        [2, 2, 1], [8, 8, 9], instance="PredInstance_RCT")
            originals = root / "originals"
            originals.mkdir()
            write_cloud(originals / "raw.las", [9.6, 10.1, 10.405])
            report = merge_collections(
                collections=[source], target_dir=None, output_tiles=root / "output_tiles",
                tile_bounds_json=layout, ready=True, originals=originals,
                instance_dimension="PredInstance_RCT")
            model = report["models"][0]
            self.assertEqual(model["reconciliation"]["accepted_pairs"], [])
            for tile in model["instance_ownership"]["tiles"]:
                self.assertEqual({i["instance"]: i["kept"] for i in tile["instances"]},
                                 {1: True, 2: False})
            for tile in range(2):
                cloud = laspy.read(root / "output_tiles" / f"tile_{tile:05d}.laz")
                self.assertEqual(cloud.PredInstance_RCT.tolist(), [(tile + 1) * 100000 + 1])
                for suffix in ("trees", "trees_info"):
                    sidecar = root / "segmented_filtered" / f"c{tile:02d}_r00_filtered_{suffix}.txt"
                    self.assertEqual(sidecar.read_text(),
                                     f"# RCT tree table\npredinstance,header\n{(tile + 1) * 100000 + 1},tree-1\n")
            output = laspy.read(root / "original_with_predictions" / "raw.las")
            original = laspy.read(originals / "raw.las")
            self.assertEqual(output.X.tolist(), original.X.tolist())
            self.assertEqual(output.intensity.tolist(), original.intensity.tolist())
            self.assertEqual(output.PredInstance_RCT.tolist(), [1, 0, 2])
            self.assertEqual(output.PredSemantic_RCT.tolist(), [5, 0, 9])
            self.assertEqual(report["state"], "validated")
            self.assertEqual(report["background_assigned_points"], 1)
            baseline = next(m for m in report["original_coverage"] if m["stage"] == "unfiltered_1cm")
            self.assertEqual((baseline["matched"], baseline["total"]), (3, 3))
            final = next(m for m in report["original_coverage"] if m["stage"] == "final_survivors")
            self.assertEqual((final["matched"], final["total"], final["background_assigned"]), (2, 3, 1))

    def test_final_remap_writes_zero_for_unmatched_rct_point(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline, filtered, originals = (root / name for name in ("baseline", "filtered", "originals"))
            for folder in (baseline, filtered, originals):
                folder.mkdir()
            write_cloud(baseline / "tile.las", [0, .02, .04], [1, 2, 3], [5, 6, 7],
                        instance="PredInstance_RCT")
            write_cloud(filtered / "tile.las", [0, .04], [1, 3], [5, 7],
                        instance="PredInstance_RCT")
            write_cloud(originals / "raw.las", [0, .02, .04])
            for suffix in ("trees", "trees_info"):
                (filtered / f"tile_{suffix}.txt").write_text("# RCT\npredinstance,header\n1,first\n3,third\n")
            report = strict_remap(collections=[filtered], baseline_collections=[baseline],
                                  originals=originals, output=root / "enriched",
                                  instance_dimension="PredInstance_RCT")
            output = laspy.read(root / "enriched" / "raw.las")
            self.assertEqual(output.PredInstance_RCT.tolist(), [1, 0, 2])
            self.assertEqual(output.PredSemantic_RCT.tolist(), [5, 0, 7])
            self.assertEqual(report["state"], "validated")
            final = next(m for m in report["original_coverage"] if m["stage"] == "final_survivors")
            self.assertEqual((final["matched"], final["total"], final["background_assigned"]), (2, 3, 1))
            self.assertEqual(report["background_assigned_points"], 1)
            with self.assertRaisesRegex(ValueError, "unfiltered_1cm"):
                strict_remap(collections=[filtered], baseline_collections=[filtered],
                             originals=originals, output=root / "invalid",
                             instance_dimension="PredInstance_RCT")
            self.assertFalse((root / "invalid").exists())

    def test_encoded_ids_survive_separate_remap_and_repeat_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, layout = self._fixture(root)
            # Explicit original IDs may already have gaps after earlier filtering.
            for file in source.glob("*.las"):
                cloud = laspy.read(file)
                cloud.PredInstance_RCT = [3 if v == 1 else 2 for v in cloud.PredInstance_RCT]
                cloud.write(file)
            for table in source.glob("*.txt"):
                table.write_text("# RCT tree table\npredinstance,header\n2,tree-2\n3,tree-3\n")
            output = root / "first" / "output_tiles"
            merge_collections(collections=[source], target_dir=None, output_tiles=output,
                              tile_bounds_json=layout, ready=True, instance_dimension="PredInstance_RCT")
            originals = root / "originals"
            originals.mkdir()
            write_cloud(originals / "raw.las", [1, 11, 10.6])
            strict_remap(collections=[output], originals=originals, output=root / "enriched",
                         instance_dimension="PredInstance_RCT")
            self.assertEqual(laspy.read(root / "enriched/raw.las").PredInstance_RCT.tolist(),
                             [1, 2, 0])
            # Co-locate the published tables as required by the filter interface.
            for table in (output.parent / "segmented_filtered").glob("*.txt"):
                shutil.copy2(table, output / table.name)
            second = root / "second" / "output_tiles"
            merge_collections(collections=[output], target_dir=None, output_tiles=second,
                              tile_bounds_json=layout, ready=True, instance_dimension="PredInstance_RCT")
            for tile in range(2):
                expected = (tile + 1) * 100000 + 3
                file = second / f"tile_{tile:05d}.laz"
                self.assertEqual(laspy.read(file).PredInstance_RCT.tolist(), [expected, expected])
                self.assertEqual(read_tile_namespace(file)["tile_id"], tile + 1)
                for suffix in ("trees", "trees_info"):
                    table = second.parent / "segmented_filtered" / f"c{tile:02d}_r00_filtered_{suffix}.txt"
                    self.assertEqual(table.read_text(), f"# RCT tree table\npredinstance,header\n{expected},tree-3\n")

    def test_subset_uses_declared_tile_id_not_collection_position(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, layout = self._fixture(root)
            for file in source.glob("c00*"):
                file.unlink()
            merge_collections(collections=[source], target_dir=None, output_tiles=root / "output",
                              tile_bounds_json=layout, ready=True, instance_dimension="PredInstance_RCT")
            cloud = laspy.read(root / "output/tile_00000.laz")
            self.assertEqual(cloud.PredInstance_RCT.tolist(), [200001, 200001])

    def test_remap_validates_namespaces_in_single_prediction_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, layout = self._fixture(root)
            output = root / 'output'
            merge_collections(collections=[source], target_dir=None, output_tiles=output,
                              tile_bounds_json=layout, ready=True, instance_dimension='PredInstance_RCT')
            originals = root / 'originals'
            originals.mkdir()
            write_cloud(originals / 'raw.las', [1, 11])
            reads = {file: 0 for file in output.glob('*.laz')}
            real_open = laspy.open

            def count_open(path, *args, **kwargs):
                reader = real_open(path, *args, **kwargs)
                if Path(path) in reads and kwargs.get('mode', 'r') == 'r':
                    chunks = reader.chunk_iterator

                    def count_chunks(size):
                        reads[Path(path)] += 1
                        return chunks(size)

                    reader.chunk_iterator = count_chunks
                return reader

            with patch('laspy.open', side_effect=count_open):
                strict_remap(collections=[output], originals=originals, output=root / 'enriched',
                             instance_dimension='PredInstance_RCT')
            self.assertEqual(list(reads.values()), [1, 1])
            self.assertEqual(laspy.read(root / 'enriched/raw.las').PredInstance_RCT.tolist(),
                             [1, 2])
            file = output / 'tile_00000.laz'
            cloud = laspy.read(file)
            cloud.PredInstance_RCT = [200001] * len(cloud.points)
            cloud.write(file)  # Labels now contradict tile 1's namespace VLR.
            with self.assertRaisesRegex(ValueError, 'disagree with their tile namespace'):
                strict_remap(collections=[output], originals=originals, output=root / 'invalid',
                             instance_dimension='PredInstance_RCT')
            self.assertFalse((root / 'invalid').exists())

    def test_background_is_not_offset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, layout = self._fixture(root)
            write_cloud(source / "c00_r00_segmented.las", [1, 2, 10.5], [0, 1, 2],
                        instance="PredInstance_RCT")
            merge_collections(collections=[source], target_dir=None, output_tiles=root / "output",
                              tile_bounds_json=layout, ready=True, instance_dimension="PredInstance_RCT")
            self.assertEqual(laspy.read(root / "output/tile_00000.laz").PredInstance_RCT.tolist(), [0, 100001])

    def test_compact_ids_and_qsm_payloads_are_shared_across_originals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, layout = self._fixture(root)
            for tile in ("c00_r00", "c01_r00"):
                for suffix in ("trees", "trees_info"):
                    (source / f"{tile}_{suffix}.txt").write_text(
                        f"# RCT\nheader\n{tile}-{suffix},1,2,3;4,5,6\nremoved\n")
            output = root / "output"
            merge_collections(collections=[source], target_dir=None, output_tiles=output,
                              tile_bounds_json=layout, ready=True, instance_dimension="PredInstance_RCT")
            originals = root / "originals"
            originals.mkdir()
            write_cloud(originals / "a.las", [1, 11, 10.6])
            add_extended_metadata(originals / "a.las")
            write_cloud(originals / "b.laz", [11])
            write_cloud(originals / "c.las", [10.6])
            write_cloud(originals / "empty.las", [])
            report = strict_remap(collections=[output], originals=originals, output=root / "final")
            expected_ids = {"a.las": [1, 2, 0], "b.laz": [2], "c.las": [0], "empty.las": []}
            assert_extended_metadata(self, root / "final/a.las")
            for name, ids in expected_ids.items():
                final = laspy.read(root / "final" / name)
                original = laspy.read(originals / name)
                self.assertEqual(final.PredInstance_RCT.tolist(), ids)
                self.assertEqual(final.PredInstance_RCT.dtype, np.dtype("uint32"))
                for dimension in original.point_format.dimension_names:
                    np.testing.assert_array_equal(final[dimension], original[dimension])
                for suffix in ("trees", "trees_info"):
                    rows = (root / "final" / f"{Path(name).stem}_{suffix}.txt").read_text().splitlines()[2:]
                    self.assertEqual(rows, [f"{uid},c{uid-1:02d}_r00-{suffix},1,2,3;4,5,6"
                                            for uid in sorted(set(ids) - {0})])
                self.assertIsNone(read_tile_namespace(root / "final" / name))
                compact_vlr = [v for v in final.header.vlrs if v.user_id == "3DTrees" and v.record_id == 24003]
                self.assertEqual(len(compact_vlr), 1)
            metadata = json.loads((root / "final/rct_instance_mapping.json").read_text())
            self.assertEqual(metadata["tree_count"], 2)
            self.assertEqual([row["source_instance_id"] for row in metadata["instances"]], [100001, 200001])
            self.assertEqual([row["source_tile_id"] for row in metadata["instances"]], [1, 2])
            self.assertEqual([row["source_local_id"] for row in metadata["instances"]], [1, 1])
            self.assertEqual([row["tree_count"] for row in metadata["files"]], [2, 1, 0, 0])
            self.assertEqual(report["rct_finalization"]["tree_count"], 2)
            # Subset originals omit tile 1 completely: IDs must still start at 1.
            subset = root / "subset"
            subset.mkdir()
            shutil.copy2(originals / "b.laz", subset / "b.laz")
            strict_remap(collections=[output], originals=subset, output=root / "subset_final")
            self.assertEqual(laspy.read(root / "subset_final/b.laz").PredInstance_RCT.tolist(), [1])

    def test_combined_and_standalone_finalization_agree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, layout = self._fixture(root)
            originals = root / "originals"
            originals.mkdir()
            write_cloud(originals / "raw.las", [1, 11, 10.6])
            output = root / "output"
            merge_collections(collections=[source], target_dir=None, output_tiles=output,
                              tile_bounds_json=layout, ready=True, instance_dimension="PredInstance_RCT",
                              originals=originals)
            strict_remap(collections=[output], originals=originals, output=root / "separate")
            for name in ("raw.las", "raw_trees.txt", "raw_trees_info.txt", "rct_instance_mapping.json"):
                self.assertEqual((root / "original_with_predictions" / name).read_bytes(),
                                 (root / "separate" / name).read_bytes())

    def test_compaction_preserves_other_models_in_three_collection_remap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            originals = root / "originals"
            originals.mkdir()
            write_cloud(originals / "raw.las", [0, 1, 2])
            collections = []
            for model, ids in [("RCT", [7, 0, 999]), ("A", [77, 78, 79]), ("B", [88, 89, 90])]:
                folder = root / model
                folder.mkdir()
                write_cloud(folder / "tile.las", [0, 1, 2], ids, [5, 6, 7], instance=f"PredInstance_{model}")
                collections.append(folder)
            for suffix in ("trees", "trees_info"):
                (collections[0] / f"tile_{suffix}.txt").write_text("# RCT\npredinstance,header\n7,first\n999,last\n")
            strict_remap(collections=collections, baseline_collections=[collections[0]],
                         originals=originals, output=root / "final")
            final = laspy.read(root / "final/raw.las")
            for model, ids in [("RCT", [1, 0, 2]), ("A", [1, 2, 3]), ("B", [1, 2, 3])]:
                self.assertEqual(final[f"PredInstance_{model}"].tolist(), ids)
                self.assertEqual(final[f"PredSemantic_{model}"].tolist(), [5, 6, 7])

    def test_finalization_rejects_missing_or_inconsistent_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, layout = self._fixture(root)
            output = root / "output"
            merge_collections(collections=[source], target_dir=None, output_tiles=output,
                              tile_bounds_json=layout, ready=True, instance_dimension="PredInstance_RCT")
            originals = root / "originals"
            originals.mkdir()
            write_cloud(originals / "raw.las", [1, 11])
            table = root / "segmented_filtered/c00_r00_filtered_trees_info.txt"
            saved = table.read_text()
            for content, error in [(None, "Missing RayCloudTools"),
                                   ("# RCT\npredinstance,header\n", "instance IDs differ"),
                                   (saved.replace("predinstance,header", "predinstance,incompatible"), "incompatible")]:
                with self.subTest(error=error):
                    if content is None:
                        table.unlink()
                    else:
                        table.write_text(content)
                    with self.assertRaisesRegex(ValueError, error):
                        strict_remap(collections=[output], originals=originals, output=root / "invalid")
                    self.assertFalse((root / "invalid").exists())
            # Matching tables that both omit a represented instance also fail.
            for suffix in ("trees", "trees_info"):
                (root / f"segmented_filtered/c00_r00_filtered_{suffix}.txt").write_text(
                    "# RCT\npredinstance,header\n")
            with self.assertRaisesRegex(ValueError, "has no tree row"):
                strict_remap(collections=[output], originals=originals, output=root / "invalid")
            self.assertFalse((root / "invalid").exists())

    def test_unencoded_multiple_tiles_cannot_publish_ambiguous_originals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, _ = self._fixture(root)
            originals = root / "originals"
            originals.mkdir()
            write_cloud(originals / "raw.las", [1, 11])
            with self.assertRaisesRegex(ValueError, "encoded tile namespaces"):
                strict_remap(collections=[source], baseline_collections=[source], originals=originals,
                             output=root / "enriched", instance_dimension="PredInstance_RCT")
            self.assertFalse((root / "enriched").exists())

    def test_namespace_guards_collisions_and_uint32_overflow(self):
        np.testing.assert_array_equal(encode_instance_ids(1, [0, 1, 99999]), [0, 100001, 199999])
        np.testing.assert_array_equal(encode_instance_ids(2, [1]), [200001])
        self.assertEqual(int(encode_instance_ids(42949, [67295])[0]), 2**32 - 1)
        for tile, ids in [(1, [100000]), (1, [-1]), (1, [1.5]), (1, [np.nan]),
                          (0, [1]), (42949, [67296]), (42950, [0])]:
            with self.subTest(tile=tile, ids=ids), self.assertRaises(ValueError):
                encode_instance_ids(tile, ids)

    def test_sidecars_validate_explicit_ids(self):
        for content, expected_error in [
                ("1,first\n1,duplicate\n", "duplicate"),
                ("0,background\n", "invalid"),
                ("bad,tree\n", "invalid"),
                ("1\n", "missing explicit")]:
            with self.subTest(content=content), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = write_cloud(root / "c00_r00_segmented.las", [0], [1], instance="PredInstance_RCT")
                tables = [root / f"c00_r00_{suffix}.txt" for suffix in ("trees", "trees_info")]
                for table in tables:
                    table.write_text("# RCT\npredinstance,header\n" + content)
                ownership = {"tiles": [{"instances": [{"instance": 1, "kept": True}]}]}
                with self.assertRaisesRegex(ValueError, expected_error):
                    filter_tree_sidecars([(source, source, "tile")], tables, ownership, root / "out")

    def test_rejects_unsupported_intermediate_merged_product(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, layout = self._fixture(root)
            with self.assertRaisesRegex(ValueError, "intermediate merged output"):
                merge_collections(collections=[source], target_dir=None,
                                  output_tiles=root / "output_tiles", tile_bounds_json=layout,
                                  ready=True, instance_dimension="PredInstance_RCT",
                                  merged_output=root / "merged.laz")


if __name__ == "__main__":
    unittest.main()
