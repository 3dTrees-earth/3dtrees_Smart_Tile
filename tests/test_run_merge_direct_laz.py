import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import laspy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from parameters import Parameters
import main_merge
import run
from test_dense_tile_merge import write_cloud
from test_strict_prediction_pipeline import layout


class RunMergeDirectLazTests(unittest.TestCase):
    def fixture(self, root):
        source, target, originals = [root / n for n in ("source", "target", "originals")]
        for path in (source, target, originals):
            path.mkdir()
        write_cloud(source / "a.las", [0, .1], [7, 7], [2, 2])
        write_cloud(target / "a.las", [0, .009, .1])
        write_cloud(originals / "a.las", [0, .009, .1])
        return source, target, originals

    def params(self, root, **kwargs):
        return Parameters(task="merge", output_tiles_folder=root / "output/tiles",
                          tile_bounds_json=layout(root, 1), workers=1,
                          transfer_original_dims_to_merged=False,
                          _cli_parse_args=False, **kwargs)

    def test_merge_remaps_unfiltered_predictions_to_dense_geometry_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, target, originals = self.fixture(root)
            params = self.params(root, subsampled_10cm_folder=source, subsampled_target_folder=target,
                                 original_laz_input_dir=originals)
            with mock.patch("filter_buffer_instances.filter_buffer_instances_dir") as old_filter:
                run.run_merge_task(params)
            old_filter.assert_not_called()
            report = json.loads((root / "output/remap_first_report.json").read_text())
            self.assertEqual(report["models"][0]["transfer"][0]["matched"], 3)
            self.assertEqual(report["state"], "validated")
            self.assertTrue((root / "output/merged.laz").is_file())
            self.assertTrue((root / "output/original_with_predictions/a.las").is_file())

    def test_main_merge_adapter_preserves_dense_target_scales(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, target, _ = self.fixture(root)
            main_merge.run_merge(source, root / "out", target, layout(root, 1),
                                 skip_merged_file=True, num_threads=1)
            result = laspy.read(root / "out/tile_00000.laz")
            np.testing.assert_array_equal(result.header.scales, laspy.read(target / "a.las").header.scales)
            self.assertEqual(len(result.points), 3)

    def test_input_area_stays_unchanged_and_no_originals_is_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, target, _ = self.fixture(root)
            before = list(source.iterdir())
            run.run_merge_task(self.params(root, subsampled_10cm_folder=source, subsampled_target_folder=target))
            self.assertEqual(list(source.iterdir()), before)
            report = json.loads((root / "output/remap_first_report.json").read_text())
            self.assertIn("pending", report["state"])
            self.assertFalse((root / "output/original_with_predictions").exists())

    def test_skip_merged_file_still_enriches_raw_originals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, _, originals = self.fixture(root)
            params = self.params(root, segmented_remapped_folder=source, original_laz_input_dir=originals,
                                 skip_merged_file=True)
            run.run_merge_task(params)
            self.assertFalse((root / "output/merged.laz").exists())
            self.assertTrue((root / "output/original_with_predictions/a.las").exists())

    def test_copc_lane_is_validation_only_and_raw_metadata_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, target, originals = self.fixture(root)
            params = self.params(root, subsampled_10cm_folder=source, subsampled_target_folder=target,
                                 original_laz_input_dir=originals, original_copc_input_dir=root / "copc")
            with mock.patch.object(run, "_validate_copc_original_lane") as validate, mock.patch.object(run, "_validate_copc_laz_source_pairs") as pairs:
                run.run_merge_task(params)
            validate.assert_called_once_with(root / "copc")
            pairs.assert_called_once_with(root / "copc", originals)
            before, after = laspy.read(originals / "a.las"), laspy.read(root / "output/original_with_predictions/a.las")
            for name in before.points.array.dtype.names:
                np.testing.assert_array_equal(before.points.array[name], after.points.array[name])

    def test_prod_generation_only_runs_after_strict_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, target, originals = self.fixture(root)
            params = self.params(root, subsampled_10cm_folder=source, subsampled_target_folder=target,
                                 original_laz_input_dir=originals, num_spatial_chunks=3)
            params.transfer_original_dims_to_merged = True
            with mock.patch("main_create_merged_file.create_prod_merged_files", return_value=[]) as create:
                run.run_merge_task(params)
            self.assertEqual(create.call_args.kwargs["original_with_predictions_dir"],
                             root / "output/original_with_predictions")
            self.assertEqual(create.call_args.kwargs["chunk_workers"], 3)

    def test_independent_model_collections_are_not_combined_before_matching(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sat, fm = root / "SAT", root / "FM"
            sat.mkdir()
            fm.mkdir()
            write_cloud(sat / "a.las", [0], [3], instance="PredInstance_SAT")
            write_cloud(fm / "a.las", [0], [900], instance="PredInstance_FM")
            run.run_merge_task(self.params(root, segmented_folders=f"{sat},{fm}", skip_merged_file=True))
            report = json.loads((root / "output/remap_first_report.json").read_text())
            self.assertEqual(len(report["models"]), 2)
            self.assertEqual(report["models"][0]["reconciliation"]["ids"], [[0, 3, 1]])
            self.assertEqual(report["models"][1]["reconciliation"]["ids"], [[0, 900, 1]])

    def test_ambiguous_sources_and_missing_tile_bounds_fail_before_processing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(SystemExit):
                run.run_merge_task(self.params(root, subsampled_10cm_folder=root / "a",
                                              segmented_remapped_folder=root / "b"))
            params = self.params(root, segmented_remapped_folder=root / "a")
            params.tile_bounds_json = None
            with self.assertRaises(SystemExit):
                run.run_merge_task(params)
            self.assertFalse((root / "output").exists())


if __name__ == "__main__":
    unittest.main()
