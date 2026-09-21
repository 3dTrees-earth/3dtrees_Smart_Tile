import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import laspy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from parameters import Parameters
import run
from test_dense_tile_merge import write_cloud


class RunRemapDirectLazTests(unittest.TestCase):
    def fixture(self, root):
        predictions, originals, baseline = [root / n for n in ("predictions", "originals", "baseline")]
        for path in (predictions, originals, baseline):
            path.mkdir()
        write_cloud(predictions / "tile.las", [0, .1], [1, 1], [2, 2])
        write_cloud(baseline / "tile.las", [0, .1])
        write_cloud(originals / "original.las", [0, .101])
        return predictions, originals, baseline

    def params(self, root, predictions, originals, baseline, **kwargs):
        return Parameters(task="remap", segmented_folders=str(predictions),
                          original_laz_input_dir=originals, original_laz_output_dir=root / "out",
                          baseline_1cm_folders=str(baseline), workers=1,
                          transfer_original_dims_to_merged=False, _cli_parse_args=False, **kwargs)

    def test_collections_enrich_raw_originals_directly_with_copc_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred, originals, baseline = self.fixture(root)
            params = self.params(root, pred, originals, baseline, original_copc_input_dir=root / "copc")
            with mock.patch.object(run, "_validate_copc_original_lane") as validate, mock.patch.object(run, "_validate_copc_laz_source_pairs") as pairs:
                run.run_remap_task(params)
            validate.assert_called_once_with(root / "copc")
            pairs.assert_called_once_with(root / "copc", originals)
            before, after = laspy.read(originals / "original.las"), laspy.read(root / "out/original.las")
            for name in before.points.array.dtype.names:
                np.testing.assert_array_equal(before.points.array[name], after.points.array[name])

    def test_legacy_original_input_alias_and_merged_file_use_strict_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred, originals, baseline = self.fixture(root)
            params = Parameters(task="remap", merged_laz=pred / "tile.las",
                                original_input_dir=originals, output_dir=root / "out",
                                baseline_1cm_folders=str(baseline), _cli_parse_args=False)
            run.run_remap_task(params)
            self.assertTrue((root / "out/original.las").exists())

    def test_two_models_and_prod_creation_use_enriched_originals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred, originals, baseline = self.fixture(root)
            fm = root / "FM"
            fm.mkdir()
            write_cloud(fm / "tile.las", [0, .1], [4, 4], instance="PredInstance_FM")
            params = self.params(root, f"{pred},{fm}", originals, baseline, num_spatial_chunks=3)
            params.transfer_original_dims_to_merged = True
            with mock.patch("main_create_merged_file.create_prod_merged_files", return_value=[]) as create:
                run.run_remap_task(params)
            result = laspy.read(root / "out/original.las")
            self.assertIn("PredInstance_FM", result.point_format.dimension_names)
            self.assertEqual(create.call_args.kwargs["original_with_predictions_dir"], root / "out")
            self.assertEqual(create.call_args.kwargs["chunk_workers"], 3)

    def test_missing_baseline_and_coverage_failure_publish_no_originals(self):
        for missing_baseline in (True, False):
            with self.subTest(missing_baseline=missing_baseline), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                pred, originals, baseline = self.fixture(root)
                params = self.params(root, pred, originals, baseline)
                if missing_baseline:
                    params.baseline_1cm_folders = None
                else:
                    write_cloud(originals / "bad.las", [2])
                with mock.patch.object(run, "_create_prod_merged_outputs") as create:
                    with self.assertRaises(SystemExit):
                        run.run_remap_task(params)
                self.assertFalse((root / "out").exists())
                create.assert_not_called()

    def test_label_reassignment_cannot_bypass_reconciled_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred, originals, baseline = self.fixture(root)
            params = self.params(root, pred, originals, baseline, pre_remap_reassign_instances=True)
            with self.assertRaises(SystemExit):
                run.run_remap_task(params)

    def test_no_copc_only_original_lane_or_conflicting_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred, originals, baseline = self.fixture(root)
            params = Parameters(task="remap", segmented_folders=str(pred),
                                original_copc_input_dir=root / "copc", _cli_parse_args=False)
            with self.assertRaises(SystemExit):
                run.run_remap_task(params)
            params = self.params(root, pred, originals, baseline, merged_laz=pred / "tile.las")
            with self.assertRaises(SystemExit):
                run.run_remap_task(params)


if __name__ == "__main__":
    unittest.main()
