import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from parameters import Parameters  # noqa: E402
    import run  # noqa: E402
except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
    if exc.name not in {"pydantic_settings", "pydantic"}:
        raise
    Parameters = None
    run = None


class RunRemapDirectLazTests(unittest.TestCase):
    @unittest.skipIf(Parameters is None, "pydantic_settings is not installed")
    def test_remap_rejects_ambiguous_prediction_and_merged_sources(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            laz_dir = root / "raw_laz"
            pred_dir = root / "pred"
            merged_laz = root / "merged.laz"
            for directory in (laz_dir, pred_dir):
                directory.mkdir()
            (laz_dir / "source.laz").write_text("placeholder")
            (pred_dir / "pred.laz").write_text("placeholder")
            merged_laz.write_text("placeholder")

            params = Parameters(
                task="remap",
                segmented_folders=str(pred_dir),
                merged_laz=merged_laz,
                original_laz_input_dir=laz_dir,
                transfer_original_dims_to_merged=False,
                workers=1,
                _cli_parse_args=False,
            )

            with mock.patch("prediction_collection_remap.remap_prediction_collections_to_original_files") as collections_remap:
                with mock.patch("output_remap.remap_merged_file_to_original_input_files") as merged_remap:
                    with self.assertRaises(SystemExit):
                        run.run_remap_task(params)

            collections_remap.assert_not_called()
            merged_remap.assert_not_called()

    @unittest.skipIf(Parameters is None, "pydantic_settings is not installed")
    def test_remap_rejects_collection_mode_only_options_in_wrong_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            laz_dir = root / "raw_laz"
            merged_laz = root / "merged.laz"
            laz_dir.mkdir()
            (laz_dir / "source.laz").write_text("placeholder")
            merged_laz.write_text("placeholder")

            params = Parameters(
                task="remap",
                merged_laz=merged_laz,
                original_laz_input_dir=laz_dir,
                remap_dims="PredInstance_SAT",
                transfer_original_dims_to_merged=False,
                workers=1,
                _cli_parse_args=False,
            )

            with mock.patch("output_remap.remap_merged_file_to_original_input_files") as merged_remap:
                with self.assertRaises(SystemExit):
                    run.run_remap_task(params)

            merged_remap.assert_not_called()

    @unittest.skipIf(Parameters is None, "pydantic_settings is not installed")
    def test_remap_rejects_pre_reassign_for_prediction_collections(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            laz_dir = root / "raw_laz"
            pred_dir = root / "pred"
            for directory in (laz_dir, pred_dir):
                directory.mkdir()
            (laz_dir / "source.laz").write_text("placeholder")
            (pred_dir / "pred.laz").write_text("placeholder")

            params = Parameters(
                task="remap",
                segmented_folders=str(pred_dir),
                original_laz_input_dir=laz_dir,
                pre_remap_reassign_instances=True,
                transfer_original_dims_to_merged=False,
                workers=1,
                _cli_parse_args=False,
            )

            with mock.patch("prediction_collection_remap.remap_prediction_collections_to_original_files") as collections_remap:
                with self.assertRaises(SystemExit):
                    run.run_remap_task(params)

            collections_remap.assert_not_called()

    @unittest.skipIf(Parameters is None, "pydantic_settings is not installed")
    def test_default_transfer_dims_follow_custom_instance_dimension(self):
        params = Parameters(
            task="remap",
            instance_dimension="PredInstance_SAT",
            _cli_parse_args=False,
        )

        self.assertEqual(
            run._effective_threedtrees_dims(params),
            ["PredInstance_SAT", "PredSemantic_SAT"],
        )

    @unittest.skipIf(Parameters is None, "pydantic_settings is not installed")
    def test_explicit_transfer_dims_are_not_rewritten(self):
        params = Parameters(
            task="remap",
            instance_dimension="PredInstance_SAT",
            threedtrees_dims="PredInstance",
            _cli_parse_args=False,
        )

        self.assertEqual(run._effective_threedtrees_dims(params), ["PredInstance"])

    @unittest.skipIf(Parameters is None, "pydantic_settings is not installed")
    def test_pre_remap_reassign_uses_effective_custom_instance_dimension(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            laz_dir = root / "raw_laz"
            laz_out = root / "raw_enriched"
            laz_dir.mkdir()
            (laz_dir / "source.laz").write_text("placeholder")
            merged_laz = root / "merged.laz"
            merged_laz.write_text("placeholder")

            params = Parameters(
                task="remap",
                merged_laz=merged_laz,
                original_laz_input_dir=laz_dir,
                original_laz_output_dir=laz_out,
                instance_dimension="PredInstance_SAT",
                pre_remap_reassign_instances=True,
                transfer_original_dims_to_merged=False,
                workers=2,
                _cli_parse_args=False,
            )

            points = np.zeros((3, 3), dtype=np.float64)
            merged_dims = {
                "PredInstance_SAT": np.array([1, 1, 2], dtype=np.uint16),
                "PredSemantic_SAT": np.array([1, 1, 1], dtype=np.uint8),
            }

            with mock.patch.object(run, "_validate_raw_original_lane"):
                with mock.patch("merge_tiles.load_merged_file", return_value=(points, merged_dims, {})):
                    with mock.patch(
                        "merge_tiles.reassign_small_instances_in_dims",
                        return_value={
                            "changed_points": 0,
                            "instances_before": 2,
                            "instances_after": 2,
                        },
                    ) as reassign:
                        with mock.patch("merge_tiles.remap_to_original_input_files") as remap:
                            run.run_remap_task(params)

            reassign.assert_called_once()
            self.assertEqual(reassign.call_args.kwargs["instance_dimension"], "PredInstance_SAT")
            remap.assert_called_once()
            self.assertEqual(
                remap.call_args.kwargs["threedtrees_dims"],
                ["PredInstance_SAT", "PredSemantic_SAT"],
            )

    @unittest.skipIf(Parameters is None, "pydantic_settings is not installed")
    def test_copc_original_lane_accepts_uppercase_copc_extension(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            copc_dir = Path(tmpdir)
            (copc_dir / "SOURCE.COPC.LAZ").write_text("placeholder")

            run._validate_copc_original_lane(copc_dir)

    @unittest.skipIf(Parameters is None, "pydantic_settings is not installed")
    def test_segmented_collections_enrich_laz_directly_when_laz_lane_is_set(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            copc_dir = root / "copc"
            laz_dir = root / "raw_laz"
            pred_dir = root / "pred"
            processing_out = root / "processing_unused"
            laz_out = root / "raw_enriched"
            for directory in (copc_dir, laz_dir, pred_dir):
                directory.mkdir()
            (copc_dir / "source.copc.laz").write_text("placeholder")
            (laz_dir / "source.laz").write_text("placeholder")
            (pred_dir / "pred.laz").write_text("placeholder")

            params = Parameters(
                task="remap",
                segmented_folders=str(pred_dir),
                original_copc_input_dir=copc_dir,
                output_dir=processing_out,
                original_laz_input_dir=laz_dir,
                original_laz_output_dir=laz_out,
                remap_dims="PredInstance_SAT,PredSemantic_SAT",
                transfer_original_dims_to_merged=False,
                workers=1,
                _cli_parse_args=False,
            )

            with mock.patch.object(run, "_validate_copc_original_lane") as validate_copc:
                with mock.patch.object(run, "_validate_copc_laz_source_pairs") as validate_pairs:
                    with mock.patch(
                        "prediction_collection_remap.remap_prediction_collections_to_original_files"
                    ) as remap:
                        run.run_remap_task(params)

            validate_copc.assert_called_once_with(copc_dir)
            validate_pairs.assert_called_once_with(copc_dir, laz_dir)
            remap.assert_called_once()
            args, kwargs = remap.call_args
            self.assertEqual(args[0], [pred_dir])
            self.assertEqual(args[1], laz_dir)
            self.assertEqual(args[2], laz_out)
            self.assertEqual(kwargs["target_dims"], {"PredInstance_SAT", "PredSemantic_SAT"})
            self.assertFalse(kwargs["prefer_copc_sources"])

    @unittest.skipIf(Parameters is None, "pydantic_settings is not installed")
    def test_prod_merged_uses_enriched_laz_dir_when_laz_lane_is_set(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            copc_dir = root / "copc"
            laz_dir = root / "raw_laz"
            pred_dir = root / "pred"
            processing_out = root / "processing_unused"
            laz_out = root / "raw_enriched"
            for directory in (copc_dir, laz_dir, pred_dir):
                directory.mkdir()
            (copc_dir / "source.copc.laz").write_text("placeholder")
            (laz_dir / "source.laz").write_text("placeholder")
            (pred_dir / "pred.laz").write_text("placeholder")

            params = Parameters(
                task="remap",
                segmented_folders=str(pred_dir),
                original_copc_input_dir=copc_dir,
                output_dir=processing_out,
                original_laz_input_dir=laz_dir,
                original_laz_output_dir=laz_out,
                remap_dims="PredInstance_SAT",
                transfer_original_dims_to_merged=True,
                workers=1,
                _cli_parse_args=False,
            )

            def remap_side_effect(*args, **kwargs):
                args[2].mkdir(parents=True, exist_ok=True)

            with mock.patch.object(run, "_validate_copc_original_lane"):
                with mock.patch.object(run, "_validate_copc_laz_source_pairs"):
                    with mock.patch(
                        "prediction_collection_remap.remap_prediction_collections_to_original_files",
                        side_effect=remap_side_effect,
                    ):
                        with mock.patch(
                            "main_create_merged_file.create_prod_merged_files",
                            return_value=[root / "prod_merged_1cm.copc.laz"],
                        ) as create_prod:
                            run.run_remap_task(params)

            create_prod.assert_called_once()
            _, kwargs = create_prod.call_args
            self.assertEqual(kwargs["original_with_predictions_dir"], laz_out)
            self.assertEqual(kwargs["output_dir"], laz_out.parent)
            self.assertEqual(kwargs["chunk_workers"], 1)

    @unittest.skipIf(Parameters is None, "pydantic_settings is not installed")
    def test_two_collections_enrich_originals_then_create_prod_merged_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            laz_dir = root / "raw_laz"
            sat_dir = root / "sat"
            foma_dir = root / "foma"
            laz_out = root / "raw_enriched"
            for directory in (laz_dir, sat_dir, foma_dir):
                directory.mkdir()
            (laz_dir / "source.laz").write_text("placeholder")
            (sat_dir / "sat.laz").write_text("placeholder")
            (foma_dir / "foma.laz").write_text("placeholder")

            params = Parameters(
                task="remap",
                segmented_folders=f"{sat_dir},{foma_dir}",
                original_laz_input_dir=laz_dir,
                original_laz_output_dir=laz_out,
                remap_dims="PredInstance_SAT,PredInstance_ForestMamba",
                transfer_original_dims_to_merged=True,
                merged_resolutions="1cm",
                merged_output_formats="laz,copc.laz",
                workers=1,
                _cli_parse_args=False,
            )

            def remap_side_effect(*args, **kwargs):
                args[2].mkdir(parents=True, exist_ok=True)

            with mock.patch(
                "prediction_collection_remap.remap_prediction_collections_to_original_files",
                side_effect=remap_side_effect,
            ) as remap:
                with mock.patch(
                    "main_create_merged_file.create_prod_merged_files",
                    return_value=[root / "prod_merged_1cm.laz", root / "prod_merged_1cm.copc.laz"],
                ) as create_prod:
                    run.run_remap_task(params)

            remap.assert_called_once()
            args, kwargs = remap.call_args
            self.assertEqual(args[0], [sat_dir, foma_dir])
            self.assertEqual(args[1], laz_dir)
            self.assertEqual(args[2], laz_out)
            self.assertEqual(kwargs["target_dims"], {"PredInstance_SAT", "PredInstance_ForestMamba"})
            create_prod.assert_called_once()
            _, create_kwargs = create_prod.call_args
            self.assertEqual(create_kwargs["original_with_predictions_dir"], laz_out)
            self.assertEqual(create_kwargs["resolution_selector"], "1cm")
            self.assertEqual(create_kwargs["output_format_selector"], "laz,copc.laz")
            self.assertEqual(create_kwargs["chunk_workers"], 1)

    @unittest.skipIf(Parameters is None, "pydantic_settings is not installed")
    def test_legacy_original_input_dir_is_treated_as_laz_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            laz_dir = root / "legacy_originals"
            pred_dir = root / "pred"
            laz_out = root / "raw_enriched"
            for directory in (laz_dir, pred_dir):
                directory.mkdir()
            (laz_dir / "source.laz").write_text("placeholder")
            (pred_dir / "pred.laz").write_text("placeholder")

            params = Parameters(
                task="remap",
                segmented_folders=str(pred_dir),
                original_input_dir=laz_dir,
                output_dir=laz_out,
                remap_dims="PredInstance_SAT",
                transfer_original_dims_to_merged=False,
                workers=1,
                _cli_parse_args=False,
            )

            with mock.patch(
                "prediction_collection_remap.remap_prediction_collections_to_original_files"
            ) as remap:
                run.run_remap_task(params)

            remap.assert_called_once()
            args, kwargs = remap.call_args
            self.assertEqual(args[0], [pred_dir])
            self.assertEqual(args[1], laz_dir)
            self.assertEqual(args[2], laz_out)
            self.assertFalse(kwargs["prefer_copc_sources"])

    @unittest.skipIf(Parameters is None, "pydantic_settings is not installed")
    def test_copc_only_remap_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            copc_dir = root / "copc"
            pred_dir = root / "pred"
            for directory in (copc_dir, pred_dir):
                directory.mkdir()
            (copc_dir / "source.copc.laz").write_text("placeholder")
            (pred_dir / "pred.laz").write_text("placeholder")

            params = Parameters(
                task="remap",
                segmented_folders=str(pred_dir),
                original_copc_input_dir=copc_dir,
                output_dir=root / "unused",
                remap_dims="PredInstance_SAT",
                transfer_original_dims_to_merged=False,
                workers=1,
                _cli_parse_args=False,
            )

            with self.assertRaises(SystemExit):
                run.run_remap_task(params)

    @unittest.skipIf(Parameters is None, "pydantic_settings is not installed")
    def test_merged_copc_remap_uses_streaming_entrypoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            laz_dir = root / "raw_laz"
            laz_out = root / "raw_enriched"
            laz_dir.mkdir()
            (laz_dir / "source.laz").write_text("placeholder")
            merged_copc = root / "merged.copc.laz"
            merged_copc.write_text("placeholder")

            params = Parameters(
                task="remap",
                merged_laz=merged_copc,
                original_laz_input_dir=laz_dir,
                original_laz_output_dir=laz_out,
                threedtrees_dims="PredInstance",
                threedtrees_suffix="SAT",
                transfer_original_dims_to_merged=False,
                workers=2,
                chunk_size=123,
                num_spatial_chunks=4,
                _cli_parse_args=False,
            )

            with mock.patch.object(run, "_validate_raw_original_lane"):
                with mock.patch("output_remap.remap_merged_file_to_original_input_files") as remap:
                    run.run_remap_task(params)

            remap.assert_called_once()
            args, kwargs = remap.call_args
            self.assertEqual(args[0], merged_copc)
            self.assertEqual(args[1], laz_dir)
            self.assertEqual(args[2], laz_out)
            self.assertEqual(kwargs["chunk_size"], 123)
            self.assertEqual(kwargs["num_threads"], 2)
            self.assertEqual(kwargs["num_spatial_chunks"], 4)
            self.assertFalse(kwargs["prefer_copc_sources"])


if __name__ == "__main__":
    unittest.main()
