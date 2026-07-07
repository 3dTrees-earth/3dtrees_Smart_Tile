import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import laspy
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from parameters import Parameters  # noqa: E402
    import main_merge  # noqa: E402
    import run  # noqa: E402
except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
    if exc.name not in {"pydantic_settings", "pydantic"}:
        raise
    Parameters = None
    main_merge = None
    run = None


class RunMergeDirectLazTests(unittest.TestCase):
    def _write_las(
        self,
        path: Path,
        xyz: np.ndarray,
        *,
        scales: tuple[float, float, float],
        with_instance: bool,
    ) -> None:
        header = laspy.LasHeader(point_format=3, version="1.2")
        header.offsets = xyz.min(axis=0)
        header.scales = np.asarray(scales)
        las = laspy.LasData(header)
        las.x = xyz[:, 0]
        las.y = xyz[:, 1]
        las.z = xyz[:, 2]
        if with_instance:
            las.add_extra_dim(laspy.ExtraBytesParams(name="PredInstance", type=np.uint16))
            las.PredInstance = np.arange(1, len(xyz) + 1, dtype=np.uint16)
        las.write(path)

    @unittest.skipIf(Parameters is None, "pydantic_settings is not installed")
    def test_single_file_merge_preserves_target_tile_scales(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            segmented = root / "segmented"
            target = root / "target"
            output_tiles = root / "output_tiles"
            tile_bounds = root / "tile_bounds_tindex.json"
            segmented.mkdir()
            target.mkdir()
            tile_bounds.write_text('{"tile_buffer": 20.0, "tiles": []}', encoding="utf-8")

            xyz = np.array(
                [
                    [1000.001, 2000.001, 50.001],
                    [1000.011, 2000.011, 50.011],
                    [1000.021, 2000.021, 50.021],
                ],
                dtype=np.float64,
            )
            self._write_las(
                segmented / "tile.las",
                xyz,
                scales=(0.001, 0.001, 0.001),
                with_instance=True,
            )
            self._write_las(
                target / "target.las",
                xyz,
                scales=(0.001, 0.001, 0.001),
                with_instance=False,
            )

            main_merge.run_merge(
                segmented_dir=segmented,
                output_tiles_dir=output_tiles,
                original_tiles_dir=target,
                tile_bounds_json=tile_bounds,
                output_merged=root / "merged.laz",
                skip_merged_file=True,
                transfer_original_dims_to_merged=False,
                num_threads=1,
            )

            output = laspy.read(output_tiles / "target_segmented.laz")
            np.testing.assert_allclose(output.header.scales, np.array([0.001, 0.001, 0.001]))
            self.assertEqual(len(output.points), len(xyz))
            self.assertIn("PredInstance", output.point_format.dimension_names)

    @unittest.skipIf(Parameters is None, "pydantic_settings is not installed")
    def test_merge_rejects_ambiguous_segmented_sources(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            segmented_10cm = root / "subsampled_res2"
            segmented_remapped = root / "segmented_remapped"
            tile_bounds = root / "tile_bounds_tindex.json"
            for directory in (segmented_10cm, segmented_remapped):
                directory.mkdir()
            tile_bounds.write_text("{}", encoding="utf-8")

            params = Parameters(
                task="merge",
                subsampled_10cm_folder=segmented_10cm,
                segmented_remapped_folder=segmented_remapped,
                tile_bounds_json=tile_bounds,
                transfer_original_dims_to_merged=False,
                workers=1,
                _cli_parse_args=False,
            )

            with mock.patch("filter_buffer_instances.filter_buffer_instances_dir") as filter_dir:
                with mock.patch("main_remap.remap_all_tiles") as remap:
                    with mock.patch("main_merge.run_merge") as run_merge:
                        with self.assertRaises(SystemExit):
                            run.run_merge_task(params)

            filter_dir.assert_not_called()
            remap.assert_not_called()
            run_merge.assert_not_called()

    @unittest.skipIf(Parameters is None, "pydantic_settings is not installed")
    def test_merge_filters_source_predictions_before_remap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            segmented_10cm = root / "subsampled_res2"
            target_1cm = root / "subsampled_res1"
            remapped = root / "segmented_remapped"
            tile_bounds = root / "tile_bounds_tindex.json"
            for directory in (segmented_10cm, target_1cm, remapped):
                directory.mkdir()
            tile_bounds.write_text("{}", encoding="utf-8")

            params = Parameters(
                task="merge",
                subsampled_10cm_folder=segmented_10cm,
                tile_bounds_json=tile_bounds,
                output_merged_laz=root / "merged.laz",
                transfer_original_dims_to_merged=False,
                workers=1,
                _cli_parse_args=False,
            )

            calls = []

            def filter_side_effect(**_kwargs):
                calls.append("filter")
                return {"input_files": 1, "output_files": [root / "segmented_filtered/c0_r0_filtered.laz"]}

            def remap_side_effect(**_kwargs):
                calls.append("remap")
                return remapped

            with mock.patch("main_remap.remap_all_tiles", side_effect=remap_side_effect) as remap:
                with mock.patch("filter_buffer_instances.filter_buffer_instances_dir", side_effect=filter_side_effect) as filter_dir:
                    with mock.patch("main_merge.run_merge", return_value=root / "merged.laz") as run_merge:
                        run.run_merge_task(params)

            self.assertEqual(calls, ["filter", "remap"])
            remap.assert_called_once()
            _, remap_kwargs = remap.call_args
            self.assertEqual(remap_kwargs["source_folder"], root / "segmented_filtered")
            self.assertEqual(remap_kwargs["target_folder"], target_1cm)
            self.assertEqual(remap_kwargs["output_folder"], root / "segmented_remapped")
            filter_dir.assert_called_once_with(
                input_dir=segmented_10cm,
                output_dir=root / "segmented_filtered",
                buffer=10.0,
                suffix="_filtered",
                instance_dimension="PredInstance",
                output_extension=".laz",
            )
            run_merge.assert_called_once()
            _, merge_kwargs = run_merge.call_args
            self.assertEqual(merge_kwargs["segmented_dir"], remapped)
            self.assertIsNone(merge_kwargs["original_input_dir"])

    @unittest.skipIf(Parameters is None, "pydantic_settings is not installed")
    def test_merge_uses_output_area_for_intermediates_when_configured(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_root = root / "readonly_inputs"
            output_root = root / "outputs"
            segmented_10cm = input_root / "subsampled_res2"
            target_1cm = input_root / "subsampled_res1"
            tile_bounds = input_root / "tile_bounds_tindex.json"
            for directory in (segmented_10cm, target_1cm, output_root):
                directory.mkdir(parents=True)
            tile_bounds.write_text("{}", encoding="utf-8")

            params = Parameters(
                task="merge",
                subsampled_10cm_folder=segmented_10cm,
                subsampled_target_folder=target_1cm,
                tile_bounds_json=tile_bounds,
                output_tiles_folder=output_root / "output_tiles",
                output_merged_laz=output_root / "merged.laz",
                transfer_original_dims_to_merged=False,
                workers=1,
                _cli_parse_args=False,
            )

            with mock.patch(
                "filter_buffer_instances.filter_buffer_instances_dir",
                return_value={"input_files": 1, "output_files": [output_root / "segmented_filtered/c0_r0_filtered.laz"]},
            ) as filter_dir:
                with mock.patch("main_remap.remap_all_tiles", return_value=output_root / "segmented_remapped") as remap:
                    with mock.patch("main_merge.run_merge", return_value=output_root / "merged.laz"):
                        run.run_merge_task(params)

            self.assertEqual(filter_dir.call_args.kwargs["output_dir"], output_root / "segmented_filtered")
            self.assertEqual(remap.call_args.kwargs["source_folder"], output_root / "segmented_filtered")
            self.assertEqual(remap.call_args.kwargs["output_folder"], output_root / "segmented_remapped")

    @unittest.skipIf(Parameters is None, "pydantic_settings is not installed")
    def test_merge_original_remap_uses_merged_1cm_tiles_when_copc_lane_is_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            segmented = root / "segmented_remapped"
            laz_dir = root / "raw_laz"
            laz_out = root / "raw_enriched"
            out_tiles = root / "output_tiles"
            tile_bounds = root / "tile_bounds_tindex.json"
            for directory in (segmented, laz_dir):
                directory.mkdir()
            tile_bounds.write_text('{"tile_buffer": 24.0, "tiles": []}', encoding="utf-8")

            params = Parameters(
                task="merge",
                segmented_remapped_folder=segmented,
                tile_bounds_json=tile_bounds,
                output_tiles_folder=out_tiles,
                output_merged_laz=root / "merged.laz",
                original_laz_input_dir=laz_dir,
                original_laz_output_dir=laz_out,
                transfer_original_dims_to_merged=False,
                workers=1,
                _cli_parse_args=False,
            )

            with mock.patch.object(run, "_validate_raw_original_lane") as validate_raw:
                with mock.patch.object(run, "_validate_copc_original_lane") as validate_copc:
                    with mock.patch.object(run, "_validate_copc_laz_source_pairs") as validate_pairs:
                        with mock.patch(
                            "filter_buffer_instances.filter_buffer_instances_dir",
                            return_value={"input_files": 1, "output_files": [root / "segmented_filtered/c0_r0_filtered.laz"]},
                        ):
                            with mock.patch("main_merge.run_merge", return_value=root / "merged.laz") as run_merge:
                                with mock.patch(
                                    "prediction_collection_remap.remap_prediction_collections_to_original_files",
                                ) as remap_original:
                                    run.run_merge_task(params)

            validate_copc.assert_called_once_with(copc_dir)
            validate_pairs.assert_called_once_with(copc_dir, laz_dir)
            validate_raw.assert_called_once_with(laz_dir, laz_out)
            run_merge.assert_called_once()
            _, kwargs = run_merge.call_args
            self.assertEqual(kwargs["segmented_dir"], root / "segmented_filtered")
            self.assertIsNone(kwargs["original_input_dir"])
            remap_original.assert_called_once()
            remap_args = remap_original.call_args.args
            self.assertEqual(remap_args[0], [out_tiles])
            self.assertEqual(remap_args[1], laz_dir)
            self.assertEqual(remap_args[2], laz_out)
            self.assertEqual(remap_original.call_args.kwargs["target_dims"], {"PredInstance", "PredSemantic"})
            self.assertFalse(remap_original.call_args.kwargs["prefer_copc_sources"])

    @unittest.skipIf(Parameters is None, "pydantic_settings is not installed")
    def test_merge_prod_merged_creation_receives_worker_budget(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            segmented = root / "segmented_remapped"
            laz_dir = root / "raw_laz"
            laz_out = root / "raw_enriched"
            out_tiles = root / "output_tiles"
            tile_bounds = root / "tile_bounds_tindex.json"
            for directory in (segmented, laz_dir):
                directory.mkdir()
            tile_bounds.write_text("{}", encoding="utf-8")

            params = Parameters(
                task="merge",
                segmented_remapped_folder=segmented,
                tile_bounds_json=tile_bounds,
                output_tiles_folder=out_tiles,
                output_merged_laz=root / "merged.laz",
                original_laz_input_dir=laz_dir,
                original_laz_output_dir=laz_out,
                transfer_original_dims_to_merged=True,
                merged_resolutions="10cm",
                merged_output_formats="laz",
                workers=5,
                num_spatial_chunks=10,
                _cli_parse_args=False,
            )

            def remap_side_effect(*args, **kwargs):
                args[2].mkdir(parents=True, exist_ok=True)

            with mock.patch.object(run, "_validate_raw_original_lane"):
                with mock.patch(
                    "filter_buffer_instances.filter_buffer_instances_dir",
                    return_value={"input_files": 1, "output_files": [root / "segmented_filtered/c0_r0_filtered.laz"]},
                ):
                    with mock.patch("main_merge.run_merge", return_value=root / "merged.laz"):
                        with mock.patch(
                            "prediction_collection_remap.remap_prediction_collections_to_original_files",
                            side_effect=remap_side_effect,
                        ):
                            with mock.patch(
                                "main_create_merged_file.create_prod_merged_files",
                                return_value=[root / "prod_merged_10cm.laz"],
                            ) as create_prod:
                                run.run_merge_task(params)

            create_prod.assert_called_once()
            _, kwargs = create_prod.call_args
            self.assertEqual(kwargs["original_with_predictions_dir"], laz_out)
            self.assertEqual(kwargs["resolution_selector"], "10cm")
            self.assertEqual(kwargs["output_format_selector"], "laz")
            self.assertEqual(kwargs["num_spatial_chunks"], 10)
            self.assertEqual(kwargs["chunk_workers"], 5)

    @unittest.skipIf(Parameters is None, "pydantic_settings is not installed")
    def test_merge_without_laz_original_lane_runs_without_original_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            segmented = root / "segmented_remapped"
            tile_bounds = root / "tile_bounds_tindex.json"
            segmented.mkdir()
            tile_bounds.write_text('{"tile_buffer": 20.0, "tiles": []}', encoding="utf-8")

            params = Parameters(
                task="merge",
                segmented_remapped_folder=segmented,
                tile_bounds_json=tile_bounds,
                output_tiles_folder=root / "output_tiles",
                transfer_original_dims_to_merged=False,
                workers=1,
                _cli_parse_args=False,
            )

            with mock.patch("filter_buffer_instances.filter_buffer_instances_dir") as filter_dir:
                with mock.patch("main_merge.run_merge") as run_merge:
                    with self.assertRaises(SystemExit):
                        run.run_merge_task(params)

            filter_dir.assert_not_called()
            run_merge.assert_not_called()

    @unittest.skipIf(Parameters is None, "pydantic_settings is not installed")
    def test_merge_original_remap_can_skip_processed_merged_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            segmented = root / "segmented_remapped"
            laz_dir = root / "raw_laz"
            tile_bounds = root / "tile_bounds_tindex.json"
            for directory in (segmented, laz_dir):
                directory.mkdir()
            tile_bounds.write_text("{}", encoding="utf-8")

            params = Parameters(
                task="merge",
                segmented_remapped_folder=segmented,
                tile_bounds_json=tile_bounds,
                output_tiles_folder=root / "output_tiles",
                original_laz_input_dir=laz_dir,
                skip_merged_file=True,
                transfer_original_dims_to_merged=False,
                workers=1,
                _cli_parse_args=False,
            )

            with mock.patch(
                "filter_buffer_instances.filter_buffer_instances_dir",
                return_value={"input_files": 1, "output_files": [root / "segmented_filtered/c0_r0_filtered.laz"]},
            ):
                with mock.patch("main_merge.run_merge", return_value=root / "merged.laz") as run_merge:
                    with mock.patch(
                        "prediction_collection_remap.remap_prediction_collections_to_original_files",
                    ) as remap_original:
                        run.run_merge_task(params)

            run_merge.assert_called_once()
            _, merge_kwargs = run_merge.call_args
            self.assertTrue(merge_kwargs["skip_merged_file"])
            remap_original.assert_called_once()
            self.assertEqual(remap_original.call_args.args[0], [root / "output_tiles"])


if __name__ == "__main__":
    unittest.main()
