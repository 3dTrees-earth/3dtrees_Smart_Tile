import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from parameters import Parameters  # noqa: E402
    import run  # noqa: E402
except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
    if exc.name != "pydantic_settings":
        raise
    Parameters = None
    run = None


class ParameterSubsamplingMethodTests(unittest.TestCase):
    @unittest.skipIf(Parameters is None, "pydantic_settings is not installed")
    def test_default_subsampling_method_is_center_of_mass(self):
        params = Parameters(_cli_parse_args=False)

        self.assertEqual(params.subsampling_method, "center-of-mass")

    @unittest.skipIf(Parameters is None, "pydantic_settings is not installed")
    def test_default_worker_policy_is_two_files_and_cpu_spatial_chunks(self):
        params = Parameters(_cli_parse_args=False)

        self.assertEqual(params.workers, 2)
        self.assertGreaterEqual(params.num_spatial_chunks, 1)

    @unittest.skipIf(Parameters is None, "pydantic_settings is not installed")
    def test_subsampling_method_alias_normalizes_to_nearest_to_centroid(self):
        params = Parameters(subsampling_method="centroid", _cli_parse_args=False)

        self.assertEqual(params.subsampling_method, "nearest-to-centroid")

    @unittest.skipIf(Parameters is None, "pydantic_settings is not installed")
    def test_invalid_subsampling_method_is_rejected(self):
        with self.assertRaises(ValueError):
            Parameters(subsampling_method="voxel-center", _cli_parse_args=False)

    @unittest.skipIf(Parameters is None, "pydantic_settings is not installed")
    def test_default_prod_merged_generation_is_off_and_copc_when_enabled(self):
        params = Parameters(_cli_parse_args=False)

        self.assertFalse(params.transfer_original_dims_to_merged)
        self.assertEqual(params.merged_output_formats, "copc.laz")
        self.assertIsNone(params.staged_copc_dir)
        self.assertIsNone(params.remap_tolerance)
        self.assertEqual(params.prediction_transfer_tolerance, 0.1732)

    @unittest.skipIf(Parameters is None, "pydantic_settings is not installed")
    def test_staged_copc_dir_alias_is_available(self):
        params = Parameters(staged_copc_dir="/tmp/staged-copc", _cli_parse_args=False)

        self.assertEqual(params.staged_copc_dir, Path("/tmp/staged-copc"))

    @unittest.skipIf(Parameters is None, "pydantic_settings is not installed")
    def test_raw_original_lane_aliases_are_available(self):
        params = Parameters(
            original_laz_input_dir="/tmp/raw",
            original_laz_output_dir="/tmp/raw-out",
            _cli_parse_args=False,
        )

        self.assertEqual(params.original_raw_input_dir, Path("/tmp/raw"))
        self.assertEqual(params.original_raw_output_dir, Path("/tmp/raw-out"))

    @unittest.skipIf(Parameters is None, "pydantic_settings is not installed")
    def test_merged_output_format_aliases_are_normalized_and_deduped(self):
        params = Parameters(merged_output_formats="copc,laz,.ply,copc.laz", _cli_parse_args=False)

        self.assertEqual(params.merged_output_formats, "copc.laz,laz,ply")

    @unittest.skipIf(Parameters is None, "pydantic_settings is not installed")
    def test_merged_output_format_list_like_values_are_normalized(self):
        params = Parameters(merged_output_formats=["copc.laz", "laz"], _cli_parse_args=False)
        self.assertEqual(params.merged_output_formats, "copc.laz,laz")

        params = Parameters(merged_output_formats="['copc.laz', 'ply']", _cli_parse_args=False)
        self.assertEqual(params.merged_output_formats, "copc.laz,ply")

    @unittest.skipIf(Parameters is None, "pydantic_settings is not installed")
    def test_invalid_merged_output_format_is_rejected(self):
        with self.assertRaises(ValueError):
            Parameters(merged_output_formats="copc.laz,txt", _cli_parse_args=False)

    @unittest.skipIf(Parameters is None, "pydantic_settings is not installed")
    def test_zero_tile_buffer_is_allowed(self):
        params = Parameters(tile_buffer=0, _cli_parse_args=False)

        self.assertEqual(params.tile_buffer, 0)

    @unittest.skipIf(Parameters is None, "pydantic_settings is not installed")
    def test_negative_tile_buffer_is_rejected(self):
        with self.assertRaises(ValueError):
            Parameters(tile_buffer=-1, _cli_parse_args=False)

    @unittest.skipIf(Parameters is None, "pydantic_settings is not installed")
    def test_num_spatial_chunks_must_be_positive(self):
        with self.assertRaises(ValueError):
            Parameters(num_spatial_chunks=0, _cli_parse_args=False)

    @unittest.skipIf(Parameters is None, "pydantic_settings is not installed")
    def test_cli_unknown_long_flags_fail_fast(self):
        self.assertEqual(run._unknown_cli_flags(["--task", "tile", "--tilng-threshold", "1"]), ["tilng-threshold"])

    @unittest.skipIf(Parameters is None, "pydantic_settings is not installed")
    def test_inactive_reassignment_options_fail_instead_of_being_ignored(self):
        import contextlib
        import io

        removed = (
            "min_cluster_size", "enable_volume_merge", "disable_volume_merge",
            "pre_remap_reassign_instance_dimension", "pre_remap_reassign_min_cluster_size",
            "pre_remap_reassign_hull_point_threshold", "pre_remap_reassign_max_volume",
            "pre_remap_reassigned_laz",
        )
        for name in removed:
            for flag in (name, name.replace("_", "-")):
                with self.subTest(flag=flag), contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        run._validate_known_cli_flags(["--task", "merge", "--" + flag, "1"])
        run._validate_known_cli_flags([
            "--task", "merge", "--reassign-small-instances",
            "--max-cluster-size", "3000", "--max-volume-for-merge", "4",
        ])

    @unittest.skipIf(Parameters is None, "pydantic_settings is not installed")
    def test_cli_known_alias_and_preprocessor_flags_are_accepted(self):
        self.assertEqual(
            run._unknown_cli_flags(
                [
                    "--task",
                    "remap",
                    "--subsampled-segmented-folder",
                    "segmented",
                    "--produce-merged-file",
                    "--no-produce-merged-file",
                    "--no-produce_merged_file",
                    "--no-transfer-original-dims-to-merged",
                    "--no-transfer_original_dims_to_merged",
                    "--transfer-original-dims-to-merged",
                    "--show-params",
                    "--output-copc-res1",
                    "True",
                ]
            ),
            [],
        )

    @unittest.skipIf(Parameters is None, "pydantic_settings is not installed")
    def test_cli_help_lists_current_options_without_removed_dimension_flags(self):
        import io
        from contextlib import redirect_stdout

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            run._print_cli_help()
        help_text = stdout.getvalue()

        self.assertIn("--task", help_text)
        self.assertIn("--subsampled-10cm-folder", help_text)
        self.assertIn("--segmented-folders", help_text)
        self.assertIn("--produce-merged-file", help_text)
        self.assertIn("--no-produce-merged-file", help_text)
        self.assertNotIn("skip-dimension-reduction", help_text)
        self.assertNotIn("dimension_reduction", help_text)
        self.assertNotIn("keep all dims", help_text.lower())


if __name__ == "__main__":
    unittest.main()
