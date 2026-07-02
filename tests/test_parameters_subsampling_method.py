import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from parameters import Parameters, get_tile_params  # noqa: E402
    import run  # noqa: E402
except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
    if exc.name != "pydantic_settings":
        raise
    Parameters = None
    get_tile_params = None
    run = None


class ParameterSubsamplingMethodTests(unittest.TestCase):
    @unittest.skipIf(Parameters is None, "pydantic_settings is not installed")
    def test_default_subsampling_method_is_center_of_mass(self):
        params = Parameters(_cli_parse_args=False)

        self.assertEqual(params.subsampling_method, "center-of-mass")
        self.assertEqual(get_tile_params(params)["subsampling_method"], "center-of-mass")

    @unittest.skipIf(Parameters is None, "pydantic_settings is not installed")
    def test_subsampling_method_alias_normalizes_to_nearest_to_centroid(self):
        params = Parameters(subsampling_method="centroid", _cli_parse_args=False)

        self.assertEqual(params.subsampling_method, "nearest-to-centroid")

    @unittest.skipIf(Parameters is None, "pydantic_settings is not installed")
    def test_invalid_subsampling_method_is_rejected(self):
        with self.assertRaises(ValueError):
            Parameters(subsampling_method="voxel-center", _cli_parse_args=False)

    @unittest.skipIf(Parameters is None, "pydantic_settings is not installed")
    def test_default_merged_output_format_is_copc_laz(self):
        params = Parameters(_cli_parse_args=False)

        self.assertEqual(params.merged_output_formats, "copc.laz")
        self.assertIsNone(params.staged_copc_dir)

    @unittest.skipIf(Parameters is None, "pydantic_settings is not installed")
    def test_staged_copc_dir_alias_is_available(self):
        params = Parameters(staged_copc_dir="/tmp/staged-copc", _cli_parse_args=False)

        self.assertEqual(params.staged_copc_dir, Path("/tmp/staged-copc"))

    @unittest.skipIf(Parameters is None, "pydantic_settings is not installed")
    def test_raw_original_lane_aliases_are_available(self):
        params = Parameters(
            original_copc_input_dir="/tmp/copc",
            original_laz_input_dir="/tmp/raw",
            original_laz_output_dir="/tmp/raw-out",
            _cli_parse_args=False,
        )

        self.assertEqual(params.original_copc_input_dir, Path("/tmp/copc"))
        self.assertEqual(params.original_raw_input_dir, Path("/tmp/raw"))
        self.assertEqual(params.original_raw_output_dir, Path("/tmp/raw-out"))

    @unittest.skipIf(Parameters is None, "pydantic_settings is not installed")
    def test_filter_output_extension_alias_is_available(self):
        params = Parameters(filter_output_extension=".laz", _cli_parse_args=False)

        self.assertEqual(params.filter_output_extension, ".laz")

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
    def test_cli_known_alias_and_preprocessor_flags_are_accepted(self):
        self.assertEqual(
            run._unknown_cli_flags(
                [
                    "--task",
                    "remap",
                    "--subsampled-segmented-folder",
                    "segmented",
                    "--no-transfer-original-dims-to-merged",
                    "--show-params",
                    "--output-copc-res1",
                    "True",
                ]
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
