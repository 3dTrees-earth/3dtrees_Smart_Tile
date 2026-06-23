import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from parameters import Parameters, get_tile_params  # noqa: E402
except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
    if exc.name != "pydantic_settings":
        raise
    Parameters = None
    get_tile_params = None


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

    @unittest.skipIf(Parameters is None, "pydantic_settings is not installed")
    def test_merged_output_format_aliases_are_normalized_and_deduped(self):
        params = Parameters(merged_output_formats="copc,laz,.ply,copc.laz", _cli_parse_args=False)

        self.assertEqual(params.merged_output_formats, "copc.laz,laz,ply")

    @unittest.skipIf(Parameters is None, "pydantic_settings is not installed")
    def test_invalid_merged_output_format_is_rejected(self):
        with self.assertRaises(ValueError):
            Parameters(merged_output_formats="copc.laz,txt", _cli_parse_args=False)


if __name__ == "__main__":
    unittest.main()
