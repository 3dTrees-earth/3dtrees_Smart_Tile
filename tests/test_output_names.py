import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from output_names import subsample_output_name  # noqa: E402
from subsample_methods import (  # noqa: E402
    SUBSAMPLING_METHOD_CENTER_OF_MASS,
    SUBSAMPLING_METHOD_NEAREST_TO_CENTROID,
    normalize_subsampling_method,
)


class SubsamplingMethodTests(unittest.TestCase):
    def test_normalize_subsampling_method_defaults_to_center_of_mass(self):
        self.assertEqual(normalize_subsampling_method(None), SUBSAMPLING_METHOD_CENTER_OF_MASS)
        self.assertEqual(normalize_subsampling_method("com"), SUBSAMPLING_METHOD_CENTER_OF_MASS)
        self.assertEqual(normalize_subsampling_method("centroid"), SUBSAMPLING_METHOD_NEAREST_TO_CENTROID)


class SubsampleOutputNameTests(unittest.TestCase):
    def test_copc_extension_when_requested(self):
        self.assertEqual(
            subsample_output_name(Path("tile.copc.laz"), 0.01, None, True),
            "tile_subsampled_1cm.copc.laz",
        )

    def test_previous_subsampled_cm_suffix_is_stripped(self):
        self.assertEqual(
            subsample_output_name(Path("tile_subsampled_1cm.copc.laz"), 0.1, None, False),
            "tile_subsampled_10cm.laz",
        )

    def test_tile_id_is_kept_and_output_prefix_removed(self):
        self.assertEqual(
            subsample_output_name(Path("c01_r02.copc.laz"), 0.005, "run_100m", False),
            "c01_r02_subsampled_5mm.laz",
        )


if __name__ == "__main__":
    unittest.main()
