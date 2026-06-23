import sys
import tempfile
import types
import unittest
from unittest import mock
from pathlib import Path

import laspy
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.modules.setdefault(
    "parameters",
    types.SimpleNamespace(
        TILE_PARAMS={
            "subsampling_method": "center-of-mass",
            "resolution_1": 0.01,
            "resolution_2": 0.1,
            "threads": 2,
        }
    ),
)

from main_subsample import (  # noqa: E402
    SUBSAMPLING_METHOD_CENTER_OF_MASS,
    SUBSAMPLING_METHOD_NEAREST_TO_CENTROID,
    _aggregate_center_of_mass_xyz,
    _iter_copc_center_of_mass_windows,
    _voxel_subsampling_filter,
    _write_center_of_mass_points,
    center_of_mass_subsample_las,
    normalize_subsampling_method,
)
import main_subsample  # noqa: E402


class FakePoints:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __len__(self):
        return len(self.x)


def _write_test_las(path: Path) -> None:
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.array([0.0, 0.0, 0.0])

    las = laspy.LasData(header)
    las.x = np.array([0.0, 0.02, 0.08, 0.12])
    las.y = np.array([0.0, 0.0, 0.0, 0.0])
    las.z = np.array([0.0, 0.0, 0.0, 0.0])
    las.intensity = np.array([10, 20, 80, 120], dtype=np.uint16)
    las.classification = np.array([1, 2, 8, 12], dtype=np.uint8)
    las.add_extra_dim(laspy.ExtraBytesParams(name="PredInstance", type=np.uint16))
    las.PredInstance = np.array([10, 20, 80, 120], dtype=np.uint16)
    las.write(path)


class SubsamplingMethodTests(unittest.TestCase):
    def test_normalize_subsampling_method_defaults_to_center_of_mass(self):
        self.assertEqual(normalize_subsampling_method(None), SUBSAMPLING_METHOD_CENTER_OF_MASS)
        self.assertEqual(normalize_subsampling_method("com"), SUBSAMPLING_METHOD_CENTER_OF_MASS)
        self.assertEqual(normalize_subsampling_method("centroid"), SUBSAMPLING_METHOD_NEAREST_TO_CENTROID)

    def test_nearest_to_centroid_uses_existing_pdal_filter(self):
        self.assertEqual(
            _voxel_subsampling_filter(0.1, "nearest-to-centroid"),
            {"type": "filters.voxelcentroidnearestneighbor", "cell": 0.1},
        )

    def test_center_of_mass_averages_xyz_only_and_preserves_selected_attributes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            src = tmp_path / "source.las"
            out = tmp_path / "out.las"
            _write_test_las(src)

            count = center_of_mass_subsample_las(src, out, 0.1, dimension_reduction=False)

            self.assertEqual(count, 2)
            result = laspy.read(out)
            self.assertTrue(np.allclose(result.x, [0.033, 0.12], atol=0.001))
            self.assertTrue(np.allclose(result.y, [0.0, 0.0], atol=0.001))
            self.assertTrue(np.allclose(result.z, [0.0, 0.0], atol=0.001))
            # Non-XYZ dimensions come from the real point nearest to the averaged XYZ.
            self.assertEqual(list(result.intensity), [20, 120])
            self.assertEqual(list(result.classification), [2, 12])
            self.assertEqual(list(result.PredInstance), [20, 120])

    def test_center_of_mass_dimension_reduction_drops_extra_dimensions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            src = tmp_path / "source.las"
            out = tmp_path / "out.las"
            _write_test_las(src)

            center_of_mass_subsample_las(src, out, 0.1, dimension_reduction=True)

            result = laspy.read(out)
            self.assertEqual(result.header.point_format.id, 0)
            self.assertNotIn("PredInstance", set(result.point_format.dimension_names))

    def test_center_of_mass_xyz_aggregates_without_non_coordinate_attributes(self):
        points = FakePoints(
            x=np.array([0.0, 0.02, 0.11]),
            y=np.array([0.0, 0.02, 0.0]),
            z=np.array([0.0, 0.02, 0.0]),
        )

        centers = _aggregate_center_of_mass_xyz(points, 0.1)

        self.assertEqual(len(centers), 2)
        self.assertTrue(any(np.allclose(center, [0.01, 0.01, 0.01]) for center in centers))
        self.assertTrue(any(np.allclose(center, [0.11, 0.0, 0.0]) for center in centers))

    def test_copc_center_of_mass_windows_are_voxel_aligned_and_half_open(self):
        header = types.SimpleNamespace(
            x_min=0.0,
            x_max=0.4,
            y_min=0.0,
            y_max=0.2,
            z_min=0.0,
            z_max=1.0,
            scales=np.array([0.01, 0.01, 0.01]),
        )

        with mock.patch.object(main_subsample, "COPC_COM_TARGET_WINDOW_CELLS", 1):
            windows = list(_iter_copc_center_of_mass_windows(header, 0.2))

        self.assertEqual(len(windows), 2)
        self.assertTrue(np.allclose(windows[0].mins, [0.0, 0.0, 0.0]))
        self.assertTrue(np.allclose(windows[0].maxs, [0.195, 0.2, 1.0]))
        self.assertTrue(np.allclose(windows[1].mins, [0.2, 0.0, 0.0]))
        self.assertTrue(np.allclose(windows[1].maxs, [0.4, 0.2, 1.0]))

    def test_copc_xyz_center_of_mass_bypasses_stripe_chunking(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_file = tmp_path / "tile.copc.laz"
            output_file = tmp_path / "tile_subsampled_20cm.laz"

            with mock.patch.object(
                main_subsample,
                "center_of_mass_subsample_copc",
                return_value=42,
            ) as optimized:
                result = main_subsample.subsample_single_file(
                    (
                        input_file,
                        output_file,
                        0.2,
                        tmp_path,
                        8,
                        True,
                        "center-of-mass",
                    )
                )

            self.assertEqual(result, (input_file.name, True, "Success", 42))
            optimized.assert_called_once_with(input_file, output_file, 0.2, num_workers=8)

    def test_center_of_mass_points_can_be_written_in_streamed_batches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            out = tmp_path / "streamed.laz"
            header = laspy.LasHeader(point_format=0, version="1.2")
            header.scales = np.array([0.001, 0.001, 0.001])
            header.offsets = np.array([0.0, 0.0, 0.0])

            with laspy.open(str(out), mode="w", header=header, do_compress=True) as writer:
                first = _write_center_of_mass_points(
                    writer,
                    header,
                    np.array([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]]),
                )
                second = _write_center_of_mass_points(
                    writer,
                    header,
                    np.array([[0.07, 0.08, 0.09]]),
                )

            result = laspy.read(out)
            self.assertEqual(first, 2)
            self.assertEqual(second, 1)
            self.assertEqual(len(result.points), 3)
            self.assertTrue(np.allclose(result.x, [0.01, 0.04, 0.07]))
            self.assertTrue(np.allclose(result.y, [0.02, 0.05, 0.08]))
            self.assertTrue(np.allclose(result.z, [0.03, 0.06, 0.09]))


if __name__ == "__main__":
    unittest.main()
