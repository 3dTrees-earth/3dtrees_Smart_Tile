import sys
import tempfile
import types
import unittest
from unittest import mock
from pathlib import Path

import laspy
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from main_subsample import (  # noqa: E402
    SUBSAMPLING_METHOD_CENTER_OF_MASS,
    SUBSAMPLING_METHOD_NEAREST_TO_CENTROID,
    _aggregate_center_of_mass_xyz,
    _iter_copc_center_of_mass_windows,
    _subsample_input_files,
    _voxel_subsampling_filter,
    _write_center_of_mass_points,
    center_of_mass_subsample_las,
    normalize_subsampling_method,
)
import main_subsample  # noqa: E402
import subsample_com  # noqa: E402


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

        with mock.patch.object(subsample_com, "COPC_COM_TARGET_WINDOW_CELLS", 1):
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

    def test_subsample_parallel_uses_copc_extension_when_requested(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_dir = tmp_path / "tiles"
            output_dir = tmp_path / "subsampled"
            input_dir.mkdir()
            (input_dir / "tile.copc.laz").write_bytes(b"placeholder")

            def fake_subsample(args):
                output_file = args[1]
                output_copc = args[-1]
                output_file.write_bytes(b"placeholder")
                return (args[0].name, True, "Success", 1 if output_copc else 0)

            with mock.patch.object(main_subsample, "subsample_single_file", side_effect=fake_subsample) as worker:
                outputs = main_subsample.subsample_parallel(
                    input_dir=input_dir,
                    output_dir=output_dir,
                    resolution=0.01,
                    num_cores=1,
                    num_threads=1,
                    output_copc=True,
                )

            self.assertEqual([path.name for path in outputs], ["tile_subsampled_1cm.copc.laz"])
            self.assertTrue(worker.call_args.args[0][-1])

    def test_subsample_parallel_strips_previous_subsampled_cm_suffix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_dir = tmp_path / "subsampled_res1"
            output_dir = tmp_path / "subsampled_res2"
            input_dir.mkdir()
            (input_dir / "tile_subsampled_1cm.copc.laz").write_bytes(b"placeholder")

            def fake_subsample(args):
                output_file = args[1]
                output_file.write_bytes(b"placeholder")
                return (args[0].name, True, "Success", 1)

            with mock.patch.object(main_subsample, "subsample_single_file", side_effect=fake_subsample):
                outputs = main_subsample.subsample_parallel(
                    input_dir=input_dir,
                    output_dir=output_dir,
                    resolution=0.1,
                    num_cores=1,
                    num_threads=1,
                    output_copc=False,
                )

            self.assertEqual([path.name for path in outputs], ["tile_subsampled_10cm.laz"])
            self.assertFalse((output_dir / "tile_subsampled_subsampled_10cm.laz").exists())

    def test_subsample_parallel_manifest_skips_matching_rerun(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_dir = tmp_path / "tiles"
            output_dir = tmp_path / "subsampled"
            input_dir.mkdir()
            (input_dir / "tile.laz").write_bytes(b"placeholder")

            def fake_subsample(args):
                output_file = args[1]
                output_file.write_bytes(b"subsampled")
                return (args[0].name, True, "Success", 1)

            with mock.patch.object(main_subsample, "subsample_single_file", side_effect=fake_subsample) as worker:
                first = main_subsample.subsample_parallel(
                    input_dir=input_dir,
                    output_dir=output_dir,
                    resolution=0.01,
                    num_cores=1,
                    num_threads=1,
                    output_copc=False,
                    subsampling_method="center-of-mass",
                )
                second = main_subsample.subsample_parallel(
                    input_dir=input_dir,
                    output_dir=output_dir,
                    resolution=0.01,
                    num_cores=1,
                    num_threads=1,
                    output_copc=False,
                    subsampling_method="center-of-mass",
                )

            self.assertEqual([path.name for path in first], ["tile_subsampled_1cm.laz"])
            self.assertEqual([path.name for path in second], ["tile_subsampled_1cm.laz"])
            self.assertEqual(worker.call_count, 1)

    def test_subsample_parallel_rebuilds_when_method_changes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_dir = tmp_path / "tiles"
            output_dir = tmp_path / "subsampled"
            input_dir.mkdir()
            (input_dir / "tile.laz").write_bytes(b"placeholder")

            writes = []

            def fake_subsample(args):
                output_file = args[1]
                method = args[6]
                writes.append(method)
                output_file.write_bytes(method.encode("ascii"))
                return (args[0].name, True, "Success", 1)

            with mock.patch.object(main_subsample, "subsample_single_file", side_effect=fake_subsample):
                main_subsample.subsample_parallel(
                    input_dir=input_dir,
                    output_dir=output_dir,
                    resolution=0.01,
                    num_cores=1,
                    num_threads=1,
                    output_copc=False,
                    subsampling_method="center-of-mass",
                )
                main_subsample.subsample_parallel(
                    input_dir=input_dir,
                    output_dir=output_dir,
                    resolution=0.01,
                    num_cores=1,
                    num_threads=1,
                    output_copc=False,
                    subsampling_method="nearest-to-centroid",
                )

            self.assertEqual(writes, ["center-of-mass", "nearest-to-centroid"])
            self.assertEqual((output_dir / "tile_subsampled_1cm.laz").read_bytes(), b"nearest-to-centroid")

    def test_subsample_input_files_include_mixed_laz_and_las(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir)
            (input_dir / "first.laz").write_bytes(b"placeholder")
            (input_dir / "second.las").write_bytes(b"placeholder")

            files = [path.name for path in _subsample_input_files(input_dir)]

        self.assertEqual(files, ["first.laz", "second.las"])

    def test_subsample_input_files_prefer_copc_twin(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir)
            (input_dir / "tile.laz").write_bytes(b"placeholder")
            (input_dir / "tile.copc.laz").write_bytes(b"placeholder")

            files = [path.name for path in _subsample_input_files(input_dir)]

        self.assertEqual(files, ["tile.copc.laz"])

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
