import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import laspy
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from output_remap import (  # noqa: E402
    remap_merged_file_to_original_input_files,
    remap_to_original_input_files,
    retile_to_original_files,
)


def _write_original(path: Path) -> np.ndarray:
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.array([100.0, 200.0, 50.0])
    las = laspy.LasData(header)
    points = np.array(
        [
            [100.0, 200.0, 50.0],
            [100.1, 200.1, 50.1],
            [100.2, 200.2, 50.2],
        ],
        dtype=np.float64,
    )
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]
    las.intensity = np.array([10, 20, 30], dtype=np.uint16)
    las.add_extra_dim(laspy.ExtraBytesParams(name="OriginalExtra", type=np.uint16))
    las.OriginalExtra = np.array([1, 2, 3], dtype=np.uint16)
    las.write(path)
    return points


def _write_merged_with_predictions(path: Path, points: np.ndarray) -> None:
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.array([100.0, 200.0, 50.0])
    las = laspy.LasData(header)
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]
    las.add_extra_dim(laspy.ExtraBytesParams(name="PredInstance", type=np.uint16))
    las.PredInstance = np.array([1, 0, 2], dtype=np.uint16)
    las.write(path)


class _FakeCopcRecord:
    def __init__(self, points: np.ndarray, pred_instance: np.ndarray):
        self.x = points[:, 0]
        self.y = points[:, 1]
        self.z = points[:, 2]
        self._dims = {"PredInstance": pred_instance}

    def __len__(self):
        return len(self.x)

    def __getitem__(self, name):
        return self._dims[name]


class _FakeCopcReader:
    def __init__(self, header, points: np.ndarray, pred_instance: np.ndarray):
        self.header = header
        self.points = points
        self.pred_instance = pred_instance
        self.query_count = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def spatial_query(self, bounds):
        self.query_count += 1
        return _FakeCopcRecord(self.points, self.pred_instance)


class OutputRemapTests(unittest.TestCase):
    def test_remap_to_original_preserves_original_dims_and_adds_branded_predictions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            original_dir = root / "originals"
            output_dir = root / "out"
            original_dir.mkdir()
            points = _write_original(original_dir / "source.las")

            remap_to_original_input_files(
                merged_points=points,
                merged_extra_dims={
                    "PredInstance": np.array([1, 0, 2], dtype=np.uint16),
                    "PredSemantic": np.array([1, 0, 1], dtype=np.uint8),
                    "IgnoredDim": np.array([9, 9, 9], dtype=np.uint16),
                },
                merged_extra_dim_params=None,
                original_input_dir=original_dir,
                output_dir=output_dir,
                tolerance=0.001,
                num_threads=1,
                threedtrees_dims=["PredInstance", "PredSemantic"],
                threedtrees_suffix="SAT",
            )

            out = laspy.read(output_dir / "source.las")
            dims = set(out.point_format.dimension_names) | {dim.name for dim in out.point_format.extra_dimensions}
            self.assertIn("OriginalExtra", dims)
            self.assertIn("PredInstance_SAT", dims)
            self.assertIn("PredSemantic_SAT", dims)
            self.assertNotIn("IgnoredDim_SAT", dims)
            np.testing.assert_array_equal(out.intensity, np.array([10, 20, 30], dtype=np.uint16))
            np.testing.assert_array_equal(out.OriginalExtra, np.array([1, 2, 3], dtype=np.uint16))
            np.testing.assert_array_equal(out.PredInstance_SAT, np.array([1, 0, 2], dtype=np.uint16))
            np.testing.assert_array_equal(out.PredSemantic_SAT, np.array([1, 0, 1], dtype=np.uint8))

    def test_remap_to_original_preserves_prediction_extra_byte_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            original_dir = root / "originals"
            output_dir = root / "out"
            original_dir.mkdir()
            points = _write_original(original_dir / "source.las")

            remap_to_original_input_files(
                merged_points=points,
                merged_extra_dims={
                    "PredInstance": np.array([1, 0, 2], dtype=np.uint16),
                },
                merged_extra_dim_params={
                    "PredInstance": laspy.ExtraBytesParams(
                        name="PredInstance",
                        type=np.uint16,
                        description="model instance id",
                    ),
                },
                original_input_dir=original_dir,
                output_dir=output_dir,
                tolerance=0.001,
                num_threads=1,
                threedtrees_dims=["PredInstance"],
                threedtrees_suffix="SAT",
            )

            with laspy.open(output_dir / "source.las") as reader:
                descriptions = {
                    dim.name: dim.description
                    for dim in reader.header.point_format.extra_dimensions
                }
            self.assertEqual(descriptions["PredInstance_SAT"], "model instance id")

    def test_remap_to_original_reprocesses_stale_existing_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            original_dir = root / "originals"
            output_dir = root / "out"
            original_dir.mkdir()
            output_dir.mkdir()
            points = _write_original(original_dir / "source.las")
            _write_original(output_dir / "source.las")

            remap_to_original_input_files(
                merged_points=points,
                merged_extra_dims={
                    "PredInstance": np.array([1, 0, 2], dtype=np.uint16),
                    "PredSemantic": np.array([1, 0, 1], dtype=np.uint8),
                },
                merged_extra_dim_params=None,
                original_input_dir=original_dir,
                output_dir=output_dir,
                tolerance=0.001,
                num_threads=1,
                threedtrees_dims=["PredInstance", "PredSemantic"],
                threedtrees_suffix="SAT",
            )

            out = laspy.read(output_dir / "source.las")
            dims = set(out.point_format.dimension_names) | {dim.name for dim in out.point_format.extra_dimensions}
            self.assertIn("PredInstance_SAT", dims)
            self.assertIn("PredSemantic_SAT", dims)

    def test_existing_output_reuse_respects_prediction_name_collisions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            original_dir = root / "originals"
            output_dir = root / "out"
            original_dir.mkdir()
            output_dir.mkdir()
            source_file = original_dir / "source.las"
            points = _write_original(source_file)

            source = laspy.read(source_file)
            source.add_extra_dim(laspy.ExtraBytesParams(name="PredInstance_SAT", type=np.uint16))
            source.PredInstance_SAT = np.array([99, 99, 99], dtype=np.uint16)
            source.write(source_file)
            source.write(output_dir / "source.las")

            remap_to_original_input_files(
                merged_points=points,
                merged_extra_dims={
                    "PredInstance": np.array([1, 0, 2], dtype=np.uint16),
                },
                merged_extra_dim_params=None,
                original_input_dir=original_dir,
                output_dir=output_dir,
                tolerance=0.001,
                num_threads=1,
                threedtrees_dims=["PredInstance"],
                threedtrees_suffix="SAT",
            )

            out = laspy.read(output_dir / "source.las")
            dims = set(out.point_format.dimension_names) | {dim.name for dim in out.point_format.extra_dimensions}
            self.assertIn("PredInstance_SAT", dims)
            self.assertIn("PredInstance_SAT_1", dims)
            np.testing.assert_array_equal(out.PredInstance_SAT, np.array([99, 99, 99], dtype=np.uint16))
            np.testing.assert_array_equal(out.PredInstance_SAT_1, np.array([1, 0, 2], dtype=np.uint16))

    def test_remap_to_original_fails_when_nearest_points_exceed_tolerance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            original_dir = root / "originals"
            output_dir = root / "out"
            original_dir.mkdir()
            points = _write_original(original_dir / "source.las")
            shifted_points = points + np.array([0.01, 0.0, 0.0])

            with self.assertRaisesRegex(RuntimeError, "matched 0/3"):
                remap_to_original_input_files(
                    merged_points=shifted_points,
                    merged_extra_dims={
                        "PredInstance": np.array([1, 0, 2], dtype=np.uint16),
                    },
                    merged_extra_dim_params=None,
                    original_input_dir=original_dir,
                    output_dir=output_dir,
                    tolerance=0.001,
                    num_threads=1,
                    threedtrees_dims=["PredInstance"],
                    threedtrees_suffix="SAT",
                )

            self.assertFalse((output_dir / "source.las").exists())

    def test_retile_to_original_fails_when_nearest_points_exceed_tolerance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            original_dir = root / "original_tiles"
            output_dir = root / "out"
            original_dir.mkdir()
            points = _write_original(original_dir / "tile.las")
            shifted_points = points + np.array([0.01, 0.0, 0.0])

            with self.assertRaisesRegex(RuntimeError, "matched 0/3"):
                retile_to_original_files(
                    merged_points=shifted_points,
                    merged_instances=np.array([1, 0, 2], dtype=np.uint16),
                    merged_extra_dims={},
                    merged_extra_dim_params=None,
                    original_tiles_dir=original_dir,
                    output_dir=output_dir,
                    tolerance=0.001,
                    num_threads=1,
                    parallel_tiles=1,
                    instance_dimension="PredInstance",
                )

            self.assertFalse((output_dir / "tile.las").exists())

    def test_retile_to_original_includes_mixed_laz_and_las_inputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            original_dir = root / "original_tiles"
            output_dir = root / "out"
            original_dir.mkdir()
            (original_dir / "tile_a.laz").write_text("placeholder")
            (original_dir / "tile_b.las").write_text("placeholder")

            import output_remap

            calls = []
            original_process = output_remap._process_single_tile

            def fake_process(args):
                calls.append(args)
                return (args[0].name, 1, 1, 1, True, "OK")

            try:
                output_remap._process_single_tile = fake_process
                retile_to_original_files(
                    merged_points=np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
                    merged_instances=np.array([1], dtype=np.uint16),
                    merged_extra_dims={},
                    merged_extra_dim_params=None,
                    original_tiles_dir=original_dir,
                    output_dir=output_dir,
                    tolerance=0.001,
                    num_threads=1,
                    parallel_tiles=1,
                    instance_dimension="PredInstance",
                )
            finally:
                output_remap._process_single_tile = original_process

        self.assertEqual(
            [args[0].name for args in calls],
            ["tile_a.laz", "tile_b.las"],
        )

    def test_legacy_remap_uses_copc_spatial_fast_path_for_copc_original(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            original_dir = root / "originals"
            output_dir = root / "out"
            original_dir.mkdir()
            copc_original = original_dir / "source.copc.laz"
            raw_twin = original_dir / "source.laz"
            copc_original.write_text("placeholder")
            raw_twin.write_text("placeholder")

            with mock.patch(
                "output_remap._process_single_original_copc_file",
                return_value=("source.copc.laz", 3, 3, 2, True, "Success"),
            ) as fast_path:
                remap_to_original_input_files(
                    merged_points=np.array(
                        [
                            [100.0, 200.0, 50.0],
                            [100.1, 200.1, 50.1],
                            [100.2, 200.2, 50.2],
                        ],
                        dtype=np.float64,
                    ),
                    merged_extra_dims={
                        "PredInstance": np.array([1, 0, 2], dtype=np.uint16),
                    },
                    merged_extra_dim_params=None,
                    original_input_dir=original_dir,
                    output_dir=output_dir,
                    tolerance=0.001,
                    num_threads=1,
                    threedtrees_dims=["PredInstance"],
                    threedtrees_suffix="SAT",
                    num_spatial_chunks=6,
                )

            fast_path.assert_called_once()
            self.assertEqual(fast_path.call_args.args[0], copc_original)
            self.assertEqual(fast_path.call_args.kwargs["num_spatial_chunks"], 6)

    def test_legacy_raw_original_mode_ignores_matching_copc_twin(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            original_dir = root / "originals"
            output_dir = root / "out"
            original_dir.mkdir()
            raw_original = original_dir / "source.las"
            copc_original = original_dir / "source.copc.laz"
            points = _write_original(raw_original)
            copc_original.write_text("placeholder")

            with mock.patch(
                "output_remap._process_single_original_input_file",
                return_value=("source.las", 3, 3, 2, True, "Success"),
            ) as raw_path:
                with mock.patch("output_remap._process_single_original_copc_file") as copc_path:
                    remap_to_original_input_files(
                        merged_points=points,
                        merged_extra_dims={
                            "PredInstance": np.array([1, 0, 2], dtype=np.uint16),
                        },
                        merged_extra_dim_params=None,
                        original_input_dir=original_dir,
                        output_dir=output_dir,
                        tolerance=0.001,
                        num_threads=1,
                        threedtrees_dims=["PredInstance"],
                        threedtrees_suffix="SAT",
                        prefer_copc_sources=False,
                    )

            raw_path.assert_called_once()
            copc_path.assert_not_called()
            process_args = raw_path.call_args.args[0]
            self.assertEqual(process_args[0], raw_original)
            self.assertEqual(process_args[1], output_dir / "source.las")

    def test_merged_copc_remap_streams_original_chunks_and_adds_predictions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            original_dir = root / "originals"
            output_dir = root / "out"
            original_dir.mkdir()
            points = _write_original(original_dir / "source.las")
            merged_copc = root / "merged.copc.laz"
            _write_merged_with_predictions(merged_copc, points)
            pred_instance = np.array([1, 0, 2], dtype=np.uint16)
            with laspy.open(str(merged_copc), laz_backend=laspy.LazBackend.LazrsParallel) as reader:
                fake_reader = _FakeCopcReader(reader.header, points, pred_instance)

            with mock.patch("laspy.CopcReader.open", return_value=fake_reader):
                remap_merged_file_to_original_input_files(
                    merged_file=merged_copc,
                    original_input_dir=original_dir,
                    output_dir=output_dir,
                    tolerance=0.001,
                    num_threads=1,
                    threedtrees_dims=["PredInstance"],
                    threedtrees_suffix="SAT",
                    chunk_size=2,
                    prefer_copc_sources=False,
                )

            out = laspy.read(output_dir / "source.las")
            dims = set(out.point_format.dimension_names) | {dim.name for dim in out.point_format.extra_dimensions}
            self.assertIn("OriginalExtra", dims)
            self.assertIn("PredInstance_SAT", dims)
            np.testing.assert_array_equal(out.OriginalExtra, np.array([1, 2, 3], dtype=np.uint16))
            np.testing.assert_array_equal(out.PredInstance_SAT, pred_instance)
            self.assertEqual(fake_reader.query_count, 2)

    def test_merged_copc_remap_fails_when_nearest_points_exceed_tolerance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            original_dir = root / "originals"
            output_dir = root / "out"
            original_dir.mkdir()
            points = _write_original(original_dir / "source.las")
            merged_copc = root / "merged.copc.laz"
            _write_merged_with_predictions(merged_copc, points)
            shifted_points = points + np.array([0.01, 0.0, 0.0])
            pred_instance = np.array([1, 0, 2], dtype=np.uint16)
            with laspy.open(str(merged_copc), laz_backend=laspy.LazBackend.LazrsParallel) as reader:
                fake_reader = _FakeCopcReader(reader.header, shifted_points, pred_instance)

            with mock.patch("laspy.CopcReader.open", return_value=fake_reader):
                with self.assertRaisesRegex(RuntimeError, "matched 0/3"):
                    remap_merged_file_to_original_input_files(
                        merged_file=merged_copc,
                        original_input_dir=original_dir,
                        output_dir=output_dir,
                        tolerance=0.001,
                        num_threads=1,
                        threedtrees_dims=["PredInstance"],
                        threedtrees_suffix="SAT",
                        chunk_size=2,
                        prefer_copc_sources=False,
                    )

            self.assertFalse((output_dir / "source.las").exists())


if __name__ == "__main__":
    unittest.main()
