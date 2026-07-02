import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

import laspy
import numpy as np
from laspy.vlrs.vlr import VLR


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from instance_labels import (  # noqa: E402
    instance_output_dtype,
    validate_prediction_instance_labels,
)
from merge_tiles import load_tile  # noqa: E402
from point_cloud_metadata import copc_files, point_cloud_files, raw_point_cloud_files  # noqa: E402
from point_cloud_outputs import merged_product_header, write_loaded_point_cloud  # noqa: E402


def _write_las_with_predinstance(path: Path, values: np.ndarray) -> None:
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.scales = np.array([0.01, 0.01, 0.01])
    header.offsets = np.array([0.0, 0.0, 0.0])

    las = laspy.LasData(header)
    las.x = np.arange(len(values), dtype=np.float64)
    las.y = np.zeros(len(values), dtype=np.float64)
    las.z = np.zeros(len(values), dtype=np.float64)
    las.add_extra_dim(laspy.ExtraBytesParams(name="PredInstance", type=values.dtype))
    las.PredInstance = values
    las.write(path)


def _write_source_las(path: Path, system_identifier: str, projection_payload: bytes) -> None:
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.array([100.0, 200.0, 300.0])
    header.system_identifier = system_identifier
    header.generating_software = "source-writer"
    header.creation_date = date(2026, 6, 21)
    header.vlrs.append(
        VLR(
            user_id="LASF_Projection",
            record_id=34735,
            description="synthetic projection",
            record_data=projection_payload,
        )
    )

    las = laspy.LasData(header)
    las.x = np.array([100.0])
    las.y = np.array([200.0])
    las.z = np.array([300.0])
    las.write(path)


class InstanceLabelContractTests(unittest.TestCase):
    def test_cli_help_documents_uint32_threshold(self):
        main_remap = (Path(__file__).resolve().parents[1] / "src" / "main_remap.py").read_text()

        self.assertIn("uint32 above 65,535", main_remap)
        self.assertNotIn("63,535", main_remap)

    def test_point_cloud_files_prefers_copc_twins_but_accepts_raw_only_inputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            raw = directory / "source.laz"
            copc = directory / "source.copc.laz"
            raw.write_text("raw")
            copc.write_text("copc")

            self.assertEqual(point_cloud_files(directory), [copc])

            copc.unlink()
            self.assertEqual(point_cloud_files(directory), [raw])

    def test_point_cloud_file_discovery_is_case_insensitive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            raw = directory / "UPPER.LAZ"
            copc = directory / "UPPER.COPC.LAZ"
            las = directory / "OTHER.LAS"
            raw.write_text("raw")
            copc.write_text("copc")
            las.write_text("las")

            self.assertEqual(point_cloud_files(directory), [las, copc])
            self.assertEqual(raw_point_cloud_files(directory), [las, raw])
            self.assertEqual(copc_files(directory), [copc])

    def test_point_cloud_file_discovery_missing_directory_is_empty(self):
        missing = Path("/tmp/smarttile_missing_point_cloud_dir_for_test")

        self.assertEqual(point_cloud_files(missing), [])
        self.assertEqual(raw_point_cloud_files(missing), [])
        self.assertEqual(copc_files(missing), [])

    def test_accepts_background_and_positive_instances(self):
        validate_prediction_instance_labels(np.array([0, 1, 65_535], dtype=np.uint16))

    def test_rejects_negative_prediction_instance_labels(self):
        with self.assertRaisesRegex(ValueError, "SmartTile expects PredInstance=0"):
            validate_prediction_instance_labels(
                np.array([0, -1, 7], dtype=np.int16),
                "PredInstance",
                "tile.laz",
            )

    def test_dtype_selection_still_uses_uint32_only_above_threshold(self):
        self.assertEqual(instance_output_dtype(np.array([0, 65_535], dtype=np.uint32)), np.dtype(np.uint16))
        self.assertEqual(instance_output_dtype(np.array([0, 65_536], dtype=np.uint32)), np.dtype(np.uint32))

    def test_load_tile_fails_fast_on_negative_predinstance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "c00_r00.las"
            _write_las_with_predinstance(path, np.array([0, -1, 1], dtype=np.int16))

            with self.assertRaisesRegex(ValueError, "negative PredInstance values"):
                load_tile(
                    path,
                    {"c00_r00": (0.0, 1.0, 0.0, 0.0)},
                    buffer=10.0,
                    instance_dimension="PredInstance",
                )

    def test_single_source_merged_header_preserves_source_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            tiles_dir = Path(tmpdir) / "tiles"
            input_dir.mkdir()
            tiles_dir.mkdir()
            source = input_dir / "source.las"
            payload = b"\x01\x00source-projection"
            _write_source_las(source, "single-source", payload)

            header = merged_product_header(
                np.array([[100.0, 200.0, 300.0], [101.0, 201.0, 301.0]]),
                input_dir,
                tiles_dir,
            )

            self.assertEqual(header.version, "1.2")
            self.assertEqual(header.point_format.id, 3)
            self.assertEqual(header.system_identifier, "single-source")
            self.assertEqual(header.generating_software, "source-writer")
            self.assertEqual(header.creation_date, date(2026, 6, 21))
            self.assertEqual(list(header.scales), [0.01, 0.01, 0.01])
            self.assertTrue(any(v.user_id == "LASF_Projection" and v.record_id == 34735 for v in header.vlrs))

    def test_multi_source_merged_header_preserves_projection_without_source_identity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            tiles_dir = Path(tmpdir) / "tiles"
            input_dir.mkdir()
            tiles_dir.mkdir()
            payload = b"\x01\x00shared-projection"
            _write_source_las(input_dir / "a.las", "source-a", payload)
            _write_source_las(input_dir / "b.las", "source-b", payload)

            header = merged_product_header(
                np.array([[100.0, 200.0, 300.0], [101.0, 201.0, 301.0]]),
                input_dir,
                tiles_dir,
            )

            self.assertEqual(header.version, "1.4")
            self.assertEqual(header.point_format.id, 6)
            self.assertNotEqual(header.system_identifier, "source-a")
            self.assertNotEqual(header.generating_software, "source-writer")
            self.assertEqual(list(header.scales), [0.01, 0.01, 0.01])
            self.assertEqual(
                [(v.user_id, v.record_id) for v in header.vlrs],
                [("LASF_Projection", 34735)],
            )

    def test_write_loaded_point_cloud_preserves_selected_standard_dimensions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source.las"
            output = Path(tmpdir) / "filtered.laz"
            header = laspy.LasHeader(point_format=3, version="1.2")
            header.scales = np.array([0.001, 0.001, 0.001])
            header.offsets = np.array([100.0, 200.0, 300.0])
            las = laspy.LasData(header)
            las.x = np.array([100.0, 100.1, 100.2])
            las.y = np.array([200.0, 200.1, 200.2])
            las.z = np.array([300.0, 300.1, 300.2])
            las.intensity = np.array([11, 22, 33], dtype=np.uint16)
            las.classification = np.array([2, 3, 4], dtype=np.uint8)
            las.write(source)

            write_loaded_point_cloud(
                source,
                output,
                np.array([[100.0, 200.0, 300.0], [100.2, 200.2, 300.2]]),
                {"PredInstance": np.array([1, 2], dtype=np.uint16)},
                source_indices=np.array([0, 2]),
            )

            out = laspy.read(output)
            np.testing.assert_array_equal(out.intensity, np.array([11, 33], dtype=np.uint16))
            np.testing.assert_array_equal(out.classification, np.array([2, 4], dtype=np.uint8))
            np.testing.assert_array_equal(out.PredInstance, np.array([1, 2], dtype=np.uint16))


if __name__ == "__main__":
    unittest.main()
