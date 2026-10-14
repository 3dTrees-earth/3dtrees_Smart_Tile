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
from point_cloud_metadata import copc_files, point_cloud_files, raw_point_cloud_files  # noqa: E402

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


if __name__ == "__main__":
    unittest.main()
