import sys
import tempfile
import unittest
from pathlib import Path

import laspy
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from copc_staging import (  # noqa: E402
    staged_copc_matches_source,
    write_copc_stage_manifest_entry,
)


def _write_las(path: Path, n_points: int = 3) -> None:
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.array([100.0, 200.0, 10.0])
    las = laspy.LasData(header)
    las.x = np.arange(n_points, dtype=np.float64) * 0.1 + 100.0
    las.y = np.arange(n_points, dtype=np.float64) * 0.1 + 200.0
    las.z = np.arange(n_points, dtype=np.float64) * 0.1 + 10.0
    las.write(path)


def _write_las_with_extra_description(path: Path, description: str, n_points: int = 3) -> None:
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.array([100.0, 200.0, 10.0])
    las = laspy.LasData(header)
    las.x = np.arange(n_points, dtype=np.float64) * 0.1 + 100.0
    las.y = np.arange(n_points, dtype=np.float64) * 0.1 + 200.0
    las.z = np.arange(n_points, dtype=np.float64) * 0.1 + 10.0
    las.add_extra_dim(
        laspy.ExtraBytesParams(
            name="PredInstance",
            type=np.uint16,
            description=description,
        )
    )
    las.PredInstance = np.arange(n_points, dtype=np.uint16)
    las.write(path)


class CopcStagingTests(unittest.TestCase):
    def test_manifest_allows_exact_source_and_staged_header_match(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.las"
            staged = root / "source.copc.las"
            _write_las(source)
            _write_las(staged)

            write_copc_stage_manifest_entry(source, staged)

            self.assertTrue(staged_copc_matches_source(staged, source))

    def test_manifest_rejects_changed_source_or_changed_staged_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.las"
            staged = root / "source.copc.las"
            _write_las(source)
            _write_las(staged)
            write_copc_stage_manifest_entry(source, staged)

            _write_las(source, n_points=4)
            self.assertFalse(staged_copc_matches_source(staged, source))

            _write_las(source)
            _write_las(staged, n_points=4)
            self.assertFalse(staged_copc_matches_source(staged, source))

    def test_no_manifest_allows_structural_header_match(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.las"
            staged = root / "source.copc.las"
            _write_las_with_extra_description(source, "source description")
            _write_las_with_extra_description(staged, "")

            self.assertTrue(staged_copc_matches_source(staged, source))

    def test_no_manifest_rejects_changed_structural_header(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.las"
            staged = root / "source.copc.las"
            _write_las_with_extra_description(source, "source description", n_points=3)
            _write_las_with_extra_description(staged, "", n_points=4)

            self.assertFalse(staged_copc_matches_source(staged, source))


if __name__ == "__main__":
    unittest.main()
