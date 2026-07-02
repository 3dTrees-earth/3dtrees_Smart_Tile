import sys
import tempfile
import unittest
from pathlib import Path

import laspy
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from main_remap import remap_single_tile  # noqa: E402


def _write_source(path: Path, offset: np.ndarray, scale: np.ndarray) -> None:
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = offset
    header.scales = scale
    las = laspy.LasData(header)
    las.x = np.array([1000.001, 1000.011, 1000.021])
    las.y = np.array([2000.001, 2000.011, 2000.021])
    las.z = np.array([50.001, 50.011, 50.021])
    las.add_extra_dim(laspy.ExtraBytesParams(name="PredInstance", type=np.uint16))
    las.PredInstance = np.array([1, 2, 3], dtype=np.uint16)
    las.write(path)


def _write_target(path: Path, offset: np.ndarray, scale: np.ndarray) -> np.ndarray:
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = offset
    header.scales = scale
    las = laspy.LasData(header)
    expected_xyz = np.array(
        [
            [1000.001, 2000.001, 50.001],
            [1000.011, 2000.011, 50.011],
            [1000.021, 2000.021, 50.021],
        ]
    )
    las.x = expected_xyz[:, 0]
    las.y = expected_xyz[:, 1]
    las.z = expected_xyz[:, 2]
    las.write(path)
    return expected_xyz


class RemapScalePreservationTests(unittest.TestCase):
    def test_output_scales_do_not_move_target_coordinates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.las"
            target = root / "target.las"
            output = root / "out.laz"
            offset = np.array([1000.0, 2000.0, 50.0])

            _write_source(source, offset, np.array([0.001, 0.001, 0.001]))
            expected_xyz = _write_target(target, offset, np.array([0.001, 0.001, 0.001]))

            _, success, message, point_count = remap_single_tile(
                source,
                target,
                output,
                instance_dimension="PredInstance",
                output_scales=(0.01, 0.01, 0.01),
            )

            self.assertTrue(success, message)
            self.assertEqual(point_count, len(expected_xyz))
            remapped = laspy.read(output)
            actual_xyz = np.vstack([remapped.x, remapped.y, remapped.z]).T
            np.testing.assert_allclose(actual_xyz, expected_xyz, atol=0.005)
            np.testing.assert_allclose(remapped.header.scales, np.array([0.01, 0.01, 0.01]))
            np.testing.assert_array_equal(remapped.PredInstance, np.array([1, 2, 3], dtype=np.uint16))


if __name__ == "__main__":
    unittest.main()
