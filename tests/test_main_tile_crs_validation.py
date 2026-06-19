import sys
import tempfile
import types
import unittest
from pathlib import Path

import laspy
import numpy as np
from laspy.vlrs.vlr import VLR


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.modules.setdefault("plot_tiles_and_copc", types.SimpleNamespace())
sys.modules.setdefault("parameters", types.SimpleNamespace(TILE_PARAMS=types.SimpleNamespace()))

from main_tile import _copc_preserves_source_crs  # noqa: E402


def _write_las(path: Path, projection_payload: bytes | None) -> None:
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.scales = np.array([0.01, 0.01, 0.01])
    header.offsets = np.array([0.0, 0.0, 0.0])
    if projection_payload is not None:
        header.vlrs.append(
            VLR(
                user_id="LASF_Projection",
                record_id=2112,
                description="synthetic CRS metadata",
                record_data=projection_payload,
            )
        )

    las = laspy.LasData(header)
    las.x = np.array([0.0])
    las.y = np.array([0.0])
    las.z = np.array([0.0])
    las.write(path)


class CopcCrsValidationTests(unittest.TestCase):
    def test_accepts_matching_projection_vlr(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            source = tmp_path / "source.las"
            output = tmp_path / "output.las"
            payload = b'LOCAL_CS["3dtrees-test"]'
            _write_las(source, payload)
            _write_las(output, payload)

            ok, message = _copc_preserves_source_crs(source, output)

            self.assertTrue(ok, message)

    def test_rejects_missing_projection_vlr(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            source = tmp_path / "source.las"
            output = tmp_path / "output.las"
            _write_las(source, b'LOCAL_CS["3dtrees-test"]')
            _write_las(output, None)

            ok, message = _copc_preserves_source_crs(source, output)

            self.assertFalse(ok)
            self.assertIn("missing or changed", message)

    def test_rejects_changed_projection_vlr(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            source = tmp_path / "source.las"
            output = tmp_path / "output.las"
            _write_las(source, b'LOCAL_CS["3dtrees-test"]')
            _write_las(output, b'LOCAL_CS["different"]')

            ok, message = _copc_preserves_source_crs(source, output)

            self.assertFalse(ok)
            self.assertIn("missing or changed", message)

    def test_accepts_source_without_crs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            source = tmp_path / "source.las"
            output = tmp_path / "output.las"
            _write_las(source, None)
            _write_las(output, None)

            ok, message = _copc_preserves_source_crs(source, output)

            self.assertTrue(ok, message)


if __name__ == "__main__":
    unittest.main()
