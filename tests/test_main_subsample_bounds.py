import json
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.modules.setdefault("parameters", types.SimpleNamespace(TILE_PARAMS=types.SimpleNamespace()))

from main_subsample import get_file_bounds  # noqa: E402


class PdalBoundsParsingTests(unittest.TestCase):
    def test_get_file_bounds_preserves_scientific_notation_values(self):
        stdout = json.dumps(
            {
                "metadata": {
                    "readers.copc": {
                        "minx": 4.000000047e-07,
                        "maxx": 30.0,
                        "miny": "5.000000058e-07",
                        "maxy": "31.5",
                    }
                }
            }
        )

        with mock.patch(
            "main_subsample.subprocess.run",
            return_value=subprocess.CompletedProcess(["pdal"], 0, stdout=stdout, stderr=""),
        ):
            bounds = get_file_bounds(Path("tile.copc.laz"))

        self.assertEqual(bounds, (4.000000047e-07, 30.0, 5.000000058e-07, 31.5))

    def test_get_file_bounds_accepts_direct_metadata_bounds(self):
        stdout = json.dumps(
            {
                "metadata": {
                    "minx": 0,
                    "maxx": 30,
                    "miny": 0,
                    "maxy": 30,
                }
            }
        )

        with mock.patch(
            "main_subsample.subprocess.run",
            return_value=subprocess.CompletedProcess(["pdal"], 0, stdout=stdout, stderr=""),
        ):
            bounds = get_file_bounds(Path("tile.laz"))

        self.assertEqual(bounds, (0.0, 30.0, 0.0, 30.0))

    def test_get_file_bounds_returns_none_for_invalid_ranges(self):
        stdout = json.dumps(
            {
                "metadata": {
                    "readers.las": {
                        "minx": 30,
                        "maxx": 0,
                        "miny": 0,
                        "maxy": 30,
                    }
                }
            }
        )

        with mock.patch(
            "main_subsample.subprocess.run",
            return_value=subprocess.CompletedProcess(["pdal"], 0, stdout=stdout, stderr=""),
        ):
            bounds = get_file_bounds(Path("tile.laz"))

        self.assertIsNone(bounds)

    def test_get_file_bounds_returns_none_for_invalid_json(self):
        with mock.patch(
            "main_subsample.subprocess.run",
            return_value=subprocess.CompletedProcess(["pdal"], 0, stdout="not-json", stderr=""),
        ):
            bounds = get_file_bounds(Path("tile.laz"))

        self.assertIsNone(bounds)


if __name__ == "__main__":
    unittest.main()
