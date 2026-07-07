import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filter_task_support import derive_tile_buffer_from_json  # noqa: E402


class TileBufferMetadataTests(unittest.TestCase):
    def test_derive_tile_buffer_prefers_root_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tile_bounds_tindex.json"
            path.write_text(json.dumps({"tile_buffer": 18.5, "tiles": []}), encoding="utf-8")

            self.assertEqual(derive_tile_buffer_from_json(path), 18.5)

    def test_derive_tile_buffer_can_use_bounds_and_core_padding(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tile_bounds_tindex.json"
            path.write_text(
                json.dumps(
                    {
                        "tiles": [
                            {
                                "bounds": [[80.0, 220.0], [180.0, 320.0]],
                                "core": [[100.0, 200.0], [200.0, 300.0]],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(derive_tile_buffer_from_json(path), 20.0)


if __name__ == "__main__":
    unittest.main()
