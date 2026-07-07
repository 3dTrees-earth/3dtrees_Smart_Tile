import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import merge_tiles  # noqa: E402


class MergeTilesTileBufferTests(unittest.TestCase):
    def test_core_merge_uses_tile_bounds_json_buffer_for_already_tiled_inputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "segmented_remapped"
            original_tiles_dir = root / "subsampled_res1"
            output_tiles_dir = root / "output_tiles"
            tile_bounds = root / "tile_bounds_tindex.json"
            for directory in (input_dir, original_tiles_dir):
                directory.mkdir()
            tile_bounds.write_text(
                json.dumps({"tile_buffer": 27.0, "tiles": []}),
                encoding="utf-8",
            )

            stdout = io.StringIO()
            with mock.patch.object(merge_tiles, "_merge_input_files", return_value=[]):
                with redirect_stdout(stdout):
                    merge_tiles.merge_tiles(
                        input_dir=input_dir,
                        original_tiles_dir=original_tiles_dir,
                        output_merged=root / "merged.laz",
                        output_tiles_dir=output_tiles_dir,
                        tile_bounds_json=tile_bounds,
                    )

            self.assertIn("Buffer: 27.0m (from tile_bounds_tindex.json)", stdout.getvalue())

    def test_core_merge_does_not_accept_manual_buffer_argument(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "segmented_remapped"
            original_tiles_dir = root / "subsampled_res1"
            output_tiles_dir = root / "output_tiles"
            tile_bounds = root / "tile_bounds_tindex.json"
            for directory in (input_dir, original_tiles_dir):
                directory.mkdir()
            tile_bounds.write_text(
                json.dumps({"tile_buffer": 27.0, "tiles": []}),
                encoding="utf-8",
            )

            with self.assertRaises(TypeError):
                merge_tiles.merge_tiles(
                    input_dir=input_dir,
                    original_tiles_dir=original_tiles_dir,
                    output_merged=root / "merged.laz",
                    output_tiles_dir=output_tiles_dir,
                    tile_bounds_json=tile_bounds,
                    buffer=99.0,
                )


if __name__ == "__main__":
    unittest.main()
