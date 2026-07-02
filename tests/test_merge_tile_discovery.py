import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from merge_tile_loading import merge_tile_name  # noqa: E402
from merge_tiles import _merge_input_files  # noqa: E402


class MergeTileDiscoveryTests(unittest.TestCase):
    def test_merge_input_files_include_mixed_laz_and_las(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "c00_r00_segmented_remapped.laz").write_text("placeholder")
            (root / "c01_r00_segmented_remapped.las").write_text("placeholder")

            files = [path.name for path in _merge_input_files(root)]

        self.assertEqual(
            files,
            ["c00_r00_segmented_remapped.laz", "c01_r00_segmented_remapped.las"],
        )

    def test_merge_input_files_prefer_copc_twin(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "c00_r00_segmented_remapped.laz").write_text("placeholder")
            (root / "c00_r00_segmented_remapped.copc.laz").write_text("placeholder")

            files = [path.name for path in _merge_input_files(root)]

        self.assertEqual(files, ["c00_r00_segmented_remapped.copc.laz"])

    def test_merge_tile_name_strips_copc_and_processing_suffixes(self):
        self.assertEqual(
            merge_tile_name(Path("c00_r00_segmented_remapped.copc.laz")),
            "c00_r00",
        )
        self.assertEqual(
            merge_tile_name(Path("c01_r00_segmented.las")),
            "c01_r00",
        )


if __name__ == "__main__":
    unittest.main()
