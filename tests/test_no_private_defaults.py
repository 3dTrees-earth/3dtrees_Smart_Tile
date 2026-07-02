import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prepare_tile_jobs import DEFAULT_BOUNDS_JSON  # noqa: E402


class NoPrivateDefaultsTests(unittest.TestCase):
    def test_helper_cli_defaults_are_workspace_relative(self):
        self.assertEqual(DEFAULT_BOUNDS_JSON, Path("tile_bounds_tindex.json"))
        self.assertFalse(DEFAULT_BOUNDS_JSON.is_absolute())

    def test_tool_source_has_no_machine_specific_default_paths(self):
        src_dir = Path(__file__).resolve().parents[1] / "src"
        source_text = "\n".join(path.read_text(encoding="utf-8") for path in src_dir.glob("*.py"))
        self.assertNotIn("/home/kg281", source_text)
        self.assertNotIn("pdal_experiments", source_text)


if __name__ == "__main__":
    unittest.main()
