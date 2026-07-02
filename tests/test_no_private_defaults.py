import sys
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import prepare_tile_jobs  # noqa: E402
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

    def test_prepare_tile_jobs_accepts_grid_offset_from_tindex_caller(self):
        argv = [
            "prepare_tile_jobs.py",
            "input.gpkg",
            "--tile-length=300",
            "--tile-buffer=20",
            "--jobs-out=jobs.txt",
            "--bounds-out=bounds.json",
            "--grid-offset=1.0",
        ]
        with mock.patch.object(sys, "argv", argv):
            with mock.patch.object(prepare_tile_jobs, "run_get_bounds", return_value={"tile_count": "1"}):
                with mock.patch.object(prepare_tile_jobs, "write_job_list") as write_jobs:
                    prepare_tile_jobs.main()

        write_jobs.assert_called_once_with(Path("bounds.json"), Path("jobs.txt"))


if __name__ == "__main__":
    unittest.main()
