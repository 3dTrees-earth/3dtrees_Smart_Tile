import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from parameters import Parameters
import run
from test_merge_stages import write_cloud
from test_remap_first_pipeline import layout
import laspy
from output_tiles import output_tile  # noqa: E402


class MergeFilterOnlyTests(unittest.TestCase):
    def test_filter_preserves_same_tile_points_and_uses_dense_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input"
            source.mkdir()
            write_cloud(source / "a.las", [0, 0, .1], [1, 1, 1])
            params = Parameters(task="merge", segmented_remapped_folder=source, output_dir=root / "output",
                                tile_bounds_json=layout(root, 1), skip_merged_file=True, _cli_parse_args=False)
            run.run_merge_task(params)
            self.assertEqual(len(laspy.read(output_tile(root / "output/output_tiles", 0)).points), 3)
            # Everything stays inside --output-dir.
            self.assertTrue((root / "output/remap_first_report.json").is_file())
            self.assertEqual(sorted(p.name for p in root.iterdir()), ["bounds.json", "input", "output"])

    def test_filter_requires_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            params = Parameters(task="merge", segmented_remapped_folder=root, output_dir=root / "out",
                                _cli_parse_args=False)
            with self.assertRaises(SystemExit):
                run.run_merge_task(params)


if __name__ == "__main__":
    unittest.main()
