import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from parameters import Parameters
import run
from test_dense_tile_merge import write_cloud
from test_strict_prediction_pipeline import layout
import laspy


class RunFilterTaskTests(unittest.TestCase):
    def test_filter_preserves_same_tile_points_and_uses_dense_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input"
            source.mkdir()
            write_cloud(source / "a.las", [0, 0, .1], [1, 1, 1])
            params = Parameters(task="filter", input_dir=source, output_dir=root / "output",
                                tile_bounds_json=layout(root, 1), _cli_parse_args=False)
            run.run_filter_task(params)
            self.assertEqual(len(laspy.read(root / "output/tile_00000.laz").points), 3)

    def test_filter_requires_layout_and_refuses_in_place_operation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for params in [
                Parameters(task="filter", input_dir=root, output_dir=root, _cli_parse_args=False),
                Parameters(task="filter", input_dir=root, output_dir=root / "out", _cli_parse_args=False),
            ]:
                with self.assertRaises(SystemExit):
                    run.run_filter_task(params)


if __name__ == "__main__":
    unittest.main()
