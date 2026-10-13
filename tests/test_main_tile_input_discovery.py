import sys
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import laspy


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.modules.setdefault(
    "plot_tiles_and_copc",
    types.SimpleNamespace(plot_extents=lambda *_args, **_kwargs: None),
)

import main_tile  # noqa: E402


class MainTileInputDiscoveryTests(unittest.TestCase):
    def test_tiling_input_files_prefer_copc_twin(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir)
            (input_dir / "source.laz").write_text("raw")
            (input_dir / "source.copc.laz").write_text("copc")
            (input_dir / "other.las").write_text("raw")

            files = [path.name for path in main_tile._tiling_input_files(input_dir)]

        self.assertEqual(files, ["other.las", "source.copc.laz"])

    def test_small_single_input_is_planned_as_one_whole_cloud_tile(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            source = input_dir / "source.copc.laz"
            las = laspy.LasData(laspy.LasHeader(point_format=0, version="1.4"))
            las.x = [100., 782.]; las.y = [200., 772.]; las.z = [0., 10.]
            las.write(source)
            (output_dir / "bounds.json").write_text(json.dumps({
                "tile_buffer": 20, "tiles": [
                    {"col": c, "row": r, "bounds": [[100+c*300, 400+c*300], [200+r*300, 500+r*300]]}
                    for c in range(3) for r in range(2)]}))

            def fake_build_tindex(_input_dir, output_gpkg):
                output_gpkg.parent.mkdir(parents=True, exist_ok=True)
                output_gpkg.write_bytes(b"gpkg")
                return output_gpkg

            with mock.patch.object(main_tile, "build_tindex", side_effect=fake_build_tindex):
                with mock.patch.object(
                    main_tile,
                    "calculate_tile_bounds",
                    return_value=(output_dir / "jobs.txt", output_dir / "bounds.json", {}),
                ):
                    with mock.patch.object(main_tile.plot_tiles_and_copc, "plot_extents") as plot:
                        plan = main_tile.plan_tiling(input_dir, output_dir, 100, 10, tiling_threshold=10000)

            self.assertTrue(plan.skip_tiling)
            self.assertEqual(plan.source_files, [source])
            plot.assert_called_once()
            layout = json.loads((output_dir / "bounds.json").read_text())
            self.assertTrue(layout["tiling_skipped"])
            self.assertEqual(layout["tile_buffer"], 0)
            self.assertEqual(len(layout["tiles"]), 1)
            self.assertEqual(layout["tiles"][0]["core"], [[100., 782.], [200., 772.]])
            from dense_instance_ownership import ownership_regions
            regions = ownership_regions([(source, source, "source")], output_dir / "bounds.json")
            self.assertEqual(regions[0]["core"], [[100., 782.], [200., 772.]])
            self.assertTrue(all(n is None for n in regions[0]["neighbors"].values()))


if __name__ == "__main__":
    unittest.main()
