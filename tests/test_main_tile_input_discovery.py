import sys
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import laspy
import numpy as np


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

    def test_small_single_copc_input_is_reused_for_skip_tiling(self):
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
                    with mock.patch.object(main_tile.plot_tiles_and_copc, "plot_extents"):
                        with mock.patch.object(main_tile, "_convert_laz_to_copc") as convert:
                            result = main_tile.run_tiling_pipeline(
                                input_dir=input_dir,
                                output_dir=output_dir,
                                tile_length=100,
                                tile_buffer=10,
                                tiling_threshold=10000,
                            )

            self.assertEqual(result, output_dir / "copc_single")
            self.assertEqual((result / "source.copc.laz").read_bytes(), source.read_bytes())
            layout = json.loads((output_dir / "bounds.json").read_text())
            self.assertTrue(layout["tiling_skipped"])
            self.assertEqual(layout["tile_buffer"], 0)
            self.assertEqual(len(layout["tiles"]), 1)
            self.assertEqual(layout["tiles"][0]["core"], [[100., 782.], [200., 772.]])
            from dense_instance_ownership import ownership_regions
            regions = ownership_regions([(source, source, "source")], output_dir / "bounds.json")
            self.assertEqual(regions[0]["core"], [[100., 782.], [200., 772.]])
            self.assertTrue(all(n is None for n in regions[0]["neighbors"].values()))
            convert.assert_not_called()

    def test_single_source_range_tasks_split_large_source_with_bounded_chunks(self):
        task = (0, "/tmp/source.laz", [("tile", (0, 0, 10, 10))], Path("/tmp/tiles"), 10, 20_000_000)

        tasks, worker_chunk_size = main_tile._single_source_range_tasks(
            task,
            total_points=100_000_000,
            max_parallel=5,
            chunk_size=20_000_000,
        )

        self.assertEqual(len(tasks), 5)
        self.assertEqual(worker_chunk_size, 4_000_000)
        self.assertEqual(tasks[0][6:9], (0, 20_000_000, 4_000_000))
        self.assertEqual(tasks[-1][6:9], (80_000_000, 20_000_000, 4_000_000))

    def test_distribute_source_file_can_read_one_point_range(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.las"
            tiles_dir = root / "tiles"
            tiles_dir.mkdir()

            header = laspy.LasHeader(point_format=6, version="1.4")
            header.scales = np.array([0.01, 0.01, 0.01])
            header.offsets = np.array([0.0, 0.0, 0.0])
            las = laspy.LasData(header)
            las.x = np.arange(10, dtype=float)
            las.y = np.zeros(10)
            las.z = np.zeros(10)
            las.write(source)

            results = main_tile._distribute_source_file(
                (
                    0,
                    str(source),
                    [("tile", (-1.0, -1.0, 20.0, 20.0))],
                    tiles_dir,
                    1,
                    10,
                    3,
                    4,
                    2,
                )
            )

            self.assertEqual(results, [("tile", 4)])
            part_files = sorted((tiles_dir / "tile").glob("part_*.las"))
            self.assertEqual(len(part_files), 2)
            point_counts = []
            for part_file in part_files:
                with laspy.open(part_file) as reader:
                    point_counts.append(reader.header.point_count)
            self.assertEqual(sum(point_counts), 4)


if __name__ == "__main__":
    unittest.main()
