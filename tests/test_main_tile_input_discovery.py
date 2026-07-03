import sys
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
sys.modules.setdefault(
    "tile_tindex",
    types.SimpleNamespace(
        bounds_overlap=lambda *_args, **_kwargs: False,
        build_tindex=lambda *_args, **_kwargs: None,
        calculate_tile_bounds=lambda *_args, **_kwargs: None,
        filter_source_files_for_tile=lambda *_args, **_kwargs: [],
        get_bounds=lambda *_args, **_kwargs: None,
        get_pdal_path=lambda: "pdal",
        get_pdal_wrench_path=lambda: "pdal_wrench",
        get_source_bounds_from_tindex=lambda *_args, **_kwargs: {},
        get_source_files_from_tindex=lambda *_args, **_kwargs: [],
        parse_proj_bounds=lambda *_args, **_kwargs: None,
        update_tile_bounds_json_from_files=lambda *_args, **_kwargs: 0,
    ),
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
            source.write_bytes(b"copc")

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
            self.assertEqual((result / "source.copc.laz").read_bytes(), b"copc")
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
