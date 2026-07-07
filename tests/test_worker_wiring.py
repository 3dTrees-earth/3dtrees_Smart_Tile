import inspect
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from worker_budget import DEFAULT_FILE_WORKERS  # noqa: E402


class WorkerWiringTests(unittest.TestCase):
    def test_main_remap_cli_defaults_to_two_file_workers_and_cpu_spatial_workers(self):
        import main_remap  # noqa: E402

        argv = [
            "main_remap.py",
            "--source_folder",
            "/tmp/source",
            "--target_folder",
            "/tmp/target",
            "--output_folder",
            "/tmp/out",
        ]
        with mock.patch("sys.argv", argv):
            with mock.patch("main_remap.available_cpu_count", return_value=13):
                with mock.patch("main_remap.remap_all_tiles", return_value=Path("/tmp/out")) as remap:
                    main_remap.main()

        remap.assert_called_once()
        self.assertEqual(remap.call_args.kwargs["num_workers"], DEFAULT_FILE_WORKERS)
        self.assertEqual(remap.call_args.kwargs["spatial_workers"], 13)

    def test_subsample_pipeline_defaults_spatial_chunks_to_available_cpus(self):
        import main_subsample  # noqa: E402

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tiles = root / "tiles"
            tiles.mkdir()
            res1_file = root / "subsampled_res1" / "tile.copc.laz"
            res2_file = root / "subsampled_res2" / "tile.laz"

            with mock.patch("main_subsample.get_cpu_count", return_value=9):
                with mock.patch(
                    "main_subsample.subsample_parallel",
                    side_effect=[[res1_file], [res2_file]],
                ) as subsample:
                    main_subsample.run_subsample_pipeline(tiles)

        self.assertEqual(subsample.call_count, 2)
        for call in subsample.call_args_list:
            self.assertEqual(call.kwargs["num_cores"], 9)
            self.assertEqual(call.kwargs["num_threads"], 9)

    def test_main_tile_standalone_defaults_use_two_file_policy(self):
        fake_plot = types.SimpleNamespace(plot_extents=mock.Mock())
        with mock.patch.dict(sys.modules, {"plot_tiles_and_copc": fake_plot}):
            import main_tile  # noqa: E402

            signature = inspect.signature(main_tile.run_tiling_pipeline)
            self.assertEqual(signature.parameters["num_workers"].default, DEFAULT_FILE_WORKERS)
            self.assertEqual(signature.parameters["max_tile_procs"].default, DEFAULT_FILE_WORKERS)

            with tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                input_dir = root / "input"
                output_dir = root / "output"
                input_dir.mkdir()
                argv = [
                    "main_tile.py",
                    "--input_dir",
                    str(input_dir),
                    "--output_dir",
                    str(output_dir),
                ]
                with mock.patch("sys.argv", argv):
                    with mock.patch("main_tile.run_tiling_pipeline", return_value=output_dir / "tiles") as tile:
                        main_tile.main()

        tile.assert_called_once()
        self.assertEqual(tile.call_args.kwargs["num_workers"], DEFAULT_FILE_WORKERS)
        self.assertEqual(tile.call_args.kwargs["max_tile_procs"], DEFAULT_FILE_WORKERS)


if __name__ == "__main__":
    unittest.main()
