import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


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


if __name__ == "__main__":
    unittest.main()
