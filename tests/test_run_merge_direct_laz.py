import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from parameters import Parameters  # noqa: E402
    import run  # noqa: E402
except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
    if exc.name not in {"pydantic_settings", "pydantic"}:
        raise
    Parameters = None
    run = None


class RunMergeDirectLazTests(unittest.TestCase):
    @unittest.skipIf(Parameters is None, "pydantic_settings is not installed")
    def test_merge_stage7_uses_laz_lane_when_copc_lane_is_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            segmented = root / "segmented_remapped"
            copc_dir = root / "copc"
            laz_dir = root / "raw_laz"
            out_tiles = root / "output_tiles"
            tile_bounds = root / "tile_bounds_tindex.json"
            for directory in (segmented, copc_dir, laz_dir):
                directory.mkdir()
            tile_bounds.write_text("{}", encoding="utf-8")

            params = Parameters(
                task="merge",
                segmented_remapped_folder=segmented,
                tile_bounds_json=tile_bounds,
                output_tiles_folder=out_tiles,
                output_merged_laz=root / "merged.laz",
                original_copc_input_dir=copc_dir,
                original_laz_input_dir=laz_dir,
                transfer_original_dims_to_merged=False,
                workers=1,
                _cli_parse_args=False,
            )

            with mock.patch.object(run, "_validate_raw_original_lane") as validate_raw:
                with mock.patch.object(run, "_validate_copc_original_lane") as validate_copc:
                    with mock.patch.object(run, "_validate_copc_laz_source_pairs") as validate_pairs:
                        with mock.patch("main_merge.run_merge", return_value=root / "merged.laz") as run_merge:
                            run.run_merge_task(params)

            validate_copc.assert_called_once_with(copc_dir)
            validate_pairs.assert_called_once_with(copc_dir, laz_dir)
            validate_raw.assert_called_once_with(laz_dir, out_tiles.parent / "original_with_predictions")
            run_merge.assert_called_once()
            _, kwargs = run_merge.call_args
            self.assertEqual(kwargs["original_input_dir"], laz_dir)

    @unittest.skipIf(Parameters is None, "pydantic_settings is not installed")
    def test_merge_copc_only_original_lane_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            segmented = root / "segmented_remapped"
            copc_dir = root / "copc"
            tile_bounds = root / "tile_bounds_tindex.json"
            for directory in (segmented, copc_dir):
                directory.mkdir()
            tile_bounds.write_text("{}", encoding="utf-8")

            params = Parameters(
                task="merge",
                segmented_remapped_folder=segmented,
                tile_bounds_json=tile_bounds,
                output_tiles_folder=root / "output_tiles",
                original_copc_input_dir=copc_dir,
                transfer_original_dims_to_merged=False,
                workers=1,
                _cli_parse_args=False,
            )

            with mock.patch("main_merge.run_merge") as run_merge:
                with self.assertRaises(SystemExit):
                    run.run_merge_task(params)

            run_merge.assert_not_called()


if __name__ == "__main__":
    unittest.main()
