import sys
import tempfile
import types
import unittest
from io import StringIO
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from parameters import Parameters  # noqa: E402
    import tile_task  # noqa: E402
except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
    if exc.name not in {"pydantic_settings", "pydantic"}:
        raise
    Parameters = None
    tile_task = None


class StandaloneTileTaskTests(unittest.TestCase):
    @unittest.skipIf(Parameters is None, "pydantic_settings is not installed")
    def test_standalone_tile_task_forwards_current_tiling_and_subsample_api(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "input"
            output_dir = root / "output"
            tiles_dir = output_dir / "tiles_100m"
            res1_dir = output_dir / "subsampled_res1"
            res2_dir = output_dir / "subsampled_res2"
            input_dir.mkdir()

            params = Parameters(
                task="tile",
                input_dir=input_dir,
                output_dir=output_dir,
                workers=3,
                threads=5,
                num_spatial_chunks=7,
                subsampling_method="nearest-to-centroid",
                _cli_parse_args=False,
            )

            fake_main_tile = types.SimpleNamespace(
                run_tiling_pipeline=mock.Mock(return_value=tiles_dir),
            )
            fake_main_subsample = types.SimpleNamespace(
                run_subsample_pipeline=mock.Mock(return_value=(res1_dir, res2_dir)),
            )

            stdout = StringIO()
            with (
                mock.patch.dict(
                    sys.modules,
                    {
                        "main_tile": fake_main_tile,
                        "main_subsample": fake_main_subsample,
                    },
                ),
                mock.patch("sys.stdout", stdout),
            ):
                tile_task.run_tile_task(params)

            self.assertNotIn("Dimension reduction", stdout.getvalue())

            fake_main_tile.run_tiling_pipeline.assert_called_once()
            _, tiling_kwargs = fake_main_tile.run_tiling_pipeline.call_args
            self.assertTrue(tiling_kwargs["dimension_reduction"])
            self.assertEqual(tiling_kwargs["chunk_size"], 20_000_000)
            self.assertNotIn("chunkwise_copc_source_creation", tiling_kwargs)

            fake_main_subsample.run_subsample_pipeline.assert_called_once()
            _, subsample_kwargs = fake_main_subsample.run_subsample_pipeline.call_args
            self.assertTrue(subsample_kwargs["dimension_reduction"])
            self.assertEqual(subsample_kwargs["subsampling_method"], "nearest-to-centroid")
            self.assertEqual(subsample_kwargs["num_threads"], 7)
            self.assertNotIn("fallback_laz_dir", subsample_kwargs)
            self.assertNotIn("chunk_size", subsample_kwargs)


if __name__ == "__main__":
    unittest.main()
