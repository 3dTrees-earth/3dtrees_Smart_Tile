import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import tile_copc  # noqa: E402


class TileCopcDimensionPolicyTests(unittest.TestCase):
    def setUp(self):
        self._previous_empty_dims_supported = tile_copc._UNTWINE_EMPTY_DIMS_SUPPORTED
        tile_copc._UNTWINE_EMPTY_DIMS_SUPPORTED = None

    def tearDown(self):
        tile_copc._UNTWINE_EMPTY_DIMS_SUPPORTED = self._previous_empty_dims_supported

    def test_pdal_copc_conversion_strips_extra_dims_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_laz = tmp_path / "input.laz"
            output_copc = tmp_path / "output.copc.laz"
            input_laz.write_text("input")
            seen_pipeline = {}

            def run_pipeline(command, **_):
                with open(command[-1]) as handle:
                    seen_pipeline.update(json.load(handle))
                output_copc.write_text("copc")
                return subprocess.CompletedProcess(command, 0, "", "")

            with mock.patch("tile_copc.subprocess.run", side_effect=run_pipeline):
                self.assertTrue(tile_copc.convert_laz_to_copc_pdal(input_laz, output_copc))

            writer = seen_pipeline["pipeline"][-1]
            self.assertEqual(writer["type"], "writers.copc")
            self.assertNotIn("extra_dims", writer)

    def test_pdal_copc_conversion_can_preserve_extra_dims_for_merged_products(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_laz = tmp_path / "input.laz"
            output_copc = tmp_path / "output.copc.laz"
            input_laz.write_text("input")
            seen_pipeline = {}

            def run_pipeline(command, **_):
                with open(command[-1]) as handle:
                    seen_pipeline.update(json.load(handle))
                output_copc.write_text("copc")
                return subprocess.CompletedProcess(command, 0, "", "")

            with mock.patch("tile_copc.subprocess.run", side_effect=run_pipeline):
                self.assertTrue(
                    tile_copc.convert_laz_to_copc_pdal(
                        input_laz,
                        output_copc,
                        preserve_extra_dims=True,
                    )
                )

            writer = seen_pipeline["pipeline"][-1]
            self.assertEqual(writer["extra_dims"], "all")

    def test_untwine_finalizer_strips_extra_dims_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            part = tmp_path / "part_0.las"
            final = tmp_path / "tile.copc.laz"
            part.write_text("part")

            def run_untwine(*_, **__):
                final.write_text("copc")
                return subprocess.CompletedProcess(["untwine"], 0, "", "")

            with mock.patch("tile_copc.shutil.which", return_value="/usr/bin/untwine"):
                with mock.patch("tile_copc._output_has_no_extra_dimensions", return_value=True):
                    with mock.patch(
                        "tile_copc.finalize_tile_to_copc_pdal",
                        return_value=(True, "OK"),
                    ) as pdal:
                        with mock.patch("tile_copc.subprocess.run", side_effect=run_untwine) as run:
                            success, message = tile_copc.finalize_tile_to_copc_untwine(
                                [part],
                                final,
                                tmp_path,
                                "tile",
                            )

            self.assertTrue(success)
            self.assertEqual(message, "untwine-stripped")
            command = run.call_args.args[0]
            self.assertIn("--dims", command)
            self.assertEqual(
                command[command.index("--dims") + 1],
                tile_copc.UNTWINE_STRIP_EXTRA_DIMS_ARG,
            )
            pdal.assert_not_called()

    def test_untwine_finalizer_strips_to_temp_laz_when_direct_dims_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            part = tmp_path / "part_0.las"
            final = tmp_path / "tile.copc.laz"
            part.write_text("part")

            with mock.patch("tile_copc.shutil.which", return_value="/usr/bin/untwine"):
                with mock.patch(
                    "tile_copc.finalize_tile_to_copc_pdal",
                    return_value=(True, "OK"),
                ) as pdal:
                    with mock.patch("tile_copc._strip_las_to_standard_dims", return_value=(True, "stripped")) as strip:
                        with mock.patch(
                            "tile_copc._run_untwine",
                            side_effect=[(False, "untwine failed"), (True, "untwine")],
                        ) as untwine:
                            with mock.patch("tile_copc._output_has_no_extra_dimensions", return_value=True):
                                success, message = tile_copc.finalize_tile_to_copc_untwine(
                                    [part],
                                    final,
                                    tmp_path,
                                    "tile",
                                )

            self.assertTrue(success)
            self.assertEqual(message, "pdal-strip+untwine")
            strip.assert_called_once()
            self.assertEqual(untwine.call_count, 2)
            self.assertTrue(untwine.call_args_list[0].kwargs["strip_extra_dims"])
            self.assertFalse(untwine.call_args_list[1].kwargs["strip_extra_dims"])
            pdal.assert_not_called()

    def test_standard_dimension_staging_uses_minimal_las_point_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_laz = tmp_path / "input.laz"
            output_laz = tmp_path / "stripped.laz"
            input_laz.write_text("input")
            seen_pipeline = {}

            def run_pipeline(command, **_):
                with open(command[-1]) as handle:
                    seen_pipeline.update(json.load(handle))
                output_laz.write_text("laz")
                return subprocess.CompletedProcess(command, 0, "", "")

            with mock.patch("tile_copc.subprocess.run", side_effect=run_pipeline):
                success, message = tile_copc._strip_las_to_standard_dims([input_laz], output_laz)

            self.assertTrue(success, message)
            writer = seen_pipeline["pipeline"][-1]
            self.assertEqual(writer["type"], "writers.las")
            self.assertEqual(writer["minor_version"], 2)
            self.assertEqual(writer["dataformat_id"], 0)
            self.assertNotIn("extra_dims", writer)

    def test_run_untwine_uses_empty_dimension_keep_list_for_stripping(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_laz = tmp_path / "input.laz"
            output_copc = tmp_path / "output.copc.laz"
            input_laz.write_text("input")

            def run_untwine(*_, **__):
                output_copc.write_text("copc")
                return subprocess.CompletedProcess(["untwine"], 0, "", "")

            with mock.patch("tile_copc.shutil.which", return_value="/usr/bin/untwine"):
                with mock.patch("tile_copc.subprocess.run", side_effect=run_untwine) as run:
                    success, message = tile_copc._run_untwine(
                        [input_laz],
                        output_copc,
                        "EPSG:32632",
                        strip_extra_dims=True,
                    )

            self.assertTrue(success, message)
            command = run.call_args.args[0]
            self.assertIn("--dims", command)
            self.assertEqual(
                command[command.index("--dims") + 1],
                tile_copc.UNTWINE_STRIP_EXTRA_DIMS_ARG,
            )
            self.assertTrue(tile_copc._UNTWINE_EMPTY_DIMS_SUPPORTED)

    def test_run_untwine_caches_unsupported_empty_dimension_keep_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_laz = tmp_path / "input.laz"
            output_copc = tmp_path / "output.copc.laz"
            input_laz.write_text("input")

            def reject_empty_dims(command, **_):
                self.assertIn("--dims", command)
                return subprocess.CompletedProcess(
                    command,
                    255,
                    "",
                    "Untwine Error: Missing value for argument 'dims'.",
                )

            with mock.patch("tile_copc.shutil.which", return_value="/usr/bin/untwine"):
                with mock.patch("tile_copc.subprocess.run", side_effect=reject_empty_dims) as run:
                    success, message = tile_copc._run_untwine(
                        [input_laz],
                        output_copc,
                        "EPSG:32632",
                        strip_extra_dims=True,
                    )
                    self.assertFalse(success)
                    self.assertIn("untwine failed", message)
                    self.assertFalse(tile_copc._UNTWINE_EMPTY_DIMS_SUPPORTED)

                    success, message = tile_copc._run_untwine(
                        [input_laz],
                        output_copc,
                        "EPSG:32632",
                        strip_extra_dims=True,
                    )

            self.assertFalse(success)
            self.assertEqual(message, "untwine empty --dims unsupported")
            self.assertEqual(run.call_count, 1)

    def test_convert_laz_to_copc_uses_direct_untwine_dims_when_supported(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_laz = tmp_path / "input.laz"
            output_copc = tmp_path / "output.copc.laz"
            input_laz.write_text("input")

            def run_untwine(*_, **__):
                output_copc.write_text("copc")
                return subprocess.CompletedProcess(["untwine"], 0, "", "")

            with mock.patch("tile_copc.shutil.which", return_value="/usr/bin/untwine"):
                with mock.patch("tile_copc._output_has_no_extra_dimensions", return_value=True):
                    with mock.patch("tile_copc.srs_assignment_from_file", return_value="EPSG:32632"):
                        with mock.patch("tile_copc.append_source_geotiff_projection_evlrs"):
                            with mock.patch("tile_copc.copc_preserves_source_crs", return_value=(True, "ok")):
                                with mock.patch("tile_copc.subprocess.run", side_effect=run_untwine) as run:
                                    self.assertTrue(tile_copc.convert_laz_to_copc(input_laz, output_copc))

            command = run.call_args.args[0]
            self.assertIn("--dims", command)
            self.assertEqual(
                command[command.index("--dims") + 1],
                tile_copc.UNTWINE_STRIP_EXTRA_DIMS_ARG,
            )

    def test_convert_laz_to_copc_strips_to_temp_laz_when_direct_dims_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_laz = tmp_path / "input.laz"
            output_copc = tmp_path / "output.copc.laz"
            input_laz.write_text("input")

            with mock.patch("tile_copc.shutil.which", return_value="/usr/bin/untwine"):
                with mock.patch("tile_copc._strip_las_to_standard_dims", return_value=(True, "stripped")) as strip:
                    with mock.patch(
                        "tile_copc._run_untwine",
                        side_effect=[(False, "untwine failed"), (True, "untwine")],
                    ) as untwine:
                        with mock.patch("tile_copc.convert_laz_to_copc_pdal") as pdal:
                            with mock.patch("tile_copc.append_source_geotiff_projection_evlrs", return_value=(True, "ok")):
                                with mock.patch("tile_copc.copc_preserves_source_crs", return_value=(True, "ok")):
                                    with mock.patch("tile_copc._output_has_no_extra_dimensions", return_value=True):
                                        self.assertTrue(tile_copc.convert_laz_to_copc(input_laz, output_copc))

            strip.assert_called_once()
            self.assertEqual(untwine.call_count, 2)
            self.assertTrue(untwine.call_args_list[0].kwargs["strip_extra_dims"])
            self.assertFalse(untwine.call_args_list[1].kwargs["strip_extra_dims"])
            pdal.assert_not_called()

if __name__ == "__main__":
    unittest.main()
