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

    def test_untwine_finalizer_accepts_reduced_standard_dimensions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            part = tmp_path / "part_0.las"
            final = tmp_path / "tile.copc.laz"
            import laspy
            laspy.LasData(laspy.LasHeader()).write(part)

            def run_untwine(*_, **__):
                final.write_text("copc")
                return subprocess.CompletedProcess(["untwine"], 0, "", "")

            with mock.patch("tile_copc.shutil.which", return_value="/usr/bin/untwine"):
                with mock.patch(
                    "tile_copc.finalize_tile_to_copc_pdal",
                    return_value=(True, "OK"),
                ) as pdal:
                    with mock.patch("tile_copc._output_has_no_extra_dimensions", return_value=True):
                        with mock.patch("tile_copc.subprocess.run", side_effect=run_untwine) as run:
                            success, message = tile_copc.finalize_tile_to_copc_untwine(
                                [part],
                                final,
                                tmp_path,
                                "tile",
                            )

            self.assertTrue(success)
            self.assertEqual(message, "untwine-standard-dimensions")
            command = run.call_args.args[0]
            self.assertNotIn("--dims", command)
            pdal.assert_not_called()

    def test_untwine_finalizer_falls_back_to_pdal_when_dims_classification_keeps_extras(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            part = tmp_path / "part_0.las"
            final = tmp_path / "tile.copc.laz"
            part.write_text("part")

            with mock.patch("tile_copc.shutil.which", return_value="/usr/bin/untwine"):
                with mock.patch("tile_copc._run_untwine", return_value=(True, "untwine")) as untwine:
                    with mock.patch("tile_copc._output_has_no_extra_dimensions", return_value=False):
                        with mock.patch(
                            "tile_copc.finalize_tile_to_copc_pdal",
                            return_value=(True, "pdal-stripped"),
                        ) as pdal:
                            success, message = tile_copc.finalize_tile_to_copc_untwine(
                                [part],
                                final,
                                tmp_path,
                                "tile",
                            )

            self.assertTrue(success)
            self.assertEqual(message, "pdal-stripped")
            untwine.assert_called_once()
            self.assertTrue(untwine.call_args.kwargs["strip_extra_dims"])
            pdal.assert_called_once_with(
                [part],
                final,
                tmp_path,
                "tile",
                tile_bounds=None,
                preserve_extra_dims=False,
            )

    def test_tile_finalizer_retries_pdal_when_untwine_dims_crs_validation_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            tiles_dir = tmp_path / "tiles"
            log_dir = tmp_path / "logs"
            tile_dir = tiles_dir / "c0_r0"
            tile_dir.mkdir(parents=True)
            log_dir.mkdir()
            part = tile_dir / "part_0.las"
            part.write_text("part")

            def first_untwine_then_pdal(parts, final_tile, *_args, **_kwargs):
                final_tile.write_text("copc")
                return (True, "untwine-standard-dimensions")

            def pdal_retry(parts, final_tile, *_args, **_kwargs):
                final_tile.write_text("copc")
                return (True, "pdal-stripped")

            with mock.patch("tile_copc.finalize_tile_to_copc_untwine", side_effect=first_untwine_then_pdal):
                with mock.patch("tile_copc.finalize_tile_to_copc_pdal", side_effect=pdal_retry) as pdal:
                    with mock.patch("tile_copc.first_crs_source", return_value=part):
                        with mock.patch("tile_copc.append_source_geotiff_projection_evlrs", return_value=(True, "ok")):
                            with mock.patch(
                                "tile_copc.copc_preserves_source_crs",
                                side_effect=[(False, "bad crs"), (True, "ok")],
                            ):
                                label, success, message = tile_copc.finalize_tile_to_copc(
                                    ("c0_r0", tiles_dir, log_dir, None)
                                )

            self.assertEqual(label, "c0_r0")
            self.assertTrue(success, message)
            self.assertIn("pdal-stripped", message)
            pdal.assert_called_once()

    def test_run_untwine_streams_standard_records_before_stripping(self):
        import laspy
        import numpy as np
        from laspy.vlrs.vlrlist import VLRList
        for extra in [False, True]:
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir); source = root / "input.las"; output = root / "output.copc.laz"
                header = laspy.LasHeader(point_format=3, version="1.4")
                header.scales = [.001] * 3
                header.offsets = [1234000, 2345000, 10]
                cloud = laspy.LasData(header)
                cloud.X, cloud.Y, cloud.Z = [1, 2, 3], [5, 6, 7], [8, 9, 10]
                cloud.gps_time = [3.5, 7.25, 9.125]
                cloud.red = [200, 300, 400]
                cloud.return_number = [1, 2, 3]
                cloud.header.evlrs = VLRList([laspy.VLR(user_id="test", record_id=42, record_data=b"retain")])
                if extra:
                    cloud.add_extra_dim(laspy.ExtraBytesParams(name="prediction", type=np.uint32))
                    cloud.prediction = [70000, 80000, 90000]
                cloud.write(source)
                staged_paths = []
                def convert(command, **kwargs):
                    staged = Path(command[command.index("-i") + 1]); staged_paths.append(staged)
                    got = laspy.read(staged)
                    self.assertEqual(list(got.point_format.extra_dimension_names), [])
                    for name in ['X', 'Y', 'Z', 'gps_time', 'red', 'return_number']:
                        np.testing.assert_array_equal(got[name], cloud[name])
                    np.testing.assert_array_equal(got.header.scales, cloud.header.scales)
                    np.testing.assert_array_equal(got.header.offsets, cloud.header.offsets)
                    self.assertEqual(got.header.evlrs[0].record_data_bytes(), b"retain")
                    self.assertNotIn("--dims", command)
                    output.write_text("copc")
                    return subprocess.CompletedProcess(command, 0, "", "")
                with mock.patch("tile_copc.shutil.which", return_value="/usr/bin/untwine"), \
                     mock.patch("tile_copc.subprocess.run", side_effect=convert):
                    success, message = tile_copc._run_untwine([source], output, None, strip_extra_dims=True)
                self.assertTrue(success, message)
                self.assertTrue(source.exists())
                if extra:
                    self.assertNotEqual(staged_paths[0], source)
                    self.assertFalse(staged_paths[0].exists())

    def test_convert_laz_to_copc_uses_direct_untwine_for_standard_dimensions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_laz = tmp_path / "input.laz"
            output_copc = tmp_path / "output.copc.laz"
            import laspy
            laspy.LasData(laspy.LasHeader()).write(input_laz)

            def run_untwine(*_, **__):
                output_copc.write_text("copc")
                return subprocess.CompletedProcess(["untwine"], 0, "", "")

            with mock.patch("tile_copc.shutil.which", return_value="/usr/bin/untwine"):
                with mock.patch("tile_copc.srs_assignment_from_file", return_value="EPSG:32632"):
                    with mock.patch("tile_copc.append_source_geotiff_projection_evlrs"):
                        with mock.patch("tile_copc.copc_preserves_source_crs", return_value=(True, "ok")):
                            with mock.patch("tile_copc._output_has_no_extra_dimensions", return_value=True):
                                with mock.patch("tile_copc.subprocess.run", side_effect=run_untwine) as run:
                                    self.assertTrue(tile_copc.convert_laz_to_copc(input_laz, output_copc))

            command = run.call_args.args[0]
            self.assertNotIn("--dims", command)

    def test_convert_laz_to_copc_falls_back_to_pdal_when_untwine_dims_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_laz = tmp_path / "input.laz"
            output_copc = tmp_path / "output.copc.laz"
            input_laz.write_text("input")

            with mock.patch("tile_copc.shutil.which", return_value="/usr/bin/untwine"):
                with mock.patch("tile_copc._run_untwine", return_value=(False, "untwine failed")) as untwine:
                    with mock.patch("tile_copc.convert_laz_to_copc_pdal", return_value=True) as pdal:
                        with mock.patch("tile_copc.append_source_geotiff_projection_evlrs", return_value=(True, "ok")):
                            with mock.patch("tile_copc.copc_preserves_source_crs", return_value=(True, "ok")):
                                self.assertTrue(tile_copc.convert_laz_to_copc(input_laz, output_copc))

            untwine.assert_called_once()
            self.assertTrue(untwine.call_args.kwargs["strip_extra_dims"])
            pdal.assert_called_once_with(input_laz, output_copc, preserve_extra_dims=False)

    def test_convert_laz_to_copc_falls_back_to_pdal_when_untwine_dims_keeps_extras(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_laz = tmp_path / "input.laz"
            output_copc = tmp_path / "output.copc.laz"
            input_laz.write_text("input")

            with mock.patch("tile_copc.shutil.which", return_value="/usr/bin/untwine"):
                with mock.patch("tile_copc._run_untwine", return_value=(True, "untwine")) as untwine:
                    with mock.patch("tile_copc._output_has_no_extra_dimensions", return_value=False):
                        with mock.patch("tile_copc.convert_laz_to_copc_pdal", return_value=True) as pdal:
                            with mock.patch("tile_copc.append_source_geotiff_projection_evlrs", return_value=(True, "ok")):
                                with mock.patch("tile_copc.copc_preserves_source_crs", return_value=(True, "ok")):
                                    self.assertTrue(tile_copc.convert_laz_to_copc(input_laz, output_copc))

            untwine.assert_called_once()
            self.assertTrue(untwine.call_args.kwargs["strip_extra_dims"])
            pdal.assert_called_once_with(input_laz, output_copc, preserve_extra_dims=False)

if __name__ == "__main__":
    unittest.main()
