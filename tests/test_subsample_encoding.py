import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import laspy
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import subsample_outputs  # noqa: E402
from subsample_encoding import safe_las_writer_encoding_options  # noqa: E402


def _fake_reader(header):
    reader = mock.Mock()
    reader.__enter__ = mock.Mock(return_value=reader)
    reader.__exit__ = mock.Mock(return_value=False)
    reader.header = header
    return reader


def _fake_header(scales, offsets, bounds):
    min_x, max_x, min_y, max_y, min_z, max_z = bounds
    return types.SimpleNamespace(
        scales=np.asarray(scales, dtype=np.float64),
        offsets=np.asarray(offsets, dtype=np.float64),
        x_min=min_x,
        x_max=max_x,
        y_min=min_y,
        y_max=max_y,
        z_min=min_z,
        z_max=max_z,
    )


def _write_epsg5650_style_las(path: Path) -> None:
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.0001, 0.0001, 0.0001])
    header.offsets = np.array([33368483.433600053, 5630000.0, 120.0])
    las = laspy.LasData(header)
    las.x = np.array([33368483.4336, 33368484.1234, 33368496.62])
    las.y = np.array([5630000.0, 5630001.5, 5630002.5])
    las.z = np.array([120.0, 121.0, 122.0])
    las.write(path, do_compress=True, laz_backend=laspy.LazBackend.LazrsParallel)


class SubsampleEncodingTests(unittest.TestCase):
    def test_epsg5650_source_encoding_is_preserved_when_safe(self):
        header = _fake_header(
            scales=[0.0001, 0.0001, 0.0001],
            offsets=[33368483.433600053, 5630000.0, 120.0],
            bounds=[33368483.4336, 33368496.62, 5630000.0, 5630002.5, 120.0, 122.0],
        )

        with mock.patch("subsample_encoding.laspy.open", return_value=_fake_reader(header)):
            opts = safe_las_writer_encoding_options(Path("epsg5650.copc.laz"))

        self.assertEqual(opts["scale_x"], 0.0001)
        self.assertEqual(opts["scale_y"], 0.0001)
        self.assertEqual(opts["scale_z"], 0.0001)
        self.assertEqual(opts["offset_x"], 33368483.433600053)
        self.assertEqual(opts["offset_y"], 5630000.0)
        self.assertEqual(opts["offset_z"], 120.0)

    def test_unsafe_source_offset_falls_back_to_bounds_minimum(self):
        header = _fake_header(
            scales=[0.0001, 0.0001, 0.0001],
            offsets=[0.0, 0.0, 0.0],
            bounds=[33368483.4336, 33368496.62, 5630000.0, 5630002.5, 120.0, 122.0],
        )

        with mock.patch("subsample_encoding.laspy.open", return_value=_fake_reader(header)):
            opts = safe_las_writer_encoding_options(Path("unsafe.laz"))

        self.assertEqual(opts["scale_x"], 0.0001)
        self.assertEqual(opts["offset_x"], 33368483.4336)
        self.assertEqual(opts["offset_y"], 5630000.0)
        self.assertEqual(opts["offset_z"], 120.0)

    def test_bounds_that_cannot_fit_int32_raise_before_pdal(self):
        header = _fake_header(
            scales=[1e-9, 1e-9, 1e-9],
            offsets=[0.0, 0.0, 0.0],
            bounds=[0.0, 10.0, 0.0, 1.0, 0.0, 1.0],
        )

        with mock.patch("subsample_encoding.laspy.open", return_value=_fake_reader(header)):
            with self.assertRaisesRegex(ValueError, "Could not choose safe LAS writer encoding"):
                safe_las_writer_encoding_options(Path("too-wide.laz"))

    def test_pdal_copc_fallback_uses_explicit_safe_encoding(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_file = root / "input.laz"
            output_file = root / "out.copc.laz"
            _write_epsg5650_style_las(input_file)
            captured = {}

            fake_copc_metadata = types.SimpleNamespace(
                validate_single_crs_record=mock.Mock(return_value=(True, "ok")),
            )
            fake_tile_copc = types.SimpleNamespace(
                convert_laz_to_copc=mock.Mock(side_effect=RuntimeError("force fallback")),
            )

            def run(command, **_kwargs):
                with open(command[-1]) as handle:
                    captured.update(json.load(handle))
                output_file.write_bytes(b"copc")
                return subprocess.CompletedProcess(command, 0, "", "")

            with mock.patch.dict(
                sys.modules,
                {"copc_metadata": fake_copc_metadata, "crs_records": fake_copc_metadata, "tile_copc": fake_tile_copc},
            ):
                with mock.patch("subsample_outputs.subprocess.run", side_effect=run):
                    ok = subsample_outputs.convert_laz_output_to_copc(
                        input_file,
                        output_file,
                        source_metadata_file=input_file,
                    )

        self.assertTrue(ok)
        writer = captured["pipeline"][-1]
        self.assertEqual(writer["type"], "writers.copc")
        self.assertEqual(writer["scale_x"], 0.0001)
        self.assertAlmostEqual(writer["offset_x"], 33368483.433600053)


if __name__ == "__main__":
    unittest.main()
