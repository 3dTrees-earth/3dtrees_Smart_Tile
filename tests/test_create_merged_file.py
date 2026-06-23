import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from main_create_merged_file import (  # noqa: E402
    parse_merged_output_formats,
    parse_merged_resolutions,
    point_cloud_files,
    prepare_copc_inputs,
    prod_merged_output_path,
    prod_merged_pipeline,
    source_key,
)


class CreateMergedFileTests(unittest.TestCase):
    def test_parse_merged_resolutions_defaults_to_res1_and_res2(self):
        self.assertEqual(
            parse_merged_resolutions("res1,res2", 0.01, 0.1),
            [("1cm", 0.01), ("10cm", 0.1)],
        )

    def test_parse_merged_resolutions_accepts_numeric_and_centimeter_tokens(self):
        self.assertEqual(
            parse_merged_resolutions("1cm,0.1,res2", 0.01, 0.1),
            [("1cm", 0.01), ("10cm", 0.1)],
        )

    def test_parse_merged_output_formats_accepts_supported_formats(self):
        self.assertEqual(
            parse_merged_output_formats("laz,copc,ply,copc.laz"),
            ["laz", "copc.laz", "ply"],
        )

    def test_parse_merged_output_formats_defaults_to_copc_laz(self):
        self.assertEqual(parse_merged_output_formats(""), ["copc.laz"])

    def test_parse_merged_output_formats_rejects_unknown_format(self):
        with self.assertRaisesRegex(ValueError, "Unsupported merged output format"):
            parse_merged_output_formats("laz,txt")

    def test_prod_merged_output_path_uses_resolution_label(self):
        self.assertEqual(
            prod_merged_output_path(Path("/tmp/out"), "10cm"),
            Path("/tmp/out/prod_merged_10cm.copc.laz"),
        )

    def test_prod_merged_output_path_uses_selected_format(self):
        self.assertEqual(
            prod_merged_output_path(Path("/tmp/out"), "10cm", "copc.laz"),
            Path("/tmp/out/prod_merged_10cm.copc.laz"),
        )
        self.assertEqual(
            prod_merged_output_path(Path("/tmp/out"), "10cm", "ply"),
            Path("/tmp/out/prod_merged_10cm.ply"),
        )

    def test_point_cloud_files_excludes_copc_derivatives(self):
        paths = [
            Path("/tmp/a.laz"),
            Path("/tmp/b.las"),
            Path("/tmp/c.copc.laz"),
        ]

        with mock.patch.object(Path, "glob") as glob:
            glob.side_effect = [[paths[0], paths[2]], [paths[1]]]
            self.assertEqual(point_cloud_files(Path("/tmp")), [paths[0], paths[1]])

    def test_source_key_treats_raw_and_copc_as_same_source(self):
        self.assertEqual(source_key(Path("/tmp/original.laz")), "original")
        self.assertEqual(source_key(Path("/tmp/original.copc.laz")), "original")

    def test_prod_merged_pipeline_uses_nearest_to_centroid_and_forwards_dimensions(self):
        pipeline = prod_merged_pipeline(
            [Path("/tmp/a.copc.laz"), Path("/tmp/b.copc.laz")],
            Path("/tmp/out/prod_merged_1cm.laz"),
            0.01,
            "laz",
        )["pipeline"]

        self.assertEqual(pipeline[0]["type"], "readers.copc")
        self.assertEqual(pipeline[1]["type"], "readers.copc")
        self.assertEqual(pipeline[2], {"type": "filters.merge"})
        self.assertEqual(
            pipeline[3],
            {"type": "filters.voxelcentroidnearestneighbor", "cell": 0.01},
        )
        self.assertEqual(pipeline[4]["type"], "writers.las")
        self.assertEqual(pipeline[4]["forward"], "all")
        self.assertEqual(pipeline[4]["extra_dims"], "all")

    def test_prod_merged_pipeline_defaults_to_copc_laz(self):
        pipeline = prod_merged_pipeline(
            [Path("/tmp/a.copc.laz")],
            Path("/tmp/out/prod_merged_1cm.copc.laz"),
            0.01,
        )["pipeline"]

        self.assertEqual(pipeline[-1]["type"], "writers.copc")

    def test_prod_merged_pipeline_can_write_copc_laz(self):
        pipeline = prod_merged_pipeline(
            [Path("/tmp/a.copc.laz")],
            Path("/tmp/out/prod_merged_1cm.copc.laz"),
            0.01,
            "copc.laz",
        )["pipeline"]

        self.assertEqual(pipeline[-1]["type"], "writers.copc")
        self.assertEqual(pipeline[-1]["forward"], "all")
        self.assertEqual(pipeline[-1]["extra_dims"], "all")

    def test_prod_merged_pipeline_can_write_ply(self):
        pipeline = prod_merged_pipeline(
            [Path("/tmp/a.copc.laz")],
            Path("/tmp/out/prod_merged_1cm.ply"),
            0.01,
            "ply",
        )["pipeline"]

        self.assertEqual(pipeline[-1]["type"], "writers.ply")
        self.assertEqual(pipeline[-1]["storage_mode"], "little endian")
        self.assertNotIn("forward", pipeline[-1])

    def test_prod_merged_pipeline_can_still_read_plain_laz(self):
        pipeline = prod_merged_pipeline(
            [Path("/tmp/a.laz")],
            Path("/tmp/out/prod_merged_1cm.laz"),
            0.01,
        )["pipeline"]

        self.assertEqual(pipeline[0]["type"], "readers.las")

    def test_prepare_copc_inputs_uses_tiling_converter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_dir = tmp_path / "input"
            output_dir = tmp_path / "out"
            input_dir.mkdir()

            convert = mock.Mock(side_effect=lambda _, output: output.write_text("copc") or True)
            fake_main_tile = mock.Mock(_convert_laz_to_copc=convert)
            with mock.patch.dict(sys.modules, {"main_tile": fake_main_tile}):
                with mock.patch("main_create_merged_file.point_cloud_files") as files:
                    files.return_value = [input_dir / "original.laz"]
                    with mock.patch("main_create_merged_file.copc_files", return_value=[]):
                        outputs = prepare_copc_inputs(input_dir, output_dir)

            expected = output_dir / "original_with_predictions_copc/original.copc.laz"
            self.assertEqual(outputs, [expected])
            convert.assert_called_once_with(input_dir / "original.laz", expected)

    def test_prepare_copc_inputs_prefers_existing_copc_over_matching_raw_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_dir = tmp_path / "input"
            output_dir = tmp_path / "out"
            input_dir.mkdir()

            raw = input_dir / "original.laz"
            copc = input_dir / "original.copc.laz"
            raw.write_text("raw")
            copc.write_text("copc")

            convert = mock.Mock()
            fake_main_tile = mock.Mock(_convert_laz_to_copc=convert)
            with mock.patch.dict(sys.modules, {"main_tile": fake_main_tile}):
                outputs = prepare_copc_inputs(input_dir, output_dir)

            self.assertEqual(outputs, [copc])
            convert.assert_not_called()

    def test_prepare_copc_inputs_accepts_existing_copc_without_converter_import(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_dir = tmp_path / "input"
            output_dir = tmp_path / "out"
            input_dir.mkdir()

            copc = input_dir / "original.copc.laz"
            copc.write_text("copc")

            outputs = prepare_copc_inputs(input_dir, output_dir)

            self.assertEqual(outputs, [copc])

    def test_prepare_copc_inputs_reuses_staged_copc_for_raw_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_dir = tmp_path / "input"
            output_dir = tmp_path / "out"
            input_dir.mkdir()

            raw = input_dir / "original.laz"
            staged = output_dir / "original_with_predictions_copc/original.copc.laz"
            raw.write_text("raw")
            staged.parent.mkdir(parents=True)
            staged.write_text("copc")

            convert = mock.Mock()
            fake_main_tile = mock.Mock(_convert_laz_to_copc=convert)
            with mock.patch.dict(sys.modules, {"main_tile": fake_main_tile}):
                outputs = prepare_copc_inputs(input_dir, output_dir)

            self.assertEqual(outputs, [staged])
            convert.assert_not_called()


if __name__ == "__main__":
    unittest.main()
