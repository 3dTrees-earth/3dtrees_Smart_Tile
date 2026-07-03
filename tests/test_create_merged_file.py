import os
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from main_create_merged_file import (  # noqa: E402
    _merge_chunk_files_pipeline,
    _merge_prod_chunks,
    _create_prod_merged_chunks,
    _untwine_chunk_files_to_copc,
    _validate_expected_dims,
    _validate_preserved_product_dims,
    create_chunked_prod_merged_file,
    create_chunked_prod_merged_files_for_resolution,
    create_prod_merged_files,
    create_prod_merged_file,
    expensive_prod_merged_warning,
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

    def test_parse_merged_output_formats_accepts_list_like_values(self):
        self.assertEqual(
            parse_merged_output_formats(["copc.laz", "laz", "ply"]),
            ["copc.laz", "laz", "ply"],
        )
        self.assertEqual(
            parse_merged_output_formats("['copc.laz', 'laz']"),
            ["copc.laz", "laz"],
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

    def test_expensive_warning_only_for_full_resolution_copc(self):
        warning = expensive_prod_merged_warning("1cm", 0.01, "copc.laz")
        self.assertIsNotNone(warning)
        self.assertIn("scratch-disk", warning)
        self.assertIsNone(expensive_prod_merged_warning("10cm", 0.1, "copc.laz"))
        self.assertIsNone(expensive_prod_merged_warning("1cm", 0.01, "laz"))
        self.assertIsNone(expensive_prod_merged_warning("1cm", 0.01, "ply"))

    def test_point_cloud_files_excludes_copc_derivatives(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_laz = root / "a.laz"
            raw_las = root / "b.las"
            copc = root / "c.copc.laz"
            raw_laz.write_text("raw")
            raw_las.write_text("raw")
            copc.write_text("copc")

            self.assertEqual(point_cloud_files(root), [raw_laz, raw_las])

    def test_source_key_treats_raw_and_copc_as_same_source(self):
        self.assertEqual(source_key(Path("/tmp/original.laz")), "original")
        self.assertEqual(source_key(Path("/tmp/original.copc.laz")), "original")
        self.assertEqual(source_key(Path("/tmp/SOURCE.COPC.LAZ")), "source")
        self.assertEqual(source_key(Path("/tmp/source.laz")), "source")

    def test_standardization_dim_validation_fails_for_missing_dims(self):
        with mock.patch(
            "main_create_merged_file.point_cloud_dimension_names",
            return_value={"X", "Y", "Z", "intensity"},
        ):
            with self.assertRaisesRegex(RuntimeError, "missing 1 standardized dimension"):
                _validate_expected_dims(
                    [Path("/tmp/source.copc.laz")],
                    {"intensity", "Amplitude"},
                    "staged inputs",
                )

    def test_product_dim_validation_fails_when_writer_drops_dims(self):
        with mock.patch(
            "main_create_merged_file.point_cloud_dimension_names",
            side_effect=[
                {"X", "Y", "Z", "PredInstance", "Amplitude"},
                {"X", "Y", "Z", "Amplitude"},
            ],
        ):
            valid, message = _validate_preserved_product_dims(
                [Path("/tmp/chunk.laz")],
                Path("/tmp/prod_merged.copc.laz"),
            )

        self.assertFalse(valid)
        self.assertIn("PredInstance", message)

    def test_create_prod_merged_files_validates_standardization_dims(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            summary = tmp_path / "collection_summary.json"
            summary.write_text(
                """
                {
                  "collection": {
                    "reference_attribute_names": ["X", "Y", "Z", "Intensity", "Amplitude", "ZeroDim"],
                    "global_attribute_stats": [
                      {"name": "Intensity", "variance": 1.0},
                      {"name": "Amplitude", "variance": 2.0},
                      {"name": "ZeroDim", "variance": 0.0}
                    ]
                  }
                }
                """
            )
            created = tmp_path / "prod_merged_10cm.laz"

            def create_file(*args, **kwargs):
                created.write_text("laz")
                return created

            with mock.patch(
                "main_create_merged_file.prepare_copc_inputs",
                return_value=[tmp_path / "source.copc.laz"],
            ) as prepare:
                with mock.patch(
                    "main_create_merged_file.point_cloud_dimension_names",
                    return_value={"X", "Y", "Z", "intensity", "Amplitude"},
                ) as dims:
                    with mock.patch(
                        "main_create_merged_file.create_prod_merged_file",
                        side_effect=create_file,
                    ):
                        outputs = create_prod_merged_files(
                            tmp_path / "original_with_predictions",
                            tmp_path,
                            "10cm",
                            "laz",
                            0.01,
                            0.1,
                            standardization_json=summary,
                        )

            self.assertEqual(outputs, [created])
            prepare.assert_called_once()
            self.assertGreaterEqual(dims.call_count, 2)

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

    def test_chunked_ply_merge_does_not_receive_las_scale_options(self):
        pipeline = _merge_chunk_files_pipeline(
            [Path("/tmp/chunk.laz")],
            Path("/tmp/out/prod_merged_1cm.ply"),
            "ply",
            {"scale_x": 0.01, "offset_x": 500000.0},
        )["pipeline"]

        self.assertEqual(pipeline[-1]["type"], "writers.ply")
        self.assertNotIn("scale_x", pipeline[-1])
        self.assertNotIn("offset_x", pipeline[-1])

    def test_chunked_copc_merge_prefers_direct_untwine_path(self):
        with mock.patch(
            "main_create_merged_file._untwine_chunk_files_to_copc",
            return_value=(True, "untwine"),
        ) as untwine:
            with mock.patch("main_create_merged_file._run_pdal_pipeline") as pdal:
                _merge_prod_chunks(
                    [Path("/tmp/chunk.laz")],
                    Path("/tmp/out/prod_merged_1cm.copc.laz"),
                    "copc.laz",
                    Path("/tmp/work"),
                    Path("/tmp/source.copc.laz"),
                    {"scale_x": 0.01},
                )

        untwine.assert_called_once()
        pdal.assert_not_called()

    def test_chunked_copc_pdal_fallback_validates_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            output = tmp_path / "prod_merged_1cm.copc.laz"

            def run_pipeline(*_, **__):
                output.write_text("copc")
                return mock.Mock(returncode=0, stdout="", stderr="")

            with mock.patch(
                "main_create_merged_file._untwine_chunk_files_to_copc",
                return_value=(False, "untwine unavailable"),
            ):
                with mock.patch("main_create_merged_file._run_pdal_pipeline", side_effect=run_pipeline):
                    with mock.patch(
                        "main_create_merged_file._validate_preserved_product_dims",
                        return_value=(True, "ok"),
                    ):
                        with mock.patch(
                            "main_create_merged_file._preserve_and_validate_las_metadata",
                            return_value=(True, "ok"),
                        ) as validate:
                            _merge_prod_chunks(
                                [tmp_path / "chunk.laz"],
                                output,
                                "copc.laz",
                                tmp_path / "work",
                                tmp_path / "source.copc.laz",
                                {"scale_x": 0.01},
                            )

        validate.assert_called_once_with(tmp_path / "source.copc.laz", output)

    def test_chunked_multiple_formats_reuse_one_chunk_set(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            source = tmp_path / "source.copc.laz"
            outputs = [
                (tmp_path / "prod_merged_10cm.laz", "laz"),
                (tmp_path / "prod_merged_10cm.copc.laz", "copc.laz"),
                (tmp_path / "prod_merged_10cm.ply", "ply"),
            ]
            merge_calls = []

            def run_pipeline(pipeline, *_):
                Path(pipeline["pipeline"][-1]["filename"]).write_text("chunk")
                return mock.Mock(returncode=0, stdout="", stderr="")

            def merge_chunks(chunk_files, output_file, output_format, *_args, **_kwargs):
                merge_calls.append((list(chunk_files), output_file, output_format))
                output_file.write_text(output_format)

            with mock.patch(
                "main_create_merged_file._prod_merged_chunk_bounds",
                return_value=["bounds-a", "bounds-b"],
            ):
                with mock.patch("main_create_merged_file._scale_offset_options", return_value={}):
                    with mock.patch("main_create_merged_file._run_pdal_pipeline", side_effect=run_pipeline) as pdal:
                        with mock.patch(
                            "main_create_merged_file._merge_prod_chunks",
                            side_effect=merge_chunks,
                        ):
                            created = create_chunked_prod_merged_files_for_resolution(
                                [source],
                                outputs,
                                0.1,
                                2,
                            )

        self.assertEqual(created, [path for path, _ in outputs])
        self.assertEqual(pdal.call_count, 2)
        self.assertEqual(len(merge_calls), 3)
        first_chunk_set = merge_calls[0][0]
        self.assertTrue(first_chunk_set)
        for chunk_set, _, _ in merge_calls:
            self.assertEqual(chunk_set, first_chunk_set)

    def test_chunk_creation_uses_capped_parallel_workers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            source = tmp_path / "source.copc.laz"
            source.write_text("copc")
            work_dir = tmp_path / "chunks"
            work_dir.mkdir()
            active = 0
            max_active = 0
            lock = threading.Lock()

            def run_pipeline(pipeline, *_):
                nonlocal active, max_active
                with lock:
                    active += 1
                    max_active = max(max_active, active)
                try:
                    time.sleep(0.05)
                    Path(pipeline["pipeline"][-1]["filename"]).write_text("chunk")
                    return mock.Mock(returncode=0, stdout="", stderr="")
                finally:
                    with lock:
                        active -= 1

            with mock.patch("main_create_merged_file._run_pdal_pipeline", side_effect=run_pipeline):
                chunks = _create_prod_merged_chunks(
                    [source],
                    work_dir,
                    "prod_merged",
                    ["bounds-a", "bounds-b", "bounds-c", "bounds-d"],
                    0.1,
                    {},
                    chunk_workers=2,
                )

        self.assertEqual(len(chunks), 4)
        self.assertEqual(max_active, 2)

    def test_single_copc_pipeline_validates_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            source = tmp_path / "source.copc.laz"
            output = tmp_path / "prod_merged_10cm.copc.laz"
            source.write_text("copc")

            def run_pipeline(*_, **__):
                output.write_text("copc")
                return mock.Mock(returncode=0, stdout="", stderr="")

            with mock.patch("main_create_merged_file._run_pdal_pipeline", side_effect=run_pipeline):
                with mock.patch(
                    "main_create_merged_file._validate_preserved_product_dims",
                    return_value=(True, "ok"),
                ) as validate_dims:
                    with mock.patch(
                        "main_create_merged_file._preserve_and_validate_las_metadata",
                        return_value=(True, "ok"),
                    ) as validate:
                        created = create_prod_merged_file(
                            [source],
                            output,
                            0.1,
                            "copc.laz",
                            num_spatial_chunks=1,
                        )

        self.assertEqual(created, output)
        validate_dims.assert_called_once_with([source], output)
        validate.assert_called_once_with(source, output)

    def test_single_laz_pipeline_validates_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            source = tmp_path / "source.copc.laz"
            output = tmp_path / "prod_merged_10cm.laz"
            source.write_text("copc")

            def run_pipeline(*_, **__):
                output.write_text("laz")
                return mock.Mock(returncode=0, stdout="", stderr="")

            with mock.patch("main_create_merged_file._run_pdal_pipeline", side_effect=run_pipeline):
                with mock.patch(
                    "main_create_merged_file._validate_preserved_product_dims",
                    return_value=(True, "ok"),
                ) as validate_dims:
                    with mock.patch(
                        "main_create_merged_file._preserve_and_validate_las_metadata",
                        return_value=(True, "ok"),
                    ) as validate:
                        created = create_prod_merged_file(
                            [source],
                            output,
                            0.1,
                            "laz",
                            num_spatial_chunks=1,
                        )

        self.assertEqual(created, output)
        validate_dims.assert_called_once_with([source], output)
        validate.assert_called_once_with(source, output)

    def test_single_product_removes_stale_output_before_rewrite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            source = tmp_path / "source.copc.laz"
            output = tmp_path / "prod_merged_10cm.laz"
            source.write_text("copc")
            output.write_text("stale")

            def run_pipeline(*_, **__):
                self.assertFalse(output.exists())
                output.write_text("fresh")
                return mock.Mock(returncode=0, stdout="", stderr="")

            with mock.patch("main_create_merged_file._run_pdal_pipeline", side_effect=run_pipeline):
                with mock.patch(
                    "main_create_merged_file._validate_preserved_product_dims",
                    return_value=(True, "ok"),
                ):
                    with mock.patch(
                        "main_create_merged_file._preserve_and_validate_las_metadata",
                        return_value=(True, "ok"),
                    ):
                        created = create_prod_merged_file([source], output, 0.1, "laz")

            self.assertEqual(created, output)
            self.assertEqual(output.read_text(), "fresh")

    def test_chunked_product_failure_cleans_scratch_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            source = tmp_path / "source.copc.laz"
            output = tmp_path / "prod_merged_10cm.laz"
            source.write_text("copc")

            def fail_chunk(pipeline, *_):
                Path(pipeline["pipeline"][-1]["filename"]).write_text("partial")
                return mock.Mock(returncode=1, stdout="", stderr="chunk failed")

            with mock.patch("main_create_merged_file._prod_merged_chunk_bounds", return_value=["bounds-a"]):
                with mock.patch("main_create_merged_file._scale_offset_options", return_value={}):
                    with mock.patch("main_create_merged_file._run_pdal_pipeline", side_effect=fail_chunk):
                        with self.assertRaisesRegex(RuntimeError, "chunk 1 failed"):
                            create_chunked_prod_merged_file([source], output, 0.1, "laz", 2)

            self.assertFalse((tmp_path / "_prod_merged_10cm_chunks").exists())

    def test_chunked_product_failure_can_keep_scratch_for_debug(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            source = tmp_path / "source.copc.laz"
            output = tmp_path / "prod_merged_10cm.laz"
            source.write_text("copc")

            def fail_chunk(pipeline, *_):
                Path(pipeline["pipeline"][-1]["filename"]).write_text("partial")
                return mock.Mock(returncode=1, stdout="", stderr="chunk failed")

            with mock.patch.dict(os.environ, {"SMARTTILE_KEEP_FAILED_CHUNKS": "1"}):
                with mock.patch("main_create_merged_file._prod_merged_chunk_bounds", return_value=["bounds-a"]):
                    with mock.patch("main_create_merged_file._scale_offset_options", return_value={}):
                        with mock.patch("main_create_merged_file._run_pdal_pipeline", side_effect=fail_chunk):
                            with self.assertRaisesRegex(RuntimeError, "chunk 1 failed"):
                                create_chunked_prod_merged_file([source], output, 0.1, "laz", 2)

            self.assertTrue((tmp_path / "_prod_merged_10cm_chunks").exists())

    def test_direct_untwine_uses_configured_temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            output = tmp_path / "out.copc.laz"
            temp_dir = tmp_path / "untwine_tmp"
            output.write_text("copc")
            fake_copc_metadata = mock.Mock(
                append_source_geotiff_projection_evlrs=mock.Mock(return_value=(True, "ok")),
                copc_preserves_source_crs=mock.Mock(return_value=(True, "ok")),
                srs_assignment_from_file=mock.Mock(return_value="EPSG:32632"),
            )
            with mock.patch("main_create_merged_file.shutil.which", return_value="/usr/bin/untwine"):
                with mock.patch.dict(sys.modules, {"copc_metadata": fake_copc_metadata}):
                    with mock.patch("main_create_merged_file.subprocess.run") as run:
                        def run_untwine(*_, **__):
                            output.write_text("copc")
                            return mock.Mock(returncode=0, stdout="", stderr="")

                        run.side_effect = run_untwine
                        with mock.patch(
                            "main_create_merged_file._point_cloud_point_count",
                            side_effect=[1, 1],
                        ):
                            with mock.patch(
                                "main_create_merged_file._validate_preserved_product_dims",
                                return_value=(True, "ok"),
                            ):
                                success, _ = _untwine_chunk_files_to_copc(
                                    [tmp_path / "chunk.laz"],
                                    output,
                                    tmp_path / "source.copc.laz",
                                    temp_dir=temp_dir,
                                )

        self.assertTrue(success)
        command = run.call_args.args[0]
        self.assertIn("--temp_dir", command)
        self.assertEqual(command[command.index("--temp_dir") + 1], str(temp_dir))

    def test_direct_untwine_rejects_point_count_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            output = tmp_path / "out.copc.laz"
            fake_copc_metadata = mock.Mock(
                append_source_geotiff_projection_evlrs=mock.Mock(return_value=(True, "ok")),
                copc_preserves_source_crs=mock.Mock(return_value=(True, "ok")),
                srs_assignment_from_file=mock.Mock(return_value="EPSG:32632"),
            )

            def run_untwine(*_, **__):
                output.write_text("copc")
                return mock.Mock(returncode=0, stdout="", stderr="")

            with mock.patch("main_create_merged_file.shutil.which", return_value="/usr/bin/untwine"):
                with mock.patch.dict(sys.modules, {"copc_metadata": fake_copc_metadata}):
                    with mock.patch("main_create_merged_file.subprocess.run", side_effect=run_untwine):
                        with mock.patch(
                            "main_create_merged_file._point_cloud_point_count",
                            side_effect=[3, 2],
                        ):
                            success, message = _untwine_chunk_files_to_copc(
                                [tmp_path / "chunk.laz"],
                                output,
                                tmp_path / "source.copc.laz",
                            )

        self.assertFalse(success)
        self.assertIn("point-count mismatch", message)
        self.assertFalse(output.exists())

    def test_direct_untwine_rejects_dimension_loss(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            output = tmp_path / "out.copc.laz"
            fake_copc_metadata = mock.Mock(
                append_source_geotiff_projection_evlrs=mock.Mock(return_value=(True, "ok")),
                copc_preserves_source_crs=mock.Mock(return_value=(True, "ok")),
                srs_assignment_from_file=mock.Mock(return_value="EPSG:32632"),
            )

            def run_untwine(*_, **__):
                output.write_text("copc")
                return mock.Mock(returncode=0, stdout="", stderr="")

            with mock.patch("main_create_merged_file.shutil.which", return_value="/usr/bin/untwine"):
                with mock.patch.dict(sys.modules, {"copc_metadata": fake_copc_metadata}):
                    with mock.patch("main_create_merged_file.subprocess.run", side_effect=run_untwine):
                        with mock.patch(
                            "main_create_merged_file._point_cloud_point_count",
                            side_effect=[3, 3],
                        ):
                            with mock.patch(
                                "main_create_merged_file._validate_preserved_product_dims",
                                return_value=(False, "prod-merged output dropped dimensions"),
                            ):
                                success, message = _untwine_chunk_files_to_copc(
                                    [tmp_path / "chunk.laz"],
                                    output,
                                    tmp_path / "source.copc.laz",
                                )

        self.assertFalse(success)
        self.assertIn("dropped dimensions", message)
        self.assertFalse(output.exists())

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

            convert = mock.Mock(side_effect=lambda _, output, **__: output.write_text("copc") or True)
            fake_main_tile = mock.Mock(_convert_laz_to_copc=convert)
            with mock.patch.dict(sys.modules, {"main_tile": fake_main_tile}):
                with mock.patch("main_create_merged_file.point_cloud_files") as files:
                    files.return_value = [input_dir / "original.laz"]
                    with mock.patch("main_create_merged_file.copc_files", return_value=[]):
                        with mock.patch("main_create_merged_file.write_copc_stage_manifest_entry"):
                            outputs = prepare_copc_inputs(input_dir, output_dir)

            expected = output_dir / "original_with_predictions_copc/original.copc.laz"
            self.assertEqual(outputs, [expected])
            convert.assert_called_once_with(
                input_dir / "original.laz",
                expected,
                preserve_extra_dims=True,
            )

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
            with mock.patch("main_create_merged_file._is_reusable_copc", return_value=True):
                with mock.patch("main_create_merged_file.staged_copc_matches_source", return_value=True):
                    with mock.patch.dict(sys.modules, {"main_tile": fake_main_tile}):
                        outputs = prepare_copc_inputs(input_dir, output_dir)

            self.assertEqual(outputs, [copc])
            convert.assert_not_called()

    def test_prepare_copc_inputs_rebuilds_existing_copc_without_fresh_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_dir = tmp_path / "input"
            output_dir = tmp_path / "out"
            input_dir.mkdir()

            raw = input_dir / "original.laz"
            stale_copc = input_dir / "original.copc.laz"
            raw.write_text("raw")
            stale_copc.write_text("stale copc")

            def convert_input(_, output, **__):
                output.write_text("converted")
                return True

            convert = mock.Mock(side_effect=convert_input)
            fake_main_tile = mock.Mock(_convert_laz_to_copc=convert)
            with mock.patch("main_create_merged_file._is_reusable_copc", return_value=True):
                with mock.patch("main_create_merged_file.staged_copc_matches_source", return_value=False):
                    with mock.patch("main_create_merged_file.write_copc_stage_manifest_entry") as manifest:
                        with mock.patch.dict(sys.modules, {"main_tile": fake_main_tile}):
                            outputs = prepare_copc_inputs(input_dir, output_dir)

            expected = output_dir / "original_with_predictions_copc/original.copc.laz"
            self.assertEqual(outputs, [expected])
            convert.assert_called_once_with(raw, expected, preserve_extra_dims=True)
            manifest.assert_called_once_with(raw, expected)

    def test_prepare_copc_inputs_accepts_existing_copc_without_converter_import(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_dir = tmp_path / "input"
            output_dir = tmp_path / "out"
            input_dir.mkdir()

            copc = input_dir / "original.copc.laz"
            copc.write_text("copc")

            with mock.patch("main_create_merged_file._is_reusable_copc", return_value=True):
                outputs = prepare_copc_inputs(input_dir, output_dir)

            self.assertEqual(outputs, [copc])

    def test_prepare_copc_inputs_keeps_raw_and_copc_only_sources(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_dir = tmp_path / "input"
            output_dir = tmp_path / "out"
            input_dir.mkdir()

            raw = input_dir / "alpha.laz"
            copc = input_dir / "bravo.copc.laz"
            raw.write_text("raw")
            copc.write_text("copc")

            def convert_input(_, output, **__):
                output.write_text("converted")
                return True

            convert = mock.Mock(side_effect=convert_input)
            fake_main_tile = mock.Mock(_convert_laz_to_copc=convert)
            with mock.patch("main_create_merged_file._is_reusable_copc", return_value=True):
                with mock.patch("main_create_merged_file.write_copc_stage_manifest_entry"):
                    with mock.patch.dict(sys.modules, {"main_tile": fake_main_tile}):
                        outputs = prepare_copc_inputs(input_dir, output_dir)

            converted = output_dir / "original_with_predictions_copc/alpha.copc.laz"
            self.assertEqual(outputs, [converted, copc])
            convert.assert_called_once_with(raw, converted, preserve_extra_dims=True)

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
            with mock.patch("main_create_merged_file._is_reusable_copc", return_value=True):
                with mock.patch("main_create_merged_file.staged_copc_matches_source", return_value=True):
                    with mock.patch.dict(sys.modules, {"main_tile": fake_main_tile}):
                        outputs = prepare_copc_inputs(input_dir, output_dir)

            self.assertEqual(outputs, [staged])
            convert.assert_not_called()

    def test_prepare_copc_inputs_reuses_explicit_staged_copc_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_dir = tmp_path / "input"
            output_dir = tmp_path / "out"
            staged_dir = tmp_path / "staged"
            input_dir.mkdir()
            staged_dir.mkdir()

            raw = input_dir / "original.laz"
            staged = staged_dir / "original.copc.laz"
            raw.write_text("raw")
            staged.write_text("copc")

            convert = mock.Mock()
            fake_main_tile = mock.Mock(_convert_laz_to_copc=convert)
            with mock.patch("main_create_merged_file._is_reusable_copc", return_value=True):
                with mock.patch("main_create_merged_file.staged_copc_matches_source", return_value=True):
                    with mock.patch.dict(sys.modules, {"main_tile": fake_main_tile}):
                        outputs = prepare_copc_inputs(input_dir, output_dir, staged_copc_dir=staged_dir)

            self.assertEqual(outputs, [staged])
            convert.assert_not_called()

    def test_prepare_copc_inputs_ignores_unmatched_staged_copc(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_dir = tmp_path / "input"
            output_dir = tmp_path / "out"
            staged_dir = tmp_path / "staged"
            input_dir.mkdir()
            staged_dir.mkdir()

            raw = input_dir / "original.laz"
            staged = staged_dir / "original.copc.laz"
            unrelated = staged_dir / "unrelated.copc.laz"
            raw.write_text("raw")
            staged.write_text("copc")
            unrelated.write_text("stale copc")

            convert = mock.Mock()
            fake_main_tile = mock.Mock(_convert_laz_to_copc=convert)
            with mock.patch("main_create_merged_file._is_reusable_copc", return_value=True):
                with mock.patch("main_create_merged_file.staged_copc_matches_source", return_value=True):
                    with mock.patch.dict(sys.modules, {"main_tile": fake_main_tile}):
                        outputs = prepare_copc_inputs(input_dir, output_dir, staged_copc_dir=staged_dir)

            self.assertEqual(outputs, [staged])
            self.assertNotIn(unrelated, outputs)
            convert.assert_not_called()

    def test_prepare_copc_inputs_ignores_unreadable_staged_copc(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_dir = tmp_path / "input"
            output_dir = tmp_path / "out"
            staged_dir = tmp_path / "staged"
            input_dir.mkdir()
            staged_dir.mkdir()

            raw = input_dir / "original.laz"
            staged = staged_dir / "original.copc.laz"
            raw.write_text("raw")
            staged.write_text("partial")

            def convert_input(_, output, **__):
                output.write_text("converted")
                return True

            convert = mock.Mock(side_effect=convert_input)
            fake_main_tile = mock.Mock(_convert_laz_to_copc=convert)
            with mock.patch("main_create_merged_file._is_reusable_copc", return_value=False):
                with mock.patch("main_create_merged_file.write_copc_stage_manifest_entry"):
                    with mock.patch.dict(sys.modules, {"main_tile": fake_main_tile}):
                        outputs = prepare_copc_inputs(input_dir, output_dir, staged_copc_dir=staged_dir)

            expected = output_dir / "original_with_predictions_copc/original.copc.laz"
            self.assertEqual(outputs, [expected])
            convert.assert_called_once_with(raw, expected, preserve_extra_dims=True)


if __name__ == "__main__":
    unittest.main()
