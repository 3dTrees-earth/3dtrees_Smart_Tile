import sys
import tempfile
import unittest
from unittest import mock
from datetime import date
from pathlib import Path

import laspy
import numpy as np
from laspy.vlrs.vlr import VLR


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prediction_collection_remap import (  # noqa: E402
    _copc_spatial_windows,
    _parallel_raw_chunk_plan,
    load_collection_subset_for_bounds,
    prediction_collection_files,
    remap_prediction_collections_to_original_files,
    stream_add_collections_to_file,
)
from parameters import Parameters  # noqa: E402
from run import _prepare_merge_prediction_collection_source  # noqa: E402


def _base_header() -> laspy.LasHeader:
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.array([500.0, 600.0, 50.0])
    header.system_identifier = "synthetic-source"
    header.generating_software = "unit-test"
    header.creation_date = date(2026, 6, 24)
    header.vlrs.append(
        VLR(
            user_id="LASF_Projection",
            record_id=34735,
            description="synthetic projection",
            record_data=b"projection-payload",
        )
    )
    return header


def _write_las(path: Path, extra_dims: dict[str, np.ndarray] | None = None) -> None:
    extra_dims = extra_dims or {}
    header = _base_header()
    las = laspy.LasData(header)
    n_points = 4
    las.x = np.array([500.0, 500.1, 500.2, 500.3])
    las.y = np.array([600.0, 600.0, 600.1, 600.1])
    las.z = np.array([50.0, 50.1, 50.2, 50.3])
    las.intensity = np.array([10, 11, 12, 13], dtype=np.uint16)
    for name, values in extra_dims.items():
        values = np.asarray(values)
        assert len(values) == n_points
        las.add_extra_dim(laspy.ExtraBytesParams(name=name, type=values.dtype))
        setattr(las, name, values)
    las.write(path)


def _write_shifted_prediction_las(path: Path) -> None:
    header = _base_header()
    las = laspy.LasData(header)
    las.x = np.array([500.01, 500.11, 500.21, 500.31])
    las.y = np.array([600.01, 600.01, 600.11, 600.11])
    las.z = np.array([50.0, 50.1, 50.2, 50.3])
    las.add_extra_dim(laspy.ExtraBytesParams(name="PredInstance_Shifted", type=np.uint16))
    las.PredInstance_Shifted = np.array([1, 1, 0, 2], dtype=np.uint16)
    las.write(path)


class MultiCollectionRemapTests(unittest.TestCase):
    def test_prediction_collection_files_prefer_copc_twins(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            collection = root / "collection"
            collection.mkdir()
            (collection / "source.laz").write_text("placeholder")
            (collection / "source.copc.laz").write_text("placeholder")
            (collection / "other.las").write_text("placeholder")

            files = [path.name for path in prediction_collection_files(collection)]

        self.assertEqual(files, ["other.las", "source.copc.laz"])

    def test_copc_spatial_windows_cover_header_extent(self):
        header = mock.Mock(x_min=0.0, x_max=10.0)
        self.assertEqual(
            _copc_spatial_windows(header, 4),
            [
                (0.0, 2.5, False),
                (2.5, 5.0, False),
                (5.0, 7.5, False),
                (7.5, 10.0, True),
            ],
        )

    def test_parallel_raw_chunk_plan_keeps_total_budget_bounded(self):
        self.assertEqual(_parallel_raw_chunk_plan(20_000_000, 20), (5_000_000, 4))
        self.assertEqual(_parallel_raw_chunk_plan(20_000_000, 1), (20_000_000, 1))
        self.assertEqual(_parallel_raw_chunk_plan(10, 3), (10, 1))

    def test_remaps_distinct_model_named_dims_and_preserves_source_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            original_dir = root / "originals"
            sat_dir = root / "sat"
            foma_dir = root / "foma"
            output_dir = root / "original_with_predictions"
            original_dir.mkdir()
            sat_dir.mkdir()
            foma_dir.mkdir()

            _write_las(original_dir / "source.las", {"OriginalExtra": np.array([1, 2, 3, 4], dtype=np.uint16)})
            _write_las(
                sat_dir / "source_sat.las",
                {
                    "PredInstance_SAT": np.array([1, 1, 0, 2], dtype=np.uint16),
                    "PredSemantic_SAT": np.array([1, 1, 0, 1], dtype=np.uint8),
                    "species_id_sat": np.array([10, 10, 0, 11], dtype=np.uint16),
                    "species_prob_sat": np.array([0.9, 0.8, 0.0, 0.7], dtype=np.float32),
                },
            )
            _write_las(
                foma_dir / "source_foma.las",
                {
                    "PredInstance_ForestMamba": np.array([5, 5, 0, 6], dtype=np.uint32),
                    "PredSemantic_ForestMamba": np.array([1, 1, 0, 1], dtype=np.uint8),
                    "ForestMambaConfidence": np.array([0.5, 0.6, 0.0, 0.7], dtype=np.float32),
                    "species_id_foma": np.array([20, 20, 0, 21], dtype=np.uint16),
                    "species_prob_foma": np.array([0.4, 0.5, 0.0, 0.6], dtype=np.float32),
                },
            )

            remap_prediction_collections_to_original_files(
                [sat_dir, foma_dir],
                original_dir,
                output_dir,
                tolerance=0.001,
                num_threads=1,
            )

            out = laspy.read(output_dir / "source.las")
            self.assertEqual(len(out), 4)
            self.assertFalse(any(output_dir.glob("*.collection_*.laz")))
            out_dims = set(out.point_format.dimension_names) | {dim.name for dim in out.point_format.extra_dimensions}
            for name in (
                "OriginalExtra",
                "PredInstance_SAT",
                "PredSemantic_SAT",
                "species_id_sat",
                "species_prob_sat",
                "PredInstance_ForestMamba",
                "PredSemantic_ForestMamba",
                "ForestMambaConfidence",
                "species_id_foma",
                "species_prob_foma",
            ):
                self.assertIn(name, out_dims)

            np.testing.assert_array_equal(out.PredInstance_SAT, np.array([1, 1, 0, 2], dtype=np.uint16))
            np.testing.assert_array_equal(out.PredInstance_ForestMamba, np.array([5, 5, 0, 6], dtype=np.uint32))
            np.testing.assert_array_equal(out.OriginalExtra, np.array([1, 2, 3, 4], dtype=np.uint16))
            self.assertEqual(out.header.system_identifier, "synthetic-source")
            self.assertEqual(out.header.generating_software, "unit-test")
            self.assertEqual(out.header.creation_date, date(2026, 6, 24))
            self.assertTrue(any(v.user_id == "LASF_Projection" and v.record_id == 34735 for v in out.header.vlrs))

    def test_prepares_merge_source_from_multiple_segmented_collections(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            primary_dir = root / "primary"
            secondary_dir = root / "secondary"
            output_dir = root / "merge_work"
            primary_dir.mkdir()
            secondary_dir.mkdir()

            _write_las(
                primary_dir / "source.las",
                {
                    "PredInstance_FM": np.array([1, 1, 0, 2], dtype=np.uint16),
                    "PredSemantic_FM": np.array([1, 1, 0, 1], dtype=np.uint8),
                },
            )
            _write_las(
                secondary_dir / "source.las",
                {
                    "PredInstance_FM2": np.array([5, 5, 0, 6], dtype=np.uint16),
                    "PredSemantic_FM2": np.array([1, 1, 0, 1], dtype=np.uint8),
                },
            )

            combined_dir = _prepare_merge_prediction_collection_source(
                prediction_collections=[primary_dir, secondary_dir],
                reference_dir=None,
                output_folder=output_dir,
                params=Parameters(remap_tolerance=0.001, workers=1, _cli_parse_args=False),
                retile_buffer=0.0,
                workers=1,
            )

            out = laspy.read(combined_dir / "source.las")
            self.assertEqual(len(out), 4)
            dims = set(out.point_format.dimension_names) | {dim.name for dim in out.point_format.extra_dimensions}
            for name in (
                "PredInstance_FM",
                "PredSemantic_FM",
                "PredInstance_FM2",
                "PredSemantic_FM2",
            ):
                self.assertIn(name, dims)
            np.testing.assert_array_equal(out.PredInstance_FM, np.array([1, 1, 0, 2], dtype=np.uint16))
            np.testing.assert_array_equal(out.PredInstance_FM2, np.array([5, 5, 0, 6], dtype=np.uint16))

    def test_duplicate_prediction_dim_names_fail_before_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            original_dir = root / "originals"
            a_dir = root / "a"
            b_dir = root / "b"
            output_dir = root / "out"
            original_dir.mkdir()
            a_dir.mkdir()
            b_dir.mkdir()

            _write_las(original_dir / "source.las")
            _write_las(a_dir / "a.las", {"PredInstance_SAT": np.array([1, 1, 0, 2], dtype=np.uint16)})
            _write_las(b_dir / "b.las", {"PredInstance_SAT": np.array([5, 5, 0, 6], dtype=np.uint16)})

            with self.assertRaisesRegex(ValueError, "Duplicate prediction dimension name"):
                remap_prediction_collections_to_original_files(
                    [a_dir, b_dir],
                    original_dir,
                    output_dir,
                    tolerance=0.001,
                    num_threads=1,
                )
            self.assertFalse(output_dir.exists())

    def test_existing_original_prediction_dim_is_preserved_and_new_dim_is_suffixed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            original_dir = root / "originals"
            sat_dir = root / "sat"
            output_dir = root / "out"
            original_dir.mkdir()
            sat_dir.mkdir()

            _write_las(
                original_dir / "source.las",
                {"PredInstance_SAT": np.array([9, 9, 9, 9], dtype=np.uint16)},
            )
            _write_las(
                sat_dir / "source_sat.las",
                {"PredInstance_SAT": np.array([1, 1, 0, 2], dtype=np.uint16)},
            )

            remap_prediction_collections_to_original_files(
                [sat_dir],
                original_dir,
                output_dir,
                tolerance=0.001,
                num_threads=1,
            )

            out = laspy.read(output_dir / "source.las")
            dims = set(out.point_format.dimension_names) | {dim.name for dim in out.point_format.extra_dimensions}
            self.assertIn("PredInstance_SAT", dims)
            self.assertIn("PredInstance_SAT_1", dims)
            np.testing.assert_array_equal(out.PredInstance_SAT, np.array([9, 9, 9, 9], dtype=np.uint16))
            np.testing.assert_array_equal(out.PredInstance_SAT_1, np.array([1, 1, 0, 2], dtype=np.uint16))

    def test_rerun_reprocesses_stale_existing_output_missing_selected_dims(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            original_dir = root / "originals"
            sat_dir = root / "sat"
            output_dir = root / "out"
            original_dir.mkdir()
            sat_dir.mkdir()
            output_dir.mkdir()

            _write_las(original_dir / "source.las")
            _write_las(output_dir / "source.las")
            _write_las(
                sat_dir / "source_sat.las",
                {"PredInstance_SAT": np.array([1, 1, 0, 2], dtype=np.uint16)},
            )

            remap_prediction_collections_to_original_files(
                [sat_dir],
                original_dir,
                output_dir,
                tolerance=0.001,
                num_threads=1,
            )

            out = laspy.read(output_dir / "source.las")
            dims = set(out.point_format.dimension_names) | {dim.name for dim in out.point_format.extra_dimensions}
            self.assertIn("PredInstance_SAT", dims)
            np.testing.assert_array_equal(out.PredInstance_SAT, np.array([1, 1, 0, 2], dtype=np.uint16))

    def test_incomplete_spatial_match_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            original_dir = root / "originals"
            shifted_dir = root / "shifted"
            output_dir = root / "out"
            original_dir.mkdir()
            shifted_dir.mkdir()

            _write_las(original_dir / "source.las")
            _write_shifted_prediction_las(shifted_dir / "shifted.las")

            with self.assertRaisesRegex(RuntimeError, "matched 0/4 points"):
                remap_prediction_collections_to_original_files(
                    [shifted_dir],
                    original_dir,
                    output_dir,
                    tolerance=0.001,
                    num_threads=1,
                )

    def test_stream_remap_honors_caller_spatial_buffer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_file = root / "source.las"
            output_file = root / "out.las"
            _write_las(input_file)

            collection_meta = [{
                "path": root / "collection",
                "dims": ["PredInstance_Buffered"],
                "extra_params": {
                    "PredInstance_Buffered": laspy.ExtraBytesParams(
                        name="PredInstance_Buffered",
                        type=np.uint16,
                    )
                },
            }]
            source_points = np.column_stack([
                np.array([500.0, 500.1, 500.2, 500.3]),
                np.array([600.0, 600.0, 600.1, 600.1]),
                np.array([50.0, 50.1, 50.2, 50.3]),
            ])
            captured_buffers = []

            def load_subset(_, __, spatial_buffer):
                captured_buffers.append(spatial_buffer)
                return (
                    source_points,
                    {"PredInstance_Buffered": np.array([1, 1, 0, 2], dtype=np.uint16)},
                )

            with mock.patch(
                "prediction_collection_remap.load_collection_subset_for_bounds",
                side_effect=load_subset,
            ):
                stream_add_collections_to_file(
                    input_file,
                    output_file,
                    collection_meta,
                    spatial_buffer=3.0,
                    tolerance=0.001,
                    chunk_size=10,
                    kdtree_workers=1,
                )

            self.assertEqual(captured_buffers, [3.0])

    def test_stream_remap_can_enrich_raw_chunks_in_parallel_preserving_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_file = root / "source.las"
            pred_dir = root / "pred"
            output_file = root / "out.las"
            pred_dir.mkdir()
            _write_las(input_file)
            _write_las(
                pred_dir / "source_pred.las",
                {"PredInstance_Model": np.array([10, 11, 12, 13], dtype=np.uint16)},
            )

            collection_meta = [{
                "path": pred_dir,
                "files": [{
                    "path": pred_dir / "source_pred.las",
                    "bounds": (499.0, 501.0, 599.0, 601.0),
                    "z_bounds": (49.0, 51.0),
                    "extra_names": {"PredInstance_Model"},
                }],
                "dims": ["PredInstance_Model"],
                "extra_params": {
                    "PredInstance_Model": laspy.ExtraBytesParams(
                        name="PredInstance_Model",
                        type=np.uint16,
                    )
                },
            }]

            n_points, matched = stream_add_collections_to_file(
                input_file,
                output_file,
                collection_meta,
                spatial_buffer=1.0,
                tolerance=0.001,
                chunk_size=1,
                kdtree_workers=1,
                chunk_parallel_workers=2,
            )

            out = laspy.read(output_file)
            self.assertEqual(n_points, 4)
            self.assertEqual(matched, 4)
            np.testing.assert_array_equal(out.x, np.array([500.0, 500.1, 500.2, 500.3]))
            np.testing.assert_array_equal(
                out.PredInstance_Model,
                np.array([10, 11, 12, 13], dtype=np.uint16),
            )

    def test_single_raw_original_uses_chunk_workers_from_total_budget(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            original_dir = root / "originals"
            pred_dir = root / "pred"
            output_dir = root / "out"
            original_dir.mkdir()
            pred_dir.mkdir()

            _write_las(original_dir / "source.las")
            _write_las(
                pred_dir / "source_pred.las",
                {"PredInstance_Model": np.array([1, 1, 0, 2], dtype=np.uint16)},
            )

            with mock.patch(
                "prediction_collection_remap.stream_add_collections_to_file",
                return_value=(4, 4),
            ) as raw_path:
                remap_prediction_collections_to_original_files(
                    [pred_dir],
                    original_dir,
                    output_dir,
                    tolerance=0.001,
                    num_threads=6,
                    num_spatial_chunks=3,
                    prefer_copc_sources=False,
                )

            raw_path.assert_called_once()
            _, kwargs = raw_path.call_args
            self.assertEqual(kwargs["chunk_parallel_workers"], 1)
            self.assertEqual(kwargs["kdtree_workers"], 6)

    def test_copc_original_routes_through_spatial_query_fast_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            original_dir = root / "originals"
            pred_dir = root / "pred"
            output_dir = root / "out"
            original_dir.mkdir()
            pred_dir.mkdir()

            copc_original = original_dir / "source.copc.laz"
            raw_twin = original_dir / "source.laz"
            copc_original.write_text("placeholder")
            raw_twin.write_text("placeholder")
            _write_las(
                pred_dir / "source_pred.las",
                {"PredInstance_Model": np.array([1, 1, 0, 2], dtype=np.uint16)},
            )

            fake_header = mock.Mock(point_count=4)
            fake_reader = mock.Mock()
            fake_reader.__enter__ = mock.Mock(return_value=fake_reader)
            fake_reader.__exit__ = mock.Mock(return_value=False)
            fake_reader.header = fake_header

            real_laspy_open = laspy.open

            def open_side_effect(path, *args, **kwargs):
                if Path(path).name == "source.copc.laz":
                    return fake_reader
                return real_laspy_open(path, *args, **kwargs)

            with mock.patch("prediction_collection_remap.laspy.open", side_effect=open_side_effect):
                with mock.patch(
                    "prediction_collection_remap.stream_add_collections_to_copc_file_spatial",
                    return_value=(4, 4),
                ) as fast_path:
                    remap_prediction_collections_to_original_files(
                        [pred_dir],
                        original_dir,
                        output_dir,
                        tolerance=0.001,
                        num_threads=1,
                        num_spatial_chunks=7,
                    )

            fast_path.assert_called_once()
            args, kwargs = fast_path.call_args
            self.assertEqual(args[0], copc_original)
            self.assertEqual(args[1], output_dir / "source.laz")
            self.assertEqual(kwargs["num_spatial_chunks"], 7)

    def test_raw_original_mode_ignores_matching_copc_twin(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            original_dir = root / "originals"
            pred_dir = root / "pred"
            output_dir = root / "out"
            original_dir.mkdir()
            pred_dir.mkdir()

            raw_original = original_dir / "source.las"
            copc_twin = original_dir / "source.copc.laz"
            _write_las(raw_original, {"OriginalExtra": np.array([1, 2, 3, 4], dtype=np.uint16)})
            copc_twin.write_text("placeholder")
            _write_las(
                pred_dir / "source_pred.las",
                {"PredInstance_Model": np.array([1, 1, 0, 2], dtype=np.uint16)},
            )

            with mock.patch(
                "prediction_collection_remap.stream_add_collections_to_file",
                return_value=(4, 4),
            ) as raw_path:
                remap_prediction_collections_to_original_files(
                    [pred_dir],
                    original_dir,
                    output_dir,
                    tolerance=0.001,
                    num_threads=1,
                )

            raw_path.assert_called_once()
            args, _ = raw_path.call_args
            self.assertEqual(args[0], raw_original)
            self.assertEqual(args[1], output_dir / "source.las")

    def test_copc_prediction_subset_uses_spatial_query(self):
        root = Path("/tmp")
        fake_points = mock.MagicMock()
        fake_points.__len__.return_value = 4
        fake_points.x = np.array([500.0, 500.1, 500.2, 500.3])
        fake_points.y = np.array([600.0, 600.0, 600.1, 600.1])
        fake_points.z = np.array([50.0, 50.1, 50.2, 50.3])
        fake_points.__getitem__.side_effect = lambda key: {
            "PredInstance_Model": np.array([1, 1, 0, 2], dtype=np.uint16),
        }[key]

        fake_reader = mock.Mock()
        fake_reader.__enter__ = mock.Mock(return_value=fake_reader)
        fake_reader.__exit__ = mock.Mock(return_value=False)
        fake_reader.spatial_query.return_value = fake_points

        coll_meta = {
            "path": root / "collection",
            "files": [{
                "path": root / "source.copc.laz",
                "bounds": (499.0, 501.0, 599.0, 601.0),
                "z_bounds": (49.0, 51.0),
            }],
            "dims": ["PredInstance_Model"],
        }

        with mock.patch("prediction_collection_remap.laspy.CopcReader.open", return_value=fake_reader):
            points, dims = load_collection_subset_for_bounds(
                coll_meta,
                (500.0, 500.3, 600.0, 600.1),
                spatial_buffer=0.01,
            )

        fake_reader.spatial_query.assert_called_once()
        self.assertEqual(points.shape, (4, 3))
        np.testing.assert_array_equal(
            dims["PredInstance_Model"],
            np.array([1, 1, 0, 2], dtype=np.uint16),
        )


if __name__ == "__main__":
    unittest.main()
