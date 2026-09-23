import json
import sys
import tempfile
import unittest
from pathlib import Path

import laspy
import numpy as np
from laspy.vlrs.vlrlist import VLRList

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from strict_prediction_pipeline import merge_collections, strict_remap
from test_dense_tile_merge import write_cloud


def layout(root, number):
    path = root / "bounds.json"
    path.write_text(json.dumps({"tile_buffer": 1, "tiles": [
        # These fixtures isolate reconciliation/deduplication; both cores own the test points.
        {"bounds": [[-.1 + i * .001, 2 + i * .001], [-1, 1]],
         "core": [[-.1, 2], [-1, 1]], "col": i, "row": 0}
        for i in range(number)
    ]}))
    return path


def add_extended_metadata(path):
    cloud = laspy.convert(laspy.read(path), file_version="1.4")
    cloud.add_extra_dim(laspy.ExtraBytesParams(name="quality", type=np.int16, no_data=[-99]))
    cloud.quality = [-99] * len(cloud.points)
    cloud.header.evlrs = VLRList([
        laspy.VLR(user_id="test", record_id=99, record_data=b"important schema")])
    cloud.write(path)


def assert_extended_metadata(test, path):
    cloud = laspy.read(path)
    test.assertIn(b"important schema", [v.record_data_bytes() for v in cloud.header.evlrs])
    descriptor = next(s for v in cloud.header.vlrs for s in getattr(v, "extra_bytes_structs", ())
                      if s.format_name() == "quality")
    np.testing.assert_array_equal(descriptor.no_data, [-99])
    np.testing.assert_array_equal(cloud.quality, [-99] * len(cloud.points))


class StrictPipelineTests(unittest.TestCase):
    def test_original_voxel_diagonal_gap_is_covered_and_scales_with_resolution(self):
        # 3057: the center-of-mass representative was 10.69 mm from one
        # original point in the same 1 cm voxel.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions, originals = root / "predictions", root / "originals"
            predictions.mkdir()
            originals.mkdir()
            write_cloud(predictions / "a.las", [0], [7], [3])
            write_cloud(originals / "a.las", [.0058], ys=[-.0062], zs=[-.0065])
            report = strict_remap(collections=[predictions], baseline_collections=[predictions],
                                  originals=originals, output=root / "enriched")
            self.assertAlmostEqual(report["original_radius_m"], np.sqrt(3) * .01)
            self.assertEqual(report["state"], "validated")
            self.assertEqual(laspy.read(root / "enriched/a.las").PredInstance.tolist(), [7])
            with self.assertRaisesRegex(ValueError, "100% original coverage"):
                strict_remap(collections=[predictions], baseline_collections=[predictions],
                             originals=originals, output=root / "finer", resolution_1=.005)

    def test_remap_uses_manifest_resolution_when_not_explicitly_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions, originals = root / "predictions", root / "originals"
            predictions.mkdir()
            originals.mkdir()
            write_cloud(predictions / "a.las", [0], [7], [3])
            write_cloud(originals / "a.las", [.0107])
            (predictions / "smarttile_merge.json").write_text(json.dumps({
                "baseline": ".", "resolution_1_m": .005}))
            with self.assertRaisesRegex(ValueError, "100% original coverage"):
                strict_remap(collections=[predictions], originals=originals,
                             output=root / "finer")
            report = json.loads((root / "finer_coverage.json").read_text())
            self.assertAlmostEqual(report["original_radius_m"], np.sqrt(3) * .005)
            self.assertFalse((root / "finer").exists())

    def test_transfer_radius_accepts_com_gap_without_relaxing_original_coverage(self):
        for distance, original_x, expected in [(.131, 0, 'pass'), (.174, 0, 'transfer'), (.131, .018, 'original')]:
            with self.subTest(distance=distance, original_x=original_x), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source, target, originals = [root / n for n in ('source', 'target', 'originals')]
                for folder in (source, target, originals):folder.mkdir()
                write_cloud(source / 'a.las', [distance], [7], [3])
                write_cloud(target / 'a.las', [0])
                write_cloud(originals / 'a.las', [original_x])
                kwargs = dict(collections=[source], target_dir=target, output_tiles=root / 'out',
                              tile_bounds_json=layout(root, 1), originals=originals)
                if expected == 'pass':
                    report = merge_collections(**kwargs)
                    self.assertEqual(report['models'][0]['transfer'][0]['radius_m'], .1732)
                    self.assertAlmostEqual(report['original_radius_m'], np.sqrt(3) * .01)
                    self.assertEqual(report['state'], 'validated')
                    self.assertEqual(laspy.read(root / 'original_with_predictions/a.las').PredSemantic.tolist(), [3])
                else:
                    message = 'incomplete prediction assignment' if expected == 'transfer' else '100% original coverage'
                    with self.assertRaisesRegex(ValueError, message):merge_collections(**kwargs)
                    self.assertFalse((root / 'out').exists())

    def test_copc_hierarchy_is_dropped_but_passenger_evlrs_are_retained(self):
        from laspy.copc import CopcHierarchyVlr
        from laspy.vlrs.vlrlist import VLRList
        from point_cloud_metadata import copy_single_source_header, write_retained_evlrs
        header = laspy.LasHeader(point_format=6, version="1.4")
        header.evlrs = VLRList([CopcHierarchyVlr(),
            laspy.VLR(user_id="test", record_id=99, record_data=b"passenger")])
        copied = copy_single_source_header(header)
        self.assertEqual([(v.user_id, v.record_id) for v in copied.evlrs], [("test", 99)])
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "dense.laz"
            with laspy.open(output, mode="w", header=copied) as writer:
                writer.write_points(laspy.ScaleAwarePointRecord.zeros(1, header=copied))
                write_retained_evlrs(writer, copied)
            with laspy.open(output) as reader:
                self.assertEqual(reader.header.evlrs[0].record_data_bytes(), b"passenger")

    def test_streaming_stages_preserve_evlrs_and_source_no_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, targets, originals = [root / n for n in ("source", "targets", "originals")]
            for folder in (source, targets, originals):
                folder.mkdir()
            write_cloud(source / "a.las", [0, .1], [1, 1])
            for folder in (targets, originals):
                add_extended_metadata(write_cloud(folder / "a.las", [0, .1]))
            merge_collections(collections=[source], target_dir=targets, output_tiles=root / "out",
                              tile_bounds_json=layout(root, 1), originals=originals,
                              merged_output=root / "merged.laz")
            for file in (root / "out_unfiltered_1cm/tile_00000.laz", root / "out/tile_00000.laz",
                         root / "merged.laz", root / "original_with_predictions/a.las"):
                with self.subTest(file=file):
                    assert_extended_metadata(self, file)

    def test_standalone_enrichment_preserves_evlrs_and_source_no_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, originals = root / "source", root / "originals"
            source.mkdir()
            originals.mkdir()
            write_cloud(source / "a.las", [0], [1])
            add_extended_metadata(write_cloud(originals / "a.las", [0]))
            strict_remap(collections=[source], baseline_collections=[source], originals=originals,
                         output=root / "enriched")
            assert_extended_metadata(self, root / "enriched/a.las")

    def test_complete_transfer_precedes_dedup_and_original_metadata_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions, targets, originals = [root / n for n in ("predictions", "targets", "originals")]
            for folder in (predictions, targets, originals):
                folder.mkdir()
            write_cloud(predictions / "a.las", [0, .1], [1, 1], [2, 2])
            write_cloud(targets / "a.las", [0, .009, .1])
            original = write_cloud(originals / "original.las", [.001, .01, .101])
            report = merge_collections(collections=[predictions], target_dir=targets,
                output_tiles=root / "out", tile_bounds_json=layout(root, 1), originals=originals)
            self.assertEqual(report["state"], "validated")
            self.assertEqual(report["models"][0]["transfer"][0]["matched"], 3)
            before, after = laspy.read(original), laspy.read(root / "original_with_predictions/original.las")
            for dim in before.points.array.dtype.names:
                np.testing.assert_array_equal(before.points.array[dim], after.points.array[dim])
            np.testing.assert_array_equal(before.header.scales, after.header.scales)
            self.assertEqual(after.PredInstance.tolist(), [1, 1, 1])

    def test_triangle_gap_fails_final_coverage_without_restoration_or_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions, originals = root / "predictions", root / "originals"
            predictions.mkdir()
            originals.mkdir()
            write_cloud(predictions / "a.las", [0], [0])
            write_cloud(predictions / "b.las", [.009], [0])
            write_cloud(originals / "a.las", [.018])
            with self.assertRaisesRegex(ValueError, "100% original coverage"):
                merge_collections(collections=[predictions], target_dir=None, output_tiles=root / "out",
                    tile_bounds_json=layout(root, 2), originals=originals, ready=True)
            report = json.loads((root / "remap_first_report.json").read_text())
            self.assertEqual([m["matched"] for m in report["original_coverage"]], [1, 0])
            self.assertFalse((root / "out").exists())
            self.assertFalse((root / "original_with_predictions").exists())
            self.assertEqual(report["models"][0]["deduplication"]["tiles"][1]["removed"], 1)

    def test_baseline_sampling_gap_is_separate_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions, originals = root / "predictions", root / "originals"
            predictions.mkdir()
            originals.mkdir()
            write_cloud(predictions / "a.las", [0], [0])
            write_cloud(originals / "a.las", [.018])
            with self.assertRaisesRegex(ValueError, "unfiltered_1cm"):
                merge_collections(collections=[predictions], target_dir=None, output_tiles=root / "out",
                    tile_bounds_json=layout(root, 1), originals=originals, ready=True)
            report = json.loads((root / "remap_first_report.json").read_text())
            self.assertEqual([m["matched"] for m in report["original_coverage"]], [0, 0])

    def test_models_fail_independently_and_successful_originals_are_not_published(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sat, fm, originals = [root / n for n in ("sat", "fm", "originals")]
            for folder in (sat, fm, originals):
                folder.mkdir()
            write_cloud(sat / "a.las", [0, 1], [1, 2], instance="PredInstance_SAT")
            write_cloud(fm / "a.las", [0], [10], instance="PredInstance_FM")
            write_cloud(originals / "a_good.las", [0])
            write_cloud(originals / "z_bad.las", [1])
            with self.assertRaisesRegex(ValueError, "100% original coverage"):
                merge_collections(collections=[sat, fm], target_dir=None, output_tiles=root / "out",
                    tile_bounds_json=layout(root, 1), originals=originals, ready=True)
            report = json.loads((root / "remap_first_report.json").read_text())
            failures = report["coverage_failures"]
            self.assertTrue(all(f["model"] == "fm" and f["file"] == "z_bad.las" for f in failures))
            self.assertFalse((root / "original_with_predictions").exists())

    def test_separate_final_remap_reads_baseline_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, originals = root / "source", root / "originals"
            source.mkdir()
            originals.mkdir()
            write_cloud(source / "tile.las", [0, .1], [1, 1])
            write_cloud(originals / "original.las", [0, .101])
            merge_collections(collections=[source], target_dir=None, output_tiles=root / "out",
                              tile_bounds_json=layout(root, 1), ready=True)
            report = strict_remap(collections=[root / "out"], originals=originals, output=root / "enriched")
            self.assertEqual(report["state"], "validated")
            self.assertTrue((root / "enriched/original.las").exists())

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "out"
            output.mkdir()
            (output / "sentinel").write_text("keep")
            with self.assertRaises(FileExistsError):
                merge_collections(collections=[root / "predictions"], target_dir=None, output_tiles=output,
                                  tile_bounds_json=layout(root, 1))
            self.assertEqual((output / "sentinel").read_text(), "keep")

    def test_original_vectors_header_and_vlrs_survive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, originals = root / "source", root / "originals"
            source.mkdir()
            originals.mkdir()
            write_cloud(source / "a.las", [0, .1], [1, 1])
            original = write_cloud(originals / "a.las", [0, .1])
            cloud = laspy.read(original)
            cloud.header.vlrs.append(laspy.VLR(user_id="test", record_id=8, record_data=b"source metadata"))
            cloud.add_extra_dim(laspy.ExtraBytesParams(name="vector", type="3i2", scales=[.1, .2, .3], offsets=[1, 2, 3]))
            cloud.points.array["vector"] = [[-5, 6, 7], [1, 2, 3]]
            cloud.write(original)
            merge_collections(collections=[source], target_dir=None, output_tiles=root / "out",
                              tile_bounds_json=layout(root, 1), originals=originals, ready=True)
            after = laspy.read(root / "original_with_predictions/a.las")
            np.testing.assert_array_equal(cloud.points.array["vector"], after.points.array["vector"])
            np.testing.assert_array_equal(cloud.header.point_format.dimension_by_name("vector").scales,
                                          after.header.point_format.dimension_by_name("vector").scales)
            self.assertIn(b"source metadata", [v.record_data_bytes() for v in after.header.vlrs])

    def test_projected_offset_centimetre_boundary_is_inclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, originals = root / "source", root / "originals"
            source.mkdir()
            originals.mkdir()
            pred = write_cloud(source / "a.las", [0], [1])
            orig = write_cloud(originals / "a.las", [.01])
            for file in (pred, orig):
                cloud = laspy.read(file)
                cloud.header.offsets = np.array([500_000., 5_000_000., 0.])
                cloud.points.offsets = cloud.header.offsets
                cloud.write(file)
            report = merge_collections(collections=[source], target_dir=None, output_tiles=root / "out",
                                       tile_bounds_json=layout(root, 1), originals=originals, ready=True)
            self.assertEqual(report["state"], "validated")

    def test_standalone_scalar_prediction_collection_is_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred, original = root / "pred", root / "original"
            pred.mkdir()
            original.mkdir()
            file = write_cloud(pred / "species.las", [0])
            cloud = laspy.read(file)
            cloud.add_extra_dim(laspy.ExtraBytesParams(name="species_probability", type=np.float32))
            cloud.species_probability = [.75]
            cloud.write(file)
            write_cloud(original / "a.las", [0])
            strict_remap(collections=[pred], baseline_collections=[pred], originals=original, output=root / "out")
            self.assertEqual(laspy.read(root / "out/a.las").species_probability[0], .75)

    def test_standalone_remap_accepts_nan_descriptor_and_retains_finite_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions, originals = root / "predictions", root / "originals"
            predictions.mkdir()
            originals.mkdir()
            file = write_cloud(predictions / "a.las", [0, .1], [1, 1])
            cloud = laspy.read(file)
            cloud.add_extra_dim(laspy.ExtraBytesParams(name="score", type="f4", no_data=[np.nan]))
            cloud.score = [.5, .75]
            cloud.write(file)
            write_cloud(originals / "a.las", [0, .1])
            report = strict_remap(collections=[predictions], baseline_collections=[predictions],
                                  originals=originals, output=root / "out")
            self.assertEqual(report["state"], "validated")
            result = laspy.read(root / "out/a.las")
            np.testing.assert_array_equal(result.score, [.5, .75])
            descriptor = next(d for v in result.header.vlrs for d in getattr(v, "extra_bytes_structs", ())
                              if d.format_name() == "score")
            self.assertTrue(np.isnan(descriptor.no_data[0]))


if __name__ == "__main__":
    unittest.main()
