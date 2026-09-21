import json
import sys
import tempfile
import unittest
from pathlib import Path

import laspy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from strict_prediction_pipeline import merge_collections, strict_remap
from test_dense_tile_merge import write_cloud


def layout(root, number):
    path = root / "bounds.json"
    path.write_text(json.dumps({"tile_buffer": 1, "tiles": [
        {"bounds": [[-.1 + i * .001, 2 + i * .001], [-1, 1]], "col": i, "row": 0}
        for i in range(number)
    ]}))
    return path


class StrictPipelineTests(unittest.TestCase):
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
            write_cloud(predictions / "a.las", [0], [1])
            write_cloud(predictions / "b.las", [.009], [5])
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
            write_cloud(originals / "a.las", [.015])
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


if __name__ == "__main__":
    unittest.main()
