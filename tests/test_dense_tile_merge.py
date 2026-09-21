import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import laspy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bounded_point_index import PointIndex
from dense_tile_merge import describe_model, prepare_dense, reconcile_instances, deduplicate


def write_cloud(path, xs, ids=None, semantics=None, instance="PredInstance", ys=None, zs=None):
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.000001] * 3)
    cloud = laspy.LasData(header)
    cloud.x, cloud.y, cloud.z = xs, (np.zeros(len(xs)) if ys is None else ys), (np.zeros(len(xs)) if zs is None else zs)
    cloud.intensity = np.arange(len(xs), dtype=np.uint16) + 5
    if ids is not None:
        cloud.add_extra_dim(laspy.ExtraBytesParams(name=instance, type=np.uint32))
        cloud[instance] = ids
    if semantics is not None:
        semantic = instance.replace("PredInstance", "PredSemantic")
        cloud.add_extra_dim(laspy.ExtraBytesParams(name=semantic, type=np.uint8))
        cloud[semantic] = semantics
    cloud.write(path)
    return path


class DenseMergeTests(unittest.TestCase):
    def run_case(self, root, tiles, *, matching=True, threshold=0.3, overlaps=None):
        source = root / "source"
        source.mkdir()
        paths = [write_cloud(source / f"{i}.las", *tile) for i, tile in enumerate(tiles)]
        model = describe_model(source, "PredInstance")
        report = {}
        with ExitStack() as stack:
            dense = stack.enter_context(PointIndex(root / "dense.sqlite", {n: p.type for n, p in model.dimensions.items()}))
            survivors = stack.enter_context(PointIndex(root / "survivors.sqlite", dense.dimensions))
            files, counts = prepare_dense(model, [(p, p, str(i)) for i, p in enumerate(paths)],
                                          root / "dense", dense, np.zeros(3), .125, report, ready=True)
            mapping = reconcile_instances(model, files, dense, np.zeros(3), counts, threshold, .05,
                                          report, enabled=matching, overlaps=overlaps)
            outputs = deduplicate(model, files, dense, survivors, np.zeros(3), mapping,
                                  root / "outputs", report, overlaps=overlaps)
        return [laspy.read(p) for p in outputs], report

    def test_exact_duplicates_reconcile_ids_and_preserve_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            outputs, report = self.run_case(Path(tmp), [([0, .1], [7, 7], [2, 2]),
                                                        ([0, .1, .5], [42, 42, 42], [2, 2, 2])])
            self.assertEqual([len(c.points) for c in outputs], [2, 1])
            self.assertAlmostEqual(outputs[1].x[0], .5)
            self.assertEqual(outputs[0].PredInstance[0], outputs[1].PredInstance[0])
            self.assertEqual(report["deduplication"]["tiles"][1]["removed"], 2)

    def test_same_tile_duplicates_are_never_thinned(self):
        with tempfile.TemporaryDirectory() as tmp:
            outputs, _ = self.run_case(Path(tmp), [([0, 0, .001], [1, 1, 1])])
            self.assertEqual(len(outputs[0].points), 3)

    def test_radius_boundary_and_xyz_distance(self):
        for distance, expected in [(0.009999, 0), (0.01, 0), (0.010001, 1)]:
            with self.subTest(distance=distance), tempfile.TemporaryDirectory() as tmp:
                outputs, _ = self.run_case(Path(tmp), [([0], [1]), ([distance], [1])])
                self.assertEqual(len(outputs[1].points), expected)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            write_cloud(source / "a.las", [0], [1], zs=[0])
            write_cloud(source / "b.las", [0], [1], zs=[.02])
            model = describe_model(source, "PredInstance")
            with PointIndex(root / "index.sqlite", {"PredInstance": np.uint32}) as index:
                from dense_tile_merge import index_file
                index_file(index, source / "a.las", 0, np.zeros(3), model)
                d, _, _ = index.nearest(np.array([[0, 0, .02]]), .01)
                self.assertFalse(np.isfinite(d[0]))

    def test_survivor_chain_does_not_erase_last_point(self):
        with tempfile.TemporaryDirectory() as tmp:
            outputs, _ = self.run_case(Path(tmp), [([0], [1]), ([.009], [5]), ([.018], [9])])
            self.assertEqual([len(c.points) for c in outputs], [1, 0, 1])

    def test_semantic_conflict_and_unreconciled_instance_are_hard_failures(self):
        for matching, labels in [(True, ([2], [3])), (False, ([2], [2]))]:
            with self.subTest(matching=matching), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaisesRegex(ValueError, "label conflict"):
                    self.run_case(Path(tmp), [([0], [1], labels[0]), ([0], [9], labels[1])], matching=matching)

    def test_non_nearest_conflicting_point_is_not_hidden(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "label conflict"):
                self.run_case(Path(tmp), [([0, .005], [1, 2], [2, 3]), ([0], [7], [2])])

    def test_missing_transfer_does_not_become_background(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = write_cloud(root / "source.las", [0], [0])
            target = write_cloud(root / "target.las", [0, 1])
            model = describe_model(source, "PredInstance")
            with PointIndex(root / "dense.sqlite", {"PredInstance": np.uint32}) as index:
                report = {}
                with self.assertRaisesRegex(ValueError, "incomplete prediction assignment"):
                    prepare_dense(model, [(source, target, "tile")], root / "dense", index,
                                  np.zeros(3), .125, report)
                self.assertEqual(report["transfer"][0]["matched"], 1)

    def test_points_outside_declared_buffer_overlap_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            outputs, _ = self.run_case(Path(tmp), [([0], [1]), ([0], [2])], overlaps=[{}, {}])
            self.assertEqual([len(c.points) for c in outputs], [1, 1])

    def test_chunk_boundaries_do_not_change_survivors_or_ids(self):
        tiles = [([0, .1, .2], [1, 1, 1]), ([.009, .1, .209], [5, 5, 5]),
                 ([.018, .1, .218], [9, 9, 9])]
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            expected, _ = self.run_case(Path(first), tiles)
            with mock.patch("dense_tile_merge.MAX_BATCH_POINTS", 1):
                actual, _ = self.run_case(Path(second), tiles)
            for a, b in zip(expected, actual):
                np.testing.assert_array_equal(a.points.array, b.points.array)

    def test_background_and_tree_disagreement_is_a_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "label conflict"):
                self.run_case(Path(tmp), [([0], [0]), ([0], [1])])

    def test_no_data_sentinel_is_not_a_valid_model_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            file = write_cloud(root / "source.las", [0], [999])
            cloud = laspy.read(file)
            cloud.remove_extra_dim("PredInstance")
            cloud.add_extra_dim(laspy.ExtraBytesParams(name="PredInstance", type=np.uint16, no_data=[999]))
            cloud.PredInstance = [999]
            cloud.write(file)
            model = describe_model(file, "PredInstance")
            with PointIndex(root / "index.sqlite", {"PredInstance": np.uint16}) as index:
                with self.assertRaisesRegex(ValueError, "no-data sentinel"):
                    prepare_dense(model, [(file, file, "tile")], root / "dense", index,
                                  np.zeros(3), .125, {}, ready=True)

    def test_nearest_ties_and_attribute_names_are_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            with PointIndex(Path(tmp) / "index.sqlite", {"xyz": np.uint16, "file": np.dtype((np.int32, 3))}) as index:
                index.add(0, np.array([[-.01, 0, 0], [.01, 0, 0]]),
                          {"xyz": np.array([2, 1], dtype=np.uint16),
                           "file": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)}, np.array([4, 2]))
                d, values, refs = index.nearest(np.zeros((1, 3)), .01)
                self.assertTrue(np.isfinite(d[0]))
                self.assertEqual(refs[0].tolist(), [0, 2])
                self.assertEqual(values["xyz"].tolist(), [1])
                self.assertEqual(values["file"].tolist(), [[4, 5, 6]])


if __name__ == "__main__":
    unittest.main()
