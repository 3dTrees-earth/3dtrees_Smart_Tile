import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import laspy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from small_instance_reassignment import (
    InstanceStatistics, SEARCH_RADIUS_M, SmallInstancePolicy, plan_reassignment,
)
from strict_prediction_pipeline import merge_collections
from test_dense_tile_merge import write_cloud
from test_strict_prediction_pipeline import layout


def cube(center, count, edge):
    """``count`` distinct points on a regular grid inside an axis-aligned cube."""
    side = int(np.ceil(count ** (1 / 3)))
    grid = np.stack(np.meshgrid(*[np.linspace(-edge / 2, edge / 2, side)] * 3, indexing="ij"), -1)
    return grid.reshape(-1, 3)[:count] + np.asarray(center, dtype=float)


def statistics(instances):
    stats = InstanceStatistics()
    for label, xyz in instances.items():
        stats.add(np.full(len(xyz), label), xyz)
    return stats


class PlanTests(unittest.TestCase):
    def test_count_and_bbox_limits_are_strict(self):
        stats = statistics({
            1: cube((0, 0, 0), 5000, 2.0),     # large target
            2: cube((1, 0, 0), 2999, 1.0),     # 2999 points, 1 m3 -> small
            3: cube((0, 1, 0), 3000, 0.5),     # 3000 points -> not small
            4: cube((0, -1, 0), 27, 1.6),      # full 1.6 m cube: 4.096 m3 -> not small
        })
        mapping, report = plan_reassignment(stats, SmallInstancePolicy(3000, 4.0))
        self.assertEqual(mapping, {2: 1})
        self.assertEqual(report["small"], 1)
        self.assertEqual(report["reassigned"][0]["target"], 1)

    def test_nearest_non_small_centroid_within_search_radius(self):
        stats = statistics({
            1: cube((0, 0, 0), 4000, 2.0),
            2: cube((10, 0, 0), 4000, 2.0),
            3: cube((8, 0, 0), 20, 0.2),                       # nearer to 2
            4: cube((0, SEARCH_RADIUS_M + 1, 0), 20, 0.2),     # no target within radius
        })
        mapping, report = plan_reassignment(stats, SmallInstancePolicy())
        self.assertEqual(mapping, {3: 2})
        self.assertEqual([d["instance"] for d in report["kept"]], [4])

    def test_distance_is_horizontal_between_xy_centroids(self):
        stats = statistics({
            1: cube((0, 0, 12), 4096, 2.0),    # crown centroid high above the fragment
            2: cube((3, 0, 0), 4096, 2.0),     # nearer in 3D, farther in XY
            3: cube((0.5, 0, 0), 27, 0.2),     # fragment at the stem base
        })
        mapping, report = plan_reassignment(stats, SmallInstancePolicy())
        self.assertEqual(mapping, {3: 1})
        self.assertAlmostEqual(report["reassigned"][0]["xy_distance_m"], 0.5, places=6)

    def test_only_small_instances_are_kept(self):
        mapping, report = plan_reassignment(statistics({1: cube((0, 0, 0), 5, .1),
                                                        2: cube((1, 0, 0), 5, .1)}),
                                            SmallInstancePolicy())
        self.assertEqual(mapping, {})
        self.assertEqual(len(report["kept"]), 2)

    def test_streamed_batches_match_one_batch(self):
        xyz = np.random.default_rng(0).random((500, 3))
        ids = np.random.default_rng(1).integers(0, 6, 500)
        whole, parts = InstanceStatistics(), InstanceStatistics()
        whole.add(ids, xyz)
        for chunk in np.array_split(np.arange(500), 7):
            parts.add(ids[chunk], xyz[chunk])
        for name in ("ids", "counts", "lo", "hi", "sums"):
            np.testing.assert_allclose(getattr(whole, name), getattr(parts, name))
        self.assertNotIn(0, whole.ids)

    def test_policy_rejects_invalid_limits(self):
        with self.assertRaises(ValueError):
            SmallInstancePolicy(0, 4.0)
        with self.assertRaises(ValueError):
            SmallInstancePolicy(3000, 0.0)


class MergeTests(unittest.TestCase):
    def build(self, root):
        source, originals = root / "source", root / "originals"
        source.mkdir()
        originals.mkdir()
        tree = cube((1.0, 0.0, 2.0), 3375, 1.5)
        fragment = cube((1.2, 0.2, 3.0), 27, 0.1)
        xyz = np.vstack([tree, fragment])
        ids = np.r_[np.full(len(tree), 1), np.full(len(fragment), 2)]
        write_cloud(source / "a.las", xyz[:, 0], ids, ys=xyz[:, 1], zs=xyz[:, 2])
        write_cloud(originals / "a.las", xyz[:, 0], ys=xyz[:, 1], zs=xyz[:, 2])
        return source, originals, len(fragment)

    def run_merge(self, root, policy):
        source, originals, fragment = self.build(root)
        report = merge_collections(collections=[source], target_dir=None, output_tiles=root / "out",
                                   tile_bounds_json=layout(root, 1), originals=originals, ready=True,
                                   small_instances=policy)
        return report, fragment

    def test_fragment_joins_tree_in_tiles_originals_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report, fragment = self.run_merge(root, SmallInstancePolicy())
            self.assertEqual(report["state"], "validated")
            section = report["models"][0]["small_instance_reassignment"]
            self.assertEqual(section["reassigned_count"], 1)
            self.assertEqual(section["relabeled_points_per_tile"], [fragment])
            labels = laspy.read(next((root / "out").glob("*.la*"))).PredInstance
            self.assertEqual(set(np.unique(labels)), {1})
            original = laspy.read(root / "original_with_predictions/a.las")
            self.assertEqual(set(np.unique(original.PredInstance)), {1})
            with (root / "out/instance_metadata.csv").open() as stream:
                self.assertEqual(list(csv.reader(stream))[1:], [["1", "1"]])
            summary = json.loads((root / "out/PredInstance_summary.json").read_text())
            self.assertTrue(summary["small_instance_reassignment"])
            [tree] = summary["instances"]
            self.assertEqual((tree["id"], tree["points"], tree["reassigned_from"]), (1, 3375 + fragment, [2]))
            self.assertEqual(json.loads((root / "out/smarttile_merge.json").read_text())["instance_summary"],
                             "PredInstance_summary.json")

    def test_disabled_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report, _ = self.run_merge(root, None)
            self.assertNotIn("small_instance_reassignment", report["models"][0])
            labels = laspy.read(next((root / "out").glob("*.la*"))).PredInstance
            self.assertEqual(set(np.unique(labels)), {1, 2})
            summary = json.loads((root / "out/PredInstance_summary.json").read_text())
            self.assertFalse(summary["small_instance_reassignment"])
            by_id = {i["id"]: i for i in summary["instances"]}
            self.assertEqual((by_id[1]["points"], by_id[2]["points"]), (3375, 27))
            np.testing.assert_allclose(by_id[2]["centroid"], [1.2, 0.2, 3.0], atol=1e-6)
            np.testing.assert_allclose(by_id[2]["centroid_xy"], [1.2, 0.2], atol=1e-6)
            self.assertAlmostEqual(by_id[2]["bbox_volume_m3"], 0.001, places=6)

    def test_rejected_for_raycloudtools_tree_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, _, _ = self.build(root)
            (source / "a_trees.txt").write_text("# tree\nx,y,z,radius\n")
            with self.assertRaisesRegex(ValueError, "RayCloudTools"):
                merge_collections(collections=[source], target_dir=None, output_tiles=root / "out",
                                  tile_bounds_json=layout(root, 1), ready=True,
                                  small_instances=SmallInstancePolicy())


if __name__ == "__main__":
    unittest.main()
