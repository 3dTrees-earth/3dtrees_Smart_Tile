"""Coarse-only merge uses shared samples without widening identity distances."""
import json
from pathlib import Path
import tempfile
import unittest

import laspy
import numpy as np

from test_dense_instance_ownership import core_layout
from test_merge_stages import write_cloud
from remap_first_pipeline import merge_collections


def coarse_fixture(root):
    """Thirty coarse representatives, independent tile IDs and four known trees."""
    source = root / "input"
    source.mkdir()
    for tile in (0, 1):
        rows = [(x, 0., 0., 1) for x in
                ([.8, .9, 1., 1.1] if tile == 0 else [.8, .9, 1., 1.1, 1.2])]
        # These distinct opposite-core trees are only 8 cm apart: changing the
        # identity radius to a 10 cm voxel diagonal would falsely link them.
        rows += [(x, .5, z, truth) for truth, x in ((2, .96), (3, 1.04))
                 for z in (1., 1.1, 1.2)]
        # Right-tile centroid lies outside its core. Its unique tip must survive
        # orphan recovery, even after it is reconciled with the left fragment.
        rows += [(x, .8, 3., 4) for x in ([.9] if tile == 0 else [.9, 1.05])]
        corners = ([(-1., -1.), (1.2, 1.), (1., .9)] if tile == 0
                   else [(.8, -1.), (3., 1.), (1., .9)])
        rows += [(x, y, -1., 0) for x, y in corners]
        rows = np.asarray(rows)
        truth = rows[:, 3].astype(np.uint8)
        ids = np.where(truth > 0, truth + (10 if tile == 0 else 20), 0)
        path = write_cloud(source / f"c{tile:02d}_r00.las", rows[:, 0], ids,
                           np.where(truth > 0, 2 + tile, 0), ys=rows[:, 1], zs=rows[:, 2])
        cloud = laspy.read(path)
        cloud.add_extra_dim(laspy.ExtraBytesParams(name="TruthTree", type=np.uint8))
        cloud.add_extra_dim(laspy.ExtraBytesParams(name="Confidence", type=np.uint16,
                                                 no_data=[65535]))
        cloud.TruthTree = truth
        cloud.Confidence = np.full(len(rows), 65535, dtype=np.uint16)
        cloud.intensity = np.arange(len(rows), dtype=np.uint16) + 100 * (tile + 1)
        cloud.write(path)
    return source, core_layout(root)


def records(cloud, *, include_id=False):
    """Canonical XYZ + source attributes, independently of the merge report."""
    fields = [np.round(cloud.x, 6), np.round(cloud.y, 6), np.round(cloud.z, 6),
              cloud.TruthTree, cloud.PredSemantic, cloud.intensity, cloud.Confidence]
    if include_id:
        fields.append(cloud.PredInstance)
    return sorted(zip(*fields))


def declared_no_data(cloud, name):
    # laspy's reread DimensionInfo omits no_data; inspect the actual file VLR.
    descriptors = [entry for vlr in cloud.header.vlrs
                   for entry in getattr(vlr, "extra_bytes_structs", ())
                   if entry.format_name() == name]
    assert len(descriptors) == 1, (name, len(descriptors))
    return descriptors[0].no_data


class CoarseFilteringTests(unittest.TestCase):
    def check_mode(self, matching):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, layout = coarse_fixture(root)
            inputs = [laspy.read(p) for p in sorted(source.glob("*.las"))]
            for cloud in inputs:
                np.testing.assert_array_equal(declared_no_data(cloud, "Confidence"), [65535])
            source_records = {row for cloud in inputs for row in records(cloud)}
            self.assertEqual(sum(len(c.points) for c in inputs), 30)
            expected_xyz = {
                0: {(-1., -1., -1.), (3., 1., -1.), (1., .9, -1.)},
                1: {(x, 0., 0.) for x in (.8, .9, 1., 1.1, 1.2)},
                2: {(.96, .5, z) for z in (1., 1.1, 1.2)},
                3: {(1.04, .5, z) for z in (1., 1.1, 1.2)},
                4: {(x, .8, 3.) for x in (.9, 1.05)},
            }
            canonical = None
            for workers in (1, 2):
                with self.subTest(matching=matching, workers=workers):
                    output = root / f"workers_{workers}" / "output_tiles"
                    merge_collections(collections=[source], target_dir=None,
                                      output_tiles=output, tile_bounds_json=layout,
                                      ready=True, resolution_1=.1, matching=matching,
                                      workers=workers, keep_unfiltered=False)
                    clouds = [(p.name, laspy.read(p)) for p in sorted(output.glob("*.laz"))]
                    actual_records = [row for _, cloud in clouds for row in records(cloud)]
                    self.assertEqual(len(actual_records), 16)
                    self.assertTrue(set(actual_records).issubset(source_records))
                    per_truth, per_id, owners = {}, {}, {}
                    for filename, cloud in clouds:
                        np.testing.assert_array_equal(declared_no_data(cloud, "Confidence"), [65535])
                        for row in records(cloud, include_id=True):
                            xyz, truth, uid = tuple(row[:3]), int(row[3]), int(row[-1])
                            per_truth.setdefault(truth, []).append((xyz, uid))
                            if truth == 0:
                                self.assertEqual(uid, 0)
                            else:
                                self.assertGreater(uid, 0)
                                per_id.setdefault(uid, set()).add(truth)
                                owners.setdefault(uid, set()).add(filename)
                    self.assertEqual(set(per_truth), set(expected_xyz))
                    for truth, expected in expected_xyz.items():
                        actual = [xyz for xyz, _ in per_truth[truth]]
                        self.assertEqual(set(actual), expected)
                        self.assertEqual(len(actual), len(expected))  # no repeated samples
                        if truth:
                            expected_groups = 1 if matching or truth in (2, 3) else 2
                            self.assertEqual(len({uid for _, uid in per_truth[truth]}), expected_groups)
                    # Ground truth is designed independently of the algorithm:
                    # neither reconciliation nor deduplication may join trees.
                    self.assertTrue(all(len(truths) == 1 for truths in per_id.values()))
                    self.assertTrue(all(len(files) == 1 for files in owners.values()))
                    self.assertEqual(len(per_id), 4 if matching else 6)
                    # File placement, IDs, semantics and attributes must all be
                    # deterministic; compare data records, not compressed bytes.
                    current = {name: records(cloud, include_id=True) for name, cloud in clouds}
                    if canonical is not None:
                        self.assertEqual(current, canonical)
                    canonical = current
                    manifest = json.loads((output / "smarttile_merge.json").read_text())
                    self.assertEqual(manifest["resolution_1_m"], .1)

    def test_matching_reconciles_shared_coarse_samples_without_false_links(self):
        self.check_mode(matching=True)

    def test_disabled_matching_keeps_fragments_distinct_and_recovers_geometry(self):
        self.check_mode(matching=False)


if __name__ == "__main__":
    unittest.main()
