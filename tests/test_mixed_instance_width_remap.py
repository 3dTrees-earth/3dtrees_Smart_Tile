"""Dataset 2924 (3DT-2172): tiles with uint16 and uint32 instance IDs must not wrap."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import laspy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from remap_first_pipeline import strict_remap
from test_merge_stages import write_cloud


def write_tile(path, xs, ids, dtype):
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.000001] * 3)
    cloud = laspy.LasData(header)
    cloud.x, cloud.y, cloud.z = xs, np.zeros(len(xs)), np.zeros(len(xs))
    cloud.add_extra_dim(laspy.ExtraBytesParams(name="PredInstance", type=dtype))
    cloud.PredInstance = np.asarray(ids, dtype=dtype)
    cloud.write(path)


class MixedInstanceWidthRemapTests(unittest.TestCase):
    def test_uint32_ids_from_a_later_tile_are_not_wrapped_into_a_uint16_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions, originals = root / "predictions", root / "originals"
            predictions.mkdir()
            originals.mkdir()
            # 70,674 wraps to 5,138 in uint16, which would merge it with tree 5,138.
            write_tile(predictions / "tile_00000.las", [0, 1, 2], [5138, 5138, 7], np.uint16)
            write_tile(predictions / "tile_00001.las", [10, 11, 12], [70674, 70674, 70675], np.uint32)
            write_cloud(originals / "a.las", [0, 1, 2, 10, 11, 12])
            strict_remap(collections=[predictions], baseline_collections=[predictions],
                         originals=originals, output=root / "out")
            final = laspy.read(root / "out/a.las")
            labels = final.PredInstance.tolist()
            # Four distinct trees stay four distinct trees after compaction.
            self.assertEqual(len(set(labels)), 4)
            self.assertEqual(labels[0], labels[1])
            self.assertNotEqual(labels[0], labels[3])
            self.assertEqual(labels[3], labels[4])
            mapping = json.loads((root / "out/instance_mapping.json").read_text())
            sources = {row["source_instance_id"] for row in mapping["models"]["PredInstance"]["instances"]}
            self.assertEqual(sources, {7, 5138, 70674, 70675})


if __name__ == "__main__":
    unittest.main()
