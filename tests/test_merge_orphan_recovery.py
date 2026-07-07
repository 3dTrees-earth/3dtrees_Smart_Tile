import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from merge_orphan_recovery import recover_orphaned_instances  # noqa: E402
from merge_tile_loading import TileData  # noqa: E402


class MergeOrphanRecoveryTests(unittest.TestCase):
    def test_no_filtered_instances_returns_without_building_bboxes(self):
        tile = TileData(
            name="c00_r00",
            points=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]),
            instances=np.array([1, 1], dtype=np.int32),
            boundary=(0.0, 1.0, 0.0, 1.0),
        )

        with mock.patch("merge_orphan_recovery._build_instance_bboxes") as build_bboxes:
            result = recover_orphaned_instances(
                tiles=[tile],
                kept_instances_per_tile={"c00_r00": {1}},
                filtered_instances_per_tile={"c00_r00": set()},
                buffer_direction_per_tile={"c00_r00": {}},
                neighbors_by_tile={
                    "c00_r00": {
                        "east": None,
                        "west": None,
                        "north": None,
                        "south": None,
                    }
                },
                global_to_merged={1: 1},
                merged_instance_sources={1: [1]},
                buffer=10.0,
                border_zone_width=10.0,
                num_threads=20,
            )

        self.assertEqual(result, {0: "c00_r00"})
        build_bboxes.assert_not_called()


if __name__ == "__main__":
    unittest.main()
