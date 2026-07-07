import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from merge_tiles import _instance_order_north_to_south  # noqa: E402


class MergeInstanceOrderTests(unittest.TestCase):
    def test_orders_instances_by_bbox_center_north_then_west(self):
        points = np.array(
            [
                [0.0, 0.0, 0.0],   # ground ignored
                [10.0, 10.0, 0.0],
                [12.0, 12.0, 0.0],  # instance 7 center y=11, x=11
                [1.0, 10.0, 0.0],
                [3.0, 12.0, 0.0],   # instance 3 center y=11, x=2
                [8.0, 4.0, 0.0],
                [10.0, 6.0, 0.0],   # instance 42 center y=5, x=9
            ],
            dtype=np.float64,
        )
        instances = np.array([0, 7, 7, 3, 3, 42, 42], dtype=np.int32)

        self.assertEqual(_instance_order_north_to_south(points, instances), [3, 7, 42])

    def test_returns_empty_when_only_ground(self):
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float64)
        instances = np.array([0, 0], dtype=np.int32)

        self.assertEqual(_instance_order_north_to_south(points, instances), [])


if __name__ == "__main__":
    unittest.main()
