import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from merge_small_instances import merge_small_volume_instances  # noqa: E402


class MergeSmallInstancesTests(unittest.TestCase):
    def test_unsorted_summary_matches_presorted_reassignment(self):
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [20.0, 20.0, 0.0],
                [20.0, 21.0, 0.0],
                [21.0, 20.0, 0.0],
                [21.0, 21.0, 0.0],
                [0.2, 0.2, 0.0],
                [0.3, 0.3, 0.0],
                [20.2, 20.2, 0.0],
                [20.3, 20.3, 0.0],
            ],
            dtype=np.float64,
        )
        instances = np.array([1, 1, 1, 1, 9, 9, 9, 9, 3, 3, 7, 7], dtype=np.int32)

        sorted_order = np.argsort(instances)
        sorted_instances = instances[sorted_order]
        sorted_points = points[sorted_order]
        unique_inst, first_idx, inst_counts = np.unique(
            sorted_instances,
            return_index=True,
            return_counts=True,
        )

        unsorted_result, _ = merge_small_volume_instances(
            points,
            instances.copy(),
            min_points_for_hull_check=4,
            min_cluster_size=3,
            max_volume_for_merge=4.0,
            max_search_radius=5.0,
            num_threads=2,
            verbose=False,
        )
        presorted_result, _ = merge_small_volume_instances(
            points,
            instances.copy(),
            min_points_for_hull_check=4,
            min_cluster_size=3,
            max_volume_for_merge=4.0,
            max_search_radius=5.0,
            num_threads=2,
            verbose=False,
            presorted_points=sorted_points,
            presorted_instances=sorted_instances,
            presorted_unique_inst=unique_inst,
            presorted_first_idx=first_idx,
            presorted_inst_counts=inst_counts,
        )

        np.testing.assert_array_equal(unsorted_result, presorted_result)
        np.testing.assert_array_equal(unsorted_result, [1, 1, 1, 1, 9, 9, 9, 9, 1, 1, 9, 9])


if __name__ == "__main__":
    unittest.main()
