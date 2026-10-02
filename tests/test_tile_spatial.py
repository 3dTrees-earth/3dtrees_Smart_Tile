import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tile_spatial import (  # noqa: E402
    compute_centroids_vectorized,
    find_overlap_region,
    find_spatial_neighbors,
)


class TileSpatialTests(unittest.TestCase):
    def test_find_overlap_region(self):
        self.assertEqual(
            find_overlap_region((0, 10, 0, 10), (5, 15, 2, 8)),
            (5, 10, 2, 8),
        )
        self.assertIsNone(find_overlap_region((0, 10, 0, 10), (10, 20, 0, 10)))

    def test_compute_centroids_vectorized_ignores_background(self):
        points = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [99.0, 99.0, 99.0],
        ])
        instances = np.array([1, 1, 2, 0])

        centroids = compute_centroids_vectorized(points, instances)

        np.testing.assert_allclose(centroids[1], np.array([1.0, 0.0, 0.0]))
        np.testing.assert_allclose(centroids[2], np.array([10.0, 0.0, 0.0]))
        self.assertNotIn(0, centroids)

    def test_find_spatial_neighbors_prefers_cardinal_overlap(self):
        all_tiles = {
            "center": (0.0, 10.0, 0.0, 10.0),
            "east": (8.0, 18.0, 0.0, 10.0),
            "north": (0.0, 10.0, 8.0, 18.0),
            "diagonal": (8.0, 18.0, 8.0, 18.0),
        }

        neighbors = find_spatial_neighbors(all_tiles["center"], "center", all_tiles, tolerance=1.5)

        self.assertEqual(neighbors["east"], "east")
        self.assertEqual(neighbors["north"], "north")
        self.assertIsNone(neighbors["west"])
        self.assertIsNone(neighbors["south"])


if __name__ == "__main__":
    unittest.main()
