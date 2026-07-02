import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tile_spatial import (  # noqa: E402
    compute_centroids_vectorized,
    filter_by_centroid_in_buffer,
    find_overlap_region,
    find_spatial_neighbors,
    get_border_region_mask,
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

    def test_filter_by_centroid_in_buffer_uses_precomputed_neighbors(self):
        points = np.array([
            [1.0, 5.0, 0.0],
            [2.0, 5.0, 0.0],
            [8.0, 5.0, 0.0],
            [9.0, 5.0, 0.0],
        ])
        instances = np.array([1, 1, 2, 2])

        removed, directions = filter_by_centroid_in_buffer(
            points,
            instances,
            (0.0, 10.0, 0.0, 10.0),
            "center",
            {},
            buffer=3.0,
            precomputed_neighbors={"west": "west", "east": None, "north": None, "south": None},
        )

        self.assertEqual(removed, {1})
        self.assertEqual(directions, {1: "west"})

    def test_get_border_region_mask_preserves_edge_inclusivity(self):
        points = np.array([
            [1.0, 5.0, 0.0],
            [2.0, 5.0, 0.0],
            [8.0, 5.0, 0.0],
            [9.0, 5.0, 0.0],
        ])

        mask = get_border_region_mask(
            points,
            (0.0, 10.0, 0.0, 10.0),
            inner_dist=1.0,
            outer_dist=2.0,
            neighbors={"west": "w", "east": "e", "north": None, "south": None},
        )

        np.testing.assert_array_equal(mask, np.array([True, False, False, True]))


if __name__ == "__main__":
    unittest.main()
