import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tile_bounds_graph import (  # noqa: E402
    build_neighbor_graph_from_bounds_json,
    match_tiles_to_json_bounds,
)


class TileBoundsGraphTests(unittest.TestCase):
    def test_grid_neighbors_use_col_row_when_available(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tile_bounds_tindex.json"
            path.write_text(json.dumps({
                "tiles": [
                    {"col": 0, "row": 0, "bounds": [[0, 10], [0, 10]]},
                    {"col": 1, "row": 0, "bounds": [[10, 20], [0, 10]]},
                    {"col": 0, "row": 1, "bounds": [[0, 10], [10, 20]]},
                ]
            }))

            bounds, centers, neighbors = build_neighbor_graph_from_bounds_json(path)

        self.assertEqual(bounds[0], (0.0, 10.0, 0.0, 10.0))
        self.assertEqual(centers[0], (5.0, 5.0))
        self.assertEqual(neighbors[0]["east"], 1)
        self.assertEqual(neighbors[0]["north"], 2)
        self.assertIsNone(neighbors[0]["west"])
        self.assertIsNone(neighbors[0]["south"])

    def test_match_tiles_to_json_bounds_supports_tolerance(self):
        json_bounds = [(0.0, 10.0, 0.0, 10.0), (10.0, 20.0, 0.0, 10.0)]
        centers = [(5.0, 5.0), (15.0, 5.0)]
        tile_boundaries = {
            "c00_r00": (0.05, 10.05, 0.0, 10.0),
            "c01_r00": (10.04, 20.04, 0.0, 10.0),
        }

        tile_to_json, json_to_tile = match_tiles_to_json_bounds(tile_boundaries, json_bounds, centers)

        self.assertEqual(tile_to_json, {"c00_r00": 0, "c01_r00": 1})
        self.assertEqual(json_to_tile, {0: "c00_r00", 1: "c01_r00"})

    def test_match_tiles_to_json_bounds_assigns_slightly_offset_tiles_by_best_match(self):
        json_bounds = [
            (0.0, 10.0, 0.0, 10.0),
            (10.0, 20.0, 0.0, 10.0),
            (20.0, 30.0, 0.0, 10.0),
        ]
        centers = [(5.0, 5.0), (15.0, 5.0), (25.0, 5.0)]
        tile_boundaries = {
            "left_shifted": (0.42, 10.42, 0.03, 10.03),
            "middle_shifted": (9.61, 19.61, -0.04, 9.96),
            "right_shifted": (20.35, 30.35, 0.02, 10.02),
        }

        tile_to_json, json_to_tile = match_tiles_to_json_bounds(tile_boundaries, json_bounds, centers)

        self.assertEqual(
            tile_to_json,
            {"left_shifted": 0, "middle_shifted": 1, "right_shifted": 2},
        )
        self.assertEqual(
            json_to_tile,
            {0: "left_shifted", 1: "middle_shifted", 2: "right_shifted"},
        )

    def test_match_tiles_to_json_bounds_fails_for_unmatched_tile(self):
        with self.assertRaisesRegex(ValueError, "Unmatched tiles: far_away"):
            match_tiles_to_json_bounds(
                {
                    "c00_r00": (0.0, 10.0, 0.0, 10.0),
                    "far_away": (100.0, 110.0, 100.0, 110.0),
                },
                [(0.0, 10.0, 0.0, 10.0), (10.0, 20.0, 0.0, 10.0)],
                [(5.0, 5.0), (15.0, 5.0)],
            )


if __name__ == "__main__":
    unittest.main()
