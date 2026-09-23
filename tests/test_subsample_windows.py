import json
from pathlib import Path
import subprocess
import sys
import unittest

import laspy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from subsample_com import aligned_edges, iter_copc_center_of_mass_windows, _query_copc_window


class SubsampleWindowTests(unittest.TestCase):
    def test_real_3083_upper_bound_terminates_without_duplicate_windows(self):
        # Run in a child so a regression fails with a timeout instead of OOM.
        source = str(Path(__file__).resolve().parents[1] / "src")
        code = (f"import sys,json;sys.path.insert(0,{source!r});"
                "from subsample_com import aligned_edges;"
                "print(json.dumps(aligned_edges(0,228.35000000000002,5,.01)))")
        completed = subprocess.run([sys.executable, "-c", code],
                                   capture_output=True, text=True, check=True, timeout=5)
        edges = json.loads(completed.stdout)
        self.assertEqual(len(edges), 46)
        self.assertEqual(edges[0][0], 0)
        self.assertEqual(edges[-1][1], 228.35000000000002)
        self.assertTrue(all(a < b for a, b in edges))
        self.assertTrue(all(a[1] == b[0] for a, b in zip(edges, edges[1:])))

    def test_unrepresentable_step_fails_instead_of_looping(self):
        with self.assertRaisesRegex(ValueError, "cannot advance"):
            aligned_edges(1e16, 1e16 + 4, .01, .01)


class RoundedCopcReader:
    """The integer-coordinate inclusion rule used by laspy.CopcReader.query."""
    def __init__(self, las):
        self.header = las.header
        self.points = las.points

    def query(self, bounds):
        mins = np.round((bounds.mins - self.header.offsets) / self.header.scales)
        maxs = np.round((bounds.maxs - self.header.offsets) / self.header.scales)
        keep = np.ones(len(self.points), dtype=bool)
        for axis, name in enumerate(("X", "Y", "Z")):
            values = getattr(self.points, name)
            keep &= (values >= mins[axis]) & (values <= maxs[axis])
        return self.points[keep]


class CopcWindowOwnershipTests(unittest.TestCase):
    def assert_partition(self, offsets):
        header = laspy.LasHeader(point_format=0, version="1.4")
        header.scales = np.array([.01, .01, .01])
        header.offsets = np.array(offsets)
        las = laspy.LasData(header)
        # Both sides, exact internal edges, corners, and inclusive outer edges.
        coords = np.array([(x, y, 0) for x in (0, 499, 500, 501, 999, 1000, 1001)
                           for y in (0, 499, 500, 501, 999, 1000, 1001)])
        las.X, las.Y, las.Z = coords.T
        las.intensity = np.arange(len(coords))
        las.update_header()
        reader = RoundedCopcReader(las)
        owned = [_query_copc_window(reader, b)
                 for b in iter_copc_center_of_mass_windows(las.header, .01)]
        ids = np.concatenate([p.intensity for p in owned])
        np.testing.assert_array_equal(np.sort(ids), np.arange(len(coords)))

    def test_exact_grid_boundaries_have_one_owner(self):
        self.assert_partition([0, 0, 0])

    def test_non_grid_offsets_and_negative_coordinates_have_one_owner(self):
        self.assert_partition([-5.003, 2.577, 0])
