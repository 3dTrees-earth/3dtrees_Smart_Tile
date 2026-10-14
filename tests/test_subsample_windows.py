import json
from pathlib import Path
import subprocess
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from subsample_com import aligned_edges


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
