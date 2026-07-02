import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from worker_budget import kdtree_query_workers  # noqa: E402


class WorkerBudgetTests(unittest.TestCase):
    def test_kdtree_query_workers_share_total_budget_across_outer_workers(self):
        self.assertEqual(kdtree_query_workers(total_workers=10, outer_workers=1), 10)
        self.assertEqual(kdtree_query_workers(total_workers=10, outer_workers=2), 5)
        self.assertEqual(kdtree_query_workers(total_workers=10, outer_workers=8), 1)

    def test_kdtree_query_workers_never_return_zero(self):
        self.assertEqual(kdtree_query_workers(total_workers=0, outer_workers=10), 1)
        self.assertEqual(kdtree_query_workers(total_workers=10, outer_workers=0), 10)


if __name__ == "__main__":
    unittest.main()
