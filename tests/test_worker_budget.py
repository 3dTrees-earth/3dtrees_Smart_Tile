import sys
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from worker_budget import DEFAULT_FILE_WORKERS, available_cpu_count, file_worker_count, kdtree_query_workers  # noqa: E402


class WorkerBudgetTests(unittest.TestCase):
    def test_kdtree_query_workers_share_total_budget_across_outer_workers(self):
        self.assertEqual(kdtree_query_workers(total_workers=10, outer_workers=1), 10)
        self.assertEqual(kdtree_query_workers(total_workers=10, outer_workers=2), 5)
        self.assertEqual(kdtree_query_workers(total_workers=10, outer_workers=8), 1)

    def test_kdtree_query_workers_never_return_zero(self):
        self.assertEqual(kdtree_query_workers(total_workers=0, outer_workers=10), 1)
        self.assertEqual(kdtree_query_workers(total_workers=10, outer_workers=0), 10)

    def test_file_worker_count_defaults_to_two_and_caps_to_items(self):
        self.assertEqual(file_worker_count(None), DEFAULT_FILE_WORKERS)
        self.assertEqual(file_worker_count(None, item_count=1), 1)
        self.assertEqual(file_worker_count(8, item_count=2), 2)

    def test_available_cpu_count_is_positive(self):
        self.assertGreaterEqual(available_cpu_count(), 1)

    def test_available_cpu_count_uses_os_cpu_count_with_one_cpu_floor(self):
        with mock.patch("worker_budget.os.cpu_count", return_value=12):
            self.assertEqual(available_cpu_count(), 12)
        with mock.patch("worker_budget.os.cpu_count", return_value=None):
            self.assertEqual(available_cpu_count(), 1)


if __name__ == "__main__":
    unittest.main()
