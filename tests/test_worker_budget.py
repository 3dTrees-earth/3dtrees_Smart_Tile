import sys
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from worker_budget import (  # noqa: E402
    DEFAULT_FILE_WORKERS,
    available_cpu_count,
)


class WorkerBudgetTests(unittest.TestCase):
    def test_available_cpu_count_is_positive(self):
        self.assertGreaterEqual(available_cpu_count(), 1)

    def test_available_cpu_count_uses_os_cpu_count_with_one_cpu_floor(self):
        with mock.patch("worker_budget.os.cpu_count", return_value=12):
            self.assertEqual(available_cpu_count(), 12)
        with mock.patch("worker_budget.os.cpu_count", return_value=None):
            self.assertEqual(available_cpu_count(), 1)


if __name__ == "__main__":
    unittest.main()
