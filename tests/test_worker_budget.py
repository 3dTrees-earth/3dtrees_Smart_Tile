import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from worker_budget import NATIVE_THREAD_VARIABLES, available_cpu_count, requested_workers  # noqa: E402

SRC = Path(__file__).resolve().parents[1] / "src"


class WorkerBudgetTests(unittest.TestCase):
    def test_available_cpu_count_is_positive(self):
        self.assertGreaterEqual(available_cpu_count(), 1)

    def test_available_cpu_count_uses_os_cpu_count_with_one_cpu_floor(self):
        with mock.patch("worker_budget.os.cpu_count", return_value=12):
            self.assertEqual(available_cpu_count(), 12)
        with mock.patch("worker_budget.os.cpu_count", return_value=None):
            self.assertEqual(available_cpu_count(), 1)


    def test_requested_workers_reads_every_flag_spelling(self):
        self.assertEqual(requested_workers(["--task", "merge", "--workers", "16"]), 16)
        self.assertEqual(requested_workers(["--number-of-threads=3"]), 3)
        self.assertEqual(requested_workers(["--number_of_threads", "2"]), 2)
        for argv in ([], ["--workers"], ["--workers", "0"], ["--workers=many"]):
            with self.subTest(argv=argv):
                self.assertIsNone(requested_workers(argv))


class NativeThreadLimitTests(unittest.TestCase):
    """Native pools stay within --workers in the main and in spawned worker processes."""

    def run_child(self, code, env=None):
        clean = {k: v for k, v in os.environ.items()
                 if k not in (*NATIVE_THREAD_VARIABLES, "RAYON_NUM_THREADS")}
        clean.update(env or {})
        clean["PYTHONPATH"] = str(SRC)
        result = subprocess.run([sys.executable, "-c", code], env=clean, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout.splitlines()[-1])

    def caps(self, workers, worker_process, env=None):
        return self.run_child(
            "import json, os, worker_budget as w\n"
            f"w.limit_native_threads({workers}, worker_process={worker_process})\n"
            "print(json.dumps({n: os.environ[n] for n in (*w.NATIVE_THREAD_VARIABLES, 'RAYON_NUM_THREADS')}))",
            env)

    def test_main_process_gets_workers_lazrs_threads_and_one_blas_thread(self):
        caps = self.caps(6, False)
        self.assertEqual(caps.pop("RAYON_NUM_THREADS"), "6")
        self.assertEqual(set(caps.values()), {"1"})

    def test_worker_processes_get_one_thread_everywhere(self):
        self.assertEqual(set(self.caps(6, True).values()), {"1"})

    def test_smaller_environment_caps_stay(self):
        self.assertEqual(self.caps(8, False, {"RAYON_NUM_THREADS": "2"})["RAYON_NUM_THREADS"], "2")

    def test_spawned_run_module_caps_numpy_to_one_thread(self):
        # multiprocessing's spawn re-runs run.py as __mp_main__ before any task module loads.
        threads = self.run_child(
            "import json, os, runpy, sys\n"
            f"sys.argv = [{str(SRC / 'run.py')!r}, '--task', 'merge', '--workers', '4']\n"
            "runpy.run_path(sys.argv[0], run_name='__mp_main__')\n"
            "import numpy\n"
            "print(json.dumps([os.environ['RAYON_NUM_THREADS'], len(os.listdir('/proc/self/task'))]))")
        self.assertEqual(threads, ["1", 1])

if __name__ == "__main__":
    unittest.main()
