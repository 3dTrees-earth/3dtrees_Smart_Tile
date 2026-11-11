import sys
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import worker_budget  # noqa: E402
from worker_budget import (OUT_OF_MEMORY_EXIT, MemoryBudgetError, failure_exit_code,  # noqa: E402
                           memory_capped_workers, memory_limit_bytes)

GB = 1024**3


@pytest.mark.parametrize("grant_gb, workers", [(2.0, 2), (3.06, 4), (3.55, 5), (6.0, 9), (20.0, 10)])
def test_tile_workers_fit_the_galaxy_grants_of_the_failed_datasets(grant_gb, workers):
    # 2548 (2 GB), Brandenburg 1129 (3.06 GB) and 1134 (3.55 GB) failed with 10 workers.
    assert memory_capped_workers(["tile"], 10, limit=int(grant_gb * GB)) == workers


def test_a_run_needs_the_largest_worker_share_of_its_tasks():
    remap = memory_capped_workers(["remap"], 10, limit=4 * GB)
    with_products = memory_capped_workers(["remap", "create_merged_file"], 10, limit=4 * GB)
    assert remap == 10 and with_products == 3


def test_no_limit_keeps_the_workers():
    assert memory_capped_workers(["tile"], 7, limit=None) in (7, memory_capped_workers(["tile"], 7))


def test_memory_is_named_when_one_worker_does_not_fit():
    with pytest.raises(MemoryBudgetError, match="memory"):
        memory_capped_workers(["tile"], 10, limit=GB // 2)


def test_galaxy_memory_is_a_limit(monkeypatch):
    monkeypatch.setenv("GALAXY_MEMORY_MB", "3072")
    assert memory_limit_bytes() <= 3 * GB


def test_out_of_memory_failures_exit_137(monkeypatch, capsys):
    monkeypatch.setattr(worker_budget, "oom_kill_count", lambda: 2)
    assert failure_exit_code(BrokenProcessPool("abrupt"), oom_kills_before=0) == OUT_OF_MEMORY_EXIT
    assert "out of memory" in capsys.readouterr().out
    assert failure_exit_code(MemoryError()) == OUT_OF_MEMORY_EXIT
    # A broken pool without a kernel OOM kill, and ordinary errors, stay exit 1.
    assert failure_exit_code(BrokenProcessPool("abrupt"), oom_kills_before=2) == 1
    assert failure_exit_code(ValueError("bad input")) == 1


@pytest.mark.parametrize("grant_gb, workers", [(2.0, 4), (4.0, 10)])
def test_merge_workers_fit_a_small_grant(grant_gb, workers):
    # 3753 merge at 10 workers: 3.8 GB; in a 2 GB container it was killed for memory, with 4 it passes (1.8 GB).
    assert memory_capped_workers(["merge"], 10, limit=int(grant_gb * GB)) == workers
