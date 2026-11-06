"""Cleanup on network storage can neither fail a finished task nor hide its error."""
import errno
from pathlib import Path
import shutil
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import scratch
from scratch import remove_tree, scratch_directory


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr(scratch.time, "sleep", lambda seconds: None)


def busy_rmtree(monkeypatch, failures, code=errno.EBUSY):
    """rmtree that fails ``failures`` times, as NFS does while a .nfsXXXX file exists."""
    real, calls = shutil.rmtree, []

    def rmtree(path, *args, **kwargs):
        calls.append(path)
        if len(calls) <= failures:
            raise OSError(code, "Device or resource busy", str(path))
        real(path, *args, **kwargs)

    monkeypatch.setattr(scratch.shutil, "rmtree", rmtree)
    return calls


def test_retries_while_the_folder_is_busy(tmp_path, monkeypatch):
    folder = tmp_path / ".scratch"
    folder.mkdir()
    calls = busy_rmtree(monkeypatch, 2)
    assert remove_tree(folder)
    assert len(calls) == 3 and not folder.exists()


@pytest.mark.parametrize("code", [errno.EBUSY, errno.ENOTEMPTY])
def test_a_folder_that_stays_busy_is_only_a_warning(tmp_path, monkeypatch, capsys, code):
    folder = tmp_path / ".scratch"
    folder.mkdir()
    calls = busy_rmtree(monkeypatch, 100, code)
    assert not remove_tree(folder)
    assert len(calls) == len(scratch.RETRY_SECONDS) + 1
    assert "Warning: scratch folder not removed" in capsys.readouterr().out


def test_other_errors_are_not_retried(tmp_path, monkeypatch):
    folder = tmp_path / ".scratch"
    folder.mkdir()
    calls = busy_rmtree(monkeypatch, 100, errno.EACCES)
    assert not remove_tree(folder)
    assert len(calls) == 1


def test_a_missing_folder_is_removed(tmp_path):
    assert remove_tree(tmp_path / "gone")


def test_cleanup_failure_after_success_does_not_fail_the_task(tmp_path, monkeypatch):
    busy_rmtree(monkeypatch, 100)
    with scratch_directory(".scratch-", dir=tmp_path) as folder:
        assert Path(folder).is_dir()


def test_cleanup_failure_does_not_replace_the_first_error(tmp_path, monkeypatch, capfd):
    busy_rmtree(monkeypatch, 100)
    with pytest.raises(ValueError, match="the real error"):
        with scratch_directory(".scratch-", dir=tmp_path):
            raise ValueError("the real error")
    captured = capfd.readouterr()
    assert "ValueError: the real error" in captured.err
    first_error = captured.out.index("Error before scratch cleanup")
    assert first_error < captured.out.index("Warning: scratch folder not removed")


def test_nested_scratch_reports_an_error_once(tmp_path, capfd):
    with pytest.raises(ValueError):
        with scratch_directory(".outer-", dir=tmp_path) as outer:
            with scratch_directory(".inner-", dir=outer):
                raise ValueError("once")
    assert capfd.readouterr().out.count("Error before scratch cleanup") == 1


def test_exit_requests_pass_silently(tmp_path, capfd):
    with pytest.raises(SystemExit):
        with scratch_directory(".scratch-", dir=tmp_path) as folder:
            raise SystemExit(1)
    assert "Error before" not in capfd.readouterr().out
    assert not Path(folder).exists()
