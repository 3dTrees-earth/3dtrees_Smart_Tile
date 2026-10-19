"""Fail any test that removes a SmartTile scratch folder with files still open.

Galaxy EU job folders are on NFS, where an open or mapped file cannot be
removed, so the folder removal fails with EBUSY. Local disks allow it.
"""
import os
import tempfile
from pathlib import Path

import pytest

PROC = Path("/proc/self")
SCRATCH_PREFIXES = (".", "smarttile")


def open_files(folder):
    """Files under `folder` this process still maps or holds open."""
    prefix = str(folder) + os.sep
    mapped = (line.split(maxsplit=5)[-1].strip() for line in (PROC / "maps").read_text().splitlines())
    held = []
    for fd in (PROC / "fd").iterdir():
        try:
            held.append(os.readlink(fd))
        except OSError:
            pass
    return sorted({path for path in [*mapped, *held] if path.startswith(prefix)})


class CheckedTemporaryDirectory(tempfile.TemporaryDirectory):
    def cleanup(self):
        if os.path.basename(self.name).startswith(SCRATCH_PREFIXES):
            still_open = open_files(self.name)
            if still_open:
                pytest.fail(f"scratch folder removed with open files: {still_open}")
        super().cleanup()


@pytest.fixture(autouse=True)
def scratch_folders_are_closed_before_removal(monkeypatch):
    if (PROC / "maps").exists():
        monkeypatch.setattr(tempfile, "TemporaryDirectory", CheckedTemporaryDirectory)
