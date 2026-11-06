"""Scratch folders on network job storage.

Galaxy EU job folders are on NFS. A file that something still holds is renamed
to a hidden ``.nfsXXXX`` file, and removing its folder fails with EBUSY or
ENOTEMPTY until the holder lets go. Scratch cleanup therefore retries, and a
folder that stays behind is a warning: it cannot change an output, and
``TemporaryDirectory`` would raise from its cleanup and replace the error that
actually stopped the task.
"""
from __future__ import annotations

from contextlib import contextmanager
import errno
import gc
from pathlib import Path
import shutil
import tempfile
import time
import traceback

BUSY = {errno.EBUSY, errno.ENOTEMPTY}
RETRY_SECONDS = (0.1, 0.5, 1, 2, 5, 10)


def remove_tree(path) -> bool:
    """Remove a scratch folder, retrying while network storage reports it busy.

    Never raises: returns False and warns when the folder stays behind.
    """
    for delay in (*RETRY_SECONDS, None):
        try:
            shutil.rmtree(path)
            return True
        except FileNotFoundError:
            return True
        except OSError as exc:
            if delay is None or exc.errno not in BUSY:
                print(f"  Warning: scratch folder not removed: {path}: {exc}", flush=True)
                return False
        gc.collect()  # a memory map in a reference cycle holds its file until collected
        time.sleep(delay)


@contextmanager
def scratch_directory(prefix: str, dir=None):
    """``tempfile.TemporaryDirectory`` whose cleanup cannot fail the task or hide its error."""
    path = tempfile.mkdtemp(prefix=prefix, dir=dir)
    try:
        yield path
    except Exception as exc:
        if not getattr(exc, "scratch_reported", False):
            exc.scratch_reported = True
            print(f"Error before scratch cleanup of {Path(path).name}:", flush=True)
            traceback.print_exc()
        raise
    finally:
        remove_tree(path)
