"""PointIndex.append_index: a failed append reports its own error and leaves no partial rows."""
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bounded_point_index import PointIndex  # noqa: E402


def test_failed_append_keeps_original_error_and_rolls_back(tmp_path):
    dims = {"id": np.uint32}
    with PointIndex(tmp_path / "good.sqlite", dims) as shard:
        shard.add(3, np.array([[1.0, 2.0, 3.0]]), {"id": np.array([7], dtype=np.uint32)}, np.array([0]))
        shard.flush()
    broken = tmp_path / "broken.sqlite"
    with sqlite3.connect(broken) as db:  # batches present, bounds missing: fails mid-append
        db.execute("CREATE TABLE batches (id INTEGER PRIMARY KEY, tile INTEGER, data BLOB)")
        db.execute("INSERT INTO batches VALUES (1, 5, x'00')")
    with PointIndex(tmp_path / "main.sqlite", dims) as index:
        index.add(0, np.array([[0.0, 0.0, 0.0]]), {"id": np.array([1], dtype=np.uint32)}, np.array([0]))
        index.flush()
        with pytest.raises(sqlite3.OperationalError, match="no such table: shard.bounds"):
            index.append_index(broken)
        assert index.db.execute("SELECT COUNT(*) FROM batches").fetchone()[0] == 1
        assert 5 not in index._cache_tiles
        index.append_index(tmp_path / "good.sqlite")  # shard was detached: the index still works
        index.flush()
        assert index.db.execute("SELECT COUNT(*) FROM batches").fetchone()[0] == 2
        assert index.nearest(np.array([[1.0, 2.0, 3.0]]), 0.01)[1]["id"][0] == 7
