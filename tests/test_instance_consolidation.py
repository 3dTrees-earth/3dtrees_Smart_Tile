"""Every final instance ends up whole in exactly one tile."""
import sys
from pathlib import Path

import laspy
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from instance_consolidation import consolidate_instances, instance_tiles  # noqa: E402


def write_tile(path, xyz, ids, offset):
    header = laspy.LasHeader(point_format=3, version="1.4")
    header.add_extra_dim(laspy.ExtraBytesParams(name="PredInstance", type=np.uint32))
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.array(offset, dtype=np.float64)
    las = laspy.LasData(header)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    las.PredInstance = np.asarray(ids, dtype=np.uint32)
    las.intensity = np.arange(len(xyz), dtype=np.uint16)
    las.write(path)


def regions():
    # Two tiles split at x = 10, each with a 2 m buffer into the other.
    return [{"core": [[0.0, 10.0], [0.0, 10.0]]}, {"core": [[10.0, 20.0], [0.0, 10.0]]}]


def points(files):
    rows = []
    for f in files:
        las = laspy.read(f)
        rows += list(zip(np.round(las.x, 3), np.round(las.y, 3), np.round(las.z, 3),
                         las.PredInstance.tolist(), las.intensity.tolist()))
    return sorted(rows)


def test_split_instance_moves_whole_to_the_core_holding_its_centroid(tmp_path):
    a, b = tmp_path / "c00_r00.laz", tmp_path / "c01_r00.laz"
    # Instance 7: 7 points in tile a (x < 10), 3 points in tile b; centroid in a's core.
    write_tile(a, np.c_[np.linspace(6, 9.5, 7), np.full(7, 5.0), np.arange(7.0)], [7] * 7, (5.0, 5.0, 0.0))
    write_tile(b, np.c_[[10.5, 11.0, 11.5, 15.0, 16.0], np.full(5, 5.0), np.arange(5.0)],
               [7, 7, 7, 0, 9], (15.0, 5.0, 0.0))
    before = points([a, b])
    report = {}
    metric = consolidate_instances([a, b], "PredInstance", regions(), tmp_path / "work", report)
    assert metric["split_instances"] == 1 and metric["moved_points"] == 3
    assert points([a, b]) == before  # moved, never dropped or changed
    counts, _ = instance_tiles([a, b], "PredInstance")
    assert counts[7] == {0: 10} and counts[9] == {1: 1}
    assert (laspy.read(b).PredInstance == 0).sum() == 1  # background stays
    assert not list((tmp_path / "work").iterdir())


def test_owner_falls_back_to_most_points_and_is_deterministic(tmp_path):
    a, b = tmp_path / "a.laz", tmp_path / "b.laz"
    # Centroid at y = 12 lies outside both cores: the tile with most points (b) wins.
    write_tile(a, np.c_[[9.0, 9.5], [12.0, 12.0], [0.0, 1.0]], [4, 4], (5.0, 5.0, 0.0))
    write_tile(b, np.c_[[10.5, 11.0, 11.5], [12.0, 12.0, 12.0], [0.0, 1.0, 2.0]], [4, 4, 4], (15.0, 5.0, 0.0))
    consolidate_instances([a, b], "PredInstance", regions(), tmp_path / "work", {})
    counts, _ = instance_tiles([a, b], "PredInstance")
    assert counts[4] == {1: 5}
    assert len(laspy.read(a).points) == 0


def test_nothing_to_move_leaves_files_untouched(tmp_path):
    a, b = tmp_path / "a.laz", tmp_path / "b.laz"
    write_tile(a, np.c_[[1.0], [1.0], [0.0]], [1], (5.0, 5.0, 0.0))
    write_tile(b, np.c_[[15.0], [1.0], [0.0]], [2], (15.0, 5.0, 0.0))
    stamp = [f.stat().st_mtime_ns for f in (a, b)]
    metric = consolidate_instances([a, b], "PredInstance", regions(), tmp_path / "work", {})
    assert metric["split_instances"] == 0
    assert [f.stat().st_mtime_ns for f in (a, b)] == stamp
