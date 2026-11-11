"""Point-range readers on LAZ with variable-size chunks (COPC): every record once, the right one.

laspy/lazrs misplace ``seek`` in such files (point_ranges), so readers that
split a file into ranges must read it in one; on a fixed-chunk LAZ they split.
"""
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import laspy
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import original_blocks  # noqa: E402
from block_nearest import SPILL_DTYPE, spill_file  # noqa: E402
from bounded_point_index import coordinates  # noqa: E402
from point_ranges import seekable  # noqa: E402

ORIGIN = np.array([500000.0, 5000000.0, 0.0])
POINTS = 200_000


def cloud(path):
    rng = np.random.default_rng(4)
    header = laspy.LasHeader(point_format=3, version="1.4")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = ORIGIN.copy()
    las = laspy.LasData(header)
    las.x = rng.uniform(0, 300, POINTS) + ORIGIN[0]
    las.y = rng.uniform(0, 200, POINTS) + ORIGIN[1]
    las.z = rng.uniform(0, 30, POINTS)
    las.write(path)
    return path


@pytest.fixture
def copc(tmp_path):
    if shutil.which("pdal") is None:
        pytest.skip("pdal writes the COPC fixture")
    target = tmp_path / "cloud.copc.laz"
    subprocess.run(["pdal", "translate", str(cloud(tmp_path / "cloud.las")), str(target)], check=True)
    assert not seekable(target), "COPC chunks are variable-size"
    return target


def test_fixed_chunk_laz_is_seekable(tmp_path):
    assert seekable(cloud(tmp_path / "cloud.laz"))
    assert seekable(cloud(tmp_path / "cloud.las"))


def test_original_blocks_read_every_copc_point_once(copc, tmp_path, monkeypatch):
    las = laspy.read(copc)
    monkeypatch.setattr(original_blocks, "READ_POINTS", 30_000)  # several point ranges
    original_blocks.spill_original(copc, ORIGIN, tmp_path / "blocks", workers=3)
    expected, seen = coordinates(las.points.array, las.header, ORIGIN), []
    for index, local in original_blocks.block_batches(tmp_path / "blocks", las.header, ORIGIN):
        np.testing.assert_array_equal(local, expected[index])
        seen.append(index)
    np.testing.assert_array_equal(np.sort(np.concatenate(seen)), np.arange(POINTS))


def test_spill_parts_equal_one_pass_on_copc(copc, tmp_path):
    def records(processes):
        spill = spill_file(copc, ORIGIN, .1732, tmp_path / f"spill{processes}", processes=processes,
                           chunk_points=20_000)
        data = np.concatenate([np.fromfile(path, dtype=SPILL_DTYPE) for path in spill.cells.values()])
        return data[np.argsort(data["row"])]
    one = records(1)
    np.testing.assert_array_equal(records(4), one)
    with laspy.open(copc) as reader:
        np.testing.assert_array_equal(one["xyz"], coordinates(reader.read_points(POINTS), reader.header, ORIGIN))


def test_range_splitters_keep_a_copc_whole(copc, tmp_path):
    from instance_tiles import _ranges
    from merge_stages import _tile_parts
    from tile_centroids import _range_tasks, sample_xy
    fixed = cloud(tmp_path / "fixed.laz")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("merge_stages.MAX_BATCH_POINTS", 10_000)
        parts = _tile_parts([copc, fixed], 4)
    assert [part for tile, part in parts if tile == 0] == [(0, 20, 10_000)]
    assert len([tile for tile, _ in parts if tile == 1]) > 1
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("instance_tiles.CHUNK_POINTS", 10_000)
        ranges, _ = _ranges([copc, fixed], 4)
    assert [(start, count) for source, start, count in ranges if source == str(copc)] == [(0, POINTS)]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("tile_centroids.CHUNK_POINTS", 10_000)
        tile = SimpleNamespace(bounds=(ORIGIN[0], ORIGIN[1], ORIGIN[0] + 300, ORIGIN[1] + 200))
        tasks, _, _ = _range_tasks([copc], [tile], (1.0,), (1,), tmp_path / "spill", 4)
        assert [(task[2], task[3]) for task in tasks] == [(0, POINTS)]
        # The sample is the evenly spaced runs a working seek would read.
        sample = sample_xy([copc], points=60_000, run=7_000)
    las = laspy.read(copc)
    starts = np.linspace(0, POINTS - 7_000, 9).astype(np.int64)
    expected = np.concatenate([np.c_[np.asarray(las.x)[s:s + 7_000], np.asarray(las.y)[s:s + 7_000]] for s in starts])
    np.testing.assert_array_equal(sample, expected)


@pytest.mark.parametrize("chunk", [333, 4_096, 50_000])
def test_read_chunks_equal_chunk_iterator(tmp_path, monkeypatch, chunk):
    import bounded_point_index
    from bounded_point_index import read_chunks
    monkeypatch.setattr(bounded_point_index, "READ_POINTS", 40_000)
    path = cloud(tmp_path / "cloud.laz")
    with laspy.open(path) as a, laspy.open(path) as b:
        got, expected = list(read_chunks(a, chunk)), list(b.chunk_iterator(chunk))
    assert [len(r) for r in got] == [len(r) for r in expected]
    for left, right in zip(got, expected):
        np.testing.assert_array_equal(left.array, right.array)
        np.testing.assert_array_equal(left.x, right.x)
