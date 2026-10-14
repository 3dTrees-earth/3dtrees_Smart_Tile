"""Spatially ordered enrichment input: every original point once, block-compact."""
import sys
from pathlib import Path

import laspy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import original_blocks  # noqa: E402
from bounded_point_index import coordinates  # noqa: E402


def test_block_batches_cover_every_point_once_with_original_coordinates(tmp_path, monkeypatch):
    rng = np.random.default_rng(4)
    # Acquisition-order-like input: interleaved far-apart strips.
    xyz = np.c_[rng.uniform(0, 300, 200_000), rng.uniform(0, 200, 200_000), rng.uniform(0, 30, 200_000)]
    header = laspy.LasHeader(point_format=3, version="1.4")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.array([500000.0, 5000000.0, 0.0])
    las = laspy.LasData(header)
    las.x, las.y, las.z = xyz[:, 0] + 500000, xyz[:, 1] + 5000000, xyz[:, 2]
    path = tmp_path / "original.las"
    las.write(path)
    las = laspy.read(path)
    origin = np.array([500000.0, 5000000.0, 0.0])
    monkeypatch.setattr(original_blocks, "READ_POINTS", 30_000)  # several point ranges
    size = original_blocks.spill_original(path, origin, tmp_path / "blocks", workers=3)
    assert size % original_blocks.REGION_SIZE == 0
    seen, expected = [], coordinates(las.points.array, las.header, origin)
    for index, local in original_blocks.block_batches(tmp_path / "blocks", las.header, origin):
        assert len(np.unique(np.floor(local[:, :2] / size), axis=0)) == 1  # one block per batch
        np.testing.assert_array_equal(local, expected[index])
        seen.append(index)
    seen = np.concatenate(seen)
    np.testing.assert_array_equal(np.sort(seen), np.arange(len(las.points)))
    assert not (tmp_path / "blocks").exists() or not any((tmp_path / "blocks").iterdir())


def test_oversized_blocks_are_processed_region_by_region(tmp_path, monkeypatch):
    # A dense 10 x 10 m plot plus one stray point 1 km away: the header extent
    # makes the blocks large, so the plot lands in one oversized block.
    rng = np.random.default_rng(8)
    plot = np.c_[rng.uniform(0, 10, 50_000), rng.uniform(0, 10, 50_000), rng.uniform(0, 30, 50_000)]
    xyz = np.r_[plot, [[1000.0, 1000.0, 0.0]]]
    header = laspy.LasHeader(point_format=3, version="1.4")
    header.scales, header.offsets = np.full(3, 0.001), np.array([500000.0, 5000000.0, 0.0])
    las = laspy.LasData(header)
    las.x, las.y, las.z = xyz[:, 0] + 500000, xyz[:, 1] + 5000000, xyz[:, 2]
    path = tmp_path / "original.las"
    las.write(path)
    las = laspy.read(path)
    origin = np.array([500000.0, 5000000.0, 0.0])
    expected = coordinates(las.points.array, las.header, origin)
    batches = {}
    for cap in (None, 5_000):
        original_blocks.spill_original(path, origin, tmp_path / f"blocks{cap}", workers=2)
        seen = []
        for index, local in original_blocks.block_batches(tmp_path / f"blocks{cap}", las.header, origin, point_cap=cap):
            np.testing.assert_array_equal(local, expected[index])
            if cap:  # capped: every batch stays within one 2 m query cell, in index order
                assert len(np.unique(np.floor(local[:, :2] / original_blocks.CELL_SIZE), axis=0)) == 1
                assert np.all(np.diff(index) > 0)
            seen.append(index)
        batches[cap] = np.sort(np.concatenate(seen))
        assert not any((tmp_path / f"blocks{cap}").iterdir())
    np.testing.assert_array_equal(batches[None], np.arange(len(las.points)))
    np.testing.assert_array_equal(batches[5_000], batches[None])
