"""Block-wise nearest answers equal PointIndex.nearest per query batch, ties included."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from block_nearest import NearestResults, NearestSpill  # noqa: E402
from bounded_point_index import PointIndex  # noqa: E402


def test_block_nearest_equals_nearest_per_batch(tmp_path):
    rng = np.random.default_rng(2804)
    dims = {'id': np.uint32, 'cls': np.uint8}
    path = tmp_path / 'source.db'
    with PointIndex(path, dims) as index:
        grid = np.stack(np.meshgrid(np.arange(-2, 42, .1), np.arange(-2, 22, .1), [0.0, .1], indexing='ij'), -1).reshape(-1, 3)
        source = np.concatenate([grid, grid[:500], rng.uniform([-2, -2, 0], [42, 22, .1], (3000, 3))])
        ids = rng.integers(0, 50, len(source)).astype(np.uint32)
        cls = rng.integers(0, 4, len(source)).astype(np.uint8)
        for start in range(0, len(source), 4000):
            stop = start + 4000
            index.add(0, source[start:stop], {'id': ids[start:stop], 'cls': cls[start:stop]}, np.arange(start, stop))
        index.flush()
        # Targets: halfway between grid points (exact ties), on duplicates, random, and out of reach.
        batches = [grid[:6000] + [.05, 0, 0], grid[6000:12000] + [.05, .05, 0],
                   rng.uniform([-3, -3, 0], [43, 23, .1], (8000, 3)), grid[:500] + [0, 0, .25]]
        expected = [index.nearest(batch, .1732) for batch in batches]
    for workers in (1, 3):
        spill = NearestSpill(tmp_path / f'points{workers}')
        for batch in batches:
            spill.add(batch, .1732)
        answers = NearestResults(tmp_path / f'answers{workers}', dims, spill.rows)
        spill.answer(path, dims, .1732, answers, workers=workers)
        matched, values = answers.open()
        offset = 0
        for batch, (distances, want, _) in zip(batches, expected):
            rows = slice(offset, offset + len(batch))
            np.testing.assert_array_equal(matched[rows], np.isfinite(distances))
            for name in dims:
                np.testing.assert_array_equal(values[name][rows], want[name])
            offset += len(batch)
        assert 0 < int(matched[:].sum()) < spill.rows
        answers.remove()


def test_dense_transfer_blockwise_equals_per_group(tmp_path):
    """The block-wise and the per-group transfer write the same tiles and index rows."""
    from unittest import mock
    import laspy
    import merge_stages
    from test_parallel_merge_stages import index_rows, transfer
    runs = []
    for density in (0.0, 1e12):   # every tile block-wise / none
        root = tmp_path / f"d{int(density > 0)}"
        root.mkdir()
        with mock.patch.object(merge_stages, "BLOCKWISE_MIN_DENSITY", density):
            files, counts, report, error = transfer(root, 3)
        assert error is None
        runs.append((files, counts, report["parallelism"]["dense_blockwise_tiles"], root))
    (f0, c0, b0, r0), (f1, c1, b1, r1) = runs
    assert b0 == [0, 1, 2] and b1 == []
    assert c0 == c1
    for a, b in zip(f0, f1):
        np.testing.assert_array_equal(laspy.read(a).points.array, laspy.read(b).points.array)
    assert index_rows(r0 / "dense.sqlite") == index_rows(r1 / "dense.sqlite")


def test_block_nearest_equals_nearest_across_tiles(tmp_path):
    """Several source tiles, before_tile, tile and overlap filters, exact cross-tile ties."""
    rng = np.random.default_rng(2805)
    dims = {'id': np.uint32}
    path = tmp_path / 'multi.db'
    with PointIndex(path, dims) as index:
        grid = np.stack(np.meshgrid(np.arange(0, 30, .1), np.arange(0, 12, .1), [0.0], indexing='ij'), -1).reshape(-1, 3)
        for tile in range(4):
            # Every tile holds the same grid (exact cross-tile ties) plus its own random points.
            pts = np.concatenate([grid, rng.uniform([0, 0, 0], [30, 12, .2], (2000, 3))])
            ids = rng.integers(1, 9, len(pts)).astype(np.uint32)
            for start in range(0, len(pts), 3000):
                stop = start + 3000
                index.add(tile, pts[start:stop], {'id': ids[start:stop]}, np.arange(start, stop))
        index.flush()
        overlaps = {0: (np.array([0., 0.]), np.array([12., 12.])), 1: (np.array([8., 0.]), np.array([22., 12.])),
                    2: (np.array([18., 0.]), np.array([30., 12.]))}
        batches = [grid[:4000] + [.05, 0, 0], grid[4000:] + rng.uniform(-.004, .004, (len(grid) - 4000, 3)),
                   rng.uniform([0, 0, 0], [30, 12, .2], (6000, 3))]
        cases = [dict(before_tile=3), dict(before_tile=3, overlaps=overlaps), dict(tile=1, overlaps=overlaps),
                 dict(before_tile=2)]
        expected = [[index.nearest(batch, .01, **case) for batch in batches] for case in cases]
    for case, want in zip(cases, expected):
        spill = NearestSpill(tmp_path / 'p')
        for batch in batches:
            spill.add(batch, .01)
        answers = NearestResults(tmp_path / 'a', dims, spill.rows)
        spill.answer(path, dims, .01, answers, workers=2, **case)
        matched, values = answers.open()
        offset = 0
        for batch, (distances, found, _) in zip(batches, want):
            rows = slice(offset, offset + len(batch))
            np.testing.assert_array_equal(matched[rows], np.isfinite(distances), err_msg=str(case))
            np.testing.assert_array_equal(values['id'][rows], found['id'], err_msg=str(case))
            offset += len(batch)
        answers.remove()
