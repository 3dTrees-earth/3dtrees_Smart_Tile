"""Single-scan tile centroids against an independent per-tile reference."""
import sys
from pathlib import Path

import laspy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tile_centroids import (RECORD, Tile, _reduce_block, block_cells, choose_block_size,  # noqa: E402
                            reduce_records, route, write_tile_centroids)

RESOLUTIONS = (0.05, 0.5)


def write_source(path, xyz, offsets):
    header = laspy.LasHeader(point_format=3, version="1.4")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.array(offsets, dtype=np.float64)
    las = laspy.LasData(header)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    las.write(path)
    return laspy.read(path)


def reference(sources, tile, resolution):
    """Mean of the encoded tile coordinates per voxel, as integer LAS output values."""
    xs = []
    for las in sources:
        x, y, z = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)
        xmin, ymin, xmax, ymax = tile.bounds
        keep = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
        ints = [np.round((v[keep] - tile.offsets[i]) / tile.scales[i]) for i, v in enumerate((x, y, z))]
        xs.append(np.stack(ints, axis=1))
    ints = np.concatenate(xs).astype(np.int64)
    coords = ints * np.array(tile.scales) + np.array(tile.offsets)
    keys = np.floor(coords / resolution).astype(np.int64)
    # Exact mean of the integer LAS coordinates per voxel, rounded half-to-even.
    from fractions import Fraction
    groups = {}
    for key, raw in zip(map(tuple, keys), ints.tolist()):
        groups.setdefault(key, []).append(raw)
    out = np.array([[round(Fraction(sum(p[axis] for p in points), len(points))) for axis in range(3)]
                    for points in groups.values()])
    return {tuple(row) for row in out.astype(np.int64)}, len(ints)


def encoded(path):
    las = laspy.read(path)
    return {tuple(row) for row in np.stack([las.X, las.Y, las.Z], axis=1).astype(np.int64)}, len(las.points)


def make_inputs(tmp_path):
    rng = np.random.default_rng(11)
    a = np.c_[rng.uniform(-6, 4, 6000), rng.uniform(-3, 7, 6000), rng.uniform(0, 3, 6000)]
    b = np.c_[rng.uniform(1, 9, 5000), rng.uniform(-2, 6, 5000), rng.uniform(0, 3, 5000)]
    # Points on tile edges and voxel faces; repeated coordinates across sources.
    edges = np.array([[0.0, 0.0, 1.0], [2.5, 1.0, 0.5], [-1.0, 2.5, 2.0], [0.05, 0.5, 0.0], [4.0, -0.5, 1.5]])
    a = np.r_[a, edges]
    b = np.r_[b, edges]
    sources = [tmp_path / "a.las", tmp_path / "b.las"]
    las = [write_source(sources[0], a, (0.0, 0.0, 0.0)), write_source(sources[1], b, (0.0, 0.0, 0.0))]
    # 2x2 tiles with a 0.5 m buffer around cores split at x=0 and y=2.5.
    tiles = []
    for col, (x0, x1) in enumerate(((-6.5, 0.5), (-0.5, 9.5))):
        for row, (y0, y1) in enumerate(((-3.5, 3.0), (2.0, 7.5))):
            tiles.append(Tile(f"c{col:02d}_r{row:02d}", (x0, y0, x1, y1), (0.001, 0.001, 0.001),
                              ((x0 + x1) / 2, (y0 + y1) / 2, 0.0)))
    return sources, las, tiles


def run(tmp_path, sources, las, tiles, workers, tag):
    outputs = [(r, {t.label: tmp_path / tag / f"{t.label}_{i}.laz" for t in tiles}, False)
               for i, r in enumerate(RESOLUTIONS)]
    written, occupied = write_tile_centroids(sources, tiles, las[0].header, outputs, workers=workers,
                                             work_dir=tmp_path / tag, metadata_source=sources[0])
    assert all(occupied)
    return outputs, written


def test_outputs_equal_independent_per_tile_means(tmp_path):
    sources, las, tiles = make_inputs(tmp_path)
    outputs, written = run(tmp_path, sources, las, tiles, 3, "w3")
    for index, (resolution, paths, _) in enumerate(outputs):
        for tile in tiles:
            expected, used = reference(las, tile, resolution)
            got, count = encoded(paths[tile.label])
            assert got == expected and count == len(expected) == written[tile.label][index]
            assert used > 0
    # Leftover scratch would leak disk on every Galaxy job.
    assert not list((tmp_path / "w3").glob(".tile-centroids-*"))


def test_worker_count_does_not_change_outputs(tmp_path, monkeypatch):
    import tile_centroids
    sources, las, tiles = make_inputs(tmp_path)
    one, _ = run(tmp_path, sources, las, tiles, 1, "w1")
    # Many small read ranges per source must not change any output.
    monkeypatch.setattr(tile_centroids, "CHUNK_POINTS", 700)
    four, _ = run(tmp_path, sources, las, tiles, 4, "w4")
    for (_, a, _), (_, b, _) in zip(one, four):
        for label in a:
            pa, pb = laspy.read(a[label]), laspy.read(b[label])
            assert np.array_equal(pa.points.array, pb.points.array)


def test_coarse_voxels_are_means_of_original_points_not_of_fine_centers():
    # 3 points in one fine voxel and 1 in another, same coarse voxel.
    records = np.zeros(4, dtype=RECORD)
    records["X"] = [10, 12, 14, 400]
    records["m"] = 3
    fine, coarse = reduce_records(records, (0.1, 1.0), (0.001, 0.001, 0.001), (0.0, 0.0, 0.0))
    assert len(fine[0]) == 2 and fine[1] == 4
    assert np.isclose(coarse[0][0, 0], 0.109)  # (10 + 12 + 14 + 400) / 4 = 109 exactly


def test_exact_rounding_is_half_to_even_and_order_independent():
    records = np.zeros(6, dtype=RECORD)
    # Three 1 m voxels whose means are exact halves: 2.5 -> 2, 1005.5 -> 1006, 2003.5 -> 2004.
    records["X"] = [2, 3, 1005, 1006, 2003, 2004]
    records["m"] = 1
    scales, offsets = (0.001, 0.001, 0.001), (0.0, 0.0, 0.0)
    centers, _ = reduce_records(records, (1.0,), scales, offsets)[0]
    assert sorted(np.round(centers[:, 0] / 0.001).astype(int)) == [2, 1006, 2004]
    shuffled, _ = reduce_records(records[::-1].copy(), (1.0,), scales, offsets)[0]
    assert np.array_equal(centers, shuffled)


def test_route_puts_each_point_once_per_resolution_in_its_voxels_block():
    rng = np.random.default_rng(5)
    # Multiples of the coarse resolution hit IEEE floor disagreements between grids.
    coords = np.c_[np.arange(1, 20001) * 0.1, rng.uniform(-50, 50, 20000), np.zeros(20000)]
    masks = np.full(len(coords), 3, dtype=np.uint8)
    resolutions, cells = (0.01, 0.1), (10, 1)
    rows, blocks, bits = route(coords, masks, resolutions, cells)
    assert (~np.isin(np.arange(len(coords)), rows[bits == 3])).any()  # some points split across blocks
    for index, (r, k) in enumerate(zip(resolutions, cells)):
        mine = (bits & (1 << index)) != 0
        assert np.array_equal(np.sort(rows[mine]), np.arange(len(coords)))
        expected = np.floor_divide(np.floor(coords[rows[mine], :2] / r).astype(np.int64), k)
        assert np.array_equal(blocks[mine], expected)


def test_oversized_block_split_matches_unsplit_reduction(tmp_path):
    rng = np.random.default_rng(9)
    records = np.zeros(30000, dtype=RECORD)
    records["X"] = rng.integers(0, 4000, len(records))
    records["Y"] = rng.integers(0, 4000, len(records))
    records["Z"] = rng.integers(0, 500, len(records))
    records["m"] = 3
    block = tmp_path / "0_0"
    block.mkdir()
    records[:12000].tofile(block / "000000.bin")
    records[12000:].tofile(block / "000001.bin")
    scales, offsets = (0.001, 0.001, 0.001), (0.0, 0.0, 0.0)
    whole = reduce_records(records, RESOLUTIONS, scales, offsets)
    split = _reduce_block((block, RESOLUTIONS, (80, 8), scales, offsets, 5000))
    for (a, used_a), (b, used_b) in zip(whole, split):
        assert used_a == used_b == len(records)
        assert {tuple(r) for r in np.round(a, 9)} == {tuple(r) for r in np.round(b, 9)}
    assert not (block / "split").exists()


def test_tiles_with_empty_cores_are_dropped_in_the_same_read(tmp_path):
    sources, las, tiles = make_inputs(tmp_path)
    # Core outside the data, buffer overlapping it: buffer points but no core points.
    empty = Tile("c09_r09", (8.0, 6.0, 12.0, 10.0), (0.001, 0.001, 0.001), (10.0, 8.0, 0.0),
                 core=(10.5, 8.5, 11.5, 9.5))
    kept = Tile(tiles[0].label, tiles[0].bounds, tiles[0].scales, tiles[0].offsets,
                core=(-6.0, -3.0, 0.0, 2.5))
    outputs = [(r, {t.label: tmp_path / "o" / f"{t.label}_{i}.laz" for t in (kept, empty)}, False)
               for i, r in enumerate(RESOLUTIONS)]
    written, occupied = write_tile_centroids(sources, [kept, empty], las[0].header, outputs, workers=2,
                                             work_dir=tmp_path / "o", metadata_source=sources[0])
    assert occupied == [True, False] and list(written) == [kept.label]
    assert not outputs[0][1][empty.label].exists()


def test_copc_output_from_worker_fragments_holds_the_same_points(tmp_path):
    sources, las, tiles = make_inputs(tmp_path)
    tiles = tiles[:2]
    outputs = [(RESOLUTIONS[0], {t.label: tmp_path / "c" / f"{t.label}.copc.laz" for t in tiles}, True),
               (RESOLUTIONS[0], {t.label: tmp_path / "c" / f"{t.label}.laz" for t in tiles}, False)]
    written, _ = write_tile_centroids(sources, tiles, las[0].header, outputs, workers=2,
                                      work_dir=tmp_path / "c", metadata_source=sources[0])
    for tile in tiles:
        copc, count = encoded(outputs[0][1][tile.label])
        laz, _ = encoded(outputs[1][1][tile.label])
        assert copc == laz and count == written[tile.label][0]
        with laspy.open(outputs[0][1][tile.label]) as reader:
            assert any(vlr.user_id == "copc" for vlr in reader.header.vlrs)


def test_block_edges_coincide_for_nested_resolutions():
    for points, area in ((329_792_556, 202_654.8), (3_682_068_346, 115_000.0), (9_026_333, 40_000.0)):
        size = choose_block_size(points, area, (0.01, 0.1))
        cells = block_cells((0.01, 0.1), size)
        assert cells[0] == 10 * cells[1], (size, cells)  # no point is split across blocks


def test_sampled_density_shrinks_blocks_for_dense_hot_spots():
    rng = np.random.default_rng(2)
    # Wide outlier extent, dense core: the mean density would give 50 m blocks.
    sample = np.r_[rng.uniform(0, 5, (19_000, 2)), rng.uniform(-2000, 2000, (1_000, 2))]
    size = choose_block_size(400_000_000, 16_000_000, (0.01, 0.1), sample)
    assert size < 50 and block_cells((0.01, 0.1), size)[0] == 10 * block_cells((0.01, 0.1), size)[1]
    assert choose_block_size(400_000_000, 16_000_000, (0.01, 0.1)) == 50.0
