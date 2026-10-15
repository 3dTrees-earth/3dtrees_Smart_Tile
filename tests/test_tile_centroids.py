"""Single-scan tile centroids against an independent per-tile reference."""
from fractions import Fraction
import sys
from pathlib import Path

import laspy
import numpy as np
import pytest

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


def voxel_reference(raw, scales, offsets, resolution):
    """Exact decimal LAS coordinates, independent of the production grid helper."""
    step = Fraction(str(resolution))
    scales = [Fraction(str(value)) for value in scales]
    offsets = [Fraction(str(value)) for value in offsets]
    return np.asarray([[(int(row[axis]) * scales[axis] + offsets[axis]) // step
                        for axis in range(3)] for row in raw], dtype=np.int64)


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
    keys = voxel_reference(ints, tile.scales, tile.offsets, resolution)
    # Exact mean of the integer LAS coordinates per voxel, rounded half-to-even.
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
    outputs = [(r, {t.label: tmp_path / tag / f"{t.label}_{i}.laz" for t in tiles}, "laz")
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


def nearest_reference(sources, tile, resolution):
    """Per voxel, the encoded source point closest to the voxel mean (ties: lowest X, Y, Z)."""
    xs = []
    for las in sources:
        x, y, z = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)
        xmin, ymin, xmax, ymax = tile.bounds
        keep = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
        xs.append(np.stack([np.round((v[keep] - tile.offsets[i]) / tile.scales[i])
                            for i, v in enumerate((x, y, z))], axis=1))
    ints = np.concatenate(xs).astype(np.int64)
    keys = voxel_reference(ints, tile.scales, tile.offsets, resolution)
    groups = {}
    for key, raw in zip(map(tuple, keys), ints.tolist()):
        groups.setdefault(key, []).append(raw)
    chosen = set()
    for points in groups.values():
        mean = np.mean(np.array(points, dtype=float), axis=0)
        chosen.add(min(map(tuple, points), key=lambda p: (
            float(np.sum(((np.array(p) - mean) * np.array(tile.scales)) ** 2)), p)))
    return chosen


def test_nearest_to_centroid_keeps_the_original_point_nearest_each_voxel_mean(tmp_path, monkeypatch):
    import tile_centroids
    sources, las, tiles = make_inputs(tmp_path)
    runs = []
    for workers, chunk in ((1, tile_centroids.CHUNK_POINTS), (4, 700)):
        monkeypatch.setattr(tile_centroids, "CHUNK_POINTS", chunk)
        outputs = [(r, {t.label: tmp_path / f"n{workers}" / f"{t.label}_{i}.laz" for t in tiles}, "laz")
                   for i, r in enumerate(RESOLUTIONS)]
        written, _ = write_tile_centroids(sources, tiles, las[0].header, outputs, workers=workers,
                                          work_dir=tmp_path / f"n{workers}", metadata_source=sources[0],
                                          nearest=True)
        runs.append(outputs)
    for index, (resolution, paths, _) in enumerate(runs[0]):
        for tile in tiles:
            got, count = encoded(paths[tile.label])
            assert got == nearest_reference(las, tile, resolution) and count == len(got)
            same = laspy.read(runs[1][index][1][tile.label])
            assert np.array_equal(laspy.read(paths[tile.label]).points.array, same.points.array)


def test_nearest_to_centroid_ties_and_order_independence():
    scales, offsets = (0.001, 0.001, 0.001), (0.0, 0.0, 0.0)

    def nearest_x(xs):
        records = np.zeros(len(xs), dtype=RECORD)
        records["X"], records["m"] = xs, 1
        (centers, _), = reduce_records(records, (1.0,), scales, offsets, nearest=True)
        (shuffled, _), = reduce_records(records[::-1].copy(), (1.0,), scales, offsets, nearest=True)
        assert np.array_equal(centers, shuffled)
        return int(np.round(centers[0, 0] / 0.001))

    assert nearest_x([400, 600, 499, 501]) == 499  # mean 500: 499 and 501 tie, lowest X wins
    assert nearest_x([400, 600, 500, 900]) == 600  # mean 600 is a point
    assert nearest_x([0, 10, 990]) == 10  # mean 333.3: a real point, not the mean


@pytest.mark.parametrize("resolutions,cells", [((0.01, 0.1), (10, 1)), ((0.03, 0.1), (10, 1))])
def test_route_puts_each_point_once_per_resolution_in_its_voxels_block(resolutions, cells):
    rng = np.random.default_rng(5)
    # Exact centimetre faces must agree across nested grids; different block
    # widths can still require one routed copy per resolution.
    raw = np.c_[np.arange(1, 2001) * 100, rng.integers(-50000, 50000, 2000), np.zeros(2000, dtype=int)]
    scales, offsets = (0.001,) * 3, (500000.0, 5400000.0, 0.0)
    masks = np.full(len(raw), 3, dtype=np.uint8)
    rows, blocks, bits = route(raw, masks, resolutions, cells, scales=scales, offsets=offsets)
    if resolutions == (0.01, 0.1):
        assert np.all(bits == 3)
    else:
        assert np.any(bits != 3)
    for index, (resolution, width) in enumerate(zip(resolutions, cells)):
        mine = (bits & (1 << index)) != 0
        assert np.array_equal(np.sort(rows[mine]), np.arange(len(raw)))
        expected = voxel_reference(raw[rows[mine]], scales, offsets, resolution)[:, :2] // width
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
    whole = reduce_records(records, RESOLUTIONS, scales, offsets, nearest=True)
    split = _reduce_block((block, RESOLUTIONS, (80, 8), scales, offsets, 5000, True))
    for (a, _), (b, _) in zip(whole, split):
        assert {tuple(r) for r in np.round(a, 9)} == {tuple(r) for r in np.round(b, 9)}


def test_tiles_with_empty_cores_are_dropped_in_the_same_read(tmp_path):
    sources, las, tiles = make_inputs(tmp_path)
    # Core outside the data, buffer overlapping it: buffer points but no core points.
    empty = Tile("c09_r09", (8.0, 6.0, 12.0, 10.0), (0.001, 0.001, 0.001), (10.0, 8.0, 0.0),
                 core=(10.5, 8.5, 11.5, 9.5))
    kept = Tile(tiles[0].label, tiles[0].bounds, tiles[0].scales, tiles[0].offsets,
                core=(-6.0, -3.0, 0.0, 2.5))
    outputs = [(r, {t.label: tmp_path / "o" / f"{t.label}_{i}.laz" for t in (kept, empty)}, "laz")
               for i, r in enumerate(RESOLUTIONS)]
    written, occupied = write_tile_centroids(sources, [kept, empty], las[0].header, outputs, workers=2,
                                             work_dir=tmp_path / "o", metadata_source=sources[0])
    assert occupied == [True, False] and list(written) == [kept.label]
    assert not outputs[0][1][empty.label].exists()


def test_copc_output_from_worker_fragments_holds_the_same_points(tmp_path):
    sources, las, tiles = make_inputs(tmp_path)
    tiles = tiles[:2]
    outputs = [(RESOLUTIONS[0], {t.label: tmp_path / "c" / f"{t.label}.copc.laz" for t in tiles}, "copc.laz"),
               (RESOLUTIONS[0], {t.label: tmp_path / "c" / f"{t.label}.laz" for t in tiles}, "laz")]
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


def read_ply_xyz(path):
    with open(path, "rb") as handle:
        header = []
        while (line := handle.readline().decode("ascii").rstrip("\n")) != "end_header":
            header.append(line)
        return header, np.frombuffer(handle.read(), dtype="<f8").reshape(-1, 3)


def test_third_resolution_and_ply_come_from_the_same_read(tmp_path):
    sources, las, tiles = make_inputs(tmp_path)
    tiles = tiles[:1]
    resolutions = (0.05, 0.5, 0.25)
    outputs = [(0.05, {tiles[0].label: tmp_path / "x" / "a.laz"}, "laz"),
               (0.5, {tiles[0].label: tmp_path / "x" / "b.laz"}, "laz"),
               (0.25, {tiles[0].label: tmp_path / "x" / "c.laz"}, "laz"),
               (0.25, {tiles[0].label: tmp_path / "x" / "c.ply"}, "ply")]
    written, _ = write_tile_centroids(sources, tiles, las[0].header, outputs, workers=2,
                                      work_dir=tmp_path / "x", metadata_source=sources[0])
    for resolution, (_, paths, _) in zip(resolutions, outputs[:3]):
        expected, _ = reference(las, tiles[0], resolution)
        assert encoded(paths[tiles[0].label])[0] == expected
    header, xyz = read_ply_xyz(outputs[3][1][tiles[0].label])
    assert header[1] == "format binary_little_endian 1.0" and f"element vertex {len(xyz)}" in header
    third = laspy.read(outputs[2][1][tiles[0].label])
    assert np.array_equal(xyz, np.c_[third.x, third.y, third.z]) and written[tiles[0].label][3] == len(xyz)


def test_tile_output_selector_parses_pairs_and_rejects_bad_ones():
    import pytest
    from product_formats import parse_tile_outputs
    assert parse_tile_outputs("0.25:laz, 0.05:copc,0.5:ply") == [(0.25, "laz"), (0.05, "copc.laz"), (0.5, "ply")]
    assert parse_tile_outputs(None) == [] and parse_tile_outputs("") == []
    for bad in ("0.25", "0.25:las", "-1:laz", "x:laz", "0.25:laz,0.25:laz", "0.25:laz,0.25:ply"):
        with pytest.raises(ValueError):
            parse_tile_outputs(bad)


def test_tiles_use_the_finest_source_scale(tmp_path):
    import run
    from parameters import Parameters
    inputs = tmp_path / "in"
    inputs.mkdir()
    rng = np.random.default_rng(3)
    for name, scale in (("coarse.las", 0.01), ("fine.las", 0.0001)):
        header = laspy.LasHeader(point_format=3, version="1.4")
        header.scales, header.offsets = np.full(3, scale), np.zeros(3)
        las = laspy.LasData(header)
        xyz = np.c_[rng.uniform(0, 20, 500), rng.uniform(0, 20, 500), rng.uniform(0, 3, 500)]
        las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        las.write(inputs / name)
    fine = laspy.read(inputs / "fine.las")
    params = Parameters(task="tile", input_dir=inputs, output_dir=tmp_path / "out", tile_length=50,
                        tile_buffer=5, tile_outputs="0.0001:laz", workers=2, _cli_parse_args=False)
    run.run_tile_task(params)
    (output,) = (tmp_path / "out/subsampled_res1").glob("*.laz")
    tile = laspy.read(output)
    assert np.allclose(tile.header.scales, 0.0001)
    # Every fine-source point survives unchanged at a 0.1 mm voxel size.
    got = {tuple(np.round(p, 4)) for p in np.c_[tile.x, tile.y, tile.z]}
    assert {tuple(np.round(p, 4)) for p in np.c_[fine.x, fine.y, fine.z]} <= got


def test_tile_publication_failure_rolls_back_every_new_output(tmp_path, monkeypatch):
    import os
    import pytest
    sources, las, tiles = make_inputs(tmp_path)
    tiles = tiles[:1]
    a, b = tmp_path / "out/a.laz", tmp_path / "out/b.laz"
    outputs = [(.05, {tiles[0].label: a}, "laz"), (.5, {tiles[0].label: b}, "laz")]
    link = os.link

    def fail_second(source, destination):
        if Path(destination) == b:
            raise OSError("injected late tile publication failure")
        return link(source, destination)

    original_bytes = [source.read_bytes() for source in sources]
    monkeypatch.setattr("output_publication.os.link", fail_second)
    with pytest.raises(OSError, match="late tile publication"):
        write_tile_centroids(sources, tiles, las[0].header, outputs, workers=1,
                             work_dir=tmp_path / "out", metadata_source=sources[0])
    assert not a.exists() and not b.exists()
    assert [source.read_bytes() for source in sources] == original_bytes
    assert not list((tmp_path / "out").glob(".tile-centroids-*"))


def test_tile_existing_output_is_preserved_before_work(tmp_path, monkeypatch):
    import pytest
    sources, las, tiles = make_inputs(tmp_path)
    output = tmp_path / "result.laz"
    output.write_bytes(b"previous result")

    def unexpected(*args, **kwargs):
        raise AssertionError("point processing started before destination preflight")

    monkeypatch.setattr("tile_centroids._range_tasks", unexpected)
    with pytest.raises(FileExistsError, match="existing output"):
        write_tile_centroids(sources, tiles[:1], las[0].header,
                             [(.05, {tiles[0].label: output}, "laz")], workers=1,
                             work_dir=tmp_path / "work", metadata_source=sources[0])
    assert output.read_bytes() == b"previous result"
    assert not (tmp_path / "work").exists()


@pytest.mark.parametrize("output_format", ("laz", "copc.laz", "ply"))
def test_original_and_subsampled_tiles_preserve_duplicates_and_buffer_points(tmp_path, output_format):
    from collections import Counter

    xyz = np.array([[0.01, 0.01, 1.0], [0.01, 0.01, 1.0], [0.02, 0.01, 1.0],
                    [0.99, 0.01, 1.0], [1.01, 0.01, 1.0], [1.5, 0.01, 1.0]])
    sources = [tmp_path / "a.las", tmp_path / "b.las"]
    inputs = [write_source(sources[0], xyz[:4], (0, 0, 0)),
              write_source(sources[1], xyz[4:], (0, 0, 0))]
    tiles = [Tile("left", (-0.1, -0.1, 1.1, 1.1), (0.001,) * 3, (0.5, 0.5, 0)),
             Tile("right", (0.9, -0.1, 2.1, 1.1), (0.001,) * 3, (1.5, 0.5, 0))]
    outputs = [(0.0, {t.label: tmp_path / "out" / f"{t.label}.{output_format}" for t in tiles}, output_format),
               (0.1, {t.label: tmp_path / "out" / f"{t.label}_10cm.laz" for t in tiles}, "laz")]
    counts, occupied = write_tile_centroids(sources, tiles, inputs[0].header, outputs, workers=2,
                                            work_dir=tmp_path / "out", metadata_source=sources[0])
    assert occupied == [True, True]
    for tile in tiles:
        path = outputs[0][1][tile.label]
        if output_format == "ply":
            _, actual = read_ply_xyz(path)
        else:
            cloud = laspy.read(path)
            actual = np.c_[cloud.x, cloud.y, cloud.z]
        xmin, ymin, xmax, ymax = tile.bounds
        expected = xyz[(xyz[:, 0] >= xmin) & (xyz[:, 0] <= xmax) &
                       (xyz[:, 1] >= ymin) & (xyz[:, 1] <= ymax)]
        assert Counter(map(tuple, np.round(actual, 6))) == Counter(map(tuple, expected))
        assert len(actual) == counts[tile.label][0] == len(expected)
        expected_reduced, _ = reference(inputs, tile, 0.1)
        assert encoded(outputs[1][1][tile.label])[0] == expected_reduced


def test_original_only_dense_block_streams_without_voxel_reduction(tmp_path, monkeypatch):
    import tile_centroids

    block, destination = tmp_path / "0_0", tmp_path / "original"
    block.mkdir()
    destination.mkdir()
    records = np.zeros(25, dtype=RECORD)
    records["X"], records["m"] = 123, 1
    records.tofile(block / "000000.bin")

    def unexpected(*args, **kwargs):
        raise AssertionError("original points must bypass voxel reduction")

    monkeypatch.setattr(tile_centroids, "_reduce_block", unexpected)
    counts = tile_centroids._reduce_and_write((block, (0.0,), (50,), (0.001,) * 3,
                                               (0.0,) * 3, 7, [destination], "fragment", False))
    dtype = laspy.PointFormat(tile_centroids.FRAGMENT_POINT_FORMAT).dtype()
    batches = [np.fromfile(path, dtype=dtype) for path in sorted(destination.glob("*.bin"))]
    assert counts == [(25, 25)]
    assert [len(batch) for batch in batches] == [7, 7, 7, 4]
    assert np.array_equal(np.concatenate(batches)["X"], records["X"])
    assert not block.exists()


def test_original_routing_and_split_keep_every_point_once_per_output(tmp_path):
    rng = np.random.default_rng(7)
    records = np.zeros(101, dtype=RECORD)
    records["X"] = rng.integers(-100, 150, len(records))
    records["Y"] = rng.integers(-100, 150, len(records))
    records["m"] = 3
    scales, offsets, resolutions = (0.001,) * 3, (0.0,) * 3, (0.0, 0.1)
    coords = np.c_[records["X"], records["Y"], records["Z"]] * scales
    cells = block_cells(resolutions, choose_block_size(len(records), 1, resolutions))
    rows, blocks, bits = route(np.c_[records["X"], records["Y"], records["Z"]], records["m"],
                               resolutions, cells, scales=scales, offsets=offsets)
    for slot in range(2):
        assert np.array_equal(np.sort(rows[(bits & (1 << slot)) != 0]), np.arange(len(records)))
    block = tmp_path / "0_0"
    block.mkdir()
    records.tofile(block / "000000.bin")
    originals, reduced = _reduce_block((block, resolutions, cells, scales, offsets, 7))
    from collections import Counter
    assert Counter(map(tuple, originals[0])) == Counter(map(tuple, coords))
    assert originals[1] == reduced[1] == len(records)
    expected = reduce_records(records, resolutions, scales, offsets)[1][0]
    assert set(map(tuple, reduced[0])) == set(map(tuple, expected))


def test_original_output_selector_and_filename():
    from product_formats import parse_tile_outputs
    from output_names import resolution_label, subsample_output_name

    assert parse_tile_outputs("original:laz,0.1:copc.laz") == [(0.0, "laz"), (0.1, "copc.laz")]
    assert parse_tile_outputs("ORIGINAL:ply") == [(0.0, "ply")]
    assert resolution_label(0) == "original"
    assert subsample_output_name(Path("c00_r00.laz"), 0, None, False) == "c00_r00_subsampled_original.laz"
    assert subsample_output_name(Path("tree_subsampled_original.laz"), 0.1, None, False) == "tree_subsampled_10cm.laz"
    for selector in ("0:laz", "original:laz,original:ply", "original:las", "nan:laz", "inf:laz"):
        with pytest.raises(ValueError):
            parse_tile_outputs(selector)


def test_original_spatial_tiles_align_encoding_to_the_source_grid(tmp_path):
    header = laspy.LasHeader(point_format=3, version="1.4")
    header.scales, header.offsets = np.full(3, 0.001), np.full(3, 0.0005)
    source = laspy.LasData(header)
    source.X, source.Y, source.Z = [0, 1, 999, 1000], [1, 2, 3, 4], [3, 4, 5, 6]
    source_path = tmp_path / "source.las"
    source.write(source_path)
    # The tile centre lies halfway between points on the source's LAS grid.
    tile = Tile("c00_r00", (-1, -1, 2, 2), (0.001,) * 3, (0.5,) * 3)
    target = tmp_path / "out/original.laz"
    counts, _ = write_tile_centroids([source_path], [tile], header,
                                     [(0.0, {tile.label: target}, "laz")], workers=1,
                                     work_dir=tmp_path / "out", metadata_source=source_path)
    actual = laspy.read(target)
    expected_xyz = np.c_[source.x, source.y, source.z]
    actual_xyz = np.c_[actual.x, actual.y, actual.z]
    expected_xyz = expected_xyz[np.argsort(expected_xyz[:, 0])]
    actual_xyz = actual_xyz[np.argsort(actual_xyz[:, 0])]
    np.testing.assert_allclose(actual_xyz, expected_xyz, rtol=0, atol=1e-12)
    assert counts[tile.label] == [4]
