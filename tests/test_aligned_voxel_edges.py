"""Product chunk boundaries must retain points on the absolute voxel grid."""
import ast
from fractions import Fraction

import laspy
import numpy as np
import pytest

from main_create_merged_file import _prod_merged_chunk_bounds, select_voxel_points
from subsample_com import aligned_edges


def _chunk_rows(cloud, bounds, resolution):
    """Rows each chunk reads (PDAL bounds, inclusive) and keeps (its voxel columns)."""
    from voxel_grid import voxel_keys

    xyz = np.asarray(cloud.xyz)
    columns = voxel_keys(np.asarray(cloud.X)[:, None], cloud.header.scales[:1], cloud.header.offsets[:1],
                         resolution)[:, 0]
    for value, lo, hi in bounds:
        ((xmin, xmax), (ymin, ymax)) = ast.literal_eval(value)
        read = ((xyz[:, 0] >= xmin) & (xyz[:, 0] <= xmax) & (xyz[:, 1] >= ymin) & (xyz[:, 1] <= ymax))
        owned = read & (columns >= (lo if lo is not None else columns.min()))
        owned &= columns < (hi if hi is not None else columns.max() + 1)
        yield np.flatnonzero(owned)


def _write(path, scale, offset, x, y):
    header = laspy.LasHeader(point_format=0, version="1.4")
    header.scales = np.full(3, scale)
    header.offsets = np.asarray(offset, dtype=float)
    cloud = laspy.LasData(header)
    cloud.x, cloud.y, cloud.z = x, y, np.zeros(len(x))
    cloud.write(path)
    return laspy.read(path)


def test_product_chunk_reads_keep_boundary_points_and_match_whole_selection(tmp_path):
    values = np.arange(1001) * .001
    cloud = _write(tmp_path / "points.las", .001, (0, 0, 0), values, values)
    bounds = _prod_merged_chunk_bounds([tmp_path / "points.las"], .1, 10)
    coverage = np.zeros(len(cloud.points), dtype=np.int64)
    selected = []
    for rows in _chunk_rows(cloud, bounds, .1):
        coverage[rows] += 1
        selected.extend(rows[select_voxel_points(cloud.points[rows], .1)].tolist())
    np.testing.assert_array_equal(coverage, np.ones(len(cloud.points), dtype=np.int64))
    np.testing.assert_array_equal(np.sort(selected), np.sort(select_voxel_points(cloud.points, .1)))


@pytest.mark.parametrize("scale, offset", [(.0001, (412345.6785, 5712345.6789, 0)), (.01, (0.005, 0.005, 0))])
def test_product_chunks_own_every_point_once_for_unaligned_offsets(tmp_path, scale, offset):
    # TLS-like 0.1 mm encodings with UTM offsets that are not multiples of the
    # voxel size: pdal info rounding and half-scale strip ends dropped points.
    rng = np.random.default_rng(7)
    x = offset[0] + rng.uniform(0, 30, 20_000)
    y = offset[1] + rng.uniform(0, 30, 20_000)
    cloud = _write(tmp_path / "points.las", scale, offset, x, y)
    whole = np.sort(select_voxel_points(cloud.points, .01))
    for chunks in (1, 7, 37):
        coverage = np.zeros(len(cloud.points), dtype=np.int64)
        selected = []
        for rows in _chunk_rows(cloud, _prod_merged_chunk_bounds([tmp_path / "points.las"], .01, chunks), .01):
            coverage[rows] += 1
            selected.extend(rows[select_voxel_points(cloud.points[rows], .01)].tolist())
        np.testing.assert_array_equal(coverage, np.ones(len(cloud.points), dtype=np.int64))
        np.testing.assert_array_equal(np.sort(selected), whole)


@pytest.mark.parametrize("centimeters", range(1, 101))
@pytest.mark.parametrize("origin", [0, 500000, -500000])
def test_aligned_grid_edges_use_exact_cell_indices(centimeters, origin):
    resolution = centimeters / 100
    # The caller computes step by multiplying a whole number of voxels.
    # For example, 3 * .07 arrives as .21000000000000002.
    start = float(Fraction(origin) + Fraction(centimeters, 100))
    stop = float(Fraction(origin) + Fraction(centimeters * 31, 100))
    edges = aligned_edges(start, stop, 3 * resolution, resolution)
    grid = Fraction(centimeters, 100)
    first = Fraction(str(start)) // grid
    expected = []
    position = first * grid
    while position < Fraction(str(stop)):
        end = position + 3 * grid
        if end > Fraction(str(start)):
            expected.append((max(float(position), start), min(float(end), stop)))
        position = end
    assert edges == expected
    assert all(a < b for a, b in edges)
