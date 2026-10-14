"""Product chunk boundaries must retain points on the absolute voxel grid."""
import ast
from fractions import Fraction

import laspy
import numpy as np
import pytest

from main_create_merged_file import _prod_merged_chunk_bounds, select_voxel_points
from subsample_com import aligned_edges


def test_product_chunk_reads_keep_boundary_points_and_match_whole_selection(tmp_path, monkeypatch):
    header = laspy.LasHeader(point_format=0, version="1.4")
    header.scales = np.full(3, .001)
    cloud = laspy.LasData(header)
    cloud.X = np.arange(1001, dtype=np.int32)
    cloud.Y = np.arange(1001, dtype=np.int32)
    cloud.Z = np.zeros(1001, dtype=np.int32)
    path = tmp_path / "points.las"
    cloud.write(path)
    # Header bounds stand in for PDAL's metadata reader; the actual chunk
    # planner reads this LAS file's point count and scales itself.
    monkeypatch.setattr("main_create_merged_file._copc_union_bounds", lambda _: (0., 1., 0., 1.))
    bounds = _prod_merged_chunk_bounds([path], .1, 10)
    xyz = np.asarray(cloud.xyz)
    coverage = np.zeros(len(cloud.points), dtype=np.int64)
    selected = []
    for value in bounds:
        ((xmin, xmax), (ymin, ymax)) = ast.literal_eval(value)
        rows = np.flatnonzero((xyz[:, 0] >= xmin) & (xyz[:, 0] <= xmax) &
                              (xyz[:, 1] >= ymin) & (xyz[:, 1] <= ymax))
        coverage[rows] += 1
        selected.extend(rows[select_voxel_points(cloud.points[rows], .1)].tolist())
    np.testing.assert_array_equal(coverage, np.ones(len(cloud.points), dtype=np.int64))
    expected = select_voxel_points(cloud.points, .1)
    np.testing.assert_array_equal(np.sort(selected), np.sort(expected))


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
