"""Original-resolution coordinate grids must preserve every source coordinate."""
from pathlib import Path
import sys

import laspy
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tile_centroids import Tile
from tile_original_geometry import original_coordinate_tiles


def write_cloud(path, scale, offset):
    header = laspy.LasHeader(point_format=0, version="1.4")
    header.scales = np.full(3, scale)
    header.offsets = np.full(3, offset)
    cloud = laspy.LasData(header)
    cloud.x = np.asarray([1.0 + offset])
    cloud.y = np.asarray([2.0 + offset])
    cloud.z = np.asarray([3.0 + offset])
    cloud.write(path)
    return path


def test_off_grid_tile_center_is_aligned_to_source_grid(tmp_path):
    source = write_cloud(tmp_path / "a.las", 0.001, 0.0002)
    tile = Tile("a", (0, 0, 10, 10), (0.001,) * 3, (5.0005,) * 3)
    adjusted = original_coordinate_tiles([source], [tile])[0]
    np.testing.assert_allclose(adjusted.offsets, (5.0002,) * 3, atol=1e-12, rtol=0)
    assert tile.offsets == (5.0005,) * 3


@pytest.mark.parametrize("scale, offset", [(0.0015, 0), (0.001, 0.0005)])
def test_incompatible_source_grid_is_rejected(tmp_path, scale, offset):
    a = write_cloud(tmp_path / "a.las", 0.001, 0)
    b = write_cloud(tmp_path / "b.las", scale, offset)
    tile = Tile("a", (0, 0, 10, 10), (0.001,) * 3, (5,) * 3)
    with pytest.raises(ValueError, match="b.las: coordinate grid cannot be preserved"):
        original_coordinate_tiles([a, b], [tile])


def test_compatible_different_scales_share_the_finer_grid(tmp_path):
    a = write_cloud(tmp_path / "a.las", 0.001, 0)
    b = write_cloud(tmp_path / "b.las", 0.01, 0.002)
    tile = Tile("a", (0, 0, 10, 10), (0.001,) * 3, (5,) * 3)
    assert original_coordinate_tiles([a, b], [tile]) == [tile]
