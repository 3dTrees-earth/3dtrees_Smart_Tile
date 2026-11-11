"""The tile grid holds every point (datasets 3860 and 3849).

PDAL's tile index keeps 15 significant digits. 3860's minimum Y,
2781364.3847499997, came back as 2781364.38475, 4.7e-10 m inside the data, and
the grid start left the one point on the edge outside every tile core.
"""
import json
from pathlib import Path
import shutil
import sys

import laspy
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from get_bounds_from_tindex import build_tiles, write_tile_bounds
from tile_tindex import point_extent

MIN_Y_3860 = 2781364.3847499997
SCALE_3860, OFFSETS_3860 = 0.00025, (508586.946, 2781614.738, -17.311)  # file d5a23bbf of dataset 3860


def write_cloud(path, xs, ys, *, scale=SCALE_3860, offsets=OFFSETS_3860):
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales, header.offsets = np.full(3, scale), np.array(offsets)
    cloud = laspy.LasData(header)
    cloud.x, cloud.y, cloud.z = np.asarray(xs), np.asarray(ys), np.zeros(len(xs))
    cloud.write(path)
    return laspy.read(path)


def test_extent_is_the_points_own_coordinates(tmp_path):
    cloud = write_cloud(tmp_path / "a.las", [508395.76025, 508768.13825], [MIN_Y_3860, 2781834.8495])
    extent = point_extent([tmp_path / "a.las"])
    assert extent[1] == float(np.min(cloud.y)) == MIN_Y_3860
    assert extent[0] == float(np.min(cloud.x)) and extent[2] == float(np.max(cloud.x))


def test_extent_follows_the_las_grid_when_the_writer_rounded_the_header(tmp_path):
    path = tmp_path / "a.las"
    cloud = write_cloud(path, [508395.76025, 508768.13825], [MIN_Y_3860, 2781834.8495])
    data = bytearray(path.read_bytes())
    data[219:227] = np.array([float(f"{MIN_Y_3860:.15g}")], "<f8").tobytes()  # header min Y
    path.write_bytes(data)
    assert point_extent([path])[1] == MIN_Y_3860


def write_index(tmp_path, monkeypatch, extent):
    import get_bounds_from_tindex
    monkeypatch.setattr(get_bounds_from_tindex, "load_extent_from_tindex", lambda path: (extent, "EPSG:32617"))


def test_source_extent_widens_the_rounded_index(tmp_path, monkeypatch):
    rounded = (508365.97675, float(f"{MIN_Y_3860:.15g}"), 508790.8795, 2781878.38125)
    assert rounded[1] > MIN_Y_3860
    write_index(tmp_path, monkeypatch, rounded)
    out = tmp_path / "bounds.json"
    write_tile_bounds(tmp_path / "t.gpkg", 300.0, 20.0, out, source_extent=(rounded[0], MIN_Y_3860, *rounded[2:]))
    data = json.loads(out.read_text())
    cores = [tile["core"] for tile in data["tiles"]]
    assert min(core[1][0] for core in cores) == MIN_Y_3860
    assert data["proj_extent"]["miny"] == MIN_Y_3860


def test_source_extent_never_narrows_the_index(tmp_path, monkeypatch):
    write_index(tmp_path, monkeypatch, (0.0, 0.0, 10.0, 10.0))
    out = tmp_path / "bounds.json"
    write_tile_bounds(tmp_path / "t.gpkg", 5.0, 0.0, out, source_extent=(1.0, 1.0, 9.0, 9.0))
    assert json.loads(out.read_text())["proj_extent"] == {"minx": 0.0, "miny": 0.0, "maxx": 10.0, "maxy": 10.0}


@pytest.mark.parametrize("length", [0.1, 0.3, 300.0, 7.77])
@pytest.mark.parametrize("start", [508365.97675, 2781364.3847499997, 0.1])
def test_neighbouring_cores_share_their_edge(start, length):
    tiles, _ = build_tiles(start, start, start + 40 * length, start + 3 * length, length, 0.0)
    by_position = {(t["col"], t["row"]): t["core"] for t in tiles}
    for (col, row), core in by_position.items():
        if (col + 1, row) in by_position:
            assert core[0][1] == by_position[(col + 1, row)][0][0]
        if (col, row + 1) in by_position:
            assert core[1][1] == by_position[(col, row + 1)][1][0]


def test_grid_reaches_a_maximum_on_the_last_edge():
    start, length = 0.1, 0.1
    stop = start + 3 * length
    tiles, _ = build_tiles(start, start, np.nextafter(stop, np.inf), start, length, 0.0)
    assert max(tile["core"][0][1] for tile in tiles) >= np.nextafter(stop, np.inf)


@pytest.mark.skipif(shutil.which("pdal") is None, reason="needs PDAL")
def test_tiling_keeps_the_point_on_the_rounded_edge(tmp_path):
    from main_tile import plan_tiling
    from tile_centroids import subsample_tiles_single_scan
    from parameters import Parameters  # noqa: F401  (run.py imports it before tiling)

    source = tmp_path / "in"
    source.mkdir()
    rng = np.random.default_rng(3)
    xs = np.r_[508395.76025, rng.uniform(508396, 508750, 2000)]
    ys = np.r_[MIN_Y_3860, rng.uniform(2781370, 2781800, 2000)]
    cloud = write_cloud(source / "a.las", xs, ys)
    assert float(np.min(cloud.y)) == MIN_Y_3860
    plan = plan_tiling(source, tmp_path / "out", 300.0, 20.0)
    directories = list(subsample_tiles_single_scan(plan, tmp_path / "out", (0.1,), ("laz",), workers=1, converters=1,
                                                   output_prefix="a"))
    tiled = sum(laspy.read(path).header.point_count for folder in directories for path in Path(folder).glob("*.laz"))
    assert tiled > 0
