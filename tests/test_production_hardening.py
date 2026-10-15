"""Regressions for the v2.4 production review: silent point loss and unsafe inputs."""
import json
import struct

import laspy
import numpy as np
import pytest
from pyproj import CRS
from laspy.vlrs.known import WktCoordinateSystemVlr
from laspy.vlrs.vlrlist import VLRList

import run
from get_bounds_from_tindex import build_tiles
from parameters import Parameters
from point_cloud_metadata import point_cloud_files, raw_point_cloud_files
from tile_centroids import tile_offsets
from tile_crs import require_metric_tiling_crs
from tile_original_geometry import _on_grid


def _cloud(path, x, y, *, scale=0.01, crs=None, evlr=False):
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.scales = np.full(3, scale)
    header.offsets = np.zeros(3)
    if crs is not None:
        record = WktCoordinateSystemVlr(CRS.from_user_input(crs).to_wkt("WKT1_GDAL"))
        if evlr:
            header.evlrs = VLRList([record])
        else:
            header.vlrs.append(record)
        header.global_encoding.wkt = True
    cloud = laspy.LasData(header)
    cloud.x, cloud.y, cloud.z = x, y, np.zeros(len(x))
    cloud.write(path)
    return path


def _tile(input_dir, output_dir, *extra):
    run.run_tile_task(Parameters(_cli_parse_args=[
        "--task", "tile", "--input-dir", str(input_dir), "--output-dir", str(output_dir),
        "--tile-length", "10", "--tile-buffer", "0", "--tile-outputs", "0.1:laz", "--workers", "1", *extra]))


def test_points_beyond_stale_header_bounds_fail_instead_of_being_dropped(tmp_path, capsys):
    source = tmp_path / "in" / "a.las"
    source.parent.mkdir()
    x = np.r_[np.linspace(0, 19.9, 200), [25.0] * 5]
    _cloud(source, x, np.linspace(0, 9.9, 205))
    data = bytearray(source.read_bytes())
    data[179:187] = struct.pack("<d", 19.9)  # header Max X written by another tool
    source.write_bytes(bytes(data))
    with pytest.raises(SystemExit):
        _tile(source.parent, tmp_path / "out")
    assert "outside every planned tile core" in capsys.readouterr().out
    assert not list((tmp_path / "out").rglob("c*_r*.laz"))


def test_tile_offsets_are_decimal_exact_so_voxel_faces_stay_put():
    # A float midpoint such as 32704.229999999996 floored face points into the lower voxel.
    x, y, z = tile_offsets((32654.23, 0.0, 32754.23, 100.0), 0.0)
    assert (x, y, z) == (32704.0, 50.0, 0.0)
    assert _on_grid(0.0, 32704.229999999996, 0.01) == 32704.23
    assert str(_on_grid(0.005, 12.3449, 0.01)) == "12.345"


def test_tile_grid_loop_is_index_based():
    tiles, _ = build_tiles(10.0, 0.0, 10.0, 25.0, 10, 2)  # zero-width extent
    assert len(tiles) == 3 and {t["col"] for t in tiles} == {0}
    # Adding 1 m to 1e16 does not advance; an accumulating loop never ended (3DT-1565).
    tiles, _ = build_tiles(1e16, 0.0, 1e16 + 100, 1.0, 1.0, 0)
    assert len(tiles) == 100


def test_sources_with_the_same_case_insensitive_name_fail(tmp_path):
    _cloud(tmp_path / "Plot1.las", [0.0], [0.0])
    _cloud(tmp_path / "plot1.laz", [1.0], [1.0])
    with pytest.raises(ValueError, match="name the same source"):
        point_cloud_files(tmp_path)
    with pytest.raises(ValueError, match="name the same source"):
        raw_point_cloud_files(tmp_path)


def test_empty_input_fails_with_a_clear_message(tmp_path, capsys):
    (tmp_path / "in").mkdir()
    _cloud(tmp_path / "in" / "a.las", [0.0, 5.0], [0.0, 5.0])
    _cloud(tmp_path / "in" / "b.las", [], [])
    with pytest.raises(SystemExit):
        _tile(tmp_path / "in", tmp_path / "out")
    assert "without points: b.las" in capsys.readouterr().out


@pytest.mark.parametrize("crs, evlr, message", [
    ("EPSG:4326", True, "geographic"),
    ("EPSG:26911+6360", False, "not metres"),  # NAVD88 height in US survey feet
])
def test_crs_check_reads_evlrs_and_vertical_units(tmp_path, crs, evlr, message):
    source = _cloud(tmp_path / "a.las", [0.0], [0.0], crs=crs, evlr=evlr)
    with pytest.raises(ValueError, match=message):
        require_metric_tiling_crs([source])


def test_unparseable_crs_records_are_rejected(tmp_path):
    source = _cloud(tmp_path / "a.las", [0.0], [0.0])
    cloud = laspy.read(source)
    cloud.header.vlrs.append(laspy.VLR("LASF_Projection", 2112, "OGC WKT", b"not a crs\0"))
    cloud.write(source)
    with pytest.raises(ValueError, match="Unreadable CRS records"):
        require_metric_tiling_crs([source])


def test_remap_refuses_copc_only_originals(tmp_path):
    from remap_first_pipeline import original_files

    _cloud(tmp_path / "a.laz", [0.0], [0.0])
    (tmp_path / "b.copc.laz").write_bytes((tmp_path / "a.laz").read_bytes())
    with pytest.raises(ValueError, match="COPC-only originals cannot be enriched.*b.copc.laz"):
        original_files(tmp_path)


def test_small_instance_relabel_keeps_laz_compressed(tmp_path):
    from small_instance_reassignment import relabel_tiles

    tile = tmp_path / "c00_r00.laz"
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.add_extra_dim(laspy.ExtraBytesParams(name="PredInstance", type=np.uint32))
    cloud = laspy.LasData(header)
    cloud.x, cloud.y, cloud.z = [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]
    cloud.PredInstance = [1, 2]
    cloud.write(tile)
    relabel_tiles([tile], "PredInstance", {2: 1})
    with laspy.open(tile) as reader:
        assert reader.header.are_points_compressed
    assert list(laspy.read(tile).PredInstance) == [1, 1]


def test_tile_bounds_cover_every_output_resolution(tmp_path):
    from tile_tindex import update_tile_bounds_json_from_files

    layout = tmp_path / "tile_bounds_tindex.json"
    layout.write_text(json.dumps({"tiles": [{"col": 0, "row": 0, "bounds": [[0, 10], [0, 10]]}]}))
    coarse, fine = tmp_path / "subsampled_res1", tmp_path / "subsampled_res2"
    coarse.mkdir(), fine.mkdir()
    _cloud(coarse / "c00_r00.laz", [0.05, 9.95], [0.05, 9.95])
    _cloud(fine / "c00_r00.laz", [0.0, 10.0], [0.01, 9.99])
    assert update_tile_bounds_json_from_files(layout, [coarse, fine]) == 1
    assert json.loads(layout.read_text())["tiles"][0]["bounds"] == [[0.0, 10.0], [0.01, 9.99]]


def test_environment_variables_do_not_configure_a_run(monkeypatch):
    monkeypatch.setenv("WORKERS", "3")
    monkeypatch.setenv("TASK", "remap")
    from worker_budget import allocated_cpu_count

    params = Parameters(_cli_parse_args=["--task", "tile"])
    assert (params.task, params.workers) == ("tile", allocated_cpu_count())


def test_full_resolution_copc_keeps_vector_extra_bytes(tmp_path):
    from main_create_merged_file import create_prod_merged_files

    originals, products = tmp_path / "originals", tmp_path / "products"
    originals.mkdir()
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.scales, header.offsets = np.full(3, 0.01), np.zeros(3)
    header.add_extra_dim(laspy.ExtraBytesParams(name="Normal", type="3f8"))
    cloud = laspy.LasData(header)
    cloud.x, cloud.y, cloud.z = np.linspace(0, 9, 10), np.linspace(0, 9, 10), np.zeros(10)
    cloud.Normal = np.tile([0.0, 0.0, 1.0], (10, 1))
    cloud.write(originals / "a.laz")
    outputs = create_prod_merged_files(originals, products, "full", "copc.laz", .01, .1, chunk_workers=1)
    assert len(outputs) == 1 and outputs[0].name.endswith(".copc.laz")
    with laspy.open(outputs[0]) as reader:
        assert reader.header.point_count == 10
