"""Original-resolution whole instances preserve source records across output formats."""
from collections import Counter
from pathlib import Path
import sys

import laspy
import numpy as np
import pytest
from pyproj import CRS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from instance_tiles import write_instance_tiles
from point_cloud_metadata import extra_bytes_params_from_dimension_info
from tile_centroids import Tile


def source(path, xs, ids, *, attribute_type=np.uint16):
    header = laspy.LasHeader(point_format=0, version="1.4")
    header.scales = [0.001] * 3
    header.offsets = [0.0] * 3
    header.add_crs(CRS.from_epsg(32632))
    header.add_extra_dims([
        laspy.ExtraBytesParams(name="PredInstance", type=np.uint64),
        laspy.ExtraBytesParams(name="Score", type=attribute_type, description="Calibrated score",
                              scales=[0.01], offsets=[-1.0], no_data=[65535]),
    ])
    cloud = laspy.LasData(header)
    cloud.x, cloud.y, cloud.z = np.asarray(xs), np.ones(len(xs)), np.ones(len(xs))
    cloud.PredInstance = np.asarray(ids, dtype=np.uint64)
    cloud.points.array["Score"] = np.arange(len(xs)) + 101
    cloud.points.array["Score"][0] = 65535
    cloud.intensity = np.arange(len(xs)) + 10
    cloud.write(path)
    return path


def tiles():
    return [Tile("left", (-2, -2, 12, 12), (0.001,) * 3, (5, 5, 0), core=(0, 0, 10, 10)),
            Tile("right", (8, -2, 22, 12), (0.001,) * 3, (15, 5, 0), core=(10, 0, 20, 10))]


def records(cloud, *, raw=True):
    # Count multiplicity, not just set membership: original mode keeps duplicates.
    return Counter(zip(np.rint(np.asarray(cloud.x) * 1000).astype(int).tolist(),
                       np.asarray(cloud.PredInstance).tolist(),
                       (cloud.points.array["Score"] if raw else np.asarray(cloud.Score)).tolist(),
                       np.asarray(cloud.intensity).tolist()))


@pytest.mark.parametrize("resolutions", [(0.0,), (0.0, 1.0)])
def test_original_instances_keep_every_record_and_exact_uint64_ids(tmp_path, resolutions):
    # Adjacent IDs above float64's exact-integer range must remain separate.
    first, second = 2**53, 2**53 + 1
    a = source(tmp_path / "a.las", [9.0, 9.0, 9.01, 11.0, 15.0, 3.0],
               [first, first, first, first, second, 0])
    b = source(tmp_path / "b.las", [9.0, 15.0], [first, second])
    output = tmp_path / "out"
    output.mkdir()
    directories = [output / f"subsampled_res{i + 1}" for i in range(len(resolutions))]
    result = write_instance_tiles([a, b], tiles(), ["PredInstance"], output, resolutions=resolutions,
                                  attributes=["Score", "intensity"], output_directories=directories)
    clouds = {path.stem: laspy.read(path) for path in directories[0].glob("*.laz")}
    assert result["PredInstance"]["instances"] == 2
    assert result["PredInstance"]["ids_in_several_files"] == 2
    assert result["PredInstance"]["outputs"][0.0]["points"] == 8
    assert sum((records(cloud) for cloud in clouds.values()), Counter()) == records(laspy.read(a)) + records(laspy.read(b))
    assert {name for name, cloud in clouds.items() if np.any(cloud.PredInstance == first)} == {"left"}
    assert {name for name, cloud in clouds.items() if np.any(cloud.PredInstance == second)} == {"right"}
    assert np.any(clouds["left"].x > 10)
    if len(resolutions) > 1:
        assert result["PredInstance"]["outputs"][1.0]["points"] < 8


def test_original_copc_preserves_duplicates_attributes_crs_and_missingness(tmp_path):
    a = source(tmp_path / "a.las", [1.0, 1.0, 1.01, 11.0, 11.01], [3, 3, 4, 5, 5])
    output = tmp_path / "out"
    output.mkdir()
    result = write_instance_tiles([a], tiles(), ["PredInstance"], output, resolutions=[0.0],
                                  formats=["copc.laz"], attributes=["Score", "intensity"])
    info = result["PredInstance"]["outputs"][0.0]
    assert info["format"] == "copc.laz" and info["points"] == 5
    clouds = [laspy.read(path) for path in info["directory"].glob("*.copc.laz")]
    assert sum((records(cloud, raw=False) for cloud in clouds), Counter()) == records(laspy.read(a), raw=False)
    for cloud in clouds:
        assert cloud.header.parse_crs().to_epsg() == 32632
        dimension = cloud.point_format.dimension_by_name("Score")
        params = extra_bytes_params_from_dimension_info(dimension, header=cloud.header)
        # COPC converters may materialize scaling into physical doubles. The
        # no-data declaration must identify the same physical missing value.
        missing = np.asarray(params.no_data) * (params.scales if params.scales is not None else 1)
        missing += params.offsets if params.offsets is not None else 0
        np.testing.assert_array_equal(missing, [654.35])


def read_ply(path):
    types = {"char": "i1", "uchar": "u1", "short": "<i2", "ushort": "<u2",
             "int": "<i4", "uint": "<u4", "float": "<f4", "double": "<f8"}
    with path.open("rb") as handle:
        lines, fields = [], []
        while True:
            line = handle.readline().decode("ascii").rstrip()
            assert line
            lines.append(line)
            if line.startswith("property "):
                _, kind, name = line.split()
                fields.append((name, types[kind]))
            if line == "end_header":
                break
        assert lines[1] == "format binary_little_endian 1.0"
        return lines, np.fromfile(handle, dtype=np.dtype(fields))


@pytest.mark.parametrize("original_ply", [True, False])
def test_mixed_original_and_sampled_formats_keep_instances_and_attributes(tmp_path, original_ply):
    a = source(tmp_path / "a.las", [1.0, 1.0, 1.01, 1.02], [2**32 - 1, 2**32 - 1, 2**32 - 1, 8])
    output = tmp_path / "out"
    output.mkdir()
    formats = ["ply", "laz"] if original_ply else ["laz", "ply"]
    result = write_instance_tiles([a], tiles(), ["PredInstance"], output, resolutions=[0.0, 0.1],
                                  formats=formats, attributes=["Score", "intensity"],
                                  output_directories=[output / "subsampled_res1", output / "subsampled_res2"])
    assert result["PredInstance"]["outputs"][0.0]["points"] == 4
    assert result["PredInstance"]["outputs"][0.1]["points"] == 2
    path = next(output.glob("subsampled_res*/*.ply"))
    lines, points = read_ply(path)
    assert "comment crs: EPSG:32632" in lines
    assert points.dtype.names == ("x", "y", "z", "PredInstance", "Score", "intensity")
    assert set(points["PredInstance"].tolist()) == {2**32 - 1, 8}
    assert points["PredInstance"].dtype == np.uint32
    assert len(points) == (4 if original_ply else 2)
    before = laspy.read(a)
    expected = Counter(zip(np.rint(np.asarray(before.x) * 1000), np.asarray(before.PredInstance),
                           np.asarray(before.Score), np.asarray(before.intensity)))
    actual = Counter(zip(np.rint(points["x"] * 1000), points["PredInstance"], points["Score"], points["intensity"]))
    if original_ply:
        assert actual == expected
    else:
        assert actual <= expected


def test_ply_overflow_rejects_entire_collection_without_partial_publication(tmp_path):
    a = source(tmp_path / "a.las", [1.0, 11.0], [1, 2**32])
    output = tmp_path / "out"
    output.mkdir()
    with pytest.raises(ValueError, match="PLY attribute PredInstance exceeds uint32 range"):
        write_instance_tiles([a], tiles(), ["PredInstance"], output, resolutions=[0.0, 0.1],
                             formats=["laz", "ply"])
    assert list(output.iterdir()) == []


def test_explicit_output_directories_reject_multiple_dimensions_before_processing(tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    with pytest.raises(ValueError, match="one instance dimension"):
        write_instance_tiles([], tiles(), ["SAT", "FM"], output, resolutions=[0.0],
                             output_directories=[output / "subsampled_res1"])


def test_original_resolution_aligns_tile_offsets_to_source_coordinate_grid(tmp_path):
    a = source(tmp_path / "a.las", [1.001, 1.002, 11.001], [1, 1, 2])
    cloud = laspy.read(a)
    cloud.change_scaling(offsets=np.full(3, 0.0002))
    cloud.write(a)
    before = laspy.read(a)
    output = tmp_path / "out"
    output.mkdir()
    result = write_instance_tiles([a], tiles(), ["PredInstance"], output, resolutions=[0.0])
    clouds = [laspy.read(path) for path in result["PredInstance"]["outputs"][0.0]["directory"].glob("*.laz")]
    actual = np.concatenate([np.c_[cloud.x, cloud.y, cloud.z] for cloud in clouds])
    expected = np.c_[before.x, before.y, before.z]
    np.testing.assert_allclose(actual[np.argsort(actual[:, 0])], expected[np.argsort(expected[:, 0])], atol=1e-12, rtol=0)
