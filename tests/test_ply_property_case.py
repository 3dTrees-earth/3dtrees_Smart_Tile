"""PDAL lowercases PLY property names; the header rewrite restores them (3DT-2352)."""
import shutil
import subprocess
import sys
from pathlib import Path

import laspy
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ply_crs import add_crs_comment_to_ply, property_case_map  # noqa: E402

PLY_TYPES = {"char": "i1", "uchar": "u1", "short": "<i2", "ushort": "<u2", "int": "<i4", "uint": "<u4",
             "float": "<f4", "double": "<f8", "int8": "i1", "uint8": "u1", "int16": "<i2", "uint16": "<u2",
             "int32": "<i4", "uint32": "<u4", "float32": "<f4", "float64": "<f8"}


def read_ply(path):
    with open(path, "rb") as stream:
        header = []
        while (line := stream.readline().decode("ascii").rstrip("\r\n")) != "end_header":
            header.append(line)
        count = next(int(line.split()[2]) for line in header if line.startswith("element vertex"))
        props = [line.split()[1:] for line in header if line.startswith("property ")]
        dtype = np.dtype([(name, PLY_TYPES[kind]) for kind, name in props])
        return header, np.fromfile(stream, dtype=dtype, count=count)


def write_source(path):
    header = laspy.LasHeader(point_format=7, version="1.4")
    header.scales, header.offsets = np.full(3, 0.001), np.array([400000.0, 5600000.0, 0.0])
    for name, kind in (("treeID", np.uint32), ("PredSemantic", np.uint8),
                       ("semantic_seg", np.uint8), ("NirChannel", np.float32)):
        header.add_extra_dim(laspy.ExtraBytesParams(name=name, type=kind))
    las = laspy.LasData(header)
    las.x = 400000.0 + np.array([1.5, 2.25, 3.0])
    las.y = 5600000.0 + np.array([4.0, 5.5, 6.75])
    las.z = np.array([0.5, 1.0, 1.5])
    las.red, las.green, las.blue = (np.array([1, 2, 3], dtype=np.uint16) * k for k in (100, 200, 300))
    las.treeID = np.array([7, 70000, 0], dtype=np.uint32)
    las.PredSemantic = np.array([1, 2, 0], dtype=np.uint8)
    las.semantic_seg = np.array([3, 3, 1], dtype=np.uint8)
    las.NirChannel = np.array([0.25, 0.5, 0.75], dtype=np.float32)
    las.write(path)
    return las


def test_case_map_keeps_conventional_names():
    assert property_case_map(["treeID", "X", "Red", "semantic_seg"]) == {"treeid": "treeID"}


@pytest.mark.parametrize("names", [("treeID", "TreeID"), ("treeID", "treeid"), ("treeid", "treeID")])
def test_case_map_rejects_case_twins_including_lowercase_names(names):
    with pytest.raises(ValueError, match="differ only in case"):
        property_case_map(names)


@pytest.mark.skipif(shutil.which("pdal") is None, reason="needs the PDAL executable")
@pytest.mark.parametrize("dims", [None, "X,Y,Z,Red,Green,Blue,treeID,PredSemantic,semantic_seg,NirChannel"])
def test_pdal_ply_gets_exact_names_and_values_back(tmp_path, dims):
    source, target = tmp_path / "in.laz", tmp_path / "out.ply"
    las = write_source(source)
    command = ["pdal", "translate", str(source), str(target), "--writers.ply.storage_mode=little endian"]
    if dims:
        command.append(f"--writers.ply.dims={dims}")
    subprocess.run(command, check=True, capture_output=True)
    before, _ = read_ply(target)
    assert any(line.endswith(" treeid") for line in before)  # stock PDAL lowercases
    names = laspy.open(source).header.point_format.extra_dimension_names
    assert add_crs_comment_to_ply(target, "EPSG:25832", restore_names=names)
    header, data = read_ply(target)
    assert "comment crs: EPSG:25832" in header
    for name in ("treeID", "PredSemantic", "semantic_seg", "NirChannel", "x", "y", "z", "red", "green", "blue"):
        assert name in data.dtype.names, (name, data.dtype.names)
    np.testing.assert_array_equal(data["treeID"], las.treeID)  # above 65535 survives
    np.testing.assert_array_equal(data["PredSemantic"], las.PredSemantic)
    np.testing.assert_allclose(data["NirChannel"], las.NirChannel)
    np.testing.assert_allclose(data["x"], las.x)  # georeferenced float64 XYZ
    np.testing.assert_array_equal(data["red"], las.red)
