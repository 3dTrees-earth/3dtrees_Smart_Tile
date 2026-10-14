"""Shared tile-output selections feed original-resolution species remapping."""
import os
from pathlib import Path
import subprocess
import sys

import laspy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def invoke(*args, success=True):
    result = subprocess.run([sys.executable, str(ROOT / "src/run.py"), *map(str, args)],
                            capture_output=True, text=True, timeout=120,
                            env={**os.environ, "MPLBACKEND": "Agg"})
    assert (result.returncode == 0) == success, result.stdout + result.stderr
    return result


def originals(tmp_path):
    source, target = tmp_path / "labeled", tmp_path / "originals"
    source.mkdir()
    target.mkdir()
    header = laspy.LasHeader(point_format=0, version="1.4")
    header.scales = [0.001] * 3
    header.offsets = [398700, 5646160, 0]
    header.add_extra_dim(laspy.ExtraBytesParams(name="CustomTreeId", type="uint32"))
    header.add_extra_dim(laspy.ExtraBytesParams(name="PredSpecies", type="uint16"))
    cloud = laspy.LasData(header)
    cloud.x = 398700 + np.array([1, 1, 1.001, 1.003, 2, 2.001, 2.003, 2.003])
    cloud.y = 5646160 + np.array([1, 1, 1.001, 1.002, 2, 2.001, 2.002, 2.002])
    cloud.z = np.array([0, 0, 0.001, 0.002, 0, 0.001, 0.002, 0.002])
    cloud.CustomTreeId = [7, 7, 7, 7, 70000, 70000, 70000, 70000]
    cloud.PredSpecies = [10] * 4 + [20] * 4
    cloud.intensity = np.arange(8, dtype=np.uint16) + 100
    cloud.write(source / "original.laz")
    cloud.remove_extra_dim("PredSpecies")
    cloud.write(target / "original.laz")
    return source, target


def test_original_copc_and_sampled_outputs_remap_to_exact_originals(tmp_path):
    source, target = originals(tmp_path)
    output = tmp_path / "tile"
    invoke("--task", "tile", "--input-dir", source, "--output-dir", output,
           "--instance-tiles", "CustomTreeId", "--instance-outputs",
           "original:copc.laz,0.1:laz,0.2:ply", "--instance-tile-dims", "PredSpecies,intensity",
           "--workers", "1")
    full = list((output / "subsampled_res1").glob("*.copc.laz"))
    coarse = list((output / "subsampled_res2").glob("*.laz"))
    ply = list((output / "subsampled_res3").glob("*.ply"))
    assert len(full) == len(coarse) == len(ply) == 1
    cloud = laspy.read(full[0])
    assert len(cloud.points) == 8
    assert sorted(cloud.intensity.tolist()) == list(range(100, 108))
    assert set(cloud.CustomTreeId) == {7, 70000}
    assert len(laspy.read(coarse[0]).points) == 2
    assert b"CustomTreeId" in ply[0].read_bytes().split(b"end_header")[0]
    remapped = tmp_path / "remapped"
    invoke("--task", "remap", "--segmented-folders", output / "subsampled_res1",
           "--original-laz-input-dir", target, "--original-laz-output-dir", remapped,
           "--prediction-resolution-original", "--instance-dimension", "CustomTreeId",
           "--match-instances", "--remap-dims", "PredSpecies", "--workers", "1")
    before, after = laspy.read(target / "original.laz"), laspy.read(remapped / "original.laz")
    for name in before.points.array.dtype.names:
        np.testing.assert_array_equal(before.points.array[name], after.points.array[name])
    np.testing.assert_array_equal(after.PredSpecies, [10] * 4 + [20] * 4)


def test_original_resolution_conflicts_fail_before_writing(tmp_path):
    source, target = originals(tmp_path)
    output = tmp_path / "tile"
    result = invoke("--task", "tile", "--input-dir", source, "--output-dir", output,
                    "--instance-tiles", "CustomTreeId", "--instance-outputs", "original:laz",
                    "--tile-outputs", "0.1:laz", success=False)
    assert "cannot be combined" in result.stdout + result.stderr
    assert not output.exists()
    output = tmp_path / "remapped"
    result = invoke("--task", "remap", "--segmented-folders", source,
                    "--original-laz-input-dir", target, "--original-laz-output-dir", output,
                    "--prediction-resolution-original", "--resolution-1", "0.01", success=False)
    assert "cannot be combined" in result.stdout + result.stderr
    assert not output.exists()
