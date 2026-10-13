"""Join versions of the same originals with attributes from several segmentation runs."""
import sys
from pathlib import Path

import laspy
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from attribute_join import join_attribute_collections  # noqa: E402
from parameters import Parameters  # noqa: E402
import run  # noqa: E402


def write_version(path, xyz, attributes):
    header = laspy.LasHeader(point_format=3, version="1.4")
    header.scales, header.offsets = np.full(3, 0.001), np.array([500000.0, 5000000.0, 0.0])
    for name in attributes:
        header.add_extra_dim(laspy.ExtraBytesParams(name=name, type=np.uint32))
    las = laspy.LasData(header)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    las.intensity = np.arange(len(xyz), dtype=np.uint16)
    for name, values in attributes.items():
        las[name] = values
    las.write(path)


def collections(tmp_path, shift=0.0):
    rng = np.random.default_rng(8)
    clouds = {name: np.c_[rng.uniform(500000, 500030, n), rng.uniform(5000000, 5000030, n), rng.uniform(0, 20, n)]
              for name, n in (("a", 1200), ("b", 800))}
    sat, fm = tmp_path / "sat", tmp_path / "fm"
    sat.mkdir(), fm.mkdir()
    for name, xyz in clouds.items():
        ids = np.arange(len(xyz), dtype=np.uint32)
        write_version(sat / f"{name}.laz", xyz, {"PredInstance_SAT": ids % 7, "PredSemantic": ids % 3})
        write_version(fm / f"{name}.laz", xyz + shift, {"PredInstance_FM": ids % 5, "PredSemantic": ids % 3})
    return sat, fm, clouds


def test_join_keeps_points_and_every_attribute(tmp_path):
    sat, fm, clouds = collections(tmp_path)
    joined = join_attribute_collections([sat, fm], tmp_path / "joined")
    for name, xyz in clouds.items():
        las = laspy.read(joined / f"{name}.laz")
        assert len(las.points) == len(xyz)
        names = [d.name for d in las.point_format.extra_dimensions]
        assert names == ["PredInstance_SAT", "PredSemantic", "PredInstance_FM"]
        ids = np.arange(len(xyz))
        assert np.array_equal(las.PredInstance_SAT, ids % 7) and np.array_equal(las.PredInstance_FM, ids % 5)
        assert np.array_equal(las.intensity, ids) and np.allclose(las.x, xyz[:, 0], atol=5e-4)


def test_join_rejects_other_points_and_conflicting_attributes(tmp_path):
    sat, fm, _ = collections(tmp_path, shift=0.01)
    with pytest.raises(ValueError, match="same points"):
        join_attribute_collections([sat, fm], tmp_path / "joined")
    other = tmp_path / "other"
    other.mkdir()
    sat, fm, clouds = collections(other)
    write_version(fm / "a.laz", clouds["a"], {"PredSemantic": np.full(len(clouds["a"]), 9, dtype=np.uint32)})
    with pytest.raises(ValueError, match="PredSemantic differs"):
        join_attribute_collections([sat, fm], other / "joined")
    (fm / "b.laz").unlink()
    with pytest.raises(ValueError, match="other originals"):
        join_attribute_collections([sat, fm], other / "joined2")


def test_create_merged_file_joins_collections_into_one_full_product(tmp_path):
    sat, fm, clouds = collections(tmp_path)
    params = Parameters(task="create_merged_file", original_with_predictions_dir=f"{sat},{fm}",
                        output_dir=tmp_path / "out", merged_resolutions="full", merged_output_formats="laz",
                        _cli_parse_args=False)
    run.run_create_merged_file_task(params)
    merged = laspy.read(tmp_path / "out/merged.laz")
    names = [d.name for d in merged.point_format.extra_dimensions]
    assert names[:2] == ["OriginalPointIndex", "PredInstance_SAT"] and "PredInstance_FM" in names
    total = sum(len(xyz) for xyz in clouds.values())
    assert np.array_equal(merged.OriginalPointIndex, np.arange(total))
    assert not list((tmp_path / "out").glob(".joined-originals-*"))
