"""Retained-point predictions remap without an invented voxel resolution."""
import json
import sys
from pathlib import Path

import laspy
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from remap_first_pipeline import strict_remap


def write_cloud(path, xyz, ids, *, species=None, scale=0.000001, offset=None, dimension="tree_id"):
    header = laspy.LasHeader(point_format=3, version="1.4")
    header.scales = np.full(3, scale)
    header.offsets = np.array([500000.0, 5800000.0, 100.0] if offset is None else offset)
    header.add_extra_dim(laspy.ExtraBytesParams(name=dimension, type=np.uint32))
    if species is not None:
        header.add_extra_dim(laspy.ExtraBytesParams(name="species", type=np.uint16))
    cloud = laspy.LasData(header)
    cloud.x, cloud.y, cloud.z = np.asarray(xyz).T
    cloud[dimension] = ids
    cloud.intensity = np.arange(len(ids)) + 100
    if species is not None:
        cloud.species = species
    cloud.write(path)


@pytest.mark.parametrize("workers", [1, 3])
def test_original_predictions_reordered_reencoded_and_same_instance(tmp_path, monkeypatch, workers):
    import remap_first_pipeline as pipeline
    monkeypatch.setattr(pipeline, "MAX_BATCH_POINTS", 2)
    monkeypatch.setattr(pipeline, "spatial_query_worker_count", lambda count: count)
    originals, predictions = tmp_path / "originals", tmp_path / "predictions"
    originals.mkdir()
    predictions.mkdir()
    offset = np.array([500000., 5800000., 100.])
    # Different trees share exact coordinates. Each tree also has several
    # point-specific predictions, so matching by tree alone would be wrong.
    xyz = offset + np.array([[0, 0, 0], [.010123, .021234, .032345],
                            [.010123, .021234, .032345], [.123456, .023456, .034567],
                            [.223456, .023456, .034567], [.323456, .023456, .034567]])
    ids = np.array([0, 7, 70000, 7, 70000, 0], dtype=np.uint32)
    species = np.array([0, 10, 20, 11, 21, 0], dtype=np.uint16)
    write_cloud(originals / "a.las", xyz, ids)
    original = laspy.read(originals / "a.las")
    order = np.array([4, 1, 5, 2, 0, 3])
    # Half-scale shifted encoding verifies the permitted LAS rounding error;
    # the retained geometry still comes from the same original points.
    write_cloud(predictions / "tile.laz", xyz[order], ids[order], species=species[order],
                offset=offset + 0.0000005)
    report = strict_remap(collections=[predictions], originals=originals, output=tmp_path / "out",
                          instance_dimension="tree_id", match_instances=True,
                          prediction_resolution_original=True, workers=workers)
    result = laspy.read(tmp_path / "out/a.las")
    np.testing.assert_array_equal(result.species, species)
    np.testing.assert_array_equal(result.tree_id, ids)
    for name in original.point_format.dimension_names:
        np.testing.assert_array_equal(result[name], original[name])
    assert report["resolution_1_m"] is None
    assert report["prediction_resolution"] == "original"
    assert 0 < report["original_radius_m"] < 0.000002
    assert all(metric["matched"] == metric["total"] for metric in report["original_coverage"])


@pytest.mark.parametrize("failure", ["missing", "shifted", "wrong-instance"])
def test_original_predictions_require_complete_precise_coverage(tmp_path, failure):
    originals, predictions = tmp_path / "originals", tmp_path / "predictions"
    originals.mkdir()
    predictions.mkdir()
    xyz = np.array([[500000., 5800000., 100.], [500000.01, 5800000., 100.]])
    ids = np.array([7, 70000], dtype=np.uint32)
    write_cloud(originals / "a.las", xyz, ids)
    prediction_xyz, prediction_ids = xyz.copy(), ids.copy()
    if failure == "missing":
        prediction_xyz, prediction_ids = prediction_xyz[:1], prediction_ids[:1]
    elif failure == "shifted":
        # The former default 1 cm radius would accept this displaced point.
        prediction_xyz[1, 0] += .002
    else:
        prediction_ids[1] = 7
    write_cloud(predictions / "tile.laz", prediction_xyz, prediction_ids,
                species=np.full(len(prediction_ids), 10))
    with pytest.raises(ValueError, match="100% original coverage"):
        strict_remap(collections=[predictions], originals=originals, output=tmp_path / "out",
                     instance_dimension="tree_id", match_instances=True,
                     prediction_resolution_original=True)
    assert not (tmp_path / "out").exists()
    report = json.loads((tmp_path / "out_coverage.json").read_text())
    assert report["state"] == "failed"
    assert report["original_coverage"][0]["matched"] == 1


@pytest.mark.parametrize("conflict", [{"resolution_1": .01}, {"remap_tolerance": .01}])
def test_original_prediction_mode_rejects_voxel_resolution_or_override(tmp_path, conflict):
    with pytest.raises(ValueError, match="cannot be combined"):
        strict_remap(collections=[], originals=tmp_path, output=tmp_path / "out",
                     prediction_resolution_original=True, **conflict)


def test_original_prediction_collections_keep_independent_precision_allowances(tmp_path):
    originals = tmp_path / "originals"
    originals.mkdir()
    xyz = np.array([[500000., 5800000., 100.]])
    write_cloud(originals / "a.las", xyz, [7])
    collections = []
    for name, scale, shifted in [("coarse", .01, False), ("precise", .000001, True)]:
        directory = tmp_path / name
        directory.mkdir()
        points = xyz + ([.002, 0, 0] if shifted else [0, 0, 0])
        path = directory / "a.las"
        write_cloud(path, points, [7], species=[10], scale=scale)
        cloud = laspy.read(path)
        cloud.add_extra_dim(laspy.ExtraBytesParams(name=f"species_{name}", type=np.uint16))
        cloud[f"species_{name}"] = cloud.species
        cloud.remove_extra_dim("species")
        cloud.write(path)
        collections.append(directory)
    with pytest.raises(ValueError, match="100% original coverage"):
        strict_remap(collections=collections, originals=originals, output=tmp_path / "out",
                     instance_dimension="tree_id", match_instances=True,
                     prediction_resolution_original=True)
    report = json.loads((tmp_path / "out_coverage.json").read_text())
    coarse, precise = report["original_coverage"]
    assert coarse["matched"] == 1 and precise["matched"] == 0
    assert coarse["radius_m"] > 1000 * precise["radius_m"]


@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("workers", [1, 2])
def test_original_resolution_rct_uses_nearest_background_and_rejects_missing(tmp_path, monkeypatch, missing, workers):
    import remap_first_pipeline as pipeline
    monkeypatch.setattr(pipeline, "MAX_BATCH_POINTS", 1)
    monkeypatch.setattr(pipeline, "spatial_query_worker_count", lambda count: count)
    originals, predictions = tmp_path / "originals", tmp_path / "predictions"
    originals.mkdir()
    predictions.mkdir()
    xyz = np.array([[500000., 5800000., 100.], [500000., 5800000., 100.]])
    if missing:
        xyz[1, 0] += .02
    write_cloud(originals / "a.las", xyz, [0, 0])
    write_cloud(predictions / "a.las", np.array([[500000., 5800000., 100.],
                                               [500000.001, 5800000., 100.]]),
                [0, 7], species=[0, 99], scale=[.0001, .0001, .01], dimension="PredInstance_RCT")
    kwargs = dict(collections=[predictions], originals=originals, output=tmp_path / "out",
                  prediction_resolution_original=True, target_dims=["species"], workers=workers)
    if missing:
        with pytest.raises(ValueError, match="100% original coverage"):
            strict_remap(**kwargs)
        assert not (tmp_path / "out").exists()
    else:
        report = strict_remap(**kwargs)
        np.testing.assert_array_equal(laspy.read(tmp_path / "out/a.las").species, [0, 0])
        assert report["original_coverage_policy"]["final_survivors"] == "100% required for every model"
