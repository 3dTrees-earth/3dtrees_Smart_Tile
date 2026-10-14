"""LAS attribute names must never choose paths in the job filesystem."""
import json
import sys
import tempfile
from pathlib import Path

import laspy
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from remap_first_pipeline import merge_collections, strict_remap
from test_merge_stages import write_cloud
from test_remap_first_pipeline import layout


@pytest.mark.parametrize("match_instances", [False, True])
@pytest.mark.parametrize("absolute", [False, True])
def test_remap_keeps_attribute_paths_inside_scratch(tmp_path, absolute, match_instances):
    # LAS names are limited to 32 bytes. A short, private directory lets the
    # absolute-path regression exercise the real header instead of truncation.
    with tempfile.TemporaryDirectory(prefix="st-") as other:
        target = (Path(other) if absolute else tmp_path) / "escaped.npy"
        target.write_bytes(b"must survive remap")
        field = str(target.with_suffix("")) if absolute else "../../escaped"
        originals, predictions = tmp_path / "raw", tmp_path / "pred"
        originals.mkdir()
        predictions.mkdir()
        original = write_cloud(originals / "a.las", [0, .1])
        prediction = write_cloud(predictions / "a.las", [0, .1])
        for path in ([original, prediction] if match_instances else [prediction]):
            cloud = laspy.read(path)
            cloud.add_extra_dim(laspy.ExtraBytesParams(name=field, type=np.uint32))
            cloud[field] = [7, 9]
            if path == prediction and match_instances:
                cloud.add_extra_dim(laspy.ExtraBytesParams(name="species_id", type=np.uint16))
                cloud.species_id = [10, 20]
            cloud.write(path)
        report = strict_remap(
            collections=[predictions], originals=originals, output=tmp_path / "enriched",
            instance_dimension=field if match_instances else "PredInstance",
            match_instances=match_instances,
        )
        assert report["state"] == "validated"
        assert target.read_bytes() == b"must survive remap"
        result = laspy.read(tmp_path / "enriched/a.las")
        np.testing.assert_array_equal(result[field], [7, 9])
        if match_instances:
            np.testing.assert_array_equal(result.species_id, [10, 20])


def test_merge_summary_cannot_escape_its_output_collection(tmp_path):
    dimension = "../../escaped"
    sentinel = tmp_path / "escaped_summary.json"
    sentinel.write_text("must survive merge")
    source = tmp_path / "predictions"
    source.mkdir()
    write_cloud(source / "a.las", [0, .1], [7, 7], instance=dimension)
    output = tmp_path / "out"
    merge_collections(collections=[source], target_dir=None, output_tiles=output,
                      tile_bounds_json=layout(tmp_path, 1), ready=True,
                      instance_dimension=dimension, keep_unfiltered=False)
    manifest = json.loads((output / "smarttile_merge.json").read_text())
    summary = output / manifest["instance_summary"]
    assert summary.parent == output
    assert json.loads(summary.read_text())["instance_dimension"] == dimension
    assert sentinel.read_text() == "must survive merge"
