"""Remap must close its scratch stores before removing them (NFS, Galaxy EU)."""
import sys
from pathlib import Path

import laspy
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from remap_first_pipeline import strict_remap
from test_merge_stages import write_cloud


@pytest.mark.parametrize("match_instances", [False, True])
def test_remap_closes_scratch_files_before_removing_them(tmp_path, match_instances):
    # conftest fails the test if a scratch folder is removed with files still open.
    originals, predictions = tmp_path / "raw", tmp_path / "pred"
    originals.mkdir()
    predictions.mkdir()
    write_cloud(originals / "a.las", [0, .1], [7, 9] if match_instances else None)
    prediction = write_cloud(predictions / "a.las", [0, .1], [7, 9])
    cloud = laspy.read(prediction)
    cloud.add_extra_dim(laspy.ExtraBytesParams(name="species_id", type=np.uint16))
    cloud.species_id = [10, 20]
    cloud.write(prediction)

    report = strict_remap(collections=[predictions], originals=originals, output=tmp_path / "enriched",
                          match_instances=match_instances)

    assert report["state"] == "validated"
    np.testing.assert_array_equal(laspy.read(tmp_path / "enriched/a.las").species_id, [10, 20])
