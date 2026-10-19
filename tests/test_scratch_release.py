"""Scratch directories must not be removed while their files are still mapped.

On NFS job directories (Galaxy EU) removing a mapped file fails with EBUSY.
"""
import sys
import tempfile
from pathlib import Path

import laspy
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from remap_first_pipeline import strict_remap
from test_merge_stages import write_cloud

MAPS = Path("/proc/self/maps")


class MappedOnCleanup(tempfile.TemporaryDirectory):
    found = []

    def cleanup(self):
        mapped = [line.split(maxsplit=5)[-1].strip() for line in MAPS.read_text().splitlines()]
        MappedOnCleanup.found += [path for path in mapped if path.startswith(self.name + "/")]
        super().cleanup()


@pytest.mark.skipif(not MAPS.exists(), reason="needs /proc/self/maps")
@pytest.mark.parametrize("match_instances", [False, True])
def test_remap_unmaps_scratch_files_before_removing_them(tmp_path, monkeypatch, match_instances):
    originals, predictions = tmp_path / "raw", tmp_path / "pred"
    originals.mkdir()
    predictions.mkdir()
    write_cloud(originals / "a.las", [0, .1], [7, 9] if match_instances else None)
    prediction = write_cloud(predictions / "a.las", [0, .1], [7, 9])
    cloud = laspy.read(prediction)
    cloud.add_extra_dim(laspy.ExtraBytesParams(name="species_id", type=np.uint16))
    cloud.species_id = [10, 20]
    cloud.write(prediction)
    MappedOnCleanup.found = []
    monkeypatch.setattr(tempfile, "TemporaryDirectory", MappedOnCleanup)

    report = strict_remap(collections=[predictions], originals=originals, output=tmp_path / "enriched",
                          match_instances=match_instances)

    assert report["state"] == "validated"
    assert MappedOnCleanup.found == []
    np.testing.assert_array_equal(laspy.read(tmp_path / "enriched/a.las").species_id, [10, 20])
