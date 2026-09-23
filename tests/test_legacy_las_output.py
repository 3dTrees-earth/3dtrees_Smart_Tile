"""Legacy uploaded LAS headers must remain readable after enrichment."""
import io
import sys
from pathlib import Path

import laspy
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from point_cloud_metadata import copy_single_source_header
from strict_prediction_pipeline import strict_remap


@pytest.mark.parametrize("preserve", [True, False])
def test_legacy_header_can_be_written_without_changing_source(preserve):
    source = laspy.LasHeader(point_format=0, version="1.1")
    source.scales = [.01, .01, .01]
    source.offsets = [610000., 4490000., 1600.]
    source.vlrs.append(laspy.VLR(user_id="test", record_id=91, record_data=b"keep"))
    encoded = io.BytesIO()
    source.write_to(encoded)
    data = bytearray(encoded.getvalue())
    data[25] = 0
    source = laspy.LasHeader.read_from(io.BytesIO(data))
    header = copy_single_source_header(source, preserve_extra_dimensions=preserve)
    assert str(source.version) == "1.0"
    assert str(header.version) == "1.4"
    assert header.point_format.id == source.point_format.id
    np.testing.assert_array_equal(header.scales, source.scales)
    np.testing.assert_array_equal(header.offsets, source.offsets)
    stream = io.BytesIO()
    with laspy.open(stream, mode="w", header=header, closefd=False) as writer:
        writer.write_points(laspy.ScaleAwarePointRecord.zeros(1, header=header))
    stream.seek(0)
    with laspy.open(stream) as reader:
        assert reader.header.point_count == 1
        assert reader.header.vlrs[0].record_data_bytes() == b"keep"


@pytest.mark.parametrize("version", ["1.1", "1.2", "1.3", "1.4"])
def test_supported_versions_are_preserved(version):
    source = laspy.LasHeader(point_format=0, version=version)
    assert copy_single_source_header(source).version == source.version


def test_strict_remap_enriches_legacy_original_and_preserves_point_fields(tmp_path):
    originals, predictions = tmp_path / "originals", tmp_path / "predictions"
    originals.mkdir()
    predictions.mkdir()
    cloud = laspy.LasData(laspy.LasHeader(point_format=0, version="1.1"))
    cloud.x, cloud.y, cloud.z = [610571.15, 610571.17], [4494812.05]*2, [1674.68]*2
    cloud.intensity = [71, 92]
    cloud.classification = [2, 5]
    cloud.header.vlrs.append(laspy.VLR(user_id="test", record_id=91, record_data=b"keep"))
    source = originals / "original.las"
    cloud.write(source)
    # LAS 1.0 is readable but cannot be produced by laspy's writer. Mark this
    # compatible format-0 fixture as the historical version seen in real inputs.
    data = bytearray(source.read_bytes())
    data[25] = 0
    source.write_bytes(data)
    cloud.add_extra_dim(laspy.ExtraBytesParams(name="PredInstance", type=np.uint16))
    cloud.add_extra_dim(laspy.ExtraBytesParams(name="PredSemantic", type=np.uint8))
    cloud.PredInstance, cloud.PredSemantic = [7, 7], [3, 3]
    cloud.write(predictions / "prediction.las")
    report = strict_remap(collections=[predictions], baseline_collections=[predictions],
                          originals=originals, output=tmp_path / "enriched")
    assert report["state"] == "validated"
    before, after = laspy.read(source), laspy.read(tmp_path / "enriched/original.las")
    assert str(before.header.version) == "1.0"
    assert str(after.header.version) == "1.4"
    np.testing.assert_array_equal(before.header.scales, after.header.scales)
    np.testing.assert_array_equal(before.header.offsets, after.header.offsets)
    for name in before.points.array.dtype.names:
        np.testing.assert_array_equal(before.points.array[name], after.points.array[name])
    np.testing.assert_array_equal(after.PredInstance, [7, 7])
    np.testing.assert_array_equal(after.PredSemantic, [3, 3])
    assert any(v.record_data_bytes() == b"keep" for v in after.header.vlrs)
