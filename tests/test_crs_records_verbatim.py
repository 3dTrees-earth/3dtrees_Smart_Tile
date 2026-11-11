"""Outputs written by SmartTile keep the source's CRS records byte for byte (3DT-2718).

The fixtures are the records of real standardized bases: 3805 (northing-first
EPSG:6678), 3827 (EPSG:3067, GeoTIFF keys plus WKT; the PROJ of the SmartTile
image knows it as EUREF-FIN) and 1919 (keys and WKT that disagree).
"""
from pathlib import Path
import sys

import laspy
from laspy.vlrs.vlrlist import VLRList
import numpy as np
import pytest
from pyproj import CRS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from crs_records import projection_record_list, validate_crs_records_unchanged, validate_single_crs_record
from full_resolution_merge import create_full_resolution_product
from point_cloud_metadata import copy_single_source_header
from remap_first_pipeline import strict_remap
from test_merge_stages import write_cloud

FIXTURES = Path(__file__).parent / "fixtures" / "crs"


def record(name):
    return (FIXTURES / name).read_bytes()


def northing_first_31467():
    return CRS.from_epsg(31467).to_wkt("WKT2_2019").encode() + b"\0"


CASES = {
    "3805_northing_first": {2112: record("3805_epsg6678.2112.bin")},
    "31467_northing_first": {2112: northing_first_31467()},
    "3827_keys_and_wkt": {34735: record("3827_epsg3067.34735.bin"), 2112: record("3827_epsg3067.2112.bin")},
    "1919_keys_and_wkt": {34735: record("1919_wkt_and_keys.34735.bin"), 2112: record("1919_wkt_and_keys.2112.bin")},
    "wkt_padded_with_nuls": {2112: record("3805_epsg6678.2112.bin") + b"\0" * 7},
    "wkt_without_nul": {2112: record("3805_epsg6678.2112.bin").rstrip(b"\0")},
    "geokeys_only": {34735: bytes([1, 0, 1, 0, 0, 0, 1, 0, 0, 12, 0, 0, 1, 0, 110, 19])},
}


def write_base(path, records, xs=(0.0, 0.1), version="1.4", point_format=3, wkt_flag=True, evlr=None):
    """A LAS file whose CRS records are exactly ``records``, as a writer other than laspy left them."""
    header = laspy.LasHeader(point_format=point_format, version=version)
    header.scales, header.offsets = np.full(3, 0.001), np.zeros(3)
    for record_id, data in records.items():
        header.vlrs.append(laspy.VLR(user_id="LASF_Projection", record_id=record_id, record_data=data))
    if evlr is not None:
        header.evlrs = VLRList([laspy.VLR(user_id="LASF_Projection", record_id=2112, record_data=evlr)])
    if version == "1.4":
        header.global_encoding.wkt = wkt_flag and 2112 in records
    cloud = laspy.LasData(header)
    cloud.x, cloud.y, cloud.z = np.asarray(xs), np.zeros(len(xs)), np.zeros(len(xs))
    cloud.write(path)


def payloads(path):
    with laspy.open(path) as reader:
        return projection_record_list(reader.header), reader.header.global_encoding.wkt


def enrich(tmp_path, base):
    originals, predictions = tmp_path / "raw", tmp_path / "pred"
    predictions.mkdir()
    write_cloud(predictions / "tile.las", [0, .1], [7, 9])
    report = strict_remap(collections=[predictions], originals=originals, output=tmp_path / "enriched")
    assert report["state"] == "validated"
    return tmp_path / "enriched" / base.name


@pytest.mark.parametrize("case", CASES)
def test_remap_member_keeps_the_source_records(tmp_path, case):
    (tmp_path / "raw").mkdir()
    base = tmp_path / "raw" / "base.las"
    write_base(base, CASES[case])
    member = enrich(tmp_path, base)
    assert payloads(member) == payloads(base)
    assert [data for _, _, data in payloads(member)[0]] == list(CASES[case].values())
    with laspy.open(base) as a, laspy.open(member) as b:
        assert a.header.parse_crs() == b.header.parse_crs()  # the sidecar check's comparison


def test_extended_records_stay_extended(tmp_path):
    (tmp_path / "raw").mkdir()
    base = tmp_path / "raw" / "base.las"
    write_base(base, {}, evlr=record("3827_epsg3067.2112.bin"))
    assert payloads(enrich(tmp_path, base)) == payloads(base)


def test_full_resolution_merge_keeps_the_shared_record(tmp_path):
    first, second = tmp_path / "a.las", tmp_path / "b.las"
    for path in (first, second):
        write_base(path, CASES["3827_keys_and_wkt"])
    output = tmp_path / "merged.laz"
    create_full_resolution_product([first, second], output, "laz", original_point_index=False)
    assert payloads(output) == payloads(first)
    assert validate_single_crs_record(first, output)[0]


def test_a_changed_record_is_written_as_changed(tmp_path):
    base = tmp_path / "base.las"
    write_base(base, {2112: record("3805_epsg6678.2112.bin") + b"\0" * 7})
    with laspy.open(base) as reader:
        header = copy_single_source_header(reader.header)
    wkt = header.vlrs.get("WktCoordinateSystemVlr")[0]
    wkt.string = "changed"
    assert wkt.record_data_bytes() == b"changed\0"


def test_unchanged_check_rejects_a_rewritten_record(tmp_path):
    base, rewritten = tmp_path / "base.las", tmp_path / "rewritten.las"
    write_base(base, {2112: record("3805_epsg6678.2112.bin")})
    write_base(rewritten, {2112: CRS.from_epsg(6678).to_wkt("WKT1_GDAL").encode() + b"\0"})
    with pytest.raises(ValueError, match="CRS records differ"):
        validate_crs_records_unchanged(base, rewritten)


def test_unchanged_check_rejects_a_changed_wkt_flag(tmp_path):
    base, changed = tmp_path / "base.las", tmp_path / "flag.las"
    write_base(base, {2112: record("3805_epsg6678.2112.bin")})
    write_base(changed, {2112: record("3805_epsg6678.2112.bin")}, wkt_flag=False)
    with pytest.raises(ValueError, match="WKT flag differs"):
        validate_crs_records_unchanged(base, changed)
