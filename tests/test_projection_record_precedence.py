"""Every output carries exactly one standardized CRS record (3DT-2200 class)."""
import laspy
import numpy as np
from laspy.vlrs.known import WktCoordinateSystemVlr
from laspy.vlrs.vlrlist import VLRList
from pyproj import CRS
from crs_records import projection_record_list, standardized_wkt_bytes, validate_single_crs_record
from point_cloud_metadata import copy_single_source_header

STANDARD = CRS.from_epsg(25833).to_wkt("WKT1_GDAL")
ORIGINAL = STANDARD.replace('"ETRS89 / UTM zone 33N"', '"ETRS89 / UTM zone 33N(Meter)"', 1)
UNKNOWN = CRS.from_epsg(32632).to_wkt("WKT1_GDAL").replace('"WGS 84 / UTM zone 32N"', '"unknown"', 1)


def write_cloud(path, wkt, extended_wkt=None):
    header = laspy.LasHeader(point_format=6, version='1.4')
    header.global_encoding.wkt = True
    header.vlrs.append(WktCoordinateSystemVlr(wkt))
    if extended_wkt is not None:
        header.evlrs = VLRList([WktCoordinateSystemVlr(extended_wkt)])
    cloud = laspy.LasData(header)
    cloud.x, cloud.y, cloud.z = [437807], [5695260], [200]
    cloud.write(path)


def test_standardized_text_matches_the_pdal_writer():
    # PDAL writes the identified EPSG definition as WKT1_GDAL, NUL-terminated.
    assert standardized_wkt_bytes(ORIGINAL) == STANDARD.encode() + b"\0"
    assert standardized_wkt_bytes(UNKNOWN) == CRS.from_epsg(32632).to_wkt("WKT1_GDAL").encode() + b"\0"
    assert standardized_wkt_bytes(STANDARD.encode() + b"\0") == STANDARD.encode() + b"\0"


def test_unidentified_compound_crs_keeps_its_text():
    compound = CRS.from_epsg(25833).to_wkt("WKT1_GDAL")
    compound = f'COMPD_CS["ETRS89 / UTM zone 33N + local height",{compound},VERT_CS["local height",VERT_DATUM["local",2005],UNIT["metre",1],AXIS["Up",UP]]]'
    assert standardized_wkt_bytes(compound) == compound.encode() + b"\0"


def test_copied_header_has_one_standardized_record(tmp_path):
    source = tmp_path / 'source.laz'
    write_cloud(source, STANDARD, ORIGINAL)       # the duplicate state of v2.3 outputs
    with laspy.open(source) as reader:
        header = copy_single_source_header(reader.header)
    records = projection_record_list(header)
    assert [(kind, rid) for kind, rid, _ in records] == [('vlr', 2112)]
    assert records[0][2] == STANDARD.encode() + b"\0"
    assert header.global_encoding.wkt


def test_upload_spelling_is_standardized_on_copy(tmp_path):
    source, target = tmp_path / 'source.laz', tmp_path / 'target.laz'
    write_cloud(source, ORIGINAL)
    with laspy.open(source) as reader:
        header = copy_single_source_header(reader.header)
    cloud = laspy.LasData(header)
    cloud.x, cloud.y, cloud.z = [437807], [5695260], [200]
    cloud.write(target)
    with laspy.open(target) as reader:
        records = projection_record_list(reader.header)
    assert records == [('vlr', 2112, STANDARD.encode() + b"\0")]
    assert validate_single_crs_record(source, target)[0]


def test_validation_is_read_only(tmp_path):
    source, target = tmp_path / 'source.laz', tmp_path / 'target.laz'
    write_cloud(source, ORIGINAL)
    write_cloud(target, STANDARD)
    before = target.read_bytes()
    for _ in range(2):
        ok, message = validate_single_crs_record(source, target)
        assert ok, message
    assert target.read_bytes() == before


def test_validator_rejects_duplicates_and_changed_crs(tmp_path):
    source, dual, changed = tmp_path / 'source.laz', tmp_path / 'dual.laz', tmp_path / 'changed.laz'
    write_cloud(source, ORIGINAL)
    write_cloud(dual, STANDARD, ORIGINAL)
    write_cloud(changed, CRS.from_epsg(25832).to_wkt("WKT1_GDAL"))
    ok, message = validate_single_crs_record(source, dual)
    assert not ok and "exactly one standardized WKT record" in message
    ok, message = validate_single_crs_record(source, changed)
    assert not ok and "differs from source CRS" in message


def test_empty_placeholder_wkt_counts_as_no_crs(tmp_path):
    # untwine writes a 1-byte NUL WKT VLR when the source has no CRS.
    source, target = tmp_path / 'source.laz', tmp_path / 'target.laz'
    header = laspy.LasHeader(point_format=6, version='1.4')
    header.vlrs.append(laspy.VLR(user_id="LASF_Projection", record_id=2112, record_data=b"\0"))
    cloud = laspy.LasData(header)
    cloud.x, cloud.y, cloud.z = [1.0], [2.0], [3.0]
    cloud.write(source)
    plain = laspy.LasData(laspy.LasHeader(point_format=6, version='1.4'))
    plain.x, plain.y, plain.z = [1.0], [2.0], [3.0]
    plain.write(target)
    ok, message = validate_single_crs_record(source, target)
    assert ok and "no readable CRS" in message
    with laspy.open(source) as reader:
        assert copy_single_source_header(reader.header).parse_crs() is None
