"""Repeated product conversion must preserve the effective source WKT (3DT-2200)."""
import laspy
import numpy as np
import pytest
from laspy.vlrs.known import WktCoordinateSystemVlr
from laspy.vlrs.vlrlist import VLRList
from pyproj import CRS
from copc_metadata import (
    append_source_geotiff_projection_evlrs, copc_preserves_source_crs,
    projection_vlr_fingerprints,
)


def write_cloud(path, wkt, extended_wkt=None):
    header = laspy.LasHeader(point_format=6, version='1.4')
    header.global_encoding.wkt = True
    header.vlrs.append(WktCoordinateSystemVlr(wkt))
    if extended_wkt is not None:
        header.evlrs = VLRList([WktCoordinateSystemVlr(extended_wkt)])
    cloud = laspy.LasData(header)
    cloud.x, cloud.y, cloud.z = [437807], [5695260], [200]
    cloud.write(path)


@pytest.mark.parametrize('output_already_has_original', [True, False])
def test_source_evlr_precedence_survives_repeated_preservation(tmp_path, output_already_has_original):
    normalized = CRS.from_epsg(25833).to_wkt()
    original = normalized.replace('ETRS89 / UTM zone 33N', 'ETRS89 / UTM zone 33N(Meter)', 1)
    assert CRS.from_wkt(normalized).equals(CRS.from_wkt(original))
    source, target = tmp_path/'source.laz', tmp_path/'target.laz'
    write_cloud(source, normalized, original)
    write_cloud(target, original if output_already_has_original else normalized)
    before = laspy.read(target).points.array.copy()
    for _ in range(2):
        ok, message = append_source_geotiff_projection_evlrs(source, target)
        assert ok, message
        ok, message = copc_preserves_source_crs(source, target)
        assert ok, message
        with laspy.open(source) as a, laspy.open(target) as b:
            assert projection_vlr_fingerprints(a.header) == projection_vlr_fingerprints(b.header)
            assert b.header.number_of_evlrs == (0 if output_already_has_original else 1)
        np.testing.assert_array_equal(laspy.read(target).points.array, before)


def test_validator_still_rejects_a_changed_crs(tmp_path):
    source, target = tmp_path/'source.laz', tmp_path/'target.laz'
    write_cloud(source, CRS.from_epsg(25833).to_wkt())
    write_cloud(target, CRS.from_epsg(25832).to_wkt())
    assert not copc_preserves_source_crs(source, target)[0]
