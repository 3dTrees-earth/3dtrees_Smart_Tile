import sys
import tempfile
import unittest
from pathlib import Path

import laspy
import numpy as np
from laspy.vlrs.vlr import VLR
from laspy.vlrs.vlrlist import VLRList


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pyproj import CRS  # noqa: E402

from copc_metadata import crs_authority_string, crs_equivalent  # noqa: E402
from crs_records import projection_record_list, validate_single_crs_record  # noqa: E402

# Same CRS, two serializations: PDAL's standardized EPSG text and a
# non-canonical upload spelling as in dataset 3433.
_STANDARD_25833 = CRS.from_epsg(25833).to_wkt("WKT1_GDAL")
_ORIGINAL_25833 = _STANDARD_25833.replace('"ETRS89 / UTM zone 33N"', '"ETRS89 / UTM zone 33N(Meter)"', 1)


class _FakeCrs:
    def __init__(self, authority=None, epsg=None, wkt="FAKE_WKT", equals_result=None):
        self.authority = authority
        self.epsg = epsg
        self.wkt = wkt
        self.equals_result = equals_result

    def to_authority(self):
        return self.authority

    def to_epsg(self):
        return self.epsg

    def equals(self, other, ignore_axis_order=False):
        return self.equals_result if self.equals_result is not None else False

    def to_wkt(self):
        return self.wkt


class _FakeHeader:
    def __init__(self, crs):
        self.crs = crs

    def parse_crs(self):
        return self.crs


def _write_las(path: Path, projection_payload: bytes | None, record_id: int = 2112,
               evlr_payload: bytes | None = None) -> None:
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.scales = np.array([0.01, 0.01, 0.01])
    header.offsets = np.array([0.0, 0.0, 0.0])
    if projection_payload is not None:
        header.vlrs.append(
            VLR(
                user_id="LASF_Projection",
                record_id=record_id,
                description="synthetic CRS metadata",
                record_data=projection_payload,
            )
        )

    if evlr_payload is not None:
        header.evlrs = VLRList([VLR(user_id="LASF_Projection", record_id=2112,
                                    description="appended original", record_data=evlr_payload)])
    las = laspy.LasData(header)
    las.x = np.array([0.0])
    las.y = np.array([0.0])
    las.z = np.array([0.0])
    las.write(path)


class CopcCrsValidationTests(unittest.TestCase):
    def test_crs_authority_string_prefers_authority_code(self):
        header = _FakeHeader(_FakeCrs(authority=("EPSG", "32632"), epsg=32633))

        self.assertEqual(crs_authority_string(header), "EPSG:32632")

    def test_crs_authority_string_falls_back_to_epsg_code(self):
        header = _FakeHeader(_FakeCrs(authority=None, epsg=32632))

        self.assertEqual(crs_authority_string(header), "EPSG:32632")

    def test_crs_equivalent_accepts_same_authority_with_different_wkt(self):
        source = _FakeCrs(authority=("EPSG", "4978"), wkt="LONG_WKT")
        output = _FakeCrs(authority=("EPSG", "4978"), wkt="SHORT_WKT")

        self.assertTrue(crs_equivalent(source, output))

    def test_crs_equivalent_accepts_pyproj_equals_match(self):
        source = _FakeCrs(wkt="LONG_WKT", equals_result=True)
        output = _FakeCrs(wkt="SHORT_WKT")

        self.assertTrue(crs_equivalent(source, output))

    def test_accepts_matching_projection_vlr(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            source = tmp_path / "source.las"
            output = tmp_path / "output.las"
            payload = b'LOCAL_CS["3dtrees-test"]'
            _write_las(source, payload)
            _write_las(output, payload)

            ok, message = validate_single_crs_record(source, output)

            self.assertTrue(ok, message)

    def test_rejects_missing_projection_vlr(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            source = tmp_path / "source.las"
            output = tmp_path / "output.las"
            _write_las(source, b"\x01\x00\x01\x00\x00\x00\x01\x00\x00\x00\x04\x00\x01\x00r\x13", record_id=34735)
            _write_las(output, None)

            ok, message = validate_single_crs_record(source, output)

            self.assertFalse(ok)
            self.assertIn("missing", message)

    def test_accepts_standardized_text_of_the_same_crs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            source = tmp_path / "source.las"
            output = tmp_path / "output.las"
            _write_las(source, _ORIGINAL_25833.encode(), record_id=2112)
            _write_las(output, _STANDARD_25833.encode(), record_id=2112)

            ok, message = validate_single_crs_record(source, output)

            self.assertTrue(ok, message)

    def test_rejects_two_crs_records_in_one_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            source = tmp_path / "source.las"
            output = tmp_path / "output.las"
            _write_las(source, _ORIGINAL_25833.encode(), record_id=2112)
            _write_las(output, _STANDARD_25833.encode(), record_id=2112, evlr_payload=_ORIGINAL_25833.encode())

            ok, message = validate_single_crs_record(source, output)

            self.assertFalse(ok)
            self.assertIn("exactly one standardized WKT record", message)

    def test_rejects_changed_projection_vlr(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            source = tmp_path / "source.las"
            output = tmp_path / "output.las"
            _write_las(source, _STANDARD_25833.encode())
            _write_las(output, CRS.from_epsg(25832).to_wkt("WKT1_GDAL").encode())

            ok, message = validate_single_crs_record(source, output)

            self.assertFalse(ok)
            self.assertIn("differs from source CRS", message)

    def test_accepts_source_without_crs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            source = tmp_path / "source.las"
            output = tmp_path / "output.las"
            _write_las(source, None)
            _write_las(output, None)

            ok, message = validate_single_crs_record(source, output)

            self.assertTrue(ok, message)


if __name__ == "__main__":
    unittest.main()
