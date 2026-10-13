from pathlib import Path
import sys
import unittest

import laspy
from laspy.vlrs.known import WktCoordinateSystemVlr
from laspy.vlrs.vlrlist import VLRList
from pyproj import CRS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from subsample_com import make_center_of_mass_header


class SubsampleEvlrTests(unittest.TestCase):
    def source(self):
        header = laspy.LasHeader(point_format=6, version="1.4")
        header.vlrs.append(WktCoordinateSystemVlr(""))
        header.evlrs = VLRList([
            WktCoordinateSystemVlr(CRS.from_user_input("EPSG:25832+7837").to_wkt()),
            laspy.VLR(user_id="copc", record_id=1000, record_data=b"obsolete hierarchy"),
        ])
        data = laspy.LasData(header)
        data.x = [.1, .2]
        data.y = [.1, .2]
        data.z = [.1, .2]
        data.update_header()
        return data

    def test_reduced_header_preserves_extended_projection_but_drops_copc_hierarchy(self):
        source = self.source()
        header = make_center_of_mass_header(source.header, True)
        self.assertEqual(str(header.version), "1.4")
        self.assertEqual(header.point_format.id, 0)
        # The effective extended CRS becomes the single standardized WKT VLR.
        self.assertEqual([v.record_id for v in header.vlrs if v.user_id == "LASF_Projection"], [2112])
        self.assertFalse(any(v.user_id == "LASF_Projection" for v in header.evlrs))
        self.assertTrue(header.parse_crs().equals(source.header.parse_crs()))
