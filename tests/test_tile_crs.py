"""3DT-1898 criterion 1: the tile task rejects CRSs it cannot tile in metres.

Regressions for 3DT-1709/1710 (2935: degree and LV95 files in one upload,
"would create 37,751,670 tiles") and 3DT-1777 (3035: geocentric EPSG:4978).
"""
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import laspy
import numpy as np
from pyproj import CRS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.modules.setdefault("plot_tiles_and_copc", types.SimpleNamespace(plot_extents=lambda *_a, **_k: None))

import main_tile  # noqa: E402
from tile_crs import require_metric_tiling_crs  # noqa: E402


def cloud(path, crs=None, xyz=((2650174.7, 1249627.9, 0.0), (2650204.6, 1249657.8, 40.0)), version="1.4"):
    header = laspy.LasHeader(point_format=6 if version == "1.4" else 3, version=version)
    header.scales = [0.001] * 3
    header.offsets = [min(p[i] for p in xyz) for i in range(3)]
    if crs is not None:
        header.add_crs(CRS.from_user_input(crs))
    las = laspy.LasData(header)
    points = np.asarray(xyz, dtype=float)
    las.x, las.y, las.z = points[:, 0], points[:, 1], points[:, 2]
    las.write(path)
    return path


LONLAT = ((7.90, 47.17, 500.0), (7.91, 47.18, 540.0))
ECEF_3035 = ((4294967.296, 0.0, 0.0), (4295363.156, 183.287, 18.64))


class MetricCrsTests(unittest.TestCase):
    def check(self, *files):
        return require_metric_tiling_crs(files)

    def test_projected_metre_crs_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for crs in ("EPSG:2056", "EPSG:25832", "EPSG:2056+5728"):
                with self.subTest(crs=crs):
                    self.assertIsNotNone(self.check(cloud(root / "a.las", crs)))

    def test_files_without_crs_are_local_metres(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(self.check(cloud(Path(tmp) / "local.las", None, xyz=((0, 0, 0), (50, 50, 20)))))

    def test_geographic_degrees_are_rejected_with_file_and_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            for version in ("1.4", "1.2"):
                with self.subTest(version=version):
                    path = cloud(Path(tmp) / f"plot_{version}.las", "EPSG:4326", LONLAT, version)
                    with self.assertRaisesRegex(ValueError, rf"plot_{version}\.las: EPSG:4326 .*geographic.*degrees.*reproject"):
                        self.check(path)

    def test_geographic_compound_crs_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "geographic"):
                self.check(cloud(Path(tmp) / "a.las", "EPSG:4258+5783", LONLAT))

    def test_geocentric_3035_layout_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for version in ("1.2", "1.4"):
                with self.subTest(version=version):
                    path = cloud(Path(tmp) / f"9451_{version}.las", "EPSG:4978", ECEF_3035, version)
                    with self.assertRaisesRegex(ValueError, r"EPSG:4978 .*geocentric"):
                        self.check(path)

    def test_non_metre_linear_unit_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "US survey foot"):
                self.check(cloud(Path(tmp) / "ny.las", "EPSG:2263", ((980000, 190000, 0), (980100, 190100, 30))))

    def test_degree_file_among_metre_files_is_named(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = [cloud(root / f"{i}.las", "EPSG:2056") for i in range(3)]
            files.append(cloud(root / "9999.las", "EPSG:4326", LONLAT))
            with self.assertRaisesRegex(ValueError, r"9999\.las: EPSG:4326") as caught:
                self.check(*files)
            self.assertNotIn("0.las", str(caught.exception))

    def test_different_metric_crss_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, r"2 different CRSs: EPSG:2056 .*EPSG:25832"):
                self.check(cloud(root / "a.las", "EPSG:2056"),
                           cloud(root / "b.las", "EPSG:25832", ((652550, 5772900, 0), (652600, 5772950, 30))))

    def test_files_with_and_without_crs_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, r"1 input file\(s\) have no CRS \(b.las\) while a.las declares EPSG:2056"):
                self.check(cloud(root / "a.las", "EPSG:2056"), cloud(root / "b.las", None))
            # Several CRS-less files alone remain local metres.
            self.assertIsNone(self.check(cloud(root / "c.las", None), cloud(root / "d.las", None)))

    def test_equivalent_crs_texts_are_one_crs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wkt = CRS.from_epsg(2056).to_wkt("WKT1_GDAL")
            crs = self.check(cloud(root / "a.las", "EPSG:2056"), cloud(root / "b.las", wkt))
            self.assertEqual(crs.to_epsg(), 2056)


class TilePipelineTests(unittest.TestCase):
    def test_tile_task_fails_before_indexing(self):
        """The planned-tile limit must not be the first error (37,751,670 tiles for 2935)."""
        for name, crs, xyz in (("2935", "EPSG:4326", LONLAT), ("3035", "EPSG:4978", ECEF_3035)):
            with self.subTest(dataset=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "in").mkdir()
                cloud(root / "in" / "a.las", "EPSG:2056")
                cloud(root / "in" / f"{name}.las", crs, xyz)
                with mock.patch.object(main_tile, "build_tindex") as tindex, \
                        mock.patch.object(main_tile, "calculate_tile_bounds") as bounds:
                    with self.assertRaisesRegex(ValueError, rf"Unsupported input CRS for tiling: {name}\.las: {crs}"):
                        main_tile.run_tiling_pipeline(root / "in", root / "out", tile_length=300, tile_buffer=20)
                tindex.assert_not_called()
                bounds.assert_not_called()


class CrsLessTileIndexTests(unittest.TestCase):
    """Review regression: CRS-less local-metre uploads must still tile.

    PDAL labels a tile index of CRS-less inputs EPSG:4326 without reprojecting;
    that label must not be read as degree coordinates.
    """

    def test_crs_less_input_tile_index_is_read_as_local_metres(self):
        import shutil
        if shutil.which("pdal") is None:
            self.skipTest("PDAL not installed")
        from get_bounds_from_tindex import load_extent_from_tindex
        from tile_tindex import build_tindex
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "in").mkdir()
            cloud(root / "in" / "local.laz", None, xyz=((0, 0, 0), (50, 40, 20)))
            self.assertIsNone(require_metric_tiling_crs([root / "in" / "local.laz"]))
            bounds, _ = load_extent_from_tindex(build_tindex(root / "in", root / "out" / "tindex.gpkg"))
            np.testing.assert_allclose(bounds, (0, 0, 50, 40), atol=1e-3)


if __name__ == "__main__":
    unittest.main()
