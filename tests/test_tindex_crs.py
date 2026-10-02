import json
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import laspy
from pyproj import CRS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
# Some older discovery tests leave a tile_tindex stub in sys.modules.
# Load the implementation independently so this regression tests real code.
spec = importlib.util.spec_from_file_location(
    "tile_tindex_crs_under_test", Path(__file__).resolve().parents[1] / "src/tile_tindex.py")
tile_tindex = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tile_tindex)
_common_header_srs = tile_tindex._common_header_srs
build_tindex = tile_tindex.build_tindex


class TindexCrsTests(unittest.TestCase):
    @staticmethod
    def cloud(path, epsg):
        header = laspy.LasHeader(point_format=3, version="1.2")
        header.add_crs(CRS.from_epsg(epsg))
        laspy.LasData(header).write(path)

    def test_missing_pdal_crs_uses_verified_common_header_crs_in_both_cli_paths(self):
        for legacy_cli in (False, True):
            with self.subTest(legacy_cli=legacy_cli), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.cloud(root / "a.las", 2056)
                self.cloud(root / "b.las", 2056)
                index_commands = []

                def run(command, **kwargs):
                    if "info" in command:
                        return subprocess.CompletedProcess(command, 0, json.dumps({"metadata": {"srs": {"compoundwkt": ""}}}), "")
                    index_commands.append(command)
                    if legacy_cli and "--filelist" in command:
                        return subprocess.CompletedProcess(command, 1, "", "Unexpected argument 'filelist'")
                    Path(command[3]).write_bytes(b"index")
                    return subprocess.CompletedProcess(command, 0, "", "")

                with mock.patch.object(tile_tindex.subprocess, "run", side_effect=run):
                    build_tindex(root, root / "out.gpkg")
                self.assertEqual(len(index_commands), 2 if legacy_cli else 1)
                for command in index_commands:
                    for flag in ("--a_srs=", "--t_srs="):
                        value = next(a[len(flag):] for a in command if a.startswith(flag))
                        self.assertEqual(CRS.from_user_input(value).to_epsg(), 2056)

    def test_differing_headers_do_not_authorize_a_common_assignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.cloud(root / "a.las", 2056)
            self.cloud(root / "b.las", 3857)
            self.assertIsNone(_common_header_srs([root / "a.las", root / "b.las"]))


class SingleCloudLayoutTests(unittest.TestCase):
    def test_subsampling_updates_single_cloud_without_grid_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); outputs = root / "subsampled_res1"; outputs.mkdir()
            source = root / "8851.las"
            cloud = laspy.LasData(laspy.LasHeader(point_format=0, version="1.4"))
            cloud.x = [100, 782]; cloud.y = [200, 772]; cloud.z = [0, 1]
            cloud.write(source)
            layout = root / "bounds.json"
            layout.write_text(json.dumps({"tile_buffer": 20, "tiles": [{"col": i, "row": 0} for i in range(6)]}))
            tile_tindex.write_single_cloud_bounds(layout, source)
            cloud.x = [100.01, 781.99]; cloud.y = [200.01, 771.99]
            cloud.write(outputs / "8851_subsampled_1cm.laz")
            self.assertEqual(tile_tindex.update_tile_bounds_json_from_files(layout, outputs), 1)
            data = json.loads(layout.read_text())
            self.assertEqual(len(data["tiles"]), 1)
            self.assertEqual(data["tiles"][0]["bounds"], [[100.01, 781.99], [200.01, 771.99]])
            self.assertEqual(data["tiles"][0]["core"], data["tiles"][0]["bounds"])
            self.assertEqual(data["tile_buffer"], 0)
