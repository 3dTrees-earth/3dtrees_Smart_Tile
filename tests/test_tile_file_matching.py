import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from main_remap import _match_files_via_json, _remap_point_cloud_files, find_matching_files  # noqa: E402


class MainRemapFileDiscoveryTests(unittest.TestCase):
    def test_remap_file_discovery_includes_mixed_laz_and_las(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "tile_a.laz").write_text("placeholder")
            (root / "tile_b.las").write_text("placeholder")

            files = [path.name for path in _remap_point_cloud_files(root)]

        self.assertEqual(files, ["tile_a.laz", "tile_b.las"])

    def test_remap_file_discovery_prefers_copc_twin(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "tile_a.laz").write_text("placeholder")
            (root / "tile_a.copc.laz").write_text("placeholder")

            files = [path.name for path in _remap_point_cloud_files(root)]

        self.assertEqual(files, ["tile_a.copc.laz"])

    def test_spatial_matching_considers_mixed_laz_and_las(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            target = root / "target"
            source.mkdir()
            target.mkdir()
            source_laz = source / "c00_r00_segmented.laz"
            source_las = source / "c01_r00_segmented.las"
            target_laz = target / "c00_r00.laz"
            target_las = target / "c01_r00.las"
            for path in (source_laz, source_las, target_laz, target_las):
                path.write_text("placeholder")

            bounds = {
                source_laz: (0.0, 10.0, 0.0, 10.0),
                target_laz: (0.0, 10.0, 0.0, 10.0),
                source_las: (20.0, 30.0, 0.0, 10.0),
                target_las: (20.0, 30.0, 0.0, 10.0),
            }

            import main_remap

            original_get_file_bounds = main_remap.get_file_bounds
            try:
                main_remap.get_file_bounds = bounds.__getitem__
                matches = find_matching_files(source, target)
            finally:
                main_remap.get_file_bounds = original_get_file_bounds

        self.assertEqual(
            [(src.name, tgt.name) for src, tgt, _ in matches],
            [
                ("c00_r00_segmented.laz", "c00_r00.laz"),
                ("c01_r00_segmented.las", "c01_r00.las"),
            ],
        )

    def test_json_matching_considers_mixed_laz_and_las(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            target = root / "target"
            source.mkdir()
            target.mkdir()
            source_laz = source / "c00_r00_segmented.laz"
            source_las = source / "c01_r00_segmented.las"
            target_laz = target / "c00_r00.laz"
            target_las = target / "c01_r00.las"
            for path in (source_laz, source_las, target_laz, target_las):
                path.write_text("placeholder")

            bounds = {
                source_laz: (0.0, 10.0, 0.0, 10.0),
                target_laz: (0.0, 10.0, 0.0, 10.0),
                source_las: (20.0, 30.0, 0.0, 10.0),
                target_las: (20.0, 30.0, 0.0, 10.0),
            }

            import main_remap

            original_get_file_bounds = main_remap.get_file_bounds
            original_build = main_remap.build_neighbor_graph_from_bounds_json
            try:
                main_remap.get_file_bounds = bounds.__getitem__
                main_remap.build_neighbor_graph_from_bounds_json = lambda *_: (
                    [
                        (0.0, 10.0, 0.0, 10.0),
                        (20.0, 30.0, 0.0, 10.0),
                    ],
                    [(5.0, 5.0), (25.0, 5.0)],
                    {},
                )
                matches = _match_files_via_json(root / "tile_bounds_tindex.json", source, target)
            finally:
                main_remap.get_file_bounds = original_get_file_bounds
                main_remap.build_neighbor_graph_from_bounds_json = original_build

        self.assertEqual(
            [(src.name, tgt.name) for src, tgt, _ in matches],
            [
                ("c00_r00_segmented.laz", "c00_r00.laz"),
                ("c01_r00_segmented.las", "c01_r00.las"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
