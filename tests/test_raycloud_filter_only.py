import json
import sys
import tempfile
import unittest
from pathlib import Path

import laspy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from strict_prediction_pipeline import merge_collections
from test_dense_tile_merge import write_cloud


class RayCloudFilterOnlyTests(unittest.TestCase):
    def _fixture(self, root):
        source = root / "rct"
        source.mkdir()
        write_cloud(source / "c00_r00_segmented.las", [1, 2, 10.5], [1, 1, 2],
                    instance="PredInstance_RCT")
        write_cloud(source / "c01_r00_segmented.las", [11, 12, 9.5], [1, 1, 2],
                    instance="PredInstance_RCT")
        for tile in ("c00_r00", "c01_r00"):
            for suffix in ("trees", "trees_info"):
                (source / f"{tile}_{suffix}.txt").write_text(
                    "# RCT tree table\nheader\ntree-1\ntree-2\n", encoding="utf-8")
        layout = root / "bounds.json"
        layout.write_text(json.dumps({"tile_buffer": 1, "tiles": [
            {"bounds": [[-1, 11], [-1, 1]], "core": [[-1, 10], [-1, 1]], "col": 0, "row": 0},
            {"bounds": [[9, 21], [-1, 1]], "core": [[10, 21], [-1, 1]], "col": 1, "row": 0},
        ]}))
        return source, layout

    def test_keeps_original_tile_ids_and_filters_both_tree_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, layout = self._fixture(root)
            report = merge_collections(
                collections=[source], target_dir=None, output_tiles=root / "output_tiles",
                tile_bounds_json=layout, ready=True, instance_dimension="PredInstance_RCT")
            self.assertIn("filter-only", report["contract"])
            self.assertEqual(report["models"][0]["reconciliation"]["accepted_pairs"], [])
            self.assertEqual((root / "output_tiles" / "instance_metadata.csv").read_text().splitlines(),
                             ["tile,PredInstance_RCT,has_added_clusters", "0,1,0", "1,1,0"])
            for tile in range(2):
                output = laspy.read(root / "output_tiles" / f"tile_{tile:05d}.laz")
                self.assertEqual(output.PredInstance_RCT.tolist(), [1, 1])
                for suffix in ("trees", "trees_info"):
                    sidecar = root / "segmented_filtered" / f"c{tile:02d}_r00_filtered_{suffix}.txt"
                    self.assertEqual(sidecar.read_text(),
                                     "# RCT tree table\npredinstance,header\n1,tree-1\n")

    def test_rejects_missing_tree_file_before_publishing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, layout = self._fixture(root)
            (source / "c01_r00_trees.txt").unlink()
            (source / "c01_r00_trees_info.txt").unlink()
            with self.assertRaisesRegex(ValueError, "Missing RayCloudTools tree or treeinfo file"):
                merge_collections(collections=[source], target_dir=None,
                                  output_tiles=root / "output_tiles", tile_bounds_json=layout,
                                  ready=True, instance_dimension="PredInstance_RCT")
            self.assertFalse((root / "output_tiles").exists())

    def test_rejects_rct_dimension_without_sidecars(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, layout = self._fixture(root)
            for sidecar in source.glob("*.txt"):
                sidecar.unlink()
            with self.assertRaisesRegex(ValueError, "require matching _trees.txt and _trees_info.txt"):
                merge_collections(collections=[source], target_dir=None,
                                  output_tiles=root / "output_tiles", tile_bounds_json=layout,
                                  ready=True, instance_dimension="PredInstance_RCT")
            self.assertFalse((root / "output_tiles").exists())

    def test_rejects_missing_treeinfo(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, layout = self._fixture(root)
            (source / "c00_r00_trees_info.txt").unlink()
            with self.assertRaisesRegex(ValueError, "Missing RayCloudTools tree or treeinfo file"):
                merge_collections(collections=[source], target_dir=None,
                                  output_tiles=root / "output_tiles", tile_bounds_json=layout,
                                  ready=True, instance_dimension="PredInstance_RCT")

    def test_rejects_treeinfo_row_count_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, layout = self._fixture(root)
            with (source / "c00_r00_trees_info.txt").open("a") as stream:
                stream.write("orphan-row\n")
            with self.assertRaisesRegex(ValueError, "row counts differ"):
                merge_collections(collections=[source], target_dir=None,
                                  output_tiles=root / "output_tiles", tile_bounds_json=layout,
                                  ready=True, instance_dimension="PredInstance_RCT")
            self.assertFalse((root / "output_tiles").exists())

    def test_rejects_merged_product_with_tile_local_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, layout = self._fixture(root)
            with self.assertRaisesRegex(ValueError, "tile-local"):
                merge_collections(collections=[source], target_dir=None,
                                  output_tiles=root / "output_tiles", tile_bounds_json=layout,
                                  ready=True, instance_dimension="PredInstance_RCT",
                                  merged_output=root / "merged.laz")


if __name__ == "__main__":
    unittest.main()
