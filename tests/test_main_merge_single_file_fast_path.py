import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import laspy
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from main_merge import run_merge  # noqa: E402


def _write_segmented_laz(path: Path) -> None:
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.offsets = np.array([0.0, 0.0, 0.0])
    header.scales = np.array([0.01, 0.01, 0.01])
    las = laspy.LasData(header)
    las.x = np.array([0.0, 1.0, 2.0])
    las.y = np.array([0.0, 1.0, 2.0])
    las.z = np.array([0.0, 0.0, 0.0])
    las.add_extra_dims(
        [
            laspy.ExtraBytesParams(name="PredInstance_SAT", type=np.uint32),
            laspy.ExtraBytesParams(name="PredSemantic_SAT", type=np.uint16),
        ]
    )
    las.PredInstance_SAT = np.array([1, 1, 2], dtype=np.uint32)
    las.PredSemantic_SAT = np.array([5, 5, 7], dtype=np.uint16)
    las.write(str(path), do_compress=True, laz_backend=laspy.LazBackend.LazrsParallel)


class MainMergeSingleFileFastPathTests(unittest.TestCase):
    def test_single_target_resolution_file_skips_core_dedup_merge(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            segmented_dir = root / "segmented"
            original_tiles_dir = root / "merge_work"
            output_tiles_dir = root / "output_tiles"
            output_merged = root / "merged_sat.laz"
            tile_bounds_json = root / "tile_bounds_tindex.json"
            segmented_dir.mkdir()
            original_tiles_dir.mkdir()
            tile_bounds_json.write_text('{"tile_buffer": 20.0, "tiles": []}', encoding="utf-8")
            source = segmented_dir / "c00_r00_segmented_filtered.laz"
            _write_segmented_laz(source)

            with mock.patch("main_merge.core_merge_tiles") as core_merge:
                result = run_merge(
                    segmented_dir=segmented_dir,
                    output_tiles_dir=output_tiles_dir,
                    original_tiles_dir=original_tiles_dir,
                    tile_bounds_json=tile_bounds_json,
                    output_merged=output_merged,
                    instance_dimension="PredInstance_SAT",
                    transfer_original_dims_to_merged=False,
                )

            core_merge.assert_not_called()
            self.assertEqual(result, output_merged)
            self.assertTrue(output_merged.exists())
            self.assertTrue((output_tiles_dir / source.name).exists())
            metadata_csv = root / "merged_sat_instance_metadata.csv"
            self.assertTrue(metadata_csv.exists())
            self.assertIn("PredInstance_SAT,has_added_clusters", metadata_csv.read_text(encoding="utf-8"))

            with laspy.open(str(output_merged), laz_backend=laspy.LazBackend.LazrsParallel) as reader:
                self.assertEqual(reader.header.point_count, 3)
                self.assertIn("PredInstance_SAT", reader.header.point_format.extra_dimension_names)
                self.assertIn("PredSemantic_SAT", reader.header.point_format.extra_dimension_names)


if __name__ == "__main__":
    unittest.main()
