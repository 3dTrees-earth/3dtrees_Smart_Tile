import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import laspy
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import filter_buffer_instances as filter_mod  # noqa: E402


class FilterBufferInstancesTests(unittest.TestCase):
    def test_get_tile_neighbors_treats_non_grid_names_as_edge_tiles(self):
        self.assertEqual(
            filter_mod.get_tile_neighbors("single_tile", ["single_tile"]),
            {"east": False, "west": False, "north": False, "south": False},
        )

    def test_filter_directory_processes_laz_and_las_with_instance_dimension(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            (input_dir / "c00_r00_segmented.laz").touch()
            (input_dir / "c01_r00_segmented.las").touch()

            def fake_process(input_file, output_file, all_tile_names, buffer, instance_dimension):
                output_file.parent.mkdir(parents=True, exist_ok=True)
                output_file.touch()
                return (100, 10, 1)

            with mock.patch.object(filter_mod, "process_tile", side_effect=fake_process) as process:
                summary = filter_mod.filter_buffer_instances_dir(
                    input_dir=input_dir,
                    output_dir=output_dir,
                    buffer=20.0,
                    suffix="_clean",
                    instance_dimension="ModelInstance",
                )

            self.assertEqual(summary["input_files"], 2)
            self.assertEqual(summary["total_original"], 200)
            self.assertEqual(summary["total_removed"], 20)
            self.assertEqual(summary["total_instances_removed"], 2)
            self.assertEqual(
                [call.args[0].name for call in process.call_args_list],
                ["c00_r00_segmented.laz", "c01_r00_segmented.las"],
            )
            self.assertEqual(
                process.call_args_list[0].args[2],
                ["c00_r00", "c01_r00"],
            )
            self.assertEqual(process.call_args_list[0].args[3], 20.0)
            self.assertEqual(process.call_args_list[0].kwargs["instance_dimension"], "ModelInstance")
            self.assertTrue((output_dir / "c00_r00_segmented_clean.laz").exists())
            self.assertTrue((output_dir / "c01_r00_segmented_clean.las").exists())

    def test_filter_directory_can_force_laz_output_extension(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            (input_dir / "c00_r00_segmented.las").touch()

            def fake_process(input_file, output_file, all_tile_names, buffer, instance_dimension):
                output_file.parent.mkdir(parents=True, exist_ok=True)
                output_file.touch()
                return (1, 0, 0)

            with mock.patch.object(filter_mod, "process_tile", side_effect=fake_process):
                summary = filter_mod.filter_buffer_instances_dir(
                    input_dir=input_dir,
                    output_dir=output_dir,
                    output_extension=".laz",
                )

            self.assertEqual(summary["input_files"], 1)
            self.assertTrue((output_dir / "c00_r00_segmented_filtered.laz").exists())
            self.assertFalse((output_dir / "c00_r00_segmented_filtered.las").exists())

    def test_filter_directory_copies_tree_sidecars_with_filtered_names(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            (input_dir / "c00_r00_segmented.laz").touch()
            (input_dir / "c00_r00_trees.txt").write_text("tree rows\n", encoding="utf-8")
            (input_dir / "c00_r00_trees_info.txt").write_text("tree info\n", encoding="utf-8")

            def fake_process(input_file, output_file, all_tile_names, buffer, instance_dimension):
                output_file.parent.mkdir(parents=True, exist_ok=True)
                output_file.touch()
                return (1, 0, 0)

            with mock.patch.object(filter_mod, "process_tile", side_effect=fake_process):
                summary = filter_mod.filter_buffer_instances_dir(
                    input_dir=input_dir,
                    output_dir=output_dir,
                    output_extension=".laz",
                )

            self.assertEqual([path.name for path in summary["tree_files"]], [
                "c00_r00_trees.txt",
                "c00_r00_trees_info.txt",
            ])
            self.assertEqual([path.name for path in summary["tree_output_files"]], [
                "c00_r00_filtered_trees.txt",
                "c00_r00_filtered_trees_info.txt",
            ])
            self.assertEqual(
                (output_dir / "c00_r00_filtered_trees.txt").read_text(encoding="utf-8"),
                "tree rows\n",
            )

    def test_filter_directory_rejects_copc_output_extension(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "input"
            input_dir.mkdir()

            with self.assertRaises(ValueError):
                filter_mod.filter_buffer_instances_dir(
                    input_dir=input_dir,
                    output_dir=root / "output",
                    output_extension=".copc.laz",
                )

    def test_copy_path_strips_stale_copc_vlrs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_file = root / "single.copc.laz"
            output_file = root / "single_filtered.laz"

            header = laspy.LasHeader(point_format=3, version="1.2")
            header.vlrs.append(
                laspy.VLR(
                    user_id="copc",
                    record_id=1,
                    description="COPC info VLR",
                    record_data=b"stale",
                )
            )
            las = laspy.LasData(header)
            las.x = np.array([0.0, 1.0])
            las.y = np.array([0.0, 1.0])
            las.z = np.array([0.0, 1.0])
            las.write(str(input_file), do_compress=True, laz_backend=laspy.LazBackend.LazrsParallel)

            original, removed, removed_instances = filter_mod.process_tile(
                input_file,
                output_file,
                all_tile_names=["single"],
                instance_dimension="MissingInstance",
            )

            self.assertEqual((original, removed, removed_instances), (2, 0, 0))
            with laspy.open(str(output_file), laz_backend=laspy.LazBackend.LazrsParallel) as reader:
                stale_vlrs = [
                    vlr for vlr in reader.header.vlrs
                    if getattr(vlr, "user_id", "") == "copc"
                ]
                self.assertEqual(stale_vlrs, [])
                self.assertEqual(reader.header.point_count, 2)


if __name__ == "__main__":
    unittest.main()
