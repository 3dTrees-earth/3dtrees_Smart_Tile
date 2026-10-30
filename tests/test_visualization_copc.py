import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import laspy  # noqa: E402
from laspy.copc import CopcReader  # noqa: E402

from visualization_copc import assign_nodes, octree_cube, write_visualization_copc  # noqa: E402


def _write_cloud(path: Path, n: int = 60_000, point_format: int = 1) -> laspy.LasData:
    rng = np.random.default_rng(3)
    header = laspy.LasHeader(version="1.4" if point_format >= 6 else "1.2", point_format=point_format)
    header.scales = [0.001] * 3
    header.offsets = [1000.0, 2000.0, 0.0]
    header.add_extra_dim(laspy.ExtraBytesParams(name="PredInstance_SAT", type=np.uint16))
    header.add_extra_dim(laspy.ExtraBytesParams(name="species_prob_SAT", type=np.float32))
    las = laspy.LasData(header)
    xyz = rng.uniform([1000, 2000, 0], [1030, 2030, 20], size=(n, 3))
    las.x, las.y, las.z = xyz.T
    las.intensity = rng.integers(0, 4000, n)
    las.return_number = np.ones(n, dtype=np.uint8)
    las.number_of_returns = np.ones(n, dtype=np.uint8)
    las.gps_time = np.sort(rng.uniform(0, 100, n))
    las.PredInstance_SAT = (np.floor(xyz[:, 0] - 1000) // 3).astype(np.uint16)  # ten instances, 0 = ground
    las.species_prob_SAT = rng.uniform(0, 1, n).astype(np.float32)
    las.write(str(path))
    return las


class AssignNodesTests(unittest.TestCase):
    def test_every_point_gets_one_node_inside_its_cube(self):
        xyz = np.random.default_rng(0).uniform(0, 10, size=(20_000, 3))
        cube_min, size = octree_cube(xyz.min(0), xyz.max(0))
        level, keys = assign_nodes(xyz, cube_min, size, grid=16, leaf_points=500)
        self.assertTrue((level >= 0).all())
        cell = size / (1 << level.astype(np.int64))
        lower = cube_min + keys * cell[:, None]
        self.assertTrue(((xyz >= lower - 1e-9) & (xyz <= lower + cell[:, None] + 1e-9)).all())

    def test_upper_levels_hold_at_most_one_point_per_voxel(self):
        xyz = np.random.default_rng(1).uniform(0, 10, size=(50_000, 3))
        cube_min, size = octree_cube(xyz.min(0), xyz.max(0))
        level, _ = assign_nodes(xyz, cube_min, size, grid=8, leaf_points=1_000)
        root = xyz[level == 0]
        voxels = np.floor((root - cube_min) / size * 8).astype(int)
        self.assertEqual(len(np.unique(voxels, axis=0)), len(root))
        self.assertLessEqual(len(root), 8 ** 3)


class WriteVisualizationCopcTests(unittest.TestCase):
    def test_writes_lossless_copc_with_coloring_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = _write_cloud(tmp / "plot.laz")
            out = tmp / "plot.copc.laz"
            report = write_visualization_copc(tmp / "plot.laz", out, grid=16, leaf_points=2_000, workers=2)

            result = laspy.read(str(out))
            self.assertEqual(result.point_format.id, 6)  # format 1 is upgraded for COPC
            self.assertEqual(len(result.points), len(source.points))
            for dim in ("X", "Y", "Z", "intensity", "gps_time", "PredInstance_SAT", "species_prob_SAT"):
                np.testing.assert_array_equal(np.sort(np.asarray(result[dim])), np.sort(np.asarray(source[dim])))

            ids = np.asarray(result.PredInstance_SAT)
            colours = np.asarray(result.coloring_id_sat)
            self.assertTrue((colours[ids == 0] == 0).all())
            self.assertTrue((colours[ids > 0] > 0).all())
            for instance in np.unique(ids):
                self.assertEqual(len(np.unique(colours[ids == instance])), 1)
            mapping = json.loads(Path(report["coloring_mappings"][0]).read_text())
            self.assertEqual(mapping["color_attribute"], "coloring_id_sat")
            self.assertEqual(set(mapping["instance_id_to_coloring_id"]), {str(i) for i in np.unique(ids)})

            with CopcReader.open(str(out)) as reader:
                self.assertEqual(len(reader.query()), len(source.points))
                self.assertEqual(len(reader.query(level=0)), report["root_points"])
            self.assertLessEqual(report["root_points"], 16 ** 3)

    def test_hierarchy_has_no_empty_nodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _write_cloud(tmp / "plot.laz", n=30_000, point_format=6)
            out = tmp / "plot.copc.laz"
            write_visualization_copc(tmp / "plot.laz", out, grid=16, leaf_points=1_000)
            raw = out.read_bytes()
            info = raw[375 + 54:375 + 54 + 160]
            root_offset, root_size = struct.unpack_from("<2Q", info, 40)
            counts = [struct.unpack_from("<4iQii", raw, root_offset + i)[-1] for i in range(0, root_size, 32)]
            self.assertTrue(all(c > 0 for c in counts))
            self.assertEqual(sum(counts), 30_000)

    def test_rejects_unknown_dimension_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _write_cloud(tmp / "plot.laz", n=2_000)
            with self.assertRaises(ValueError):
                write_visualization_copc(tmp / "plot.laz", tmp / "x.copc.laz", dimensions=["nope"])


if __name__ == "__main__":
    unittest.main()
