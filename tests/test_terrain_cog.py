import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from osgeo import gdal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from raycloud_meshes import VERTEX
from raycloud_terrain import FACE
from terrain_cog import NODATA, terrain_grid, write_terrain_cog


def write_mesh(path, xyz, faces, crs="EPSG:32632"):
    vertices = np.zeros(len(xyz), dtype=VERTEX)
    vertices["xyz"] = xyz
    vertices["rgba"] = [120, 90, 60, 255]
    records = np.zeros(len(faces), dtype=FACE)
    records["count"] = 3
    records["indices"] = faces
    header = ["ply", "format binary_little_endian 1.0", f"comment crs: {crs}", f"element vertex {len(xyz)}",
              "property double x", "property double y", "property double z", "property uchar red",
              "property uchar green", "property uchar blue", "property uchar alpha", f"element face {len(faces)}",
              "property list int int vertex_indices", "end_header"]
    Path(path).write_bytes("\n".join(header).encode() + b"\n" + vertices.tobytes() + records.tobytes())


def plane(x0, x1, y0, y1, step=0.5):
    """A sloped plane (z = 100 + 0.2 x + 0.1 y) as grid triangles."""
    xs, ys = np.arange(x0, x1 + step / 2, step), np.arange(y0, y1 + step / 2, step)
    x, y = np.meshgrid(xs, ys)
    xyz = np.column_stack([x.ravel(), y.ravel(), 100 + 0.2 * x.ravel() + 0.1 * y.ravel()])
    faces = []
    for j in range(len(ys) - 1):
        for i in range(len(xs) - 1):
            a = j * len(xs) + i
            faces += [(a, a + 1, a + len(xs) + 1), (a, a + len(xs) + 1, a + len(xs))]
    return xyz, np.array(faces)


class TerrainCogTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dir = Path(self.temp.name)

    def test_cells_hold_the_mesh_height_at_their_centre(self):
        write_mesh(self.dir / "a.ply", *plane(10, 14, 20, 23))
        grid, left, top = terrain_grid([self.dir / "a.ply"], 0.5)
        self.assertEqual((left, top, grid.shape), (10.0, 23.0, (6, 8)))
        x = left + (np.arange(8) + 0.5) * 0.5
        y = top - (np.arange(6) + 0.5) * 0.5
        expected = 100 + 0.2 * x[None, :] + 0.1 * y[:, None]
        np.testing.assert_allclose(grid, expected, atol=1e-3)

    def test_cells_off_the_mesh_stay_empty(self):
        xyz = np.array([[0.0, 0.0, 5.0], [10.0, 0.0, 5.0], [0.0, 10.0, 5.0]])
        write_mesh(self.dir / "a.ply", xyz, np.array([(0, 1, 2)]))
        grid, _, _ = terrain_grid([self.dir / "a.ply"], 0.5)
        self.assertTrue(np.isfinite(grid[-1, 0]))  # the triangle's right angle, south-west
        self.assertTrue(np.isnan(grid[0, -1]))  # beyond its hypotenuse
        self.assertEqual(np.isfinite(grid).sum(), 210)

    def test_several_meshes_share_one_grid(self):
        write_mesh(self.dir / "a.ply", *plane(0, 4, 0, 4))
        write_mesh(self.dir / "b.ply", *plane(4, 8, 0, 4))
        grid, _, _ = terrain_grid([self.dir / "a.ply", self.dir / "b.ply"], 0.5)
        self.assertEqual(grid.shape, (8, 16))
        self.assertTrue(np.isfinite(grid).all())

    def test_a_huge_triangle_is_tested_in_steps(self):
        import terrain_cog
        write_mesh(self.dir / "a.ply", *plane(0, 4, 0, 4, step=4.0))
        whole, _, _ = terrain_grid([self.dir / "a.ply"], 0.5)
        limit = terrain_cog.MAX_CANDIDATES
        terrain_cog.MAX_CANDIDATES = 8
        self.addCleanup(setattr, terrain_cog, "MAX_CANDIDATES", limit)
        steps, _, _ = terrain_grid([self.dir / "a.ply"], 0.5)
        np.testing.assert_array_equal(whole, steps)

    def test_writes_a_cog_with_the_mesh_crs(self):
        write_mesh(self.dir / "a.ply", *plane(1000, 1600, 2000, 2600, step=2.0))
        report = write_terrain_cog([self.dir / "a.ply"], self.dir / "out.tif")
        dataset = gdal.Open(str(self.dir / "out.tif"))
        self.assertEqual(dataset.GetMetadataItem("LAYOUT", "IMAGE_STRUCTURE"), "COG")
        self.assertIn("32632", dataset.GetProjection())
        self.assertEqual(dataset.GetGeoTransform(), (1000.0, 0.5, 0.0, 2600.0, 0.0, -0.5))
        band = dataset.GetRasterBand(1)
        self.assertEqual(band.GetNoDataValue(), NODATA)
        self.assertGreater(band.GetOverviewCount(), 0)
        self.assertEqual(report["cells"], 1200 * 1200)

    def test_no_triangles_is_an_error(self):
        write_mesh(self.dir / "a.ply", np.zeros((0, 3)), np.zeros((0, 3), dtype=int))
        with self.assertRaises(ValueError):
            terrain_grid([self.dir / "a.ply"])
        self.assertIsNone(write_terrain_cog([self.dir / "a.ply"], self.dir / "out.tif"))
        self.assertFalse((self.dir / "out.tif").exists())


if __name__ == "__main__":
    unittest.main()
