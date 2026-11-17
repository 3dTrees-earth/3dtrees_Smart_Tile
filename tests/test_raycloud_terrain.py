import json
import sys
import tempfile
import unittest
from pathlib import Path

import laspy
import numpy as np
from laspy.vlrs.known import WktCoordinateSystemVlr
from pyproj import CRS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from raycloud_meshes import VERTEX
from raycloud_terrain import (FACE, ORIGINAL_SCOPE, TILE_SCOPE, TerrainMesh, _clip, filtered_terrain_meshes,
                              open_tile_terrain, write_original_terrain)
from remap_first_pipeline import merge_collections, strict_remap
from test_merge_stages import write_cloud
import test_raycloud_filter_only as rct_tests
from test_raycloud_meshes import triangle, write_native_mesh

STEP = 0.5


def grid_triangles(x0, x1, y0=-3.0, y1=3.0):
    """Global grid squares inside [x0, x1] x [y0, y1], two triangles each, as integer corner keys."""
    triangles = []
    for i in range(int(round(x0 / STEP)), int(round(x1 / STEP))):
        for j in range(int(round(y0 / STEP)), int(round(y1 / STEP))):
            a, b, c, d = (i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)
            triangles += [(a, b, c), (a, c, d)]
    return triangles


def write_terrain(path, triangles, *, shift=0.0):
    """Native RCT terrain layout: tile-local vertex order, no per-face ID, constant colour."""
    corners = sorted({corner for triangle in triangles for corner in triangle}, reverse=True)
    index = {corner: n for n, corner in enumerate(corners)}
    vertices = np.zeros(len(corners), dtype=VERTEX)
    xy = np.array(corners, dtype=float) * STEP
    vertices["xyz"] = np.column_stack([xy[:, 0] + shift, xy[:, 1], 100 + 0.1 * xy[:, 0]])
    vertices["rgba"] = [120, 90, 60, 255]
    faces = np.zeros(len(triangles), dtype=FACE)
    faces["count"] = 3
    faces["indices"] = [[index[c] for c in triangle] for triangle in triangles]
    header = ["ply", "format binary_little_endian 1.0", "comment SDK generated",
              f"element vertex {len(vertices)}", "property double x", "property double y", "property double z",
              "property uchar red", "property uchar green", "property uchar blue", "property uchar alpha",
              f"element face {len(faces)}", "property list int int vertex_indices", "end_header"]
    with open(path, "wb") as stream:
        stream.write(("\n".join(header) + "\n").encode())
        stream.write(vertices.tobytes())
        stream.write(faces.tobytes())


def triangle_set(mesh):
    """Each face as a sorted tuple of its vertex coordinates, so files compare independent of order."""
    xyz = np.asarray(mesh.vertices["xyz"])[mesh.faces["indices"]]
    return [tuple(sorted(map(tuple, face.round(9).tolist()))) for face in xyz]


def expected(triangles):
    return sorted(tuple(sorted((x * STEP, y * STEP, round(100 + 0.1 * x * STEP, 9)) for x, y in t))
                  for t in triangles)


class TerrainMeshTests(unittest.TestCase):
    """Tiles c00_r00 (core x -1..10) and c01_r00 (core x 10..21), buffer 1 m."""

    def fixture(self, root, *, shift=0.0):
        source, layout = rct_tests.RayCloudFilterOnlyTests._fixture(None, root)
        write_terrain(source / "c00_r00_segmented_mesh.ply", grid_triangles(-1, 11))
        write_terrain(source / "c01_r00_segmented_mesh.ply", grid_triangles(9, 21), shift=shift)
        return source, layout

    def merge(self, root, source, layout, **kwargs):
        return merge_collections(collections=[source], target_dir=None, output_tiles=root / "output",
                                 tile_bounds_json=layout, ready=True, instance_dimension="PredInstance_RCT", **kwargs)

    def test_merge_keeps_each_triangle_in_exactly_one_tile_core(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, layout = self.fixture(root)
            report = self.merge(root, source, layout)
            pieces = [TerrainMesh.open(root / "segmented_filtered" / f"c0{t}_r00_filtered_terrain_mesh.ply")
                      for t in range(2)]
            self.assertEqual([p.scope for p in pieces], [TILE_SCOPE, TILE_SCOPE])
            for piece, (low, high) in zip(pieces, ((-1, 10), (10, 21))):
                x = np.asarray(piece.vertices["xyz"])[piece.faces["indices"]][:, :, 0].mean(axis=1)
                self.assertTrue(np.all((x >= low) & (x < high)))
                # Retained vertices and colours are copied unchanged; unused vertices go.
                self.assertEqual(len(piece.vertices), len(np.unique(piece.faces["indices"])))
                self.assertTrue(np.all(piece.vertices["rgba"] == [120, 90, 60, 255]))
            together = triangle_set(pieces[0]) + triangle_set(pieces[1])
            self.assertEqual(sorted(together), expected(grid_triangles(-1, 21)))
            terrain = report["models"][0]["terrain_meshes"]
            self.assertEqual([t["faces"] for t in terrain["tiles"]], [len(triangle_set(p)) for p in pieces])
            self.assertEqual(terrain["tiles_without_terrain"], [])
            self.assertEqual(terrain["seams"]["open_seam_edges"], 0)
            self.assertEqual(terrain["seams"]["welded_vertices"], 13)  # x = 10, y = -3..3

    def test_seam_check_reports_tiles_that_do_not_weld(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, layout = self.fixture(root, shift=1e-6)
            seams = self.merge(root, source, layout)["models"][0]["terrain_meshes"]["seams"]
            self.assertEqual(seams["welded_vertices"], 0)
            # Six edges on each side of x = 10 lie beside the 2 m core line (y = -1..1).
            self.assertEqual(seams["open_seam_edges"], 12)

    def test_each_original_gets_the_terrain_cut_straight_at_its_extent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, layout = self.fixture(root)
            self.merge(root, source, layout)
            tiles = open_tile_terrain(filtered_terrain_meshes(root / "segmented_filtered"))
            # a and b meet at x = 10.25, across the tile seam at x = 10; no extent edge is on the 0.5 m grid.
            extents = {"a": (1.1, -1.2, 10.25, 1.3), "b": (10.25, -1.2, 14.7, 1.3)}
            meshes = {}
            for stem, extent in extents.items():
                report = write_original_terrain(root / f"{stem}_terrain_mesh.ply", tiles, extent)
                mesh = meshes[stem] = TerrainMesh.open(root / f"{stem}_terrain_mesh.ply")
                self.assertEqual(mesh.scope, ORIGINAL_SCOPE)
                self.assertEqual(report["faces"], len(mesh.faces))
                x0, y0, x1, y1 = extent
                xyz = np.asarray(mesh.vertices["xyz"])
                tri = xyz[mesh.faces["indices"]]
                # Every vertex lies in the extent and the triangles cover it exactly: no holes, no overlap.
                self.assertTrue(np.all((xyz[:, 0] >= x0) & (xyz[:, 0] <= x1) & (xyz[:, 1] >= y0) & (xyz[:, 1] <= y1)))
                cross = ((tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1])
                         - (tri[:, 1, 1] - tri[:, 0, 1]) * (tri[:, 2, 0] - tri[:, 0, 0]))
                self.assertAlmostEqual(float(np.abs(cross).sum() / 2), (x1 - x0) * (y1 - y0), places=9)
                # Triangles wholly inside the extent are the tiles' own, unchanged.
                inside = [t for t in grid_triangles(-1, 21)
                          if all(x0 <= c[0] * STEP <= x1 and y0 <= c[1] * STEP <= y1 for c in t)]
                self.assertTrue(set(expected(inside)) <= set(triangle_set(mesh)))
                # Cut vertices lie on the triangle they came from (z = 100 + 0.1 x here).
                self.assertTrue(np.allclose(xyz[:, 2], 100 + 0.1 * xyz[:, 0]))
                # Vertices are welded, also across the tile seam: every vertex is distinct and used.
                self.assertEqual(len(np.unique(xyz, axis=0)), len(xyz))
                self.assertEqual(len(xyz), len(np.unique(mesh.faces["indices"])))
            # Neighbouring originals meet on one straight cut with identical vertices.
            line = lambda stem: {tuple(v) for v in np.asarray(meshes[stem].vertices["xyz"]) if v[0] == 10.25}
            self.assertEqual(line("a"), line("b"))
            self.assertGreater(len(line("a")), 2)

    def test_remap_cuts_each_original_at_the_extent_of_its_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, layout = self.fixture(root)
            self.merge(root, source, layout)
            originals = root / "originals"
            originals.mkdir()
            write_cloud(originals / "a.las", [1, 11, 10.6])
            write_cloud(originals / "empty.las", [])
            report = strict_remap(collections=[root / "output"], originals=originals, output=root / "final")
            files = {f["file"]: f for f in report["rct_finalization"]["files"]}
            # The fixture's points lie on y = 0: an extent without area holds no terrain.
            self.assertEqual(files["a.las"]["terrain"]["extent"], [1.0, 0.0, 11.0, 0.0])
            self.assertEqual(files["a.las"]["sidecars"]["terrain_mesh"], "a_terrain_mesh.ply")
            self.assertEqual(files["empty.las"]["terrain"]["extent"], None)
            for stem in ("a", "empty"):
                self.assertEqual(len(TerrainMesh.open(root / "final" / f"{stem}_terrain_mesh.ply").faces), 0)

    def test_terrain_records_the_crs_and_no_mapping_file_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, layout = self.fixture(root)
            for tile, x in (("c00_r00", 1.0), ("c01_r00", 11.0)):
                write_native_mesh(source / f"{tile}_segmented_trees_mesh.ply",
                                  [triangle(x), triangle(x + 5), triangle(x + .5)], [1, 2, 1])
            self.merge(root, source, layout)
            originals = root / "originals"
            originals.mkdir()
            write_cloud(originals / "a.las", [1, 11, 10.6])
            cloud = laspy.read(originals / "a.las")
            cloud.header.vlrs.append(WktCoordinateSystemVlr(CRS.from_epsg(32632).to_wkt("WKT1_GDAL")))
            cloud.write(originals / "a.las")
            strict_remap(collections=[root / "output"], originals=originals, output=root / "final")
            header = (root / "final/a_terrain_mesh.ply").read_bytes().split(b"end_header")[0].decode()
            self.assertIn("comment crs: EPSG:32632", header)
            for name in ("a_trees_mesh.ply", "a_terrain_mesh.ply"):
                self.assertNotIn("mapping", (root / "final" / name).read_bytes().split(b"end_header")[0].decode())
            vlrs = [v for v in laspy.read(root / "final/a.las").header.vlrs
                    if v.user_id == "3DTrees" and v.record_id in (24003, 24004)]
            self.assertEqual(len(vlrs), 2)
            for vlr in vlrs:
                self.assertNotIn("mapping", json.loads(vlr.record_data))

    def test_combined_and_standalone_terrain_agree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, layout = self.fixture(root)
            originals = root / "originals"
            originals.mkdir()
            write_cloud(originals / "raw.las", [1, 11, 10.6])
            self.merge(root, source, layout, originals=originals)
            strict_remap(collections=[root / "output"], originals=originals, output=root / "separate")
            self.assertEqual((root / "original_with_predictions/raw_terrain_mesh.ply").read_bytes(),
                             (root / "separate/raw_terrain_mesh.ply").read_bytes())

    def test_tiles_without_terrain_are_holes_not_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, layout = self.fixture(root)
            (source / "c01_r00_segmented_mesh.ply").unlink()
            terrain = self.merge(root, source, layout)["models"][0]["terrain_meshes"]
            self.assertEqual([t["tile"] for t in terrain["tiles"]], [0])
            self.assertEqual(terrain["tiles_without_terrain"], ["c01_r00_segmented.las"])

    def test_terrain_requires_tree_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, layout = self.fixture(root)
            for table in source.glob("*.txt"):
                table.unlink()
            with self.assertRaisesRegex(ValueError, "terrain meshes require the matching"):
                self.merge(root, source, layout)

    def test_a_triangle_across_a_corner_is_clipped_to_the_extent(self):
        corners = np.zeros(3, dtype=VERTEX)
        corners["xyz"] = [[0, 0, 0], [4, 0, 4], [0, 4, 0]]
        corners["rgba"] = [[0, 0, 0, 255], [200, 0, 0, 255], [0, 0, 0, 255]]
        polygon = _clip(list(corners), (1, 1, 3, 3))
        xy = np.array([p["xyz"][:2] for p in polygon])
        area = 0.5 * abs((xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1]).sum())
        self.assertAlmostEqual(area, 2.0)  # the part of the 2 x 2 square below x + y = 4
        self.assertTrue(np.allclose([p["xyz"][2] for p in polygon], xy[:, 0]))
        self.assertEqual({int(p["rgba"][0]) for p in polygon if p["xyz"][0] == 1}, {50})


if __name__ == "__main__":
    unittest.main()
