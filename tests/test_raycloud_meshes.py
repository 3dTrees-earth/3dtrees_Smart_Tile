import json
import sys
import tempfile
import unittest
from pathlib import Path

import laspy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from raycloud_meshes import DATASET_SCOPE, FACE, NATIVE_SCOPE, TILE_SCOPE, VERTEX, TreeMesh
from remap_first_pipeline import merge_collections, strict_remap
from test_merge_stages import write_cloud
import test_raycloud_filter_only as rct_tests


def write_native_mesh(path, triangles, tree_ids, *, tree_count=2, scope=NATIVE_SCOPE, with_ids=True):
    """Native RCT layout: one vertex triple per triangle plus one unused vertex."""
    xyz = np.vstack([np.asarray(t, dtype=float) for t in triangles] + [np.zeros((1, 3))])
    vertices = np.zeros(len(xyz), dtype=VERTEX)
    vertices["xyz"] = xyz
    vertices["rgba"] = (np.arange(len(xyz))[:, None] * [1, 2, 3, 0] + [0, 0, 0, 255]) % 256
    faces = np.zeros(len(triangles), dtype=FACE)
    faces["count"] = 3
    faces["indices"] = np.arange(3 * len(triangles)).reshape(-1, 3)
    faces["tree_id"] = tree_ids
    header = ["ply", "format binary_little_endian 1.0",
              f"comment rct_tree_id_scope {scope}", f"comment rct_tree_count {tree_count}",
              f"element vertex {len(vertices)}", "property double x", "property double y", "property double z",
              "property uchar red", "property uchar green", "property uchar blue", "property uchar alpha",
              f"element face {len(faces)}", "property list int int vertex_indices"]
    header += ["property uint tree_id"] if with_ids else []
    with open(path, "wb") as stream:
        stream.write(("\n".join(header + ["end_header"]) + "\n").encode())
        stream.write(vertices.tobytes())
        stream.write(faces.tobytes() if with_ids else
                     np.rec.fromarrays([faces["count"], faces["indices"]],
                                       dtype=[("c", "<i4"), ("i", "<i4", (3,))]).tobytes())


def triangle(x):
    return [[x, 0, 1], [x + .1, 0, 1], [x, .1, 2]]


class TreeMeshTests(unittest.TestCase):
    def fixture(self, root, **mesh_kwargs):
        source, layout = rct_tests.RayCloudFilterOnlyTests._fixture(None, root)
        # Tile trees: local 1 is retained (100001 / 200001), local 2 is removed.
        for tile, x in (("c00_r00", 1.0), ("c01_r00", 11.0)):
            write_native_mesh(source / f"{tile}_segmented_trees_mesh.ply",
                              [triangle(x), triangle(x + 5), triangle(x + .5)], [1, 2, 1], **mesh_kwargs)
        return source, layout

    def merge(self, root, source, layout, **kwargs):
        return merge_collections(collections=[source], target_dir=None, output_tiles=root / "output",
                                 tile_bounds_json=layout, ready=True, instance_dimension="PredInstance_RCT", **kwargs)

    def test_merge_keeps_retained_tree_faces_with_tile_prefixed_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, layout = self.fixture(root)
            report = self.merge(root, source, layout)
            for tile, x, prefix in ((0, 1.0, 100000), (1, 11.0, 200000)):
                out = TreeMesh.open(root / "segmented_filtered" / f"c{tile:02d}_r00_filtered_trees_mesh.ply")
                src = TreeMesh.open(source / f"c{tile:02d}_r00_segmented_trees_mesh.ply")
                self.assertEqual((out.scope, out.tree_count), (TILE_SCOPE, 1))
                self.assertEqual(out.faces["tree_id"].tolist(), [prefix + 1, prefix + 1])
                self.assertEqual(len(out.vertices), 6)  # removed tree and unused vertex dropped
                kept = src.faces[src.faces["tree_id"] == 1]
                np.testing.assert_array_equal(out.vertices[out.faces["indices"]]["xyz"],
                                              src.vertices[kept["indices"]]["xyz"])
                np.testing.assert_array_equal(out.vertices[out.faces["indices"]]["rgba"],
                                              src.vertices[kept["indices"]]["rgba"])
            meshes = report["models"][0]["tree_meshes"]
            self.assertEqual([(m["faces"], m["trees_with_faces"], m["removed_trees"]) for m in meshes],
                             [(2, 1, 1), (2, 1, 1)])

    def test_remap_writes_per_original_meshes_with_compact_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, layout = self.fixture(root)
            self.merge(root, source, layout)
            originals = root / "originals"
            originals.mkdir()
            write_cloud(originals / "a.las", [1, 11, 10.6])
            write_cloud(originals / "b.laz", [11])
            write_cloud(originals / "empty.las", [])
            strict_remap(collections=[root / "output"], originals=originals, output=root / "final")
            expected = {"a": [1, 1, 2, 2], "b": [2, 2], "empty": []}
            for stem, ids in expected.items():
                mesh = TreeMesh.open(root / "final" / f"{stem}_trees_mesh.ply")
                self.assertEqual((mesh.scope, mesh.tree_count), (DATASET_SCOPE, len(set(ids))))
                self.assertEqual(mesh.faces["tree_id"].tolist(), ids)
                tables = {int(line.split(",", 1)[0]) for line in
                          (root / "final" / f"{stem}_trees.txt").read_text().splitlines()[2:]}
                self.assertEqual(set(ids), tables)
            b = TreeMesh.open(root / "final/b_trees_mesh.ply")
            np.testing.assert_array_equal(b.vertices[b.faces["indices"]]["xyz"][0], triangle(11.0))
            files = json.loads((root / "final/rct_instance_mapping.json").read_text())["files"]
            self.assertEqual([f["sidecars"]["trees_mesh"] for f in files],
                             ["a_trees_mesh.ply", "b_trees_mesh.ply", "empty_trees_mesh.ply"])
            self.assertEqual([f["mesh"]["faces"] for f in files], [4, 2, 0])

    def test_meshes_record_the_point_cloud_crs_as_comment(self):
        from laspy.vlrs.known import WktCoordinateSystemVlr
        from pyproj import CRS
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, layout = self.fixture(root)
            for cloud in source.glob("*_segmented.las"):
                las = laspy.read(cloud)
                las.header.vlrs.append(WktCoordinateSystemVlr(CRS.from_epsg(25832).to_wkt("WKT1_GDAL")))
                las.write(cloud)
            self.merge(root, source, layout)
            for mesh in (root / "segmented_filtered").glob("*_trees_mesh.ply"):
                header = mesh.read_bytes().split(b"end_header", 1)[0].decode()
                self.assertIn("comment crs: EPSG:25832\n", header)
                TreeMesh.open(mesh)  # the extra comment keeps the native layout readable

    def test_rejects_meshes_without_tree_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, layout = self.fixture(root, with_ids=False)
            with self.assertRaisesRegex(ValueError, "no per-face tree_id"):
                self.merge(root, source, layout)
            self.assertFalse((root / "output").exists())

    def test_rejects_mesh_id_without_tree_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, layout = self.fixture(root)
            write_native_mesh(source / "c00_r00_segmented_trees_mesh.ply", [triangle(1.0)], [3], tree_count=3)
            with self.assertRaisesRegex(ValueError, "tree_id 3 has no tree row"):
                self.merge(root, source, layout)

    def test_requires_a_mesh_for_every_tile_or_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, layout = self.fixture(root)
            (source / "c01_r00_segmented_trees_mesh.ply").unlink()
            with self.assertRaisesRegex(ValueError, "Missing RayCloudTools tree mesh"):
                self.merge(root, source, layout)

    def test_meshes_require_tree_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, layout = self.fixture(root)
            for table in source.glob("*.txt"):
                table.unlink()
            with self.assertRaisesRegex(ValueError, "tree meshes require"):
                self.merge(root, source, layout)


if __name__ == "__main__":
    unittest.main()
