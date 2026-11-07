"""QSMs per segmentation: instance tiles with a manifest and RCT terrain, then merge_qsm per original (2.5)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from instance_tiles import write_instance_tiles  # noqa: E402
from qsm_merge import merge_qsm  # noqa: E402
from qsm_tiles import manifest_name, read_footprints  # noqa: E402
from raycloud_meshes import FACE as TREE_FACE, VERTEX  # noqa: E402
from raycloud_terrain import TerrainMesh, _faces, _header, _write  # noqa: E402
from test_instance_tiles import inputs, tiles  # noqa: E402

SAT = "PredInstance_SAT"
TREES_HEADER = ("# Tree file. Optional per-tree attributes (e.g. 'height,crown_radius, ') followed by 'x,y,z,radius' "
                "and any additional per-segment attributes:\npredinstance, x,y,z,radius,parent_id,section_id\n")
INFO_HEADER = ("# Tree file. Optional per-tree attributes (e.g. 'height,crown_radius, ') followed by 'x,y,z,radius' "
               "and any additional per-segment attributes:\npredinstance,height,crown_radius,dimension,monocotal,DBH,"
               "bend,branch_slope, x,y,z,radius,parent_id,section_id,volume,diameter,length,strength,min_strength,"
               "dominance,angle,children\n")


def grid_terrain(path, x0, x1, extra=()):
    """A 1 m triangulated ground from x0 to x1 (y 0..10) as RCT writes terrain, plus ``extra`` triangles."""
    xs, ys = np.arange(x0, x1 + 1), np.arange(0, 11)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    vertices = np.zeros(gx.size, dtype=VERTEX)
    vertices["xyz"] = np.c_[gx.ravel(), gy.ravel(), np.zeros(gx.size)]
    vertices["rgba"] = 255
    index = np.arange(gx.size).reshape(gx.shape)
    faces = [(index[i, j], index[i + 1, j], index[i, j + 1]) for i in range(len(xs) - 1) for j in range(len(ys) - 1)]
    for triangle in extra:
        start = len(vertices)
        more = np.zeros(3, dtype=VERTEX)
        more["xyz"] = triangle
        vertices = np.concatenate([vertices, more])
        faces.append((start, start + 1, start + 2))
    _write(path, _header("smarttile-terrain-original-footprint", len(vertices), len(faces), [], None),
           [vertices], [_faces(np.asarray(faces))])


def tree_mesh(path, trees):
    """A mesh as RCT --instance-dimension writes it: per tree one triangle, tree_id = instance ID."""
    vertices = np.zeros(3 * len(trees), dtype=VERTEX)
    faces = np.zeros(len(trees), dtype=TREE_FACE)
    for k, (uid, (x, y)) in enumerate(trees.items()):
        vertices["xyz"][3 * k:3 * k + 3] = [(x, y, 0), (x + 0.1, y, 0), (x, y, 2)]
        faces[k] = (3, (3 * k, 3 * k + 1, 3 * k + 2), uid)
    header = ["ply", "format binary_little_endian 1.0", "comment rct_tree_id_scope instance-dimension",
              f"comment rct_tree_count {len(trees)}", f"element vertex {len(vertices)}",
              "property double x", "property double y", "property double z", "property uchar red",
              "property uchar green", "property uchar blue", "property uchar alpha", f"element face {len(faces)}",
              "property list int int vertex_indices", "property uint tree_id", "end_header"]
    path.write_bytes(("\n".join(header) + "\n").encode() + vertices.tobytes() + faces.tobytes())


def rct_outputs(folder, label, trees):
    """The four RCT --instance-dimension files of one tile; ``trees``: {instance ID: (x, y)}."""
    folder.mkdir(exist_ok=True)
    rows = "".join(f"{uid}, {x},{y},0,0.15,-1,0, {x},{y},1.3,0.12,0,1\n" for uid, (x, y) in trees.items())
    (folder / f"{label}_trees.txt").write_text(TREES_HEADER + rows)
    if not trees:  # RCT writes an empty tile's tree-info table with the segment columns only
        (folder / f"{label}_trees_info.txt").write_text(TREES_HEADER)
        tree_mesh(folder / f"{label}_trees_mesh.ply", trees)
        grid_terrain(folder / f"{label}_mesh.ply", 0 if label == "c00_r00" else 8, 12 if label == "c00_r00" else 20)
        return
    info = "".join(f"{uid}.0000,8.0,1.5,0,0,0.24,0,0, {x},{y},0,0.15,-1,0,0.1,0.3,8,0,0,0,0,1, "
                   f"{x},{y},1.3,0.12,0,1,0.05,0.24,7,0,0,0,0,0\n" for uid, (x, y) in trees.items())
    (folder / f"{label}_trees_info.txt").write_text(INFO_HEADER + info)
    tree_mesh(folder / f"{label}_trees_mesh.ply", trees)
    grid_terrain(folder / f"{label}_mesh.ply", 0 if label == "c00_r00" else 8, 12 if label == "c00_r00" else 20)


@pytest.fixture
def tiled(tmp_path):
    """Instance tiles of two files with the QSM manifest; RCT terrain per original, overlapping at x 8..12."""
    sources = inputs(tmp_path)
    terrain = tmp_path / "rct_terrain"
    terrain.mkdir()
    grid_terrain(terrain / "a_terrain_mesh.ply", 0, 12)
    grid_terrain(terrain / "b_terrain_mesh.ply", 8, 20, extra=[[(100, 0, 0), (101, 0, 0), (100, 1, 0)]])
    out = tmp_path / "out"
    out.mkdir()
    write_instance_tiles(sources, tiles(), [SAT], out, resolutions=[0.5], rct_terrain=terrain)
    return tmp_path, out, terrain


def test_the_manifest_holds_each_instance_with_its_tile_and_every_original_with_its_points(tiled):
    _, out, _ = tiled
    manifest = json.loads((out / manifest_name(SAT)).read_text())
    assert manifest["originals"] == ["a", "b"]
    assert [t["label"] for t in manifest["tiles"]] == ["c00_r00", "c01_r00"]
    instances = manifest["instances"]
    assert instances["ids"] == [7, 8]
    assert instances["tile"] == [0, 1]
    assert instances["originals"] == [[0, 1], [1]]  # tree 7 continues across both files
    left, right = read_footprints(out / "qsm_footprints.npz", 2)
    assert len(left) and len(right)


def test_each_tile_gets_the_rct_terrain_around_it_with_shared_triangles_once(tiled):
    _, out, _ = tiled
    folder = out / f"instance_tiles_{SAT}_50cm_terrain"
    meshes = {p.stem: TerrainMesh.open(p) for p in sorted(folder.glob("*.ply"))}
    assert sorted(meshes) == ["c00_r00", "c01_r00"]
    for mesh in meshes.values():
        xyz = np.asarray(mesh.vertices["xyz"])
        assert xyz[:, 0].max() < 100  # the far triangle lies beyond the 10 m margin of both tiles
        triangles = {tuple(sorted(map(tuple, xyz[f]))) for f in np.asarray(mesh.faces["indices"])}
        assert len(triangles) == len(mesh.faces)  # a and b share x 8..12; each triangle once
        assert len(np.unique(xyz, axis=0)) == len(xyz)  # identical vertices welded
    # 20 x 10 m of 1 m cells, one triangle each.
    assert len(meshes["c00_r00"].faces) == 200
    del meshes


def run_merge(tmp_path, out, terrain, outputs, **options):
    qsm = tmp_path / "qsm"
    for label, trees in outputs.items():
        rct_outputs(qsm, label, trees)
    return merge_qsm(out / manifest_name(SAT), qsm, tmp_path / "merged", rct_terrain_folder=terrain, **options)


def table_ids(path):
    return [int(float(line.split(",", 1)[0])) for line in path.read_text().splitlines()[2:]]


def test_qsms_go_to_every_original_holding_their_points_with_ids_and_rows_kept(tiled):
    tmp_path, out, terrain = tiled
    report = run_merge(tmp_path, out, terrain, {"c00_r00": {7: (9.0, 5.0)}, "c01_r00": {8: (16.0, 5.0)}})
    merged = tmp_path / "merged"
    assert table_ids(merged / "a_trees.txt") == [7]
    assert table_ids(merged / "b_trees.txt") == [7, 8]
    assert table_ids(merged / "b_trees_info.txt") == [7, 8]
    source = (tmp_path / "qsm" / "c00_r00_trees_info.txt").read_text().splitlines()[2]
    assert (merged / "a_trees_info.txt").read_text().splitlines()[2] == source  # rows unchanged
    assert report["qsms"] == 2 and report["instances_without_qsm"] == 0
    assert report["terrain_cog"]["file"] == "rct_terrain.tif" and (tmp_path / "merged" / "rct_terrain.tif").is_file()
    for stem in ("a", "b"):
        assert (merged / f"{stem}_terrain_mesh.ply").read_bytes() == (terrain / f"{stem}_terrain_mesh.ply").read_bytes()
        assert not (merged / f"{stem}_qsm_octree").exists()


def octree_trees(folder):
    from qsm_octree import TREE
    hierarchy = json.loads((folder / "hierarchy.json").read_text())
    table = (folder / "octree.bin").read_bytes()[hierarchy["trees"]["offset"]:]
    return hierarchy, np.frombuffer(table, dtype=TREE)


def test_one_octree_per_run_points_each_tree_into_the_first_original_holding_it(tiled):
    tmp_path, out, terrain = tiled
    report = run_merge(tmp_path, out, terrain, {"c00_r00": {7: (9.0, 5.0)}, "c01_r00": {8: (16.0, 5.0)}})
    merged = tmp_path / "merged"
    hierarchy, trees = octree_trees(merged / "octree")
    assert hierarchy["contract"] == "3dtrees-qsm-octree-v2"
    assert [o["info"]["name"] for o in hierarchy["originals"]] == ["a_trees_info.txt", "b_trees_info.txt"]
    assert [o["mesh"]["name"] for o in hierarchy["originals"]] == ["a_trees_mesh.ply", "b_trees_mesh.ply"]
    assert trees["id"].tolist() == [7, 8] and trees["original"].tolist() == [0, 1]  # 7 is in a and b: from a
    assert hierarchy["totals"]["cylinders"] == 2 and report["octree"]["trees"] == 2
    for tree in trees:
        files = hierarchy["originals"][tree["original"]]
        info = (merged / files["info"]["name"]).read_bytes()
        line = info[tree["info_offset"]:tree["info_offset"] + tree["info_length"]].decode()
        assert int(float(line.split(",", 1)[0])) == tree["id"]
        mesh = (merged / files["mesh"]["name"]).read_bytes()
        face = np.frombuffer(mesh[tree["face_offset"]:tree["face_offset"] + tree["face_length"]], dtype=TREE_FACE)
        assert face["tree_id"].tolist() == [tree["id"]]


def test_a_tile_whose_instances_all_got_no_qsm_is_reported(tiled):
    tmp_path, out, terrain = tiled
    report = run_merge(tmp_path, out, terrain, {"c00_r00": {7: (9.0, 5.0)}, "c01_r00": {}})
    assert [t["qsms"] for t in report["tiles"]] == [1, 0]
    assert report["tiles"][1]["without_qsm"] == 1
    assert table_ids(tmp_path / "merged" / "b_trees_info.txt") == [7]


def test_only_empty_tiles_give_empty_tables(tiled):
    tmp_path, out, terrain = tiled
    report = run_merge(tmp_path, out, terrain, {"c00_r00": {}, "c01_r00": {}})
    assert report["qsms"] == 0 and report["octree"] is None
    assert table_ids(tmp_path / "merged" / "a_trees.txt") == []
    assert not (tmp_path / "merged" / "octree").exists()


def test_an_empty_table_beside_a_mesh_with_trees_fails(tiled):
    tmp_path, out, terrain = tiled
    qsm = tmp_path / "qsm"
    rct_outputs(qsm, "c00_r00", {7: (9.0, 5.0)})
    rct_outputs(qsm, "c01_r00", {})
    tree_mesh(qsm / "c01_r00_trees_mesh.ply", {8: (16.0, 5.0)})
    with pytest.raises(ValueError, match="c01_r00: the tables list no QSM but the mesh holds 1 trees"):
        merge_qsm(out / manifest_name(SAT), qsm, tmp_path / "merged", rct_terrain_folder=terrain)


def test_a_tile_with_its_instances_but_no_files_fails(tiled):
    tmp_path, out, terrain = tiled
    qsm = tmp_path / "qsm"
    rct_outputs(qsm, "c00_r00", {7: (9.0, 5.0)})
    rct_outputs(qsm, "c01_r00", {})
    for path in qsm.glob("c01_r00_*"):
        if path.name != "c01_r00_mesh.ply":
            path.unlink()
    with pytest.raises(ValueError, match="c01_r00 owns 1 instances but has no RayCloudTools trees, info, mesh"):
        merge_qsm(out / manifest_name(SAT), qsm, tmp_path / "merged", rct_terrain_folder=terrain)


def test_an_instance_without_qsm_is_reported_not_written(tiled):
    tmp_path, out, terrain = tiled
    report = run_merge(tmp_path, out, terrain, {"c00_r00": {7: (9.0, 5.0)}, "c01_r00": {}})
    assert report["instances_without_qsm"] == 1
    assert table_ids(tmp_path / "merged" / "b_trees.txt") == [7]


@pytest.mark.parametrize("outputs, message", [
    ({"c00_r00": {7: (9.0, 5.0), 8: (16.0, 5.0)}, "c01_r00": {}}, "does not own"),
    ({"c00_r00": {7: (9.0, 5.0)}, "c01_r00": {7: (9.0, 5.0), 8: (16.0, 5.0)}}, "tiles c00_r00 and c01_r00"),
])
def test_an_instance_in_the_wrong_or_in_two_tiles_fails(tiled, outputs, message):
    tmp_path, out, terrain = tiled
    with pytest.raises(ValueError, match=message):
        run_merge(tmp_path, out, terrain, outputs)


def test_a_tile_without_its_mesh_fails(tiled):
    tmp_path, out, terrain = tiled
    qsm = tmp_path / "qsm"
    rct_outputs(qsm, "c00_r00", {7: (9.0, 5.0)})
    rct_outputs(qsm, "c01_r00", {8: (16.0, 5.0)})
    (qsm / "c01_r00_trees_mesh.ply").unlink()
    with pytest.raises(ValueError, match="c01_r00 owns 1 instances but has no RayCloudTools mesh"):
        merge_qsm(out / manifest_name(SAT), qsm, tmp_path / "merged", rct_terrain_folder=terrain)


def test_without_rct_terrain_each_original_gets_the_tiles_terrain_over_its_footprint(tiled):
    tmp_path, out, _ = tiled
    report = run_merge(tmp_path, out, None, {"c00_r00": {7: (9.0, 5.0)}, "c01_r00": {8: (16.0, 5.0)}})
    assert report["terrain"] == "instance tiles"
    merged = tmp_path / "merged"
    both = TerrainMesh.open(merged / "b_terrain_mesh.ply")
    xyz = np.asarray(both.vertices["xyz"])
    centroids = xyz[np.asarray(both.faces["indices"])][:, :, :2].mean(axis=1)
    # Tiles overlap at x 8..12; each triangle comes once, from the tile whose core holds its centroid.
    assert len({tuple(np.round(c, 6)) for c in centroids}) == len(centroids)
    del both


def test_merge_takes_the_qsms_from_folders_by_kind_as_galaxy_collections_arrive(tiled):
    import subprocess
    tmp_path, out, terrain = tiled
    qsm = tmp_path / "qsm"
    rct_outputs(qsm, "c00_r00", {7: (9.0, 5.0)})
    rct_outputs(qsm, "c01_r00", {8: (16.0, 5.0)})
    folders = {}
    for suffix, option in (("_trees_info.txt", "--tree-info-folder"), ("_trees.txt", "--tree-tables-folder"),
                           ("_trees_mesh.ply", "--tree-meshes-folder"), ("_mesh.ply", "--terrain-meshes-folder")):
        folder = tmp_path / option.strip("-")
        folder.mkdir()
        for path in qsm.glob(f"*{suffix}"):
            if path.name.endswith(suffix) and not (suffix == "_mesh.ply" and path.name.endswith("_trees_mesh.ply")) \
                    and not (suffix == "_trees.txt" and path.name.endswith("_trees_info.txt")):
                (folder / f"{path.name[:7]}{Path(suffix).suffix}").write_bytes(path.read_bytes())
        folders[option] = folder
    run = [sys.executable, str(Path(__file__).resolve().parents[1] / "src" / "run.py"), "--task", "merge",
           "--instance-manifest", str(out / manifest_name(SAT)), "--rct-terrain-folder", str(terrain),
           "--output-dir", str(tmp_path / "merged")]
    for option, folder in folders.items():
        run += [option, str(folder)]
    result = subprocess.run(run, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert table_ids(tmp_path / "merged" / "b_trees.txt") == [7, 8]
    refused = subprocess.run(run[:-2 * len(folders)] + ["--segmented-remapped-folder", str(qsm)] + run[-2 * len(folders):],
                             capture_output=True, text=True)
    assert refused.returncode != 0 and "takes no segmented point collections" in refused.stdout
