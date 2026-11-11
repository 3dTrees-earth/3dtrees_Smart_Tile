"""RCT's own trees get the viewer's QSM octree too: one per run, beside the per-original files."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from collection_folders import original_collections  # noqa: E402
from raycloud_finalization import OCTREE, _write_octree  # noqa: E402
from test_qsm_per_segmentation import INFO_HEADER, TREES_HEADER, tree_mesh  # noqa: E402


def original(folder, stem, trees):
    """An original's tree-info table and tree mesh as the remap writes them; ``trees``: {ID: (x, y)}."""
    info = "".join(f"{uid},8.0,1.5,0,0,0.24,0,0, {x},{y},0,0.15,-1,0,0.1,0.3,8,0,0,0,0,1, "
                   f"{x},{y},1.3,0.12,0,1,0.05,0.24,7,0,0,0,0,0\n" for uid, (x, y) in trees.items())
    (folder / f"{stem}_trees_info.txt").write_text(INFO_HEADER + info)
    tree_mesh(folder / f"{stem}_trees_mesh.ply", trees)
    return {"file": f"{stem}.laz", "tree_count": len(trees),
            "sidecars": {"trees_info": f"{stem}_trees_info.txt", "trees_mesh": f"{stem}_trees_mesh.ply"}}


def test_one_octree_for_all_originals_with_a_shared_tree_once(tmp_path):
    files = [original(tmp_path, "a", {1: (1.0, 1.0), 2: (9.0, 1.0)}), original(tmp_path, "b", {2: (9.0, 1.0)}),
             {"file": "empty.laz", "tree_count": 0, "sidecars": {}}]
    totals = _write_octree(tmp_path, files)
    assert totals["trees"] == 2
    hierarchy = json.loads((tmp_path / OCTREE / "hierarchy.json").read_text())
    assert [o["info"]["name"] for o in hierarchy["originals"]] == ["a_trees_info.txt", "b_trees_info.txt"]
    assert original_collections(tmp_path)["qsm_octree"] == [tmp_path / OCTREE / "hierarchy.json",
                                                            tmp_path / OCTREE / "octree.bin"]


def test_no_octree_from_tables_without_segment_columns(tmp_path):
    (tmp_path / "a_trees_info.txt").write_text("# RCT tree table\nheader\ntree-1\n")
    files = [{"file": "a.laz", "tree_count": 1, "sidecars": {"trees_info": "a_trees_info.txt"}}]
    assert _write_octree(tmp_path, files) is None
    assert not (tmp_path / OCTREE).exists()
    assert "qsm_octree" not in original_collections(tmp_path)


def test_without_treeinfo_the_tree_table_feeds_the_octree(tmp_path):
    (tmp_path / "a_trees.txt").write_text(TREES_HEADER + "1, 1,1,0,0.15,-1,0, 1,1,1.3,0.12,0,1\n")
    files = [{"file": "a.laz", "tree_count": 1, "sidecars": {"trees": "a_trees.txt"}}]
    assert _write_octree(tmp_path, files)["trees"] == 1


INFO_HEADING = INFO_HEADER.splitlines()[1].split(",", 1)[1]  # without predinstance, as RCT writes a tile


def test_remap_writes_the_octree_of_rct_s_own_trees(tmp_path):
    """Merge and remap of RCT tiles with treeinfo tables: per-original files and one octree, published."""
    import test_raycloud_filter_only as rct_tests
    from remap_first_pipeline import merge_collections, strict_remap
    from test_merge_stages import write_cloud
    from test_raycloud_meshes import triangle, write_native_mesh

    source, layout = rct_tests.RayCloudFilterOnlyTests._fixture(None, tmp_path)
    for tile, x in (("c00_r00", 1.0), ("c01_r00", 11.0)):
        rows = "".join(f"8.0,1.5,0,0,0.24,0,0, {x + k},0,0,0.15,-1,0,0.1,0.3,8,0,0,0,0,1, "
                       f"{x + k},0,1.3,0.12,0,1,0.05,0.24,7,0,0,0,0,0\n" for k in (0, 5))
        (source / f"{tile}_trees_info.txt").write_text("# RCT tree table\n" + INFO_HEADING + "\n" + rows)
        write_native_mesh(source / f"{tile}_segmented_trees_mesh.ply",
                          [triangle(x), triangle(x + 5), triangle(x + .5)], [1, 2, 1])
    merge_collections(collections=[source], target_dir=None, output_tiles=tmp_path / "output",
                      tile_bounds_json=layout, ready=True, instance_dimension="PredInstance_RCT")
    originals = tmp_path / "originals"
    originals.mkdir()
    write_cloud(originals / "a.las", [1, 11, 10.6])
    write_cloud(originals / "b.laz", [11])
    report = strict_remap(collections=[tmp_path / "output"], originals=originals, output=tmp_path / "final")
    assert report["rct_finalization"]["octree"]["trees"] == 2  # tree 2 is in a and b, once in the octree
    hierarchy = json.loads((tmp_path / "final" / OCTREE / "hierarchy.json").read_text())
    assert [o["info"]["name"] for o in hierarchy["originals"]] == ["a_trees_info.txt", "b_trees_info.txt"]
    assert "octree/hierarchy.json" in report["checksums"]
    assert len(original_collections(tmp_path / "final")["qsm_octree"]) == 2
