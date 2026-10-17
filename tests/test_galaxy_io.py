"""Galaxy-facing I/O (v2.4.1): tolerant input names, RCT input folders, --collection-dir."""
import json
import shutil
import sys
from pathlib import Path

import laspy
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import run  # noqa: E402
from collection_folders import merged_product_collections, original_collections, publish_collections  # noqa: E402
from output_names import file_stem, subsample_output_name  # noqa: E402
from parameters import Parameters  # noqa: E402
from point_cloud_metadata import is_copc_name, point_cloud_files, point_cloud_source_key  # noqa: E402
from remap_first_pipeline import merge_collections  # noqa: E402
from test_instance_tiles import inputs  # noqa: E402
import test_raycloud_filter_only as rct_tests  # noqa: E402
from test_raycloud_meshes import triangle, write_native_mesh  # noqa: E402


def tree(root):
    """Every file below root, as relative paths."""
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


@pytest.mark.parametrize("name, stem", [
    ("plot_A.laz.laz", "plot_A"), ("c00_r00.copc.laz", "c00_r00"), ("Plot.LAZ", "Plot"),
    ("plot_A.las.las", "plot_A"), ("c00_r00_trees.txt", "c00_r00_trees"), ("mesh.ply", "mesh"),
    ("c00_r00", "c00_r00"), ("plot.v2.laz", "plot.v2"), ("scan.COPC.LAZ.laz", "scan"), (".laz", ".laz"),
])
def test_file_stem_strips_known_extensions_repeatedly(name, stem):
    assert file_stem(Path("folder") / name) == stem


def test_doubled_extensions_name_one_source():
    assert point_cloud_source_key(Path("Plot_A.laz.laz")) == "plot_a"
    assert is_copc_name(Path("x.copc.laz")) and is_copc_name(Path("x.copc.laz.laz"))
    assert not is_copc_name(Path("x.laz.laz")) and not is_copc_name(Path("copc.laz"))
    assert subsample_output_name(Path("mikro.laz.copc.laz"), 0.1, None, False) == "mikro_subsampled_10cm.laz"


def test_galaxy_named_inputs_with_one_source_key_fail(tmp_path):
    (tmp_path / "plot.laz.laz").write_bytes(b"")
    (tmp_path / "plot.las").write_bytes(b"")
    with pytest.raises(ValueError, match="name the same source"):
        point_cloud_files(tmp_path)


def test_tile_collection_dir_holds_one_flat_folder_per_output(tmp_path):
    source = inputs(tmp_path)[0]
    source.rename(source.with_name("plot_A.las.las"))  # Galaxy: '<element_identifier>.<ext>'
    (source.parent / "b.las").unlink()
    products, collections = tmp_path / "products", tmp_path / "collections"
    run.run_tile_task(Parameters(_cli_parse_args=[
        "--task", "tile", "--input-dir", str(source.parent), "--output-dir", str(products),
        "--tiling-threshold", "1000", "--workers", "1", "--collection-dir", str(collections),
        "--tile-outputs", "0.25:copc.laz,1:laz,0.5:ply"]))
    # The task's own layout is unchanged; only the input name is cleaned.
    assert {p for p in tree(products) if p.startswith("subsampled_res")} == {
        "subsampled_res1/plot_A_subsampled_25cm.copc.laz",
        "subsampled_res2/plot_A_subsampled_100cm.laz",
        "subsampled_res3/plot_A_subsampled_50cm.ply"}
    # Other test modules may stub the overview plot; publish it whenever it exists.
    preview = ["overview_copc_tiles.png"] if (products / "overview_copc_tiles.png").exists() else []
    assert tree(collections) == preview + [
        "tile_bounds_tindex.json", "tiles_1/plot_A_subsampled_25cm.laz", "tiles_2/plot_A_subsampled_100cm.laz",
        "tiles_3/plot_A_subsampled_50cm.ply"]
    published = collections / "tiles_1/plot_A_subsampled_25cm.laz"
    assert published.samefile(products / "subsampled_res1/plot_A_subsampled_25cm.copc.laz")
    with laspy.open(published) as reader:
        assert reader.header.are_points_compressed


def rct_folders(root, tables_named_like_tiles=True):
    """RCT fixture with tables and meshes moved into folders of their own, named like Galaxy elements."""
    source, layout = rct_tests.RayCloudFilterOnlyTests._fixture(None, root)
    for tile, x in (("c00_r00", 1.0), ("c01_r00", 11.0)):
        write_native_mesh(source / f"{tile}_trees_mesh.ply",
                          [triangle(x), triangle(x + 5), triangle(x + .5)], [1, 2, 1])
    folders = {kind: root / kind for kind in ("trees", "trees_info", "trees_mesh")}
    for tile in ("c00_r00", "c01_r00"):
        element = f"{tile}_subsampled_10cm" if tables_named_like_tiles else f"{tile}_subsampled_10cm_trees"
        for kind, extension in (("trees", ".txt"), ("trees_info", ".txt"), ("trees_mesh", ".ply")):
            folders[kind].mkdir(exist_ok=True)
            shutil.move(source / f"{tile}_{kind}{extension}", folders[kind] / f"{element}{extension}")
    return source, layout, folders


@pytest.mark.parametrize("tables_named_like_tiles", [True, False])
def test_rct_files_in_own_folders_match_the_colocated_convention(tmp_path, tables_named_like_tiles):
    (tmp_path / "a").mkdir()
    colocated, layout = rct_tests.RayCloudFilterOnlyTests._fixture(None, tmp_path / "a")
    for tile, x in (("c00_r00", 1.0), ("c01_r00", 11.0)):
        write_native_mesh(colocated / f"{tile}_trees_mesh.ply",
                          [triangle(x), triangle(x + 5), triangle(x + .5)], [1, 2, 1])
    (tmp_path / "b").mkdir()
    source, layout_b, folders = rct_folders(tmp_path / "b", tables_named_like_tiles)
    assert not list(source.glob("*.txt")) and not list(source.glob("*.ply"))
    for root, collection, bounds, extra in ((tmp_path / "a", colocated, layout, {}),
                                            (tmp_path / "b", source, layout_b, {"rct_folders": folders})):
        merge_collections(collections=[collection], target_dir=None, output_tiles=root / "output_tiles",
                          tile_bounds_json=bounds, ready=True, instance_dimension="PredInstance_RCT", **extra)
    expected = tree(tmp_path / "a/segmented_filtered")
    assert expected == [f"c0{t}_r00_filtered_{kind}" for t in (0, 1)
                        for kind in ("trees.txt", "trees_info.txt", "trees_mesh.ply")]
    assert tree(tmp_path / "b/segmented_filtered") == expected
    for name in expected:
        assert ((tmp_path / "a/segmented_filtered" / name).read_bytes()
                == (tmp_path / "b/segmented_filtered" / name).read_bytes())


def test_rct_table_folders_are_required_together(tmp_path):
    source, layout, folders = rct_folders(tmp_path)
    with pytest.raises(SystemExit):
        run.run_merge_task(Parameters(task="merge", segmented_remapped_folder=source, tile_bounds_json=layout,
                                      tree_tables_folder=folders["trees"], _cli_parse_args=False))
    assert not (tmp_path / "output_tiles").exists()


def test_merge_collection_dir_layout(tmp_path):
    source, layout, folders = rct_folders(tmp_path)
    collections = tmp_path / "collections"
    run.run_merge_task(Parameters(_cli_parse_args=[
        "--task", "merge", "--segmented-remapped-folder", str(source), "--tile-bounds-json", str(layout),
        "--output-tiles-folder", str(tmp_path / "out/output_tiles"), "--instance-dimension", "PredInstance_RCT",
        "--disable-matching", "True", "--skip-merged-file", "True", "--skip-unfiltered-tiles", "True",
        "--tree-tables-folder", str(folders["trees"]), "--tree-info-folder", str(folders["trees_info"]),
        "--tree-meshes-folder", str(folders["trees_mesh"]), "--collection-dir", str(collections)]))
    assert tree(collections) == [
        "merged_tiles/c00_r00.laz", "merged_tiles/c01_r00.laz",
        "tree_info_tables/c00_r00_filtered_trees_info.txt", "tree_info_tables/c01_r00_filtered_trees_info.txt",
        "tree_meshes/c00_r00_filtered_trees_mesh.ply", "tree_meshes/c01_r00_filtered_trees_mesh.ply",
        "tree_tables/c00_r00_filtered_trees.txt", "tree_tables/c01_r00_filtered_trees.txt"]
    assert (collections / "merged_tiles/c00_r00.laz").samefile(tmp_path / "out/output_tiles/c00_r00.laz")
    # The default merge layout is unchanged.
    assert (tmp_path / "out/segmented_filtered/c00_r00_filtered_trees.txt").is_file()
    assert json.loads((tmp_path / "out/output_tiles/smarttile_merge.json").read_text())["rct_tree_sidecars"]


def test_originals_and_merged_products_get_one_extension(tmp_path):
    work = tmp_path / "work"
    originals = work / "original_with_predictions"
    originals.mkdir(parents=True)
    for name in ("plot_A.laz.laz", "plot_B.las", "plot_B_trees.txt", "rct_instance_mapping.json"):
        (originals / name).write_bytes(name.encode())
    products = [work / name for name in ("prod_merged_10cm.laz", "prod_merged_10cm.copc.laz", "merged.ply")]
    for product in products:
        product.write_bytes(product.name.encode())
    collections = tmp_path / "collections"
    publish_collections(collections, {**original_collections(originals), **merged_product_collections(products)})
    assert tree(collections) == [
        "merged_copc/prod_merged_10cm.laz", "merged_laz/prod_merged_10cm.laz", "merged_ply/merged.ply",
        "originals_with_predictions/plot_A.laz", "originals_with_predictions/plot_B.las"]
    assert (collections / "merged_copc/prod_merged_10cm.laz").read_bytes() == b"prod_merged_10cm.copc.laz"


def test_collection_names_must_be_unique_and_the_folder_new(tmp_path):
    (tmp_path / "a.laz").write_bytes(b"1")
    (tmp_path / "a.copc.laz").write_bytes(b"2")
    with pytest.raises(ValueError, match="would both be named a"):
        publish_collections(tmp_path / "c", {"tiles_1": [tmp_path / "a.laz", tmp_path / "a.copc.laz"]})
    assert not list((tmp_path / "c").rglob("*.laz"))
    from collection_folders import check_collection_dir
    (tmp_path / "used").mkdir()
    (tmp_path / "used/old.laz").write_bytes(b"")
    with pytest.raises(FileExistsError):
        check_collection_dir(tmp_path / "used", tmp_path)
    check_collection_dir(tmp_path / "new", tmp_path)
