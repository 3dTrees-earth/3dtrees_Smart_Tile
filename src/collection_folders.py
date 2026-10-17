"""--collection-dir: every product of a task in one flat folder per output collection.

After a successful task, each product is hard-linked to
``<collection-dir>/<collection>/<designation>.<ext>``: the designation is the
product's name without its extensions (``file_stem``) and ``ext`` its last
extension, so ``x_subsampled_1cm.copc.laz`` becomes ``tiles_1/x_subsampled_1cm.laz``.
A Galaxy wrapper can then discover each collection from its own folder with a
plain pattern. The task's own output layout is unchanged.
"""
from __future__ import annotations

import os
from pathlib import Path

from output_names import file_stem
from output_publication import publish_files

POINT_CLOUDS = (".laz", ".las", ".ply")
TREE_FILES = {"_trees": "tree_tables", "_trees_info": "tree_info_tables", "_trees_mesh": "tree_meshes"}


def _files(folder: Path, extensions) -> list[Path]:
    """Top-level regular files of ``folder`` with one of ``extensions`` (none if it is missing)."""
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in extensions)


def _existing_ancestor(path: Path) -> Path:
    path = Path(path).absolute()
    while not path.exists():
        path = path.parent
    return path


def check_collection_dir(collection_dir: Path, output_root: Path) -> None:
    """Fail before the task runs if the collection folders cannot be published."""
    collection_dir = Path(collection_dir)
    if collection_dir.is_symlink() or collection_dir.exists() and (
            not collection_dir.is_dir() or any(collection_dir.iterdir())):
        raise FileExistsError(f"--collection-dir must be a new or empty directory: {collection_dir}")
    if os.stat(_existing_ancestor(collection_dir)).st_dev != os.stat(_existing_ancestor(output_root)).st_dev:
        raise ValueError(f"--collection-dir must be on the same filesystem as the outputs (hard links): "
                         f"{collection_dir}")


def tile_collections(output_dir: Path) -> dict[str, list[Path]]:
    """Tile output i (``subsampled_res<i>``) is collection ``tiles_<i>``."""
    collections, index = {}, 1
    while (output_dir / f"subsampled_res{index}").is_dir():
        collections[f"tiles_{index}"] = _files(output_dir / f"subsampled_res{index}", POINT_CLOUDS)
        index += 1
    return collections


def merged_tile_collections(output_tiles: Path, tree_files: Path) -> dict[str, list[Path]]:
    """Merged tiles, and the filtered RayCloudTools tables and meshes by kind."""
    collections = {"merged_tiles": _files(output_tiles, (".laz",))}
    for path in _files(tree_files, (".txt", ".ply")):
        stem = file_stem(path)
        collection = next((TREE_FILES[suffix] for suffix in TREE_FILES if stem.endswith(suffix)), None)
        if collection is None:
            raise ValueError(f"Unexpected RayCloudTools output: {path}")
        collections.setdefault(collection, []).append(path)
    return collections


def original_collections(original_output: Path) -> dict[str, list[Path]]:
    """Originals with predictions; per-original RCT tables stay in the task output."""
    return {"originals_with_predictions": _files(original_output, (".laz", ".las"))}


def merged_product_collections(products) -> dict[str, list[Path]]:
    """Prod-merged products by format: merged_laz, merged_copc and merged_ply."""
    collections = {}
    for path in map(Path, products):
        name = path.name.lower()
        collection = ("merged_copc" if name.endswith(".copc.laz") else
                      "merged_laz" if name.endswith(".laz") else
                      "merged_ply" if name.endswith(".ply") else None)
        if collection is None:
            raise ValueError(f"Unexpected prod-merged product: {path}")
        collections.setdefault(collection, []).append(path)
    return collections


def publish_collections(collection_dir: Path, collections: dict[str, list[Path]],
                        files: dict[str, Path] | None = None) -> None:
    """Hard-link every product to ``collection_dir/<collection>/<designation>.<ext>``.

    ``files`` adds single products at fixed names in ``collection_dir`` itself,
    for example the tile layout JSON. Nothing is published if any link fails.
    """
    collection_dir = Path(collection_dir)
    products = []
    for collection, paths in collections.items():
        designations = {}
        for path in paths:
            designation = file_stem(path)
            if designation.lower() in designations:
                raise ValueError(f"{collection}: {path.name} and {designations[designation.lower()].name} "
                                 f"would both be named {designation}")
            designations[designation.lower()] = path
            products.append((path, collection_dir / collection / f"{designation}{path.suffix.lower()}"))
    products += [(path, collection_dir / name) for name, path in (files or {}).items() if path.is_file()]
    collection_dir.mkdir(parents=True, exist_ok=True)
    for collection in collections:
        (collection_dir / collection).mkdir(exist_ok=True)
    publish_files(products)
    print(f"Collection folders: {collection_dir} ({', '.join(sorted(collections)) or 'none'})")
