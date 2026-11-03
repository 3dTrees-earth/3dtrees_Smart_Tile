"""``merge --instance-manifest``: QSMs of an existing segmentation, per original, without reading points (2.5).

RayCloudTools ``--instance-dimension`` writes per instance tile
``<tile>_trees.txt``, ``<tile>_trees_info.txt``, ``<tile>_trees_mesh.ply``
(``tree_id`` = instance ID) and its terrain ``<tile>_mesh.ply``. With the tile
task's ``qsm_manifest_<dimension>.json`` this task writes, per original:

- ``<original>_trees.txt``, ``<original>_trees_info.txt``: the rows of every
  instance with points in that original, by instance ID. IDs are kept, so a QSM
  joins its points by the same ID. A tree shared by originals goes to each.
- ``<original>_trees_mesh.ply``: those trees' faces, IDs kept.
- ``<original>_terrain_mesh.ply``: the original's RCT terrain when
  ``--rct-terrain-folder`` is given (the ground the QSMs were rooted on),
  otherwise the tiles' terrain triangles, each from the tile whose core holds
  its centroid, over the original's 1 m footprint (``qsm_footprints.npz``).
- ``<original>_qsm_octree/``: the viewer's QSM octree (``qsm_octree``, 3DT-2626).

It fails on an instance in two tiles, on an instance its tile does not own, on
missing tile files and on tables or meshes that disagree, and reports the
instances that got no QSM (``qsm_merge_report.json``).
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np

from instance_tiles import _core_index
from qsm_tiles import FOOTPRINTS_FILE, read_footprints
from raycloud_meshes import TreeMesh, write_tree_mesh
from raycloud_terrain import ORIGINAL_SCOPE, TerrainMesh, _faces, _header, _select_and_weld, _write, cell_keys
from raycloud_tree_files import _tile_key, read_tree_header

INSTANCE_SCOPE = "instance-dimension"  # RayCloudTools --instance-dimension: tree_id is the instance ID
KINDS = {"_trees_info.txt": "info", "_trees.txt": "trees", "_trees_mesh.ply": "mesh", "_mesh.ply": "terrain"}
REPORT = "qsm_merge_report.json"


def tile_files(folder=None, kind_folders=None):
    """{tile: {kind: path}} of the RayCloudTools instance outputs.

    Either one ``folder`` holding them under RCT's names, or ``kind_folders``
    ({kind: folder}, as Galaxy collections arrive): the folder says what a file
    is, its name which tile (``c00_r00.txt``).
    """
    from raycloud_tree_files import folder_files

    found = []
    if folder is not None:
        for path in folder_files(folder):
            kind = next((k for suffix, k in KINDS.items() if path.name.endswith(suffix)), None)
            if kind is not None:
                found.append((kind, path))
    for kind, kind_folder in (kind_folders or {}).items():
        found += [(kind, path) for path in folder_files(kind_folder)]
    files = {}
    for kind, path in found:
        tile = _tile_key(path)
        if kind in files.setdefault(tile, {}):
            raise ValueError(f"Two RayCloudTools {kind} files for tile {tile}: {files[tile][kind].name}, {path.name}")
        files[tile][kind] = path
    return files


def read_table(path):
    """(description, heading, {instance ID: row}) of an RCT tree or tree-info table; rows stay unchanged.

    The first column is ``predinstance``; treeinfo writes it as a decimal (``12.0000``).
    """
    rows = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        description, heading, explicit = read_tree_header(stream, path)
        if not explicit:
            raise ValueError(f"{path.name}: no predinstance column; run RayCloudTools with --instance-dimension")
        for number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = float(line.split(",", 1)[0])
            except ValueError:
                raise ValueError(f"{path.name}: invalid predinstance at row {number}") from None
            uid = int(value)
            if uid != value or uid <= 0 or uid > 2**32 - 1 or uid in rows:
                raise ValueError(f"{path.name}: invalid or duplicate predinstance {value:g} at row {number}")
            rows[uid] = line if line.endswith("\n") else line + ("\r\n" if heading.endswith("\r\n") else "\n")
    return description, heading, rows


def _write_table(path, description, heading, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        stream.write(description)
        stream.write(heading)
        for uid in sorted(rows):
            stream.write(rows[uid])


def _same_header(kind, seen, header, path):
    if seen.setdefault(kind, (header, path))[0] != header:
        raise ValueError(f"{path.name}: {kind} header differs from {seen[kind][1].name}; "
                         "merge only outputs of one RayCloudTools version and parameter set")


def _tile_cores(manifest):
    cores = []
    for tile in manifest["tiles"]:
        core = [(-np.inf if i < 2 else np.inf) if v is None else v for i, v in enumerate(tile["core"])]
        cores.append(tuple(core))
    return cores


def _terrain_from_tiles(output, tile_meshes, cores, footprint, crs):
    """The tiles' triangles over one footprint, each from the tile whose core holds its centroid."""
    def select(mesh, batch):
        xy = mesh.centroids(batch)
        owner = _core_index(xy[:, 0], xy[:, 1], cores)
        return (owner == mesh.tile_index) & np.isin(cell_keys(xy), footprint)

    vertices, faces, welded = _select_and_weld(tile_meshes, select)
    header = _header(ORIGINAL_SCOPE, len(vertices), len(faces),
                     [f"rct_terrain_source_tiles {len(tile_meshes)}", "rct_terrain_footprint_cell_m 1"], crs)
    _write(output, header, [vertices], [_faces(faces)])
    return {"source": "instance tiles", "vertices": int(len(vertices)), "faces": int(len(faces)),
            "welded_vertices": int(welded)}


def merge_qsm(manifest_path, qsm_folder, output_dir, *, dimension=None, rct_terrain_folder=None,
              footprints_path=None, octree=True, kind_folders=None):
    """Write the QSMs of one instance dimension per original; returns the report."""
    from qsm_octree import build as build_octree

    manifest_path, output_dir = Path(manifest_path), Path(output_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if dimension and manifest["instance_dimension"] != dimension:
        raise ValueError(f"{manifest_path.name} is for {manifest['instance_dimension']}, not {dimension}")
    dimension = manifest["instance_dimension"]
    originals, labels = manifest["originals"], [t["label"].lower() for t in manifest["tiles"]]
    ids = np.asarray(manifest["instances"]["ids"], dtype=np.int64)
    owner_tile = dict(zip(ids.tolist(), manifest["instances"]["tile"]))
    holders = dict(zip(ids.tolist(), manifest["instances"]["originals"]))
    files = tile_files(qsm_folder, kind_folders)
    if not files:
        raise ValueError("No RayCloudTools instance outputs given (--qsm-folder or the tree table, info and mesh folders)")
    unknown = sorted(set(files) - set(labels))
    if unknown:
        raise ValueError(f"RayCloudTools outputs for tiles the manifest does not list: {', '.join(unknown)}")

    rows = {"trees": {}, "info": {}}
    tile_of, headers, report_tiles, meshes = {}, {}, [], []
    for index, label in enumerate(labels):
        owned = {uid for uid, tile in owner_tile.items() if tile == index}
        tile = files.get(label, {})
        if not owned and not tile:
            continue
        missing = [k for k in ("trees", "info", "mesh") if k not in tile]
        if missing:
            raise ValueError(f"Tile {label} owns {len(owned)} instances but has no RayCloudTools "
                             f"{', '.join(missing)} file")
        table_ids = {}
        for kind in ("trees", "info"):
            description, heading, table = read_table(tile[kind])
            _same_header(kind, headers, (description, heading), tile[kind])
            table_ids[kind] = set(table)
            for uid, payload in table.items():
                if uid in tile_of:
                    if kind == "trees":
                        raise ValueError(f"Instance {uid} has QSMs in tiles {labels[tile_of[uid]]} and {label}")
                elif kind == "trees":
                    tile_of[uid] = index
                rows[kind][uid] = payload
        if table_ids["trees"] != table_ids["info"]:
            raise ValueError(f"Tile {label}: trees and tree-info tables list different instances")
        stray = sorted(table_ids["trees"] - owned)
        if stray:
            raise ValueError(f"Tile {label} has QSMs of instances it does not own, e.g. {stray[0]}")
        mesh = TreeMesh.open(tile["mesh"])
        if mesh.scope != INSTANCE_SCOPE:
            raise ValueError(f"{tile['mesh'].name}: expected tree IDs of scope {INSTANCE_SCOPE}, found {mesh.scope}")
        meshes.append((index, mesh, table_ids["trees"]))
        report_tiles.append({"tile": label, "instances": len(owned), "qsms": len(table_ids["trees"]),
                             "without_qsm": len(owned - table_ids["trees"])})

    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to write QSMs into a nonempty folder: {output_dir}")
    crs = manifest.get("crs")
    rct_terrain = {}
    if rct_terrain_folder:
        rct_terrain = {p.name[:-len("_terrain_mesh.ply")].lower(): p
                       for p in Path(rct_terrain_folder).glob("*_terrain_mesh.ply")}
    tile_terrain, footprints = [], None
    if not rct_terrain_folder:
        for index, label in enumerate(labels):
            path = files.get(label, {}).get("terrain")
            if path is not None:
                terrain = TerrainMesh.open(path)
                terrain.tile_index = index
                tile_terrain.append(terrain)
        footprints = read_footprints(footprints_path or manifest_path.with_name(FOOTPRINTS_FILE), len(originals))
    cores = _tile_cores(manifest)

    covered_ids = set(tile_of)
    report_originals = []
    for o, stem in enumerate(originals):
        mine = sorted(uid for uid in tile_of if o in holders[uid])
        description, heading = headers["trees"][0]
        _write_table(output_dir / f"{stem}_trees.txt", description, heading, {u: rows["trees"][u] for u in mine})
        description, heading = headers["info"][0]
        info = output_dir / f"{stem}_trees_info.txt"
        _write_table(info, description, heading, {u: rows["info"][u] for u in mine})
        wanted = set(mine)
        sources = [(mesh, {u: u for u in sorted(known & wanted)}) for _, mesh, known in meshes]
        mesh_path = output_dir / f"{stem}_trees_mesh.ply"
        vertices, faces, per_tree = write_tree_mesh(
            mesh_path, [s for s in sources if s[1]], scope=INSTANCE_SCOPE, tree_count=len(mine), crs=crs,
            comments=[f"rct_instance_dimension {dimension}"], allowed_ids=covered_ids)
        terrain_path = output_dir / f"{stem}_terrain_mesh.ply"
        if rct_terrain_folder:
            source = rct_terrain.get(stem.lower())
            if source is None:
                raise ValueError(f"No {stem}_terrain_mesh.ply in {rct_terrain_folder}")
            shutil.copyfile(source, terrain_path)
            terrain = {"source": f"RCT terrain {source.name}"}
        else:
            terrain = _terrain_from_tiles(terrain_path, tile_terrain, cores, footprints[o], crs)
        entry = {"original": stem, "trees": len(mine), "mesh_faces": int(faces),
                 "trees_without_faces": sorted(wanted - per_tree.keys()), "terrain": terrain}
        if octree and mine:
            entry["octree"] = build_octree(info, output_dir / f"{stem}_qsm_octree", mesh=mesh_path)
        report_originals.append(entry)
    del meshes, tile_terrain  # release the memory maps before the caller cleans up

    covered = covered_ids
    report = {
        "instance_dimension": dimension,
        "instances": int(len(ids)),
        "qsms": len(covered),
        "instances_without_qsm": int(len(ids) - len(covered)),
        "tiles": report_tiles,
        "originals": report_originals,
        "terrain": "rct" if rct_terrain_folder else "instance tiles",
    }
    (output_dir / REPORT).write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")
    return report
