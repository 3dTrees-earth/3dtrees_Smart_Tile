"""Keep RayCloudTools QSM tree meshes aligned with the output tree IDs (3DT-2232).

RCT v1.3.0 writes one ``<stem>_trees_mesh.ply`` per tile: binary little-endian
double XYZ + uchar RGBA vertices and triangle faces followed by
``property uint tree_id``, the native tile-local tree ID (1-based tree-table
row). SmartTile applies the same ID mapping as the LAZ and tree tables, drops
the faces of removed trees and unused vertices, and never changes retained
geometry, colours, topology or winding. Terrain meshes carry no tree IDs and
are not handled here.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ply_crs import crs_comment_from_file
from raycloud_tree_files import _prediction_key, _tile_key, rct_sidecar_folder

_MESH_FILE = re.compile(r"^(?P<stem>.+?)_trees_mesh\.ply$", re.IGNORECASE)
NATIVE_SCOPE = "native-rct-extraction"
TILE_SCOPE = "smarttile-rct-tile-namespace"
DATASET_SCOPE = "smarttile-rct-dataset-compact"
BATCH_FACES = 1 << 18
VERTEX = np.dtype([("xyz", "<f8", (3,)), ("rgba", "u1", (4,))])
FACE = np.dtype([("count", "<i4"), ("indices", "<i4", (3,)), ("tree_id", "<u4")])
_SCHEMA = ["ply", "format binary_little_endian 1.0", None,
           "property double x", "property double y", "property double z",
           "property uchar red", "property uchar green", "property uchar blue",
           "property uchar alpha", None, "property list int int vertex_indices",
           "property uint tree_id", "end_header"]


def tree_meshes(collection: Path) -> list[Path]:
    """Find RCT tree meshes co-located with a prediction collection."""
    collection = Path(collection)
    folder = collection if collection.is_dir() else collection.parent
    return sorted((p for p in folder.glob("*.ply") if _MESH_FILE.fullmatch(p.name)),
                  key=lambda p: p.name.lower())


def pair_tree_meshes(prediction_files, meshes) -> dict[int, Path]:
    """Require exactly one tree mesh per prediction tile, or none at all."""
    if not meshes:
        return {}
    by_key = {_prediction_key(p): i for i, p in enumerate(prediction_files)}
    result = {}
    for mesh in meshes:
        key = _tile_key(Path(_MESH_FILE.fullmatch(mesh.name).group("stem")))
        tile = by_key.get(key, 0 if len(prediction_files) == 1 else None)
        if tile is None:
            raise ValueError(f"Cannot match RayCloudTools tree mesh {mesh.name} to a prediction tile")
        if tile in result:
            raise ValueError(f"Duplicate RayCloudTools tree mesh for {prediction_files[tile].name}")
        result[tile] = mesh
    missing = [prediction_files[t].name for t in range(len(prediction_files)) if t not in result]
    if missing:
        raise ValueError(f"Missing RayCloudTools tree mesh for {missing[0]}; supply one per tile or none")
    return result


@dataclass
class TreeMesh:
    """Memory-mapped native RCT tree mesh.

    Mirrors the reader of RayCloudTools ``mesh_export.NativeMesh`` (separate
    image, so it cannot be imported); keep both in step with the native layout.
    """

    path: Path
    vertices: np.ndarray
    faces: np.ndarray
    scope: str
    tree_count: int

    @classmethod
    def open(cls, path):
        path = Path(path)
        with path.open("rb") as handle:
            lines = []
            while True:
                line = handle.readline(4096)
                if not line or len(lines) > 100 or len(line) >= 4096:
                    raise ValueError(f"{path.name}: invalid or oversized PLY header")
                lines.append(line.decode("ascii").strip())
                if lines[-1] == "end_header":
                    break
            offset = handle.tell()
        comments = {}
        for line in lines:
            parts = line.split()
            if len(parts) == 3 and parts[0] == "comment" and parts[1].startswith("rct_"):
                comments[parts[1]] = parts[2]
        schema = [line for line in lines if not line.startswith(("comment", "obj_info"))]
        if "property uint tree_id" not in schema:
            raise ValueError(f"{path.name}: tree mesh has no per-face tree_id; regenerate it with RayCloudTools >= 1.3.0")
        if len(schema) != len(_SCHEMA) or any(e is not None and e != a for e, a in zip(_SCHEMA, schema)):
            raise ValueError(f"{path.name}: unsupported tree mesh schema (expected native RCT double XYZ/RGBA triangles)")
        try:
            nv, nf = int(schema[2].split()[2]), int(schema[10].split()[2])
            assert schema[2].startswith("element vertex ") and schema[10].startswith("element face ")
            assert min(nv, nf) >= 0
            tree_count = int(comments["rct_tree_count"])
            scope = comments["rct_tree_id_scope"]
        except (AssertionError, IndexError, KeyError, ValueError):
            raise ValueError(f"{path.name}: invalid PLY element counts or missing rct_tree_count/rct_tree_id_scope") from None
        if path.stat().st_size != offset + nv * VERTEX.itemsize + nf * FACE.itemsize:
            raise ValueError(f"{path.name}: PLY length does not match the native triangle layout")
        vertices = (np.memmap(path, mode="r", offset=offset, dtype=VERTEX, shape=(nv,))
                    if nv else np.empty(0, dtype=VERTEX))
        faces = (np.memmap(path, mode="r", offset=offset + nv * VERTEX.itemsize, dtype=FACE, shape=(nf,))
                 if nf else np.empty(0, dtype=FACE))
        return cls(path, vertices, faces, scope, tree_count)

    def face_batches(self):
        for start in range(0, len(self.faces), BATCH_FACES):
            batch = self.faces[start:start + BATCH_FACES]
            if np.any(batch["count"] != 3):
                raise ValueError(f"{self.path.name}: non-triangle face")
            if len(batch) and (batch["indices"].min() < 0 or batch["indices"].max() >= len(self.vertices)):
                raise ValueError(f"{self.path.name}: face references a missing vertex")
            yield batch


def write_tree_mesh(output, sources, *, scope, tree_count, comments=(), crs=None, allowed_ids=None):
    """Write selected faces of one or more meshes with remapped tree IDs.

    ``sources`` is a list of ``(TreeMesh, {source_id: output_id})``. Faces of
    unmapped trees are dropped; vertices not used by a kept face are dropped.
    Returns ``(vertices, faces, {output_id: faces})``. Writes atomically.
    ``allowed_ids`` (the tree-table IDs) rejects any face owned by an unknown tree.
    """
    allowed = None if allowed_ids is None else np.fromiter(sorted(allowed_ids), dtype=np.uint64)
    output = Path(output)
    plans, total_vertices, total_faces, per_tree = [], 0, 0, {}
    for mesh, mapping in sources:
        source_ids = np.fromiter(sorted(mapping), dtype=np.uint64)
        target_ids = np.fromiter((mapping[i] for i in sorted(mapping)), dtype=np.uint64)
        used = np.zeros(len(mesh.vertices), dtype=bool)
        faces = 0
        for batch in mesh.face_batches():
            if allowed is not None:
                unknown = batch["tree_id"][~np.isin(batch["tree_id"], allowed)]
                if len(unknown):
                    raise ValueError(f"{mesh.path.name}: tree_id {int(unknown.min())} has no tree row")
            keep = np.isin(batch["tree_id"], source_ids)
            used[batch["indices"][keep].ravel()] = True
            faces += int(keep.sum())
            ids, counts = np.unique(batch["tree_id"][keep], return_counts=True)
            for uid, n in zip(ids.tolist(), counts.tolist()):
                target = int(target_ids[np.searchsorted(source_ids, uid)])
                per_tree[target] = per_tree.get(target, 0) + n
        remap = np.cumsum(used, dtype=np.int64) - 1 + total_vertices
        plans.append((mesh, source_ids, target_ids, used, remap))
        total_vertices += int(used.sum())
        total_faces += faces
    if total_vertices > np.iinfo(np.int32).max:
        raise ValueError(f"{output.name}: too many vertices for native int32 face indices")
    temporary = output.with_name(f".{output.name}.tmp")
    header = ["ply", "format binary_little_endian 1.0",
              f"comment rct_tree_id_scope {scope}", f"comment rct_tree_count {tree_count}",
              *[f"comment {c}" for c in comments],
              *([f"comment crs: {crs}"] if crs else []),
              f"element vertex {total_vertices}",
              "property double x", "property double y", "property double z",
              "property uchar red", "property uchar green", "property uchar blue", "property uchar alpha",
              f"element face {total_faces}", "property list int int vertex_indices",
              "property uint tree_id", "end_header"]
    try:
        with temporary.open("wb") as stream:
            stream.write(("\n".join(header) + "\n").encode("ascii"))
            for mesh, _, _, used, _ in plans:
                for start in range(0, len(mesh.vertices), BATCH_FACES):
                    stream.write(np.ascontiguousarray(mesh.vertices[start:start + BATCH_FACES][used[start:start + BATCH_FACES]]).tobytes())
            for mesh, source_ids, target_ids, _, remap in plans:
                for batch in mesh.face_batches():
                    keep = np.isin(batch["tree_id"], source_ids)
                    out = np.empty(int(keep.sum()), dtype=FACE)
                    out["count"] = 3
                    out["indices"] = remap[batch["indices"][keep]]
                    out["tree_id"] = target_ids[np.searchsorted(source_ids, batch["tree_id"][keep])]
                    stream.write(out.tobytes())
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return total_vertices, total_faces, per_tree


def filter_tree_meshes(pairs, meshes, table_results, output_dir: Path, *, id_mapping):
    """Filter each tile's tree mesh with the ID mapping used for LAZ and tables."""
    sources = [Path(source) for source, _, _ in pairs]
    paired = pair_tree_meshes(sources, meshes)
    results = []
    for tile, path in sorted(paired.items()):
        mesh = TreeMesh.open(path)
        if mesh.scope != NATIVE_SCOPE:
            raise ValueError(f"{path.name}: expected native RCT tree IDs, found scope {mesh.scope}")
        table = next(r for r in table_results if r["tile"] == tile)
        mapping = {uid: id_mapping[(tile, uid)] for uid in table["kept_source_instance_ids"]}
        output = Path(output_dir) / f"{_prediction_key(sources[tile])}_filtered_trees_mesh.ply"
        vertices, faces, per_tree = write_tree_mesh(
            output, [(mesh, mapping)], scope=TILE_SCOPE, tree_count=len(mapping),
            comments=["rct_tree_id_encoding tile_id*100000+local_id"],
            crs=crs_comment_from_file(sources[tile]),
            allowed_ids=set(table["kept_source_instance_ids"]) | set(table["removed_instance_ids"]))
        results.append({"tile": tile, "source": str(path), "output": output.name,
                        "source_faces": int(len(mesh.faces)), "faces": faces, "vertices": vertices,
                        "trees_with_faces": len(per_tree),
                        "retained_trees_without_faces": sorted(set(mapping.values()) - per_tree.keys()),
                        "removed_trees": len(set(table["removed_instance_ids"]))})
    return results


def remap_tree_meshes(collection):
    """Resolve co-located meshes or the meshes published beside merge tables."""
    return tree_meshes(collection) or tree_meshes(rct_sidecar_folder(collection))
