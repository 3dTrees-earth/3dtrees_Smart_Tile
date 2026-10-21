"""Merge RayCloudTools terrain meshes across tiles and cut them per original (3DT-2469).

RCT writes one terrain mesh per tile (``<tile>_mesh.ply``): binary little-endian
double XYZ + uchar RGBA vertices and triangle faces, with no per-face ID. Every
tile's terrain also covers its buffer, so neighbouring tiles overlap.

Merge keeps each tile's triangles whose XY centroid lies in that tile's core,
using the half-open core rule of background points, so every patch of ground
comes from exactly one tile (``<tile>_filtered_terrain_mesh.ply``). The
per-original terrain (``<original>_terrain_mesh.ply``) then holds every kept
triangle whose centroid falls in a 1 m cell that holds at least one of the
original's points. Originals that share ground share identical triangles, as
they share whole trees. Vertices with identical coordinates from different
tiles are welded. Retained geometry, colours and winding never change.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dense_instance_ownership import owned_background
from ply_crs import crs_comment_from_file
from raycloud_meshes import BATCH_FACES, VERTEX
from raycloud_tree_files import _prediction_key, _tile_key, rct_sidecar_folder

FACE = np.dtype([("count", "<i4"), ("indices", "<i4", (3,))])
TILE_SCOPE = "smarttile-terrain-tile-core"
ORIGINAL_SCOPE = "smarttile-terrain-original-footprint"
FOOTPRINT_CELL_M = 1.0
# Seam edges left open after welding lie within this XY distance of an interior core line.
SEAM_BAND_M = 2.0
_NATIVE_FILE = re.compile(r"^(?P<stem>.+?)(?<!_trees)(?<!_terrain)_mesh\.ply$", re.IGNORECASE)
_FILTERED_FILE = re.compile(r"^(?P<stem>.+?)_terrain_mesh\.ply$", re.IGNORECASE)
_SCHEMA = ["ply", "format binary_little_endian 1.0", None,
           "property double x", "property double y", "property double z",
           "property uchar red", "property uchar green", "property uchar blue",
           "property uchar alpha", None, "property list int int vertex_indices", "end_header"]


def terrain_meshes(collection: Path) -> list[Path]:
    """Native RCT terrain meshes (``<tile>_mesh.ply``) co-located with a prediction collection."""
    collection = Path(collection)
    folder = collection if collection.is_dir() else collection.parent
    return sorted((p for p in folder.glob("*.ply") if _NATIVE_FILE.fullmatch(p.name)),
                  key=lambda p: p.name.lower())


def filtered_terrain_meshes(folder: Path) -> list[Path]:
    """Core-filtered tile terrain meshes (``<tile>_filtered_terrain_mesh.ply``) published by merge."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted((p for p in folder.glob("*.ply") if _FILTERED_FILE.fullmatch(p.name)),
                  key=lambda p: p.name.lower())


def remap_terrain_meshes(collection):
    """Resolve co-located tile terrain or the terrain published beside merge tables."""
    folder = Path(collection) if Path(collection).is_dir() else Path(collection).parent
    return filtered_terrain_meshes(folder) or filtered_terrain_meshes(rct_sidecar_folder(collection))


def _terrain_key(path: Path) -> str:
    name = re.sub(r"(_terrain)?_mesh(?=\.ply$)", "", path.name, flags=re.IGNORECASE)
    return _tile_key(path.with_name(name))


def pair_terrain_meshes(prediction_files, meshes) -> dict[int, Path]:
    """At most one terrain mesh per prediction tile; RCT writes none for tiles under 3 ground points."""
    by_key = {_prediction_key(p): i for i, p in enumerate(prediction_files)}
    result = {}
    for mesh in meshes:
        tile = by_key.get(_terrain_key(mesh), 0 if len(prediction_files) == 1 else None)
        if tile is None:
            raise ValueError(f"Cannot match RayCloudTools terrain mesh {mesh.name} to a prediction tile")
        if tile in result:
            raise ValueError(f"Duplicate RayCloudTools terrain mesh for {prediction_files[tile].name}")
        result[tile] = mesh
    return result


@dataclass
class TerrainMesh:
    """Memory-mapped RCT terrain mesh (native layout, no per-face ID)."""

    path: Path
    vertices: np.ndarray
    faces: np.ndarray
    scope: str | None

    @classmethod
    def open(cls, path):
        path = Path(path)
        with path.open("rb") as handle:
            lines = []
            while True:
                line = handle.readline(8192)
                if not line or len(lines) > 100 or len(line) >= 8192:
                    raise ValueError(f"{path.name}: invalid or oversized PLY header")
                lines.append(line.decode("ascii").strip())
                if lines[-1] == "end_header":
                    break
            offset = handle.tell()
        scope = next((line.split()[2] for line in lines
                      if line.startswith("comment rct_terrain_scope ") and len(line.split()) == 3), None)
        schema = [line for line in lines if not line.startswith(("comment", "obj_info"))]
        if len(schema) != len(_SCHEMA) or any(e is not None and e != a for e, a in zip(_SCHEMA, schema)):
            raise ValueError(f"{path.name}: unsupported terrain mesh schema (expected native RCT double XYZ/RGBA triangles)")
        try:
            assert schema[2].startswith("element vertex ") and schema[10].startswith("element face ")
            nv, nf = int(schema[2].split()[2]), int(schema[10].split()[2])
            assert min(nv, nf) >= 0
        except (AssertionError, IndexError, ValueError):
            raise ValueError(f"{path.name}: invalid PLY element counts") from None
        if path.stat().st_size != offset + nv * VERTEX.itemsize + nf * FACE.itemsize:
            raise ValueError(f"{path.name}: PLY length does not match the native triangle layout")
        vertices = (np.memmap(path, mode="r", offset=offset, dtype=VERTEX, shape=(nv,))
                    if nv else np.empty(0, dtype=VERTEX))
        faces = (np.memmap(path, mode="r", offset=offset + nv * VERTEX.itemsize, dtype=FACE, shape=(nf,))
                 if nf else np.empty(0, dtype=FACE))
        return cls(path, vertices, faces, scope)

    def face_batches(self):
        """Yield ``(start, faces)`` batches of validated triangles."""
        for start in range(0, len(self.faces), BATCH_FACES):
            batch = self.faces[start:start + BATCH_FACES]
            if np.any(batch["count"] != 3):
                raise ValueError(f"{self.path.name}: non-triangle face")
            if len(batch) and (batch["indices"].min() < 0 or batch["indices"].max() >= len(self.vertices)):
                raise ValueError(f"{self.path.name}: face references a missing vertex")
            yield start, batch

    def centroids(self, batch):
        """XY centroids of a batch of faces, in absolute coordinates."""
        return self.vertices["xyz"][batch["indices"]][:, :, :2].mean(axis=1)


def _header(scope, vertices, faces, comments, crs):
    return ["ply", "format binary_little_endian 1.0", f"comment rct_terrain_scope {scope}",
            *[f"comment {c}" for c in comments], *([f"comment crs: {crs}"] if crs else []),
            f"element vertex {vertices}",
            "property double x", "property double y", "property double z",
            "property uchar red", "property uchar green", "property uchar blue", "property uchar alpha",
            f"element face {faces}", "property list int int vertex_indices", "end_header"]


def _write(output, header, vertex_chunks, face_chunks):
    output = Path(output)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(("\n".join(header) + "\n").encode("ascii"))
            for chunk in vertex_chunks:
                stream.write(np.ascontiguousarray(chunk).tobytes())
            for chunk in face_chunks:
                stream.write(chunk.tobytes())
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _faces(indices):
    out = np.empty(len(indices), dtype=FACE)
    out["count"] = 3
    out["indices"] = indices
    return out


def filter_terrain_meshes(pairs, meshes, regions, output_dir: Path):
    """Keep each tile's terrain triangles whose XY centroid lies in its half-open core."""
    sources = [Path(source) for source, _, _ in pairs]
    paired = pair_terrain_meshes(sources, meshes)
    results = []
    for tile, path in sorted(paired.items()):
        mesh = TerrainMesh.open(path)
        keep = np.zeros(len(mesh.faces), dtype=bool)
        used = np.zeros(len(mesh.vertices), dtype=bool)
        for start, batch in mesh.face_batches():
            owned = owned_background(mesh.centroids(batch), regions[tile])
            keep[start:start + len(batch)] = owned
            used[batch["indices"][owned].ravel()] = True
        remap = np.cumsum(used, dtype=np.int64) - 1
        output = Path(output_dir) / f"{_prediction_key(sources[tile])}_filtered_terrain_mesh.ply"
        header = _header(TILE_SCOPE, int(used.sum()), int(keep.sum()),
                         [f"rct_terrain_source {path.name}"], crs_comment_from_file(sources[tile]))
        vertex_chunks = (mesh.vertices[s:s + BATCH_FACES][used[s:s + BATCH_FACES]]
                         for s in range(0, len(mesh.vertices), BATCH_FACES))
        face_chunks = (_faces(remap[batch["indices"][keep[s:s + len(batch)]]])
                       for s, batch in mesh.face_batches())
        _write(output, header, vertex_chunks, face_chunks)
        results.append({"tile": tile, "source": str(path), "output": output.name,
                        "source_faces": int(len(mesh.faces)), "faces": int(keep.sum()),
                        "vertices": int(used.sum())})
    missing = [sources[t].name for t in range(len(sources)) if t not in paired]
    return {"tiles": results, "tiles_without_terrain": missing,
            "policy": "triangle XY centroid in the tile's half-open core"}


def cell_keys(xy, cell=FOOTPRINT_CELL_M) -> np.ndarray:
    """int64 keys of the ``cell`` m grid cells holding ``xy`` (absolute coordinates), in input order."""
    ij = np.floor(np.asarray(xy, dtype=np.float64) / cell).astype(np.int64)
    if len(ij) and np.abs(ij).max() >= 1 << 31:
        raise ValueError("Coordinates too large for the terrain footprint grid")
    return (ij[:, 0] << 32) | (ij[:, 1] & 0xFFFFFFFF)


def _interior_lines(regions):
    """Core edges shared with a neighbour, as ((x0, y0), (x1, y1)) segments."""
    lines = set()
    for region in regions:
        (x0, x1), (y0, y1) = region["core"]
        n = region["neighbors"]
        if n["east"] is not None:
            lines.add(((x1, y0), (x1, y1)))
        if n["north"] is not None:
            lines.add(((x0, y1), (x1, y1)))
    return sorted(lines)


def _near_lines(xy, lines, band):
    near = np.zeros(len(xy), dtype=bool)
    for (x0, y0), (x1, y1) in lines:
        dx = np.maximum(np.maximum(min(x0, x1) - xy[:, 0], xy[:, 0] - max(x0, x1)), 0)
        dy = np.maximum(np.maximum(min(y0, y1) - xy[:, 1], xy[:, 1] - max(y0, y1)), 0)
        near |= dx * dx + dy * dy <= band * band
    return near


def _select_and_weld(tile_meshes, select):
    """Faces chosen by ``select(mesh, batch)`` from every tile, with identical vertices welded.

    A welded vertex keeps the first copy in tile order. Returns
    ``(vertices, faces as int64 index triples, welded vertex count)``.
    """
    pieces = []
    for mesh in tile_meshes:
        keep = [batch["indices"][select(mesh, batch)] for _, batch in mesh.face_batches()]
        indices = np.concatenate(keep) if keep else np.empty((0, 3), dtype=np.int32)
        used = np.unique(indices)
        pieces.append((np.asarray(mesh.vertices[used]), np.searchsorted(used, indices)))
    vertices = np.concatenate([v for v, _ in pieces]) if pieces else np.empty(0, dtype=VERTEX)
    keys = np.ascontiguousarray(vertices["xyz"]).view(np.dtype((np.void, 24))).ravel()
    _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    rank = np.empty(len(first), dtype=np.int64)
    rank[np.argsort(first, kind="stable")] = np.arange(len(first))
    faces, offset = [], 0
    for local_vertices, local in pieces:
        faces.append(rank[inverse.ravel()[offset + local]])
        offset += len(local_vertices)
    faces = np.concatenate(faces) if faces else np.empty((0, 3), dtype=np.int64)
    return vertices[np.sort(first)], faces, len(vertices) - len(first)


def write_original_terrain(output, tile_meshes, footprint, *, crs=None, cell=FOOTPRINT_CELL_M):
    """Write the core-filtered tile triangles over one original's footprint as one welded mesh.

    ``footprint`` holds the sorted unique ``cell_keys`` of the original's points.
    A triangle is kept when its XY centroid lies in one of those cells.
    """
    vertices, faces, welded = _select_and_weld(
        tile_meshes, lambda mesh, batch: np.isin(cell_keys(mesh.centroids(batch), cell), footprint))
    if len(vertices) > np.iinfo(np.int32).max:
        raise ValueError(f"{Path(output).name}: too many vertices for native int32 face indices")
    header = _header(ORIGINAL_SCOPE, len(vertices), len(faces),
                     [f"rct_terrain_source_tiles {len(tile_meshes)}", f"rct_terrain_footprint_cell_m {cell:g}"], crs)
    _write(output, header, [vertices], [_faces(faces)])
    return {"vertices": int(len(vertices)), "faces": int(len(faces)), "welded_vertices": int(welded)}


def terrain_seams(tile_meshes, regions):
    """Check that core-filtered tile terrains join along interior core lines.

    Takes the triangles within ``SEAM_BAND_M`` of a line shared by two tiles,
    welds identical vertices and counts boundary edges whose midpoint lies
    within a quarter band of the line. Where neighbouring tiles triangulated
    the shared buffer identically the count is near zero (a few edges where
    the outer edge of the data crosses a seam); a seam that did not weld shows
    about one open edge per triangle on each side.
    """
    lines = _interior_lines(regions)
    report = {"interior_lines": len(lines), "band_m": SEAM_BAND_M}
    if not lines or not tile_meshes:
        return dict(report, faces=0, welded_vertices=0, open_seam_edges=0)
    vertices, faces, welded = _select_and_weld(
        tile_meshes, lambda mesh, batch: _near_lines(mesh.centroids(batch), lines, SEAM_BAND_M))
    report.update(faces=int(len(faces)), welded_vertices=int(welded), open_seam_edges=0)
    if len(faces):
        edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
        keys, counts = np.unique(edges[:, 0] * len(vertices) + edges[:, 1], return_counts=True)
        boundary = keys[counts == 1]
        middle = (vertices["xyz"][boundary // len(vertices), :2] + vertices["xyz"][boundary % len(vertices), :2]) / 2
        report["open_seam_edges"] = int(_near_lines(middle, lines, SEAM_BAND_M / 4).sum())
    return report


def open_tile_terrain(meshes):
    opened = []
    for path in sorted(meshes, key=lambda p: Path(p).name.lower()):
        mesh = TerrainMesh.open(path)
        if mesh.scope != TILE_SCOPE:
            raise ValueError(f"{Path(path).name}: expected SmartTile core-filtered tile terrain, found scope {mesh.scope}; "
                             "use the terrain meshes published by merge")
        opened.append(mesh)
    return opened
