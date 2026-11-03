"""What instance tiles need for QSMs per segmentation (SmartTile 2.5).

RayCloudTools ``--instance-dimension`` builds one QSM per instance of a tile;
``merge --instance-manifest`` later writes them per original without reading any point. The
tile task, which reads every point anyway, leaves it what it needs:

- ``qsm_manifest_<dimension>.json``: the originals, each tile's core and actual
  bounds, and every instance with its tile and the originals holding its points;
- ``qsm_footprints.npz``: each original's 1 m footprint cells, to cut a terrain
  per original when no RCT terrain is given;
- with ``--rct-terrain-folder`` (the per-original ``<original>_terrain_mesh.ply``
  of an earlier RCT run), ``<tiles>_terrain/<tile>.ply`` per instance tile: the
  RCT triangles whose XY centroid lies within the tile's bounds plus
  ``TERRAIN_MARGIN_M``. The originals' terrains are merged once first:
  identical vertices welded, triangles that neighbouring originals share once. The
  QSMs of every segmentation then stand on the RCT ground (RCT ``--terrain-mesh``).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from raycloud_terrain import FOOTPRINT_CELL_M, TerrainMesh, _faces, _header, _select_and_weld, _write

TERRAIN_MARGIN_M = 10.0
TILE_TERRAIN_SCOPE = "smarttile-terrain-instance-tile"
MANIFEST_VERSION = 1
FOOTPRINTS_FILE = "qsm_footprints.npz"


def manifest_name(dimension: str) -> str:
    return f"qsm_manifest_{dimension}.json"


def _finite(values):
    return [float(v) if math.isfinite(v) else None for v in values]


def manifest(dimension, originals, tiles, bounds, instance_files, owners, crs):
    """The manifest document of one instance dimension.

    ``instance_files``: (instance, file index) pairs, ``owners``: (instance IDs,
    owner tile index), ``bounds``: {tile label: (xmin, ymin, xmax, ymax)} of the
    written tiles.
    """
    ids, owner = owners
    pair_ids, pair_files = np.asarray(instance_files["instance"]), np.asarray(instance_files["file"])
    order = np.lexsort((pair_files, pair_ids))
    pair_ids, pair_files = pair_ids[order], pair_files[order]
    starts = np.searchsorted(pair_ids, ids)
    ends = np.searchsorted(pair_ids, ids, side="right")
    return {
        "version": MANIFEST_VERSION,
        "instance_dimension": dimension,
        "crs": crs,
        "originals": list(originals),
        "tiles": [{"label": tile.label, "core": _finite(tile.core),
                   "bounds": list(bounds[tile.label]) if tile.label in bounds else None} for tile in tiles],
        "instances": {
            "ids": [int(i) for i in ids],
            "tile": [int(t) for t in owner],
            "originals": [[int(f) for f in pair_files[a:b]] for a, b in zip(starts, ends)],
        },
    }


def write_footprints(path, originals, cells):
    """``cells``: per original the sorted unique 1 m cell keys of its points."""
    np.savez_compressed(path, cell_m=np.float64(FOOTPRINT_CELL_M),
                        **{f"original_{i}": np.asarray(c, dtype=np.int64) for i, c in enumerate(cells)})


def read_footprints(path, count):
    with np.load(path) as data:
        if float(data["cell_m"]) != FOOTPRINT_CELL_M:
            raise ValueError(f"{Path(path).name}: footprint cells of {float(data['cell_m'])} m, expected {FOOTPRINT_CELL_M} m")
        return [np.asarray(data[f"original_{i}"]) for i in range(count)]


def rct_terrain_meshes(folder):
    """The per-original RCT terrain meshes (``<original>_terrain_mesh.ply``) of a folder."""
    meshes = sorted(Path(folder).glob("*_terrain_mesh.ply"), key=lambda p: p.name.lower())
    if not meshes:
        raise ValueError(f"No <original>_terrain_mesh.ply in {folder}")
    return meshes


class MergedTerrain:
    """The per-original RCT terrain meshes as one mesh: identical vertices welded, shared triangles once."""

    def __init__(self, meshes):
        self.sources = len(meshes)
        self.vertices, faces, self.welded = _select_and_weld(
            meshes, lambda mesh, batch: np.ones(len(batch), dtype=bool))
        if len(meshes) > 1:  # neighbouring originals share triangles
            _, first = np.unique(faces, axis=0, return_index=True)
            faces = faces[np.sort(first)]  # first-seen order
        self.shared_faces = int(sum(len(m.faces) for m in meshes) - len(faces))
        self.faces = faces
        self.centroids = self.vertices["xyz"][self.faces][:, :, :2].mean(axis=1)


def write_tile_terrain(output, terrain, bounds, *, margin=TERRAIN_MARGIN_M, crs=None):
    """Write the triangles of the merged RCT terrain whose XY centroid lies within ``bounds`` grown by ``margin``.

    Fails when no triangle is left, since RayCloudTools cannot root a QSM without ground.
    """
    x0, y0, x1, y1 = bounds[0] - margin, bounds[1] - margin, bounds[2] + margin, bounds[3] + margin
    xy = terrain.centroids
    faces = terrain.faces[(xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)]
    if not len(faces):
        raise ValueError(f"{Path(output).name}: the RCT terrain has no triangle within {margin:g} m of the tile")
    used = np.unique(faces)
    header = _header(TILE_TERRAIN_SCOPE, len(used), len(faces),
                     [f"rct_terrain_source_originals {terrain.sources}", f"rct_terrain_margin_m {margin:g}"], crs)
    _write(output, header, [terrain.vertices[used]], [_faces(np.searchsorted(used, faces))])
    return {"vertices": int(len(used)), "faces": int(len(faces))}


def open_rct_terrain(folder):
    """The RCT terrain of ``folder`` merged once, for cropping per tile."""
    return MergedTerrain([TerrainMesh.open(path) for path in rct_terrain_meshes(folder)])


def tile_terrain_directory(tile_directory: Path) -> Path:
    tile_directory = Path(tile_directory)
    return tile_directory.with_name(f"{tile_directory.name}_terrain")


def write_json(path, document):
    Path(path).write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
