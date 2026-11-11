"""The RayCloudTools terrain as a COG, the form the viewer's raster layers read.

An RCT terrain mesh is a height field (one triangulated surface over XY), so it
is sampled on a regular grid: each cell holds the mesh height at its centre. The
grid covers every terrain mesh of a run, so a dataset of several originals gets
one COG (originals that share ground share identical triangles). It shows what
the meshes cover: `<original>_terrain_mesh.ply` is already cut to the original's
footprint, and ground under closed canopy is the mesh's own triangles across it.

The meshes stay the originals; this COG is a view of them.

    python3 terrain_cog.py --mesh 11505_terrain_mesh.ply --out rct_terrain.tif
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from raycloud_terrain import TerrainMesh

TERRAIN_COG = "rct_terrain.tif"
RESOLUTION_M = 0.5
NODATA = -9999.0
# Cells tested at once, whatever the size of the triangles in a batch.
MAX_CANDIDATES = 1 << 24


def _mesh_crs(path) -> str | None:
    """The value of a mesh's ``comment crs:`` header line, if it has one."""
    with Path(path).open("rb") as handle:
        for line in handle:
            text = line.decode("ascii", "replace").strip()
            if text == "end_header":
                return None
            if text.startswith("comment crs: "):
                return text[len("comment crs: "):]
    return None


def _sample(grid, triangles, left, top, resolution):
    """Write the height of ``triangles`` (n, 3, 3, absolute) at each cell centre they hold."""
    height, width = grid.shape
    lo, hi = triangles[:, :, :2].min(axis=1), triangles[:, :, :2].max(axis=1)
    columns = np.ceil((lo[:, 0] - left) / resolution - 0.5).astype(np.int64)
    last_columns = np.floor((hi[:, 0] - left) / resolution - 0.5).astype(np.int64)
    rows = np.ceil((top - hi[:, 1]) / resolution - 0.5).astype(np.int64)
    last_rows = np.floor((top - lo[:, 1]) / resolution - 0.5).astype(np.int64)
    columns, rows = np.maximum(columns, 0), np.maximum(rows, 0)
    wide = np.maximum(np.minimum(last_columns, width - 1) - columns + 1, 0)
    tall = np.maximum(np.minimum(last_rows, height - 1) - rows + 1, 0)
    candidates = wide * tall
    # Most triangles are smaller than a cell and hold no centre.
    holding = np.flatnonzero(candidates)
    ends = np.cumsum(candidates[holding])
    start = 0
    while start < len(holding):
        stop = max(int(np.searchsorted(ends, (ends[start - 1] if start else 0) + MAX_CANDIDATES, "right")), start + 1)
        chosen = holding[start:stop]
        start = stop
        owner = np.repeat(np.arange(len(chosen)), candidates[chosen])
        offsets = np.arange(len(owner)) - np.repeat(np.cumsum(candidates[chosen]) - candidates[chosen], candidates[chosen])
        column = columns[chosen][owner] + offsets % wide[chosen][owner]
        row = rows[chosen][owner] + offsets // wide[chosen][owner]
        x = left + (column + 0.5) * resolution
        y = top - (row + 0.5) * resolution
        a, b, c = (triangles[chosen][owner, i] for i in range(3))
        area = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
        flat = area == 0
        area[flat] = 1.0
        wa = ((b[:, 1] - c[:, 1]) * (x - c[:, 0]) + (c[:, 0] - b[:, 0]) * (y - c[:, 1])) / area
        wb = ((c[:, 1] - a[:, 1]) * (x - c[:, 0]) + (a[:, 0] - c[:, 0]) * (y - c[:, 1])) / area
        wc = 1.0 - wa - wb
        inside = ~flat & (wa >= 0) & (wb >= 0) & (wc >= 0)
        grid[row[inside], column[inside]] = (wa * a[:, 2] + wb * b[:, 2] + wc * c[:, 2])[inside]


def terrain_grid(meshes, resolution=RESOLUTION_M):
    """``(heights as float32, NaN off the meshes, left, top)`` of the meshes on one grid."""
    opened = [TerrainMesh.open(path) for path in meshes]
    opened = [mesh for mesh in opened if len(mesh.faces)]
    if not opened:
        raise ValueError("No terrain triangles to grid")
    xyz = [mesh.vertices["xyz"] for mesh in opened]
    left = np.floor(min(v[:, 0].min() for v in xyz) / resolution) * resolution
    right = np.ceil(max(v[:, 0].max() for v in xyz) / resolution) * resolution
    bottom = np.floor(min(v[:, 1].min() for v in xyz) / resolution) * resolution
    top = np.ceil(max(v[:, 1].max() for v in xyz) / resolution) * resolution
    grid = np.full((int(round((top - bottom) / resolution)), int(round((right - left) / resolution))), np.nan)
    for mesh in opened:
        for _, batch in mesh.face_batches():
            _sample(grid, mesh.vertices["xyz"][batch["indices"]], left, top, resolution)
    return grid.astype(np.float32), float(left), float(top)


def write_terrain_cog(meshes, output, *, resolution=RESOLUTION_M, crs=None) -> dict | None:
    """Write the meshes' heights as one COG; returns what was written, or None without any triangle."""
    from osgeo import gdal, osr

    gdal.UseExceptions()
    meshes = [Path(path) for path in meshes]
    if not any(len(TerrainMesh.open(path).faces) for path in meshes):
        return None
    grid, left, top = terrain_grid(meshes, resolution)
    crs = crs or next(filter(None, map(_mesh_crs, meshes)), None)
    source = gdal.GetDriverByName("MEM").Create("", grid.shape[1], grid.shape[0], 1, gdal.GDT_Float32)
    source.SetGeoTransform((left, resolution, 0, top, 0, -resolution))
    if crs:
        reference = osr.SpatialReference()
        reference.SetFromUserInput(crs)
        source.SetProjection(reference.ExportToWkt())
    band = source.GetRasterBand(1)
    band.SetNoDataValue(NODATA)
    band.WriteArray(np.where(np.isfinite(grid), np.round(grid, 3), NODATA))
    gdal.Translate(str(output), source, format="COG", creationOptions=[
        "COMPRESS=DEFLATE", "PREDICTOR=YES", "BLOCKSIZE=512", "OVERVIEW_RESAMPLING=AVERAGE"])
    return {"file": Path(output).name, "resolution_m": resolution, "width": int(grid.shape[1]),
            "height": int(grid.shape[0]), "cells": int(np.isfinite(grid).sum())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mesh", nargs="+", required=True, help="terrain meshes, for example <original>_terrain_mesh.ply")
    parser.add_argument("--out", default=TERRAIN_COG)
    parser.add_argument("--resolution", type=float, default=RESOLUTION_M, help="cell size (m)")
    parser.add_argument("--crs", help="for meshes without a crs comment, for example EPSG:32618")
    args = parser.parse_args()
    print(write_terrain_cog(args.mesh, args.out, resolution=args.resolution, crs=args.crs))


if __name__ == "__main__":
    main()
