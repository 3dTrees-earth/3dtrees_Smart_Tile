#!/usr/bin/env python3
"""Tile-bound JSON neighbor graph and matching helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


def build_neighbor_graph_from_bounds_json(
    tile_bounds_json: Path,
) -> Tuple[List[Tuple[float, float, float, float]], List[Tuple[float, float]], List[Dict[str, Optional[int]]]]:
    """
    Build a neighbor graph from tile_bounds_tindex.json.

    Returns:
        json_bounds: list of (minx, maxx, miny, maxy) for each JSON tile
        centers: list of (cx, cy) centers for each JSON tile
        neighbors_idx: list of dicts {dir -> neighbor_index or None} for each JSON tile
    """
    if not tile_bounds_json.exists():
        raise FileNotFoundError(f"tile_bounds_tindex.json not found: {tile_bounds_json}")

    with tile_bounds_json.open() as f:
        data = json.load(f)

    tiles = data.get("tiles", [])
    if not tiles:
        raise ValueError(f"No tiles found in tile bounds JSON: {tile_bounds_json}")

    json_bounds: List[Tuple[float, float, float, float]] = []
    centers: List[Tuple[float, float]] = []

    for tile in tiles:
        bx, by = tile["bounds"]
        minx, maxx = float(bx[0]), float(bx[1])
        miny, maxy = float(by[0]), float(by[1])
        json_bounds.append((minx, maxx, miny, maxy))
        centers.append(((minx + maxx) * 0.5, (miny + maxy) * 0.5))

    n = len(json_bounds)
    neighbors_idx: List[Dict[str, Optional[int]]] = [
        {"east": None, "west": None, "north": None, "south": None} for _ in range(n)
    ]

    col_row_to_idx: Dict[Tuple[int, int], int] = {}
    for i, tile in enumerate(tiles):
        if "col" in tile and "row" in tile:
            col_row_to_idx[(int(tile["col"]), int(tile["row"]))] = i

    if col_row_to_idx:
        for i, tile in enumerate(tiles):
            if "col" not in tile or "row" not in tile:
                continue
            c, r = int(tile["col"]), int(tile["row"])
            neighbors_idx[i]["east"] = col_row_to_idx.get((c + 1, r))
            neighbors_idx[i]["west"] = col_row_to_idx.get((c - 1, r))
            neighbors_idx[i]["north"] = col_row_to_idx.get((c, r + 1))
            neighbors_idx[i]["south"] = col_row_to_idx.get((c, r - 1))
        return json_bounds, centers, neighbors_idx

    for i in range(n):
        minx_i, maxx_i, miny_i, maxy_i = json_bounds[i]
        cx_i, cy_i = centers[i]
        best_east = None
        best_west = None
        best_north = None
        best_south = None

        for j in range(n):
            if i == j:
                continue
            minx_j, maxx_j, miny_j, maxy_j = json_bounds[j]
            cx_j, cy_j = centers[j]
            overlap_y = not (maxy_i <= miny_j or maxy_j <= miny_i)
            overlap_x = not (maxx_i <= minx_j or maxx_j <= minx_i)

            if cx_j > cx_i and overlap_y:
                dx = cx_j - cx_i
                if best_east is None or dx < best_east[0]:
                    best_east = (dx, j)
            if cx_j < cx_i and overlap_y:
                dx = cx_i - cx_j
                if best_west is None or dx < best_west[0]:
                    best_west = (dx, j)
            if cy_j > cy_i and overlap_x:
                dy = cy_j - cy_i
                if best_north is None or dy < best_north[0]:
                    best_north = (dy, j)
            if cy_j < cy_i and overlap_x:
                dy = cy_i - cy_j
                if best_south is None or dy < best_south[0]:
                    best_south = (dy, j)

        if best_east is not None:
            neighbors_idx[i]["east"] = best_east[1]
        if best_west is not None:
            neighbors_idx[i]["west"] = best_west[1]
        if best_north is not None:
            neighbors_idx[i]["north"] = best_north[1]
        if best_south is not None:
            neighbors_idx[i]["south"] = best_south[1]

    return json_bounds, centers, neighbors_idx


def match_tiles_to_json_bounds(
    tile_boundaries: Dict[str, Tuple[float, float, float, float]],
    json_bounds: List[Tuple[float, float, float, float]],
    centers: List[Tuple[float, float]],
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Match loaded tiles to JSON tiles with stepwise bounds/centroid tolerances."""
    tile_items = list(tile_boundaries.items())
    tile_to_json: Dict[str, int] = {}
    json_to_tile: Dict[int, str] = {}
    used_json: Set[int] = set()

    if len(tile_items) == 1 and len(json_bounds) == 1:
        name = tile_items[0][0]
        return {name: 0}, {0: name}

    for tol in [0.1, 0.5, 1.0, 2.0, 5.0]:
        for name, bounds in tile_items:
            if name in tile_to_json:
                continue
            minx_a, maxx_a, miny_a, maxy_a = bounds
            best_j = None
            best_l1 = None
            for j, (minx_b, maxx_b, miny_b, maxy_b) in enumerate(json_bounds):
                if j in used_json:
                    continue
                if (
                    abs(minx_a - minx_b) <= tol
                    and abs(maxx_a - maxx_b) <= tol
                    and abs(miny_a - miny_b) <= tol
                    and abs(maxy_a - maxy_b) <= tol
                ):
                    l1 = (
                        abs(minx_a - minx_b)
                        + abs(maxx_a - maxx_b)
                        + abs(miny_a - miny_b)
                        + abs(maxy_a - maxy_b)
                    )
                    if best_l1 is None or l1 < best_l1:
                        best_l1 = l1
                        best_j = j
            if best_j is not None:
                tile_to_json[name] = best_j
                json_to_tile[best_j] = name
                used_json.add(best_j)

        for name, bounds in tile_items:
            if name in tile_to_json:
                continue
            minx_a, maxx_a, miny_a, maxy_a = bounds
            cx_a = (minx_a + maxx_a) * 0.5
            cy_a = (miny_a + maxy_a) * 0.5
            best_j = None
            best_dist = None
            for j, (cx_b, cy_b) in enumerate(centers):
                if j in used_json:
                    continue
                dist = math.hypot(cx_b - cx_a, cy_b - cy_a)
                if dist <= tol and (best_dist is None or dist < best_dist):
                    best_dist = dist
                    best_j = j
            if best_j is not None:
                tile_to_json[name] = best_j
                json_to_tile[best_j] = name
                used_json.add(best_j)

        if len(tile_to_json) == len(tile_boundaries):
            break

    if len(tile_to_json) != len(tile_boundaries):
        unmatched = sorted(set(tile_boundaries.keys()) - set(tile_to_json.keys()))
        raise ValueError(
            "Failed to match all tiles to entries in tile_bounds_tindex.json. "
            f"Unmatched tiles: {', '.join(unmatched)}"
        )

    return tile_to_json, json_to_tile
