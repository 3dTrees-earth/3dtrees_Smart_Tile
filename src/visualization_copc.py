"""Visualization COPC: a COPC laid out for streaming viewers (3Dtrees Potree).

Untwine fills every octree node from a 128-cell grid, so the root of a 50 m plot
holds ~276 k points (6.5 MB for dataset 3753) and a phone on 4G waits ~8 s for
its first picture. Here each level keeps one random point per voxel of a coarser
grid (``grid`` cells per node edge, 80 by default, close to PotreeConverter's
Poisson levels), and a region becomes a leaf once at most ``leaf_points`` points
remain. Every point and dimension is kept; only the node a point sits in differs.

Each ``PredInstance*`` dimension also gets the Potree tool's ``coloring_id_*``
dimension and its palette mapping, so the viewer can colour instances without a
Potree conversion.

The whole cloud is held in memory (about 320 B per point for 14 extra
dimensions); stream by spatial blocks before using this for large originals.
"""
from __future__ import annotations

import copy
import io
import json
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from laspy.vlrs.known import WktCoordinateSystemVlr
import numpy as np

import crs_records  # noqa: F401  (laspy then writes CRS records as read)
from visualization_coloring import (
    COLOR_ATTRIBUTE_DTYPE,
    GROUND_COLOR_HEX,
    GROUND_COLORING_ID,
    GROUND_INSTANCE_ID,
    N_COLORS,
    N_NEIGHBORS,
    PALETTE_NAME,
    InstanceStats,
    assign_coloring_ids,
    color_attribute_name_for_instance,
    is_ground_instance_ids,
    map_instances_to_coloring_ids,
)

DEFAULT_GRID = 80
DEFAULT_LEAF_POINTS = 20_000
MAX_DEPTH = 16
LAS14_HEADER_SIZE = 375
VLR_HEADER_SIZE = 54
# LAS point formats without COPC support -> the COPC format holding the same fields.
COPC_POINT_FORMAT = {0: 6, 1: 6, 2: 7, 3: 7, 4: 6, 5: 7, 6: 6, 7: 7, 8: 8, 9: 6, 10: 7}


def morton3(v: np.ndarray) -> np.ndarray:
    """Spread the low 21 bits of ``v`` so three of them interleave into a Morton code."""
    v = v & 0x1FFFFF
    v = (v | v << 32) & 0x1F00000000FFFF
    v = (v | v << 16) & 0x1F0000FF0000FF
    v = (v | v << 8) & 0x100F00F00F00F00F
    v = (v | v << 4) & 0x10C30C30C30C30C3
    v = (v | v << 2) & 0x1249249249249249
    return v


def octree_cube(mins: Sequence[float], maxs: Sequence[float]) -> tuple[np.ndarray, float]:
    """COPC root cube (min corner, edge) around the bounds, as untwine centres it."""
    mins = np.asarray(mins, dtype=np.float64)
    maxs = np.asarray(maxs, dtype=np.float64)
    size = float(np.max(maxs - mins)) or 1.0
    size *= 1.0 + 1e-9
    centre = (mins + maxs) / 2
    return centre - size / 2, size


def assign_nodes(
    xyz: np.ndarray,
    cube_min: np.ndarray,
    size: float,
    grid: int = DEFAULT_GRID,
    leaf_points: int = DEFAULT_LEAF_POINTS,
    max_depth: int = MAX_DEPTH,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return each point's octree level and node key (x, y, z at that level)."""
    n = len(xyz)
    order = np.random.default_rng(seed).permutation(n)  # random pick inside a voxel
    rel = np.clip((xyz[order] - cube_min) / size, 0.0, np.nextafter(1.0, 0.0))
    level = np.full(n, -1, dtype=np.int16)
    remaining = np.arange(n)
    for depth in range(max_depth + 1):
        if remaining.size == 0:
            break
        r = rel[remaining]
        node = np.floor(r * (1 << depth)).astype(np.int64)
        node_id = (node[:, 0] << 42) | (node[:, 1] << 21) | node[:, 2]
        _, inverse, counts = np.unique(node_id, return_inverse=True, return_counts=True)
        take = counts[inverse] <= leaf_points if depth < max_depth else np.ones(remaining.size, bool)
        sampled = np.flatnonzero(~take)
        if sampled.size:
            voxel = np.floor(r[sampled] * (grid << depth)).astype(np.int64)
            voxel_id = (voxel[:, 0] << 42) | (voxel[:, 1] << 21) | voxel[:, 2]
            _, first = np.unique(voxel_id, return_index=True)
            take[sampled[first]] = True
        level[remaining[take]] = depth
        remaining = remaining[~take]
    keys = np.floor(rel * (1 << level.astype(np.int64))[:, None]).astype(np.int64)
    back = np.empty(n, dtype=np.int64)
    back[order] = np.arange(n)
    return level[back], keys[back]


def instance_coloring(las, instance_attribute: str) -> tuple[np.ndarray, dict]:
    """Coloring IDs and mapping for one instance dimension (Potree tool algorithm)."""
    ids = np.asarray(las[instance_attribute]).astype(np.int64)
    stats: Dict[int, InstanceStats] = {}
    ground = is_ground_instance_ids(ids, GROUND_INSTANCE_ID)
    for instance_id, count in zip(*np.unique(ids[ground], return_counts=True)):
        stats[int(instance_id)] = InstanceStats(count=int(count))
    keep = ~ground
    unique_ids, inverse = np.unique(ids[keep], return_inverse=True)
    counts = np.bincount(inverse)
    sums = [np.bincount(inverse, weights=np.asarray(las[axis], dtype=np.float64)[keep]) for axis in "xyz"]
    for index, instance_id in enumerate(unique_ids):
        stats[int(instance_id)] = InstanceStats(int(counts[index]), *(float(s[index]) for s in sums))
    instance_to_coloring, coloring_to_color = assign_coloring_ids(stats)
    keys = np.array(sorted(instance_to_coloring), dtype=np.int64)
    values = np.array([instance_to_coloring[int(k)] for k in keys], dtype=COLOR_ATTRIBUTE_DTYPE)
    coloring = map_instances_to_coloring_ids(ids, keys, values, instance_attribute=instance_attribute)
    mapping = {
        "version": 1,
        "instance_attribute": instance_attribute,
        "ground_instance_id": GROUND_INSTANCE_ID,
        "ground_coloring_id": GROUND_COLORING_ID,
        "ground_color": GROUND_COLOR_HEX,
        "palette": PALETTE_NAME,
        "n_colors": N_COLORS,
        "n_neighbors": N_NEIGHBORS,
        "coloring_id_to_color": {str(k): v for k, v in sorted(coloring_to_color.items())},
        "instance_id_to_coloring_id": {str(k): int(v) for k, v in sorted(instance_to_coloring.items())},
    }
    return coloring, mapping


def _vlr(user_id: str, record_id: int, description: str, data: bytes) -> bytes:
    return struct.pack("<H16sHH32s", 0, user_id.encode(), record_id, len(data), description.encode()) + data


def _walk_vlrs(raw: bytes, count: int, offset: int) -> List[tuple[str, int, bytes]]:
    vlrs = []
    for _ in range(count):
        user_id = raw[offset + 2:offset + 18].split(b"\0")[0].decode()
        record_id, length = struct.unpack_from("<HH", raw, offset + 18)
        vlrs.append((user_id, record_id, raw[offset:offset + VLR_HEADER_SIZE + length]))
        offset += VLR_HEADER_SIZE + length
    return vlrs


def _viewer_las(source, dimensions: Optional[Sequence[str]]):
    """Copy of ``source`` in a COPC point format, keeping only the chosen extra dimensions."""
    import laspy

    extra = list(source.point_format.extra_dimension_names)
    keep = extra if dimensions is None else [d for d in extra if d in set(dimensions)]
    missing = [] if dimensions is None else [d for d in dimensions if d not in extra
                                              and d not in source.point_format.dimension_names]
    if missing:
        raise ValueError(f"Visualization COPC dimensions not in the source: {missing}")
    header = laspy.LasHeader(version="1.4", point_format=COPC_POINT_FORMAT[source.point_format.id])
    header.scales, header.offsets = source.header.scales, source.header.offsets
    by_name = {d.name: d for d in source.point_format.extra_dimensions}
    for name in keep:
        dim = by_name[name]
        header.add_extra_dim(laspy.ExtraBytesParams(
            name=name, type=dim.dtype, description=dim.description,
            scales=dim.scales, offsets=dim.offsets))
    # COPC needs one WKT record: the source's bytes where it has one (last record wins, as readers do).
    wkt = [vlr for vlr in [*source.header.vlrs, *(source.header.evlrs or [])]
           if isinstance(vlr, WktCoordinateSystemVlr) and vlr.string]
    crs = source.header.parse_crs()
    if wkt:
        header.vlrs.append(copy.deepcopy(wkt[-1]))
        header.global_encoding.wkt = True
    elif crs is not None:
        header.add_crs(crs)
    las = laspy.LasData(header)
    las.points = laspy.ScaleAwarePointRecord.zeros(len(source.points), header=header)
    for name in header.point_format.dimension_names:
        if name in source.point_format.dimension_names:
            las[name] = source[name]
    if "scan_angle_rank" in source.point_format.dimension_names:
        las.scan_angle = np.round(np.asarray(source.scan_angle_rank) / 0.006).astype(np.int16)
    return las


def write_visualization_copc(
    source_file: Path,
    output_file: Path,
    *,
    dimensions: Optional[Sequence[str]] = None,
    grid: int = DEFAULT_GRID,
    leaf_points: int = DEFAULT_LEAF_POINTS,
    workers: int = 1,
) -> dict:
    """Write ``output_file`` (``*.copc.laz``) plus one coloring mapping JSON per instance dimension."""
    import laspy
    import lazrs

    started = time.time()
    las = _viewer_las(laspy.read(str(source_file)), dimensions)
    color_names = {}
    for name in [d for d in las.point_format.extra_dimension_names if d.startswith("PredInstance")]:
        color_name = color_attribute_name_for_instance(name, len(color_names) + 1)
        if color_name in las.point_format.dimension_names:
            continue
        coloring, mapping = instance_coloring(las, name)
        las.add_extra_dim(laspy.ExtraBytesParams(name=color_name, type=COLOR_ATTRIBUTE_DTYPE))
        las[color_name] = coloring
        mapping["color_attribute"] = color_name
        color_names[color_name] = mapping

    xyz = np.column_stack([np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)])
    mins, maxs = xyz.min(axis=0), xyz.max(axis=0)
    cube_min, size = octree_cube(mins, maxs)
    level, keys = assign_nodes(xyz, cube_min, size, grid, leaf_points)
    node_id = (level.astype(np.int64) << 60) | (keys[:, 0] << 40) | (keys[:, 1] << 20) | keys[:, 2]
    fine = np.clip(np.floor((xyz - cube_min) / size * (1 << 20)).astype(np.int64), 0, (1 << 20) - 1)
    morton = morton3(fine[:, 0]) | (morton3(fine[:, 1]) << 1) | (morton3(fine[:, 2]) << 2)
    by_time = np.lexsort((np.asarray(las.gps_time), node_id))
    by_space = np.lexsort((morton, node_id))
    sorted_ids = node_id[by_time]
    starts = np.flatnonzero(np.r_[True, sorted_ids[1:] != sorted_ids[:-1]])
    ends = np.r_[starts[1:], len(sorted_ids)]
    nodes = [(int(level[by_time[s]]), *map(int, keys[by_time[s]]), int(s), int(e)) for s, e in zip(starts, ends)]

    records = las.points.array
    time_records, space_records = records[by_time], records[by_space]
    point_size = records.dtype.itemsize
    base_size = laspy.PointFormat(las.point_format.id).size
    laz_vlr = lazrs.LazVlr.new_for_compression(las.point_format.id, point_size - base_size, True)

    def compress(node) -> bytes:
        # Dense leaves compress best in scan (GPS time) order, sparse upper nodes in Morton order.
        _, _, _, _, s, e = node
        best = None
        for chunk in (time_records[s:e], space_records[s:e]):
            out = bytes(lazrs.compress_points(laz_vlr, chunk.tobytes(), False))
            end = int.from_bytes(out[:8], "little", signed=True)
            if best is None or end - 8 < len(best):
                best = out[8:end]
        return best

    with ThreadPoolExecutor(max(1, workers)) as pool:
        chunks = list(pool.map(compress, nodes))

    # laspy writes a valid LAS 1.4 header with the extra-bytes and CRS VLRs; COPC info and
    # laszip VLRs go first, then point counts, bounds and EVLR fields are patched in.
    template = io.BytesIO()
    empty = laspy.LasData(copy.deepcopy(las.header))
    empty.write(template)
    raw = template.getvalue()
    vlr_count = struct.unpack_from("<I", raw, 100)[0]
    kept = [v for user, rid, v in _walk_vlrs(raw, vlr_count, LAS14_HEADER_SIZE)
            if user not in ("copc", "laszip encoded")]
    centre = cube_min + size / 2
    gps = np.asarray(las.gps_time)

    def copc_info(root_offset: int, root_size: int) -> bytes:
        return struct.pack("<5d2Q2d", *centre, size / 2, size / grid, root_offset, root_size,
                           float(gps.min()), float(gps.max())) + bytes(88)

    vlrs = [_vlr("copc", 1, "copc info", copc_info(0, 0)),
            _vlr("laszip encoded", 22204, "lazrs variable chunks", bytes(laz_vlr.record_data())), *kept]
    header = bytearray(raw[:LAS14_HEADER_SIZE])
    point_offset = LAS14_HEADER_SIZE + sum(map(len, vlrs))
    returns = np.bincount(np.clip(np.asarray(las.return_number), 0, 15), minlength=16)[1:16]
    struct.pack_into("<II", header, 96, point_offset, len(vlrs))
    header[104] = las.point_format.id | 0x80
    struct.pack_into("<HI5I", header, 105, point_size, 0, 0, 0, 0, 0, 0)
    struct.pack_into("<6d", header, 179, maxs[0], mins[0], maxs[1], mins[1], maxs[2], mins[2])
    struct.pack_into("<Q15Q", header, 247, len(xyz), *map(int, returns))

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_file.with_name(f".{output_file.name}.tmp")
    with open(tmp, "w+b") as f:
        f.write(header)
        for v in vlrs:
            f.write(v)
        f.write(bytes(8))  # offset to the LAZ chunk table, patched below
        entries = []
        for (d, x, y, z, s, e), chunk in zip(nodes, chunks):
            entries.append(struct.pack("<4iQii", d, x, y, z, f.tell(), len(chunk), e - s))
            f.write(chunk)
        table_offset = f.tell()
        lazrs.write_chunk_table(f, [(e - s, len(c)) for (*_, s, e), c in zip(nodes, chunks)], laz_vlr)
        evlr_start = f.tell()
        hierarchy = b"".join(entries)
        f.write(struct.pack("<H16sHQ32s", 0, b"copc", 1000, len(hierarchy), b"EPT hierarchy"))
        root_offset = f.tell()
        f.write(hierarchy)
        f.seek(point_offset)
        f.write(struct.pack("<q", table_offset))
        f.seek(235)
        f.write(struct.pack("<QI", evlr_start, 1))
        f.seek(LAS14_HEADER_SIZE + VLR_HEADER_SIZE)
        f.write(copc_info(root_offset, len(hierarchy)))
    tmp.replace(output_file)

    stem = output_file.name[:-len(".copc.laz")] if output_file.name.endswith(".copc.laz") else output_file.stem
    mappings = []
    for color_name, mapping in color_names.items():
        path = output_file.with_name(f"{stem}_special_coloring_mapping_{color_name}.json")
        path.write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
        mappings.append(path)

    points_per_level: Dict[int, int] = {}
    for d, *_, s, e in nodes:
        points_per_level[d] = points_per_level.get(d, 0) + e - s
    return {
        "source": str(source_file),
        "output": str(output_file),
        "points": int(len(xyz)),
        "nodes": len(nodes),
        "root_points": points_per_level.get(0, 0),
        "points_per_level": points_per_level,
        "bytes": output_file.stat().st_size,
        "dimensions": list(las.point_format.extra_dimension_names),
        "coloring_mappings": [str(p) for p in mappings],
        "grid": grid,
        "leaf_points": leaf_points,
        "seconds": round(time.time() - started, 1),
    }
