"""Visualization COPC: a COPC laid out for streaming viewers (3Dtrees Potree).

Untwine fills every octree node from a 128-cell grid, so the root of a 50 m plot
holds ~276 k points (6.5 MB for dataset 3753) and a phone on 4G waits ~8 s for
its first picture. Here each level keeps one point per voxel of a coarser grid
(``grid`` cells per node edge, 80 by default, close to PotreeConverter's Poisson
levels), and a region becomes a leaf once at most ``leaf_points`` points remain.
Every point and dimension is kept; only the node a point sits in differs.

Each ``PredInstance*`` dimension also gets the Potree tool's ``coloring_id_*``
dimension and its palette mapping, so the viewer can colour instances without a
Potree conversion.

Memory does not grow with the point count. The source is read twice in chunks
(bounds and instance statistics, then the points), and a node with more than
``memory_points`` points is split through files on disk: one pass finds each
voxel's point, a second keeps those in the node and sends the rest to the eight
children. Within a voxel the point with the lowest hash of its record index
(``point_priority``) is kept, so the result is the same whichever path a node
takes. Nodes are compressed as they are found and written in key order.
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
from scratch import scratch_directory
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
MEMORY_POINTS = 4_000_000  # a node with more points is split through files
READ_POINTS = 1_000_000
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


def point_priority(index: np.ndarray) -> np.ndarray:
    """A fixed pseudo-random rank per record index (splitmix64, a bijection, so no ties)."""
    z = np.asarray(index, dtype=np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def relative(xyz: np.ndarray, cube_min: np.ndarray, size: float) -> np.ndarray:
    return np.clip((xyz - cube_min) / size, 0.0, np.nextafter(1.0, 0.0))


def assign_nodes(
    xyz: np.ndarray,
    cube_min: np.ndarray,
    size: float,
    grid: int = DEFAULT_GRID,
    leaf_points: int = DEFAULT_LEAF_POINTS,
    max_depth: int = MAX_DEPTH,
    priority: Optional[np.ndarray] = None,
    start_depth: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return each point's octree level and node key (x, y, z at that level).

    ``priority`` (default: ``point_priority`` of the position) picks the point a
    voxel keeps. From ``start_depth`` on the points must share one node.
    """
    n = len(xyz)
    if priority is None:
        priority = point_priority(np.arange(n))
    order = np.argsort(priority, kind="stable")
    rel = relative(xyz[order], cube_min, size)
    level = np.full(n, -1, dtype=np.int16)
    remaining = np.arange(n)
    for depth in range(start_depth, max_depth + 1):
        if remaining.size == 0:
            break
        r = rel[remaining]
        node = np.floor(r * (1 << depth)).astype(np.int64)
        node_id = (node[:, 0] << 42) | (node[:, 1] << 21) | node[:, 2]
        _, inverse, counts = np.unique(node_id, return_inverse=True, return_counts=True)
        take = counts[inverse] <= leaf_points if depth < max_depth else np.ones(remaining.size, bool)
        sampled = np.flatnonzero(~take)
        if sampled.size:
            voxel_id = _voxel_ids(r[sampled], grid, depth)
            _, first = np.unique(voxel_id, return_index=True)
            take[sampled[first]] = True
        level[remaining[take]] = depth
        remaining = remaining[~take]
    keys = np.floor(rel * (1 << level.astype(np.int64))[:, None]).astype(np.int64)
    back = np.empty(n, dtype=np.int64)
    back[order] = np.arange(n)
    return level[back], keys[back]


def _voxel_ids(rel: np.ndarray, grid: int, depth: int) -> np.ndarray:
    voxel = np.floor(rel * (grid << depth)).astype(np.int64)
    return (voxel[:, 0] << 42) | (voxel[:, 1] << 21) | voxel[:, 2]


def instance_coloring(stats: Dict[int, InstanceStats], instance_attribute: str) -> dict:
    """Coloring mapping for one instance dimension from per-instance statistics (Potree tool algorithm)."""
    instance_to_coloring, coloring_to_color = assign_coloring_ids(stats)
    return {
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


class _InstanceStatistics:
    """Point count and coordinate sums per instance, accumulated chunk by chunk."""

    def __init__(self):
        self.ground: Dict[int, int] = {}
        self.ids = np.zeros(0, dtype=np.int64)
        self.sums = np.zeros((0, 4))

    def add(self, ids: np.ndarray, xyz: np.ndarray) -> None:
        ids = ids.astype(np.int64)
        ground = is_ground_instance_ids(ids, GROUND_INSTANCE_ID)
        for instance_id, count in zip(*np.unique(ids[ground], return_counts=True)):
            self.ground[int(instance_id)] = self.ground.get(int(instance_id), 0) + int(count)
        unique, inverse = np.unique(ids[~ground], return_inverse=True)
        sums = np.column_stack([np.bincount(inverse, minlength=len(unique))]
                               + [np.bincount(inverse, weights=xyz[~ground, axis], minlength=len(unique))
                                  for axis in range(3)])
        merged = np.union1d(self.ids, unique)
        total = np.zeros((len(merged), 4))
        np.add.at(total, np.searchsorted(merged, self.ids), self.sums)
        np.add.at(total, np.searchsorted(merged, unique), sums)
        self.ids, self.sums = merged, total

    def stats(self) -> Dict[int, InstanceStats]:
        stats = {k: InstanceStats(count=v) for k, v in self.ground.items()}
        for instance_id, (count, x, y, z) in zip(self.ids.tolist(), self.sums):
            stats[int(instance_id)] = InstanceStats(int(count), float(x), float(y), float(z))
        return stats


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


def _viewer_header(source_header, dimensions: Optional[Sequence[str]]):
    """The COPC point format and header for a source, with the selected extra dimensions."""
    import laspy

    extra = list(source_header.point_format.extra_dimension_names)
    keep = extra if dimensions is None else [d for d in extra if d in set(dimensions)]
    missing = [] if dimensions is None else [d for d in dimensions if d not in extra
                                              and d not in source_header.point_format.dimension_names]
    if missing:
        raise ValueError(f"Visualization COPC dimensions not in the source: {missing}")
    header = laspy.LasHeader(version="1.4", point_format=COPC_POINT_FORMAT[source_header.point_format.id])
    header.scales, header.offsets = source_header.scales, source_header.offsets
    by_name = {d.name: d for d in source_header.point_format.extra_dimensions}
    for name in keep:
        dim = by_name[name]
        header.add_extra_dim(laspy.ExtraBytesParams(
            name=name, type=dim.dtype, description=dim.description,
            scales=dim.scales, offsets=dim.offsets))
    # COPC needs one WKT record: the source's bytes where it has one (last record wins, as readers do).
    wkt = [vlr for vlr in [*source_header.vlrs, *(source_header.evlrs or [])]
           if isinstance(vlr, WktCoordinateSystemVlr) and vlr.string]
    crs = source_header.parse_crs()
    if wkt:
        header.vlrs.append(copy.deepcopy(wkt[-1]))
        header.global_encoding.wkt = True
    elif crs is not None:
        header.add_crs(crs)
    return header


def _viewer_points(chunk, header):
    """A source chunk in the viewer header's point format: fields by name, extra dimensions raw."""
    import laspy

    points = laspy.ScaleAwarePointRecord.zeros(len(chunk), header=header)
    source = set(chunk.point_format.dimension_names)
    extra = set(header.point_format.extra_dimension_names)
    for name in header.point_format.dimension_names:
        if name in extra and name in source:
            points.array[name] = chunk.array[name]
        elif name in source:
            points[name] = chunk[name]
    if "scan_angle_rank" in source:
        points.scan_angle = np.round(np.asarray(chunk.scan_angle_rank) / 0.006).astype(np.int16)
    return points


def _xyz(records: np.ndarray, scales, offsets) -> np.ndarray:
    return np.column_stack([records[axis] * scales[i] + offsets[i] for i, axis in enumerate("XYZ")])


class _NodeWriter:
    """Compresses nodes as they are found (a few at a time) into a scratch file of LAZ chunks."""

    def __init__(self, path: Path, laz_vlr, scales, offsets, cube_min, size, workers):
        import lazrs

        self.lazrs, self.laz_vlr = lazrs, laz_vlr
        self.scales, self.offsets, self.cube_min, self.size = scales, offsets, cube_min, size
        self.file = open(path, "w+b")
        self.pool = ThreadPoolExecutor(max(1, workers))
        self.pending, self.limit, self.nodes = [], max(1, workers) * 2, []

    def _compress(self, records: np.ndarray, index: np.ndarray) -> bytes:
        # Dense leaves compress best in scan (GPS time) order, sparse upper nodes in Morton order.
        xyz = _xyz(records, self.scales, self.offsets)
        fine = np.clip(np.floor((xyz - self.cube_min) / self.size * (1 << 20)).astype(np.int64), 0, (1 << 20) - 1)
        morton = morton3(fine[:, 0]) | (morton3(fine[:, 1]) << 1) | (morton3(fine[:, 2]) << 2)
        best = None
        for order in (np.lexsort((index, records["gps_time"])), np.lexsort((index, morton))):
            out = bytes(self.lazrs.compress_points(self.laz_vlr, records[order].tobytes(), False))
            end = int.from_bytes(out[:8], "little", signed=True)
            if best is None or end - 8 < len(best):
                best = out[8:end]
        return best

    def add(self, level: int, key, records: np.ndarray, index: np.ndarray) -> None:
        self.pending.append(((level, *map(int, key)), len(records),
                             self.pool.submit(self._compress, records, index)))
        if len(self.pending) >= self.limit:
            self._drain(1)

    def _drain(self, keep: int = 0) -> None:
        while len(self.pending) > keep:
            node, count, future = self.pending.pop(0)
            chunk = future.result()
            self.nodes.append((node, count, self.file.tell(), len(chunk)))
            self.file.write(chunk)

    def close(self):
        self._drain()
        self.pool.shutdown()
        self.file.flush()
        return sorted(self.nodes, key=lambda n: (n[0][0] << 60) | (n[0][1] << 40) | (n[0][2] << 20) | n[0][3])


def _emit_subtree(writer, records, index, xyz, cube_min, size, grid, leaf_points, depth):
    """Every node below one node held in memory (or of the whole cloud from depth 0)."""
    level, keys = assign_nodes(xyz, cube_min, size, grid, leaf_points, priority=point_priority(index),
                               start_depth=depth)
    node_id = (level.astype(np.int64) << 60) | (keys[:, 0] << 40) | (keys[:, 1] << 20) | keys[:, 2]
    order = np.argsort(node_id, kind="stable")
    ids = node_id[order]
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    for s, e in zip(starts, np.r_[starts[1:], len(ids)]):
        rows = order[s:e]
        writer.add(int(level[rows[0]]), keys[rows[0]], records[rows], index[rows])


def _read_spill(path: Path, dtype) -> np.ndarray:
    return np.fromfile(path, dtype=dtype)


def _spill_chunks(path: Path, dtype, count: int):
    with open(path, "rb") as stream:
        while count > 0:
            part = np.fromfile(stream, dtype=dtype, count=min(READ_POINTS, count))
            count -= len(part)
            yield part


def _split_node(writer, path, count, dtype, depth, key, cube_min, size, grid, leaf_points, memory_points,
                scales, offsets):
    """One node, held in memory if small enough, else sampled from its file and split into eight."""
    if count <= memory_points or depth == MAX_DEPTH:
        spill = _read_spill(path, dtype)
        path.unlink()
        records, index = spill["point"], spill["index"]
        _emit_subtree(writer, records, index, _xyz(records, scales, offsets), cube_min, size, grid, leaf_points,
                      depth)
        return
    # More points than leaf_points: keep the lowest-priority point of every voxel of this level.
    voxels, best = np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.uint64)
    for part in _spill_chunks(path, dtype, count):
        ids = _voxel_ids(relative(_xyz(part["point"], scales, offsets), cube_min, size), grid, depth)
        priority = point_priority(part["index"])
        ids, priority = np.concatenate([voxels, ids]), np.concatenate([best, priority])
        order = np.lexsort((priority, ids))
        first = np.r_[True, ids[order][1:] != ids[order][:-1]]
        voxels, best = ids[order][first], priority[order][first]
    kept, children = [], {}
    stem = path.with_suffix("")
    for part in _spill_chunks(path, dtype, count):
        rel = relative(_xyz(part["point"], scales, offsets), cube_min, size)
        ids = _voxel_ids(rel, grid, depth)
        mine = best[np.searchsorted(voxels, ids)] == point_priority(part["index"])
        kept.append(part[mine])
        child = np.floor(rel[~mine] * (1 << (depth + 1))).astype(np.int64)
        octant = (child - 2 * np.asarray(key)) @ np.array([4, 2, 1])
        rest = part[~mine]
        for k in np.unique(octant):
            target = Path(f"{stem}{k}.bin")
            with open(target, "ab") as out:
                rest[octant == k].tofile(out)
            children[int(k)] = children.get(int(k), 0) + int(np.count_nonzero(octant == k))
    path.unlink()
    kept = np.concatenate(kept)
    writer.add(depth, key, kept["point"], kept["index"])
    for k, n in sorted(children.items()):
        child_key = tuple(2 * np.asarray(key) + np.array([(k >> 2) & 1, (k >> 1) & 1, k & 1]))
        _split_node(writer, Path(f"{stem}{k}.bin"), n, dtype, depth + 1, child_key, cube_min, size, grid,
                    leaf_points, memory_points, scales, offsets)


def write_visualization_copc(
    source_file: Path,
    output_file: Path,
    *,
    dimensions: Optional[Sequence[str]] = None,
    grid: int = DEFAULT_GRID,
    leaf_points: int = DEFAULT_LEAF_POINTS,
    workers: int = 1,
    memory_points: int = MEMORY_POINTS,
) -> dict:
    """Write ``output_file`` (``*.copc.laz``) plus one coloring mapping JSON per instance dimension."""
    import laspy
    import lazrs

    started = time.time()
    source_file, output_file = Path(source_file), Path(output_file)
    with laspy.open(str(source_file)) as reader:
        header = _viewer_header(reader.header, dimensions)
    instances = [d for d in header.point_format.extra_dimension_names if d.startswith("PredInstance")]

    # Pass 1: bounds, GPS time range, returns and per-instance statistics.
    mins, maxs = np.full(3, np.inf), np.full(3, -np.inf)
    gps, returns, total = [np.inf, -np.inf], np.zeros(16, dtype=np.int64), 0
    statistics = {name: _InstanceStatistics() for name in instances}
    with laspy.open(str(source_file)) as reader:
        for chunk in reader.chunk_iterator(READ_POINTS):
            points = _viewer_points(chunk, header)
            xyz = _xyz(points.array, header.scales, header.offsets)
            mins, maxs = np.minimum(mins, xyz.min(axis=0)), np.maximum(maxs, xyz.max(axis=0))
            gps = [min(gps[0], float(points.array["gps_time"].min())), max(gps[1], float(points.array["gps_time"].max()))]
            returns += np.bincount(np.clip(np.asarray(points.return_number), 0, 15), minlength=16)
            total += len(points)
            for name in instances:
                statistics[name].add(np.asarray(points[name]), xyz)
    if not total:
        raise ValueError(f"{source_file.name}: no points")
    color_names = {}
    for name in instances:
        color_name = color_attribute_name_for_instance(name, len(color_names) + 1)
        if color_name in header.point_format.dimension_names:
            continue
        mapping = instance_coloring(statistics[name].stats(), name)
        mapping["color_attribute"] = color_name
        color_names[color_name] = mapping
        header.add_extra_dim(laspy.ExtraBytesParams(name=color_name, type=COLOR_ATTRIBUTE_DTYPE))
    lookups = {}
    for color_name, mapping in color_names.items():
        keys = np.array(sorted(int(k) for k in mapping["instance_id_to_coloring_id"]), dtype=np.int64)
        values = np.array([mapping["instance_id_to_coloring_id"][str(k)] for k in keys], dtype=COLOR_ATTRIBUTE_DTYPE)
        lookups[color_name] = (mapping["instance_attribute"], keys, values)

    cube_min, size = octree_cube(mins, maxs)
    point_dtype = laspy.ScaleAwarePointRecord.zeros(0, header=header).array.dtype
    spill_dtype = np.dtype([("point", point_dtype), ("index", "<u8")])
    point_size = point_dtype.itemsize
    laz_vlr = lazrs.LazVlr.new_for_compression(header.point_format.id,
                                               point_size - laspy.PointFormat(header.point_format.id).size, True)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with scratch_directory(".visualization-copc-", dir=output_file.parent) as scratch:
        scratch = Path(scratch)
        # Pass 2: the points in the viewer format with their coloring IDs, into the root node's file.
        root = scratch / "n.bin"
        with laspy.open(str(source_file)) as reader, open(root, "wb") as out:
            index = 0
            for chunk in reader.chunk_iterator(READ_POINTS):
                points = laspy.ScaleAwarePointRecord.zeros(len(chunk), header=header)
                converted = _viewer_points(chunk, header)
                for name in converted.array.dtype.names:
                    points.array[name] = converted.array[name]
                for color_name, (attribute, keys, values) in lookups.items():
                    points.array[color_name] = map_instances_to_coloring_ids(
                        np.asarray(points[attribute]), keys, values, instance_attribute=attribute)
                spill = np.empty(len(points), dtype=spill_dtype)
                spill["point"], spill["index"] = points.array, np.arange(index, index + len(points))
                spill.tofile(out)
                index += len(points)
        writer = _NodeWriter(scratch / "chunks.laz", laz_vlr, header.scales, header.offsets, cube_min, size, workers)
        _split_node(writer, root, total, spill_dtype, 0, (0, 0, 0), cube_min, size, grid, leaf_points,
                    max(memory_points, leaf_points), header.scales, header.offsets)
        nodes = writer.close()
        _write_copc(output_file, header, nodes, writer.file, laz_vlr, cube_min, size, grid, gps, mins, maxs,
                    returns, total, point_size)
        writer.file.close()

    stem = output_file.name[:-len(".copc.laz")] if output_file.name.endswith(".copc.laz") else output_file.stem
    mappings = []
    for color_name, mapping in color_names.items():
        path = output_file.with_name(f"{stem}_special_coloring_mapping_{color_name}.json")
        path.write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
        mappings.append(path)

    points_per_level: Dict[int, int] = {}
    for (d, *_), count, _, _ in nodes:
        points_per_level[d] = points_per_level.get(d, 0) + count
    return {
        "source": str(source_file),
        "output": str(output_file),
        "points": int(total),
        "nodes": len(nodes),
        "root_points": points_per_level.get(0, 0),
        "points_per_level": points_per_level,
        "bytes": output_file.stat().st_size,
        "dimensions": list(header.point_format.extra_dimension_names),
        "coloring_mappings": [str(p) for p in mappings],
        "grid": grid,
        "leaf_points": leaf_points,
        "seconds": round(time.time() - started, 1),
    }


def _write_copc(output_file, header_template, nodes, chunks, laz_vlr, cube_min, size, grid, gps, mins, maxs,
                returns, total, point_size):
    """The COPC file: header, COPC and LAZ VLRs, the nodes' chunks in key order, chunk table, hierarchy."""
    import laspy
    import lazrs

    # laspy writes a valid LAS 1.4 header with the extra-bytes and CRS VLRs; COPC info and
    # laszip VLRs go first, then point counts, bounds and EVLR fields are patched in.
    template = io.BytesIO()
    laspy.LasData(copy.deepcopy(header_template)).write(template)
    raw = template.getvalue()
    vlr_count = struct.unpack_from("<I", raw, 100)[0]
    kept = [v for user, rid, v in _walk_vlrs(raw, vlr_count, LAS14_HEADER_SIZE)
            if user not in ("copc", "laszip encoded")]
    centre = cube_min + size / 2

    def copc_info(root_offset: int, root_size: int) -> bytes:
        return struct.pack("<5d2Q2d", *centre, size / 2, size / grid, root_offset, root_size,
                           float(gps[0]), float(gps[1])) + bytes(88)

    vlrs = [_vlr("copc", 1, "copc info", copc_info(0, 0)),
            _vlr("laszip encoded", 22204, "lazrs variable chunks", bytes(laz_vlr.record_data())), *kept]
    header = bytearray(raw[:LAS14_HEADER_SIZE])
    point_offset = LAS14_HEADER_SIZE + sum(map(len, vlrs))
    struct.pack_into("<II", header, 96, point_offset, len(vlrs))
    header[104] = header_template.point_format.id | 0x80
    struct.pack_into("<HI5I", header, 105, point_size, 0, 0, 0, 0, 0, 0)
    struct.pack_into("<6d", header, 179, maxs[0], mins[0], maxs[1], mins[1], maxs[2], mins[2])
    struct.pack_into("<Q15Q", header, 247, int(total), *map(int, returns[1:16]))

    tmp = output_file.with_name(f".{output_file.name}.tmp")
    with open(tmp, "w+b") as f:
        f.write(header)
        for v in vlrs:
            f.write(v)
        f.write(bytes(8))  # offset to the LAZ chunk table, patched below
        entries = []
        for (d, x, y, z), count, offset, length in nodes:
            entries.append(struct.pack("<4iQii", d, x, y, z, f.tell(), length, count))
            chunks.seek(offset)
            f.write(chunks.read(length))
        table_offset = f.tell()
        lazrs.write_chunk_table(f, [(count, length) for _, count, _, length in nodes], laz_vlr)
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
