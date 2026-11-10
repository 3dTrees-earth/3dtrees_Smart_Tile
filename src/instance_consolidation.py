"""Gather every final instance into one output tile.

Cross-tile reconciliation gives one ID to the parts of a tree that neighbouring
tiles segmented differently, but each part stays in its own tile. Per-tile
tools such as DetailView would then classify one tree several times, each time
from partial geometry (dataset 645: 11% of the instances spanned two tiles).

After deduplication, every positive instance found in more than one tile moves
whole into one owner tile: among the tiles holding it, the one whose core
contains the instance's XY centroid, else the one holding most of its points
(ties: lowest tile index). Points are moved, never dropped or changed, so the
survivor index used for original enrichment stays valid. Background (0) stays
where it is. An owner tile can then extend slightly beyond its planned buffer.
"""
from __future__ import annotations

import os
from pathlib import Path

import laspy
import numpy as np

from bounded_point_index import MAX_BATCH_POINTS
from merge_stages import copy_record
from parallel_tiles import map_tiles
from point_cloud_metadata import copy_single_source_header, write_retained_evlrs


def _tile_chunk_sums(file, instance_dimension):
    """Per chunk of one tile: positive instance IDs, their point counts and XY sums."""
    chunks = []
    with laspy.open(file) as reader:
        for record in reader.chunk_iterator(MAX_BATCH_POINTS):
            ids = np.asarray(record[instance_dimension])
            positive = ids > 0
            if not np.any(positive):
                continue
            unique, inverse, number = np.unique(ids[positive], return_inverse=True, return_counts=True)
            chunks.append((unique, number, np.bincount(inverse, np.asarray(record.x)[positive]),
                           np.bincount(inverse, np.asarray(record.y)[positive])))
    return chunks


def instance_tiles(files, instance_dimension, *, workers=1):
    """Per positive instance: points per tile and the XY coordinate sums.

    Tiles are read in up to ``workers`` processes; the sums are accumulated
    here in tile and chunk order, so they equal a one-process run bit for bit.
    """
    counts, sums = {}, {}
    jobs = [(file, instance_dimension) for file in files]
    sizes = [Path(file).stat().st_size for file in files]
    for tile, chunks in enumerate(map_tiles(_tile_chunk_sums, jobs, processes=max(1, int(workers)), sizes=sizes)):
        for unique, number, sx, sy in chunks:
            for uid, n, x, y in zip(unique.tolist(), number.tolist(), sx, sy):
                per_tile = counts.setdefault(uid, {})
                per_tile[tile] = per_tile.get(tile, 0) + n
                total = sums.setdefault(uid, [0.0, 0.0])
                total[0] += x
                total[1] += y
    return counts, sums


def plan_owners(counts, sums, regions):
    """``{instance: owner tile}`` for every instance present in several tiles."""
    owners = {}
    for uid, per_tile in counts.items():
        if len(per_tile) < 2:
            continue
        n = sum(per_tile.values())
        cx, cy = sums[uid][0] / n, sums[uid][1] / n
        tiles = sorted(per_tile)
        inside = [t for t in tiles
                  if regions[t]["core"][0][0] <= cx <= regions[t]["core"][0][1]
                  and regions[t]["core"][1][0] <= cy <= regions[t]["core"][1][1]]
        owners[uid] = inside[0] if inside else max(tiles, key=lambda t: per_tile[t])
    return owners


def _encode_like(record, header):
    """Copy ``record`` into ``header``'s scale and offsets, refusing int32 overflow."""
    result = copy_record(record, header)
    for axis, name in enumerate(("X", "Y", "Z")):
        values = np.asarray(getattr(record, name.lower()), dtype=np.float64)
        encoded = np.round((values - header.offsets[axis]) / header.scales[axis])
        if encoded.size and (encoded.min() < np.iinfo(np.int32).min or encoded.max() > np.iinfo(np.int32).max):
            raise ValueError("Moved instance points exceed the LAS int32 range of their owner tile")
        result.array[name] = encoded.astype(np.int32)
    return result


def verify_distinct_instances(files, instance_dimension, *, workers=1):
    """Fail when any positive instance ID appears in more than one tile."""
    counts, _ = instance_tiles(files, instance_dimension, workers=workers)
    split = sorted(uid for uid, per_tile in counts.items() if len(per_tile) > 1)
    if split:
        raise ValueError(f"{len(split)} instances span several tiles (e.g. {split[:5]})")


def _send_departing(tile, file, instance_dimension, ids, owner_of, headers, work_dir):
    """Pass 1 for one tile: rewrite it without its departing points.

    Departing points go, already in the owner's encoding, to one LAS per
    (owner, tile). Returns ``[(owner, points), ...]`` in the order the points
    were met, for the report.
    """
    file = Path(file)
    header = headers[tile]
    leave = ids[owner_of != tile]
    temporary = file.with_name(file.stem + ".consolidating.laz")
    writers, moved = {}, []
    try:
        with laspy.open(file) as reader, laspy.open(temporary, mode="w", header=header) as writer:
            for record in reader.chunk_iterator(MAX_BATCH_POINTS):
                labels = np.asarray(record[instance_dimension]).astype(np.uint64)
                out = np.isin(labels, leave)
                writer.write_points(record[~out] if np.any(out) else record)
                if not np.any(out):
                    continue
                destination = owner_of[np.searchsorted(ids, labels[out])]
                departing = record[out]
                for owner in np.unique(destination):
                    part = departing[destination == owner]
                    owner = int(owner)
                    if owner not in writers:
                        writers[owner] = laspy.open(_incoming(work_dir, owner, tile), mode="w",
                                                    header=copy_single_source_header(headers[owner]))
                    writers[owner].write_points(_encode_like(part, headers[owner]))
                    moved.append((owner, len(part)))
            write_retained_evlrs(writer, header)
    finally:
        for part_writer in writers.values():
            part_writer.close()
    os.replace(temporary, file)
    return moved


def _incoming(work_dir, owner, tile):
    return Path(work_dir) / f"incoming_{owner:05d}_{tile:05d}.las"


def _receive_incoming(owner, file, header, sources, work_dir):
    """Pass 2 for one owner: its own points, then the incoming points in tile order.

    The parts are first joined into one LAS, so the owner is written with the
    same chunks as when all departures went to a single file.
    """
    file = Path(file)
    joined = Path(work_dir) / f"incoming_{owner:05d}.las"
    with laspy.open(joined, mode="w", header=copy_single_source_header(header)) as writer:
        for tile in sources:
            part = _incoming(work_dir, owner, tile)
            with laspy.open(part) as reader:
                for record in reader.chunk_iterator(MAX_BATCH_POINTS):
                    writer.write_points(record)
            part.unlink()
    temporary = file.with_name(file.stem + ".consolidating.laz")
    with laspy.open(temporary, mode="w", header=header) as writer:
        for source in (file, joined):
            with laspy.open(source) as reader:
                for record in reader.chunk_iterator(MAX_BATCH_POINTS):
                    writer.write_points(record)
        write_retained_evlrs(writer, header)
    os.replace(temporary, file)
    joined.unlink()


def consolidate_instances(files, instance_dimension, regions, work_dir, report, *, workers=1):
    """Move each split instance whole into its owner tile; files are rewritten in place.

    Both passes treat tiles independently and run in up to ``workers``
    processes; their results are merged in tile order, as one process would.
    """
    files = [Path(f) for f in files]
    processes = max(1, int(workers))
    counts, sums = instance_tiles(files, instance_dimension, workers=processes)
    owners = plan_owners(counts, sums, regions)
    metric = report["instance_consolidation"] = {
        "policy": "each positive instance in exactly one tile: owner core holds its XY centroid, "
                  "else most points (ties: lowest tile); points are moved, never dropped",
        "split_instances": len(owners), "moved_points": 0,
        "moved_points_per_owner": {}}
    if not owners:
        return metric
    ids = np.array(sorted(owners), dtype=np.uint64)
    owner_of = np.array([owners[int(i)] for i in ids], dtype=np.int64)
    headers = []
    for file in files:
        with laspy.open(file) as reader:
            headers.append(copy_single_source_header(reader.header))
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    # Pass 1: tiles losing points, each without its departing points.
    departing = sorted({tile for uid, owner in owners.items() for tile in counts[uid] if tile != owner})
    jobs = []
    for tile in departing:
        present = np.array([tile in counts[int(i)] for i in ids])
        jobs.append((tile, files[tile], instance_dimension, ids[present], owner_of[present], headers, work_dir))
    sizes = [files[tile].stat().st_size for tile in departing]
    sources = {}
    for tile, moved in zip(departing, map_tiles(_send_departing, jobs, processes=processes, sizes=sizes)):
        for owner, points in moved:
            sources.setdefault(owner, [])
            if tile not in sources[owner]:
                sources[owner].append(tile)
            metric["moved_points"] += points
            key = files[owner].name
            metric["moved_points_per_owner"][key] = metric["moved_points_per_owner"].get(key, 0) + points
    # Pass 2: append each owner's incoming points after its own points.
    receiving = sorted(sources)
    jobs = [(owner, files[owner], headers[owner], sources[owner], work_dir) for owner in receiving]
    sizes = [files[owner].stat().st_size for owner in receiving]
    for _ in map_tiles(_receive_incoming, jobs, processes=processes, sizes=sizes):
        pass
    verify_distinct_instances(files, instance_dimension, workers=processes)
    return metric
