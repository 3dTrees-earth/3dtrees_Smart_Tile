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
from point_cloud_metadata import copy_single_source_header, write_retained_evlrs


def instance_tiles(files, instance_dimension):
    """Per positive instance: points per tile and the XY coordinate sums."""
    counts, sums = {}, {}
    for tile, file in enumerate(files):
        with laspy.open(file) as reader:
            for record in reader.chunk_iterator(MAX_BATCH_POINTS):
                ids = np.asarray(record[instance_dimension])
                positive = ids > 0
                if not np.any(positive):
                    continue
                unique, inverse, number = np.unique(ids[positive], return_inverse=True, return_counts=True)
                sx = np.bincount(inverse, np.asarray(record.x)[positive])
                sy = np.bincount(inverse, np.asarray(record.y)[positive])
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


def verify_distinct_instances(files, instance_dimension):
    """Fail when any positive instance ID appears in more than one tile."""
    counts, _ = instance_tiles(files, instance_dimension)
    split = sorted(uid for uid, per_tile in counts.items() if len(per_tile) > 1)
    if split:
        raise ValueError(f"{len(split)} instances span several tiles (e.g. {split[:5]})")


def consolidate_instances(files, instance_dimension, regions, work_dir, report):
    """Move each split instance whole into its owner tile; files are rewritten in place."""
    files = [Path(f) for f in files]
    counts, sums = instance_tiles(files, instance_dimension)
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
    incoming = {}
    writers = {}
    try:
        # Pass 1: rewrite each tile without its departing points; spill them,
        # already in the owner's encoding, to one file per owner.
        for tile, file in enumerate(files):
            moving = {int(i) for i, o in zip(ids, owner_of) if o != tile and tile in counts[int(i)]}
            if not moving:
                continue
            leave = np.array(sorted(moving), dtype=np.uint64)
            temporary = file.with_name(file.stem + ".consolidating.laz")
            with laspy.open(file) as reader, \
                    laspy.open(temporary, mode="w", header=headers[tile]) as writer:
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
                            incoming[owner] = work_dir / f"incoming_{owner:05d}.las"
                            header = copy_single_source_header(headers[owner])
                            writers[owner] = laspy.open(incoming[owner], mode="w", header=header)
                        writers[owner].write_points(_encode_like(part, headers[owner]))
                        metric["moved_points"] += len(part)
                        key = files[owner].name
                        metric["moved_points_per_owner"][key] = metric["moved_points_per_owner"].get(key, 0) + len(part)
                write_retained_evlrs(writer, headers[tile])
            os.replace(temporary, file)
        for writer in writers.values():
            writer.close()
        writers.clear()
        # Pass 2: append each owner's incoming points after its own points.
        for owner, path in sorted(incoming.items()):
            file = files[owner]
            temporary = file.with_name(file.stem + ".consolidating.laz")
            with laspy.open(temporary, mode="w", header=headers[owner]) as writer:
                for source in (file, path):
                    with laspy.open(source) as reader:
                        for record in reader.chunk_iterator(MAX_BATCH_POINTS):
                            writer.write_points(record)
                write_retained_evlrs(writer, headers[owner])
            os.replace(temporary, file)
            path.unlink()
    finally:
        for writer in writers.values():
            writer.close()
    verify_distinct_instances(files, instance_dimension)
    return metric
