"""Streaming core ownership on dense predictions, before cross-tile matching."""
from collections import Counter
import json
import tempfile
from contextlib import closing
from pathlib import Path
from time import perf_counter

import laspy
import numpy as np
from scipy.spatial import cKDTree

from bounded_point_index import (MAX_BATCH_POINTS, coordinates, distance_limit,
                                 spatial_batches, inside_xy)
from dense_tile_merge import DUPLICATE_RADIUS, index_written_record, mapped_values
from point_cloud_metadata import copy_single_source_header, write_retained_evlrs


ANCHORS = ("centroid", "highest_point", "lowest_point")


def ownership_regions(pairs, tile_bounds_json):
    """Use declared cores and neighbors, including neighbors absent from a subset."""
    from main_remap import get_file_bounds
    from tile_bounds_graph import (build_neighbor_graph_from_bounds_json,
                                   match_tiles_to_json_bounds, is_legacy_single_cloud_layout)

    data = json.loads(tile_bounds_json.read_text())
    if len(pairs) == 1 and is_legacy_single_cloud_layout(data, get_file_bounds(pairs[0][1])):
        x0, x1, y0, y1 = get_file_bounds(pairs[0][1])
        recovery = {"reason": "legacy_single_cloud_bypass",
                    "planned_tile_count": len(data["tiles"]), "extent_tolerance_m": 0.01}
        print(f"Recovered legacy single-cloud layout for {pairs[0][1].name}: "
              f"full projected extent verified; ignoring {len(data['tiles'])} unused planned tiles")
        return [{"layout_tile": None, "core": [[x0, x1], [y0, y1]],
                 "neighbors": dict.fromkeys(("west", "east", "south", "north")),
                 "layout_recovery": recovery}]
    if not data.get("tiles") and len(pairs) == 1:
        # SmartTile's single-file bypass has no grid and therefore no buffer owner.
        x0, x1, y0, y1 = get_file_bounds(pairs[0][1])
        return [{"layout_tile": None, "core": [[x0, x1], [y0, y1]],
                 "neighbors": dict.fromkeys(("west", "east", "south", "north"))}]
    bounds, centers, neighbors = build_neighbor_graph_from_bounds_json(tile_bounds_json)
    mapping, _ = match_tiles_to_json_bounds(
        {str(p[1]): get_file_bounds(p[1]) for p in pairs}, bounds, centers)
    regions = []
    for _, target, _ in pairs:
        idx = mapping[str(target)]
        entry = data["tiles"][idx]
        core = entry.get("core")
        if core is None:
            # Older layouts carry a buffer width instead of explicit core bounds.
            buffer = data.get("tile_buffer")
            if buffer is None and any(n is not None for n in neighbors[idx].values()):
                raise ValueError("Instance ownership requires core bounds or tile_buffer")
            buffer = float(buffer or 0)
            if not np.isfinite(buffer) or buffer < 0:
                raise ValueError("Instance ownership requires a finite nonnegative tile_buffer")
            x0, x1, y0, y1 = bounds[idx]
            n = neighbors[idx]
            core = [[x0 + (buffer if n['west'] is not None else 0),
                     x1 - (buffer if n['east'] is not None else 0)],
                    [y0 + (buffer if n['south'] is not None else 0),
                     y1 - (buffer if n['north'] is not None else 0)]]
        core = np.asarray(core, dtype=np.float64)
        if core.shape != (2, 2) or not np.all(np.isfinite(core)) or np.any(core[:, 0] > core[:, 1]):
            raise ValueError(f"Invalid core bounds for {target.name}")
        regions.append({"layout_tile": idx, "core": core.tolist(),
                        "neighbors": neighbors[idx]})
    return regions


def owned_anchor(xy, region):
    """Match legacy ownership: inclusive core edges, no clipping at outer edges."""
    (x0, x1), (y0, y1) = region["core"]
    n = region["neighbors"]
    return not ((n["west"] is not None and xy[0] < x0) or
                (n["east"] is not None and xy[0] > x1) or
                (n["south"] is not None and xy[1] < y0) or
                (n["north"] is not None and xy[1] > y1))


def owned_background(xy, region):
    """Background has no tree anchor: assign points to half-open spatial cores.

    East/north neighbors own shared upper edges, including in subset replays.
    Dataset exterior edges stay unbounded, as in tree ownership.
    """
    (x0, x1), (y0, y1) = region["core"]
    n = region["neighbors"]
    keep = np.ones(len(xy), dtype=bool)
    for axis, lower, upper, low_side, high_side in (
            (0, x0, x1, "west", "east"), (1, y0, y1, "south", "north")):
        if n[low_side] is not None:
            keep &= xy[:, axis] >= lower
        if n[high_side] is not None:
            keep &= xy[:, axis] < upper
    return keep


def core_distance(xyz, region, origin):
    """XY distance to the closed core rectangle, in common local coordinates."""
    bounds = np.asarray(region['core']) - np.asarray(origin[:2])[:, None]
    delta = np.maximum(np.maximum(bounds[:, 0] - xyz[:, :2],
                                  xyz[:, :2] - bounds[:, 1]), 0.)
    return np.linalg.norm(delta, axis=1)


def preferred_core(xyz, first, second, regions, origin):
    """Rank two tree claimants by core distance, then stable source tile order.

    Only callers with actual retained tree claims may use this comparison.
    Background and absent/removed instances cannot win through core proximity.
    """
    a = core_distance(xyz, regions[first], origin)
    b = core_distance(xyz, regions[second], origin)
    return (a < b) | ((a == b) & (first < second))


def filter_owned_instances(model, files, regions, output_dir, index, origin, report, *, anchor="centroid"):
    """Keep owned tree instances and core-owned background, with their source attributes.

    Only per-instance statistics survive between bounded chunks. Extremum ties
    select the first point in source order, matching the legacy implementation.
    """
    if anchor not in ANCHORS:
        raise ValueError(f"Unknown filter anchor: {anchor}")
    output_dir.mkdir(parents=True)
    stats = report["instance_ownership"] = {
        "anchor": anchor, "geometry": "remapped_1cm", "background": "spatial core owner; half-open east/north edges",
        "semantics": "preserve per-point values from owning tile",
        "boundary": "inclusive core; outer edges retained", "tiles": []}
    outputs, counts = [], Counter()
    for tile, (file, region) in enumerate(zip(files, regions)):
        aggregates = {}
        with laspy.open(file) as reader:
            header = copy_single_source_header(reader.header)
            for record in reader.chunk_iterator(MAX_BATCH_POINTS):
                xyz = coordinates(record, reader.header, origin)
                labels = np.asarray(record[model.instance])
                order = np.argsort(labels, kind="stable")
                ids, starts, sizes = np.unique(labels[order], return_index=True, return_counts=True)
                for uid, start, size in zip(ids, starts, sizes):
                    if uid == 0:
                        continue
                    points = xyz[order[start:start + size]]
                    key = int(uid)
                    if anchor == "centroid":
                        value = points.sum(axis=0)
                    else:
                        pos = points[:, 2].argmax() if anchor == "highest_point" else points[:, 2].argmin()
                        value = points[pos].copy()
                    if key not in aggregates:
                        aggregates[key] = [int(size), value, points.min(axis=0), points.max(axis=0)]
                    else:
                        old = aggregates[key]
                        old[0] += int(size)
                        old[2] = np.minimum(old[2], points.min(axis=0))
                        old[3] = np.maximum(old[3], points.max(axis=0))
                        if anchor == "centroid":
                            old[1] += value
                        elif (anchor == "highest_point" and value[2] > old[1][2]) or (
                                anchor == "lowest_point" and value[2] < old[1][2]):
                            old[1] = value
        decisions = []
        removed = []
        for uid, (size, value, lower, upper) in sorted(aggregates.items()):
            point = (value / size if anchor == "centroid" else value) + origin
            keep = owned_anchor(point[:2], region)
            # Ownership is horizontal: a centroid anchor acts as the XY centroid.
            decisions.append({"instance": uid, "points": size, "anchor_xy": point[:2].tolist(),
                              "anchor_xyz": point.tolist(),
                              "bbox_xyz": [(lower + origin).tolist(), (upper + origin).tolist()],
                              "source_prediction": report["tile_sources"][tile]["prediction"],
                              "normal_kept": keep, "kept": keep,
                              "disposition": "normal_owner" if keep else "rejected_buffer"})
            if keep:
                counts[tile, uid] = size
            else:
                removed.append(uid)
        metric = dict(region, tile=tile, input=0, surviving=0, removed=0,
                      background_input=0, background_removed=0,
                      kept_instances=len(aggregates) - len(removed), removed_instances=len(removed),
                      instances=decisions)
        stats["tiles"].append(metric)
        output = output_dir / file.name
        indexed_offset = 0
        with laspy.open(file) as reader, laspy.open(output, mode="w", header=header) as writer:
            for record in reader.chunk_iterator(MAX_BATCH_POINTS):
                labels = np.asarray(record[model.instance])
                keep = ~np.isin(labels, removed)
                background = labels == 0
                if np.any(background):
                    xyz = coordinates(record, reader.header, origin) + origin
                    keep[background] = owned_background(xyz[background, :2], region)
                metric["background_input"] += int(np.count_nonzero(background))
                metric["background_removed"] += int(np.count_nonzero(background & ~keep))
                retained = record[keep]
                writer.write_points(retained)
                indexed_offset = index_written_record(index, retained, tile, header, origin, model, indexed_offset)
                metric["input"] += len(record)
                metric["surviving"] += int(np.count_nonzero(keep))
            write_retained_evlrs(writer, header)
        metric["removed"] = metric["input"] - metric["surviving"]
        outputs.append(output)
        index.flush()
    return outputs, counts


def ownership_candidates(index, pts, tile, competitors, regions, origin, model,
                         mapping, *, background_only, occupancy=None, block_limit=None):
    """Reuse bounded trees containing only eligible competitor source points.

    Eligibility is fixed for a source tile/current tile pair. Query-point core
    ranking is still checked by the caller. Oversized or mutable indexes retain
    the bounded batch fallback.
    """
    from spatial_query_cache import REGION_SIZE, region_entry
    cell = tuple(np.floor(pts[0, :2] / REGION_SIZE).astype(np.int64))
    limit = distance_limit(pts, DUPLICATE_RADIUS)
    for other, bounds in sorted(competitors.items()):
        allowed = inside_xy(pts, bounds)
        if not background_only:
            allowed &= preferred_core(pts, other, tile, regions, origin)
        if not np.any(allowed):
            continue
        if occupancy is not None and not occupancy.may_contain(other, pts[allowed], limit):
            continue
        select = None if background_only else lambda xyz: preferred_core(
            xyz, other, tile, regions, origin)
        selection_key = None if background_only else (
            'ownership', tile, other, tuple(np.asarray(regions[tile]['core']).ravel()),
            tuple(np.asarray(regions[other]['core']).ravel()), tuple(origin))
        entry = None
        if index._cache_ready and index.query_cache.max_bytes:
            entry = region_entry(index, other, cell, limit if block_limit is None else block_limit, model.instance, bounds,
                                 selection_key=selection_key, select_points=select)
        if entry is not None:
            tree, payload = entry
            if tree is not None:
                yield other, {'indices': payload['indices']}, tree
            continue
        for _, data in index.candidates(pts, DUPLICATE_RADIUS, tile=other):
            selected = (mapped_values(model, data['values'], other, mapping)[model.instance] > 0)
            selected &= inside_xy(data['xyz'], bounds)
            if select is not None:
                selected &= select(data['xyz'])
            if np.any(selected):
                yield other, {'indices': data['indices'][selected]}, cKDTree(data['xyz'][selected])


def assign_shared_points(model, files, source_index, regions, mapping, output_dir,
                         index, origin, report, *, overlaps, background_only=False):
    """Resolve shared points against retained source records.

    First resolve tree/tree claims using nearest-core ownership among claimants.
    Then call with background_only=True and the resulting index: only background
    is removed,
    and every replacement is an actual surviving tree within the radius. This
    ordering keeps background replacements tied to surviving tree points.
    All source attributes travel with the winning tree point.
    """
    output_dir.mkdir(parents=True)
    stats = report["tree_background_ownership" if background_only else "shared_point_ownership"] = {
        "radius_m": DUPLICATE_RADIUS,
        "policy": ("retained tree wins over background, preserving its attributes" if background_only
                   else "all retained tree claims, including merged members: nearest claimant core in XY"),
        "tie_break": "stable source tile filename order",
        "boundary": "closed tree cores; preserve unshared buffer points",
        "tiles": []}
    from ownership_spatial_blocks import OwnershipQuerySpool, TreeOccupancy
    started = perf_counter()
    occupancy = TreeOccupancy(source_index, model.instance)
    stats['occupancy_build_seconds'] = perf_counter() - started
    outputs = []
    for tile, file in enumerate(files):
        metric = {"tile": tile, "removed": 0, "removal_examples": []}
        competitors = {other: overlaps[max(tile, other)].get(min(tile, other))
                       for other in range(len(files)) if other != tile}
        competitors = {other: bounds for other, bounds in competitors.items() if bounds is not None}
        stats["tiles"].append(metric)
        output = output_dir / file.name
        started = perf_counter()
        with tempfile.TemporaryDirectory(prefix='.ownership-blocks-', dir=output_dir.parent) as scratch:
            with closing(OwnershipQuerySpool(Path(scratch) / 'queries.sqlite')) as spool:
                with laspy.open(file) as reader:
                    header = copy_single_source_header(reader.header)
                    point_count = reader.header.point_count
                    offset = 0
                    for record in reader.chunk_iterator(MAX_BATCH_POINTS):
                        xyz = coordinates(record, reader.header, origin)
                        labels = mapped_values(model, {model.instance: np.asarray(record[model.instance])},
                                               tile, mapping)[model.instance]
                        eligible = (labels == 0) if background_only else (labels > 0)
                        if not background_only:
                            could_lose = np.zeros(len(record), dtype=bool)
                            for other, bounds in competitors.items():
                                could_lose |= (inside_xy(xyz, bounds) &
                                               preferred_core(xyz, other, tile, regions, origin))
                            eligible &= could_lose
                        positions = np.flatnonzero(eligible)
                        for group in spatial_batches(xyz[positions]):
                            rows = positions[group]
                            spool.add(xyz[rows], rows + offset, DUPLICATE_RADIUS)
                        offset += len(record)
                metric['query_staging_seconds'] = perf_counter() - started
                removed = (np.memmap(Path(scratch) / 'removed.bin', mode='w+', dtype=bool,
                                     shape=(point_count,)) if point_count else np.empty(0, dtype=bool))
                removed[:] = False
                started = perf_counter()
                for positions, pts, block_limit in spool.groups():
                    limit = distance_limit(pts, DUPLICATE_RADIUS)
                    best_core = core_distance(pts, regions[tile], origin)
                    best_tile = np.full(len(pts), tile, dtype=np.int64)
                    examples = {}
                    for other, data, tree in ownership_candidates(
                            source_index, pts, tile, competitors, regions, origin,
                            model, mapping, background_only=background_only,
                            occupancy=occupancy, block_limit=block_limit):
                        query_allowed = inside_xy(pts, competitors[other])
                        if not background_only:
                            other_core = core_distance(pts, regions[other], origin)
                            query_allowed &= ((other_core < best_core) |
                                              ((other_core == best_core) & (other < best_tile)))
                        if not np.any(query_allowed):
                            continue
                        d, nearest = source_index.query_tree(
                            tree, pts, distance_upper_bound=np.nextafter(limit, np.inf))
                        remove = (d <= limit) & query_allowed
                        if not background_only:
                            best_core[remove] = other_core[remove]
                            best_tile[remove] = other
                        else:
                            remove &= ~removed[positions]
                        example_positions = set(np.flatnonzero(remove)[:5])
                        example_positions.update(pos for pos in examples if remove[pos])
                        for pos in sorted(example_positions):
                            if pos in examples or len(examples) < 5:
                                examples[pos] = {
                                    'point': int(positions[pos]),
                                    'owner': [int(other), int(data['indices'][nearest[pos]])],
                                    'distance_m': float(d[pos])}
                        removed[positions[remove]] = True
                    # Diagnostics remain bounded and deterministic in source row order.
                    metric['removal_examples'] = sorted(
                        metric['removal_examples'] + list(examples.values()),
                        key=lambda value: value['point'])[:5]
                metric.update(query_seconds=perf_counter() - started,
                              query_points=spool.points, query_blocks=spool.blocks)
                started = perf_counter()
                offset = indexed_offset = 0
                with laspy.open(file) as reader, laspy.open(output, mode='w', header=header) as writer:
                    for record in reader.chunk_iterator(MAX_BATCH_POINTS):
                        keep = ~removed[offset:offset + len(record)]
                        retained = record[keep]
                        writer.write_points(retained)
                        indexed_offset = index_written_record(index, retained, tile, header, origin, model, indexed_offset)
                        metric['removed'] += len(record) - len(retained)
                        offset += len(record)
                    write_retained_evlrs(writer, header)
                metric['write_and_index_seconds'] = perf_counter() - started
                if isinstance(removed, np.memmap):
                    removed._mmap.close()
        index.flush()
        outputs.append(output)
    stats['spatial_occupancy'] = occupancy.report()
    return outputs
