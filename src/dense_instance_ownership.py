"""Streaming core ownership on dense predictions, before cross-tile matching."""
from collections import Counter
import json

import laspy
import numpy as np
from scipy.spatial import cKDTree

from bounded_point_index import (MAX_BATCH_POINTS, coordinates, distance_limit,
                                 spatial_batches, inside_xy)
from dense_tile_merge import DUPLICATE_RADIUS, index_file, mapped_values
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
            decisions.append({"instance": uid, "points": size, "anchor_xyz": point.tolist(),
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
                writer.write_points(record[keep])
                metric["input"] += len(record)
                metric["surviving"] += int(np.count_nonzero(keep))
            write_retained_evlrs(writer, header)
        metric["removed"] = metric["input"] - metric["surviving"]
        outputs.append(output)
        index_file(index, output, tile, origin, model)
    return outputs, counts


def instance_owners(mapping, report, admitted=()):
    """Choose one eligible core owner per reconciled instance, deterministically.

    Recovered members that supply unsupported core samples take precedence
    within their reconciled group. Otherwise use stable source filename order.
    """
    recovery_scores = {(row['tile'], row['local_instance']): row['new_locations']
                       for row in admitted}
    grouped = {}
    for key, reconciled in mapping.items():
        grouped.setdefault(reconciled, []).append(key)
    owners = {reconciled: min(keys, key=lambda key: (-recovery_scores.get(key, 0), key))
              for reconciled, keys in grouped.items()}
    report["semantic_ownership"] = {
        "policy": "retain all per-point attributes from the instance owner",
        "tie_break": "recovered member with most uncovered core samples, then source filename/local ID; otherwise first retained core owner",
        "instances": [{"instance": key, "tile": tile, "local_instance": local}
                      for key, (tile, local) in sorted(owners.items())],
        "tiles": []}
    return owners


def retain_instance_owners(model, files, mapping, owners, output_dir, index, origin, report):
    """Drop secondary copies rather than blending or inventing their semantics."""
    output_dir.mkdir(parents=True)
    outputs = []
    for tile, file in enumerate(files):
        removed = [local for (t, local), final in mapping.items()
                   if t == tile and owners[final] != (t, local)]
        metric = {"tile": tile, "removed_instances": removed, "removed_points": 0}
        report["semantic_ownership"]["tiles"].append(metric)
        output = output_dir / file.name
        with laspy.open(file) as reader:
            header = copy_single_source_header(reader.header)
            with laspy.open(output, mode="w", header=header) as writer:
                for record in reader.chunk_iterator(MAX_BATCH_POINTS):
                    keep = ~np.isin(np.asarray(record[model.instance]), removed)
                    writer.write_points(record[keep])
                    metric["removed_points"] += int(np.count_nonzero(~keep))
                write_retained_evlrs(writer, header)
        index_file(index, output, tile, origin, model)
        outputs.append(output)
    return outputs


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
                   else "distinct positive instances: nearest retained claimant core in XY"),
        "tie_break": "stable source tile filename order",
        "boundary": "closed tree cores; preserve unshared buffer points",
        "tiles": []}
    outputs = []
    for tile, file in enumerate(files):
        metric = {"tile": tile, "removed": 0, "removal_examples": []}
        competitors = {other: overlaps[max(tile, other)].get(min(tile, other))
                       for other in range(len(files)) if other != tile}
        competitors = {other: bounds for other, bounds in competitors.items() if bounds is not None}
        stats["tiles"].append(metric)
        output = output_dir / file.name
        offset = 0
        with laspy.open(file) as reader:
            header = copy_single_source_header(reader.header)
            with laspy.open(output, mode="w", header=header) as writer:
                for record in reader.chunk_iterator(MAX_BATCH_POINTS):
                    xyz = coordinates(record, reader.header, origin)
                    values = {model.instance: np.asarray(record[model.instance])}
                    labels = mapped_values(model, values, tile, mapping)[model.instance]
                    eligible = (labels == 0) if background_only else (labels > 0)
                    if not background_only:
                        # Interior core points cannot lose. Restrict expensive
                        # index searches to points a neighboring core could win.
                        could_lose = np.zeros(len(record), dtype=bool)
                        for other, bounds in competitors.items():
                            could_lose |= (inside_xy(xyz, bounds) &
                                           preferred_core(xyz, other, tile, regions, origin))
                        eligible &= could_lose
                    keep = np.ones(len(record), dtype=bool)
                    eligible_positions = np.flatnonzero(eligible)
                    for group in spatial_batches(xyz[eligible_positions]):
                        positions = eligible_positions[group]
                        pts = xyz[positions]
                        limit = distance_limit(pts, DUPLICATE_RADIUS)
                        best_core = core_distance(pts, regions[tile], origin)
                        best_tile = np.full(len(pts), tile, dtype=np.int64)
                        examples = {}
                        for other, data in source_index.candidates(pts, DUPLICATE_RADIUS):
                            if other == tile:
                                continue
                            # The pipeline stores each overlap once, under the later tile.
                            bounds = competitors.get(other)
                            if bounds is None:
                                continue
                            other_labels = mapped_values(model, data['values'], other, mapping)[model.instance]
                            selected = (other_labels > 0) & inside_xy(data['xyz'], bounds)
                            query_allowed = inside_xy(pts, bounds)
                            if not background_only:
                                # Compare at both records' coordinates. Nearby
                                # points across a core bisector keep their own
                                # owners instead of deleting each other.
                                selected &= preferred_core(data['xyz'], other, tile, regions, origin)
                                other_core = core_distance(pts, regions[other], origin)
                                query_allowed &= ((other_core < best_core) |
                                                  ((other_core == best_core) & (other < best_tile)))
                            if not np.any(query_allowed) or not np.any(selected):
                                continue
                            # Every retained tree label outranks background; one
                            # tree query per batch suffices for that pass.
                            groups = ((selected,) if background_only else
                                      (selected & (other_labels == label)
                                       for label in np.unique(other_labels[selected])))
                            for selected_group in groups:
                                subset = np.flatnonzero(selected_group)
                                if not len(subset):
                                    continue
                                d, nearest = source_index.query_tree(cKDTree(data['xyz'][subset]), pts)
                                remove = ((d <= limit) & query_allowed &
                                          (labels[positions] != other_labels[subset[nearest]]))
                                if not background_only:
                                    remove &= ((other_core < best_core) |
                                               ((other_core == best_core) & (other < best_tile)))
                                    best_core[remove] = other_core[remove]
                                    best_tile[remove] = other
                                else:
                                    remove &= keep[positions]
                                example_positions = set(np.flatnonzero(remove)[:5])
                                example_positions.update(pos for pos in examples if remove[pos])
                                for pos in sorted(example_positions):
                                    ref = subset[nearest[pos]]
                                    if pos in examples or len(examples) < 5:
                                        examples[pos] = {
                                            "point": int(offset + positions[pos]),
                                            "owner": [int(other), int(data['indices'][ref])],
                                            "distance_m": float(d[pos])}
                                keep[positions[remove]] = False
                        metric['removal_examples'].extend(
                            list(examples.values())[:5 - len(metric['removal_examples'])])
                    writer.write_points(record[keep])
                    metric['removed'] += int(np.count_nonzero(~keep))
                    offset += len(record)
                write_retained_evlrs(writer, header)
        index_file(index, output, tile, origin, model)
        outputs.append(output)
    return outputs
