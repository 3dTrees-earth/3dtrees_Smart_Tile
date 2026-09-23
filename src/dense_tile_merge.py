"""Strict per-model dense transfer, instance reconciliation and point deduplication."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import laspy
import numpy as np

from bounded_point_index import MAX_BATCH_POINTS, PointIndex, coordinates
from point_cloud_metadata import (
    copy_single_source_header, extra_bytes_params_from_dimension_info, extra_bytes_attribute_equal,
    update_extra_dimensions, write_retained_evlrs,
)
from prediction_collection_remap import prediction_collection_files, _promote_collection_extra_dim
from instance_labels import instance_extra_bytes_params


DUPLICATE_RADIUS = 0.01
ORIGINAL_RADIUS = 0.01

# A model may downgrade the LAS point format and carry native fields (notably
# SAT scan_angle) as ExtraBytes. Dense targets remain their authoritative source.
STANDARD_DIMENSIONS = frozenset(
    name for point_format in range(11)
    for name in laspy.PointFormat(point_format).standard_dimension_names
)


@dataclass
class Model:
    name: str
    source: Path
    instance: str | None
    semantic: str | None
    dimensions: dict
    no_data: dict


def describe_model(source, instance_dimension, *, require_instance=True):
    files = prediction_collection_files(source)
    if not files:
        raise ValueError(f"No prediction files in {source}")
    with laspy.open(files[0]) as reader:
        names = list(reader.header.point_format.extra_dimension_names)
        instances = [n for n in names if n.startswith("PredInstance")]
        if instance_dimension in names and instance_dimension not in instances:
            instances.append(instance_dimension)
        if not instances and "treeID" in names:
            instances.append("treeID")
        if len(instances) > 1 or (not instances and require_instance):
            raise ValueError(f"{source}: expected one model instance dimension, got {instances}; "
                             "supply each model as a separate collection")
        instance = instances[0] if instances else None
        semantic = instance.replace("PredInstance", "PredSemantic", 1) if instance else None
        semantic = semantic if semantic != instance and semantic in names else None
        dimensions = {d.name: extra_bytes_params_from_dimension_info(d, header=reader.header)
                      for d in reader.header.point_format.extra_dimensions
                      if d.name not in STANDARD_DIMENSIONS}
    for file in files:
        with laspy.open(file) as reader:
            actual = {d.name: extra_bytes_params_from_dimension_info(d, header=reader.header)
                      for d in reader.header.point_format.extra_dimensions
                      if d.name not in STANDARD_DIMENSIONS}
            if actual.keys() != dimensions.keys():
                raise ValueError(f"{source}: inconsistent prediction schema in {file.name}")
            for name, expected in dimensions.items():
                got = actual[name]
                if any(not extra_bytes_attribute_equal(getattr(expected, key), getattr(got, key))
                       for key in ("scales", "offsets", "no_data")):
                    raise ValueError(f"{file.name}: inconsistent prediction schema for {name}")
                if not extra_bytes_attribute_equal(expected.type, got.type):
                    # Instance widths may vary between tiles; promote before indexing.
                    if name != instance or {np.dtype(expected.type), np.dtype(got.type)} != {np.dtype("uint16"), np.dtype("uint32")}:
                        raise ValueError(f"{file.name}: inconsistent prediction schema for {name}")
                    dimensions[name] = _promote_collection_extra_dim(expected, reader.header.point_format.dimension_by_name(name))
    no_data = {name: param.no_data for name, param in dimensions.items()}
    for name in (instance, semantic):
        if name is None:
            continue
        param = dimensions[name]
        if ((param.scales is not None and np.any(np.asarray(param.scales) != 1)) or
            (param.offsets is not None and np.any(np.asarray(param.offsets) != 0))):
            raise ValueError(f"{source}: scaled prediction label {name} needs explicit normalization")
    return Model(Path(source).name, Path(source), instance, semantic, dimensions, no_data)


def validate_labels(model, record, source):
    for name in (model.instance, model.semantic):
        if name is None:
            continue
        values = np.asarray(record[name])
        if values.ndim != 1 or np.any(~np.isfinite(values)) or np.any(values < 0) or np.any(values != np.floor(values)):
            raise ValueError(f"{model.name}/{source}: invalid or missing {name} labels")
        if name == model.instance and np.any(values > np.iinfo(np.uint32).max):
            raise ValueError(f"{model.name}/{source}: instance IDs exceed uint32")
        # Zero is valid model background. A declared no-data sentinel is missing.
        missing = model.no_data[name]
        if missing is not None and np.any(np.isin(values, missing)):
            raise ValueError(f"{model.name}/{source}: missing {name} prediction (no-data sentinel)")


def prediction_values(record, dimensions):
    # Raw ExtraBytes values preserve vector shape, scaling and no-data bits.
    return {name: record.array[name].copy() for name in dimensions}


def copy_record(record, header):
    result = laspy.ScaleAwarePointRecord.zeros(len(record), header=header)
    for name in record.array.dtype.names:
        if name in result.array.dtype.names:
            result.array[name] = record.array[name]
    return result


def target_header(target, model, *, ready):
    header = copy_single_source_header(target)
    existing = set(header.point_format.dimension_names)
    for name in model.dimensions:
        if name in existing:
            if not ready:
                raise ValueError(f"Target already contains prediction dimension {name}")
    update_extra_dimensions(header, model.dimensions.values(), replace=ready)
    return header


def index_file(index, file, tile, origin, model):
    offset = 0
    maximum = 0
    with laspy.open(file) as reader:
        for record in reader.chunk_iterator(MAX_BATCH_POINTS):
            validate_labels(model, record, file.name)
            if model.instance and len(record):
                maximum = max(maximum, int(np.max(record[model.instance])))
            index.add(tile, coordinates(record, reader.header, origin),
                      prediction_values(record, model.dimensions),
                      np.arange(offset, offset + len(record), dtype=np.int64))
            offset += len(record)
    index.flush()
    return maximum


def prepare_dense(model, pairs, output_dir, index, origin, transfer_radius, report, *, ready=False):
    """Transfer each tile's own predictions, requiring assignment of every target."""
    output_dir.mkdir(parents=True)
    files = []
    counts = Counter()
    for tile, (source, target, _) in enumerate(pairs):
        source_index_path = output_dir / f"source_{tile}.sqlite"
        with PointIndex(source_index_path, {n: p.type for n, p in model.dimensions.items()},
                        query_workers=index.query_workers) as source_index:
            if not ready:
                index_file(source_index, source, 0, origin, model)
            output = output_dir / f"tile_{tile:05d}.laz"
            metric = {"source": str(source), "target": str(target), "tile": tile,
                      "total": 0, "matched": 0, "missing_examples": [],
                      "radius_m": None if ready else transfer_radius}
            report.setdefault("transfer", []).append(metric)
            with laspy.open(target) as reader:
                header = target_header(reader.header, model, ready=ready)
                with laspy.open(output, mode="w", header=header) as writer:
                    for record in reader.chunk_iterator(MAX_BATCH_POINTS):
                        xyz = coordinates(record, reader.header, origin)
                        if ready:
                            validate_labels(model, record, source.name)
                            matched = np.ones(len(record), dtype=bool)
                            values = prediction_values(record, model.dimensions)
                        else:
                            distances, values, _ = source_index.nearest(xyz, transfer_radius)
                            matched = np.isfinite(distances)
                        metric["total"] += len(record)
                        metric["matched"] += int(np.count_nonzero(matched))
                        for point in xyz[~matched][:5 - len(metric["missing_examples"])]:
                            metric["missing_examples"].append((point + origin).tolist())
                        out = copy_record(record, header)
                        for name, data in values.items():
                            out.array[name] = data
                        writer.write_points(out)
                        ids, sizes = np.unique(values[model.instance], return_counts=True)
                        counts.update({(tile, int(i)): int(n) for i, n in zip(ids, sizes) if i > 0})
                    write_retained_evlrs(writer, header)
            if metric["matched"] != metric["total"]:
                raise ValueError(f"{model.name}: incomplete prediction assignment on {target.name}: "
                                 f"{metric['matched']}/{metric['total']} within {transfer_radius} m")
        source_index_path.unlink()
        files.append(output)
        index_file(index, output, tile, origin, model)
    return files, counts


def reconcile_instances(model, files, index, origin, counts, overlap_threshold, radius, report, *,
                        enabled=True, overlaps=None, normal_keys=None):
    """Match mutual-best instance pairs, then form consistent cross-tile groups.

    Each pair needs the configured overlap fraction of the smaller instance.
    A component cannot contain two different instances from the same tile;
    ambiguous matches remain distinct for point ownership and conflict checks.
    Geometry is never removed during reconciliation. With recovered candidates,
    establish normal groups first and forbid a recovered path from joining two
    groups that were distinct before recovery.
    """
    parent = {key: key for key in sorted(counts)}
    members = {key: {key[0]: key[1]} for key in parent}
    normal_component = {key: key in normal_keys for key in parent} if normal_keys is not None else None

    def root(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    edges = []
    if enabled:
        for tile, file in enumerate(files):
            for other in range(tile):
                if overlaps is not None and other not in overlaps[tile]:
                    continue
                pairs = Counter()
                with laspy.open(file) as reader:
                    for record in reader.chunk_iterator(MAX_BATCH_POINTS):
                        xyz = coordinates(record, reader.header, origin)
                        d, values, _ = index.nearest(xyz, radius, tile=other,
                                                     overlaps=None if overlaps is None else overlaps[tile])
                        left, right = np.asarray(record[model.instance]), values[model.instance]
                        mask = np.isfinite(d) & (left > 0) & (right > 0)
                        if np.any(mask):
                            ids, sizes = np.unique(np.column_stack((left[mask], right[mask])), axis=0, return_counts=True)
                            pairs.update({tuple(map(int, pair)): int(n) for pair, n in zip(ids, sizes)})
                by_left, by_right = defaultdict(list), defaultdict(list)
                for (a, b), n in pairs.items():
                    by_left[a].append((n, b))
                    by_right[b].append((n, a))
                for (a, b), n in sorted(pairs.items()):
                    left_best, right_best = sorted(by_left[a], reverse=True), sorted(by_right[b], reverse=True)
                    if (left_best[0] != (n, b) or right_best[0] != (n, a) or
                        (len(left_best) > 1 and left_best[1][0] == n) or
                        (len(right_best) > 1 and right_best[1][0] == n)):
                        continue
                    if n / min(counts[(tile, a)], counts[(other, b)]) >= overlap_threshold:
                        edges.append((tile, a, other, b, n))
    if normal_keys is not None:
        # Preserve the original normal/normal edge order. Recovery edges then
        # take stronger mutual-best correspondences first, with stable ties.
        edges = [edge for _, edge in sorted(enumerate(edges), key=lambda item: (
            0 if (item[1][0], item[1][1]) in normal_keys and
                 (item[1][2], item[1][3]) in normal_keys else 1,
            0 if (item[1][0], item[1][1]) in normal_keys and
                 (item[1][2], item[1][3]) in normal_keys else -item[1][4],
            item[0]))]
    accepted, rejected_bridges, rejected_bridge_count = [], [], 0
    for tile, a, other, b, n in edges:
        left, right = root((tile, a)), root((other, b))
        if left != right:
            if any(t in members[right] and members[right][t] != label for t, label in members[left].items()):
                continue
            if (normal_component is not None and normal_component[left] and normal_component[right]
                    and not ((tile, a) in normal_keys and (other, b) in normal_keys)):
                rejected_bridge_count += 1
                if len(rejected_bridges) < 10:
                    rejected_bridges.append({"tiles": [tile, other], "instances": [a, b],
                                             "matches": n, "reason": "distinct retained groups"})
                continue
            survivor, removed = sorted((left, right))
            parent[removed] = survivor
            members[survivor].update(members[removed])
            if normal_component is not None:
                normal_component[survivor] |= normal_component[removed]
        accepted.append({"tiles": [tile, other], "instances": [a, b], "matches": n})
    roots = {key: i + 1 for i, key in enumerate(sorted({root(key) for key in parent}))}
    mapping = {key: roots[root(key)] for key in parent}
    if len(roots) > np.iinfo(np.uint32).max:
        raise ValueError(f"{model.name}: reconciled IDs exceed uint32")
    model.dimensions[model.instance] = instance_extra_bytes_params(model.instance, np.array([len(roots)], dtype=np.uint64))
    report["reconciliation"] = {"radius_m": radius, "overlap_threshold": overlap_threshold,
                                "groups": len(roots), "accepted_pairs": accepted,
                                "rejected_recovery_bridge_count": rejected_bridge_count,
                                "rejected_recovery_bridges": rejected_bridges,
                                "ids": [[tile, local, final] for (tile, local), final in mapping.items()]}
    return mapping


def mapped_values(model, values, tile, mapping):
    result = dict(values)
    source = values[model.instance]
    labels, inverse = np.unique(source, return_inverse=True)
    translated = np.array([mapping[(tile, int(label))] if label > 0 else 0 for label in labels], dtype=np.uint32)
    result[model.instance] = translated[inverse]
    return result


def label_matrix(model, values):
    names = [model.instance] + ([model.semantic] if model.semantic else [])
    return np.column_stack([values[name] for name in names])


def deduplicate(model, files, dense_index, survivor_index, origin, mapping, output_dir, report, *,
                overlaps=None, background_semantics_owned=False, core_preferred=None):
    """Keep stable tile/point order and compare only with final earlier survivors."""
    output_dir.mkdir(parents=True)
    survivor_index.dimensions = {n: p.type for n, p in model.dimensions.items()}
    stats = report["deduplication"] = {"radius_m": DUPLICATE_RADIUS, "tiles": [], "conflicts": []}
    outputs = []
    for tile, file in enumerate(files):
        output = output_dir / file.name
        metric = {"tile": tile, "input": 0, "surviving": 0, "removed": 0, "removal_examples": []}
        stats["tiles"].append(metric)
        offset = 0
        with laspy.open(file) as reader:
            header = target_header(reader.header, model, ready=True)
        with laspy.open(file) as reader, laspy.open(output, mode="w", header=header) as writer:
            for record in reader.chunk_iterator(MAX_BATCH_POINTS):
                xyz = coordinates(record, reader.header, origin)
                values = mapped_values(model, prediction_values(record, model.dimensions), tile, mapping)
                conflict = dense_index.conflicting_match(
                    xyz, label_matrix(model, values), DUPLICATE_RADIUS, before_tile=tile,
                    map_labels=lambda other, data: label_matrix(model, mapped_values(model, data, other, mapping)),
                    background_semantics_owned=background_semantics_owned,
                    core_preferred=(None if core_preferred is None else
                                    lambda first, second, pts: core_preferred(
                                        tile if first is None else first,
                                        tile if second is None else second, pts)),
                    overlaps=None if overlaps is None else overlaps[tile],
                )
                if conflict is not None:
                    pos = conflict.pop("query_index")
                    conflict.update(model=model.name, query_tile=tile, query_point=offset + pos,
                                    query_xyz=(xyz[pos] + origin).tolist(),
                                    query_labels=label_matrix(model, values)[pos].tolist())
                    if "tile_sources" in report:
                        conflict["query_source"] = report["tile_sources"][tile]["prediction"]
                        conflict["source"] = report["tile_sources"][conflict["tile"]]["prediction"]
                    conflict["xyz"] = (np.asarray(conflict["xyz"]) + origin).tolist()
                    stats["conflicts"].append(conflict)
                    raise ValueError(f"Unresolved cross-tile label conflict: {conflict}")
                distances, survivor_values, refs = survivor_index.nearest(xyz, DUPLICATE_RADIUS, before_tile=tile,
                                                           overlaps=None if overlaps is None else overlaps[tile])
                # A neighbor's class must never replace the owning tile's class.
                same_labels = np.all(label_matrix(model, values) == label_matrix(model, survivor_values), axis=1)
                keep = ~np.isfinite(distances) | ~same_labels
                for pos in np.flatnonzero(~keep)[:5 - len(metric["removal_examples"])]:
                    metric["removal_examples"].append({"point": int(offset + pos),
                        "survivor": refs[pos].tolist(), "distance_m": float(distances[pos])})
                out = copy_record(record[keep], header)
                for name, data in values.items():
                    out.array[name] = data[keep]
                writer.write_points(out)
                survivor_index.add(tile, xyz[keep], {n: v[keep] for n, v in values.items()},
                                   np.flatnonzero(keep) + offset)
                metric["input"] += len(record)
                metric["surviving"] += int(np.count_nonzero(keep))
                metric["removed"] += int(np.count_nonzero(~keep))
                offset += len(record)
            write_retained_evlrs(writer, header)
        survivor_index.flush()
        outputs.append(output)
    return outputs
