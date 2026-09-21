"""Remap-first processing with independent model gates and staged publication."""
from __future__ import annotations

import hashlib
import csv
import json
import os
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path

import laspy
import numpy as np

from bounded_point_index import MAX_BATCH_POINTS, PointIndex, coordinates
from dense_tile_merge import (
    ORIGINAL_RADIUS, copy_record, deduplicate, describe_model, index_file,
    prepare_dense, reconcile_instances,
)
from point_cloud_metadata import (
    copy_single_source_header, raw_point_cloud_files,
    update_extra_dimensions, write_retained_evlrs,
)
from prediction_collection_remap import prediction_collection_files
from instance_labels import instance_extra_bytes_params


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def available_destination(path):
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"Refusing to mix this run with existing output: {path}")


def publish(products):
    """Publish only validated artifacts; roll back new paths if a move fails."""
    for _, destination in products:
        available_destination(destination)
    moved = []
    try:
        for source, destination in products:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.is_dir():
                destination.rmdir()  # Only an empty, explicitly selected output.
            os.replace(source, destination)
            moved.append((source, destination))
    except Exception:
        for source, destination in reversed(moved):
            os.replace(destination, source)
        raise


def origin_for(files):
    with laspy.open(files[0]) as reader:
        return np.asarray(reader.header.offsets, dtype=np.float64)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            result.update(block)
    return result.hexdigest()


def new_report():
    return {"contract": "3DT-2101/v1", "state": "running", "models": [],
            "duplicate_radius_m": 0.01, "original_radius_m": ORIGINAL_RADIUS,
            "required_match_fraction": 1.0, "batch_point_limit": MAX_BATCH_POINTS,
            "distance_boundary": "inclusive Euclidean XYZ; 8 float64 ULP guard in local coordinates",
            "survivor_order": "source tile filename, then original point order",
            "original_coverage": "pending: originals not supplied"}


def _coverage_metric(file, model, stage):
    return {"file": file.name, "model": model.name, "stage": stage,
            "radius_m": ORIGINAL_RADIUS, "total": 0, "matched": 0,
            "max_matched_distance_m": 0.0, "distance_histogram": [0] * 5,
            "histogram_upper_bounds_m": [0.0025, 0.005, 0.0075, 0.01],
            "missing_examples": []}


def _record_coverage(metric, xyz, distances, origin):
    matched = np.isfinite(distances)
    metric["total"] += len(xyz)
    metric["matched"] += int(np.count_nonzero(matched))
    finite = distances[matched]
    if len(finite):
        metric["max_matched_distance_m"] = max(metric["max_matched_distance_m"], float(finite.max()))
    bins = np.searchsorted(metric["histogram_upper_bounds_m"], finite, side="left")
    hist = np.bincount(bins, minlength=5)
    metric["distance_histogram"] = (np.asarray(metric["distance_histogram"]) + hist).tolist()
    for point in xyz[~matched][:5 - len(metric["missing_examples"])]:
        metric["missing_examples"].append((point + origin).tolist())


def enrich_originals(models, indices, baseline_indices, originals, output_dir, origin, report, *, target_dims=None):
    """Stage all originals and measure every model independently before failing."""
    files = raw_point_cloud_files(originals)
    if not files:
        raise ValueError(f"No raw LAS/LAZ originals found in {originals}")
    selected = []
    used = set()
    for model in models:
        dims = {n: p for n, p in model.dimensions.items() if target_dims is None or n in target_dims}
        if not dims:
            raise ValueError(f"No selected prediction dimensions for model {model.name}")
        if used.intersection(dims):
            raise ValueError(f"Duplicate prediction dimension name across models: {sorted(used.intersection(dims))}")
        used.update(dims)
        selected.append(dims)
    output_dir.mkdir(parents=True)
    metrics = report["original_coverage"] = []
    for file in files:
        with laspy.open(file) as reader:
            header = copy_single_source_header(reader.header)
            existing = set(header.point_format.dimension_names)
            if existing.intersection(used):
                raise ValueError(f"{file.name}: original already contains prediction dimensions {sorted(existing.intersection(used))}")
            update_extra_dimensions(header, [param for dims in selected for param in dims.values()])
            final_metrics = [_coverage_metric(file, model, "final_survivors") for model in models]
            baseline_metrics = [_coverage_metric(file, model, "unfiltered_1cm") for model in models]
            metrics.extend(baseline_metrics + final_metrics)
            with laspy.open(output_dir / file.name, mode="w", header=header) as writer:
                for record in reader.chunk_iterator(MAX_BATCH_POINTS):
                    xyz = coordinates(record, reader.header, origin)
                    out = copy_record(record, header)
                    for i, (model, index, baseline) in enumerate(zip(models, indices, baseline_indices)):
                        base_distances, _, _ = baseline.nearest(xyz, ORIGINAL_RADIUS)
                        distances, values, _ = index.nearest(xyz, ORIGINAL_RADIUS)
                        _record_coverage(baseline_metrics[i], xyz, base_distances, origin)
                        _record_coverage(final_metrics[i], xyz, distances, origin)
                        for name in selected[i]:
                            out.array[name] = values[name]
                    writer.write_points(out)
                write_retained_evlrs(writer, header)
    failures = [m for m in metrics if m["matched"] != m["total"]]
    if failures:
        report["coverage_failures"] = [
            {"file": m["file"], "model": m["model"], "stage": m["stage"],
             "missing": m["total"] - m["matched"]} for m in failures
        ]
        raise ValueError("100% original coverage within 0.01 m is required; " + "; ".join(
            f"{m['file']}/{m['model']}/{m['stage']}: {m['matched']}/{m['total']}" for m in failures))


def tile_overlaps(pairs, tile_bounds_json, origin):
    """Admit only the intersection of distinct declared buffered tile bounds."""
    if len(pairs) == 1:
        return [{}]
    from main_remap import get_file_bounds
    from tile_bounds_graph import build_neighbor_graph_from_bounds_json, match_tiles_to_json_bounds
    bounds, centers, _ = build_neighbor_graph_from_bounds_json(tile_bounds_json)
    mapping, _ = match_tiles_to_json_bounds({str(p[1]): get_file_bounds(p[1]) for p in pairs}, bounds, centers)
    regions = [bounds[mapping[str(p[1])]] for p in pairs]
    result = []
    for i, (x0, x1, y0, y1) in enumerate(regions):
        previous = {}
        for j, (a0, a1, b0, b1) in enumerate(regions[:i]):
            lo = np.array([max(x0, a0), max(y0, b0)])
            hi = np.array([min(x1, a1), min(y1, b1)])
            if np.all(hi > lo):
                previous[j] = (lo - origin[:2], hi - origin[:2])
        result.append(previous)
    return result


def merge_file(files, output):
    """Stream an intermediate merged cloud, preserving the dense geometry."""
    with laspy.open(files[0]) as first:
        header = copy_single_source_header(first.header)
    with laspy.open(output, mode="w", header=header) as writer:
        for file in files:
            with laspy.open(file) as reader:
                if reader.header.point_format != header.point_format:
                    raise ValueError(f"Cannot concatenate incompatible target schemas: {file}")
                for record in reader.chunk_iterator(MAX_BATCH_POINTS):
                    record.change_scaling(scales=header.scales, offsets=header.offsets)
                    writer.write_points(record)
        write_retained_evlrs(writer, header)


def merge_collections(*, collections, target_dir, output_tiles, tile_bounds_json,
                      originals=None, original_output=None, merged_output=None,
                      transfer_radius=.125, overlap_threshold=.3, correspondence_radius=.05,
                      ready=False, matching=True, report_path=None, target_dims=None,
                      instance_dimension="PredInstance"):
    """The sole remap-first merge path for one or more independent models."""
    from main_remap import find_matching_files
    if not collections:
        raise ValueError("At least one prediction collection is required")
    if not (np.isfinite(transfer_radius) and transfer_radius > 0 and
            np.isfinite(correspondence_radius) and correspondence_radius > 0 and
            0 < overlap_threshold <= 1):
        raise ValueError("Finite positive matching radii and an overlap threshold in (0,1] are required")
    output_tiles = Path(output_tiles)
    output_tiles.parent.mkdir(parents=True, exist_ok=True)
    baseline_output = output_tiles.with_name(output_tiles.name + "_unfiltered_1cm")
    report_path = Path(report_path or output_tiles.parent / "remap_first_report.json")
    report = new_report()
    start, cpu_start = time.monotonic(), time.process_time()
    destinations = [output_tiles, baseline_output]
    if originals:
        original_output = Path(original_output or output_tiles.parent / "original_with_predictions")
        destinations.append(original_output)
    if merged_output:
        if Path(merged_output).name.lower().endswith(".copc.laz"):
            raise ValueError("Processed merge writes regular LAZ; use create_merged_file for COPC products")
        if len(collections) != 1:
            raise ValueError("A single --output-merged-laz cannot combine model geometry; use --skip-merged-file for multiple collections")
        destinations.append(Path(merged_output))
    if len({p.resolve() for p in destinations}) != len(destinations):
        raise ValueError("Output destinations must be distinct")
    for path in destinations:
        available_destination(path)
    try:
        with tempfile.TemporaryDirectory(prefix=".remap-first-", dir=output_tiles.parent) as temporary, ExitStack() as stack:
            work = Path(temporary)
            final_dir, dense_dir = work / "final", work / "dense"
            models, indices, baselines = [], [], []
            source_files = prediction_collection_files(collections[0])
            if not source_files:
                raise ValueError(f"No predictions in {collections[0]}")
            origin = origin_for(source_files)
            report["origin"] = origin.tolist()
            final_files = []
            for i, collection in enumerate(collections):
                model = describe_model(collection, instance_dimension)
                models.append(model)
                model_report = {"model": model.name, "source": str(collection),
                                "instance_dimension": model.instance, "semantic_dimension": model.semantic}
                report["models"].append(model_report)
                if ready:
                    pairs = [(p, p, p.stem) for p in sorted(prediction_collection_files(collection))]
                else:
                    if target_dir is None:
                        raise ValueError("Remap-first merge requires --subsampled-target-folder with the 1 cm tiles")
                    targets = set(prediction_collection_files(target_dir))
                    if Path(collection).is_file() and len(targets) == 1:
                        pairs = [(Path(collection), next(iter(targets)), "tile")]
                    else:
                        pairs = find_matching_files(Path(collection), Path(target_dir), tile_bounds_json=tile_bounds_json)
                    if set(p[1] for p in pairs) != targets or len(pairs) != len(targets):
                        raise ValueError(f"{model.name}: every 1 cm target tile must have exactly one prediction tile")
                    pairs.sort(key=lambda p: str(p[0]))
                model_report["tile_sources"] = [{"tile": n, "prediction": str(p[0]), "geometry": str(p[1])}
                                                for n, p in enumerate(pairs)]
                overlaps = tile_overlaps(pairs, tile_bounds_json, origin)
                dimensions = {n: p.type for n, p in model.dimensions.items()}
                dense = stack.enter_context(PointIndex(work / f"dense_{i}.sqlite", dimensions))
                survivors = stack.enter_context(PointIndex(work / f"survivors_{i}.sqlite", dimensions))
                suffix = Path(f"model_{i:03d}") if len(collections) > 1 else Path()
                dense_files, counts = prepare_dense(model, pairs, dense_dir / suffix, dense, origin,
                                                    transfer_radius, model_report, ready=ready)
                mapping = reconcile_instances(model, dense_files, dense, origin, counts, overlap_threshold,
                                              correspondence_radius, model_report, enabled=matching, overlaps=overlaps)
                final_files = deduplicate(model, dense_files, dense, survivors, origin, mapping,
                                          final_dir / suffix, model_report, overlaps=overlaps)
                with (final_dir / suffix / "instance_metadata.csv").open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.writer(stream)
                    writer.writerow([model.instance, "has_added_clusters"])
                    writer.writerows((value, 0) for value in sorted(set(mapping.values())))
                indices.append(survivors)
                baselines.append(dense)
                manifest = {"contract": report["contract"], "model": model.name,
                            "baseline": os.path.relpath(baseline_output / suffix, output_tiles / suffix)}
                write_report(final_dir / suffix / "smarttile_merge.json", manifest)
            products = [(final_dir, output_tiles), (dense_dir, baseline_output)]
            if originals:
                enrich_originals(models, indices, baselines, originals, work / "originals", origin, report,
                                 target_dims=target_dims)
                products.append((work / "originals", original_output))
            if merged_output:
                merge_file(final_files, work / "merged.laz")
                products.append((work / "merged.laz", Path(merged_output)))
                metadata = work / "merged_instance_metadata.csv"
                metadata.write_bytes((final_dir / "instance_metadata.csv").read_bytes())
                products.append((metadata, Path(merged_output).with_name(Path(merged_output).stem + "_instance_metadata.csv")))
            report["checksums"] = {str(destination / file.relative_to(source) if source.is_dir() else destination): digest(file)
                                   for source, destination in products
                                   for file in (sorted(source.rglob("*.la*")) if source.is_dir() else [source])}
            # Close every database before TemporaryDirectory cleanup on Windows.
            stack.close()
            publish(products)
            report["state"] = "validated" if originals else "intermediate; final original coverage pending"
    except Exception as exc:
        report["state"], report["error"] = "failed", str(exc)
        raise
    finally:
        report["wall_seconds"] = time.monotonic() - start
        report["cpu_seconds"] = time.process_time() - cpu_start
        write_report(report_path, report)
    return report


def strict_remap(*, collections, originals, output, baseline_collections=None, target_dims=None,
                 instance_dimension="PredInstance"):
    """Final task gate; requires baseline geometry or a merge manifest per model."""
    output = Path(output)
    available_destination(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = new_report()
    report_path = output.parent / f"{output.name}_coverage.json"
    start = time.monotonic()
    try:
        if baseline_collections is None:
            baseline_collections = []
            for collection in collections:
                manifest = Path(collection) / "smarttile_merge.json"
                if not manifest.is_file():
                    raise ValueError(f"Baseline 1 cm geometry is required for {collection}; "
                                     "supply --baseline-1cm-folders or preserve smarttile_merge.json")
                metadata = json.loads(manifest.read_text(encoding="utf-8"))
                baseline_collections.append((Path(collection) / metadata["baseline"]).resolve())
        if len(baseline_collections) not in (1, len(collections)):
            raise ValueError("Provide one shared baseline or one baseline per model")
        if len(baseline_collections) == 1:
            baseline_collections = baseline_collections * len(collections)
        with tempfile.TemporaryDirectory(prefix=".strict-remap-", dir=output.parent) as temporary, ExitStack() as stack:
            work = Path(temporary)
            models, indices, baselines = [], [], []
            source_files = raw_point_cloud_files(originals)
            if not source_files:
                raise ValueError(f"No raw original files in {originals}")
            origin = origin_for(source_files)
            for i, (collection, baseline) in enumerate(zip(collections, baseline_collections)):
                model = describe_model(collection, instance_dimension, require_instance=False)
                models.append(model)
                index = stack.enter_context(PointIndex(work / f"model_{i}.sqlite", {n: p.type for n, p in model.dimensions.items()}))
                maximum = 0
                for tile, file in enumerate(prediction_collection_files(collection)):
                    maximum = max(maximum, index_file(index, file, tile, origin, model))
                if model.instance:
                    model.dimensions[model.instance] = instance_extra_bytes_params(model.instance, np.array([maximum], dtype=np.uint64))
                baseline_index = stack.enter_context(PointIndex(work / f"baseline_{i}.sqlite", {}))
                baseline_files = prediction_collection_files(baseline)
                if not baseline_files:
                    raise ValueError(f"Baseline geometry unavailable: {baseline}")
                for tile, file in enumerate(baseline_files):
                    with laspy.open(file) as reader:
                        offset = 0
                        for record in reader.chunk_iterator(MAX_BATCH_POINTS):
                            baseline_index.add(tile, coordinates(record, reader.header, origin), {},
                                               np.arange(offset, offset + len(record)))
                            offset += len(record)
                baseline_index.flush()
                indices.append(index)
                baselines.append(baseline_index)
            enrich_originals(models, indices, baselines, originals, work / "originals", origin, report, target_dims=target_dims)
            stack.close()
            report["checksums"] = {f.name: digest(f) for f in sorted((work / "originals").iterdir())}
            publish([(work / "originals", output)])
            report["state"] = "validated"
    except Exception as exc:
        report["state"], report["error"] = "failed", str(exc)
        raise
    finally:
        report["wall_seconds"] = time.monotonic() - start
        write_report(report_path, report)
    return report
