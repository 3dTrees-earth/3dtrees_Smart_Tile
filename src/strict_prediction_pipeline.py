"""Remap-first processing with independent model gates and staged publication."""
from __future__ import annotations

import hashlib
import csv
import json
import math
import os
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path

import laspy
import numpy as np

from bounded_point_index import MAX_BATCH_POINTS, PointIndex, coordinates
from parallel_remap import RemapBatchQueries
from dense_tile_merge import (
    copy_record, deduplicate, describe_model, index_file,
    prepare_dense, reconcile_instances,
)
from point_cloud_metadata import (
    copy_single_source_header, raw_point_cloud_files,
    update_extra_dimensions, write_retained_evlrs,
)
from dense_instance_ownership import (
    ANCHORS, filter_owned_instances, ownership_regions, instance_owners, retain_instance_owners,
    assign_shared_points, preferred_core,
)
from prediction_collection_remap import prediction_collection_files, _assign_prediction_values
from instance_labels import instance_extra_bytes_params
from orphan_instance_recovery import recover_orphaned_instances, validate_recovered_geometry
from worker_budget import spatial_query_worker_count


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


def original_match_radius(resolution_1, override=None):
    """Maximum point-to-center-of-mass distance within a cubic res1 voxel."""
    if not np.isfinite(resolution_1) or resolution_1 <= 0:
        raise ValueError("First subsampling resolution must be finite and positive")
    radius = math.sqrt(3.0) * resolution_1 if override is None else override
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Original remap tolerance must be finite and positive")
    return float(radius)


def new_report(resolution_1=0.01, remap_tolerance=None):
    radius = original_match_radius(resolution_1, remap_tolerance)
    return {"contract": "3DT-2101/v6-nearest-core", "state": "running", "models": [],
            "duplicate_radius_m": 0.01, "resolution_1_m": resolution_1,
            "original_radius_m": radius,
            "required_match_fraction": 1.0, "batch_point_limit": MAX_BATCH_POINTS,
            "distance_boundary": "inclusive Euclidean XYZ; 8 float64 ULP guard in local coordinates",
            "survivor_order": "source tile filename, then original point order",
            "original_coverage": "pending: originals not supplied"}


def _coverage_metric(file, model, stage, radius):
    return {"file": file.name, "model": model.name, "stage": stage,
            "radius_m": radius, "total": 0, "matched": 0,
            "max_matched_distance_m": 0.0, "distance_histogram": [0] * 5,
            "histogram_upper_bounds_m": [radius * fraction for fraction in (.25, .5, .75, 1)],
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


def enrich_originals(models, indices, baseline_indices, originals, output_dir, origin, report, *,
                     target_dims=None, process_workers=1):
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
    radius = report["original_radius_m"]
    enrichment_start = time.monotonic()
    with RemapBatchQueries(indices, baseline_indices, workers=process_workers,
                          radius=radius) as queries:
        for file in files:
            with laspy.open(file) as reader:
                header = copy_single_source_header(reader.header)
                existing = set(header.point_format.dimension_names)
                if existing.intersection(used):
                    raise ValueError(f"{file.name}: original already contains prediction dimensions {sorted(existing.intersection(used))}")
                update_extra_dimensions(header, [param for dims in selected for param in dims.values()])
                final_metrics = [_coverage_metric(file, model, "final_survivors", radius) for model in models]
                baseline_metrics = [_coverage_metric(file, model, "unfiltered_1cm", radius) for model in models]
                metrics.extend(baseline_metrics + final_metrics)
                with laspy.open(output_dir / file.name, mode="w", header=header) as writer:
                    batches = ((record, coordinates(record, reader.header, origin))
                               for record in reader.chunk_iterator(MAX_BATCH_POINTS))
                    for record, xyz, results in queries.map(batches):
                        out = copy_record(record, header)
                        for i, (base_distances, distances, values) in enumerate(results):
                            _record_coverage(baseline_metrics[i], xyz, base_distances, origin)
                            _record_coverage(final_metrics[i], xyz, distances, origin)
                            for name in selected[i]:
                                _assign_prediction_values(out, name, values[name], raw=True)
                        writer.write_points(out)
                    write_retained_evlrs(writer, header)
    report.setdefault("timings", {})["enrichment_seconds"] = time.monotonic() - enrichment_start
    failures = [m for m in metrics if m["matched"] != m["total"]]
    if failures:
        report["coverage_failures"] = [
            {"file": m["file"], "model": m["model"], "stage": m["stage"],
             "missing": m["total"] - m["matched"]} for m in failures
        ]
        raise ValueError(f"100% original coverage within {radius:.8g} m is required; " + "; ".join(
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
                      transfer_radius=.1732, overlap_threshold=.3, correspondence_radius=.05,
                      ready=False, matching=True, report_path=None, target_dims=None,
                      instance_dimension="PredInstance", filter_anchor="centroid", workers=1,
                      resolution_1=0.01, remap_tolerance=None):
    """The sole remap-first merge path for one or more independent models."""
    from main_remap import find_matching_files
    if filter_anchor not in ANCHORS:
        raise ValueError(f"Unknown filter anchor: {filter_anchor}")
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
    report = new_report(resolution_1, remap_tolerance)
    report["contract"] = "3DT-2183/v7-orphan-recovery"
    query_workers = spatial_query_worker_count(workers)
    report["parallelism"] = {"requested_workers": workers, "query_workers": query_workers,
                             "scope": "bounded spatial queries; small batches run serially"}
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
                regions = ownership_regions(pairs, Path(tile_bounds_json))
                dimensions = {n: p.type for n, p in model.dimensions.items()}
                dense = stack.enter_context(PointIndex(work / f"dense_{i}.sqlite", dimensions, query_workers=query_workers))
                owned = stack.enter_context(PointIndex(work / f"owned_{i}.sqlite", dimensions, query_workers=query_workers))
                survivors = stack.enter_context(PointIndex(work / f"survivors_{i}.sqlite", dimensions, query_workers=query_workers))
                suffix = Path(f"model_{i:03d}") if len(collections) > 1 else Path()
                dense_files, counts = prepare_dense(model, pairs, dense_dir / suffix, dense, origin,
                                                    transfer_radius, model_report, ready=ready)
                owned_files, counts = filter_owned_instances(
                    model, dense_files, regions, work / "owned" / suffix, owned, origin, model_report,
                    anchor=filter_anchor)
                normal_keys = set(counts)
                recovered = stack.enter_context(PointIndex(work / f"recovered_{i}.sqlite", dimensions,
                                                           query_workers=query_workers))
                owned_files, counts, admitted, claims_path = recover_orphaned_instances(
                    model, dense_files, owned_files, owned, regions, overlaps,
                    work / "recovered" / suffix, recovered, origin, counts, model_report)
                if admitted:
                    owned = recovered
                mapping = reconcile_instances(model, owned_files, owned, origin, counts, overlap_threshold,
                                              correspondence_radius, model_report, enabled=matching,
                                              overlaps=overlaps, normal_keys=normal_keys if admitted else None)
                owners = instance_owners(mapping, model_report, model_report["orphan_recovery"]["admitted"])
                if len(owners) < len(mapping):
                    authoritative = stack.enter_context(PointIndex(work / f"authoritative_{i}.sqlite", dimensions, query_workers=query_workers))
                    owned_files = retain_instance_owners(
                        model, owned_files, mapping, owners, work / "authoritative" / suffix,
                        authoritative, origin, model_report)
                    owned = authoritative
                if len(owned_files) > 1:
                    resolved = stack.enter_context(PointIndex(work / f"resolved_{i}.sqlite", dimensions, query_workers=query_workers))
                    owned_files = assign_shared_points(
                        model, owned_files, owned, regions, mapping, work / "resolved" / suffix,
                        resolved, origin, model_report, overlaps=overlaps)
                    owned = resolved
                    if any(t["background_input"] > t["background_removed"]
                           for t in model_report["instance_ownership"]["tiles"]):
                        tree_priority = stack.enter_context(PointIndex(work / f"tree_priority_{i}.sqlite", dimensions, query_workers=query_workers))
                        owned_files = assign_shared_points(
                            model, owned_files, owned, regions, mapping, work / "tree_priority" / suffix,
                            tree_priority, origin, model_report, overlaps=overlaps, background_only=True)
                        owned = tree_priority
                final_files = deduplicate(model, owned_files, owned, survivors, origin, mapping,
                                          final_dir / suffix, model_report, overlaps=overlaps,
                                          background_semantics_owned=True,
                                          core_preferred=lambda first, second, pts: preferred_core(
                                              pts, first, second, regions, origin))
                validate_recovered_geometry(survivors, model, admitted, claims_path, model_report, origin)
                with (final_dir / suffix / "instance_metadata.csv").open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.writer(stream)
                    writer.writerow([model.instance, "has_added_clusters"])
                    writer.writerows((value, 0) for value in sorted(set(mapping.values())))
                indices.append(survivors)
                baselines.append(dense)
                manifest = {"contract": report["contract"], "model": model.name,
                            "resolution_1_m": resolution_1,
                            "baseline": os.path.relpath(baseline_output / suffix, output_tiles / suffix)}
                # Publish the layout used for ownership beside the predictions.
                # Keep the source immutable and bind the effective copy by checksum.
                from tile_bounds_graph import single_cloud_layout
                layout = json.loads(Path(tile_bounds_json).read_text())
                if len(regions) == 1 and "layout_recovery" in regions[0]:
                    (x0, x1), (y0, y1) = regions[0]["core"]
                    layout = single_cloud_layout(layout, (x0, x1, y0, y1))
                    layout["layout_recovery"] = dict(regions[0]["layout_recovery"],
                        source_sha256=digest(Path(tile_bounds_json)))
                layout_path = final_dir / suffix / "tile_bounds.json"
                write_report(layout_path, layout)
                manifest["tile_bounds_json"] = "tile_bounds.json"
                manifest["tile_bounds_sha256"] = digest(layout_path)
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
                 instance_dimension="PredInstance", workers=1, resolution_1=None,
                 remap_tolerance=None):
    """Final task gate; requires baseline geometry or a merge manifest per model."""
    output = Path(output)
    available_destination(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = new_report()
    query_workers = spatial_query_worker_count(workers)
    report["parallelism"] = {"requested_workers": workers, "query_workers": query_workers,
                             "scope": "bounded spatial queries; small batches run serially"}
    report_path = output.parent / f"{output.name}_coverage.json"
    start = time.monotonic()
    try:
        recorded_resolutions = set()
        inferred_baselines = []
        for collection in collections:
            manifest = Path(collection) / "smarttile_merge.json"
            if not manifest.is_file():
                if baseline_collections is None:
                    raise ValueError(f"Baseline 1 cm geometry is required for {collection}; "
                                     "supply --baseline-1cm-folders or preserve smarttile_merge.json")
                continue
            metadata = json.loads(manifest.read_text(encoding="utf-8"))
            if "resolution_1_m" in metadata:
                recorded_resolutions.add(float(metadata["resolution_1_m"]))
            if baseline_collections is None:
                inferred_baselines.append((Path(collection) / metadata["baseline"]).resolve())
        if len(recorded_resolutions) > 1:
            raise ValueError("Prediction collections declare conflicting first-stage resolutions")
        if resolution_1 is None:
            resolution_1 = next(iter(recorded_resolutions), 0.01)
        report.update(new_report(resolution_1, remap_tolerance))
        if baseline_collections is None:
            baseline_collections = inferred_baselines
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
            source_point_count = 0
            for file in source_files:
                with laspy.open(file, read_evlrs=False) as reader:
                    source_point_count += reader.header.point_count
            # Avoid process startup for tiny inputs; require a full batch per worker.
            process_workers = min(query_workers, max(1, source_point_count // MAX_BATCH_POINTS))
            indexing_start = time.monotonic()
            for i, (collection, baseline) in enumerate(zip(collections, baseline_collections)):
                model = describe_model(collection, instance_dimension, require_instance=False)
                models.append(model)
                index = stack.enter_context(PointIndex(work / f"model_{i}.sqlite", {n: p.type for n, p in model.dimensions.items()},
                                                       query_workers=query_workers))
                maximum = 0
                for tile, file in enumerate(prediction_collection_files(collection)):
                    maximum = max(maximum, index_file(index, file, tile, origin, model))
                if model.instance:
                    model.dimensions[model.instance] = instance_extra_bytes_params(model.instance, np.array([maximum], dtype=np.uint64))
                baseline_index = stack.enter_context(PointIndex(work / f"baseline_{i}.sqlite", {}, query_workers=query_workers))
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
            report.setdefault("timings", {})["indexing_seconds"] = time.monotonic() - indexing_start
            report["parallelism"].update(enrichment_processes=process_workers,
                enrichment_query_workers=1, max_pending_batches=2 * process_workers,
                scope="serial indexing; bounded process-parallel final remap; ordered writer")
            enrich_originals(models, indices, baselines, originals, work / "originals", origin, report,
                             target_dims=target_dims, process_workers=process_workers)
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
