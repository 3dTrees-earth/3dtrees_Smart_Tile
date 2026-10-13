"""Instance tiles: tile point clouds that already carry instance IDs.

For originals that already hold predictions (for example PredInstance_SAT),
per-tile tools such as a species model need tiles in which every instance
appears exactly once and whole. No buffer filtering, reconciliation or
deduplication is needed: the instances exist already.

For each requested instance dimension:

1. A statistics read sums every instance's points and XY per input file.
   Instance IDs must be unique across the dataset. Separately segmented files
   that each number from 1 reuse IDs at unrelated places, so an ID found in
   several files whose boxes for it lie more than REUSE_GAP apart is a
   suspected reuse; too many of them fail the run. One tree split across
   files under one ID (touching boxes) is fine, and so are wide instances
   within one file (segmentation models produce scattered instances).
2. Each instance is owned by the tile whose core holds its XY centroid
   (nearest core when none does). Background (ID 0) belongs to the tile whose
   core holds the point.
3. A second read spills every point to its owner tile in XY blocks aligned
   to every voxel grid (several resolutions share the read, as in tiling).
   Voxels are keyed by (instance, voxel), so touching trees are never
   blended; each voxel keeps its original point nearest the voxel mean, so
   instance ID and carried attributes stay real values.
4. Each tile is written as ``instance_tiles_<dimension>_<resolution>/<label>.laz``:
   XYZ, the instance dimension and any requested attributes. A tile extends
   as far as its instances reach. Instances wider than WIDE_INSTANCE are
   reported with a warning (segmentation models produce scattered instances).

Every source point is represented exactly once per dimension (checked).
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import math
import multiprocessing
from pathlib import Path
import shutil
import tempfile
import time
from urllib.parse import quote

import laspy
import numpy as np

from tile_centroids import (CHUNK_POINTS, Tile, _block_key, _init_worker, _nearest_to_mean,
                            _write_blocks, block_cells, choose_block_size, route, sample_xy)
from point_cloud_metadata import extra_bytes_metadata_equal, extra_bytes_params_from_dimension_info
from prediction_collections import assign_prediction_values, promote_collection_extra_dim

WIDE_INSTANCE = 50.0  # metres: wider instances are reported with a warning
REUSE_GAP = 5.0  # metres between one ID's boxes in two files
REUSE_LIMIT = (10, 0.05)  # fail above max(10 IDs, 5% of the IDs)


def _collection_schema(sources, dimensions, attributes):
    """Preflight all carried fields and promote compatible ExtraBytes losslessly.

    ExtraBytes are spilled as raw values; their scaling and no-data descriptors
    must therefore agree across the collection and survive in the output.
    """
    names = [*dimensions, *attributes]
    if len(set(names)) != len(names):
        raise ValueError("Instance dimensions and carried attributes must be distinct")
    standard = {d.name: d for d in laspy.PointFormat(0).dimensions}
    schema, attribute_types, first = {}, {}, None
    for source in sources:
        with laspy.open(source) as reader:
            header = reader.header
            if first is None:
                first = header
            available = set(header.point_format.dimension_names)
            missing = [name for name in names if name not in available]
            if missing:
                raise ValueError(f"{source.name}: missing dimensions {missing}")
            for name in names:
                info = header.point_format.dimension_by_name(name)
                if info.is_standard:
                    if name in dimensions or name not in standard or info.num_bits > standard[name].num_bits:
                        raise ValueError(f"{source.name}: {name} cannot be carried by instance tiles (point format 0)")
                    attribute_types[name] = info.dtype or np.dtype("uint8")
                    continue
                if name in standard:
                    raise ValueError(f"{source.name}: ExtraByte {name} conflicts with a standard output dimension")
                incoming = extra_bytes_params_from_dimension_info(info, header=header)
                if name in dimensions:
                    dtype = np.dtype(incoming.type)
                    if dtype.subdtype or dtype.kind not in "iu":
                        raise ValueError(f"{source.name}: instance dimension {name} must contain scalar integer labels")
                    if ((incoming.scales is not None and np.any(np.asarray(incoming.scales) != 1)) or
                            (incoming.offsets is not None and np.any(np.asarray(incoming.offsets) != 0))):
                        raise ValueError(f"{source.name}: scaled instance dimension {name} needs explicit normalization")
                if name not in schema:
                    schema[name] = incoming
                    continue
                current = schema[name]
                if not extra_bytes_metadata_equal(current, incoming):
                    raise ValueError(f"{source.name}: incompatible schema for {name}")
                left, right = np.dtype(current.type), np.dtype(incoming.type)
                if left.shape != right.shape or (left.base.kind in "iu") != (right.base.kind in "iu"):
                    raise ValueError(f"{source.name}: incompatible schema for {name}")
                promoted = promote_collection_extra_dim(current, info)
                if left.base.kind in "iu" and np.dtype(promoted.type).base.kind not in "iu":
                    raise ValueError(f"{source.name}: no lossless integer schema for {name}")
                schema[name] = promoted
    return first, schema, [schema[name].type if name in schema else attribute_types[name] for name in attributes]


def _reader(source):
    from copc_metadata import laspy_laz_backend

    backend = laspy_laz_backend()
    kwargs = {"laz_backend": backend} if str(source).lower().endswith(".laz") and backend is not None else {}
    return laspy.open(source, **kwargs)


def _ranges(sources, workers):
    counts = []
    for source in sources:
        with laspy.open(source) as reader:
            counts.append(int(reader.header.point_count))
    size = max(CHUNK_POINTS, math.ceil(sum(counts) / max(1, 2 * workers)))
    return [(str(source), start, min(size, total - start))
            for source, total in zip(sources, counts) for start in range(0, total, size)], counts


def _chunks(source, start, count):
    with _reader(source) as reader:
        reader.seek(start)
        read = 0
        while read < count:
            chunk = reader.read_points(min(CHUNK_POINTS, count - read))
            if not len(chunk):
                raise ValueError(f"{source}: expected {count} points from {start}, read {read}")
            read += len(chunk)
            yield chunk


def _statistics(task):
    """Per dimension: rows (id, file, count, sum x, sum y, min x, max x, min y, max y) of one point range."""
    source, start, count, dimensions, file_index = task
    parts = {d: [] for d in dimensions}
    for chunk in _chunks(source, start, count):
        x, y = np.asarray(chunk.x), np.asarray(chunk.y)
        for d in dimensions:
            ids = np.asarray(chunk[d]).astype(np.uint64)
            positive = ids > 0
            if not np.any(positive):
                continue
            uid, inverse, n = np.unique(ids[positive], return_inverse=True, return_counts=True)
            xs, ys = x[positive], y[positive]
            mnx = np.full(len(uid), np.inf); mxx = np.full(len(uid), -np.inf)
            mny = np.full(len(uid), np.inf); mxy = np.full(len(uid), -np.inf)
            np.minimum.at(mnx, inverse, xs); np.maximum.at(mxx, inverse, xs)
            np.minimum.at(mny, inverse, ys); np.maximum.at(mxy, inverse, ys)
            parts[d].append(np.c_[uid.astype(np.float64), np.full(len(uid), file_index), n,
                                  np.bincount(inverse, xs), np.bincount(inverse, ys), mnx, mxx, mny, mxy])
    return parts


def _group(ids, table, columns):
    """Sum count/sx/sy and reduce the box per group of ``ids``."""
    uid, inverse = np.unique(ids, return_inverse=True)
    out = {"ids": uid}
    for column, name in zip(columns[:3], ("count", "sx", "sy")):
        out[name] = np.bincount(inverse, table[:, column], minlength=len(uid))
    for column, name, reduce in zip(columns[3:], ("minx", "maxx", "miny", "maxy"),
                                    (np.minimum, np.maximum, np.minimum, np.maximum)):
        values = np.full(len(uid), np.inf if reduce is np.minimum else -np.inf)
        reduce.at(values, inverse, table[:, column])
        out[name] = values
    return out, inverse


def _combine(parts):
    """Per instance totals, plus per (instance, file) boxes for the reuse check."""
    if not parts:
        return {"ids": np.empty(0, np.uint64), "reused": np.empty(0, np.uint64), "shared": 0}
    table = np.concatenate(parts)
    ids = table[:, 0].astype(np.uint64)
    files = table[:, 1].astype(np.int64)
    out, _ = _group(ids, table, (2, 3, 4, 5, 6, 7, 8))
    # One box per (instance, file); IDs in several files compare their boxes.
    key = ids * np.uint64(1 << 20) + files.astype(np.uint64)
    per_file, _ = _group(key, table, (2, 3, 4, 5, 6, 7, 8))
    file_ids = per_file["ids"] // np.uint64(1 << 20)
    uid, inverse, n_files = np.unique(file_ids, return_inverse=True, return_counts=True)
    shared = n_files > 1
    reused = []
    if np.any(shared):
        # Separation of the boxes: a box entirely beside another in X or Y.
        lo_x = np.full(len(uid), -np.inf); hi_x = np.full(len(uid), np.inf)
        lo_y = np.full(len(uid), -np.inf); hi_y = np.full(len(uid), np.inf)
        np.maximum.at(lo_x, inverse, per_file["minx"]); np.minimum.at(hi_x, inverse, per_file["maxx"])
        np.maximum.at(lo_y, inverse, per_file["miny"]); np.minimum.at(hi_y, inverse, per_file["maxy"])
        gap = np.hypot(np.maximum(lo_x - hi_x, 0), np.maximum(lo_y - hi_y, 0))
        reused = uid[shared & (gap > REUSE_GAP)]
    out["reused"] = np.asarray(reused, dtype=np.uint64)
    out["shared"] = int(np.count_nonzero(shared))
    return out


def check_unique_instances(statistics, dimension, enabled=True, wide=WIDE_INSTANCE):
    """Fail when many IDs occur in several files at unrelated places (IDs reused across files).

    Instances wider than ``wide`` metres only produce a warning.
    """
    reused, total = statistics["reused"], len(statistics["ids"])
    widths = (np.maximum(statistics["maxx"] - statistics["minx"], statistics["maxy"] - statistics["miny"])
              if total else np.zeros(0))
    wide_ids = np.flatnonzero(widths > wide)
    if len(wide_ids):
        examples = ", ".join(f"{int(statistics['ids'][i])} ({widths[i]:.0f} m)"
                             for i in wide_ids[np.argsort(-widths[wide_ids])][:5])
        print(f"  Warning: {dimension}: {len(wide_ids)} of {total} instances are wider than {wide:g} m, "
              f"e.g. {examples}; their tiles extend accordingly")
    limit = max(REUSE_LIMIT[0], REUSE_LIMIT[1] * total)
    if enabled and len(reused) > limit:
        examples = ", ".join(str(int(i)) for i in reused[:8])
        raise ValueError(
            f"{dimension}: {len(reused)} of {total} instance IDs occur in several input files more than "
            f"{REUSE_GAP:g} m apart (e.g. {examples}). Instance IDs must be unique across all input files; "
            "separately segmented files that each number from 1 reuse IDs. Renumber them, or pass "
            "--allow-reused-instance-ids if these really are single instances.")
    return {"instances": total, "ids_in_several_files": statistics["shared"],
            "suspected_reused_ids": int(len(reused)), "wider_than_m": wide, "wide_instances": int(len(wide_ids))}


def _core_index(x, y, cores):
    """Owner tile per point: first tile whose closed core holds it, else the nearest core."""
    owner = np.full(len(x), -1, dtype=np.int64)
    for index, (x0, y0, x1, y1) in enumerate(cores):
        free = owner < 0
        inside = free & (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        owner[inside] = index
    missing = np.flatnonzero(owner < 0)
    if len(missing):
        c = np.asarray(cores, dtype=np.float64)
        dx = np.maximum(np.maximum(c[None, :, 0] - x[missing, None], 0), x[missing, None] - c[None, :, 2])
        dy = np.maximum(np.maximum(c[None, :, 1] - y[missing, None], 0), y[missing, None] - c[None, :, 3])
        owner[missing] = np.argmin(dx * dx + dy * dy, axis=1)
    return owner


def _record_dtype(dimension_dtypes):
    return np.dtype([("X", "<i4"), ("Y", "<i4"), ("Z", "<i4"), ("m", "u1"), ("instance", "<u8")]
                    + [(f"a{i}", dtype) for i, dtype in enumerate(dimension_dtypes)])


def _spill(task):
    """Second read: route each point of one range to its owner tile, per dimension."""
    (task_id, source, start, count, dimensions, dimension_dirs, owners, tiles, attributes, attribute_dtypes,
     resolutions, cells, spill_dir) = task
    full_mask = np.uint8((1 << len(resolutions)) - 1)
    cores = [tile.core for tile in tiles]
    dtype = _record_dtype(attribute_dtypes)
    name = f"{task_id:06d}.bin"
    counts = {d: np.zeros(len(tiles), dtype=np.int64) for d in dimensions}
    for chunk in _chunks(source, start, count):
        x, y, z = np.asarray(chunk.x), np.asarray(chunk.y), np.asarray(chunk.z)
        background = None
        for d in dimensions:
            ids = np.asarray(chunk[d]).astype(np.uint64)
            owner = np.empty(len(ids), dtype=np.int64)
            positive = ids > 0
            if np.any(positive):
                known, tile_of = owners[d]
                position = np.searchsorted(known, ids[positive])
                owner[positive] = tile_of[position]
            if not np.all(positive):
                if background is None:
                    background = _core_index(x, y, cores)
                owner[~positive] = background[~positive]
            for index in np.unique(owner):
                rows = np.flatnonzero(owner == index)
                tile = tiles[int(index)]
                records = np.empty(len(rows), dtype=dtype)
                for axis, values in enumerate((x, y, z)):
                    encoded = np.round((values[rows] - tile.offsets[axis]) / tile.scales[axis])
                    if encoded.min() < np.iinfo(np.int32).min or encoded.max() > np.iinfo(np.int32).max:
                        raise ValueError("Instance tile coordinates exceed the LAS int32 range")
                    records["XYZ"[axis]] = encoded.astype(np.int32)
                records["instance"] = ids[rows]
                for i, attribute in enumerate(attributes):
                    values = chunk.array[attribute] if attribute in chunk.array.dtype.names else chunk[attribute]
                    records[f"a{i}"] = np.asarray(values)[rows]
                records["m"] = full_mask
                coords = np.stack([records[a] * tile.scales[i] + tile.offsets[i]
                                   for i, a in enumerate(("X", "Y", "Z"))], axis=1)
                # A voxel never spans blocks at any resolution (tile_centroids.route).
                source_rows, blocks, bits = route(coords, records["m"], resolutions, cells)
                routed = records[source_rows]
                routed["m"] = bits
                _write_blocks(Path(spill_dir) / dimension_dirs[d] / tile.label, name, routed, blocks)
                counts[d][int(index)] += len(rows)
    return counts


def _reduce(task):
    """One block of one tile: per resolution and (instance, voxel), keep the point nearest the voxel mean."""
    block, resolutions, scales, offsets, attribute_dtypes, point_format, instance_field, attributes, fragments = task
    dtype = _record_dtype(attribute_dtypes)
    every = np.concatenate([np.fromfile(path, dtype=dtype) for path in sorted(Path(block).glob("*.bin"))])
    results = []
    for index, (resolution, fragment) in enumerate(zip(resolutions, fragments)):
        records = every[(every["m"] & np.uint8(1 << index)) != 0]
        if not len(records):
            results.append((0, 0))
            continue
        xyz = np.stack([records[a].astype(np.int64) for a in ("X", "Y", "Z")], axis=1)
        keys = np.floor((xyz * np.asarray(scales) + np.asarray(offsets)) / resolution).astype(np.int64)
        order = np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0], keys[:, 2], keys[:, 1], keys[:, 0],
                            records["instance"]))
        records, xyz, keys = records[order], xyz[order], keys[order]
        change = np.r_[True, (np.diff(keys, axis=0) != 0).any(axis=1) | (np.diff(records["instance"]) != 0)]
        starts = np.flatnonzero(change)
        counts = np.diff(np.r_[starts, len(records)]).astype(np.int64)
        sums = np.add.reduceat(xyz, starts, axis=0)
        chosen = records[_nearest_to_mean(xyz, starts, counts, sums, scales)]
        # Named assignment lets laspy pack bit fields such as classification.
        out = laspy.ScaleAwarePointRecord.zeros(len(chosen), point_format=point_format,
                                                scales=np.asarray(scales), offsets=np.asarray(offsets))
        for axis in ("X", "Y", "Z"):
            out.array[axis] = chosen[axis]
        assign_prediction_values(out, instance_field, chosen["instance"], raw=True)
        for i, attribute in enumerate(attributes):
            assign_prediction_values(out, attribute, chosen[f"a{i}"], raw=attribute in out.array.dtype.names)
        out.array.tofile(fragment)
        results.append((len(chosen), len(records)))
    shutil.rmtree(block)
    return results


def _output_header(source_header, dimension, schema, attributes, scales, offsets):
    """Point format 0 with the source CRS, plus the instance and requested attributes.

    Requested standard dimensions of format 0 (intensity, classification, ...)
    stay standard; any other requested dimension is carried as ExtraBytes.
    """
    from main_tile import _make_tile_header
    from subsample_com import make_center_of_mass_header

    base = _make_tile_header(source_header, offsets=np.asarray(offsets), scales=np.asarray(scales))
    header = make_center_of_mass_header(base, dimension_reduction=True)
    standard = set(header.point_format.dimension_names)
    extra = [schema[dimension]]
    for attribute in attributes:
        if attribute in standard:
            continue
        extra.append(schema[attribute])
    header.add_extra_dims(extra)
    return header


def output_directory(output_dir, dimension, resolution):
    from main_subsample import resolution_label
    return Path(output_dir) / f"instance_tiles_{quote(dimension, safe='')}_{resolution_label(resolution)}"


def write_instance_tiles(sources, tiles, dimensions, output_dir, *, resolutions, attributes=(),
                         check_ids=True, wide=WIDE_INSTANCE, workers=1):
    """Write ``instance_tiles_<dimension>_<resolution>/<label>.laz`` per dimension and resolution.

    All resolutions come from the same two reads. ``tiles``: [Tile] with
    ``core`` (xmin, ymin, xmax, ymax). Returns {dimension: {instance checks,
    "outputs": {resolution: {"directory", "tiles", "points"}}}}.
    """
    from point_cloud_metadata import write_retained_evlrs
    from output_publication import publish_files, validate_output_paths

    resolutions = tuple(float(r) for r in resolutions)
    if not resolutions or any(not math.isfinite(r) or r <= 0 for r in resolutions):
        raise ValueError("Instance tiles need positive resolutions")
    if len(set(resolutions)) != len(resolutions) or len(resolutions) > 8:
        raise ValueError("Instance tile resolutions must be distinct, at most eight")
    if not dimensions:
        raise ValueError("Instance tiles need at least one instance dimension")
    sources = [Path(s) for s in sources]
    if not sources:
        raise ValueError("Instance tiles need at least one input file")
    header, schema, attribute_dtypes = _collection_schema(sources, dimensions, attributes)
    dimension_dirs = {d: f"dimension_{i:03d}" for i, d in enumerate(dimensions)}
    validate_output_paths(sources, [output_directory(output_dir, d, r) for d in dimensions for r in resolutions])
    workers = max(1, int(workers))
    tasks, counts = _ranges(sources, workers)
    started = time.monotonic()
    context = multiprocessing.get_context("spawn")
    results, products = {}, []
    with tempfile.TemporaryDirectory(prefix=".instance-tiles-", dir=output_dir) as scratch, \
            ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=_init_worker) as pool:
        scratch = Path(scratch)
        parts = {d: [] for d in dimensions}
        file_of = {str(source): i for i, source in enumerate(sources)}
        for result in pool.map(_statistics, [(s, a, n, tuple(dimensions), file_of[s]) for s, a, n in tasks]):
            for d in dimensions:
                parts[d].extend(result[d])
        owners = {}
        cores = [tile.core for tile in tiles]
        for d in dimensions:
            statistics = _combine(parts[d])
            id_report = check_unique_instances(statistics, d, check_ids, wide)
            n = np.maximum(statistics.get("count", np.zeros(0)), 1)
            cx = statistics.get("sx", np.zeros(0)) / n
            cy = statistics.get("sy", np.zeros(0)) / n
            owners[d] = (statistics["ids"], _core_index(cx, cy, cores))
            results[d] = id_report
        total = sum(counts)
        area = max(1e-9, (max(c[2] for c in cores) - min(c[0] for c in cores))
                   * (max(c[3] for c in cores) - min(c[1] for c in cores))) if all(np.isfinite(cores).ravel()) else 1.0
        block = choose_block_size(total, area, resolutions, sample_xy(sources))
        cells = block_cells(resolutions, block)
        spill_dir = scratch / "blocks"
        spill_tasks = [(i, s, a, n, tuple(dimensions), dimension_dirs, owners, tiles, tuple(attributes), attribute_dtypes,
                        resolutions, cells, str(spill_dir)) for i, (s, a, n) in enumerate(tasks)]
        tile_points = {d: np.zeros(len(tiles), dtype=np.int64) for d in dimensions}
        for result in pool.map(_spill, spill_tasks):
            for d in dimensions:
                tile_points[d] += result[d]
        spilled = time.monotonic()
        for d in dimensions:
            if int(tile_points[d].sum()) != total:
                raise ValueError(f"{d}: routed {int(tile_points[d].sum()):,} of {total:,} points")
            staged = scratch / "out" / dimension_dirs[d]
            written = {r: [] for r in resolutions}
            points = {r: 0 for r in resolutions}
            for index, tile in enumerate(tiles):
                tile_dir = spill_dir / dimension_dirs[d] / tile.label
                if not tile_dir.exists():
                    continue
                out_header = _output_header(header, d, schema, attributes, tile.scales, tile.offsets)
                out_dtype = out_header.point_format.dtype()
                fragments = [staged / tile.label / str(slot) for slot in range(len(resolutions))]
                for folder in fragments:
                    folder.mkdir(parents=True)
                jobs = [(b, resolutions, tile.scales, tile.offsets, attribute_dtypes, out_header.point_format, d,
                         tuple(attributes), [str(folder / f"{n:09d}.bin") for folder in fragments])
                        for n, b in enumerate(sorted(tile_dir.iterdir(), key=_block_key))]
                kept = [0] * len(resolutions)
                used = [0] * len(resolutions)
                for block_results in pool.map(_reduce, jobs):
                    for slot, (k, u) in enumerate(block_results):
                        kept[slot] += k
                        used[slot] += u
                for slot, resolution in enumerate(resolutions):
                    if used[slot] != tile_points[d][index]:
                        raise ValueError(f"{d} {tile.label} at {resolution} m: voxels represent {used[slot]:,} of "
                                         f"{int(tile_points[d][index]):,} points")
                    target = staged / f"{tile.label}.{slot}.laz"
                    with laspy.open(target, mode="w", header=out_header) as writer:
                        for fragment in sorted(fragments[slot].glob("*.bin")):
                            array = np.fromfile(fragment, dtype=out_dtype)
                            if len(array):
                                writer.write_points(laspy.ScaleAwarePointRecord(
                                    array, out_header.point_format, out_header.scales, out_header.offsets))
                        write_retained_evlrs(writer, out_header)
                    written[resolution].append((target, tile.label))
                    points[resolution] += kept[slot]
                shutil.rmtree(staged / tile.label)
            results[d]["outputs"] = {}
            for resolution in resolutions:
                directory = output_directory(output_dir, d, resolution)
                for target, label in written[resolution]:
                    products.append((target, directory / f"{label}.laz"))
                results[d]["outputs"][resolution] = {"directory": directory, "tiles": len(written[resolution]),
                                                     "points": points[resolution]}
        publish_files(products, inputs=sources)
    print(f"  Instance tiles {list(dimensions)} at {list(resolutions)} m: {total:,} points read twice in "
          f"{spilled - started:.1f} s; written in {time.monotonic() - spilled:.1f} s")
    return results


def instance_tiles_from_plan(plan, output_dir, dimensions, *, resolutions, attributes=(),
                             check_ids=True, wide=WIDE_INSTANCE, workers=1, publish_occupancy=True):
    """Tile-task entry: the planned grid's cores own instances and background.

    ``publish_occupancy`` writes the occupied tiles into the layout; leave it to
    the subsampled outputs when the same run writes both.
    """
    import json
    from main_tile import plot_overview, read_tile_jobs
    from tile_core_occupancy import publish_occupied_tiles

    sources = [Path(s) for s in plan.source_files]
    scales = []
    for source in sources:
        with laspy.open(source) as reader:
            scales.append(tuple(float(v) for v in reader.header.scales))
            z_offset = float(reader.header.offsets[2])
    scales = tuple(min(axis) for axis in zip(*scales))
    if plan.skip_tiling:
        with laspy.open(sources[0]) as reader:
            h = reader.header
        stem = sources[0].name[:-len(".copc.laz")] if sources[0].name.endswith(".copc.laz") else sources[0].stem
        tiles = [Tile(stem, (-math.inf, -math.inf, math.inf, math.inf), scales,
                      tuple(float(v) for v in h.offsets), core=(-math.inf, -math.inf, math.inf, math.inf))]
    else:
        layout = json.loads(Path(plan.bounds_json).read_text())["tiles"]
        cores = {f"c{t['col']:02d}_r{t['row']:02d}": t["core"] for t in layout}
        jobs = read_tile_jobs(plan.jobs_file)
        tiles = [Tile(label, bounds, scales,
                      ((bounds[0] + bounds[2]) / 2.0, (bounds[1] + bounds[3]) / 2.0, z_offset),
                      core=(cores[label][0][0], cores[label][1][0], cores[label][0][1], cores[label][1][1]))
                 for label, bounds in sorted(jobs.items())]
    results = write_instance_tiles(sources, tiles, dimensions, Path(output_dir), resolutions=resolutions,
                                   attributes=attributes, check_ids=check_ids, wide=wide, workers=workers)
    if publish_occupancy and not plan.skip_tiling:
        occupied = set()
        for d in dimensions:
            for output in results[d]["outputs"].values():
                occupied |= {p.stem for p in Path(output["directory"]).glob("*.laz")}
        layout = json.loads(Path(plan.bounds_json).read_text())["tiles"]
        publish_occupied_tiles(plan.bounds_json, plan.jobs_file,
                               [f"c{t['col']:02d}_r{t['row']:02d}" in occupied for t in layout])
        plot_overview(plan)
    return results
