"""Single-scan tile subsampling: decode every original point once.

Center-of-mass tiling used to crop sources into per-tile parts, build a
full-resolution COPC per tile and read it back through ~5 m window queries.
COPC's coarse octree nodes overlap every window, so those queries decoded each
point about 25 times (3109: 4.6 B decoded for 183 M points), and the
full-resolution tiles were never published.

Here the sources are read once, in parallel point ranges. A point is
re-encoded into every tile it falls in (inclusive buffered bounds) exactly as
the tile COPC stored it, and spilled to XY blocks aligned to each output voxel
grid. Blocks are reduced independently: every output point is the mean of the
original points in its voxel at that resolution (3DT-2203), not a mean of finer
means. Sums follow a canonical point order, so results do not depend on the
worker count or on how the sources were split.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import tempfile
import time

import laspy
import numpy as np

# One spilled point: XYZ in its tile's LAS encoding plus a bit per resolution.
RECORD = np.dtype([("X", "<i4"), ("Y", "<i4"), ("Z", "<i4"), ("m", "u1")])
CHUNK_POINTS = 1_000_000
BLOCK_POINTS = 4_000_000
MAX_BLOCK_EDGE = 50.0
MIN_BLOCK_EDGE = 5.0  # bounds the number of spill files on sparse ground


@dataclass(frozen=True)
class Tile:
    """Output cloud: inclusive XY bounds and the LAS encoding its points use."""
    label: str
    bounds: tuple  # (xmin, ymin, xmax, ymax)
    scales: tuple
    offsets: tuple
    core: tuple = None  # (xmin, ymin, xmax, ymax); tiles without core points are dropped


def block_cells(resolutions, block_size):
    """Voxels per block edge, per resolution; blocks never split a voxel."""
    return tuple(max(1, int(round(block_size / r))) for r in resolutions)


def choose_block_size(point_count, area, resolutions, sample=None):
    """Block edge holding at most about BLOCK_POINTS points.

    With an XY `sample` of the sources, the edge is the largest one (down to
    MIN_BLOCK_EDGE) that leaves at most 10% of the points in blocks above
    BLOCK_POINTS; those few are split once while reducing. Header extents can
    include far outliers and terrestrial scans are very uneven, so the mean
    density over `area` is only the fallback.
    """
    coarse = max(resolutions)

    def aligned(edge):
        # A whole number of coarsest voxels: nested grids then share block
        # edges; otherwise route() writes most points once per resolution.
        return max(coarse, math.floor(edge / coarse) * coarse)

    edge = MAX_BLOCK_EDGE
    if sample is not None and len(sample):
        scale = point_count / len(sample)
        while edge / 2 >= MIN_BLOCK_EDGE:
            blocks = np.floor(sample / aligned(edge)).astype(np.int64)
            _, counts = np.unique(blocks, axis=0, return_counts=True)
            if counts[counts * scale > BLOCK_POINTS].sum() <= 0.1 * len(sample):
                break
            edge /= 2
        return aligned(edge)
    density = point_count / max(area, 1e-9)
    return aligned(min(MAX_BLOCK_EDGE, math.sqrt(BLOCK_POINTS / max(density, 1e-12))))


def sample_xy(sources, points=2_000_000, run=20_000):
    """XY of evenly spaced point runs across all sources (a few LAZ chunks each)."""
    from copc_metadata import laspy_laz_backend

    backend = laspy_laz_backend()
    counts = []
    for source in sources:
        with laspy.open(source) as reader:
            counts.append(int(reader.header.point_count))
    total = sum(counts)
    parts = []
    for source, count in zip(sources, counts):
        runs = max(1, math.ceil(points * count / max(total, 1) / run))
        kwargs = {"laz_backend": backend} if str(source).lower().endswith(".laz") and backend is not None else {}
        with laspy.open(source, **kwargs) as reader:
            for start in np.linspace(0, max(count - run, 0), runs).astype(np.int64):
                reader.seek(int(start))
                chunk = reader.read_points(min(run, count - int(start)))
                parts.append(np.c_[np.asarray(chunk.x), np.asarray(chunk.y)])
    return np.concatenate(parts) if parts else np.empty((0, 2))


def route(coords, masks, resolutions, cells):
    """Assign spilled points to blocks.

    A point's block comes from its voxel key at each resolution, so a voxel
    is never split. Where IEEE rounding puts a point's voxels into different
    blocks at different resolutions, it is written once per block, each copy
    contributing only to the resolutions whose block it is.
    """
    blocks = [np.floor_divide(np.floor(coords[:, :2] / r).astype(np.int64), k)
              for r, k in zip(resolutions, cells)]
    same = np.ones(len(coords), dtype=bool)
    for block in blocks[1:]:
        same &= (block == blocks[0]).all(axis=1)
    rows, block_ids, bits = [np.flatnonzero(same)], [blocks[0][same]], [masks[same]]
    split = np.flatnonzero(~same)
    for index, block in enumerate(blocks):
        bit = np.uint8(1 << index)
        rows_i = split[(masks[split] & bit) != 0]
        rows.append(rows_i)
        block_ids.append(block[rows_i])
        bits.append(np.full(len(rows_i), bit, dtype=np.uint8))
    return np.concatenate(rows), np.concatenate(block_ids), np.concatenate(bits)


def _coords(records, scales, offsets):
    # laspy computes scaled coordinates as X * scale + offset in float64.
    return np.stack([records[axis].astype(np.float64) * scales[i] + offsets[i]
                     for i, axis in enumerate(("X", "Y", "Z"))], axis=1)


def _write_blocks(directory, name, records, block_ids):
    if not len(records):
        return
    lo = block_ids.min(axis=0)
    span = int(block_ids[:, 1].max() - lo[1]) + 1
    linear = (block_ids[:, 0] - lo[0]) * span + (block_ids[:, 1] - lo[1])
    order = np.argsort(linear, kind="stable")
    linear = linear[order]
    starts = np.r_[0, np.flatnonzero(np.diff(linear)) + 1]
    for start, end in zip(starts, np.r_[starts[1:], len(order)]):
        bx, by = block_ids[order[start]]
        block = directory / f"{bx}_{by}"
        block.mkdir(parents=True, exist_ok=True)
        with open(block / name, "ab") as handle:
            handle.write(records[order[start:end]].tobytes())


def _encode(values, scale, offset):
    encoded = np.round((values - offset) / scale)
    if encoded.size and (encoded.min() < np.iinfo(np.int32).min or encoded.max() > np.iinfo(np.int32).max):
        raise ValueError("Tile coordinates exceed the LAS int32 range")
    return encoded.astype(np.int32)


def _spill_range(task):
    """Read one point range of one source; spill each point to every tile it falls in."""
    task_id, source, start, count, tiles, resolutions, cells, spill_dir = task
    from copc_metadata import laspy_laz_backend

    backend = laspy_laz_backend()
    kwargs = {"laz_backend": backend} if str(source).lower().endswith(".laz") and backend is not None else {}
    bounds = np.array([tile.bounds for tile in tiles], dtype=np.float64)
    counts = np.zeros(len(tiles), dtype=np.int64)
    core_counts = np.zeros(len(tiles), dtype=np.int64)
    full_mask = np.uint8((1 << len(resolutions)) - 1)
    name = f"{task_id:06d}.bin"
    read = 0
    with laspy.open(source, **kwargs) as reader:
        reader.seek(start)
        while read < count:
            chunk = reader.read_points(min(CHUNK_POINTS, count - read))
            if not len(chunk):
                raise ValueError(f"{source}: expected {count} points from {start}, read {read}")
            read += len(chunk)
            x, y, z = np.asarray(chunk.x), np.asarray(chunk.y), np.asarray(chunk.z)
            candidates = np.flatnonzero((bounds[:, 2] >= x.min()) & (bounds[:, 0] <= x.max())
                                        & (bounds[:, 3] >= y.min()) & (bounds[:, 1] <= y.max()))
            for i in candidates:
                xmin, ymin, xmax, ymax = bounds[i]
                rows = np.flatnonzero((x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax))
                if not len(rows):
                    continue
                tile = tiles[i]
                if tile.core is not None:
                    cx0, cy0, cx1, cy1 = tile.core
                    xs, ys = x[rows], y[rows]
                    core_counts[i] += int(np.count_nonzero((xs >= cx0) & (xs <= cx1) & (ys >= cy0) & (ys <= cy1)))
                records = np.empty(len(rows), dtype=RECORD)
                for axis, values in zip(("X", "Y", "Z"), (x, y, z)):
                    index = "XYZ".index(axis)
                    records[axis] = _encode(values[rows], tile.scales[index], tile.offsets[index])
                records["m"] = full_mask
                source_rows, block_ids, bits = route(_coords(records, tile.scales, tile.offsets),
                                                     records["m"], resolutions, cells)
                routed = records[source_rows]
                routed["m"] = bits
                _write_blocks(Path(spill_dir) / tile.label, name, routed, block_ids)
                counts[i] += len(rows)
    return counts, core_counts


def _round_half_even(numerator, denominator):
    """Exact integer round-half-to-even of numerator / denominator (denominator > 0)."""
    quotient, remainder = np.divmod(numerator, denominator)
    twice = 2 * remainder
    up = (twice > denominator) | ((twice == denominator) & (quotient % 2 == 1))
    return quotient + up


def reduce_records(records, resolutions, scales, offsets):
    """Mean of the points in each voxel, per resolution, in voxel-key order.

    Voxels use the scaled float coordinates (floor(x / r)), as before. Each
    center is the exact mean of the voxel's integer LAS coordinates, rounded
    half-to-even once onto the same LAS grid: no float summation, so results
    do not depend on point order or worker count. Returns
    [(centers, represented_points)]; centers are on the output LAS grid.
    """
    results = []
    for index, resolution in enumerate(resolutions):
        selected = records[(records["m"] & np.uint8(1 << index)) != 0]
        if not len(selected):
            results.append((np.empty((0, 3)), 0))
            continue
        keys = np.floor(_coords(selected, scales, offsets) / resolution).astype(np.int64)
        lo = keys.min(axis=0)
        span = [int(v) for v in keys.max(axis=0) - lo + 1]
        if span[0] * span[1] * span[2] < 2**62:
            packed = ((keys[:, 0] - lo[0]) * span[1] + (keys[:, 1] - lo[1])) * span[2] + (keys[:, 2] - lo[2])
            order = np.argsort(packed)
            packed = packed[order]
            starts = np.flatnonzero(np.r_[True, packed[1:] != packed[:-1]])
        else:
            order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
            keys = keys[order]
            starts = np.flatnonzero(np.r_[True, np.any(keys[1:] != keys[:-1], axis=1)])
        counts = np.diff(np.r_[starts, len(order)]).astype(np.int64)
        centers = np.empty((len(starts), 3))
        for axis, name in enumerate(("X", "Y", "Z")):
            sums = np.add.reduceat(selected[name][order].astype(np.int64), starts)
            centers[:, axis] = _round_half_even(sums, counts) * scales[axis] + offsets[axis]
        results.append((centers, len(selected)))
    return results


def _block_files(block):
    return sorted(path for path in block.iterdir() if path.suffix == ".bin")


def _reduce_block(task):
    """Reduce one spilled block, splitting it first when it exceeds BLOCK_POINTS."""
    block, resolutions, cells, scales, offsets, limit = task
    block = Path(block)
    files = _block_files(block)
    total = sum(path.stat().st_size for path in files) // RECORD.itemsize
    # One re-spill into enough children for the measured size (not repeated halving).
    factor = 2 ** max(1, math.ceil(math.log2(math.sqrt(max(total, 1) / limit)))) if total > limit else 1
    finer = tuple(max(1, cell // factor) for cell in cells)
    if total <= limit or finer == cells:
        records = np.concatenate([np.fromfile(path, dtype=RECORD) for path in files]) if files else \
            np.empty(0, dtype=RECORD)
        return reduce_records(records, resolutions, scales, offsets)
    # Skewed density: re-route into finer blocks without loading the whole block.
    children = block / "split"
    for path in files:
        with open(path, "rb") as handle:
            while len(records := np.fromfile(handle, dtype=RECORD, count=limit)):
                rows, block_ids, bits = route(_coords(records, scales, offsets), records["m"], resolutions, finer)
                routed = records[rows]
                routed["m"] = bits
                _write_blocks(children, path.name, routed, block_ids)
    parts = [[] for _ in resolutions]
    represented = [0] * len(resolutions)
    for child in sorted(children.iterdir(), key=_block_key):
        for index, (centers, used) in enumerate(_reduce_block((child, resolutions, finer, scales, offsets, limit))):
            parts[index].append(centers)
            represented[index] += used
    shutil.rmtree(children)
    return [(np.concatenate(p) if p else np.empty((0, 3)), used) for p, used in zip(parts, represented)]


def _block_key(path):
    bx, by = path.name.split("_")
    return int(bx), int(by)


def _range_tasks(sources, tiles, resolutions, cells, spill_dir, workers):
    """Split sources into point ranges (about two per worker, none tiny).

    Also returns the total point count and the area those points cover: the
    sources' combined XY extent clipped to the tiles (planned tiles can be
    mostly empty, so their area would underestimate the density).
    """
    counts, extent = [], [math.inf, math.inf, -math.inf, -math.inf]
    for source in sources:
        with laspy.open(source) as reader:
            counts.append(int(reader.header.point_count))
            h = reader.header
            extent = [min(extent[0], h.x_min), min(extent[1], h.y_min), max(extent[2], h.x_max), max(extent[3], h.y_max)]
    tiles_extent = [min(t.bounds[0] for t in tiles), min(t.bounds[1] for t in tiles),
                    max(t.bounds[2] for t in tiles), max(t.bounds[3] for t in tiles)]
    width = min(extent[2], tiles_extent[2]) - max(extent[0], tiles_extent[0])
    height = min(extent[3], tiles_extent[3]) - max(extent[1], tiles_extent[1])
    area = max(width, 0.0) * max(height, 0.0)
    size = max(CHUNK_POINTS, math.ceil(sum(counts) / max(1, 2 * workers)))
    tasks = []
    for source, total in zip(sources, counts):
        for start in range(0, total, size):
            tasks.append((len(tasks), str(source), start, min(size, total - start),
                          tiles, resolutions, cells, str(spill_dir)))
    return tasks, sum(counts), area


def _init_worker():
    # One LAZ decoder thread per process; the pool provides the parallelism.
    os.environ["RAYON_NUM_THREADS"] = "1"


def _ordered(executor, function, tasks, window, waited):
    """Yield results in task order with at most `window` tasks in flight.

    `waited[0]` accumulates the time spent blocked on results."""
    pending = {}
    submitted = 0
    for index in range(len(tasks)):
        while submitted < len(tasks) and submitted - index < window:
            pending[submitted] = executor.submit(function, tasks[submitted])
            submitted += 1
        started = time.monotonic()
        result = pending.pop(index).result()
        waited[0] += time.monotonic() - started
        yield result


# Fragments hold raw LAS point format 0 records (no header): assembly then
# reads them with plain file reads; thousands of LAS headers were the bottleneck.
FRAGMENT_POINT_FORMAT = 0
ASSEMBLY_POINTS = 5_000_000


def _reduce_and_write(task):
    """Reduce one block and write each resolution as a raw point-record fragment.

    Workers write their own fragments, so centers never travel through the
    parent process. Returns [(output points, represented points)] per resolution.
    """
    block, resolutions, cells, scales, offsets, limit, slot_dirs, name = task
    results = _reduce_block((block, resolutions, cells, scales, offsets, limit))
    shutil.rmtree(block)
    dtype = laspy.PointFormat(FRAGMENT_POINT_FORMAT).dtype()
    counts = []
    for slot, (centers, used) in enumerate(results):
        if len(centers):
            records = np.zeros(len(centers), dtype=dtype)
            for axis, field in enumerate(("X", "Y", "Z")):
                # Centers lie on the output grid: this recovers the exact integers.
                records[field] = np.round((centers[:, axis] - offsets[axis]) / scales[axis])
            records.tofile(Path(slot_dirs[slot]) / f"{name}.bin")
        counts.append((len(centers), used))
    return counts


def _fragments(directory):
    return sorted(Path(directory).glob("*.bin"))


def _fragment_batches(directory, header):
    """Yield point records of all fragments in block order, about ASSEMBLY_POINTS at a time."""
    dtype = laspy.PointFormat(FRAGMENT_POINT_FORMAT).dtype()
    pending, size = [], 0
    for fragment in _fragments(directory):
        pending.append(np.fromfile(fragment, dtype=dtype))
        size += len(pending[-1])
        if size >= ASSEMBLY_POINTS:
            yield laspy.ScaleAwarePointRecord(np.concatenate(pending), header.point_format,
                                              header.scales, header.offsets)
            pending, size = [], 0
    if pending:
        yield laspy.ScaleAwarePointRecord(np.concatenate(pending), header.point_format,
                                          header.scales, header.offsets)


def _assemble(directory, target, template, compress):
    """Concatenate fragments in block order into one LAS/LAZ with the tile header."""
    from copc_metadata import laspy_laz_backend
    from point_cloud_metadata import write_retained_evlrs

    with laspy.open(template) as reader:
        header = reader.header
    backend = laspy_laz_backend()
    kwargs = {"laz_backend": backend} if compress and backend is not None else {}
    with laspy.open(target, mode="w", header=header, do_compress=compress, **kwargs) as writer:
        for points in _fragment_batches(directory, header):
            writer.write_points(points)
        write_retained_evlrs(writer, header)
    shutil.rmtree(directory)
    return True


def _assemble_ply(directory, target, template, expected):
    """Binary little-endian PLY of float64 XYZ, with the CRS as `comment crs:`."""
    from crs_records import crs_comment_value
    from ply_crs import add_crs_comment_to_ply

    with laspy.open(template) as reader:
        header = reader.header
    crs = crs_comment_value(header)
    written = 0
    with open(target, "wb") as handle:
        handle.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {expected}\n"
                      "property double x\nproperty double y\nproperty double z\nend_header\n").encode("ascii"))
        for points in _fragment_batches(directory, header):
            xyz = np.empty((len(points), 3), dtype="<f8")
            xyz[:, 0], xyz[:, 1], xyz[:, 2] = points.x, points.y, points.z
            handle.write(xyz.tobytes())
            written += len(points)
    if written != expected:
        raise ValueError(f"{Path(target).name}: wrote {written:,} of {expected:,} points")
    add_crs_comment_to_ply(target, crs)
    shutil.rmtree(directory)
    return True


def _fragments_to_copc(directory, target, template, metadata_source, expected):
    """One LAS per tile from its fragments, then the shared COPC converter (Untwine, PDAL fallback)."""
    from subsample_outputs import convert_laz_output_to_copc

    merged = Path(str(directory) + ".las")
    _assemble(directory, merged, template, compress=False)
    converted = convert_laz_output_to_copc(merged, target, source_metadata_file=metadata_source)
    merged.unlink(missing_ok=True)
    if converted:
        with laspy.open(target) as reader:
            if reader.header.point_count != expected:
                raise ValueError(f"{Path(target).name}: COPC holds {reader.header.point_count:,} of {expected:,} points")
    return converted


def write_tile_centroids(sources, tiles, header, outputs, *, workers, work_dir, metadata_source,
                         converters=2):
    """Write center-of-mass outputs for every tile from one read of the sources.

    `tiles`: [Tile]. `header`: LAS header the tiles derive from (VLRs, CRS).
    `outputs`: [(resolution, {label: final path}, format)] in output order;
    format is laz, copc.laz or ply.
    Tiles with a `core` and no source point inside it are dropped (the layout
    is planned before reading). Returns ({label: [output point counts]},
    [occupied flag per tile]). Fails before publishing on any point-count
    mismatch.
    """
    from point_cloud_metadata import write_retained_evlrs
    from main_tile import _make_tile_header
    from subsample_com import make_center_of_mass_header

    resolutions = tuple(float(resolution) for resolution, _, _ in outputs)
    if not tiles or not resolutions or any(not math.isfinite(r) or r <= 0 for r in resolutions):
        raise ValueError("Tile centroids need tiles and positive resolutions")
    if len(resolutions) > 8:
        raise ValueError("At most eight resolutions per scan")
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    workers = max(1, int(workers))
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix=".tile-centroids-", dir=work_dir) as scratch:
        scratch = Path(scratch)
        spill_dir = scratch / "blocks"
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=_init_worker) as pool:
            probe, total_points, area = _range_tasks(sources, tiles, resolutions, (1,) * len(resolutions),
                                                     spill_dir, workers)
            block_size = choose_block_size(total_points, area, resolutions, sample_xy(sources))
            cells = block_cells(resolutions, block_size)
            tasks = [task[:6] + (cells, task[7]) for task in probe]
            tile_points = np.zeros(len(tiles), dtype=np.int64)
            core_points = np.zeros(len(tiles), dtype=np.int64)
            for counts, cores in pool.map(_spill_range, tasks):
                tile_points += counts
                core_points += cores
            # Occupancy comes from the same read: drop tiles whose core is empty.
            occupied = np.array([tile.core is None or core_points[i] > 0 for i, tile in enumerate(tiles)])
            if not occupied.any():
                raise ValueError("No planned tile core contains source points")
            for index in np.flatnonzero(~occupied):
                shutil.rmtree(spill_dir / tiles[index].label, ignore_errors=True)
            spilled = time.monotonic()
            spill_bytes = sum(path.stat().st_size for path in spill_dir.rglob("*.bin"))
            print(f"  Read {total_points:,} source points once with {workers} workers in {spilled - started:.1f} s; "
                  f"{block_size:.1f} m blocks; {spill_bytes / 2**30:.2f} GiB spilled")

            # Every block writes its own fragments; tiles share one header template.
            jobs, owners, templates = [], [], {}
            for index, tile in enumerate(tiles):
                if not occupied[index]:
                    continue
                tile_out = scratch / "out" / tile.label
                slot_dirs = [str(tile_out / str(slot)) for slot in range(len(outputs))]
                for slot_dir in slot_dirs:
                    Path(slot_dir).mkdir(parents=True)
                tile_header = _make_tile_header(header, offsets=np.array(tile.offsets), scales=np.array(tile.scales))
                out_header = make_center_of_mass_header(tile_header, dimension_reduction=True)
                template = tile_out / "header.las"
                with laspy.open(template, mode="w", header=out_header) as writer:
                    write_retained_evlrs(writer, out_header)
                templates[index] = template
                tile_dir = spill_dir / tile.label
                blocks = sorted(tile_dir.iterdir(), key=_block_key) if tile_dir.exists() else []
                if out_header.point_format.id != FRAGMENT_POINT_FORMAT:
                    raise ValueError("Center-of-mass outputs must use point format 0")
                for block in blocks:
                    jobs.append((block, resolutions, cells, tile.scales, tile.offsets, BLOCK_POINTS,
                                 slot_dirs, f"{len(jobs):09d}"))
                    owners.append(index)
            last_block = {owner: position for position, owner in enumerate(owners)}

            written, staged, finishing = {}, [], []
            totals = {index: [[0, 0] for _ in outputs] for index in templates}
            finish = ThreadPoolExecutor(max_workers=max(1, converters))
            waited = [0.0]
            try:
                for position, results in enumerate(_ordered(pool, _reduce_and_write, jobs, 2 * workers, waited)):
                    index = owners[position]
                    for slot, (count, used) in enumerate(results):
                        totals[index][slot][0] += count
                        totals[index][slot][1] += used
                    if last_block[index] != position:
                        continue
                    tile = tiles[index]
                    for slot, (resolution, paths, output_format) in enumerate(outputs):
                        count, used = totals[index][slot]
                        if used != tile_points[index] or not count:
                            raise ValueError(
                                f"{tile.label} at {resolution} m: voxels represent {used:,} "
                                f"points of {int(tile_points[index]):,}; {count:,} outputs")
                        directory = scratch / "out" / tile.label / str(slot)
                        target = scratch / f"{tile.label}.{slot}.{output_format}"
                        if output_format == "copc.laz":
                            future = finish.submit(_fragments_to_copc, directory, target, templates[index],
                                                   metadata_source, count)
                        elif output_format == "ply":
                            future = finish.submit(_assemble_ply, directory, target, templates[index], count)
                        else:
                            future = finish.submit(_assemble, directory, target, templates[index], True)
                        finishing.append((target, paths[tile.label], future))
                    written[tile.label] = [count for count, _ in totals[index]]
                reduced = time.monotonic()
                for target, final, future in finishing:
                    if not future.result():
                        raise RuntimeError(f"COPC conversion failed for {final.name}")
                    staged.append((target, final))
            finally:
                finish.shutdown(wait=True)
        missing = [tile.label for index, tile in enumerate(tiles)
                   if occupied[index] and tile_points[index] and tile.label not in written]
        if missing:
            raise ValueError(f"Tiles without outputs: {missing}")
        # Publish only after every tile and resolution passed its checks.
        for path, final in staged:
            final.parent.mkdir(parents=True, exist_ok=True)
            path.replace(final)
    print(f"  Tile centroids: {len(written)} tiles, {resolutions} m, in {time.monotonic() - started:.1f} s: "
          f"spill {spilled - started:.1f} s, reduce/write {reduced - spilled:.1f} s "
          f"(waiting on blocks {waited[0]:.1f} s), remaining conversion/assembly {time.monotonic() - reduced:.1f} s")
    return written, occupied.tolist()


def subsample_tiles_single_scan(plan, output_dir, resolutions, formats, *, workers, converters,
                                output_prefix):
    """Tile-task subsampling without full-resolution tiles: one output dir per resolution.

    Output i goes to subsampled_res{i+1}; `formats` are laz, copc.laz or ply.

    Tiled inputs use the planned buffered tiles, each encoded with the first
    source's scales and the tile-centre offsets the COPC tiles used. A
    single-cloud plan is one unbounded tile in the source's own encoding.
    """
    from main_subsample import subsample_output_name
    from main_tile import plot_overview, read_tile_jobs
    from tile_core_occupancy import publish_occupied_tiles

    sources = [Path(source) for source in plan.source_files]
    with laspy.open(sources[0]) as reader:
        header = reader.header
    scales = tuple(float(v) for v in header.scales)
    if plan.skip_tiling:
        stem = sources[0].name[:-len(".copc.laz")] if sources[0].name.endswith(".copc.laz") else sources[0].stem
        tiles = [Tile(stem, (-math.inf, -math.inf, math.inf, math.inf), scales,
                      tuple(float(v) for v in header.offsets))]
    else:
        # The plan holds every grid tile; occupancy is decided by the single read.
        layout = json.loads(Path(plan.bounds_json).read_text())["tiles"]
        cores = {f"c{t['col']:02d}_r{t['row']:02d}": t["core"] for t in layout}
        jobs = read_tile_jobs(plan.jobs_file)
        if sorted(jobs) != sorted(cores):
            raise ValueError("Tile jobs and tile layout disagree")
        tiles = [Tile(label, bounds, scales,
                      ((bounds[0] + bounds[2]) / 2.0, (bounds[1] + bounds[3]) / 2.0, float(header.offsets[2])),
                      core=(cores[label][0][0], cores[label][1][0], cores[label][0][1], cores[label][1][1]))
                 for label, bounds in sorted(jobs.items())]
    output_dirs = [Path(output_dir) / f"subsampled_res{index + 1}" for index in range(len(resolutions))]
    def name(tile, resolution, output_format):
        stem = subsample_output_name(Path(f"{tile.label}.copc.laz"), resolution, output_prefix, False)
        return stem[:-len(".laz")] + "." + output_format

    outputs = [(resolution, {tile.label: directory / name(tile, resolution, output_format) for tile in tiles},
                output_format)
               for resolution, directory, output_format in zip(resolutions, output_dirs, formats)]
    print()
    print("=" * 60)
    print("Single-scan tile subsampling")
    print("=" * 60)
    print(f"  Tiles: {len(tiles)}; resolutions: {', '.join(f'{r} m' for r in resolutions)}; "
          "full-resolution tiles are not written")
    _, occupied = write_tile_centroids(sources, tiles, header, outputs, workers=workers, work_dir=output_dir,
                                       metadata_source=sources[0], converters=converters)
    if not plan.skip_tiling:
        by_label = dict(zip((tile.label for tile in tiles), occupied))
        layout = json.loads(Path(plan.bounds_json).read_text())["tiles"]
        publish_occupied_tiles(plan.bounds_json, plan.jobs_file,
                               [by_label[f"c{t['col']:02d}_r{t['row']:02d}"] for t in layout])
        plot_overview(plan)
    return tuple(output_dirs)
