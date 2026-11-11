"""Deduplication queries for one dense tile, answered block by block.

``deduplicate`` asks two questions per 1 M-point chunk: does any earlier tile
hold a conflicting label within the duplicate radius (``conflicting_match``),
and which earlier survivor is nearest (``survivors.nearest``). Both read their
regions per 2 m group from SQLite, again for every chunk that visits a block.

Here the tile's points are spilled once (``block_nearest.spill_file``) with
their mapped labels, and:

* the conflict check runs per 8 m block as a conservative screen: a group is
  checked exactly, by ``conflicting_match`` itself, when one of a point's
  eight nearest earlier dense points within its bound carries a different
  label (or all eight lie within it). A conflict anywhere sends the caller back
  to the per-chunk path, which raises exactly the error it always raised;
* survivors are answered by ``block_nearest`` with ``before_tile`` and the
  tile's overlaps, which equals ``survivors.nearest`` point by point.
"""
from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import laspy
import numpy as np
from scipy.spatial import cKDTree

from block_nearest import SPILL_DTYPE, NearestResults, _indexed_points, cell_blocks, cell_jobs, spill_file
from bounded_point_index import MAX_BATCH_POINTS, PointIndex, inside_xy, read_chunks
from point_ranges import seekable

NEIGHBORS = 8


def dedup_answers(model, mapping, tile, file, dense_index, survivor_index, overlaps, background_semantics_owned,
                  core_regions, origin, radius, work, *, processes):
    """``(answers, conflict_free)``: survivor answers by record number, and whether no conflict exists.

    Without conflict, ``answers`` (a NearestResults) holds what
    ``survivor_index.nearest(..., before_tile=tile, overlaps=overlaps)``
    returns for every record; with one, the caller must use the per-chunk path.
    """
    work = Path(work)
    spill = spill_file(file, origin, radius, work / 'points', processes=processes, chunk_points=MAX_BATCH_POINTS)
    labels = _label_rows(model, mapping, tile, file, work / 'labels.npy', spill.rows, processes)
    jobs = [(paths, str(dense_index.path), dense_index.dimensions, model, mapping, tile, overlaps,
             background_semantics_owned, core_regions, origin, radius, str(labels))
            for paths in cell_jobs(spill.cells, processes)]
    conflict_free = all(_map(_screen_cells, jobs, processes))
    Path(labels).unlink()
    answers = NearestResults(work / 'answers', survivor_index.dimensions, spill.rows)
    if conflict_free:
        spill.answer(survivor_index.path, survivor_index.dimensions, radius, answers, workers=processes,
                     before_tile=tile, overlaps=overlaps)
    else:
        for path in spill.cells.values():
            Path(path).unlink()
        spill.folder.rmdir()
    return answers, conflict_free


def _map(function, jobs, processes):
    if processes > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(processes, len(jobs)),
                                 mp_context=multiprocessing.get_context('spawn')) as pool:
            return list(pool.map(function, *zip(*jobs)))
    return [function(*job) for job in jobs]


def _label_rows(model, mapping, tile, file, path, rows, processes):
    """Write the mapped label matrix (instance, semantic) of every record to ``path``, by record number."""
    columns = 2 if model.semantic else 1
    np.lib.format.open_memmap(path, mode='w+', dtype=np.int64, shape=(rows, columns)).flush()
    chunks = max(1, -(-rows // MAX_BATCH_POINTS))
    # Parts after the first seek: a variable-chunk file is labelled in one part.
    share = max(1, -(-chunks // max(1, processes))) if seekable(file) else chunks
    jobs = [(model, mapping, tile, str(file), str(path), first, min(first + share, chunks))
            for first in range(0, chunks, share)]
    _map(_label_part, jobs, processes)
    return path


def _label_part(model, mapping, tile, file, path, first, stop):
    from merge_stages import label_matrix, mapped_values, prediction_values
    labels = np.load(path, mmap_mode='r+')
    with laspy.open(file) as reader:
        if first:
            reader.seek(first * MAX_BATCH_POINTS)
        for number, record in enumerate(read_chunks(reader, MAX_BATCH_POINTS), start=first):
            if number >= stop:
                break
            values = mapped_values(model, prediction_values(record, model.dimensions), tile, mapping)
            start = number * MAX_BATCH_POINTS
            labels[start:start + len(record)] = label_matrix(model, values)
    labels.flush()


def _screen_cells(spill_paths, dense_path, dimensions, model, mapping, tile, overlaps, background_semantics_owned,
                   core_regions, origin, radius, labels_path):
    """True when no group of these cells has a conflict (exact check for every suspicious group)."""
    with PointIndex(dense_path, dimensions, query_workers=1, read_only=True) as dense:
        return all(_screen_cell(dense, path, model, mapping, tile, overlaps, background_semantics_owned,
                                core_regions, origin, radius, labels_path) for path in spill_paths)


def _screen_cell(dense, spill_path, model, mapping, tile, overlaps, background_semantics_owned, core_regions,
                 origin, radius, labels_path):
    from merge_stages import _core_rule, label_matrix, mapped_values
    records = np.fromfile(spill_path, dtype=SPILL_DTYPE)
    labels = np.load(labels_path, mmap_mode='r')[records['row']]
    xyz = records['xyz']
    rule = _core_rule(core_regions, origin)
    dtype = dense._storage_dtype()
    names = list(dense.dimensions)
    for block in cell_blocks(xyz):
        suspicious = _suspicious(dense, dtype, names, model, mapping, tile, overlaps, background_semantics_owned,
                                 xyz[block], records['limit'][block], labels[block], mapped_values, label_matrix)
        for group in np.unique(records['group'][block][suspicious]):
            members = block[records['group'][block] == group]
            conflict = dense.conflicting_match(
                xyz[members], labels[members], radius, before_tile=tile,
                map_labels=lambda other, data: label_matrix(model, mapped_values(model, data, other, mapping)),
                background_semantics_owned=background_semantics_owned,
                core_preferred=(None if rule is None else
                                lambda first, second, pts: rule(tile if first is None else first,
                                                                tile if second is None else second, pts)),
                overlaps=overlaps)
            if conflict is not None:
                return False
    return True


def _suspicious(dense, dtype, names, model, mapping, tile, overlaps, background_semantics_owned,
                xyz, limits, labels, mapped_values, label_matrix):
    """Points that might have a conflicting earlier neighbor within their bound (a superset)."""
    halo = limits.max()
    lo, hi = xyz.min(axis=0) - halo, xyz.max(axis=0) + halo
    data, tiles = _indexed_points(dense.db, dtype, lo, hi, None, tile)
    suspicious = np.zeros(len(xyz), dtype=bool)
    for source in sorted(set(tiles.tolist())):
        if overlaps is not None and source not in overlaps:
            continue
        selected = np.flatnonzero(tiles == source)
        allowed = np.ones(len(xyz), dtype=bool)
        if overlaps is not None:
            selected = selected[inside_xy(data['xyz'][selected], overlaps[source])]
            allowed = inside_xy(xyz, overlaps[source])
        if not len(selected) or not np.any(allowed):
            continue
        values = {name: data[f'value_{i}'][selected] for i, name in enumerate(names)}
        other = label_matrix(model, mapped_values(model, values, source, mapping))
        k = min(NEIGHBORS, len(selected))
        ds, ns = cKDTree(data['xyz'][selected]).query(xyz, k=k, distance_upper_bound=np.nextafter(halo, np.inf))
        ds, ns = ds.reshape(len(xyz), k), ns.reshape(len(xyz), k)
        near = ds <= limits[:, None]
        # Possibly more neighbors within the bound than were looked at.
        suspicious |= allowed & near[:, -1] & (k == NEIGHBORS)
        neighbor = other[np.minimum(ns, len(selected) - 1)]
        disagree = np.any(neighbor != labels[:, None, :], axis=2)
        if background_semantics_owned:
            disagree &= ~((neighbor[:, :, 0] == 0) & (labels[:, None, 0] == 0))
        suspicious |= allowed & np.any(near & disagree, axis=1)
    return suspicious
