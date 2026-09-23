"""Recover rejected whole trees only where no retained tree covers core geometry.

The source geometry and full-instance anchors are fixed before this stage. A
scratch SQLite table keeps distinct uncovered 3D samples and claimant IDs on
disk, so selection does not grow with the number of dense points in memory.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import laspy
import numpy as np
from scipy.spatial import cKDTree

from bounded_point_index import MAX_BATCH_POINTS, PointIndex, coordinates, distance_limit, spatial_batches
from dense_tile_merge import DUPLICATE_RADIUS, index_file
from point_cloud_metadata import copy_single_source_header, write_retained_evlrs


def _positive_support(index, xyz, instance_dimension):
    """A nearby background record never counts as a tree claimant."""
    found = np.zeros(len(xyz), dtype=bool)
    for group in spatial_batches(xyz):
        points = xyz[group]
        limit = distance_limit(points, DUPLICATE_RADIUS)
        for _, data in index.candidates(points, DUPLICATE_RADIUS):
            positive = np.asarray(data['values'][instance_dimension]) > 0
            if not np.any(positive):
                continue
            distances, _ = index.query_tree(cKDTree(data['xyz'][positive]), points)
            found[group] |= distances <= limit
            if np.all(found[group]):
                break
    return found


def _core_union_mask(xyz, tile, regions, overlaps, origin):
    result = np.zeros(len(xyz), dtype=bool)
    for other, region in enumerate(regions):
        if other != tile and min(tile, other) not in overlaps[max(tile, other)]:
            continue
        (x0, x1), (y0, y1) = region['core']
        x0, x1, y0, y1 = x0 - origin[0], x1 - origin[0], y0 - origin[1], y1 - origin[1]
        result |= ((xyz[:, 0] >= x0) & (xyz[:, 0] <= x1) &
                   (xyz[:, 1] >= y0) & (xyz[:, 1] <= y1))
    return result


def recover_orphaned_instances(model, dense_files, owned_files, owned_index,
                               regions, overlaps, output_dir, recovered_index,
                               origin, counts, report):
    """Admit rejected whole instances with the most unsupported core samples.

    The provisional ownership decisions were calculated on complete dense
    instances by filter_owned_instances. Exact XYZ keys identify distinct dense
    source samples; the support check itself uses the pipeline's 1 cm spatial
    duplicate radius. Stable source order and local ID break equal-score ties.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    decisions = report['instance_ownership']['tiles']
    rejected = [{d['instance'] for d in tile['instances'] if not d['kept']}
                for tile in decisions]
    metric = report['orphan_recovery'] = {
        'policy': 'whole rejected instance only for unsupported positive core geometry',
        'support_radius_m': DUPLICATE_RADIUS,
        'tie_break': 'descending new distinct XYZ core samples, then source tile and local ID',
        'candidates': 0, 'uncovered_locations': 0, 'admitted': [],
        'final_support': 'pending'}
    db_path = output_dir / 'orphan_claims.sqlite'
    db = sqlite3.connect(db_path)
    try:
        db.execute('PRAGMA journal_mode=OFF')
        db.execute('PRAGMA synchronous=OFF')
        db.execute('CREATE TABLE claims (tile INTEGER, instance INTEGER, loc BLOB, '
                   'PRIMARY KEY(tile,instance,loc)) WITHOUT ROWID')
        db.execute('CREATE INDEX claims_loc ON claims(loc)')
        db.execute('CREATE TABLE covered (loc BLOB PRIMARY KEY) WITHOUT ROWID')
        for tile, file in enumerate(dense_files):
            if not rejected[tile]:
                continue
            with laspy.open(file) as reader:
                for record in reader.chunk_iterator(MAX_BATCH_POINTS):
                    labels = np.asarray(record[model.instance])
                    eligible = np.isin(labels, list(rejected[tile]))
                    if not np.any(eligible):
                        continue
                    xyz = coordinates(record, reader.header, origin)
                    eligible &= _core_union_mask(xyz, tile, regions, overlaps, origin)
                    if not np.any(eligible):
                        continue
                    positions = np.flatnonzero(eligible)
                    uncovered = ~_positive_support(owned_index, xyz[positions], model.instance)
                    positions = positions[uncovered]
                    if len(positions):
                        db.executemany('INSERT OR IGNORE INTO claims VALUES (?,?,?)',
                            ((tile, int(labels[pos]), xyz[pos].astype('<f8').tobytes())
                             for pos in positions))
        db.commit()
        metric['candidates'] = db.execute(
            'SELECT COUNT(*) FROM (SELECT 1 FROM claims GROUP BY tile,instance)').fetchone()[0]
        metric['uncovered_locations'] = db.execute(
            'SELECT COUNT(DISTINCT loc) FROM claims').fetchone()[0]
        selected = []
        # A second candidate may use different sampled XYZ coordinates within
        # the duplicate radius of the first. Recheck support after every
        # admission instead of treating byte-distinct coordinates as gaps.
        with PointIndex(output_dir / 'selection.sqlite', {model.instance: np.uint8},
                        query_workers=owned_index.query_workers) as selection_index:
            while True:
                best = db.execute('SELECT tile,instance,COUNT(*) AS score FROM claims '
                    'WHERE loc NOT IN (SELECT loc FROM covered) '
                    'GROUP BY tile,instance ORDER BY score DESC,tile,instance LIMIT 1').fetchone()
                if best is None:
                    break
                tile, uid, score = map(int, best)
                if score <= 0:
                    break
                selected.append((tile, uid))
                cursor = db.execute('SELECT loc FROM claims WHERE tile=? AND instance=?', (tile, uid))
                offset = 0
                while rows := cursor.fetchmany(MAX_BATCH_POINTS):
                    xyz = np.array([np.frombuffer(row[0], dtype='<f8') for row in rows])
                    selection_index.add(tile, xyz, {model.instance: np.ones(len(xyz), dtype=np.uint8)},
                                        np.arange(offset, offset + len(xyz), dtype=np.int64))
                    offset += len(xyz)
                selection_index.flush()
                # Read claims in bounded batches. The claims table is immutable
                # while covered grows; each subsequent score uses updated support.
                cursor = db.execute('SELECT DISTINCT loc FROM claims '
                                    'WHERE loc NOT IN (SELECT loc FROM covered)')
                while rows := cursor.fetchmany(MAX_BATCH_POINTS):
                    xyz = np.array([np.frombuffer(row[0], dtype='<f8') for row in rows])
                    support = _positive_support(selection_index, xyz, model.instance)
                    db.executemany('INSERT OR IGNORE INTO covered(loc) VALUES (?)',
                                   (rows[pos] for pos in np.flatnonzero(support)))
                metric['admitted'].append({'tile': tile, 'local_instance': uid,
                                          'new_locations': score,
                                          'source': report['tile_sources'][tile]['prediction']})
        db.commit()
        if not selected:
            metric['final_support'] = 'no recovery needed'
            return owned_files, counts, selected, db_path
        by_tile = {}
        for tile, uid in selected:
            by_tile.setdefault(tile, set()).add(uid)
            decision = next(d for d in decisions[tile]['instances'] if d['instance'] == uid)
            decision['kept'] = True
            decision['disposition'] = 'recovered_orphan'
            decision['recovered'] = True
            counts[tile, uid] = decision['points']
        outputs = []
        for tile, (dense, owned) in enumerate(zip(dense_files, owned_files)):
            if tile not in by_tile:
                output = owned
            else:
                output = output_dir / dense.name
                original_rejected = rejected[tile] - by_tile[tile]
                with laspy.open(dense) as reader:
                    header = copy_single_source_header(reader.header)
                    with laspy.open(output, mode='w', header=header) as writer:
                        for record in reader.chunk_iterator(MAX_BATCH_POINTS):
                            labels = np.asarray(record[model.instance])
                            keep = ~np.isin(labels, list(original_rejected))
                            background = labels == 0
                            if np.any(background):
                                from dense_instance_ownership import owned_background
                                xy = coordinates(record, reader.header, origin)[background, :2] + origin[:2]
                                keep[background] = owned_background(xy, regions[tile])
                            writer.write_points(record[keep])
                        write_retained_evlrs(writer, header)
                with laspy.open(output) as reader:
                    revised = int(reader.header.point_count)
                decisions[tile]['surviving'] = revised
                decisions[tile]['removed'] = decisions[tile]['input'] - revised
                decisions[tile]['kept_instances'] += len(by_tile[tile])
                decisions[tile]['removed_instances'] -= len(by_tile[tile])
                decisions[tile]['recovered_instances'] = sorted(by_tile[tile])
            outputs.append(output)
            index_file(recovered_index, output, tile, origin, model)
        metric['final_support'] = 'pending' if selected else 'no recovery needed'
        return outputs, counts, selected, db_path
    finally:
        db.close()


def validate_recovered_geometry(index, model, selected, claims_path, report, origin):
    """Detect loss of required recovered geometry after owner selection/dedup."""
    if not selected:
        return
    checked, missing, examples = 0, 0, []
    with sqlite3.connect(claims_path) as db:
        for tile, uid in selected:
            cursor = db.execute('SELECT loc FROM claims WHERE tile=? AND instance=?', (tile, uid))
            while rows := cursor.fetchmany(MAX_BATCH_POINTS):
                xyz = np.array([np.frombuffer(row[0], dtype='<f8') for row in rows])
                supported = _positive_support(index, xyz, model.instance)
                checked += len(xyz)
                missing += int(np.count_nonzero(~supported))
                for point in xyz[~supported][:max(0, 5 - len(examples))]:
                    examples.append({'tile': tile, 'local_instance': uid,
                                     'xyz': (point + origin).tolist()})
    metric = report['orphan_recovery']
    metric['final_support'] = {'checked_samples': checked, 'missing_samples': missing,
                               'missing_examples': examples}
    if missing:
        raise ValueError(f'{model.name}: recovered tree geometry lost after ownership '
                         f'and deduplication: {missing}/{checked} core samples unsupported; '
                         f'see orphan_recovery.final_support in report')
