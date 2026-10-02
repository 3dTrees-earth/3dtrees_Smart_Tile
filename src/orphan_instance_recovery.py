"""Recover rejected whole trees only where no retained tree covers core geometry.

The source geometry and full-instance anchors are fixed before this stage. A
scratch SQLite table keeps distinct uncovered 3D samples and claimant IDs on
disk, so selection does not grow with the number of dense points in memory.
"""
from __future__ import annotations

import sqlite3
from time import perf_counter

import laspy
import numpy as np
from bounded_point_index import MAX_BATCH_POINTS, coordinates
from dense_tile_merge import DUPLICATE_RADIUS, index_file
from point_cloud_metadata import copy_single_source_header, write_retained_evlrs
from orphan_claims import select_claims
from parallel_index_queries import IndexQueries


def _positive_support(index, xyz, instance_dimension):
    """A nearby background record never counts as a tree claimant."""
    return index.covered(xyz, DUPLICATE_RADIUS, positive_dimension=instance_dimension)


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


def _claim_candidates(model, dense_files, rejected, regions, overlaps, origin):
    """Yield ``((tile, labels), xyz)`` for rejected-instance points inside any relevant core."""
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
                if np.any(eligible):
                    yield (tile, labels[eligible]), xyz[eligible]


def recover_orphaned_instances(model, dense_files, owned_files, owned_index,
                               regions, overlaps, output_dir, recovered_index,
                               origin, counts, report, *, admit_candidate=None, workers=1):
    """Admit rejected whole instances with the most unsupported core samples.

    The provisional ownership decisions were calculated on complete dense
    instances by filter_owned_instances. Exact XYZ keys identify distinct dense
    source samples; the support check itself uses the pipeline's 1 cm spatial
    duplicate radius. Stable source order and local ID break equal-score ties.
    An optional admission gate may reject a whole candidate before it supplies
    coverage. Pass no recovered_index when a later output stage builds its own.
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
        db.execute('PRAGMA temp_store=FILE')
        db.execute('CREATE TABLE claims (tile INTEGER, instance INTEGER, loc BLOB, '
                   'PRIMARY KEY(tile,instance,loc)) WITHOUT ROWID')
        db.execute('CREATE INDEX claims_loc ON claims(loc)')
        db.execute('CREATE TABLE covered (loc BLOB PRIMARY KEY) WITHOUT ROWID')
        started = perf_counter()
        # Support checks are GIL-bound per-cell searches on the finished owned
        # index: answer them in worker processes; claims form a set, so only
        # their content (not insertion order) matters.
        with IndexQueries(owned_index, DUPLICATE_RADIUS, method='covered', positive_dimension=model.instance,
                          workers=max(1, int(workers)) if any(rejected) else 1) as queries:
            for (tile, labels), xyz, covered in queries.map(
                    _claim_candidates(model, dense_files, rejected, regions, overlaps, origin)):
                uncovered = ~covered
                if np.any(uncovered):
                    db.executemany('INSERT OR IGNORE INTO claims VALUES (?,?,?)',
                        ((tile, int(label), point.astype('<f8').tobytes())
                         for label, point in zip(labels[uncovered], xyz[uncovered])))
        db.commit()
        metric['claim_seconds'] = perf_counter() - started
        metric['candidates'] = db.execute(
            'SELECT COUNT(*) FROM (SELECT 1 FROM claims GROUP BY tile,instance)').fetchone()[0]
        metric['uncovered_locations'] = db.execute(
            'SELECT COUNT(DISTINCT loc) FROM claims').fetchone()[0]
        selected = []
        started = perf_counter()
        for tile, uid, score in select_claims(db, owned_index.query_tree, metric,
                                            admit_candidate=admit_candidate):
            selected.append((tile, uid))
            metric['admitted'].append({'tile': tile, 'local_instance': uid,
                                      'new_locations': score,
                                      'source': report['tile_sources'][tile]['prediction']})
        db.commit()
        metric['selection_seconds'] = perf_counter() - started
        if not selected:
            metric['final_support'] = 'no recovery needed'
            return owned_files, counts, selected, db_path
        by_tile = {}
        decisions_by_id = [{d['instance']: d for d in tile['instances']} for tile in decisions]
        for tile, uid in selected:
            by_tile.setdefault(tile, set()).add(uid)
            decision = decisions_by_id[tile][uid]
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
            if recovered_index is not None:
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
