"""Publish the tiles whose core contains source points (counted by the single-scan read)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from prepare_tile_jobs import write_job_list


def publish_occupied_tiles(bounds_json, jobs_file, keep):
    """Rewrite the layout and job list with only the populated cores.

    Grid coordinates are preserved and skipped cores recorded separately, so
    neighbor discovery uses the actual layout.
    """
    bounds_json, jobs_file = Path(bounds_json), Path(jobs_file)
    data = json.loads(bounds_json.read_text())
    tiles = data['tiles']
    keep = np.asarray(keep, dtype=bool)
    if len(keep) != len(tiles):
        raise ValueError('Occupancy must cover every planned tile')
    if not keep.any():
        raise ValueError('No planned tile core contains source points')
    skipped = [tile for tile, populated in zip(tiles, keep) if not populated]
    data['tiles'] = [tile for tile, populated in zip(tiles, keep) if populated]
    data['tile_count'] = len(data['tiles'])
    data['core_occupancy'] = {
        'policy': 'at least one original source point inside the inclusive XY core',
        'planned_tiles': len(tiles),
        'retained_tiles': len(data['tiles']),
        'skipped_tiles': [dict(tile, reason='no_source_points_in_core') for tile in skipped],
    }
    temporary = bounds_json.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(bounds_json)
    write_job_list(bounds_json, jobs_file)
    print(f"  Core occupancy: {len(data['tiles'])} tiles retained; {len(skipped)} empty cores skipped")
    return len(data['tiles'])
