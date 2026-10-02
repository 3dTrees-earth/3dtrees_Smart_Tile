"""Select populated cores before writing buffered tiles or scheduling models."""
from __future__ import annotations

import json
from pathlib import Path

import laspy
import numpy as np

from prepare_tile_jobs import write_job_list


def occupied_cores(source_files, tiles, *, chunk_size=1_000_000):
    """Return one boolean per declared core, using inclusive XY boundaries.

    All sources contribute to occupancy. Once a core is populated it needs no
    further checks; stop reading when every core is populated. No point payload
    is retained between chunks, and no bounds-only inference counts as a hit.
    """
    cores = np.asarray([tile['core'] for tile in tiles], dtype=np.float64)
    if (cores.shape != (len(tiles), 2, 2) or not np.all(np.isfinite(cores))
            or np.any(cores[:, :, 0] > cores[:, :, 1])):
        raise ValueError('Core occupancy requires finite, ordered XY core bounds')
    occupied = np.zeros(len(tiles), dtype=bool)
    for source in source_files:
        if occupied.all():
            break
        with laspy.open(source) as reader:
            for chunk in reader.chunk_iterator(max(1, min(int(chunk_size), 1_000_000))):
                if not len(chunk):
                    continue
                x, y = np.asarray(chunk.x), np.asarray(chunk.y)
                candidates = np.flatnonzero(
                    ~occupied & (cores[:, 0, 0] <= x.max())
                    & (cores[:, 0, 1] >= x.min())
                    & (cores[:, 1, 0] <= y.max())
                    & (cores[:, 1, 1] >= y.min()))
                for i in candidates:
                    (x0, x1), (y0, y1) = cores[i]
                    occupied[i] = np.any((x >= x0) & (x <= x1) & (y >= y0) & (y <= y1))
                if occupied.all():
                    break
    return occupied


def select_occupied_tile_jobs(source_files, bounds_json, jobs_file, tiles_dir, *, chunk_size):
    """Publish only cores containing source points as active tiles and jobs.

    Preserve grid coordinates and record skipped cores separately, so neighbor
    discovery uses the actual layout. Refuse stale outputs from an older layout
    rather than letting directory-based subsampling pick them up on a resume.
    """
    bounds_json, jobs_file, tiles_dir = map(Path, (bounds_json, jobs_file, tiles_dir))
    data = json.loads(bounds_json.read_text())
    tiles = data['tiles']
    keep = occupied_cores(source_files, tiles, chunk_size=chunk_size)
    if not keep.any():
        raise ValueError('No planned tile core contains source points')
    skipped = [tile for tile, populated in zip(tiles, keep) if not populated]
    for tile in skipped:
        label = f"c{tile['col']:02d}_r{tile['row']:02d}"
        if (tiles_dir / f'{label}.copc.laz').exists() or (tiles_dir / label).exists():
            raise FileExistsError(
                f'Empty-core tile {label} has existing output; use a fresh output directory')
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
