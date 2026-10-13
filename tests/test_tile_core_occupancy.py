import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from tile_core_occupancy import publish_occupied_tiles
from tile_bounds_graph import build_neighbor_graph_from_bounds_json


def layout(tmp_path):
    tiles = [dict(col=c, row=r, core=[[c*10, c*10+10], [r*10, r*10+10]],
                  bounds=[[c*10-2, c*10+12], [r*10-2, r*10+12]])
             for c in range(2) for r in range(2)]
    path = tmp_path / 'tile_bounds_tindex.json'
    path.write_text(json.dumps(dict(tile_length=10, tile_buffer=2, tiles=tiles)))
    return path, tiles


def test_buffer_only_tiles_are_absent_from_jobs_and_neighbors(tmp_path):
    bounds, tiles = layout(tmp_path)
    jobs = tmp_path/'jobs.txt'
    assert publish_occupied_tiles(bounds, jobs, [True, False, False, True]) == 2
    data = json.loads(bounds.read_text())
    assert [(t['col'], t['row']) for t in data['tiles']] == [(0, 0), (1, 1)]
    assert data['tiles'] == [tiles[0], tiles[3]]  # Full buffers are unchanged.
    assert len(data['core_occupancy']['skipped_tiles']) == 2
    assert [line.split('|')[0] for line in jobs.read_text().splitlines()] == ['c00_r00', 'c01_r01']
    _, _, neighbors = build_neighbor_graph_from_bounds_json(bounds)
    assert all(v is None for n in neighbors for v in n.values())


def test_no_occupied_core_fails_without_rewriting_metadata(tmp_path):
    bounds, _ = layout(tmp_path)
    before = bounds.read_bytes()
    with pytest.raises(ValueError, match='No planned tile core'):
        publish_occupied_tiles(bounds, tmp_path/'jobs', [False] * 4)
    assert bounds.read_bytes() == before
    assert not (tmp_path/'jobs').exists()


def test_occupancy_must_cover_every_planned_tile(tmp_path):
    bounds, _ = layout(tmp_path)
    with pytest.raises(ValueError, match='every planned tile'):
        publish_occupied_tiles(bounds, tmp_path/'jobs', [True])
