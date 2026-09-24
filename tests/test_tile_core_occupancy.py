import json
import sys
import types
from pathlib import Path
from unittest import mock

import laspy
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.modules.setdefault('plot_tiles_and_copc', types.SimpleNamespace(plot_extents=lambda *a: None))

from tile_core_occupancy import occupied_cores, select_occupied_tile_jobs
from tile_bounds_graph import build_neighbor_graph_from_bounds_json
import main_tile


def cloud(path, xy):
    las = laspy.LasData(laspy.LasHeader(point_format=0, version='1.2'))
    las.x, las.y = np.asarray(xy, dtype=float).T
    las.z = np.zeros(len(xy))
    las.classification = np.full(len(xy), 2)  # Ground also qualifies.
    las.write(path)
    return path


def layout(tmp_path):
    tiles = [dict(col=c, row=r, core=[[c*10, c*10+10], [r*10, r*10+10]],
                  bounds=[[c*10-2, c*10+12], [r*10-2, r*10+12]])
             for c in range(2) for r in range(2)]
    path = tmp_path / 'tile_bounds_tindex.json'
    path.write_text(json.dumps(dict(tile_length=10, tile_buffer=2, tiles=tiles)))
    return path, tiles


def test_buffer_only_tiles_are_absent_from_jobs_and_neighbors(tmp_path):
    bounds, tiles = layout(tmp_path)
    sources = [cloud(tmp_path/'first.las', [(9, 9)]),
               cloud(tmp_path/'later.las', [(15, 15)])]
    jobs = tmp_path/'jobs.txt'
    assert select_occupied_tile_jobs(sources, bounds, jobs, tmp_path/'tiles', chunk_size=1) == 2
    data = json.loads(bounds.read_text())
    assert [(t['col'], t['row']) for t in data['tiles']] == [(0, 0), (1, 1)]
    assert data['tiles'] == [tiles[0], tiles[3]]  # Full buffers are unchanged.
    assert len(data['core_occupancy']['skipped_tiles']) == 2
    assert [line.split('|')[0] for line in jobs.read_text().splitlines()] == ['c00_r00', 'c01_r01']
    _, _, neighbors = build_neighbor_graph_from_bounds_json(bounds)
    assert all(v is None for n in neighbors for v in n.values())


def test_later_chunks_populate_cores_and_boundary_points_count(tmp_path):
    _, tiles = layout(tmp_path)
    source = cloud(tmp_path/'points.las', [(1, 1), (10, 10)])
    assert occupied_cores([source], tiles, chunk_size=1).tolist() == [True]*4


def test_file_extent_is_not_evidence_of_core_occupancy(tmp_path):
    _, tiles = layout(tmp_path)
    # Bounding box intersects every core, but only two contain actual points.
    source = cloud(tmp_path/'points.las', [(1, 1), (19, 19)])
    assert occupied_cores([source], tiles, chunk_size=1).tolist() == [True, False, False, True]


def test_no_occupied_core_fails_without_rewriting_metadata(tmp_path):
    bounds, _ = layout(tmp_path)
    before = bounds.read_bytes()
    source = cloud(tmp_path/'points.las', [(-1, -1)])
    with pytest.raises(ValueError, match='No planned tile core'):
        select_occupied_tile_jobs([source], bounds, tmp_path/'jobs', tmp_path/'tiles', chunk_size=1)
    assert bounds.read_bytes() == before
    assert not (tmp_path/'jobs').exists()


def test_stale_buffer_tile_requires_fresh_destination(tmp_path):
    bounds, _ = layout(tmp_path)
    before = bounds.read_bytes()
    source = cloud(tmp_path/'points.las', [(9, 9)])
    out = tmp_path/'tiles'
    out.mkdir()
    stale = out/'c00_r01.copc.laz'
    stale.write_bytes(b'existing result')
    with pytest.raises(FileExistsError, match='fresh output directory'):
        select_occupied_tile_jobs([source], bounds, tmp_path/'jobs', out, chunk_size=1)
    assert bounds.read_bytes() == before
    assert stale.read_bytes() == b'existing result'


def test_tile_pipeline_selects_cores_before_crop_or_overview(tmp_path):
    source_dir = tmp_path/'input'
    source_dir.mkdir()
    source = cloud(source_dir/'points.las', [(9, 9), (15, 15)])
    bounds, _ = layout(tmp_path)
    jobs = tmp_path/'jobs.txt'
    observed = []

    def verify_layout(*args, **kwargs):
        assert len(json.loads(bounds.read_text())['tiles']) == 2
        assert len(jobs.read_text().splitlines()) == 2
        observed.append(True)
        return []

    with mock.patch.object(main_tile, 'build_tindex', return_value=tmp_path/'index.gpkg'), \
         mock.patch.object(main_tile, 'calculate_tile_bounds', return_value=(jobs, bounds, {})), \
         mock.patch.object(main_tile.plot_tiles_and_copc, 'plot_extents', side_effect=verify_layout), \
         mock.patch.object(main_tile, 'create_tiles', side_effect=verify_layout):
        main_tile.run_tiling_pipeline(source_dir, tmp_path/'output', tile_length=10, tiling_threshold=None)
    assert len(observed) == 2


def test_occupied_tiles_still_receive_buffer_points(tmp_path):
    _, tiles = layout(tmp_path)
    source = cloud(tmp_path/'points.las', [(9, 9), (11, 9)])
    out = tmp_path/'tiles'
    out.mkdir()
    result = main_tile._distribute_source_file((0, str(source), [('c00_r00', (-2, -2, 12, 12))], out, 1, 1))
    assert result == [('c00_r00', 2)]
    parts = list((out/'c00_r00').glob('part_*.las'))
    assert sum(laspy.read(p).header.point_count for p in parts) == 2
