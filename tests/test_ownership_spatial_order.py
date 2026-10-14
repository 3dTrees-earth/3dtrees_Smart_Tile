from collections import Counter
from unittest.mock import patch

import laspy
import numpy as np
import pytest

import dense_instance_ownership as ownership
from bounded_point_index import PointIndex
from merge_stages import describe_model, index_file
from spatial_query_cache import SpatialQueryCache
from test_merge_stages import write_cloud


def run_ownership(tmp_path, files, *, cache_bytes=64 * 1024**2):
    model = describe_model(files[0].parent, 'PredInstance')
    dims = {name: p.type for name, p in model.dimensions.items()}
    regions = [{'core': [[0, 20], [-1, 1]]} for _ in files]
    overlap = (np.array([-1., -1.]), np.array([21., 1.]))
    overlaps = [{j: overlap for j in range(i)} for i in range(len(files))]
    mapping = {(i, i + 1): i + 1 for i in range(len(files))}
    report = {}
    with PointIndex(tmp_path/'source.db', dims, query_cache=SpatialQueryCache(cache_bytes)) as source:
        for tile, file in enumerate(files):
            index_file(source, file, tile, np.zeros(3), model)
        with PointIndex(tmp_path/'output.db', dims) as output:
            with patch.object(source, 'query_tree', wraps=source.query_tree) as query:
                result = ownership.assign_shared_points(model, files, source, regions, mapping,
                    tmp_path/'out', output, np.zeros(3), report, overlaps=overlaps)
            calls = query.call_count
    return result, report, calls


def test_skip_queries_without_competing_tree_in_xyz(tmp_path):
    folder = tmp_path/'input'; folder.mkdir()
    files = [write_cloud(folder/'0.las', [.5, .5], [1, 0], zs=[100, 0]),
             write_cloud(folder/'1.las', [.5], [2], zs=[0])]
    result, report, calls = run_ownership(tmp_path, files)
    assert [len(laspy.read(p).points) for p in result] == [2, 1]
    assert calls == 0, 'A tree 100 m above cannot contest this point; skip the nearest query'


@pytest.mark.parametrize('budget', [0, 64 * 1024**2])
def test_process_spatial_block_together_across_file_chunks(tmp_path, budget):
    folder = tmp_path/'input'; folder.mkdir()
    files = [write_cloud(folder/'0.las', [.5, 8.5], [1, 1]),
             write_cloud(folder/'1.las', [.5, 8.5, .5, 8.5], [2] * 4)]
    visited = []
    original = ownership.ownership_candidates
    def record(index, pts, *args, **kwargs):
        visited.append(int(np.floor(pts[0, 0] / 4)))
        yield from original(index, pts, *args, **kwargs)
    with patch.object(ownership, 'MAX_BATCH_POINTS', 1), patch.object(ownership, 'ownership_candidates', record):
        result, _, _ = run_ownership(tmp_path, files, cache_bytes=budget)
    assert [len(laspy.read(p).points) for p in result] == [2, 0]
    assert visited == sorted(visited), 'Return to the same spatial region only after all its chunks are resolved'


def test_occupancy_budget_fallback_and_boundary_halo(tmp_path):
    from ownership_spatial_blocks import TreeOccupancy
    with PointIndex(tmp_path/'occupancy.db', {'instance': np.uint32}) as index:
        xyz = np.array([[2.0001, -2.0001, 2.0001], [100, 100, 100]])
        index.add(0, xyz, {'instance': np.array([1, 0], np.uint32)}, np.arange(2))
        index.flush()
        occupancy = TreeOccupancy(index, 'instance')
        assert occupancy.may_contain(0, np.array([[1.9999, -1.9999, 1.9999]]), .01)
        assert not occupancy.may_contain(0, np.array([[100., 100., 100.]]), .01)
        with patch.object(TreeOccupancy, 'MAX_VOXELS', 0):
            occupancy = TreeOccupancy(index, 'instance')
        assert occupancy.may_contain(0, np.array([[100., 100., 100.]]), .01)
        assert not occupancy.report()['enabled']


def test_block_halo_does_not_expand_individual_query_allowance(tmp_path):
    folder = tmp_path/'input'; folder.mkdir()
    files = []
    for tile in (0, 1):
        header = laspy.LasHeader(point_format=3, version='1.2')
        header.scales = [1e-12, .001, .001]
        header.offsets = [.01 + 5e-12 if tile == 0 else 0, 0, 0]
        cloud = laspy.LasData(header)
        cloud.X = [0, 0]; cloud.Y = [0, 0]; cloud.Z = [0, 16384000]
        cloud.add_extra_dim(laspy.ExtraBytesParams(name='PredInstance', type=np.uint32))
        cloud.PredInstance = [tile + 1] * 2
        file = folder/f'{tile}.las'; cloud.write(file); files.append(file)
    with patch.object(ownership, 'MAX_BATCH_POINTS', 1):
        result, _, _ = run_ownership(tmp_path, files)
    cloud = laspy.read(result[1])
    assert len(cloud.points) == 1
    assert cloud.Z[0] == 0, 'The high-Z group halo must not enlarge the low-Z query tolerance'
