"""Shared output selections keep independent segmentation dimensions in separate collections."""
from collections import Counter
import os
from pathlib import Path
import subprocess
import sys

import laspy
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from instance_tiles import write_instance_tiles


def source(directory):
    directory.mkdir()
    header = laspy.LasHeader(point_format=0, version='1.4')
    header.scales = np.full(3, 0.001)
    header.offsets = np.zeros(3)
    header.add_extra_dims([
        laspy.ExtraBytesParams(name='PredInstance_SAT', type=np.uint32),
        laspy.ExtraBytesParams(name='PredInstance_FM', type=np.uint16),
        laspy.ExtraBytesParams(name='PointToken', type=np.uint16),
    ])
    cloud = laspy.LasData(header)
    cloud.x = np.asarray([1.001, 1.001, 1.002, 1.003, 2.001, 2.001, 2.002, 2.003])
    cloud.y = np.asarray([1.001, 1.001, 1.002, 1.003] * 2)
    cloud.z = np.full(8, 1.001)
    cloud.PredInstance_SAT = [7] * 4 + [70000] * 4
    cloud.PredInstance_FM = [1001, 1002] * 4
    cloud.PointToken = np.arange(8, dtype=np.uint16) + 100
    path = directory / 'source.laz'
    cloud.write(path)
    return path


def invoke(source_dir, output, *options, success=True):
    result = subprocess.run([sys.executable, str(ROOT / 'src/run.py'), '--task', 'tile',
                             '--input-dir', str(source_dir), '--output-dir', str(output),
                             '--workers', '1', *options], capture_output=True, text=True, timeout=180,
                            env={**os.environ, 'MPLBACKEND': 'Agg'})
    assert (result.returncode == 0) == success, result.stdout + result.stderr
    return result


def inventory(cloud, dimension):
    xyz = np.rint(np.c_[cloud.x, cloud.y, cloud.z] * 1000).astype(np.int64)
    return Counter((*map(int, point), int(label), int(token)) for point, label, token in
                   zip(xyz, np.asarray(cloud[dimension]), np.asarray(cloud.PointToken)))


def test_two_dimensions_two_resolutions_produce_four_independent_cli_collections(tmp_path):
    path = source(tmp_path / 'input')
    output = tmp_path / 'output'
    invoke(path.parent, output, '--instance-tiles', 'PredInstance_SAT,PredInstance_FM',
           '--instance-outputs', 'original:copc.laz,0.1:laz', '--instance-tile-dims', 'PointToken')
    before = laspy.read(path)
    expected_paths = {output / 'subsampled_res1', output / 'subsampled_res2',
                      output / 'instance_dim2/subsampled_res1', output / 'instance_dim2/subsampled_res2'}
    actual_paths = {p for p in output.rglob('subsampled_res*') if p.is_dir()}
    assert actual_paths == expected_paths
    for dimension, root, expected_ids, coarse_count in [
        ('PredInstance_SAT', output, {7, 70000}, 2),
        ('PredInstance_FM', output / 'instance_dim2', {1001, 1002}, 4),
    ]:
        expected = inventory(before, dimension)
        for slot, extension, count in [(1, 'copc.laz', 8), (2, 'laz', coarse_count)]:
            files = list((root / f'subsampled_res{slot}').glob('*.' + extension))
            assert len(files) == 1
            cloud = laspy.read(files[0])
            assert len(cloud.points) == count
            assert set(cloud.point_format.extra_dimension_names) == {dimension, 'PointToken'}
            assert set(map(int, cloud[dimension])) == expected_ids
            if slot == 1:
                assert inventory(cloud, dimension) == expected
                assert any(vlr.user_id == 'copc' and vlr.record_id == 1 for vlr in cloud.header.vlrs)
            else:
                assert inventory(cloud, dimension) <= expected


@pytest.mark.parametrize('dimension_args', [[], ['--instance-tiles', 'PredInstance_SAT,PredInstance_SAT']])
def test_cli_rejects_missing_or_duplicate_dimensions_before_creating_outputs(tmp_path, dimension_args):
    path = source(tmp_path / 'input')
    output = tmp_path / 'output'
    result = invoke(path.parent, output, *dimension_args, '--instance-outputs', 'original:laz', success=False)
    message = result.stdout + result.stderr
    assert ('at least one' if not dimension_args else 'duplicate fields') in message
    assert not output.exists()


@pytest.mark.parametrize('mapping', [
    {'SAT': ['sat']},
    {'SAT': ['sat'], 'FM': ['fm'], 'OTHER': ['other']},
    {'SAT': ['sat'], 'FM': []},
    {'SAT': ['sat'], 'FM': ['fm1', 'fm2']},
    {'SAT': ['sat'], 'FM': 'fm'},
    {'SAT': ['sat'], 'FM': None},
])
def test_mapping_shape_is_rejected_before_reading_sources(tmp_path, mapping):
    output = tmp_path / 'output'
    with pytest.raises(ValueError, match='mapping|directories'):
        write_instance_tiles([tmp_path / 'nonexistent.laz'], [], ['SAT', 'FM'], output,
                             resolutions=[0.0], output_directories=mapping)
    assert not output.exists()


def test_duplicate_dimension_is_rejected_before_mapping_or_source_reads(tmp_path):
    with pytest.raises(ValueError, match='duplicate fields'):
        write_instance_tiles([tmp_path / 'nonexistent.laz'], [], ['SAT', 'SAT'], tmp_path / 'output',
                             resolutions=[0.0], output_directories={'SAT': [tmp_path / 'sat']})
    assert not (tmp_path / 'output').exists()


@pytest.mark.parametrize('same_dimension', [False, True])
def test_mapping_destination_aliases_are_rejected_before_reading_sources(tmp_path, same_dimension):
    output = tmp_path / 'output'
    one = output / 'same'
    alias = output / 'parent/../same'
    mapping = {'SAT': [one, alias]} if same_dimension else {'SAT': [one], 'FM': [alias]}
    with pytest.raises(ValueError, match='Duplicate output destination'):
        write_instance_tiles([tmp_path / 'nonexistent.laz'], [], list(mapping), output,
                             resolutions=[0.0, 0.1] if same_dimension else [0.0], output_directories=mapping)
    assert not output.exists()


@pytest.mark.parametrize('dimensions', [
    'PredInstance_SAT,,PredInstance_FM',
    ',PredInstance_SAT',
    'PredInstance_SAT,',
    'PredInstance_SAT, ,PredInstance_FM',
])
def test_cli_rejects_empty_instance_fields_before_creating_outputs(tmp_path, dimensions):
    path = source(tmp_path / 'input')
    output = tmp_path / 'output'
    result = invoke(path.parent, output, '--instance-tiles', dimensions,
                    '--instance-outputs', 'original:laz', success=False)
    assert 'nonempty comma-separated field names' in result.stdout + result.stderr
    assert not output.exists()
