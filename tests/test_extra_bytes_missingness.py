"""Missing-value declarations must describe the stored output values after conversion."""
import hashlib

import laspy
import numpy as np
import pytest
from pyproj import CRS

from main_create_merged_file import create_prod_merged_files
from point_cloud_metadata import extra_bytes_params_from_dimension_info


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(path):
    header = laspy.LasHeader(point_format=6, version='1.4')
    header.scales = [.001] * 3
    header.offsets = [398700, 5646100, 300]
    header.add_crs(CRS.from_epsg(32632))
    header.add_extra_dims([
        laspy.ExtraBytesParams(name='Amplitude', type=np.int16, scales=[.01], offsets=[0], no_data=[0]),
        laspy.ExtraBytesParams(name='Reflectance', type=np.int16, scales=[.01], offsets=[0], no_data=[0]),
        laspy.ExtraBytesParams(name='Deviation', type=np.int16, no_data=[-1]),
        laspy.ExtraBytesParams(name='ScaledSentinel', type=np.int16, scales=[.25], offsets=[10.5], no_data=[-7]),
    ])
    cloud = laspy.LasData(header)
    cloud.X = [1, 101, 201, 301, 401, 501, 1, 101]
    cloud.Y = [1, 101, 201, 301, 401, 501, 1, 101]
    cloud.Z = [1, 101, 201, 301, 401, 501, 1, 101]
    cloud.gps_time = [6, 5, 4, 3, 2, 1, 20, 21]
    cloud.points.array['Amplitude'] = [0, 3, -4, 12, 30, 45, 0, 3]
    cloud.points.array['Reflectance'] = [7, 0, 9, -1, 2, 4, 7, 0]
    cloud.points.array['Deviation'] = [3, 2, -1, 0, 1, 4, 3, 2]
    cloud.points.array['ScaledSentinel'] = [1, 2, 3, -7, 0, 7, 1, 2]
    cloud.write(path)
    return laspy.read(path)


def params(cloud, name):
    return extra_bytes_params_from_dimension_info(cloud.point_format.dimension_by_name(name), header=cloud.header)


def assert_missingness(source, output):
    order = np.argsort(np.asarray(output.x))
    source_order = np.argsort(np.asarray(source.x))
    np.testing.assert_array_equal(np.asarray(output.x)[order], np.asarray(source.x)[source_order])
    for name in source.point_format.extra_dimension_names:
        before, after = params(source, name), params(output, name)
        assert after.no_data is not None, name
        raw = np.asarray(output.points.array[name])[order]
        dtype = raw.dtype
        declaration = np.asarray(after.no_data).astype(dtype)[0]
        source_missing = np.asarray(source.points.array[name])[source_order] == before.no_data[0]
        np.testing.assert_array_equal(raw == declaration, source_missing, err_msg=name)
        assert np.count_nonzero(raw == declaration) == np.count_nonzero(source_missing) > 0, name
        np.testing.assert_allclose(np.asarray(output[name])[order], np.asarray(source[name])[source_order], rtol=0, atol=1e-14)
        expected_physical = np.asarray(source[name])[source_order][source_missing][0]
        output_physical = declaration * (1 if after.scales is None else after.scales[0]) + (0 if after.offsets is None else after.offsets[0])
        assert output_physical == expected_physical, name


def test_full_and_subsampled_products_preserve_actual_missing_points(tmp_path):
    inputs = tmp_path / 'inputs'; inputs.mkdir()
    original = inputs / 'a.laz'; source = fixture(original); before = digest(original)
    output = tmp_path / 'products'
    products = create_prod_merged_files(inputs, output, 'full,10cm', 'laz,copc.laz', .01, .1,
                                       num_spatial_chunks=2, chunk_workers=2)
    assert len(products) == 4
    for path in products:
        expected = source if path.name.startswith('merged.') else laspy.LasData(source.header.copy(), points=source.points[:6])
        assert_missingness(expected, laspy.read(path))
    assert digest(original) == before


def test_metadata_failure_publishes_no_products(tmp_path, monkeypatch):
    import point_cloud_metadata
    inputs = tmp_path / 'inputs'; inputs.mkdir()
    original = inputs / 'a.laz'; fixture(original); before = digest(original)
    output = tmp_path / 'products'
    actual = point_cloud_metadata.preserve_extra_bytes_missingness
    def fail_late(source, target):
        if target.name == 'prod_merged_10cm.copc.laz':
            raise ValueError('sentinel cannot be represented')
        return actual(source, target)
    monkeypatch.setattr(point_cloud_metadata, 'preserve_extra_bytes_missingness', fail_late)
    with pytest.raises((ValueError, RuntimeError), match='sentinel cannot be represented'):
        create_prod_merged_files(inputs, output, '10cm', 'laz,copc.laz', .01, .1)
    assert not list(output.glob('prod_merged*'))
    assert digest(original) == before


def write_field(path, field):
    header = laspy.LasHeader(point_format=6, version='1.4')
    header.add_extra_dim(field)
    cloud = laspy.LasData(header)
    cloud.x = [1.]; cloud.y = [2.]; cloud.z = [3.]
    cloud.write(path)


@pytest.mark.parametrize('source_type,source_options,target_type,target_options,expected', [
    (np.int16, dict(scales=[.25], offsets=[10.5], no_data=[-7]), np.float64, {}, [8.75]),
    (np.int16, dict(no_data=[300]), np.int16, dict(scales=[.5], offsets=[100.]), [400]),
    (np.uint64, dict(no_data=[2**64-1]), np.uint64, {}, [2**64-1]),
    (np.float32, dict(no_data=[np.nan]), np.float64, {}, [np.nan]),
    (np.float32, dict(no_data=[-1]), np.float64, {}, [-1]),
])
def test_descriptor_repair_matches_output_encoding_without_moving_points(tmp_path, source_type, source_options,
                                                                        target_type, target_options, expected):
    from point_cloud_metadata import preserve_extra_bytes_missingness
    source, output = tmp_path/'source.las', tmp_path/'output.las'
    write_field(source, laspy.ExtraBytesParams(name='value', type=source_type, **source_options))
    write_field(output, laspy.ExtraBytesParams(name='value', type=target_type, **target_options))
    before = output.read_bytes(); original_source = digest(source)
    with laspy.open(output) as reader:offset = reader.header.offset_to_point_data
    preserve_extra_bytes_missingness(source, output)
    changed = output.read_bytes()
    assert len(changed) == len(before) and changed[offset:] == before[offset:]
    got = params(laspy.read(output), 'value')
    np.testing.assert_array_equal(got.no_data, expected)
    assert np.dtype(got.type) == np.dtype(target_type)
    preserve_extra_bytes_missingness(source, output)
    assert output.read_bytes() == changed
    assert digest(source) == original_source


@pytest.mark.parametrize('source_type,options,target_type', [
    (np.int16, dict(no_data=[300]), np.uint8),
    (np.int16, dict(scales=[.5], offsets=[1.], no_data=[-7]), np.int16),
    (np.uint64, dict(no_data=[2**53+1]), np.float64),
    (np.float64, dict(no_data=[.1]), np.float32),
    (np.float64, dict(no_data=[np.nan]), np.int32),
])
def test_unrepresentable_sentinel_does_not_modify_output(tmp_path, source_type, options, target_type):
    from point_cloud_metadata import preserve_extra_bytes_missingness
    source, output = tmp_path/'source.las', tmp_path/'output.las'
    write_field(source, laspy.ExtraBytesParams(name='value', type=source_type, **options))
    write_field(output, laspy.ExtraBytesParams(name='value', type=target_type))
    before = output.read_bytes()
    with pytest.raises(ValueError, match='no_data'):
        preserve_extra_bytes_missingness(source, output)
    assert output.read_bytes() == before


def test_descriptor_validation_failure_restores_original_bytes(tmp_path, monkeypatch):
    import point_cloud_metadata
    source, output = tmp_path/'source.las', tmp_path/'output.las'
    write_field(source, laspy.ExtraBytesParams(name='value', type=np.int16, no_data=[-1]))
    write_field(output, laspy.ExtraBytesParams(name='value', type=np.int16))
    before = output.read_bytes()
    def fail(*args):raise ValueError('post-write metadata validation failed')
    monkeypatch.setattr(point_cloud_metadata, 'validate_extra_bytes_missingness', fail)
    with pytest.raises(ValueError, match='post-write'):
        point_cloud_metadata.preserve_extra_bytes_missingness(source, output)
    assert output.read_bytes() == before


def test_legacy_lossy_staging_cache_is_regenerated(tmp_path, monkeypatch):
    import json
    import shutil
    import tile_copc
    from copc_staging import write_copc_stage_manifest_entry, staged_copc_matches_source, MANIFEST_FILENAME
    from main_create_merged_file import prepare_copc_inputs
    inputs, cache, outputs = [tmp_path / n for n in ['inputs', 'legacy', 'output']]
    inputs.mkdir(); cache.mkdir()
    source, old = inputs/'a.laz', cache/'a.copc.laz'
    cloud = fixture(source)
    for vlr in cloud.header.vlrs:
        for descriptor in getattr(vlr, 'extra_bytes_structs', []):descriptor.no_data = None
    cloud.write(old)
    before = digest(old)
    write_copc_stage_manifest_entry(source, old)
    manifest_path = cache/MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest['sources'].values():
        for header in [entry['source']['header'], entry['copc_header']]:
            for dim in header['extra_dimensions']:
                for key in ['scales','offsets','no_data']:dim.pop(key)
    manifest_path.write_text(json.dumps(manifest))
    assert not staged_copc_matches_source(old, source)
    conversions = []
    def convert(origin, target, **options):
        assert options['preserve_extra_dims']
        conversions.append((origin, target)); shutil.copyfile(origin, target); return True
    monkeypatch.setattr(tile_copc, 'convert_laz_to_copc', convert)
    result = prepare_copc_inputs(inputs, outputs, staged_copc_dir=cache)
    assert len(conversions) == 1 and conversions[0][0] == source
    assert result[0] != old and digest(old) == before
    assert_missingness(laspy.read(source), laspy.read(result[0]))
    assert staged_copc_matches_source(result[0], source)
    assert prepare_copc_inputs(inputs, outputs) == result
    assert len(conversions) == 1


@pytest.mark.parametrize('sentinel', [2**53, 2**53+1])
def test_scaled_uint64_sentinel_cannot_alias_a_valid_point(tmp_path, sentinel):
    from point_cloud_metadata import preserve_extra_bytes_missingness
    source, output = tmp_path/'source.las', tmp_path/'output.las'
    write_field(source, laspy.ExtraBytesParams(name='value', type=np.uint64,
                                             scales=[1.], offsets=[0.], no_data=[sentinel]))
    cloud = laspy.read(source)
    cloud.points = laspy.ScaleAwarePointRecord.zeros(2, header=cloud.header)
    cloud.points.array['value'] = [2**53, 2**53+1]
    cloud.write(source)
    write_field(output, laspy.ExtraBytesParams(name='value', type=np.float64))
    converted = laspy.read(output)
    converted.points = laspy.ScaleAwarePointRecord.zeros(2, header=converted.header)
    converted.value = np.asarray(cloud.value)
    converted.write(output)
    assert np.count_nonzero(cloud.points.array['value'] == sentinel) == 1
    assert len(set(np.asarray(converted.value))) == 1
    before = output.read_bytes()
    with pytest.raises(ValueError, match='no_data.*indistinguishable'):
        preserve_extra_bytes_missingness(source, output)
    assert output.read_bytes() == before


@pytest.mark.parametrize('source_type,options,target_type', [
    (np.int16, dict(scales=[1.], offsets=[2.**53], no_data=[0]), np.float64),
    (np.int32, dict(no_data=[2**24]), np.float32),
    (np.float64, dict(no_data=[0.]), np.float32),
    (np.float64, dict(no_data=[np.inf]), np.float32),
])
def test_representable_sentinel_rejects_neighbour_collisions(tmp_path, source_type, options, target_type):
    from point_cloud_metadata import preserve_extra_bytes_missingness
    source, output = tmp_path/'source.las', tmp_path/'output.las'
    write_field(source, laspy.ExtraBytesParams(name='value', type=source_type, **options))
    write_field(output, laspy.ExtraBytesParams(name='value', type=target_type))
    before = output.read_bytes()
    with pytest.raises(ValueError, match='no_data.*indistinguishable'):
        preserve_extra_bytes_missingness(source, output)
    assert output.read_bytes() == before


@pytest.mark.parametrize('existing', [False, True])
def test_failed_scalar_conversion_preserves_destination(tmp_path, monkeypatch, existing):
    import tile_copc
    source, output = tmp_path/'source.las', tmp_path/'output.copc.laz'
    write_field(source, laspy.ExtraBytesParams(name='value', type=np.int16, no_data=[300]))
    if existing:output.write_bytes(b'previous product')
    def writer(origin, candidate, **kwargs):
        assert candidate != output
        write_field(candidate, laspy.ExtraBytesParams(name='value', type=np.uint8))
        return True
    monkeypatch.setattr(tile_copc, '_convert_laz_to_copc_impl', writer)
    before = digest(source)
    assert not tile_copc.convert_laz_to_copc(source, output, preserve_extra_dims=True)
    assert output.read_bytes() == b'previous product' if existing else not output.exists()
    assert digest(source) == before
    assert not list(tmp_path.glob('.smarttile-vector-extra-*'))


@pytest.mark.parametrize('changed', ['no_data', 'scales', 'offsets'])
def test_incompatible_collection_metadata_fails_before_staging(tmp_path, monkeypatch, changed):
    import main_create_merged_file
    inputs = tmp_path/'inputs'; inputs.mkdir()
    first, second = inputs/'a.las', inputs/'b.las'
    options = dict(scales=[.25], offsets=[10.5], no_data=[-7])
    write_field(first, laspy.ExtraBytesParams(name='value', type=np.int16, **options))
    options[changed] = dict(no_data=[-8], scales=[.5], offsets=[11.5])[changed]
    write_field(second, laspy.ExtraBytesParams(name='value', type=np.int16, **options))
    before = [digest(first), digest(second)]
    def unexpected(*args, **kwargs):pytest.fail('incompatible originals reached conversion')
    monkeypatch.setattr(main_create_merged_file, 'prepare_copc_inputs', unexpected)
    output = tmp_path/'products'
    with pytest.raises(ValueError, match='ExtraBytes encoding or missingness differs'):
        create_prod_merged_files(inputs, output, '10cm', 'laz,copc.laz', .01, .1)
    assert not output.exists()
    assert [digest(first), digest(second)] == before


def test_integer_narrowing_cannot_wrap_a_valid_point_onto_sentinel(tmp_path):
    from point_cloud_metadata import preserve_extra_bytes_missingness
    source, output = tmp_path/'source.las', tmp_path/'output.las'
    write_field(source, laspy.ExtraBytesParams(name='value', type=np.int16, no_data=[1]))
    cloud = laspy.read(source)
    cloud.points = laspy.ScaleAwarePointRecord.zeros(2, header=cloud.header)
    cloud.points.array['value'] = [1, 257]
    cloud.write(source)
    write_field(output, laspy.ExtraBytesParams(name='value', type=np.uint8))
    converted = laspy.read(output)
    converted.points = laspy.ScaleAwarePointRecord.zeros(2, header=converted.header)
    converted.value = cloud.points.array['value'].astype(np.uint8)
    converted.write(output)
    np.testing.assert_array_equal(converted.value, [1, 1])
    assert np.count_nonzero(cloud.points.array['value'] == 1) == 1
    before = output.read_bytes()
    with pytest.raises(ValueError, match='no_data.*narrowing integer'):
        preserve_extra_bytes_missingness(source, output)
    assert output.read_bytes() == before


def test_colocated_copc_cache_preserves_source_selection(tmp_path):
    from copc_staging import write_copc_stage_manifest_entry
    from tile_copc import convert_laz_to_copc
    inputs = tmp_path/'inputs'; inputs.mkdir()
    original, cache = inputs/'a.laz', inputs/'a.copc.laz'
    source = fixture(original)
    assert convert_laz_to_copc(original, cache, preserve_extra_dims=True)
    write_copc_stage_manifest_entry(original, cache)
    assert params(source, 'Amplitude').type != params(laspy.read(cache), 'Amplitude').type
    before = [digest(original), digest(cache)]
    products = create_prod_merged_files(inputs, tmp_path/'products', 'full,10cm', 'laz,copc.laz', .01, .1)
    assert len(products) == 4
    for product in products:
        expected = source if product.name.startswith('merged.') else laspy.LasData(source.header.copy(), points=source.points[:6])
        assert_missingness(expected, laspy.read(product))
    assert [digest(original), digest(cache)] == before


def test_distinct_copc_only_source_participates_in_subsampled_product(tmp_path):
    from tile_copc import convert_laz_to_copc
    inputs = tmp_path/'inputs'; inputs.mkdir()
    raw, other, cache = inputs/'a.laz', tmp_path/'b.laz', inputs/'b.copc.laz'
    for path, x, value in [(raw, 1., 0), (other, 2., 2)]:
        write_field(path, laspy.ExtraBytesParams(name='value', type=np.uint16, no_data=[0]))
        cloud = laspy.read(path); cloud.x = [x]; cloud.value = [value]; cloud.write(path)
    assert convert_laz_to_copc(other, cache, preserve_extra_dims=True)
    products = create_prod_merged_files(inputs, tmp_path/'products', '10cm', 'laz', .01, .1)
    output = laspy.read(products[0]); order = np.argsort(np.asarray(output.x))
    np.testing.assert_array_equal(np.asarray(output.x)[order], [1., 2.])
    np.testing.assert_array_equal(np.asarray(output.value)[order], [0, 2])
    np.testing.assert_array_equal(params(output, 'value').no_data, [0])


def test_distinct_copc_only_source_retains_strict_metadata_preflight(tmp_path, monkeypatch):
    import main_create_merged_file
    inputs = tmp_path/'inputs'; inputs.mkdir()
    write_field(inputs/'a.las', laspy.ExtraBytesParams(name='value', type=np.uint16, no_data=[0]))
    write_field(inputs/'b.copc.laz', laspy.ExtraBytesParams(name='value', type=np.uint16, no_data=[1]))
    def unexpected(*args, **kwargs):pytest.fail('incompatible COPC-only source reached conversion')
    monkeypatch.setattr(main_create_merged_file, 'prepare_copc_inputs', unexpected)
    with pytest.raises(ValueError, match='ExtraBytes encoding or missingness differs'):
        create_prod_merged_files(inputs, tmp_path/'products', '10cm', 'laz', .01, .1)
    assert not (tmp_path/'products').exists()
