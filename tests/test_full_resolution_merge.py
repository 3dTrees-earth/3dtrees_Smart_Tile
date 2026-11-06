"""Full-resolution prod-merged product with OriginalPointIndex (ForestStructure order key)."""
import sys
from pathlib import Path

import laspy
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from full_resolution_merge import ORIGINAL_POINT_INDEX, create_full_resolution_product, merged_header  # noqa: E402
from main_create_merged_file import create_prod_merged_files, parse_merged_resolutions  # noqa: E402
from merge_stages import describe_model  # noqa: E402
from point_cloud_metadata import ensure_point_capacity  # noqa: E402


def write_cloud(path, xyz, *, scale=0.001, extras=("species_id", "PredInstance_SAT"), index=False):
    header = laspy.LasHeader(point_format=3, version="1.4")
    header.scales, header.offsets = np.full(3, scale), np.array([500000.0, 5000000.0, 0.0])
    for name in extras:
        header.add_extra_dim(laspy.ExtraBytesParams(name=name, type=np.uint32))
    if index:
        header.add_extra_dim(laspy.ExtraBytesParams(name=ORIGINAL_POINT_INDEX, type=np.uint64))
    las = laspy.LasData(header)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    las.intensity = np.arange(len(xyz), dtype=np.uint16) + 7
    for n, name in enumerate(extras):
        las[name] = np.arange(len(xyz), dtype=np.uint32) * (n + 2)
    las.write(path)
    return laspy.read(path)


def inputs(tmp_path, scales=(0.001, 0.001)):
    rng = np.random.default_rng(4)
    folder = tmp_path / "in"
    folder.mkdir()
    clouds = []
    for name, count, scale in zip(("b.las", "a.las"), (1500, 2500), scales):
        xyz = np.c_[rng.uniform(500000, 500050, count), rng.uniform(5000000, 5000050, count), rng.uniform(0, 30, count)]
        clouds.append((name, write_cloud(folder / name, xyz, scale=scale)))
    # Record order is the sorted file names (a.las, then b.las).
    return folder, [cloud for _, cloud in sorted(clouds, key=lambda c: c[0])]


@pytest.mark.parametrize("output_format", ["laz", "copc.laz"])
def test_every_point_in_record_order_with_index_first(tmp_path, output_format):
    folder, clouds = inputs(tmp_path)
    output = tmp_path / f"full.{output_format}"
    create_full_resolution_product(sorted(folder.glob("*.las")), output, output_format, original_point_index=True)
    merged = laspy.read(output)
    names = [d.name for d in merged.point_format.extra_dimensions]
    assert names[:2] == [ORIGINAL_POINT_INDEX, "PredInstance_SAT"]
    total = sum(len(c.points) for c in clouds)
    assert len(merged.points) == total
    order = np.argsort(merged[ORIGINAL_POINT_INDEX])
    assert np.array_equal(merged[ORIGINAL_POINT_INDEX][order], np.arange(total))
    if output_format == "laz":  # a plain LAZ keeps record order; COPC reorders points
        assert np.array_equal(order, np.arange(total))
    for field in ("X", "Y", "Z", "intensity", "species_id", "PredInstance_SAT"):
        expected = np.concatenate([np.asarray(c[field]) for c in clouds])
        assert np.array_equal(np.asarray(merged[field])[order], expected), field
    assert merged.header.parse_crs() == clouds[0].header.parse_crs()


def test_finest_scale_keeps_every_coordinate(tmp_path):
    folder, clouds = inputs(tmp_path, scales=(0.01, 0.0001))
    output = tmp_path / "full.laz"
    create_full_resolution_product(sorted(folder.glob("*.las")), output, "laz", original_point_index=True)
    merged = laspy.read(output)
    assert np.allclose(merged.header.scales, 0.0001)
    expected = np.concatenate([np.c_[c.x, c.y, c.z] for c in clouds])
    assert np.abs(np.c_[merged.x, merged.y, merged.z] - expected).max() < 1e-9


def test_index_can_be_disabled_and_is_never_duplicated(tmp_path):
    folder, _ = inputs(tmp_path)
    files = sorted(folder.glob("*.las"))
    output = tmp_path / "plain.laz"
    create_full_resolution_product(files, output, "laz", original_point_index=False)
    assert ORIGINAL_POINT_INDEX not in laspy.read(output).point_format.dimension_names
    indexed = tmp_path / "indexed"
    indexed.mkdir()
    write_cloud(indexed / "x.las", np.array([[500000.0, 5000000.0, 1.0]]), index=True)
    with pytest.raises(ValueError, match="already contain"):
        create_full_resolution_product([indexed / "x.las"], tmp_path / "y.laz", "laz", original_point_index=True)


def test_disabling_index_generation_preserves_existing_indices(tmp_path):
    source = tmp_path / "indexed.las"
    cloud = write_cloud(source, np.array([[500000.0, 5000000.0, z] for z in range(3)]), index=True)
    cloud.OriginalPointIndex = [5, 6, 2 ** 40]
    cloud.write(source)
    output = tmp_path / "full.laz"
    create_full_resolution_product([source], output, "laz", original_point_index=False)
    assert laspy.read(output).OriginalPointIndex.tolist() == [5, 6, 2 ** 40]


def metadata_cloud(path, **overrides):
    header = laspy.LasHeader(point_format=3, version="1.4")
    options = dict(name="confidence", type=np.uint16, no_data=[65535], scales=[0.1], offsets=[2.0])
    options.update(overrides)
    header.add_extra_dim(laspy.ExtraBytesParams(**options))
    header.add_extra_dim(laspy.ExtraBytesParams(name="normal", type="3i2", no_data=[-32768] * 3,
                                             scales=[0.1, 0.2, 0.3], offsets=[1.0, 2.0, 3.0]))
    cloud = laspy.LasData(header)
    cloud.x, cloud.y, cloud.z = [1, 2], [3, 4], [5, 6]
    cloud.points.array["confidence"] = [12, 65535]
    cloud.points.array["normal"] = [[3, 4, 5], [-32768, -32768, -32768]]
    cloud.write(path)
    return laspy.read(path)


def test_full_merge_preserves_extra_byte_metadata_and_raw_values(tmp_path):
    source = tmp_path / "metadata.las"
    original = metadata_cloud(source)
    output = tmp_path / "full.laz"
    create_full_resolution_product([source], output, "laz", original_point_index=True)
    result = laspy.read(output)
    descriptors = {entry.format_name(): entry for vlr in result.header.vlrs
                   for entry in getattr(vlr, "extra_bytes_structs", ())}
    assert descriptors["confidence"].no_data.tolist() == [65535]
    assert list(descriptors["confidence"].scale) == [0.1]
    assert list(descriptors["confidence"].offset) == [2.0]
    assert descriptors["normal"].no_data.tolist() == [-32768] * 3
    assert list(descriptors["normal"].scale) == [0.1, 0.2, 0.3]
    assert list(descriptors["normal"].offset) == [1.0, 2.0, 3.0]
    for name in ("confidence", "normal"):
        assert np.array_equal(result.points.array[name], original.points.array[name])


@pytest.mark.parametrize("change", [{"no_data": [65534]}, {"scales": [0.2]}, {"offsets": [3.0]}])
def test_full_merge_rejects_incompatible_extra_byte_definitions(tmp_path, change):
    first, second = tmp_path / "first.las", tmp_path / "second.las"
    metadata_cloud(first)
    metadata_cloud(second, **change)
    output = tmp_path / "full.laz"
    with pytest.raises(ValueError, match="schema"):
        create_full_resolution_product([first, second], output, "laz", original_point_index=True)
    assert not output.exists()


def test_mismatched_schema_is_rejected(tmp_path):
    folder, _ = inputs(tmp_path)
    write_cloud(folder / "c.las", np.array([[500000.0, 5000000.0, 1.0]]), extras=("species_id",))
    with pytest.raises(ValueError, match="one schema"):
        create_full_resolution_product(sorted(folder.glob("*.las")), tmp_path / "z.laz", "laz",
                                       original_point_index=True)


def test_full_selector_and_task_entry(tmp_path):
    assert parse_merged_resolutions("full,10cm,original", 0.01, 0.1) == [("full", 0.0), ("10cm", 0.1)]
    folder, clouds = inputs(tmp_path)
    out = tmp_path / "products"
    created = create_prod_merged_files(folder, out, "full", "laz", 0.01, 0.1, original_point_index=True)
    assert [p.name for p in created] == ["merged.laz"]
    assert len(laspy.read(created[0]).points) == sum(len(c.points) for c in clouds)
    # Only subsampled products stage COPCs.
    assert not (out / "original_with_predictions_copc").exists()


def test_index_is_never_a_prediction_dimension(tmp_path):
    folder = tmp_path / "pred"
    folder.mkdir()
    write_cloud(folder / "c00_r00.las", np.array([[500000.0, 5000000.0, 1.0]]), index=True)
    model = describe_model(folder, "PredInstance")
    assert ORIGINAL_POINT_INDEX not in model.dimensions and "PredInstance_SAT" in model.dimensions


@pytest.mark.parametrize("bad_index", [2 ** 32, -1])
def test_ply_index_overflow_rolls_back_all_requested_products(tmp_path, bad_index):
    folder = tmp_path / "input"
    folder.mkdir()
    source = folder / "indexed.las"
    original = write_cloud(source, np.array([[500000.0, 5000000.0, 1.0]]))
    original.add_extra_dim(laspy.ExtraBytesParams(name=ORIGINAL_POINT_INDEX, type=np.int64))
    original[ORIGINAL_POINT_INDEX] = [bad_index]
    original.write(source)
    before = source.read_bytes()
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="OriginalPointIndex.*uint32 range"):
        create_prod_merged_files(folder, out, "full", "laz,ply", 0.01, 0.1,
                                 original_point_index=False)
    assert source.read_bytes() == before
    assert not list(out.iterdir())


@pytest.mark.parametrize("dtype", [np.uint64, np.int64])
def test_ply_rejects_other_64_bit_integer_attributes_without_truncation(tmp_path, dtype):
    source = tmp_path / "input.las"
    original = write_cloud(source, np.array([[500000.0, 5000000.0, 1.0]]))
    original.add_extra_dim(laspy.ExtraBytesParams(name="wide_value", type=dtype))
    original.wide_value = [2 ** 40]
    original.write(source)
    output = tmp_path / "full.ply"
    with pytest.raises(ValueError, match="64-bit integer attribute wide_value"):
        create_full_resolution_product([source], output, "ply", original_point_index=True)
    assert not output.exists()


@pytest.mark.parametrize("dtype,scales,offsets", [
    (np.float64, None, None), (np.uint64, [0.5], [0]), (np.uint64, [1], [2]),
])
def test_ply_rejects_noninteger_index_interpretations(tmp_path, dtype, scales, offsets):
    source = tmp_path / "input.las"
    original = write_cloud(source, np.array([[500000.0, 5000000.0, 1.0]]))
    original.add_extra_dim(laspy.ExtraBytesParams(name=ORIGINAL_POINT_INDEX, type=dtype,
                                               scales=scales, offsets=offsets))
    original.write(source)
    output = tmp_path / "full.ply"
    with pytest.raises(ValueError, match="unscaled scalar integer"):
        create_full_resolution_product([source], output, "ply", original_point_index=False)
    assert not output.exists()


def test_ply_conversion_failure_does_not_publish_partial_file(tmp_path, monkeypatch):
    import full_resolution_merge
    from types import SimpleNamespace

    source = tmp_path / "input.las"
    write_cloud(source, np.array([[500000.0, 5000000.0, 1.0]]))
    output = tmp_path / "full.ply"

    def fail_after_write(command, **kwargs):
        Path(command[3]).write_bytes(b"partial PLY")
        return SimpleNamespace(returncode=1, stderr="injected writer failure")

    monkeypatch.setattr(full_resolution_merge.subprocess, "run", fail_after_write)
    with pytest.raises(RuntimeError, match="injected writer failure"):
        create_full_resolution_product([source], output, "ply", original_point_index=True)
    assert not output.exists()
    assert not list(tmp_path.glob(".full-merge-*"))


@pytest.mark.parametrize("generate_index", [True, False])
def test_ply_point_count_overflow_rejected_from_headers_before_reading_points(tmp_path, generate_index):
    source = tmp_path / "oversize.las"
    write_cloud(source, np.array([[500000.0, 5000000.0, 1.0]]))
    # A header-only huge-count fixture ensures preflight never reads billions of records.
    with source.open("r+b") as stream:
        header = laspy.LasHeader.read_from(stream)
        header.point_count = 2 ** 32
        stream.seek(0)
        header.write_to(stream)
    output = tmp_path / "full.ply"
    with pytest.raises(ValueError, match="PLY point count.*writer limit"):
        create_full_resolution_product([source], output, "ply", original_point_index=generate_index)
    assert not output.exists()
    assert not list(tmp_path.glob(".full-merge-*"))


def format_cloud(path, point_format, *, extras=("species_id",), version="1.4", n=3, seed=1):
    header = laspy.LasHeader(point_format=point_format, version=version)
    header.scales, header.offsets = np.full(3, 0.001), np.array([500000.0, 5000000.0, 0.0])
    for name in extras:
        header.add_extra_dim(laspy.ExtraBytesParams(name=name, type=np.uint32))
    las = laspy.LasData(header)
    rng = np.random.default_rng(seed)
    las.x, las.y, las.z = 500000 + rng.uniform(0, 9, n), 5000000 + rng.uniform(0, 9, n), rng.uniform(0, 9, n)
    las.intensity = np.arange(n, dtype=np.uint16) + 7
    las.classification = np.arange(n, dtype=np.uint8) + 2
    las.return_number, las.number_of_returns = np.full(n, 1), np.full(n, 2)
    las.user_data, las.point_source_id = np.full(n, 3), np.full(n, 11)
    if "gps_time" in las.point_format.dimension_names:
        las.gps_time = np.arange(n) + 0.5
    if "red" in las.point_format.dimension_names:
        las.red, las.green, las.blue = np.full(n, 100), np.full(n, 200), np.full(n, 300)
    if "nir" in las.point_format.dimension_names:
        las.nir = np.full(n, 400)
    if point_format < 6:
        las.scan_angle_rank = np.array([-15, 0, 30][:n] + [0] * (n - 3), dtype=np.int8)
    else:
        las.scan_angle = np.array([-2500, 0, 5000][:n] + [0] * (n - 3), dtype=np.int16)
    for i, name in enumerate(extras):
        las[name] = np.arange(n, dtype=np.uint32) * (i + 2)
    las.write(path)
    return laspy.read(path)


@pytest.mark.parametrize("output_format", ["laz", "copc.laz"])
@pytest.mark.parametrize("formats, expected", [((1, 3), 7), ((3, 7), 7), ((1, 6), 6), ((6, 8), 8), ((0, 1), 6)])
def test_mixed_point_formats_are_promoted(tmp_path, formats, expected, output_format):
    first, second = (tmp_path / "a.las", tmp_path / "b.las")
    a, b = format_cloud(first, formats[0]), format_cloud(second, formats[1], seed=2)
    output = tmp_path / f"merged.{output_format}"
    create_full_resolution_product([first, second], output, output_format, original_point_index=True)
    merged = laspy.read(output)
    assert merged.header.point_format.id == expected and str(merged.header.version) == "1.4"
    assert len(merged) == 6
    merged = merged.points[np.argsort(merged[ORIGINAL_POINT_INDEX])]  # COPC reorders points
    for dimension in ("intensity", "classification", "return_number", "user_data", "point_source_id", "species_id"):
        np.testing.assert_array_equal(np.asarray(merged[dimension]),
                                      np.r_[np.asarray(a[dimension]), np.asarray(b[dimension])], err_msg=dimension)
    np.testing.assert_allclose(merged.x, np.r_[a.x, b.x], atol=1e-9)
    for name, cloud, part in (("first", a, slice(0, 3)), ("second", b, slice(3, 6))):
        if "gps_time" in cloud.point_format.dimension_names:
            np.testing.assert_array_equal(merged.gps_time[part], cloud.gps_time, err_msg=name)
        else:
            assert not np.any(merged.gps_time[part]), name
        if "red" in cloud.point_format.dimension_names:
            np.testing.assert_array_equal(merged.red[part], cloud.red, err_msg=name)
        elif "red" in merged.point_format.dimension_names:
            assert not np.any(merged.red[part]), name


def test_scan_angle_rank_becomes_scan_angle(tmp_path):
    first, second = tmp_path / "a.las", tmp_path / "b.las"
    format_cloud(first, 1)
    format_cloud(second, 6)
    output = tmp_path / "merged.laz"
    create_full_resolution_product([first, second], output, "laz", original_point_index=False)
    np.testing.assert_array_equal(np.asarray(laspy.read(output).points["scan_angle"]),
                                  [-2500, 0, 5000, -2500, 0, 5000])


def test_extra_bytes_in_another_order_merge_by_name(tmp_path):
    first, second = tmp_path / "a.las", tmp_path / "b.las"
    a = format_cloud(first, 3, extras=("alpha", "beta"))
    b = format_cloud(second, 3, extras=("beta", "alpha"))
    output = tmp_path / "merged.laz"
    create_full_resolution_product([first, second], output, "laz", original_point_index=False)
    merged = laspy.read(output)
    assert merged.header.point_format.id == 3
    for name in ("alpha", "beta"):
        np.testing.assert_array_equal(np.asarray(merged[name]), np.r_[np.asarray(a[name]), np.asarray(b[name])])


def test_one_point_format_stays_as_it_is(tmp_path):
    first, second = tmp_path / "a.las", tmp_path / "b.las"
    format_cloud(first, 3, version="1.2")
    format_cloud(second, 3, version="1.2")
    header = merged_header([first, second], False)
    assert header.point_format.id == 3 and str(header.version) == "1.2"


def legacy_count(path, count):
    """Claim ``count`` points in a LAS 1.2 header; merged_header reads headers only."""
    data = bytearray(path.read_bytes())
    data[107:111] = int(count).to_bytes(4, "little")
    path.write_bytes(data)


@pytest.mark.parametrize("counts, version", [((2_147_483_647, 2_147_483_648), "1.2"),
                                             ((2_147_483_648, 2_147_483_648), "1.4"),
                                             ((4_000_000_000, 3_044_924_732), "1.4")])
def test_version_can_count_every_point(tmp_path, counts, version):
    paths = [tmp_path / "a.las", tmp_path / "b.las"]
    for path, count in zip(paths, counts):
        format_cloud(path, 3, version="1.2")
        legacy_count(path, count)
    header = merged_header(paths, True)
    assert str(header.version) == version and header.point_format.id == 3


def test_writer_passes_the_32_bit_limit_after_promotion(tmp_path):
    # The job's state on dataset 2585: 4,294,255,408 points written, one million more to come.
    source = format_cloud(tmp_path / "a.las", 3, extras=(), version="1.2")
    points = laspy.ScaleAwarePointRecord.zeros(1_000_000, header=source.header)
    with pytest.raises(laspy.errors.LaspyException, match="maximum number of points"):
        with laspy.open(tmp_path / "old.las", mode="w", header=source.header.copy()) as writer:
            writer.header.point_count = 4_294_255_408
            writer.write_points(points)
    header = ensure_point_capacity(source.header.copy(), 7_044_924_732)
    assert str(header.version) == "1.4"
    with laspy.open(tmp_path / "new.las", mode="w", header=header) as writer:
        writer.header.point_count = 4_294_255_408
        writer.write_points(points)


def test_intermediate_merge_counts_every_point(tmp_path, monkeypatch):
    from remap_first_pipeline import merge_file

    paths = [tmp_path / "a.las", tmp_path / "b.las"]
    for path in paths:
        format_cloud(path, 3, extras=(), version="1.2")
    monkeypatch.setattr("remap_first_pipeline.ensure_point_capacity",
                        lambda header, total: (captured.append(total), header)[1])
    captured = []
    merge_file(paths, tmp_path / "merged.las")
    assert captured == [6]


@pytest.mark.parametrize("formats", [(1, 3), (3, 7), (1, 6), (6, 8)])
def test_mixed_point_formats_through_the_task(tmp_path, formats):
    folder = tmp_path / "in"
    folder.mkdir()
    for name, point_format in zip(("a.las", "b.las"), formats):
        format_cloud(folder / name, point_format)
    products = create_prod_merged_files(folder, tmp_path / "out", "full", "copc.laz", 0.01, 0.1)
    assert laspy.read(products[0]).header.point_count == 6
