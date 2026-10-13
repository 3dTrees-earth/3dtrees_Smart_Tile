"""Full-resolution prod-merged product with OriginalPointIndex (ForestStructure order key)."""
import sys
from pathlib import Path

import laspy
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from full_resolution_merge import ORIGINAL_POINT_INDEX, create_full_resolution_product  # noqa: E402
from main_create_merged_file import create_prod_merged_files, parse_merged_resolutions  # noqa: E402
from merge_stages import describe_model  # noqa: E402


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
