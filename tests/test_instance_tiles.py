"""Instance tiles: every instance whole in exactly one tile, per instance dimension."""
import sys
from pathlib import Path

import laspy
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from instance_tiles import write_instance_tiles  # noqa: E402
from tile_centroids import Tile  # noqa: E402


def write_source(path, xyz, sat, fm, intensity, scale=0.001):
    header = laspy.LasHeader(point_format=3, version="1.4")
    header.add_extra_dim(laspy.ExtraBytesParams(name="PredInstance_SAT", type=np.uint32))
    header.add_extra_dim(laspy.ExtraBytesParams(name="PredInstance_FM", type=np.uint16))
    header.add_extra_dim(laspy.ExtraBytesParams(name="PredScore_FM", type=np.float32))
    header.scales = np.full(3, scale)
    header.offsets = np.zeros(3)
    las = laspy.LasData(header)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    las.PredInstance_SAT = sat
    las.PredInstance_FM = fm
    las.PredScore_FM = np.linspace(0, 1, len(xyz)).astype(np.float32)
    las.intensity = intensity
    las.classification = np.where(np.asarray(sat) > 0, 5, 2)
    las.write(path)


def tiles():
    # Two 10 m tiles split at x = 10, buffered by 2 m.
    return [Tile("c00_r00", (-2.0, -2.0, 12.0, 12.0), (0.001,) * 3, (5.0, 5.0, 0.0), core=(0.0, 0.0, 10.0, 10.0)),
            Tile("c01_r00", (8.0, -2.0, 22.0, 12.0), (0.001,) * 3, (15.0, 5.0, 0.0), core=(10.0, 0.0, 20.0, 10.0))]


def inputs(tmp_path):
    rng = np.random.default_rng(1)
    # Tree 7 straddles x = 10 with its centroid at x ~ 9 (owner: left tile) and
    # reaches 3 m into the right tile, beyond the 2 m buffer. Tree 8 is right.
    tree7 = np.c_[rng.uniform(6, 13, 400), rng.uniform(4, 6, 400), rng.uniform(0, 10, 400)]
    tree8 = np.c_[rng.uniform(15, 17, 300), rng.uniform(4, 6, 300), rng.uniform(0, 10, 300)]
    ground = np.c_[rng.uniform(0, 20, 500), rng.uniform(0, 10, 500), np.zeros(500)]
    a = tmp_path / "in" / "a.las"
    b = tmp_path / "in" / "b.las"
    a.parent.mkdir()
    # Tree 7 continues across both files under the same ID.
    write_source(a, np.r_[tree7[:200], ground[:250]], np.r_[[7] * 200, [0] * 250],
                 np.r_[[1] * 200, [0] * 250], np.arange(450) % 100)
    write_source(b, np.r_[tree7[200:], tree8, ground[250:]], np.r_[[7] * 200, [8] * 300, [0] * 250],
                 np.r_[[1] * 200, [2] * 300, [0] * 250], np.arange(750) % 100)
    return [a, b]


def read(directory):
    return {p.stem: laspy.read(p) for p in sorted(Path(directory).glob("*.laz"))}


@pytest.mark.parametrize("workers", [1, 3])
def test_every_instance_is_whole_in_one_tile_per_dimension(tmp_path, workers):
    sources = inputs(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    result = write_instance_tiles(sources, tiles(), ["PredInstance_SAT", "PredInstance_FM"], out,
                                  resolutions=[0.5], attributes=["intensity", "PredScore_FM"], workers=workers)
    for dimension, ids in (("PredInstance_SAT", {7: "c00_r00", 8: "c01_r00"}),
                           ("PredInstance_FM", {1: "c00_r00", 2: "c01_r00"})):
        clouds = read(out / f"instance_tiles_{dimension}_50cm")
        for uid, label in ids.items():
            holders = [name for name, cloud in clouds.items() if np.any(cloud[dimension] == uid)]
            assert holders == [label]
        assert result[dimension]["instances"] == 2
        # Tree 7 / 1 keeps its part beyond the buffer: the left tile reaches x > 12.
        left = clouds["c00_r00"]
        assert left.x[left[dimension] == min(ids)].max() > 12.0
        # Background stays with the tile whose core holds it.
        for name, cloud in clouds.items():
            ground = cloud[dimension] == 0
            assert np.all((cloud.x[ground] <= 10) if name == "c00_r00" else (cloud.x[ground] >= 10))
        # Carried attributes are real values of an original point.
        assert set(np.unique(left.intensity)) <= set(range(100))
        assert left.PredScore_FM.dtype == np.float32


def test_voxels_never_blend_instances_and_keep_original_points(tmp_path):
    a = tmp_path / "in" / "a.las"
    a.parent.mkdir()
    # Two trees and background in one 1 m voxel: three output points, each an input point.
    xyz = np.array([[1.1, 1.1, 1.1], [1.2, 1.2, 1.2], [1.3, 1.3, 1.3], [1.4, 1.4, 1.4], [1.5, 1.5, 1.5]])
    write_source(a, xyz, [3, 3, 4, 4, 0], [0] * 5, [1, 2, 3, 4, 5])
    out = tmp_path / "out"
    out.mkdir()
    write_instance_tiles([a], tiles()[:1], ["PredInstance_SAT"], out, resolutions=[1.0])
    cloud = read(out / "instance_tiles_PredInstance_SAT_100cm")["c00_r00"]
    assert sorted(cloud.PredInstance_SAT.tolist()) == [0, 3, 4]
    original = {tuple(np.round(p, 3)) for p in xyz}
    assert {tuple(np.round(p, 3)) for p in np.c_[cloud.x, cloud.y, cloud.z]} <= original


def test_reused_ids_across_files_fail_but_wide_instances_and_border_trees_pass(tmp_path):
    a, b = tmp_path / "in" / "a.las", tmp_path / "in" / "b.las"
    a.parent.mkdir()
    # Two separately segmented files both number 12 trees from 1, 80 m apart.
    trees = np.arange(1, 13)
    write_source(a, np.c_[trees * 2.0, np.ones(12), np.ones(12)], trees, [0] * 12, [0] * 12)
    write_source(b, np.c_[80 + trees * 2.0, np.ones(12), np.ones(12)], trees, [0] * 12, [0] * 12)
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(ValueError, match="unique across all input files"):
        write_instance_tiles([a, b], tiles(), ["PredInstance_SAT"], out, resolutions=[0.5])
    # The same files pass when the IDs really are single instances.
    result = write_instance_tiles([a, b], tiles(), ["PredInstance_SAT"], out, resolutions=[0.5], check_ids=False)
    assert result["PredInstance_SAT"]["suspected_reused_ids"] == 12
    # One scattered 300 m instance within one file is real model output, not reuse.
    c = tmp_path / "in2" / "c.las"
    c.parent.mkdir()
    write_source(c, np.array([[1.0, 1.0, 1.0], [300.0, 1.0, 1.0]]), [5, 5], [0, 0], [0, 0])
    out2 = tmp_path / "out2"
    out2.mkdir()
    assert write_instance_tiles([c], tiles(), ["PredInstance_SAT"], out2, resolutions=[0.5])["PredInstance_SAT"]["instances"] == 1


def test_missing_dimension_is_rejected(tmp_path):
    sources = inputs(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(ValueError, match="missing dimensions"):
        write_instance_tiles(sources, tiles(), ["PredInstance_RCT"], out, resolutions=[0.5])


@pytest.mark.parametrize("attributes, ordinary_outputs", [
    pytest.param(None, False, id="default-attributes-instance-only"),
    pytest.param("", True, id="empty-attributes-with-ordinary-tiles"),
    pytest.param("intensity", True, id="optional-attribute-with-ordinary-tiles"),
])
def test_tile_cli_carries_instance_ids_without_requesting_them_as_attributes(tmp_path, attributes, ordinary_outputs):
    import run
    from parameters import Parameters
    sources = inputs(tmp_path)
    products = tmp_path / "products"
    args = ["--task", "tile", "--input-dir", str(sources[0].parent), "--output-dir", str(products),
            "--tile-length", "10", "--tile-buffer", "2", "--workers", "2",
            "--instance-tiles", "PredInstance_SAT,PredInstance_FM", "--instance-resolutions", "0.25,1"]
    if attributes is not None:
        args.extend(["--instance-tile-dims", attributes])
    if ordinary_outputs:
        args.extend(["--tile-outputs", "0.25:laz"])
    run.run_tile_task(Parameters(_cli_parse_args=args))

    def source_records(cloud, dimension):
        # Source coordinates are on a 1 mm grid; tile offsets may differ.
        xyz = np.rint(np.c_[cloud.x, cloud.y, cloud.z] * 1000).astype(np.int64)
        fields = [xyz, np.asarray(cloud[dimension])[:, None]]
        if attributes:
            fields.append(np.asarray(cloud[attributes])[:, None])
        return {tuple(row) for row in np.column_stack(fields)}

    originals = [laspy.read(source) for source in sources]
    ordinary = list(products.glob("subsampled_res*"))
    assert bool(ordinary) == ordinary_outputs
    for directory in ordinary:
        ordinary_clouds = read(directory)
        assert ordinary_clouds
        assert all(not list(cloud.point_format.extra_dimension_names) for cloud in ordinary_clouds.values())
    for dimension, label in [(d, r) for d in ("PredInstance_SAT", "PredInstance_FM") for r in ("25cm", "100cm")]:
        clouds = read(products / f"instance_tiles_{dimension}_{label}")
        expected_records = set().union(*(source_records(cloud, dimension) for cloud in originals))
        expected_ids = set().union(*(set(np.asarray(cloud[dimension]).tolist()) for cloud in originals))
        actual_ids = set()
        holders = {}
        for name, cloud in clouds.items():
            assert list(cloud.point_format.extra_dimension_names) == [dimension]
            expected_dtype = originals[0].point_format.dimension_by_name(dimension).dtype
            assert cloud.point_format.dimension_by_name(dimension).dtype == expected_dtype
            assert source_records(cloud, dimension) <= expected_records
            actual_ids.update(np.asarray(cloud[dimension]).tolist())
            for uid in np.unique(cloud[dimension]):
                if uid:
                    holders.setdefault(int(uid), []).append(name)
        assert actual_ids == expected_ids
        assert holders and all(len(names) == 1 for names in holders.values())
        assert all(name.startswith("c") and "_r" in name for name in clouds)


def test_species_from_instance_tiles_remap_onto_originals_within_each_instance(tmp_path):
    """Scenario: originals with predictions -> instance tiles -> species model -> remap."""
    import json
    import run
    from parameters import Parameters
    sources = inputs(tmp_path)
    run.run_tile_task(Parameters(task="tile", input_dir=sources[0].parent, output_dir=tmp_path / "products",
                                 tile_length=10, tile_buffer=2, instance_tiles="PredInstance_SAT",
                                 instance_resolutions="0.25", workers=2, _cli_parse_args=False))
    # A stand-in species model: per point species and probability, written per tile.
    species_dir = tmp_path / "species"
    species_dir.mkdir()
    for path in sorted((tmp_path / "products/instance_tiles_PredInstance_SAT_25cm").glob("*.laz")):
        cloud = laspy.read(path)
        cloud.add_extra_dims([laspy.ExtraBytesParams(name="species_id", type=np.uint16),
                              laspy.ExtraBytesParams(name="species_prob", type=np.float32)])
        ids = np.asarray(cloud.PredInstance_SAT)
        cloud.species_id = np.where(ids > 0, ids * 10, 0).astype(np.uint16)
        cloud.species_prob = (0.5 + 0.01 * (np.asarray(cloud.z) % 10)).astype(np.float32)
        cloud.write(species_dir / path.name)
    params = Parameters(task="remap", segmented_folders=str(species_dir), original_laz_input_dir=sources[0].parent,
                        original_laz_output_dir=tmp_path / "enriched", match_instances=True,
                        resolution_1=0.25, workers=2, transfer_original_dims_to_merged=False,
                        _cli_parse_args=False)
    run.run_remap_task(params)
    for source in sources:
        before, after = laspy.read(source), laspy.read(tmp_path / "enriched" / source.name)
        np.testing.assert_array_equal(after.PredInstance_SAT, before.PredInstance_SAT)
        np.testing.assert_array_equal(after.species_id, np.where(before.PredInstance_SAT > 0,
                                                                 before.PredInstance_SAT * 10, 0))
        assert np.all((after.species_prob >= 0.5) & (after.species_prob < 0.61))
    report = json.loads((tmp_path / "enriched_coverage.json").read_text())
    assert all(m["matched"] == m["total"] for m in report["original_coverage"])


def test_match_never_takes_a_neighbouring_instance_and_workers_agree(tmp_path, monkeypatch):
    import remap_first_pipeline as pipeline
    from remap_first_pipeline import strict_remap
    monkeypatch.setattr(pipeline, "MAX_BATCH_POINTS", 8)
    monkeypatch.setattr(pipeline, "spatial_query_worker_count", lambda n: n)
    originals = tmp_path / "originals"
    originals.mkdir()
    # Instance 1 at even x, instance 2 at odd x, 1 cm apart: the nearest point
    # of the other instance is always closer than the next point of one's own.
    xs = np.arange(40) * 0.01
    ids = np.where(np.arange(40) % 2 == 0, 1, 2)
    write_source(originals / "a.las", np.c_[xs, np.zeros(40), np.zeros(40)], ids, [0] * 40, [0] * 40)
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    header = laspy.LasHeader(point_format=0, version="1.4")
    header.add_extra_dim(laspy.ExtraBytesParams(name="PredInstance_SAT", type=np.uint32))
    header.add_extra_dim(laspy.ExtraBytesParams(name="species_id", type=np.uint16))
    header.scales, header.offsets = np.full(3, 0.001), np.zeros(3)
    cloud = laspy.LasData(header)
    # Each original's own prediction is 8 mm away, the previous point's (the
    # other instance) only 2 mm: unconstrained matching would swap species.
    cloud.x, cloud.y, cloud.z = xs + 0.008, np.zeros(40), np.zeros(40)
    cloud.PredInstance_SAT = ids
    cloud.species_id = ids * 10
    cloud.write(predictions / "c00_r00.laz")
    results = []
    for workers in (1, 4):
        output = tmp_path / f"out{workers}"
        strict_remap(collections=[predictions], originals=originals, output=output, workers=workers,
                     resolution_1=0.01, match_instances=True)
        enriched = laspy.read(output / "a.las")
        np.testing.assert_array_equal(enriched.species_id, ids * 10)
        np.testing.assert_array_equal(enriched.PredInstance_SAT, ids)
        results.append(enriched.points.array)
    np.testing.assert_array_equal(results[0], results[1])


def test_several_resolutions_share_the_reads_and_keep_instances_whole(tmp_path, capsys):
    sources = inputs(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    result = write_instance_tiles(sources, tiles(), ["PredInstance_SAT"], out, resolutions=[0.1, 1.0], wide=5.0)
    outputs = result["PredInstance_SAT"]["outputs"]
    assert outputs[0.1]["points"] > outputs[1.0]["points"] > 0
    for label in ("10cm", "100cm"):
        clouds = read(out / f"instance_tiles_PredInstance_SAT_{label}")
        assert [n for n, c in clouds.items() if np.any(c.PredInstance_SAT == 7)] == ["c00_r00"]
        assert [n for n, c in clouds.items() if np.any(c.PredInstance_SAT == 8)] == ["c01_r00"]
    # Tree 7 is about 7 m wide: above the 5 m warning limit, reported, not rejected.
    assert result["PredInstance_SAT"]["wide_instances"] == 1
    assert "wider than 5 m" in capsys.readouterr().out


def write_schema_source(path, instance_type, instance, score_type=np.uint16, score=1111,
                        score_scale=0.001, score_no_data=65535, dimension="PredInstance_SAT"):
    header = laspy.LasHeader(point_format=0, version="1.4")
    header.scales, header.offsets = np.full(3, 0.001), np.zeros(3)
    header.add_extra_dim(laspy.ExtraBytesParams(name=dimension, type=instance_type, description="Tree identity"))
    header.add_extra_dim(laspy.ExtraBytesParams(name="score", type=score_type, description="Calibrated score",
                                              scales=[score_scale], offsets=[-1], no_data=[score_no_data]))
    cloud = laspy.LasData(header)
    cloud.x, cloud.y, cloud.z = [1.0 if instance < 65536 else 2.0], [1.0], [1.0]
    cloud[dimension] = [instance]
    cloud.points.array["score"] = score
    cloud.write(path)


@pytest.mark.parametrize("reverse", [False, True])
def test_collection_schema_preserves_large_ids_and_scaled_attributes_in_either_source_order(tmp_path, reverse):
    from point_cloud_metadata import extra_bytes_params_from_dimension_info

    a, b = tmp_path / "a.las", tmp_path / "b.las"
    write_schema_source(a, np.uint16, 65535)
    write_schema_source(b, np.uint32, 70000, score_type=np.uint32, score=70000)
    out = tmp_path / "out"
    out.mkdir()
    write_instance_tiles([b, a] if reverse else [a, b], tiles()[:1], ["PredInstance_SAT"], out,
                         resolutions=[0.01], attributes=["score"], workers=1)
    cloud = read(out / "instance_tiles_PredInstance_SAT_1cm")["c00_r00"]
    order = np.argsort(cloud.x)
    np.testing.assert_array_equal(cloud.PredInstance_SAT[order], [65535, 70000])
    assert cloud.PredInstance_SAT.dtype == np.uint32
    np.testing.assert_array_equal(cloud.points.array["score"][order], [1111, 70000])
    np.testing.assert_allclose(np.asarray(cloud.score)[order], [0.111, 69.0])
    info = cloud.point_format.dimension_by_name("score")
    params = extra_bytes_params_from_dimension_info(info, header=cloud.header)
    assert params.type == np.dtype("uint32")
    np.testing.assert_array_equal(params.scales, [0.001])
    np.testing.assert_array_equal(params.offsets, [-1])
    np.testing.assert_array_equal(params.no_data, [65535])
    assert params.description == "Calibrated score"


@pytest.mark.parametrize("different", [dict(score_scale=0.01), dict(score_no_data=65534),
                                        dict(score_type=np.float32)])
def test_incompatible_carried_attribute_schema_fails_before_publishing(tmp_path, different):
    a, b = tmp_path / "a.las", tmp_path / "b.las"
    write_schema_source(a, np.uint16, 1)
    write_schema_source(b, np.uint32, 70000, **different)
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(ValueError, match="incompatible schema for score"):
        write_instance_tiles([a, b], tiles()[:1], ["PredInstance_SAT"], out,
                             resolutions=[0.01], attributes=["score"], workers=1)
    assert list(out.iterdir()) == []


def test_instance_dimension_names_are_data_not_scratch_paths(tmp_path):
    dimension = "../../escaped"
    source = tmp_path / "source.las"
    write_schema_source(source, np.uint32, 70000, dimension=dimension)
    out = tmp_path / "out"
    out.mkdir()
    result = write_instance_tiles([source], tiles()[:1], [dimension], out, resolutions=[0.01], workers=1)
    directory = result[dimension]["outputs"][0.01]["directory"]
    assert directory.parent == out
    assert directory.name == "instance_tiles_..%2F..%2Fescaped_1cm"
    cloud = read(directory)["c00_r00"]
    np.testing.assert_array_equal(cloud[dimension], [70000])
    assert not (tmp_path / "escaped").exists()
