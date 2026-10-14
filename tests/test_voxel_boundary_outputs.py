"""Regression for the 20-point UTM fixture that incorrectly kept three FM voxels."""
from collections import Counter
from pathlib import Path
import sys

import laspy
import numpy as np
import pytest
from pyproj import CRS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from instance_tiles import write_instance_tiles
from tile_centroids import Tile, write_tile_centroids


@pytest.fixture
def boundary_source(tmp_path):
    header = laspy.LasHeader(point_format=0, version="1.4")
    header.scales = (0.001,) * 3
    header.offsets = (500000.0, 5400000.0, 0.0)
    header.add_crs(CRS.from_epsg(32632))
    header.add_extra_dims([
        laspy.ExtraBytesParams(name="PredInstance_SAT", type=np.uint32),
        laspy.ExtraBytesParams(name="PredInstance_FM", type=np.uint32),
        laspy.ExtraBytesParams(name="PointToken", type=np.uint16),
        laspy.ExtraBytesParams(name="Amplitude", type=np.int16, no_data=[-1]),
    ])
    cloud = laspy.LasData(header)
    cloud.X = np.arange(1000, 1200, 10, dtype=np.int32)
    cloud.Y = cloud.Z = np.full(20, 1000, dtype=np.int32)
    cloud.PredInstance_SAT = np.tile([7, 70000], 10)
    cloud.PredInstance_FM = np.repeat([11, 22], 10)
    cloud.PointToken = np.arange(20, dtype=np.uint16)
    cloud.Amplitude = np.where(np.arange(20) % 3 == 0, -1, 42).astype(np.int16)
    source = tmp_path / "boundary.las"
    cloud.write(source)
    bounds = (500000.0, 5400000.0, 500002.0, 5400002.0)
    tile = Tile("c00_r00", bounds, tuple(header.scales), tuple(header.offsets), core=bounds)
    return source, cloud, tile


def instance_rows(cloud, dimension):
    return Counter(zip(cloud.X.tolist(), cloud.Y.tolist(), cloud.Z.tolist(),
                       cloud[dimension].tolist(), cloud.PointToken.tolist(), cloud.Amplitude.tolist()))


def test_multidimension_boundary_outputs_preserve_exact_cells_and_source_records(boundary_source, tmp_path, monkeypatch):
    import instance_tiles

    source, before, tile = boundary_source
    snapshots = []
    for workers, chunk, block_size in ((1, 1000, 10.0), (3, 3, 0.1)):
        # The small chunk creates separate read tasks; the smaller blocks put
        # the previously problematic X=500001.1 face on a spill boundary.
        monkeypatch.setattr(instance_tiles, "CHUNK_POINTS", chunk)
        monkeypatch.setattr(instance_tiles, "choose_block_size", lambda *args: block_size)
        output = tmp_path / f"instances_{workers}"
        output.mkdir()
        result = write_instance_tiles([source], [tile], ["PredInstance_SAT", "PredInstance_FM"], output,
                                      resolutions=[0.0, 0.1], attributes=["PointToken", "Amplitude"], workers=workers)
        snapshot = {}
        for dimension, expected_tokens in (("PredInstance_SAT", [4, 5, 14, 15]),
                                           ("PredInstance_FM", [4, 14])):
            for resolution in (0.0, 0.1):
                info = result[dimension]["outputs"][resolution]
                files = list(info["directory"].glob("*.laz"))
                assert len(files) == 1
                after = laspy.read(files[0])
                rows = instance_rows(after, dimension)
                assert after.header.parse_crs() == before.header.parse_crs()
                missing = next(struct for vlr in after.header.vlrs
                               if vlr.user_id == "LASF_Spec" and vlr.record_id == 4
                               for struct in vlr.extra_bytes_structs if struct.name == b"Amplitude")
                np.testing.assert_array_equal(missing.no_data, [-1])
                if resolution == 0:
                    assert rows == instance_rows(before, dimension)
                    assert len(after.points) == info["points"] == 20
                else:
                    assert sorted(after.PointToken.tolist()) == expected_tokens
                    assert rows <= instance_rows(before, dimension)
                    assert len(after.points) == info["points"] == len(expected_tokens)
                snapshot[(dimension, resolution)] = rows
        snapshots.append(snapshot)
    assert snapshots[0] == snapshots[1]


@pytest.mark.parametrize("nearest", [False, True], ids=["center-of-mass", "nearest-to-centroid"])
def test_spatial_boundary_outputs_are_invariant_to_chunks_blocks_and_workers(boundary_source, tmp_path, monkeypatch, nearest):
    import tile_centroids

    source, before, tile = boundary_source
    snapshots = []
    for workers, chunk, block_limit, block_size in ((1, 1000, 1000, 10.0), (3, 3, 3, 0.2)):
        monkeypatch.setattr(tile_centroids, "CHUNK_POINTS", chunk)
        monkeypatch.setattr(tile_centroids, "BLOCK_POINTS", block_limit)
        monkeypatch.setattr(tile_centroids, "choose_block_size", lambda *args: block_size)
        output = tmp_path / f"spatial_{workers}"
        selections = [(resolution, {tile.label: output / f"resolution_{index}.laz"}, "laz")
                      for index, resolution in enumerate((0.0, 0.1))]
        written, occupied = write_tile_centroids([source], [tile], before.header, selections,
                                                  workers=workers, work_dir=output, metadata_source=source,
                                                  nearest=nearest)
        assert occupied == [True]
        snapshot = {}
        for resolution, files, _ in selections:
            after = laspy.read(files[tile.label])
            snapshot[resolution] = Counter(zip(after.X.tolist(), after.Y.tolist(), after.Z.tolist()))
            assert after.header.parse_crs() == before.header.parse_crs()
            if resolution == 0:
                assert snapshot[resolution] == Counter(zip(before.X.tolist(), before.Y.tolist(), before.Z.tolist()))
                assert len(after.points) == written[tile.label][0] == 20
            else:
                expected_x = [1040, 1140] if nearest else [1045, 1145]
                assert sorted(after.X.tolist()) == expected_x
                assert len(after.points) == written[tile.label][1] == 2
                np.testing.assert_array_equal(after.Y, [1000, 1000])
                np.testing.assert_array_equal(after.Z, [1000, 1000])
        snapshots.append(snapshot)
    assert snapshots[0] == snapshots[1]


@pytest.mark.parametrize("dimension,expected_tokens", [("PredInstance_SAT", [4, 5, 14, 15]),
                                                       ("PredInstance_FM", [4, 14])])
@pytest.mark.parametrize("reverse,chunk_size", [(False, 20), (True, 3)])
@pytest.mark.parametrize("resolutions", [(0.0,), (0.0, 0.1)])
def test_instance_reducer_exact_boundary_records(boundary_source, tmp_path, dimension, expected_tokens,
                                                reverse, chunk_size, resolutions):
    """Exercise real spill reduction without conversion/plotting dependencies."""
    from instance_tiles import _record_dtype, _reduce

    _, before, _ = boundary_source
    attributes = ("PointToken", "Amplitude")
    dtypes = [before.points.array.dtype[name] for name in attributes]
    records = np.zeros(len(before.points), dtype=_record_dtype(dtypes))
    for axis in ("X", "Y", "Z"):
        records[axis] = before.points.array[axis]
    records["instance"] = before[dimension]
    records["m"] = (1 << len(resolutions)) - 1
    for index, attribute in enumerate(attributes):
        records[f"a{index}"] = before.points.array[attribute]
    if reverse:
        records = records[::-1].copy()
    block = tmp_path / "block"
    block.mkdir()
    for index, start in enumerate(range(0, len(records), chunk_size)):
        records[start:start + chunk_size].tofile(block / f"{index:06d}.bin")
    header = before.header.copy()
    header.remove_extra_dim("PredInstance_FM" if dimension == "PredInstance_SAT" else "PredInstance_SAT")
    fragments = [tmp_path / f"fragment_{index}.bin" for index in range(len(resolutions))]
    results = _reduce((block, resolutions, header.scales, header.offsets, dtypes, header.point_format,
                       dimension, attributes, fragments))
    for resolution, fragment, (kept, used) in zip(resolutions, fragments, results):
        array = np.fromfile(fragment, dtype=header.point_format.dtype())
        points = laspy.ScaleAwarePointRecord(array, header.point_format, header.scales, header.offsets)
        after = laspy.LasData(header, points)
        assert used == 20
        assert kept == len(points)
        if resolution == 0:
            assert kept == 20
            assert instance_rows(after, dimension) == instance_rows(before, dimension)
        else:
            assert kept == len(expected_tokens)
            assert sorted(after.PointToken.tolist()) == expected_tokens
            assert instance_rows(after, dimension) <= instance_rows(before, dimension)
    assert not block.exists()


@pytest.mark.parametrize("nearest", [False, True], ids=["center-of-mass", "nearest-to-centroid"])
@pytest.mark.parametrize("reverse,chunk_size,block_limit", [(False, 20, 1000), (True, 3, 3)])
def test_spatial_reducer_exact_boundary_records(boundary_source, tmp_path, nearest, reverse,
                                               chunk_size, block_limit):
    """Compare actual whole and split-block reducers with the known exact result."""
    from tile_centroids import RECORD, _reduce_block, reduce_records

    _, before, tile = boundary_source
    records = np.zeros(len(before.points), dtype=RECORD)
    for axis in ("X", "Y", "Z"):
        records[axis] = before.points.array[axis]
    records["m"] = 3
    if reverse:
        records = records[::-1].copy()
    block = tmp_path / "block"
    block.mkdir()
    for index, start in enumerate(range(0, len(records), chunk_size)):
        records[start:start + chunk_size].tofile(block / f"{index:06d}.bin")
    direct = reduce_records(records, (0.0, 0.1), tile.scales, tile.offsets, nearest=nearest)
    split = _reduce_block((block, (0.0, 0.1), (2, 2), tile.scales, tile.offsets, block_limit, nearest))
    expected_x = [1040, 1140] if nearest else [1045, 1145]
    for result in (direct, split):
        for slot, (xyz, represented) in enumerate(result):
            assert represented == 20
            raw = np.rint((xyz - tile.offsets) / tile.scales).astype(np.int64)
            if slot == 0:
                assert len(raw) == 20
                assert Counter(map(tuple, raw)) == Counter(zip(before.X.tolist(), before.Y.tolist(), before.Z.tolist()))
            else:
                assert len(raw) == 2
                assert sorted(raw[:, 0].tolist()) == expected_x
                np.testing.assert_array_equal(raw[:, 1:], np.full((2, 2), 1000))
