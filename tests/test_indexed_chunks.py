"""Chunks read back from a tile's index equal decoding the tile, chunk by chunk."""
import sys
from pathlib import Path
from unittest.mock import patch

import laspy
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import merge_stages  # noqa: E402
from bounded_point_index import PointIndex, coordinates  # noqa: E402
from indexed_chunks import IndexMismatch, indexed_chunks, tile_rows  # noqa: E402
from merge_stages import describe_model, index_file, prediction_values  # noqa: E402
from test_merge_stages import write_cloud  # noqa: E402

ORIGIN = np.array([1.5, -2.25, 0.0])


def clouds(folder, sizes):
    rng = np.random.default_rng(311)
    folder.mkdir()
    return [write_cloud(folder / f'{tile}.las', rng.uniform(-10, 30, size), rng.integers(0, 9, size),
                        rng.integers(0, 4, size), ys=rng.uniform(-5, 5, size), zs=rng.uniform(0, 20, size))
            for tile, size in enumerate(sizes)]


@pytest.mark.parametrize('chunk', [100, 333, 1000])
def test_indexed_chunks_equal_decoded_chunks(tmp_path, chunk):
    files = clouds(tmp_path / 'in', [2500, 0, 1777])
    model = describe_model(files[0].parent, 'PredInstance')
    dims = {n: p.type for n, p in model.dimensions.items()}
    with patch.object(merge_stages, 'MAX_BATCH_POINTS', 250), PointIndex(tmp_path / 'i.db', dims) as index:
        for tile, file in enumerate(files):
            index_file(index, file, tile, ORIGIN, model)
        for tile, file in enumerate(files):
            with laspy.open(file) as reader:
                count = reader.header.point_count
                expected = [(coordinates(r, reader.header, ORIGIN), prediction_values(r, model.dimensions))
                            for r in reader.chunk_iterator(chunk)] if count else []
            assert (tile_rows(index.db, tile) is not None) == (count > 0)
            got = list(indexed_chunks(index.db, index._storage_dtype(), list(dims), tile, count, chunk))
            assert len(got) == len(expected)
            for (xyz, values), (want_xyz, want_values) in zip(got, expected):
                np.testing.assert_array_equal(xyz, want_xyz)
                assert values.keys() == want_values.keys()
                for name in values:
                    assert values[name].dtype == want_values[name].dtype
                    np.testing.assert_array_equal(values[name], want_values[name])


def test_index_without_every_record_is_refused(tmp_path):
    files = clouds(tmp_path / 'in', [600])
    model = describe_model(files[0].parent, 'PredInstance')
    dims = {n: p.type for n, p in model.dimensions.items()}
    with PointIndex(tmp_path / 'i.db', dims) as index:
        with laspy.open(files[0]) as reader:
            record = reader.read_points(600)
            xyz, values = coordinates(record, reader.header, ORIGIN), prediction_values(record, model.dimensions)
        keep = np.arange(600) != 17
        index.add(0, xyz[keep], {k: v[keep] for k, v in values.items()}, np.arange(600)[keep])
        index.flush()
        with pytest.raises(IndexMismatch):
            list(indexed_chunks(index.db, index._storage_dtype(), list(dims), 0, 600, 200))
