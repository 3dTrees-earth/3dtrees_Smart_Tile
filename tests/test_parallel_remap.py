import hashlib
import sys
from pathlib import Path
import sqlite3

import laspy
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bounded_point_index import PointIndex
from parallel_remap import RemapBatchQueries
from strict_prediction_pipeline import strict_remap
from test_dense_tile_merge import write_cloud


def test_readonly_index_queries_without_mutation_or_creation(tmp_path):
    path = tmp_path / "index.sqlite"
    with PointIndex(path, {"label": np.uint16}) as index:
        index.add(0, np.zeros((1, 3)), {"label": np.array([7])}, np.array([0]))
        index.flush()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with PointIndex(path, {"label": np.uint16}, read_only=True) as index:
        assert index.nearest(np.zeros((1, 3)), .01)[1]["label"].tolist() == [7]
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            index.db.execute("DELETE FROM batches")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    missing = tmp_path / "missing.sqlite"
    with pytest.raises(sqlite3.OperationalError):
        PointIndex(missing, {}, read_only=True)
    assert not missing.exists()


def test_batch_queries_are_ordered_bounded_and_match_serial(tmp_path):
    with PointIndex(tmp_path / "i.sqlite", {"label": np.uint16}) as index:
        # Duplicate coordinates exercise stable tie-breaking across source tiles.
        for tile, label in [(0, 7), (1, 9)]:
            index.add(tile, np.array([[0., 0., 0.], [.02, 0., 0.]]),
                      {"label": np.array([label, label])}, np.array([0, 1]))
        serial = []
        for workers in (1, 4):
            produced = consumed = 0
            def batches():
                nonlocal produced
                for i in range(13):
                    produced += 1
                    assert produced - consumed <= 2 * workers
                    yield i, np.array([[0., 0., 0.], [.01, 0., 0.], [1., 0., 0.]])
            result = []
            with RemapBatchQueries([index], [index], workers=workers, radius=.01) as queries:
                for item, xyz, models in queries.map(batches()):
                    consumed += 1
                    assert item == consumed - 1
                    base, final, values = models[0]
                    assert values['label'].tolist() == [7, 7, 0]
                    assert np.isinf(final[-1])
                    result.append((base, final, values['label']))
            if workers == 1:
                serial = result
            else:
                for a, b in zip(serial, result):
                    for x, y in zip(a, b):
                        np.testing.assert_array_equal(x, y)


def test_worker_failure_propagates_and_pool_closes(tmp_path):
    with PointIndex(tmp_path / "i.sqlite", {}) as index:
        index.add(0, np.zeros((1, 3)), {}, np.array([0]))
        with pytest.raises(IndexError), RemapBatchQueries([index], [index], workers=2, radius=.01) as queries:
            list(queries.map([(None, np.array([0., 0., 0.]))]))
    # A subsequent pool must be usable after the first failed.
    with PointIndex(tmp_path / "next.sqlite", {}) as index:
        with RemapBatchQueries([index], [index], workers=2, radius=.01) as queries:
            assert len(list(queries.map([(None, np.zeros((1, 3)))]))) == 1


@pytest.mark.parametrize("missing", [False, True])
def test_strict_remap_processes_match_serial_outputs_and_coverage(tmp_path, monkeypatch, missing):
    import strict_prediction_pipeline as pipeline
    # Use multiple bounded batches without generating a large synthetic cloud.
    monkeypatch.setattr(pipeline, "MAX_BATCH_POINTS", 8)
    monkeypatch.setattr(pipeline, "spatial_query_worker_count", lambda n: n)
    originals = tmp_path / "originals"
    originals.mkdir()
    xs = np.arange(35) * .02
    source = write_cloud(originals / "a.laz", xs)
    cloud = laspy.read(source)
    cloud.header.vlrs.append(laspy.VLR(user_id="test", record_id=77, record_data=b"preserve"))
    cloud.write(source)
    collections = []
    for name in ("sat", "fm"):
        folder = tmp_path / name
        folder.mkdir()
        values = xs[:-1] if missing else xs
        write_cloud(folder / "a.laz", values, np.full(len(values), 7), np.full(len(values), 3),
                    instance=name + "PredInstance")
        collections.append(folder)
    reports = []
    for workers in (1, 4):
        output = tmp_path / f"out{workers}"
        kwargs = dict(collections=collections, baseline_collections=collections,
                      originals=originals, output=output, workers=workers)
        if missing:
            import json
            with pytest.raises(ValueError, match="100% original coverage"):
                strict_remap(**kwargs)
            assert not output.exists()
            reports.append(json.loads((tmp_path / f"out{workers}_coverage.json").read_text()))
        else:
            reports.append(strict_remap(**kwargs))
            result = laspy.read(output / "a.laz")
            for n in cloud.points.array.dtype.names:
                np.testing.assert_array_equal(cloud.points.array[n], result.points.array[n])
            assert any(v.record_data_bytes() == b"preserve" for v in result.header.vlrs)
    assert reports[0]["original_coverage"] == reports[1]["original_coverage"]
    assert reports[1]["parallelism"]["enrichment_processes"] == 4
    if not missing:
        np.testing.assert_array_equal(laspy.read(tmp_path / 'out1/a.laz').points.array,
                                      laspy.read(tmp_path / 'out4/a.laz').points.array)


def test_tiny_original_avoids_process_startup(tmp_path, monkeypatch):
    import strict_prediction_pipeline as pipeline
    monkeypatch.setattr(pipeline, "spatial_query_worker_count", lambda n: n)
    originals, predictions = tmp_path / "raw", tmp_path / "pred"
    originals.mkdir(); predictions.mkdir()
    write_cloud(originals / "a.las", [0])
    write_cloud(predictions / "a.las", [0], [7])
    report = strict_remap(collections=[predictions], baseline_collections=[predictions],
                          originals=originals, output=tmp_path / "out", workers=4)
    assert report["parallelism"]["enrichment_processes"] == 1
