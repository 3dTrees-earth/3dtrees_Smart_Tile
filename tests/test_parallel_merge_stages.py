"""Tile-parallel dense transfer must equal the one-process result exactly.

Covers equal-distance ties (1 cm targets halfway between 10 cm predictions),
instances crossing tile buffers, failure reporting order and the full merge.
"""
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import laspy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import merge_stages  # noqa: E402
from bounded_point_index import PointIndex  # noqa: E402
from merge_stages import describe_model, prepare_dense, query_process_shares  # noqa: E402
from remap_first_pipeline import merge_collections  # noqa: E402
from test_merge_stages import write_cloud  # noqa: E402

TILE, BUFFER, TILES = 4.0, 0.5, 3


def grid(x0, x1, y0, y1, step):
    xs, ys = np.meshgrid(np.arange(x0, x1 + 1e-9, step), np.arange(y0, y1 + 1e-9, step), indexing="ij")
    return xs.ravel(), ys.ravel()


def build(root, *, gap_tile=None, count=TILES, wide_tile=None, target_shift=0.0, exact_bands=False):
    """Three buffered tiles; predictions every 10 cm, targets every 2.5 cm (exact ties at 5 cm)."""
    source, target, originals = (root / n for n in ("source", "target", "originals"))
    for folder in (source, target, originals):
        folder.mkdir()
    tiles, core_points = [], []
    for i in range(count):
        x0, x1 = i * TILE, (i + 1) * TILE
        name = f"c{i:02d}_r00"
        sx, sy = grid(x0 - BUFFER, x1 + BUFFER, 0, TILE, 0.1)
        if exact_bands:   # integer decimetres: shared points get identical labels in every tile
            ids = (1 + np.floor_divide(np.round(sx * 10).astype(np.int64), 13)).astype(np.uint32)
        else:             # 1.3 m instance bands crossing tile borders
            ids = (1 + np.floor(sx / 1.3)).astype(np.uint32)
        write_cloud(source / f"{name}_subsampled_10cm.las", sx, ids, (np.floor(sy) % 3).astype(np.uint8), ys=sy,
                    zs=0.2 * np.sin(sx))
        tx, ty = grid(x0 - BUFFER, x1 + BUFFER, 0, TILE, 0.0125 if i == wide_tile else 0.025)
        tx = tx + target_shift
        tz = 0.2 * np.sin(tx)
        if i == gap_tile:
            tx, ty, tz = np.r_[tx, x0 + 1.0], np.r_[ty, TILE + 2.0], np.r_[tz, 0.0]   # no prediction nearby
        write_cloud(target / f"{name}_subsampled_1cm.las", tx, ys=ty, zs=tz)
        core = (tx >= x0) & (tx < x1)
        core_points.append((tx[core], ty[core], tz[core]))
        tiles.append({"col": i, "row": 0, "bounds": [[x0 - BUFFER, x1 + BUFFER], [-BUFFER, TILE + BUFFER]],
                      "core": [[x0, x1], [0, TILE]]})
    ox, oy, oz = (np.concatenate(axis) for axis in zip(*core_points))
    write_cloud(originals / "plot.las", ox, ys=oy, zs=oz)
    bounds = root / "tile_bounds.json"
    bounds.write_text(json.dumps({"tile_buffer": BUFFER, "tiles": tiles}))
    return source, target, originals, bounds


def index_rows(path):
    with sqlite3.connect(path) as db:
        return (db.execute("SELECT id, tile, data FROM batches ORDER BY id").fetchall(),
                db.execute("SELECT * FROM bounds ORDER BY id").fetchall())


def transfer(root, workers, **build_kwargs):
    source, target, _, _ = build(root, **build_kwargs)
    model = describe_model(source, "PredInstance")
    pairs = [(s, t, s.name) for s, t in zip(sorted(source.iterdir()), sorted(target.iterdir()))]
    report = {}
    dims = {n: p.type for n, p in model.dimensions.items()}
    with PointIndex(root / "dense.sqlite", dims, query_workers=workers) as index:
        try:
            files, counts = prepare_dense(model, pairs, root / "dense", index, np.zeros(3), .1732, report,
                                          workers=workers)
        except ValueError as error:
            return None, None, report, error
    return files, counts, report, None


class ParallelDenseTransferTests(unittest.TestCase):
    def test_tile_processes_equal_one_process(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            serial, parallel = Path(a), Path(b)
            files_1, counts_1, report_1, _ = transfer(serial, 1)
            files_3, counts_3, report_3, _ = transfer(parallel, 3)
            self.assertEqual(report_1["parallelism"], {"dense_transfer_processes": 1, "dense_query_processes": [1, 1, 1]})
            self.assertEqual(report_3["parallelism"], {"dense_transfer_processes": 3, "dense_query_processes": [1, 1, 1]})
            self.assertEqual([f.name for f in files_1], [f.name for f in files_3])
            for f1, f3 in zip(files_1, files_3):
                p1, p3 = laspy.read(f1).points.array, laspy.read(f3).points.array
                self.assertEqual(p1.dtype, p3.dtype)
                np.testing.assert_array_equal(p1, p3)
            self.assertEqual(list(counts_1.items()), list(counts_3.items()))
            strip = lambda r: [{k: v for k, v in m.items() if k not in ("source", "target")} for m in r["transfer"]]
            self.assertEqual(strip(report_1), strip(report_3))
            self.assertEqual(index_rows(serial / "dense.sqlite"), index_rows(parallel / "dense.sqlite"))
            self.assertEqual(sorted(p.name for p in (parallel / "dense").iterdir()), [f.name for f in files_3])

    def test_ties_are_present_and_resolved_identically(self):
        with tempfile.TemporaryDirectory() as tmp:
            files, _, _, _ = transfer(Path(tmp), 3)
            cloud = laspy.read(files[1])
            tied = np.isclose(np.mod(np.asarray(cloud.x) + 1e-9, 0.1), 0.05, atol=1e-6)
            self.assertGreater(int(tied.sum()), 1000)

    def test_incomplete_tile_fails_like_one_process(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            _, _, report_1, error_1 = transfer(Path(a), 1, gap_tile=1)
            _, _, report_3, error_3 = transfer(Path(b), 3, gap_tile=1)
            self.assertIsNotNone(error_3)
            self.assertEqual(str(error_1), str(error_3))
            self.assertIn("c01_r00_subsampled_1cm.las", str(error_3))
            self.assertEqual([m["tile"] for m in report_3["transfer"]], [0, 1])
            self.assertEqual(report_1["transfer"][1]["missing_examples"], report_3["transfer"][1]["missing_examples"])

    def test_merge_with_worker_processes_equals_one_process(self):
        outputs = []
        for workers in (1, 4):
            tmp = tempfile.TemporaryDirectory()
            self.addCleanup(tmp.cleanup)
            root = Path(tmp.name)
            source, target, originals, bounds = build(root)
            report = merge_collections(collections=[source], target_dir=target, output_tiles=root / "out",
                                       tile_bounds_json=bounds, originals=originals, workers=workers)
            self.assertEqual(report["state"], "validated")
            outputs.append((root, report))
        (r1, rep1), (r4, rep4) = outputs
        tiles = sorted(p.name for p in (r1 / "out").glob("*.la[sz]"))
        self.assertEqual(len(tiles), TILES)
        self.assertEqual(tiles, sorted(p.name for p in (r4 / "out").glob("*.la[sz]")))
        for name in tiles:
            np.testing.assert_array_equal(laspy.read(r1 / "out" / name).points.array,
                                          laspy.read(r4 / "out" / name).points.array)
        np.testing.assert_array_equal(laspy.read(r1 / "original_with_predictions/plot.las").points.array,
                                      laspy.read(r4 / "original_with_predictions/plot.las").points.array)
        for name in ("instance_metadata.csv", "PredInstance_summary.json"):
            self.assertEqual((r1 / "out" / name).read_bytes(), (r4 / "out" / name).read_bytes())
        labels = np.unique(laspy.read(r4 / "original_with_predictions/plot.las").PredInstance)
        self.assertEqual(labels.tolist(), list(range(11)))   # background 0 and ten trees crossing tile borders
        self.assertEqual(rep4["models"][0]["parallelism"]["dense_transfer_processes"], 3)

    def test_uneven_tiles_get_proportional_query_processes(self):
        self.assertEqual(query_process_shares([190, 14, 133, 37], 10), [5, 1, 4, 1])   # 3109-like layout
        self.assertEqual(query_process_shares([5, 5], 10, ready=True), [1, 1])
        self.assertEqual(query_process_shares([0, 0], 4), [1, 1])

    def test_uneven_tiles_with_nested_query_processes_equal_one_process(self):
        """A large tile next to small ones runs its queries in its own processes."""
        # No batch-size patch here: spawned tile workers would not see it.
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            serial, parallel = Path(a), Path(b)
            files_1, counts_1, report_1, _ = transfer(serial, 1, count=2, wide_tile=0)
            self.assertGreater(report_1["transfer"][0]["total"], 3 * 32_768)
            files_6, counts_6, report_6, _ = transfer(parallel, 6, count=2, wide_tile=0)
            self.assertEqual(report_6["parallelism"]["dense_transfer_processes"], 2)
            self.assertGreater(report_6["parallelism"]["dense_query_processes"][0], 1)
            for f1, f6 in zip(files_1, files_6):
                np.testing.assert_array_equal(laspy.read(f1).points.array, laspy.read(f6).points.array)
            self.assertEqual(list(counts_1.items()), list(counts_6.items()))
            self.assertEqual(index_rows(serial / "dense.sqlite"), index_rows(parallel / "dense.sqlite"))

    def test_single_tile_query_processes_equal_one_process(self):
        # Small batches put many queries in flight at once (results must come back in order).
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b, \
                mock.patch.object(merge_stages, "MAX_BATCH_POINTS", 2048):
            files_1, counts_1, report_1, _ = transfer(Path(a), 1, count=1)
            files_4, counts_4, report_4, _ = transfer(Path(b), 4, count=1)
            self.assertEqual(report_4["parallelism"], {"dense_transfer_processes": 1, "dense_query_processes": [4]})
            np.testing.assert_array_equal(laspy.read(files_1[0]).points.array, laspy.read(files_4[0]).points.array)
            self.assertEqual(list(counts_1.items()), list(counts_4.items()))
            self.assertEqual(index_rows(Path(a) / "dense.sqlite"), index_rows(Path(b) / "dense.sqlite"))
            self.assertGreater(report_4["transfer"][0]["total"], 10 * 2048)

    def test_core_filter_processes_equal_one_process(self):
        from dense_instance_ownership import filter_owned_instances, ownership_regions
        results = []
        for workers in (1, 3):
            tmp = tempfile.TemporaryDirectory()
            self.addCleanup(tmp.cleanup)
            root = Path(tmp.name)
            files, _, report, _ = transfer(root, 1)
            source, target = root / "source", root / "target"
            pairs = [(s, t, s.name) for s, t in zip(sorted(source.iterdir()), sorted(target.iterdir()))]
            report["tile_sources"] = [{"tile": n, "prediction": str(p[0].name)} for n, p in enumerate(pairs)]
            model = describe_model(source, "PredInstance")
            dims = {n: p.type for n, p in model.dimensions.items()}
            with PointIndex(root / "owned.sqlite", dims) as owned:
                outputs, counts = filter_owned_instances(model, files, ownership_regions(pairs, root / "tile_bounds.json"),
                                                         root / "owned", owned, np.zeros(3), report, workers=workers)
            results.append((root, outputs, counts, report))
        (r1, o1, c1, rep1), (r3, o3, c3, rep3) = results
        self.assertEqual(rep3["parallelism"]["core_filter_processes"], 3)
        self.assertEqual(rep1["instance_ownership"], rep3["instance_ownership"])
        self.assertEqual(list(c1.items()), list(c3.items()))
        self.assertGreater(sum(t["removed_instances"] for t in rep3["instance_ownership"]["tiles"]), 0)
        for a, b in zip(o1, o3):
            np.testing.assert_array_equal(laspy.read(a).points.array, laspy.read(b).points.array)
        self.assertEqual(index_rows(r1 / "owned.sqlite"), index_rows(r3 / "owned.sqlite"))
        self.assertEqual(sorted(p.name for p in (r3 / "owned").iterdir()), [p.name for p in o3])

    def test_reconciliation_and_ownership_processes_equal_one_process(self):
        from dense_instance_ownership import assign_shared_points, filter_owned_instances, ownership_regions
        from merge_stages import reconcile_instances
        from remap_first_pipeline import tile_overlaps
        runs = []
        for workers in (1, 3):
            tmp = tempfile.TemporaryDirectory()
            self.addCleanup(tmp.cleanup)
            root = Path(tmp.name)
            files, _, report, _ = transfer(root, 1)
            source, target, bounds = root / "source", root / "target", root / "tile_bounds.json"
            pairs = [(s, t, s.name) for s, t in zip(sorted(source.iterdir()), sorted(target.iterdir()))]
            report["tile_sources"] = [{"tile": n, "prediction": p[0].name} for n, p in enumerate(pairs)]
            model = describe_model(source, "PredInstance")
            dims = {n: p.type for n, p in model.dimensions.items()}
            regions, overlaps = ownership_regions(pairs, bounds), tile_overlaps(pairs, bounds, np.zeros(3))
            with PointIndex(root / "owned.sqlite", dims) as owned, \
                    PointIndex(root / "resolved.sqlite", dims) as resolved, \
                    PointIndex(root / "priority.sqlite", dims) as priority:
                owned_files, counts = filter_owned_instances(model, files, regions, root / "owned", owned,
                                                             np.zeros(3), report, workers=workers)
                mapping = reconcile_instances(model, owned_files, owned, np.zeros(3), counts, .3, .05, report,
                                              overlaps=overlaps, workers=workers)
                shared = assign_shared_points(model, owned_files, owned, regions, mapping, root / "resolved",
                                              resolved, np.zeros(3), report, overlaps=overlaps, workers=workers)
                final = assign_shared_points(model, shared, resolved, regions, mapping, root / "priority",
                                             priority, np.zeros(3), report, overlaps=overlaps,
                                             background_only=True, workers=workers)
            runs.append((root, mapping, final, report))
        (r1, m1, f1, rep1), (r3, m3, f3, rep3) = runs
        self.assertEqual(rep3["parallelism"]["reconciliation_processes"], 3)
        self.assertEqual(rep3["parallelism"]["shared_point_ownership_processes"], 3)
        self.assertEqual(m1, m3)
        self.assertGreater(len(set(m3.values())), 0)
        volatile = lambda section: {k: v for k, v in section.items() if not k.endswith("_seconds") and k != "tiles"}
        tiles = lambda section: [{k: v for k, v in t.items() if not k.endswith("_seconds")} for t in section["tiles"]]
        for key in ("reconciliation", "shared_point_ownership", "tree_background_ownership"):
            if key in rep1:
                self.assertEqual(volatile(rep1[key]), volatile(rep3[key]), key)
                if "tiles" in rep1[key]:
                    self.assertEqual(tiles(rep1[key]), tiles(rep3[key]), key)
        self.assertGreater(sum(t["removed"] for t in rep3["shared_point_ownership"]["tiles"]), 0)
        for a, b in zip(f1, f3):
            np.testing.assert_array_equal(laspy.read(a).points.array, laspy.read(b).points.array)
        for name in ("resolved.sqlite", "priority.sqlite"):
            self.assertEqual(index_rows(r1 / name), index_rows(r3 / name), name)

    def test_dedup_chunk_processes_equal_one_process(self):
        """A large later tile deduplicates its chunks in worker processes."""
        runs = []
        for workers in (1, 4):
            tmp = tempfile.TemporaryDirectory()
            self.addCleanup(tmp.cleanup)
            root = Path(tmp.name)
            source, target, originals, bounds = build(root, wide_tile=1)
            report = merge_collections(collections=[source], target_dir=target, output_tiles=root / "out",
                                       tile_bounds_json=bounds, originals=originals, workers=workers)
            self.assertEqual(report["state"], "validated")
            runs.append((root, report["models"][0]))
        (r1, m1), (r4, m4) = runs
        self.assertEqual(m1["parallelism"]["deduplication_parallel_tiles"], [])
        self.assertEqual(m4["parallelism"]["deduplication_parallel_tiles"], [1])
        self.assertEqual(m1["deduplication"], m4["deduplication"])
        for name in sorted(p.name for p in (r1 / "out").glob("*.la[sz]")):
            np.testing.assert_array_equal(laspy.read(r1 / "out" / name).points.array,
                                          laspy.read(r4 / "out" / name).points.array)
        np.testing.assert_array_equal(laspy.read(r1 / "original_with_predictions/plot.las").points.array,
                                      laspy.read(r4 / "original_with_predictions/plot.las").points.array)
        self.assertEqual((r1 / "out/PredInstance_summary.json").read_bytes(),
                         (r4 / "out/PredInstance_summary.json").read_bytes())

    def test_dedup_removes_duplicates_identically_in_processes(self):
        from dense_instance_ownership import ownership_regions
        from merge_stages import deduplicate, reconcile_instances
        from remap_first_pipeline import tile_overlaps
        runs = []
        for workers in (1, 4):
            tmp = tempfile.TemporaryDirectory()
            self.addCleanup(tmp.cleanup)
            root = Path(tmp.name)
            # Raw dense tiles (no ownership stages): labels of shared points must agree across tiles.
            files, counts, report, _ = transfer(root, 1, wide_tile=1, target_shift=0.003, exact_bands=True)
            source, target, bounds = root / "source", root / "target", root / "tile_bounds.json"
            pairs = [(s, t, s.name) for s, t in zip(sorted(source.iterdir()), sorted(target.iterdir()))]
            model = describe_model(source, "PredInstance")
            dims = {n: p.type for n, p in model.dimensions.items()}
            regions, overlaps = ownership_regions(pairs, bounds), tile_overlaps(pairs, bounds, np.zeros(3))
            with PointIndex(root / "dense.sqlite", dims, read_only=True) as dense, \
                    PointIndex(root / "survivors.sqlite", dims) as survivors:
                mapping = reconcile_instances(model, files, dense, np.zeros(3), counts, .3, .05, report,
                                              overlaps=overlaps)
                outputs = deduplicate(model, files, dense, survivors, np.zeros(3), mapping, root / "final", report,
                                      overlaps=overlaps, background_semantics_owned=True, core_regions=regions,
                                      workers=workers)
            runs.append((root, outputs, report))
        (r1, o1, rep1), (r4, o4, rep4) = runs
        self.assertEqual(rep4["parallelism"]["deduplication_parallel_tiles"], [1])   # tile 2 is a single batch
        self.assertEqual(rep1["deduplication"], rep4["deduplication"])
        self.assertGreater(rep4["deduplication"]["tiles"][1]["removed"], 1000)
        for a, b in zip(o1, o4):
            np.testing.assert_array_equal(laspy.read(a).points.array, laspy.read(b).points.array)
        self.assertEqual(index_rows(r1 / "survivors.sqlite"), index_rows(r4 / "survivors.sqlite"))


if __name__ == "__main__":
    unittest.main()
