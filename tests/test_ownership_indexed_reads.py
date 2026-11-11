"""Ownership reading its source index equals ownership decoding the tiles, and falls back to decoding."""
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import dense_instance_ownership  # noqa: E402
from bounded_point_index import PointIndex  # noqa: E402
from indexed_chunks import IndexMismatch  # noqa: E402
from merge_stages import describe_model, reconcile_instances  # noqa: E402
from test_parallel_merge_stages import index_rows, transfer  # noqa: E402


def ownership(root, workers):
    from dense_instance_ownership import assign_shared_points, filter_owned_instances, ownership_regions
    from remap_first_pipeline import tile_overlaps
    files, _, report, _ = transfer(root, 1)
    source, target, bounds = root / "source", root / "target", root / "tile_bounds.json"
    pairs = [(s, t, s.name) for s, t in zip(sorted(source.iterdir()), sorted(target.iterdir()))]
    report["tile_sources"] = [{"tile": n, "prediction": p[0].name} for n, p in enumerate(pairs)]
    model = describe_model(source, "PredInstance")
    dims = {n: p.type for n, p in model.dimensions.items()}
    regions, overlaps = ownership_regions(pairs, bounds), tile_overlaps(pairs, bounds, np.zeros(3))
    with PointIndex(root / "dense.sqlite", dims, read_only=True) as dense, \
            PointIndex(root / "owned.sqlite", dims) as owned, \
            PointIndex(root / "resolved.sqlite", dims) as resolved, \
            PointIndex(root / "priority.sqlite", dims) as priority:
        owned_files, counts = filter_owned_instances(model, files, regions, root / "owned", owned,
                                                     np.zeros(3), report, workers=workers, source=dense)
        mapping = reconcile_instances(model, owned_files, owned, np.zeros(3), counts, .3, .05, report,
                                      overlaps=overlaps, workers=workers)
        shared = assign_shared_points(model, owned_files, owned, regions, mapping, root / "resolved",
                                      resolved, np.zeros(3), report, overlaps=overlaps, workers=workers)
        final = assign_shared_points(model, shared, resolved, regions, mapping, root / "priority",
                                     priority, np.zeros(3), report, overlaps=overlaps,
                                     background_only=True, workers=workers)
    return final, report


def outcome(root, final, report):
    stages = {key: [{k: v for k, v in t.items() if not k.endswith("_seconds")} for t in report[key]["tiles"]]
              for key in ("instance_ownership", "shared_point_ownership", "tree_background_ownership")}
    return ([Path(f).read_bytes() for f in final], stages,
            [index_rows(root / name) for name in ("owned.sqlite", "resolved.sqlite", "priority.sqlite")])


@pytest.mark.parametrize("failure", ["staging", "shard"])
def test_index_reads_equal_decoding_and_fall_back(tmp_path, failure):
    runs = {}
    calls = []

    def failing(*args):
        calls.append(args)
        raise IndexMismatch("forced")

    for mode in ("decoded", "indexed", "fallback"):
        root = tmp_path / mode
        root.mkdir()
        patches = []
        if mode == "decoded":
            patches.append(mock.patch.object(dense_instance_ownership, "tile_rows", return_value=None))
        if mode == "fallback" and failure == "staging":
            patches.append(mock.patch.object(dense_instance_ownership, "indexed_chunks", failing))
        if mode == "fallback" and failure == "shard":
            patches.append(mock.patch.object(dense_instance_ownership, "ProcessPoolExecutor", FailingShardPool))
        with _stack(patches):
            final, report = ownership(root, 1)
        runs[mode] = (root, final, report)
    decoded = outcome(*runs["decoded"])
    assert sum(t["removed"] for t in runs["decoded"][2]["shared_point_ownership"]["tiles"]) > 0
    assert outcome(*runs["indexed"]) == decoded
    assert outcome(*runs["fallback"]) == decoded
    for stage in ("core_filter", "shared_point_ownership", "tree_background_ownership"):
        assert runs["indexed"][2]["parallelism"][f"{stage}_indexed_reads"], stage
        assert runs["decoded"][2]["parallelism"][f"{stage}_indexed_reads"] == [], stage
        if failure == "staging":
            assert runs["fallback"][2]["parallelism"][f"{stage}_indexed_reads"] == [], stage
    assert not calls or failure == "staging"


class FailingShardPool:
    """Stands in for the output shard's process: its index read finds a mismatch."""

    def __init__(self, **_):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def submit(self, *_):
        from concurrent.futures import Future
        future = Future()
        future.set_exception(IndexMismatch("forced"))
        return future


def _stack(patches):
    from contextlib import ExitStack
    stack = ExitStack()
    for patch in patches:
        stack.enter_context(patch)
    return stack
