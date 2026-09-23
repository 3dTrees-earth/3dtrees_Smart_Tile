"""Reproducible synthetic smoke test; not a replacement for incident replay.

Run with the SmartTile dependencies plus psutil:
    python benchmarks/remap_first_smoke.py --output-root out/remap-first-smoke
The output directory must not exist. Inputs and reports remain for inspection.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import laspy
import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from strict_prediction_pipeline import merge_collections


def cloud(path, x_values, y_values, instance=None):
    x, y = np.meshgrid(x_values, y_values, indexing="ij")
    header = laspy.LasHeader(point_format=3, version="1.4")
    header.scales = np.array([.000001] * 3)
    data = laspy.LasData(header)
    data.x, data.y, data.z = x.ravel(), y.ravel(), np.zeros(x.size)
    if instance is not None:
        data.add_extra_dims([
            laspy.ExtraBytesParams(name="PredInstance", type=np.uint32),
            laspy.ExtraBytesParams(name="PredSemantic", type=np.uint8)])
        data.PredInstance = np.full(x.size, instance, dtype=np.uint32)
        data.PredSemantic = np.full(x.size, 2, dtype=np.uint8)
    data.write(path)


def run(root):
    root.mkdir(parents=True, exist_ok=False)
    predictions, targets, originals = [root / n for n in ("predictions", "targets", "originals")]
    for path in (predictions, targets, originals):
        path.mkdir()
    for i, shift in enumerate((0, 2, 4)):
        cloud(predictions / f"tile_{i}.las", np.arange(41) * .1 + shift,
              np.arange(26) * .1, instance=(i + 1) * 10)
        cloud(targets / f"tile_{i}.las", np.arange(400) * .01 + shift, np.arange(250) * .01)
    cloud(originals / "original.las", np.arange(800) * .01, np.arange(250) * .01)
    bounds = root / "bounds.json"
    bounds.write_text(json.dumps({"tile_buffer": .1, "tiles": [
        {"bounds": [[shift - .1, shift + 4.1], [-.1, 2.6]], "col": i, "row": 0}
        for i, shift in enumerate((0, 2, 4))]}), encoding="utf-8")
    stop = threading.Event()
    process = psutil.Process()
    peak = [process.memory_info().rss]

    def sample():
        while not stop.wait(.05):
            peak[0] = max(peak[0], process.memory_info().rss)

    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    start = time.monotonic()
    try:
        report = merge_collections(collections=[predictions], target_dir=targets,
                                  output_tiles=root / "filtered", tile_bounds_json=bounds,
                                  originals=originals)
    finally:
        stop.set()
        monitor.join()
    before = laspy.read(originals / "original.las")
    after = laspy.read(root / "original_with_predictions/original.las")
    for name in before.points.array.dtype.names:
        np.testing.assert_array_equal(before.points.array[name], after.points.array[name])
    assert report["state"] == "validated"
    assert all(m["matched"] == m["total"] == 200_000 for m in report["original_coverage"])
    assert sum(t["matched"] for t in report["models"][0]["transfer"]) == 300_000
    result = {"fixture": "synthetic; not production datasets 3110/3111",
              "dense_input_points": 300_000, "original_points": 200_000,
              "wall_seconds": time.monotonic() - start,
              "sampled_pipeline_peak_rss_mib": peak[0] / 1024 ** 2,
              "original_coverage": report["original_coverage"],
              "deduplication": report["models"][0]["deduplication"]}
    (root / "smoke_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("original_coverage", "deduplication")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    run(parser.parse_args().output_root.resolve())
