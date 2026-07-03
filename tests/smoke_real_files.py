#!/usr/bin/env python3
"""Real-file SmartTile smoke tests for the five public tasks.

This is intentionally separate from unittest discovery because it exercises
PDAL/Untwine-backed paths and should be run in the SmartTile Docker image.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import laspy


def _run(cmd: list[str], cwd: Path) -> None:
    print("+ " + " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def _copy_inputs(data_dir: Path, work_dir: Path, names: list[str]) -> dict[str, Path]:
    copied = {}
    for name in names:
        source = data_dir / name
        target = work_dir / name
        shutil.copy2(source, target)
        copied[name] = target
    return copied


def _extra_dims(path: Path) -> set[str]:
    with laspy.open(path) as reader:
        return set(reader.header.point_format.extra_dimension_names)


def _point_count(path: Path) -> int:
    with laspy.open(path) as reader:
        return int(reader.header.point_count)


def _assert_exists(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise AssertionError(f"Expected non-empty output: {path}")


def run_smoke(src_dir: Path, data_dir: Path, workers: int, keep_dir: Path | None = None) -> Path:
    with tempfile.TemporaryDirectory(prefix="smarttile_real_smoke_") as tmp:
        root = Path(tmp)
        inputs = root / "inputs"
        inputs.mkdir()
        files = _copy_inputs(
            data_dir,
            inputs,
            [
                "mikro.laz",
                "mikro.copc.laz",
                "mikro_segmented.laz",
                "mikro_res1.laz",
                "mikro_prediction_collection.laz",
                "mikro_prediction_collection_fm.laz",
                "tile_bounds_tindex.json",
            ],
        )
        run_py = src_dir / "run.py"

        tile_out = root / "tile"
        tile_in = root / "tile_input"
        tile_in.mkdir()
        shutil.copy2(files["mikro.laz"], tile_in / "mikro.laz")
        _run(
            [
                sys.executable,
                str(run_py),
                "--task",
                "tile",
                "--input-dir",
                str(tile_in),
                "--output-dir",
                str(tile_out),
                "--tiling-threshold",
                "10000",
                "--num-spatial-chunks",
                "2",
                "--workers",
                str(workers),
            ],
            src_dir,
        )
        res1 = next((tile_out / "subsampled_res1").glob("*.la[sz]"))
        res2 = next((tile_out / "subsampled_res2").glob("*.la[sz]"))
        if _extra_dims(res1) or _extra_dims(res2):
            raise AssertionError("Tile/subsample outputs must not carry extra dimensions")

        filter_in = root / "filter_input"
        filter_out = root / "filter_out"
        filter_in.mkdir()
        shutil.copy2(files["mikro_segmented.laz"], filter_in / "mikro_segmented.laz")
        _run(
            [
                sys.executable,
                str(run_py),
                "--task",
                "filter",
                "--input-dir",
                str(filter_in),
                "--output-dir",
                str(filter_out),
                "--instance-dimension",
                "PredInstance",
            ],
            src_dir,
        )
        _assert_exists(next(filter_out.glob("*.laz")))

        merge_out = root / "merge"
        merge_segmented = root / "merge_segmented"
        merge_target = root / "merge_target"
        merge_originals = root / "merge_originals"
        merge_copc = root / "merge_copc"
        for directory in (merge_segmented, merge_target, merge_originals, merge_copc):
            directory.mkdir()
        shutil.copy2(files["mikro_segmented.laz"], merge_segmented / "mikro_segmented.laz")
        shutil.copy2(files["mikro_res1.laz"], merge_target / "mikro_res1.laz")
        shutil.copy2(files["mikro.laz"], merge_originals / "mikro.laz")
        shutil.copy2(files["mikro.copc.laz"], merge_copc / "mikro.copc.laz")
        _run(
            [
                sys.executable,
                str(run_py),
                "--task",
                "merge",
                "--subsampled-segmented-folder",
                str(merge_segmented),
                "--subsampled-target-folder",
                str(merge_target),
                "--tile-bounds-json",
                str(files["tile_bounds_tindex.json"]),
                "--output-dir",
                str(merge_out),
                "--output-merged-laz",
                str(merge_out / "merged.laz"),
                "--output-tiles-folder",
                str(merge_out / "output_tiles"),
                "--original-laz-input-dir",
                str(merge_originals),
                "--original-laz-output-dir",
                str(merge_out / "original_with_predictions"),
                "--original-copc-input-dir",
                str(merge_copc),
                "--merged-resolutions",
                "10cm",
                "--merged-output-formats",
                "laz",
                "--num-spatial-chunks",
                "2",
                "--workers",
                str(workers),
            ],
            src_dir,
        )
        merge_enriched = merge_out / "original_with_predictions" / "mikro.laz"
        _assert_exists(merge_enriched)
        if "PredInstance" not in _extra_dims(merge_enriched):
            raise AssertionError("Merge original enrichment missing PredInstance")
        _assert_exists(merge_out / "prod_merged_10cm.laz")

        remap_out = root / "remap"
        remap_originals = root / "remap_originals"
        remap_copc = root / "remap_copc"
        remap_originals.mkdir()
        remap_copc.mkdir()
        shutil.copy2(files["mikro.laz"], remap_originals / "mikro.laz")
        shutil.copy2(files["mikro.copc.laz"], remap_copc / "mikro.copc.laz")
        _run(
            [
                sys.executable,
                str(run_py),
                "--task",
                "remap",
                "--segmented-folders",
                f"{files['mikro_prediction_collection.laz']},{files['mikro_prediction_collection_fm.laz']}",
                "--remap-dims",
                "PredInstance_SAT,PredInstance_FM",
                "--original-laz-input-dir",
                str(remap_originals),
                "--original-copc-input-dir",
                str(remap_copc),
                "--original-laz-output-dir",
                str(remap_out / "original_with_predictions"),
                "--output-dir",
                str(remap_out),
                "--produce-merged-file",
                "--merged-resolutions",
                "10cm",
                "--merged-output-formats",
                "laz",
                "--num-spatial-chunks",
                "2",
                "--workers",
                str(workers),
            ],
            src_dir,
        )
        remap_enriched = remap_out / "original_with_predictions" / "mikro.laz"
        dims = _extra_dims(remap_enriched)
        for name in ("PredInstance_SAT", "PredInstance_FM"):
            if name not in dims:
                raise AssertionError(f"Two-collection remap missing {name}")
        _assert_exists(remap_out / "prod_merged_10cm.laz")

        create_out = root / "create_merged_file"
        _run(
            [
                sys.executable,
                str(run_py),
                "--task",
                "create_merged_file",
                "--original-with-predictions-dir",
                str(remap_out / "original_with_predictions"),
                "--output-dir",
                str(create_out),
                "--merged-resolutions",
                "10cm",
                "--merged-output-formats",
                "laz,copc.laz",
                "--num-spatial-chunks",
                "2",
                "--workers",
                str(workers),
            ],
            src_dir,
        )
        for output in (create_out / "prod_merged_10cm.laz", create_out / "prod_merged_10cm.copc.laz"):
            _assert_exists(output)
            dims = _extra_dims(output)
            for name in ("PredInstance_SAT", "PredInstance_FM"):
                if name not in dims:
                    raise AssertionError(f"create_merged_file output missing {name}: {output}")
            if _point_count(output) <= 0:
                raise AssertionError(f"create_merged_file output has no points: {output}")

        if keep_dir is None:
            keep_dir = Path(tempfile.mkdtemp(prefix="smarttile_real_smoke_pass_"))
        else:
            keep_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(root, keep_dir, dirs_exist_ok=True)
        print(f"Smoke outputs copied to {keep_dir}", flush=True)
        return keep_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dir", type=Path, default=Path(__file__).resolve().parents[1] / "src")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--keep-dir", type=Path, help="Directory where passing smoke outputs are copied.")
    args = parser.parse_args()
    run_smoke(args.src_dir, args.data_dir, args.workers, args.keep_dir)


if __name__ == "__main__":
    main()
