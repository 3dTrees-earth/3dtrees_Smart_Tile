"""The tile task publishes geometry, layouts and both tile modes as one job."""
import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.modules.setdefault("plot_tiles_and_copc", types.SimpleNamespace(plot_extents=lambda *a, **k: None))
import main_tile
import instance_tiles
import run
import tile_centroids
from parameters import Parameters


def setup_job(tmp_path, monkeypatch):
    source, output = tmp_path / "originals", tmp_path / "products"
    source.mkdir()
    (source / "source.las").write_bytes(b"untouched source")

    def plan(inputs, work, *args, **kwargs):
        tindex = work / "tindex_10m.gpkg"
        tindex.write_bytes(b"index")
        (work / "tindex.gpkg").symlink_to(tindex.name)
        bounds = work / "tile_bounds_tindex.json"
        bounds.write_text(json.dumps({"tiles": [], "tindex": str(tindex)}))
        return types.SimpleNamespace(bounds_json=bounds)

    def subsample(plan, work, *args, **kwargs):
        tiles = work / "subsampled_res1"
        tiles.mkdir()
        (tiles / "tile.laz").write_bytes(b"prepared geometry")
        return [tiles]

    monkeypatch.setattr(main_tile, "plan_tiling", plan)
    monkeypatch.setattr(tile_centroids, "subsample_tiles_single_scan", subsample)
    params = Parameters(task="tile", input_dir=source, output_dir=output,
                        tile_outputs="0.1:laz", workers=1, _cli_parse_args=False)
    return params, source, output


@pytest.mark.parametrize("existing_empty", [False, True])
def test_later_tile_mode_failure_publishes_no_partial_job(tmp_path, monkeypatch, existing_empty):
    params, source, output = setup_job(tmp_path, monkeypatch)
    if existing_empty:
        output.mkdir()
    params = params.model_copy(update={"instance_tiles": "PredInstance_SAT"})

    def fail(*args, **kwargs):
        raise ValueError("instance schema mismatch")

    monkeypatch.setattr(instance_tiles, "instance_tiles_from_plan", fail)
    with pytest.raises(SystemExit):
        run.run_tile_task(params)
    assert not output.exists() or list(output.iterdir()) == []
    assert (source / "source.las").read_bytes() == b"untouched source"
    assert not list(tmp_path.glob(".smarttile-*"))


def test_completed_job_has_usable_published_layout_and_relative_symlink(tmp_path, monkeypatch):
    params, source, output = setup_job(tmp_path, monkeypatch)
    assert run.run_tile_task(params) == source
    assert (output / "subsampled_res1/tile.laz").read_bytes() == b"prepared geometry"
    layout = json.loads((output / "tile_bounds_tindex.json").read_text())
    assert Path(layout["tindex"]) == output / "tindex_10m.gpkg"
    assert Path(layout["tindex"]).read_bytes() == b"index"
    assert (output / "tindex.gpkg").read_bytes() == b"index"


def test_existing_layout_and_collection_are_rejected_before_planning(tmp_path, monkeypatch):
    params, _, output = setup_job(tmp_path, monkeypatch)
    output.mkdir()
    old = output / "tile_bounds_tindex.json"
    old.write_text("existing layout")

    def unexpected(*args, **kwargs):
        pytest.fail("planning started before the destination was checked")

    monkeypatch.setattr(main_tile, "plan_tiling", unexpected)
    with pytest.raises(SystemExit):
        run.run_tile_task(params)
    assert old.read_text() == "existing layout"
