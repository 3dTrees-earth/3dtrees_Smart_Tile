"""map_tiles keeps job order; a window bounds the results waiting to be consumed."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from parallel_tiles import map_tiles  # noqa: E402


def _leave_marker(tile, folder):
    (Path(folder) / f"{tile}.done").write_text("")
    return tile


def test_window_keeps_order_and_bounds_waiting_results(tmp_path):
    jobs = [(tile, tmp_path) for tile in range(9)]
    seen = []
    for tile in map_tiles(_leave_marker, jobs, processes=4, ahead=3):
        seen.append(tile)
        waiting = sorted(int(p.stem) for p in tmp_path.glob("*.done"))
        assert len(waiting) <= 3 and all(tile <= other < tile + 3 for other in waiting), (tile, waiting)
        (tmp_path / f"{tile}.done").unlink()
    assert seen == list(range(9))
    assert list(map_tiles(_leave_marker, jobs[:3], processes=1, ahead=2)) == [0, 1, 2]
