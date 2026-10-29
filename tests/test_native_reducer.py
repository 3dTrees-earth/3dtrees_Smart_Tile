"""The optional Rust reducer returns exactly what the NumPy reducer returns."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import tile_centroids  # noqa: E402
from tile_centroids import RECORD, reduce_records  # noqa: E402

pytest.importorskip("smarttile_native")


def records(xyz, masks):
    out = np.empty(len(xyz), dtype=RECORD)
    out["X"], out["Y"], out["Z"] = xyz.T
    out["m"] = masks
    return out


def both(monkeypatch, rec, resolutions, scales, offsets, nearest):
    monkeypatch.setenv("SMARTTILE_REDUCER", "numpy")
    expected = reduce_records(rec, resolutions, scales, offsets, nearest)
    monkeypatch.setenv("SMARTTILE_REDUCER", "native")
    with monkeypatch.context() as patch:  # prove the extension, not a fallback, answered
        patch.setattr(tile_centroids, "_numpy_points", lambda *args: pytest.fail("NumPy fallback used"))
        return expected, reduce_records(rec, resolutions, scales, offsets, nearest)


def assert_identical(expected, actual):
    assert len(expected) == len(actual)
    for (want, want_used), (got, got_used) in zip(expected, actual):
        assert want_used == got_used
        assert want.dtype == got.dtype and want.shape == got.shape
        assert want.tobytes() == got.tobytes()


@pytest.mark.parametrize("case", range(30))
@pytest.mark.parametrize("nearest", [False, True])
def test_random_encodings_masks_and_orders_match_numpy(monkeypatch, case, nearest):
    rng = np.random.default_rng(242 + case)
    scales = tuple(rng.choice([0.0001, 0.001, 0.01, 0.1], 3))
    offsets = tuple(rng.choice([-500000.1, -10.03, 0.0, 615000.0, 32704.23], 3))
    xyz = rng.integers(-1000, 1000, (3000, 3), dtype=np.int32)
    xyz[:100] = xyz[100:200]  # duplicates
    rec = records(xyz, rng.integers(0, 16, len(xyz), dtype=np.uint8))
    for permutation in (np.arange(len(rec)), rng.permutation(len(rec))):
        assert_identical(*both(monkeypatch, rec[permutation], (0.01, 0.07, 0.2, 0.0), scales, offsets, nearest))


@pytest.mark.parametrize("nearest", [False, True])
def test_decimal_faces_far_from_the_origin_match_numpy(monkeypatch, nearest):
    # Points exactly on 5 cm / 50 cm faces under a large projected offset.
    x = np.arange(-2000, 2000, 25, dtype=np.int32)
    xyz = np.stack(np.meshgrid(x, x[::7], np.arange(0, 1000, 125, dtype=np.int32)), axis=-1).reshape(-1, 3)
    rec = records(xyz, np.full(len(xyz), 3, dtype=np.uint8))
    for offsets in ((521948.1, 5485047.5, 312.33), (-0.0005, 1e-7, 0.0)):
        assert_identical(*both(monkeypatch, rec, (0.05, 0.5), (0.001, 0.001, 0.001), offsets, nearest))


def test_nearest_exact_distance_ties_match_numpy(monkeypatch):
    # Symmetric points around every mean: several equidistant candidates per voxel.
    base = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [2, 2, 0], [1, 1, 1], [1, 1, -1]], dtype=np.int32)
    xyz = np.concatenate([base + np.array([10 * i, 10 * j, 5], dtype=np.int32) for i in range(20) for j in range(20)])
    rec = records(xyz, np.full(len(xyz), 1, dtype=np.uint8))
    assert_identical(*both(monkeypatch, rec, (0.01,), (0.001, 0.003, 0.001), (0.0, 0.0, 0.0), True))


def test_keys_outside_int64_are_rejected_either_way(monkeypatch):
    rec = records(np.array([[1, 2, 3]], dtype=np.int32), np.array([1], dtype=np.uint8))
    for choice in ("numpy", "native"):
        monkeypatch.setenv("SMARTTILE_REDUCER", choice)
        with pytest.raises(ValueError, match="int64"):
            reduce_records(rec, (0.1,), (0.001, 0.001, 0.001), (1e30, 0.0, 0.0))


def test_encodings_beyond_native_widths_fall_back_to_numpy(monkeypatch):
    # Offsets far beyond int128 voxel arithmetic are declined by the extension.
    rec = records(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32), np.array([1, 1], dtype=np.uint8))
    for choice in ("numpy", "native"):
        monkeypatch.setenv("SMARTTILE_REDUCER", choice)
        with pytest.raises(ValueError, match="int64"):
            reduce_records(rec, (0.1,), (0.001, 0.001, 0.001), (1e300, 0.0, 0.0))


def test_unknown_reducer_choice_is_rejected(monkeypatch):
    rec = records(np.array([[1, 2, 3]], dtype=np.int32), np.array([1], dtype=np.uint8))
    monkeypatch.setenv("SMARTTILE_REDUCER", "rust")
    with pytest.raises(ValueError, match="SMARTTILE_REDUCER"):
        reduce_records(rec, (0.1,), (0.001, 0.001, 0.001), (0.0, 0.0, 0.0))
