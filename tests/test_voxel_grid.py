"""LAS voxel ownership against exact rational coordinates, including decimal faces."""
from fractions import Fraction
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voxel_grid import voxel_keys


def reference(raw, scales, offsets, resolution):
    """Independent scalar arithmetic: a voxel is [k*r, (k+1)*r)."""
    scale = [Fraction(str(value)) for value in scales]
    offset = [Fraction(str(value)) for value in offsets]
    step = Fraction(str(resolution))
    return np.asarray([[(int(row[axis]) * scale[axis] + offset[axis]) // step
                        for axis in range(3)] for row in raw], dtype=np.int64).reshape(-1, 3)


@pytest.mark.parametrize("centimetres", range(1, 101))
def test_every_integer_centimetre_resolution_has_exact_decimal_faces(centimetres):
    # Millimetre-encoded points immediately before, on, and after both signed
    # faces. Translate to UTM-sized offsets without changing their cell ownership.
    step_mm = centimetres * 10
    faces = np.array([-100000, -1, 0, 1, 100000], dtype=np.int64) * step_mm
    values = (faces[:, None] + [-1, 0, 1]).reshape(-1)
    raw = np.c_[values, values[::-1], -values]
    scales = (0.001,) * 3
    offsets = tuple((value // step_mm * step_mm) / 1000
                    for value in (500000000, 5400000000, -500000000))
    resolution = centimetres / 100
    actual = voxel_keys(raw, scales, offsets, resolution)
    np.testing.assert_array_equal(actual, reference(raw, scales, offsets, resolution))
    assert actual.dtype == np.int64
    np.testing.assert_array_equal(actual[1::3, 0] - actual[::3, 0], np.ones(5, dtype=int))
    np.testing.assert_array_equal(actual[2::3, 0], actual[1::3, 0])


@pytest.mark.parametrize("offsets", [(0.0,) * 3, (500000.0, 5400000.0, -500000.0),
                                      (1e12, -1e12, 1e9)])
def test_near_faces_are_not_snapped_to_a_neighbouring_voxel(offsets):
    raw = np.array([[99999999, -1, -100000001], [100000000, 0, -100000000],
                    [100000001, 1, -99999999]], dtype=np.int64)
    scales = (1e-9,) * 3
    actual = voxel_keys(raw, scales, offsets, 0.1)
    np.testing.assert_array_equal(actual, reference(raw, scales, offsets, 0.1))
    # Even when decoded float coordinates coincide at large offsets, the raw
    # LAS coordinates distinguish just below the face from the face itself.
    np.testing.assert_array_equal(actual[1] - actual[0], [1, 1, 1])
    np.testing.assert_array_equal(actual[2], actual[1])


def test_anisotropic_scales_and_fractional_offsets_match_rational_coordinates():
    raw = np.array([[-2147483648, 0, 2147483647], [-101, 205, -309],
                    [-1, 0, 1], [100001, -200002, 300003]], dtype=np.int64)
    scales = (0.003, 0.0007, 0.0000001)
    offsets = (500000.0002, 5400000.0003, -10.0000002)
    for resolution in (0.01, 0.03, 0.1, 0.99, 1.0):
        np.testing.assert_array_equal(voxel_keys(raw, scales, offsets, resolution),
                                      reference(raw, scales, offsets, resolution))


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint64])
def test_empty_input_preserves_xyz_shape_and_integer_key_type(dtype):
    result = voxel_keys(np.empty((0, 3), dtype=dtype), (0.001,) * 3, (500000.0,) * 3, 0.1)
    assert result.shape == (0, 3)
    assert result.dtype == np.int64


def test_integer_intermediate_overflow_uses_exact_fallback():
    # The decimal numerator cannot fit int64, while every resulting key can.
    raw = np.array([[np.iinfo(np.int64).min, 0, np.iinfo(np.int64).max],
                    [np.iinfo(np.int64).max, -1, np.iinfo(np.int64).min]], dtype=np.int64)
    scales = (1e-16, 0.000000000000003, 0.0000000000000007)
    offsets = (1000000000.0001, -1000000000.0003, 500000.0002)
    np.testing.assert_array_equal(voxel_keys(raw, scales, offsets, 0.03),
                                  reference(raw, scales, offsets, 0.03))


def test_unsigned_coordinates_above_int64_do_not_wrap():
    raw = np.array([[2**63 - 1, 2**63, 2**64 - 1]], dtype=np.uint64)
    np.testing.assert_array_equal(voxel_keys(raw, (1e-8,) * 3, (0.0,) * 3, 0.1),
                                  reference(raw, (1e-8,) * 3, (0.0,) * 3, 0.1))


@pytest.mark.parametrize("offset", [-1e20, 1e20])
def test_unrepresentable_voxel_keys_fail_instead_of_wrapping(offset):
    with pytest.raises(ValueError):
        voxel_keys(np.zeros((1, 3), dtype=np.int32), (0.001,) * 3, (offset,) * 3, 0.01)


@pytest.mark.parametrize("raw", [np.zeros(3, dtype=int), np.zeros((2, 2), dtype=int),
                                  np.zeros((1, 3, 1), dtype=int), np.zeros((1, 3)),
                                  np.zeros((1, 3), dtype=bool)])
def test_invalid_raw_coordinate_shape_or_type_is_rejected(raw):
    with pytest.raises(ValueError):
        voxel_keys(raw, (0.001,) * 3, (0.0,) * 3, 0.1)


@pytest.mark.parametrize("scales,offsets,resolution", [
    ((0.0, 0.001, 0.001), (0.0,) * 3, 0.1),
    ((-0.001, 0.001, 0.001), (0.0,) * 3, 0.1),
    ((np.nan, 0.001, 0.001), (0.0,) * 3, 0.1),
    ((np.inf, 0.001, 0.001), (0.0,) * 3, 0.1),
    ((0.001,) * 2, (0.0,) * 3, 0.1),
    ((0.001,) * 3, (0.0,) * 2, 0.1),
    ((0.001,) * 3, (np.nan, 0.0, 0.0), 0.1),
    ((0.001,) * 3, (0.0, np.inf, 0.0), 0.1),
    ((0.001,) * 3, (0.0,) * 3, 0.0),
    ((0.001,) * 3, (0.0,) * 3, -0.1),
    ((0.001,) * 3, (0.0,) * 3, np.nan),
    ((0.001,) * 3, (0.0,) * 3, np.inf),
])
def test_invalid_encoding_or_resolution_is_rejected(scales, offsets, resolution):
    with pytest.raises(ValueError):
        voxel_keys(np.zeros((1, 3), dtype=np.int32), scales, offsets, resolution)
