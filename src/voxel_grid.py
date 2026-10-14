"""Absolute voxel keys from LAS integer coordinates, without float boundary drift."""
from fractions import Fraction
from functools import lru_cache
import math

import numpy as np


@lru_cache(maxsize=512)
def _coefficients(scale, offset, resolution):
    # LAS encodings and requested resolutions describe decimal lengths. Work
    # with those values, not their inexact binary products/quotients. In
    # particular, 500001.1 belongs to voxel 5000011 at a 0.1 m resolution.
    step = Fraction(str(scale)) / Fraction(str(resolution))
    origin = Fraction(str(offset)) / Fraction(str(resolution))
    denominator = math.lcm(step.denominator, origin.denominator)
    return (step.numerator * (denominator // step.denominator),
            origin.numerator * (denominator // origin.denominator), denominator)


def voxel_keys(raw_xyz, scales, offsets, resolution):
    """Floor exact scaled LAS coordinates onto an absolute, half-open grid.

    Accept an N-by-D integer array and D scales/offsets. Decimal metadata is
    interpreted exactly; no epsilon snaps genuine points below a boundary.
    Normal LAS encodings use vectorized int64 arithmetic. Unusual encodings
    that overflow intermediate int64 values use Python integers instead.
    Keys outside int64 are rejected rather than silently wrapping.
    """
    raw = np.asarray(raw_xyz)
    scales, offsets = tuple(map(float, scales)), tuple(map(float, offsets))
    resolution = float(resolution)
    if raw.ndim != 2 or raw.dtype.kind not in "iu":
        raise ValueError("Voxel coordinates must be a two-dimensional integer array")
    if len(scales) != raw.shape[1] or len(offsets) != raw.shape[1]:
        raise ValueError("Voxel scales and offsets must match the coordinate axes")
    if (not math.isfinite(resolution) or resolution <= 0 or
            any(not math.isfinite(s) or s <= 0 for s in scales) or
            any(not math.isfinite(o) for o in offsets)):
        raise ValueError("Voxel resolution/scales must be finite and positive; offsets must be finite")
    keys = np.empty(raw.shape, dtype=np.int64)
    if not len(raw):
        return keys
    lower, upper = -(2**63), 2**63 - 1
    for axis, (scale, offset) in enumerate(zip(scales, offsets)):
        a, b, denominator = _coefficients(scale, offset, resolution)
        values = raw[:, axis]
        lo, hi = int(values.min()), int(values.max())
        key_lo, key_hi = (lo * a + b) // denominator, (hi * a + b) // denominator
        if key_lo < lower or key_hi > upper:
            raise ValueError("Voxel keys exceed the int64 range")
        # Splitting off whole origin voxels keeps large geographic offsets
        # out of the multiplication, preserving the fast path for fine scales.
        origin, remainder = divmod(b, denominator)
        if (lower <= lo <= hi <= upper and a <= upper and denominator <= upper and
                lower <= origin <= upper and lower <= lo * a and hi * a <= upper - remainder):
            values = values.astype(np.int64, copy=False)
            keys[:, axis] = (values * a + remainder) // denominator + origin
        else:
            keys[:, axis] = np.fromiter(((int(v) * a + b) // denominator for v in values),
                                        dtype=np.int64, count=len(values))
    return keys
