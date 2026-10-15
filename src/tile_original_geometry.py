"""Preserve the source coordinate grid when retaining original tile points."""
from fractions import Fraction
from pathlib import Path

import laspy
import numpy as np


def original_coordinate_tiles(sources, tiles):
    """Align tile offsets to the input grid; original resolution must not move points."""
    from dataclasses import replace

    with laspy.open(sources[0]) as reader:
        origin = reader.header.offsets
    scales = np.asarray(tiles[0].scales)
    if any(not np.array_equal(tile.scales, scales) for tile in tiles):
        raise ValueError("Original-resolution tiles require one coordinate scale")
    for source in sources:
        with laspy.open(source) as reader:
            ratio = reader.header.scales / scales
            phase = (reader.header.offsets - origin) / scales
        if (not np.allclose(ratio, np.rint(ratio), rtol=0, atol=1e-9)
                or not np.allclose(phase, np.rint(phase), rtol=0, atol=1e-6)):
            raise ValueError(f"{Path(source).name}: coordinate grid cannot be preserved at original resolution")
    return [replace(tile, offsets=tuple(_on_grid(o, offset, scale)
                                        for o, offset, scale in zip(origin, tile.offsets, scales)))
            for tile in tiles]


def _on_grid(origin, value, scale):
    """The decimal lattice point origin + k * scale nearest value, as the closest float.

    Float products such as 0 + 3270423 * 0.01 give 32704.229999999996; voxel
    keys read offsets as decimals, so faces would then floor into the lower voxel.
    """
    steps = round((float(value) - float(origin)) / float(scale))
    return float(Fraction(str(float(origin))) + steps * Fraction(str(float(scale))))
