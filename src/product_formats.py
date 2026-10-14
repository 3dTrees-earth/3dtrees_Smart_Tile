"""Output format selection: prod-merged products and subsampled tile outputs."""
from __future__ import annotations

from collections.abc import Iterable
from typing import List


def selector_tokens(value) -> Iterable[str]:
    """Yield selector tokens from CLI strings, Galaxy lists, or list-like strings."""
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        for item in value:
            yield from selector_tokens(item)
        return
    text = str(value or "")
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    for raw_token in text.split(","):
        yield raw_token.strip().strip("'\"")


def parse_merged_output_formats(value: str) -> List[str]:
    """Parse prod-merged output format selector into unique normalized formats."""
    aliases = {
        "las": "laz",
        "laz": "laz",
        ".laz": "laz",
        "copc": "copc.laz",
        "copc_laz": "copc.laz",
        "copc-laz": "copc.laz",
        "copc.laz": "copc.laz",
        ".copc.laz": "copc.laz",
        "ply": "ply",
        ".ply": "ply",
    }
    parsed = []
    seen = set()
    for raw_token in selector_tokens(value or "copc.laz"):
        token = raw_token.strip().lower()
        if not token:
            continue
        output_format = aliases.get(token)
        if output_format is None:
            raise ValueError(
                f"Unsupported merged output format '{raw_token}'. "
                "Use laz, copc.laz, or ply."
            )
        if output_format in seen:
            continue
        seen.add(output_format)
        parsed.append(output_format)
    if not parsed:
        raise ValueError("No merged output formats selected")
    return parsed


TILE_OUTPUT_FORMATS = ("laz", "copc.laz", "ply")


def parse_tile_outputs(value) -> List[tuple]:
    """Parse 'resolution:format' pairs into [(resolution_m, format)].

    Formats are laz, copc.laz (alias: copc) and ply. ``original`` selects all
    source points, represented internally by 0.0. Each resolution may appear once.
    """
    outputs = []
    for token in selector_tokens(value):
        if not token:
            continue
        resolution, separator, output_format = token.partition(":")
        output_format = {"copc": "copc.laz"}.get(output_format.strip().lower(), output_format.strip().lower())
        original = resolution.strip().lower() == "original"
        try:
            resolution = 0.0 if original else float(resolution)
        except ValueError:
            resolution = float("nan")
        if not separator or output_format not in TILE_OUTPUT_FORMATS or not (original or 0 < resolution < 1e6):
            raise ValueError(f"Tile output '{token}' must be 'resolution:format' with 'original' or a positive "
                             f"resolution in metres and one of {', '.join(TILE_OUTPUT_FORMATS)}")
        if any(resolution == seen for seen, _ in outputs):
            raise ValueError(f"Tile output '{token}': resolution {resolution} m is listed twice; "
                             "each resolution may appear once")
        outputs.append((resolution, output_format))
    return outputs
