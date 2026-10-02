"""Prod-merged output format selection shared by the CLI parameters and product creation."""
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
