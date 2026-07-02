#!/usr/bin/env python3
"""COPC staging-cache freshness checks for prod-merged creation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from copc_metadata import laspy_laz_backend
from point_cloud_metadata import point_cloud_source_key


MANIFEST_FILENAME = ".smarttile_copc_manifest.json"


def _extra_dimension_signature(dim) -> Dict[str, object]:
    """Return the structural extra-dimension schema used for cache freshness.

    Descriptions are intentionally excluded. Some LAZ->COPC conversions drop or
    normalize descriptive text while preserving the actual point record schema.
    """
    return {
        "name": dim.name,
        "dtype": str(np.dtype(dim.dtype)),
    }


def point_cloud_header_signature(path: Path) -> Dict[str, object]:
    """Return cheap header metadata that should match across LAZ->COPC staging."""
    import laspy

    with laspy.open(str(path), laz_backend=laspy_laz_backend()) as reader:
        header = reader.header
        return {
            "point_count": int(header.point_count),
            "bounds": [
                float(header.x_min),
                float(header.x_max),
                float(header.y_min),
                float(header.y_max),
                float(header.z_min),
                float(header.z_max),
            ],
            "scales": [float(value) for value in header.scales],
            "offsets": [float(value) for value in header.offsets],
            "point_format": int(header.point_format.id),
            "version": str(header.version),
            "extra_dimensions": [
                _extra_dimension_signature(dim)
                for dim in header.point_format.extra_dimensions
            ],
        }


def source_file_fingerprint(source_file: Path) -> Dict[str, object]:
    """Return the source identity used to decide whether a staged COPC is fresh."""
    stat = source_file.stat()
    return {
        "name": source_file.name,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "header": point_cloud_header_signature(source_file),
    }


def _manifest_path(copc_dir: Path) -> Path:
    return copc_dir / MANIFEST_FILENAME


def read_copc_stage_manifest(copc_dir: Path) -> Dict[str, object]:
    manifest_file = _manifest_path(copc_dir)
    if not manifest_file.exists():
        return {"version": 1, "sources": {}}
    try:
        with manifest_file.open() as handle:
            manifest = json.load(handle)
    except Exception:
        return {"version": 1, "sources": {}}
    if not isinstance(manifest, dict):
        return {"version": 1, "sources": {}}
    manifest.setdefault("version", 1)
    manifest.setdefault("sources", {})
    return manifest


def write_copc_stage_manifest_entry(source_file: Path, copc_file: Path) -> None:
    """Record which source file produced a staged COPC."""
    copc_file.parent.mkdir(parents=True, exist_ok=True)
    manifest = read_copc_stage_manifest(copc_file.parent)
    sources = manifest.setdefault("sources", {})
    key = point_cloud_source_key(source_file)
    sources[key] = {
        "source": source_file_fingerprint(source_file),
        "copc_file": copc_file.name,
        "copc_header": point_cloud_header_signature(copc_file),
    }
    with _manifest_path(copc_file.parent).open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)


def staged_copc_matches_source(copc_file: Path, source_file: Optional[Path]) -> bool:
    """Return True when a staged COPC is compatible with source_file.

    A SmartTile manifest is preferred when available. Older validation runs may
    have reusable COPCs without a manifest, so fall back to comparing structural
    header metadata.
    """
    if source_file is None:
        return True

    manifest = read_copc_stage_manifest(copc_file.parent)
    entry = manifest.get("sources", {}).get(point_cloud_source_key(source_file))
    try:
        source_header = point_cloud_header_signature(source_file)
        copc_header = point_cloud_header_signature(copc_file)
        if not entry:
            return source_header == copc_header
        if entry.get("copc_file") != copc_file.name:
            return False
        return (
            entry.get("source") == source_file_fingerprint(source_file)
            and entry.get("copc_header") == copc_header
        )
    except Exception:
        return False
