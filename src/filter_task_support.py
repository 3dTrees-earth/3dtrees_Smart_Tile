from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple

import laspy


_TREE_FILE_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"^input_trees(?:_[^.]+)?\.txt$", re.IGNORECASE),
    re.compile(r"^input_trees_info(?:_[^.]+)?\.txt$", re.IGNORECASE),
    re.compile(r"^trees(?:_[^.]+)?\.txt$", re.IGNORECASE),
    re.compile(r"^trees_info(?:_[^.]+)?\.txt$", re.IGNORECASE),
    re.compile(r"^.+_trees(?:_[^.]+)?\.txt$", re.IGNORECASE),
    re.compile(r"^.+_trees_info(?:_[^.]+)?\.txt$", re.IGNORECASE),
)


def is_tree_sidecar_file(path: Path) -> bool:
    """Return True for tree sidecar text files."""
    path = Path(path)
    name = path.name
    if any(pattern.match(name) for pattern in _TREE_FILE_PATTERNS):
        return True
    return path.suffix.lower() == ".txt"


def classify_tree_sidecar_file(path: Path) -> Optional[str]:
    """Return the canonical output suffix for a supported tree sidecar text file."""
    if not is_tree_sidecar_file(path):
        return None
    lower_name = Path(path).name.lower()
    if "trees_info" in lower_name:
        return "_trees_info.txt"
    return "_trees.txt"


def validate_pointcloud_header(path: Path) -> Tuple[bool, Optional[str]]:
    """Return whether a LAS/LAZ/COPC header can be opened by laspy."""
    try:
        with laspy.open(str(path), laz_backend=laspy.LazBackend.LazrsParallel) as reader:
            _ = reader.header.point_count
        return True, None
    except Exception as exc:
        return False, str(exc)
