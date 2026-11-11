"""Instance coloring IDs for viewer files.

Vendored from the 3Dtrees Potree tool 1.2.1 (`special_coloring.py`) so SmartTile's
visualization COPC carries the same `coloring_id_*` values and palette mapping as the
Potree octree: neighbouring instances get distinct palette entries, ground stays grey.
Keep both copies in step until the Potree step is retired.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np

INSTANCE_ATTRIBUTE = "PredInstance"
COLOR_ATTRIBUTE = "coloring_id"
COLOR_ATTRIBUTE_DTYPE = np.uint16
GROUND_INSTANCE_ID = 0
GROUND_COLORING_ID = 0
GROUND_COLOR_HEX = "#808080"
PALETTE_NAME = "candy"
N_COLORS = 10
N_NEIGHBORS = 10
RANDOM_SEED = 0
PALETTES = {
    "sky": ["#f2c13a", "#ef7239", "#f23184", "#955ae8", "#5c96f5"],
    "sea": ["#d9ed92", "#99d98c", "#52b69a", "#168aad", "#1e6091", "#154366", "#0C304C"],
    "cozy": ["#41764c", "#a7c957", "#eadbb3", "#d68b8b", "#bc4749"],
    "fairy": ["#cd92ef", "#ffadcb", "#fed2e2", "#b1e2fc", "#9bc1ff"],
    "winter": ["#007DA3", "#00afb9", "#fdfcdc", "#fdca9b", "#ee6258"],
    "rainbow": ["#f52e2e", "#f5982e", "#f5d42e", "#dcf636", "#acf636", "#36f6a6", "#36e9f6", "#3d90ee", "#7336f6", "#c236f6"],
    "pastel": ["#fcf7b7", "#fed9bb", "#ffbfc3", "#f7aae7", "#c4a6f5", "#90bbf8", "#7cd9f8", "#7beff9", "#85f9e0", "#a4fdac"],
    "candy": ["#9137ff", "#ff5ce4", "#ff4545", "#fee440", "#00bbf9", "#00f5bc"],
    "boring": ["#DEE2E6", "#191C1F"],
}
SRGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)
XYZ_TO_SRGB = np.array(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ],
    dtype=np.float64,
)
D65_WHITE = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)


@dataclass
class InstanceStats:
    count: int = 0
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_z: float = 0.0


def color_attribute_name_for_instance(attribute: str, index: int) -> str:
    suffix = attribute.rsplit("_", 1)[1] if "_" in attribute else ""
    suffix_key = re.sub(r"[^a-z0-9]+", "", suffix.lower())
    if attribute.lower() == "predinstance" or suffix_key == "sat":
        return "coloring_id_sat"
    if suffix_key in {"fm", "foma"}:
        return "coloring_id_fm"
    if suffix_key:
        return f"coloring_id_{suffix_key}"
    return f"coloring_id{index}"


def assign_coloring_ids(
    instance_stats: Mapping[int, InstanceStats],
    *,
    n_colors: int = N_COLORS,
    n_neighbors: int = N_NEIGHBORS,
    palette_name: str = PALETTE_NAME,
    seed: int = RANDOM_SEED,
    ground_instance_id: int = GROUND_INSTANCE_ID,
    ground_color: str = GROUND_COLOR_HEX,
) -> Tuple[Dict[int, int], Dict[int, Dict[str, object]]]:
    if n_colors < 1:
        raise ValueError("n_colors must be at least 1.")
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be at least 1.")
    palette_hex, palette_rgb, palette_lab = build_palette(palette_name, n_colors)
    coloring_id_to_color: Dict[int, Dict[str, object]] = {
        GROUND_COLORING_ID: {
            "hex": normalize_hex_color(ground_color),
            "rgb": [int(value) for value in np.rint(hex_to_rgb01(ground_color) * 255).astype(int)],
        }
    }
    for index, hex_color in enumerate(palette_hex, start=1):
        rgb = np.rint(palette_rgb[index - 1] * 255).astype(int)
        coloring_id_to_color[index] = {
            "hex": hex_color,
            "rgb": [int(rgb[0]), int(rgb[1]), int(rgb[2])],
        }

    valid_items = sorted(
        (instance_id, stat)
        for instance_id, stat in instance_stats.items()
        if not is_ground_instance_id(instance_id, ground_instance_id) and stat.count > 0
    )
    instance_to_coloring_id: Dict[int, int] = {
        int(instance_id): GROUND_COLORING_ID
        for instance_id, stat in instance_stats.items()
        if is_ground_instance_id(int(instance_id), ground_instance_id) and stat.count > 0
    }
    instance_to_coloring_id.setdefault(ground_instance_id, GROUND_COLORING_ID)
    if not valid_items:
        return instance_to_coloring_id, coloring_id_to_color

    instance_ids = np.array([instance_id for instance_id, _ in valid_items], dtype=np.int64)
    coords = np.array(
        [
            [stat.sum_x / stat.count, stat.sum_y / stat.count, stat.sum_z / stat.count]
            for _, stat in valid_items
        ],
        dtype=np.float64,
    )

    neighbor_query_count = min(n_neighbors + 1, len(instance_ids))
    indices, distances = nearest_neighbors(coords, neighbor_query_count)
    mean_distances = distances.mean(axis=1)
    sorted_order = np.lexsort((instance_ids, mean_distances))
    id_to_row = {int(instance_id): index for index, instance_id in enumerate(instance_ids)}
    color_distances = np.linalg.norm(
        palette_lab[:, None, :] - palette_lab[None, :, :],
        axis=2,
    )
    assigned = np.full(len(instance_ids), -1, dtype=np.int64)
    rng = np.random.default_rng(seed)

    for row in sorted_order:
        neighbor_colors = assigned[indices[row, 1 : n_neighbors + 1]]
        used = sorted(int(value) for value in np.unique(neighbor_colors) if value >= 0)
        if not used:
            new_color = int(rng.integers(0, n_colors))
        else:
            unused = [color for color in range(n_colors) if color not in used]
            if not unused:
                unused = list(range(n_colors))
            if len(unused) == 1:
                new_color = unused[0]
            else:
                dvals = color_distances[np.ix_(unused, used)].sum(axis=1)
                new_color = unused[int(np.argmax(dvals))]
        assigned[row] = new_color

    for instance_id in instance_ids:
        row = id_to_row[int(instance_id)]
        instance_to_coloring_id[int(instance_id)] = int(assigned[row]) + 1

    return instance_to_coloring_id, coloring_id_to_color


def nearest_neighbors(coords: np.ndarray, n_neighbors: int) -> Tuple[np.ndarray, np.ndarray]:
    # The Potree tool uses sklearn's exact kNN; SmartTile ships scipy, whose KD-tree gives the
    # same neighbours (order may differ only for equal distances).
    from scipy.spatial import cKDTree

    distances, indices = cKDTree(coords).query(coords, k=n_neighbors)
    return indices.reshape(len(coords), -1), distances.reshape(len(coords), -1)


def map_instances_to_coloring_ids(
    raw_instance_ids: object,
    mapping_keys: np.ndarray,
    mapping_values: np.ndarray,
    *,
    instance_attribute: str = INSTANCE_ATTRIBUTE,
    ground_instance_id: int = GROUND_INSTANCE_ID,
) -> np.ndarray:
    ids = _coerce_instance_ids(raw_instance_ids)
    coloring_ids = np.zeros(ids.shape[0], dtype=COLOR_ATTRIBUTE_DTYPE)
    if mapping_keys.size == 0 or ids.size == 0:
        return coloring_ids

    positions = np.searchsorted(mapping_keys, ids)
    in_range = positions < mapping_keys.size
    matches = np.zeros(ids.shape[0], dtype=bool)
    matches[in_range] = mapping_keys[positions[in_range]] == ids[in_range]
    coloring_ids[matches] = mapping_values[positions[matches]]

    unmatched = (~matches) & (~is_ground_instance_ids(ids, ground_instance_id))
    if np.any(unmatched):
        missing = np.unique(ids[unmatched])[:10]
        raise ValueError(f"Missing coloring_id assignments for {instance_attribute} values: {missing.tolist()}")

    return coloring_ids


def is_ground_instance_id(instance_id: int, ground_instance_id: int = GROUND_INSTANCE_ID) -> bool:
    return instance_id == ground_instance_id or instance_id < 0


def is_ground_instance_ids(ids: np.ndarray, ground_instance_id: int = GROUND_INSTANCE_ID) -> np.ndarray:
    return (ids == ground_instance_id) | (ids < 0)


def build_palette(
    palette_name: str,
    n_colors: int,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    base_palette = resolve_palette(palette_name)
    base_rgb = np.array([hex_to_rgb01(color) for color in base_palette], dtype=np.float64)
    base_lab = xyz_to_lab(srgb_to_xyz(base_rgb))
    if n_colors <= len(base_palette):
        rgb = base_rgb[:n_colors]
        lab = base_lab[:n_colors]
    else:
        base_positions = np.linspace(0.0, 1.0, len(base_palette))
        target_positions = np.linspace(0.0, 1.0, n_colors)
        lab = np.column_stack(
            [
                np.interp(target_positions, base_positions, base_lab[:, channel])
                for channel in range(3)
            ]
        )
        rgb = np.clip(xyz_to_srgb(lab_to_xyz(lab)), 0.0, 1.0)
    return [rgb01_to_hex(color) for color in rgb], rgb, lab


def resolve_palette(palette_name: str) -> List[str]:
    key = palette_name.strip().lower()
    if key in PALETTES:
        return PALETTES[key]
    colors = [part.strip() for part in palette_name.split(",") if part.strip()]
    if colors:
        return [normalize_hex_color(color) for color in colors]
    raise ValueError(
        f"Unknown special coloring palette {palette_name!r}. "
        f"Expected one of {sorted(PALETTES)} or a comma-separated #RRGGBB list."
    )


def normalize_hex_color(value: str) -> str:
    raw = value.strip().replace("__pd__", "#")
    if not raw.startswith("#"):
        raw = f"#{raw}"
    hex_part = raw[1:]
    if len(hex_part) != 6:
        raise ValueError(f"Expected #RRGGBB color, got {value!r}")
    try:
        int(hex_part, 16)
    except ValueError as exc:
        raise ValueError(f"Expected #RRGGBB color, got {value!r}") from exc
    return f"#{hex_part.lower()}"


def hex_to_rgb01(value: str) -> np.ndarray:
    raw = normalize_hex_color(value).lstrip("#")
    return np.array(
        [int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16)],
        dtype=np.float64,
    ) / 255.0


def rgb01_to_hex(rgb: Iterable[float]) -> str:
    values = np.rint(np.clip(np.array(list(rgb), dtype=np.float64), 0.0, 1.0) * 255).astype(int)
    return f"#{values[0]:02x}{values[1]:02x}{values[2]:02x}"


def srgb_to_xyz(rgb: np.ndarray) -> np.ndarray:
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return linear @ SRGB_TO_XYZ.T


def xyz_to_srgb(xyz: np.ndarray) -> np.ndarray:
    linear = xyz @ XYZ_TO_SRGB.T
    linear = np.clip(linear, 0.0, None)
    return np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * (linear ** (1 / 2.4)) - 0.055)


def xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    scaled = xyz / D65_WHITE
    delta = 6 / 29
    transformed = np.where(
        scaled > delta**3,
        np.cbrt(scaled),
        scaled / (3 * delta**2) + 4 / 29,
    )
    return np.column_stack(
        [
            116 * transformed[:, 1] - 16,
            500 * (transformed[:, 0] - transformed[:, 1]),
            200 * (transformed[:, 1] - transformed[:, 2]),
        ]
    )


def lab_to_xyz(lab: np.ndarray) -> np.ndarray:
    fy = (lab[:, 0] + 16) / 116
    fx = fy + lab[:, 1] / 500
    fz = fy - lab[:, 2] / 200
    f = np.column_stack([fx, fy, fz])
    delta = 6 / 29
    scaled = np.where(
        f > delta,
        f**3,
        3 * delta**2 * (f - 4 / 29),
    )
    return scaled * D65_WHITE


def _coerce_instance_ids(raw_ids: object) -> np.ndarray:
    ids = np.asarray(raw_ids)
    if ids.ndim > 1:
        ids = ids[:, 0]
    if ids.size == 0:
        return np.empty(0, dtype=np.int64)
    if np.issubdtype(ids.dtype, np.floating):
        finite = np.isfinite(ids)
        if not np.all(finite):
            raise ValueError("PredInstance contains non-finite values.")
        rounded = np.rint(ids)
        if not np.allclose(ids, rounded):
            raise ValueError("PredInstance contains non-integer values.")
        ids = rounded
    return ids.astype(np.int64, copy=False)

