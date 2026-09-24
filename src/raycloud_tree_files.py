"""Keep RayCloudTools tree rows aligned with the output instance IDs."""
from __future__ import annotations

import re
from pathlib import Path

from point_cloud_metadata import point_cloud_source_key
from raycloud_instance_ids import read_tile_namespace


_TREE_FILE = re.compile(r"^(?P<stem>.+?)_trees(?P<info>_info)?\.txt$", re.IGNORECASE)
_GRID_TILE = re.compile(r"c\d+_r\d+", re.IGNORECASE)


def tree_sidecars(collection: Path) -> list[Path]:
    """Find RCT tree and tree-info tables co-located with a prediction collection."""
    collection = Path(collection)
    folder = collection if collection.is_dir() else collection.parent
    return sorted((p for p in folder.glob("*.txt") if _TREE_FILE.fullmatch(p.name)),
                  key=lambda p: p.name.lower())


def _tile_key(path: Path) -> str:
    match = _GRID_TILE.search(path.name)
    if match:
        return match.group().lower()
    stem = point_cloud_source_key(path)
    for suffix in ("_segmented_remapped", "_segmented", "_remapped", "_filtered"):
        if stem.endswith(suffix):
            return stem[:-len(suffix)]
    return stem


def pair_tree_sidecars(prediction_files: list[Path], sidecars: list[Path]) -> dict[int, list[Path]]:
    """Require an unambiguous tree table for every RCT prediction tile."""
    result: dict[int, list[Path]] = {i: [] for i in range(len(prediction_files))}
    by_key = {}
    for i, prediction in enumerate(prediction_files):
        key = _prediction_key(prediction)
        if key in by_key:
            raise ValueError(f"Ambiguous RayCloudTools prediction tile key {key}")
        by_key[key] = i
    for sidecar in sidecars:
        match = _TREE_FILE.fullmatch(sidecar.name)
        if match is None:
            continue
        key = _tile_key(Path(match.group("stem")))
        if key not in by_key:
            if len(prediction_files) != 1:
                raise ValueError(f"Cannot match RayCloudTools tree file {sidecar.name} to a prediction tile")
            tile = 0
        else:
            tile = by_key[key]
        suffix = "_trees_info.txt" if match.group("info") else "_trees.txt"
        if any(other.name.lower().endswith(suffix) for other in result[tile]):
            raise ValueError(f"Duplicate RayCloudTools {suffix} file for {prediction_files[tile].name}")
        result[tile].append(sidecar)
    for tile, files in result.items():
        kinds = {"_trees_info.txt" if p.name.lower().endswith("_trees_info.txt") else "_trees.txt"
                 for p in files}
        if kinds != {"_trees.txt", "_trees_info.txt"}:
            raise ValueError(f"Missing RayCloudTools tree or treeinfo file for {prediction_files[tile].name}")
    return result


def _prediction_key(path):
    info = read_tile_namespace(path)
    return _tile_key(Path(info["source_name"]) if info else path)


def filter_tree_sidecars(pairs, sidecars, ownership, output_dir: Path, *, id_mapping=None):
    """Filter implicit or explicit tree IDs and apply the LAZ's ID mapping."""
    sources = [Path(source) for source, _, _ in pairs]
    paired = pair_tree_sidecars(sources, sidecars)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True)
    results = []
    for tile, files in paired.items():
        decisions = ownership["tiles"][tile]["instances"]
        all_ids = {int(row["instance"]) for row in decisions}
        kept_ids = {int(row["instance"]) for row in decisions if row["kept"]}
        table_ids = {}
        for source in files:
            suffix = "_trees_info.txt" if source.name.lower().endswith("_trees_info.txt") else "_trees.txt"
            stem = _prediction_key(sources[tile])
            output = output_dir / f"{stem}_filtered{suffix}"
            with source.open("r", encoding="utf-8") as input_stream, output.open("w", encoding="utf-8") as output_stream:
                description = input_stream.readline()
                heading = input_stream.readline()
                if not description or not heading:
                    raise ValueError(f"{source.name}: expected two header lines followed by tree rows")
                newline = "\r\n" if heading.endswith("\r\n") else "\n"
                explicit_ids = heading.split(",", 1)[0].strip().lower() == "predinstance"
                output_stream.write(description)
                output_stream.write(("" if explicit_ids else "predinstance,") + heading.rstrip("\r\n") + newline)
                seen = set()
                for row_count, line in enumerate(input_stream, start=1):
                    uid, payload = row_count, line
                    if explicit_ids:
                        token, separator, payload = line.partition(",")
                        if not separator:
                            raise ValueError(f"{source.name}: missing explicit predinstance at row {row_count}")
                        try:
                            uid = int(token)
                        except ValueError:
                            raise ValueError(f"{source.name}: invalid predinstance at row {row_count}") from None
                    if uid <= 0 or uid > 2**32 - 1 or uid in seen:
                        raise ValueError(f"{source.name}: invalid or duplicate predinstance {uid}")
                    seen.add(uid)
                    if uid in kept_ids:
                        output_id = id_mapping[(tile, uid)] if id_mapping is not None else uid
                        output_stream.write(f"{output_id},{payload}")
            table_ids[source.name] = seen
            missing = all_ids - seen
            if missing:
                raise ValueError(f"{source.name}: instance ID {min(missing)} has no tree row")
            results.append({"tile": tile, "source": str(source), "output": output.name,
                            "kept_instance_ids": sorted(id_mapping[(tile, uid)] if id_mapping is not None else uid
                                                        for uid in kept_ids),
                            "kept_source_instance_ids": sorted(kept_ids),
                            "removed_instance_ids": sorted(all_ids - kept_ids)})
        if len({len(ids) for ids in table_ids.values()}) != 1:
            raise ValueError(f"RayCloudTools tree and treeinfo row counts differ for {sources[tile].name}")
        if len({frozenset(ids) for ids in table_ids.values()}) != 1:
            raise ValueError(f"RayCloudTools tree and treeinfo instance IDs differ for {sources[tile].name}")
    return results
