"""Keep RayCloudTools tree rows aligned with the output instance IDs."""
from __future__ import annotations

import json
import re
from pathlib import Path

from point_cloud_metadata import point_cloud_source_key
from raycloud_instance_ids import read_tile_namespace


_TREE_FILE = re.compile(r"^(?P<stem>.+?)_trees(?P<info>_info)?\.txt$", re.IGNORECASE)
_GRID_TILE = re.compile(r"c\d+_r\d+", re.IGNORECASE)
_RCT_SUFFIX = re.compile(r"_trees(_info|_mesh)?$")
# Tree tables by kind: {"trees": [...], "trees_info": [...]}. Filtered outputs
# are always named <tile>_filtered_<kind>.txt.
TABLE_KINDS = ("trees", "trees_info")


def tree_sidecars(collection: Path) -> dict[str, list[Path]]:
    """Find RCT tree and tree-info tables co-located with a prediction collection.

    They are named ``<tile>_trees.txt`` and ``<tile>_trees_info.txt``; returns
    the tables by kind, or ``{}`` when there are none.
    """
    collection = Path(collection)
    folder = collection if collection.is_dir() else collection.parent
    tables = {}
    for path in sorted(folder.glob("*.txt"), key=lambda p: p.name.lower()):
        match = _TREE_FILE.fullmatch(path.name)
        if match:
            tables.setdefault("trees_info" if match.group("info") else "trees", []).append(path)
    return tables


def folder_files(folder: Path | None) -> list[Path]:
    """Every regular, non-hidden file of a folder; the folder says what the files are."""
    if folder is None:
        return []
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"Folder not found: {folder}")
    return sorted((p for p in folder.iterdir() if p.is_file() and not p.name.startswith(".")),
                  key=lambda p: p.name.lower())


def rct_sidecar_folder(collection: Path) -> Path:
    """Folder holding a collection's RCT tables and meshes.

    Co-located sidecars win; otherwise the merge manifest's ``rct_tree_sidecars``
    link, or the sibling ``segmented_filtered/`` of manifests predating it.
    """
    collection = Path(collection)
    folder = collection if collection.is_dir() else collection.parent
    if tree_sidecars(folder):
        return folder
    manifest = folder / "smarttile_merge.json"
    if manifest.is_file():
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        if metadata.get("rct_tree_sidecars"):
            return folder / metadata["rct_tree_sidecars"]
        if "rct_instance_ids" in metadata:
            return folder.parent / "segmented_filtered"
    return folder


def _tile_key(path: Path) -> str:
    """The tile a prediction, table or mesh belongs to: its grid name, else its stem.

    The stem drops known extensions, an RCT ``_trees``/``_trees_info``/``_trees_mesh``
    suffix and then one SmartTile stage suffix.
    """
    match = _GRID_TILE.search(path.name)
    if match:
        return match.group().lower()
    stem = _RCT_SUFFIX.sub("", point_cloud_source_key(path))
    for suffix in ("_segmented_remapped", "_segmented", "_remapped", "_filtered"):
        if stem.endswith(suffix):
            return stem[:-len(suffix)]
    return stem


def pair_tree_sidecars(prediction_files: list[Path], sidecars: dict[str, list[Path]]) -> dict[int, dict[str, Path]]:
    """Require an unambiguous tree table of each kind for every RCT prediction tile."""
    result: dict[int, dict[str, Path]] = {i: {} for i in range(len(prediction_files))}
    by_key = {}
    for i, prediction in enumerate(prediction_files):
        key = _prediction_key(prediction)
        if key in by_key:
            raise ValueError(f"Ambiguous RayCloudTools prediction tile key {key}")
        by_key[key] = i
    for kind, paths in sidecars.items():
        for sidecar in paths:
            tile = by_key.get(_tile_key(sidecar), 0 if len(prediction_files) == 1 else None)
            if tile is None:
                raise ValueError(f"Cannot match RayCloudTools tree file {sidecar.name} to a prediction tile")
            if kind in result[tile]:
                raise ValueError(f"Duplicate RayCloudTools {kind} file for {prediction_files[tile].name}")
            result[tile][kind] = sidecar
    for tile, files in result.items():
        if set(files) != set(TABLE_KINDS):
            raise ValueError(f"Missing RayCloudTools tree or treeinfo file for {prediction_files[tile].name}")
    return result


def _prediction_key(path):
    info = read_tile_namespace(path)
    return _tile_key(Path(info["source_name"]) if info else path)


def read_tree_header(stream, source):
    description, heading = stream.readline(), stream.readline()
    if not description or not heading:
        raise ValueError(f"{source.name}: expected two header lines followed by tree rows")
    explicit = heading.split(",", 1)[0].strip().lower() == "predinstance"
    return description, heading, explicit


def tree_rows(stream, source, explicit):
    """Yield validated IDs and unchanged payloads, including legacy row IDs."""
    seen = set()
    for row_count, line in enumerate(stream, start=1):
        uid, payload = row_count, line
        if explicit:
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
        yield uid, payload


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
        for kind in TABLE_KINDS:
            source = files[kind]
            output = output_dir / f"{_prediction_key(sources[tile])}_filtered_{kind}.txt"
            with source.open("r", encoding="utf-8") as input_stream, output.open("w", encoding="utf-8") as output_stream:
                description, heading, explicit_ids = read_tree_header(input_stream, source)
                newline = "\r\n" if heading.endswith("\r\n") else "\n"
                output_stream.write(description)
                output_stream.write(("" if explicit_ids else "predinstance,") + heading.rstrip("\r\n") + newline)
                seen = set()
                for uid, payload in tree_rows(input_stream, source, explicit_ids):
                    seen.add(uid)
                    if uid in kept_ids:
                        output_id = id_mapping[(tile, uid)] if id_mapping is not None else uid
                        output_stream.write(f"{output_id},{payload}")
            table_ids[kind] = seen
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
