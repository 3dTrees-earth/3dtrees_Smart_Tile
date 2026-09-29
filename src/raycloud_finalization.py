"""Compact final RCT identities and export matching tables for each original."""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import laspy
import numpy as np
from laspy.vlrs.vlr import VLR

from bounded_point_index import MAX_BATCH_POINTS
from dense_tile_merge import copy_record
from point_cloud_metadata import copy_single_source_header, update_extra_dimensions, write_retained_evlrs
from raycloud_instance_ids import RCT_ID_STRIDE, read_tile_namespace, validate_namespaced_labels
from raycloud_tree_files import pair_tree_sidecars, read_tree_header, tree_rows, tree_sidecars


INSTANCE = "PredInstance_RCT"
MAPPING_FILE = "rct_instance_mapping.json"


def remap_tree_sidecars(collection):
    """Resolve co-located tables or the tables published with a merge result."""
    collection = Path(collection)
    files = tree_sidecars(collection)
    if files:
        return files
    folder = collection if collection.is_dir() else collection.parent
    manifest = folder / "smarttile_merge.json"
    if manifest.is_file():
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        location = metadata.get("rct_tree_sidecars")
        if location:
            return tree_sidecars(folder / location)
        # Compatibility with merge manifests predating the explicit link.
        if "rct_instance_ids" in metadata:
            return tree_sidecars(folder.parent / "segmented_filtered")
    return []


def _catalogue_tables(db, predictions, sidecars):
    """Keep large QSM payloads on disk; validate both tables before rewriting."""
    db.execute("CREATE TABLE trees (kind TEXT, id INTEGER, payload TEXT, PRIMARY KEY(kind, id))")
    headers, provenance = {}, {}
    for tile, files in pair_tree_sidecars(predictions, sidecars).items():
        namespace = read_tile_namespace(predictions[tile])
        table_ids = []
        for source in files:
            kind = "trees_info" if source.name.lower().endswith("_trees_info.txt") else "trees"
            with source.open(encoding="utf-8", newline="") as stream:
                description, heading, explicit = read_tree_header(stream, source)
                heading = (heading.split(",", 1)[1] if explicit else heading).rstrip("\r\n")
                if kind in headers and heading != headers[kind][1]:
                    raise ValueError(f"{source.name}: incompatible RCT {kind} table columns")
                headers.setdefault(kind, (description, heading))
                ids = set()
                for uid, payload in tree_rows(stream, source, explicit):
                    if namespace:
                        validate_namespaced_labels(np.array([uid], dtype=np.uint32), namespace)
                    ids.add(uid)
                    try:
                        db.execute("INSERT INTO trees VALUES (?, ?, ?)", (kind, uid, payload))
                    except sqlite3.IntegrityError:
                        raise ValueError(f"{source.name}: duplicate RCT source instance {uid}") from None
                    if kind == "trees":
                        provenance[uid] = {
                            "source_instance_id": uid,
                            "source_tile_id": namespace["tile_id"] if namespace else None,
                            "source_local_id": uid % RCT_ID_STRIDE if namespace else uid,
                            "source_file": namespace["source_name"] if namespace else predictions[tile].name,
                        }
                table_ids.append(ids)
        if table_ids[0] != table_ids[1]:
            raise ValueError(f"{predictions[tile].name}: RCT tree and treeinfo instance IDs differ")
    return headers, provenance


def _rewrite_cloud(file, source_ids):
    """Rewrite only the RCT ID field using bounded point batches."""
    temporary = file.with_name(f".compact-{file.name}")
    with laspy.open(file) as reader:
        header = copy_single_source_header(reader.header)
        update_extra_dimensions(header, [laspy.ExtraBytesParams(
            name=INSTANCE, type=np.uint32, description="Dataset-wide compact RCT ID")], replace=True)
        header.vlrs[:] = [v for v in header.vlrs
                         if not (v.user_id == "3DTrees" and v.record_id in (24002, 24003))]
        header.vlrs.append(VLR(user_id="3DTrees", record_id=24003,
            description="RCT compact IDs v1", record_data=json.dumps({
                "version": 1, "scope": "dataset", "background": 0,
                "mapping": MAPPING_FILE,
            }).encode()))
        with laspy.open(temporary, mode="w", header=header) as writer:
            for record in reader.chunk_iterator(MAX_BATCH_POINTS):
                values = np.asarray(record[INSTANCE])
                positive = values > 0
                positions = np.searchsorted(source_ids, values[positive])
                if (np.any(positions >= len(source_ids)) or
                        np.any(source_ids[positions] != values[positive])):
                    raise ValueError(f"{file.name}: RCT IDs changed after remapping")
                out = copy_record(record, header)
                compact = np.zeros(len(record), dtype=np.uint32)
                compact[positive] = positions + 1
                out[INSTANCE] = compact
                writer.write_points(out)
            write_retained_evlrs(writer, header)
    os.replace(temporary, file)


def finalize_rct_originals(output_dir, ids_by_file, predictions, sidecars, *, rewrite_clouds=True):
    """Finalize a private staging folder; callers publish it only on success.

    IDs are collected while remapping, avoiding a separate cloud census read.
    Intermediate tile namespaces and their sidecars remain reusable unchanged.
    """
    output_dir = Path(output_dir)
    stems = [Path(name).stem.casefold() for name in ids_by_file]
    if len(stems) != len(set(stems)):
        raise ValueError("RCT originals require distinct stems for per-original tree files")
    all_ids = sorted({uid for ids in ids_by_file.values() for uid in ids if uid > 0})
    if len(all_ids) > np.iinfo(np.uint32).max:
        raise ValueError("Compact RCT instance IDs exceed uint32")
    mapping = {uid: i + 1 for i, uid in enumerate(all_ids)}
    source_ids = np.asarray(all_ids, dtype=np.uint32)
    metadata = {"version": 1, "dimension": INSTANCE, "scope": "dataset", "background": 0,
                "tree_count": len(all_ids), "instances": [], "files": [],
                "qsm_quality": "not assessed; file boundaries do not measure model reliability"}
    # The database lives outside the published folder and bounds QSM payload RAM.
    database = output_dir.parent / "rct_final_tables.sqlite"
    db = sqlite3.connect(database)
    try:
        headers, provenance = _catalogue_tables(db, predictions, sidecars)
        missing = set(all_ids) - provenance.keys()
        if missing:
            raise ValueError(f"RCT instance ID {min(missing)} has no tree row")
        metadata["instances"] = [dict(predinstance=mapping[uid], **provenance[uid]) for uid in all_ids]
        for name, ids in sorted(ids_by_file.items()):
            file = output_dir / name
            kept = sorted(uid for uid in ids if uid > 0)
            entry = {"file": name, "tree_count": len(kept), "sidecars": {}}
            for kind, (description, heading) in headers.items():
                table = file.with_name(f"{file.stem}_{kind}.txt")
                with table.open("w", encoding="utf-8", newline="") as stream:
                    stream.write(description)
                    stream.write(f"predinstance,{heading}\n")
                    for uid in kept:
                        payload, = db.execute("SELECT payload FROM trees WHERE kind=? AND id=?", (kind, uid)).fetchone()
                        stream.write(f"{mapping[uid]},{payload}")
                        if not payload.endswith(("\n", "\r")):
                            stream.write("\n")
                entry["sidecars"][kind] = table.name
            if rewrite_clouds:
                _rewrite_cloud(file, source_ids)
            metadata["files"].append(entry)
        (output_dir / MAPPING_FILE).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    finally:
        db.close()
        database.unlink(missing_ok=True)
    return {"scope": "dataset", "tree_count": len(all_ids), "mapping": MAPPING_FILE,
            "files": metadata["files"]}
