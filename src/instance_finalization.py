"""Dataset-wide final identity compaction, independently for each model."""
from __future__ import annotations

import json
import os
from pathlib import Path

import laspy
import numpy as np
from laspy.vlrs.vlr import VLR

from bounded_point_index import MAX_BATCH_POINTS
from dense_tile_merge import copy_record
from point_cloud_metadata import copy_single_source_header, update_extra_dimensions, write_retained_evlrs

MAPPING_FILE = "instance_mapping.json"


def _rewrite_cloud(file, mappings):
    temporary = file.with_name(f".compact-{file.name}")
    try:
        with laspy.open(file) as reader:
            header = copy_single_source_header(reader.header)
            update_extra_dimensions(header, [laspy.ExtraBytesParams(
                name=name, type=np.uint32, description="Dataset-wide compact instance ID")
                for name in mappings], replace=True)
            removed = {24004}
            if "PredInstance_RCT" in mappings:
                removed.update((24002, 24003))
            header.vlrs[:] = [v for v in header.vlrs
                             if not (v.user_id == "3DTrees" and v.record_id in removed)]
            header.vlrs.append(VLR(user_id="3DTrees", record_id=24004,
                description="Compact instance IDs v1", record_data=json.dumps({
                    "version": 1, "scope": "dataset_per_model", "background": 0,
                    "mapping": MAPPING_FILE, "dimensions": sorted(mappings),
                }).encode()))
            if "PredInstance_RCT" in mappings:
                header.vlrs.append(VLR(user_id="3DTrees", record_id=24003,
                    description="RCT compact IDs v1", record_data=json.dumps({
                        "version": 1, "scope": "dataset", "background": 0,
                        "mapping": "rct_instance_mapping.json",
                    }).encode()))
            with laspy.open(temporary, mode="w", header=header) as writer:
                for record in reader.chunk_iterator(MAX_BATCH_POINTS):
                    out = copy_record(record, header)
                    for name, source_ids in mappings.items():
                        values = np.asarray(record[name])
                        positive = values > 0
                        positions = np.searchsorted(source_ids, values[positive])
                        if (np.any(values < 0) or np.any(positions >= len(source_ids)) or
                                np.any(source_ids[positions] != values[positive])):
                            raise ValueError(f"{file.name}: {name} IDs changed after remapping")
                        compact = np.zeros(len(record), dtype=np.uint32)
                        compact[positive] = positions + 1
                        out[name] = compact
                    writer.write_points(out)
                write_retained_evlrs(writer, header)
        os.replace(temporary, file)
    finally:
        temporary.unlink(missing_ok=True)


def compact_originals(output_dir, ids_by_dimension):
    """Compact a private staging collection before publication in one cloud pass.

    The census is collected during enrichment; no extra source scan is needed.
    The same source label gets the same final label across all original files.
    """
    output_dir = Path(output_dir)
    mappings, metadata = {}, {"version": 1, "background": 0, "models": {}}
    files = set()
    for name, ids_by_file in sorted(ids_by_dimension.items()):
        ids = sorted({int(uid) for values in ids_by_file.values() for uid in values if uid > 0})
        if len(ids) > np.iinfo(np.uint32).max:
            raise ValueError(f"{name}: compact IDs exceed uint32")
        mappings[name] = np.asarray(ids, dtype=np.uint32)
        files.update(ids_by_file)
        metadata["models"][name] = {
            "scope": "dataset", "instance_count": len(ids),
            "instances": [{"source_instance_id": uid, "instance_id": i + 1}
                          for i, uid in enumerate(ids)],
        }
    if mappings:
        for name in sorted(files):
            _rewrite_cloud(output_dir / name, mappings)
        (output_dir / MAPPING_FILE).write_text(json.dumps(metadata, indent=2) + "\n")
    return {"mapping": MAPPING_FILE if mappings else None,
            "counts": {name: len(ids) for name, ids in mappings.items()}}
