"""Give RCT trees a reversible tile namespace without changing membership."""
from __future__ import annotations

import json
from pathlib import Path

import laspy
import numpy as np
from laspy.vlrs.vlr import VLR

from bounded_point_index import MAX_BATCH_POINTS, coordinates
from merge_stages import copy_record, index_file, mapped_values, prediction_values
from point_cloud_metadata import copy_single_source_header, update_extra_dimensions, write_retained_evlrs


RCT_ID_STRIDE = 100_000
_USER_ID = "3DTrees"
_RECORD_ID = 24002
_UINT32_MAX = np.iinfo(np.uint32).max


def encode_instance_ids(tile_id, local_ids):
    """Encode positive local IDs; zero stays background and overflow fails."""
    if not isinstance(tile_id, int) or isinstance(tile_id, bool) or tile_id < 1:
        raise ValueError("RCT tile_id must be a positive integer")
    if tile_id * RCT_ID_STRIDE > _UINT32_MAX:
        raise ValueError("RCT tile namespace exceeds uint32")
    values = np.asarray(local_ids)
    if (np.any(~np.isfinite(values)) or np.any(values < 0) or
            np.any(values != np.floor(values)) or np.any(values >= RCT_ID_STRIDE)):
        raise ValueError("RCT local instance IDs must be integers from 0 to 99999 for a 100000 tile stride")
    values = values.astype(np.uint64)
    encoded = np.where(values > 0, values + tile_id * RCT_ID_STRIDE, 0)
    if np.any(encoded > _UINT32_MAX):
        raise ValueError("RCT encoded instance ID exceeds uint32")
    return encoded.astype(np.uint32)


def read_tile_namespace(path):
    """Explicit metadata distinguishes encoded IDs from large raw local IDs."""
    with laspy.open(path, read_evlrs=False) as reader:
        records = [v for v in reader.header.vlrs
                   if v.user_id == _USER_ID and v.record_id == _RECORD_ID]
    if not records:
        return None
    if len(records) != 1:
        raise ValueError(f"{path}: duplicate RCT tile namespace metadata")
    info = json.loads(records[0].record_data_bytes())
    if info.get("version") != 1 or info.get("stride") != RCT_ID_STRIDE:
        raise ValueError(f"{path}: unsupported RCT tile namespace")
    encode_instance_ids(info.get("tile_id"), [])
    if not isinstance(info.get("source_name"), str) or not info["source_name"]:
        raise ValueError(f"{path}: RCT tile namespace lacks a source name")
    return info


def validate_namespaced_labels(labels, info):
    positive = np.asarray(labels)[np.asarray(labels) > 0].astype(np.uint64)
    if np.any(positive // RCT_ID_STRIDE != info["tile_id"]) or np.any(positive % RCT_ID_STRIDE == 0):
        raise ValueError("RCT instance IDs disagree with their tile namespace")


def namespace_rct_tiles(model, files, pairs, regions, ownership, index, origin, output_dir, *,
                        instance_statistics=None):
    """Publish one consistent ID mapping to points, table callers and manifests.

    ``instance_statistics`` optionally collects per-tree summaries while writing.
    """
    mapping, identities = {}, []
    seen_tiles = set()
    for tile, ((source, _, _), region) in enumerate(zip(pairs, regions)):
        existing = read_tile_namespace(source)
        info = existing or {
            "version": 1, "stride": RCT_ID_STRIDE,
            "tile_id": (region["layout_tile"] + 1 if region["layout_tile"] is not None else 1),
            "source_name": Path(source).name,
        }
        encode_instance_ids(info["tile_id"], [])
        if info["tile_id"] in seen_tiles:
            raise ValueError("Duplicate RCT tile namespace in prediction collection")
        seen_tiles.add(info["tile_id"])
        identities.append(info)
        decisions = ownership["tiles"][tile]["instances"]
        ids = np.array([row["instance"] for row in decisions], dtype=np.uint64)
        if existing:
            validate_namespaced_labels(ids, info)
            encoded = ids
        else:
            encoded = encode_instance_ids(info["tile_id"], ids)
        mapping.update({(tile, int(old)): int(new)
                        for row, old, new in zip(decisions, ids, encoded) if row["kept"]})

    # Every non-background tile namespace exceeds uint16, including tiny trees.
    param = laspy.ExtraBytesParams(name=model.instance, type=np.uint32,
                                 description="RCT tile*100000 + local ID")
    model.dimensions[model.instance] = param
    model.no_data[model.instance] = None
    index.dimensions = {name: p.type for name, p in model.dimensions.items()}
    output_dir.mkdir(parents=True)
    outputs = []
    for tile, (file, info) in enumerate(zip(files, identities)):
        output = output_dir / file.name
        with laspy.open(file) as reader:
            header = copy_single_source_header(reader.header)
            update_extra_dimensions(header, [param], replace=True)
            header.vlrs[:] = [v for v in header.vlrs
                             if not (v.user_id == _USER_ID and v.record_id == _RECORD_ID)]
            header.vlrs.append(VLR(user_id=_USER_ID, record_id=_RECORD_ID,
                                   description="RCT tile namespace v1",
                                   record_data=json.dumps(info, sort_keys=True).encode()))
            with laspy.open(output, mode="w", header=header) as writer:
                for record in reader.chunk_iterator(MAX_BATCH_POINTS):
                    out = copy_record(record, header)
                    values = mapped_values(model, prediction_values(record, model.dimensions), tile, mapping)
                    out[model.instance] = values[model.instance]
                    writer.write_points(out)
                    if instance_statistics is not None:
                        instance_statistics.add(values[model.instance], coordinates(record, reader.header, origin))
                write_retained_evlrs(writer, header)
        index_file(index, output, tile, origin, model)
        outputs.append(output)
    metadata = {"stride": RCT_ID_STRIDE, "background": 0,
                "encoding": "tile_id * 100000 + local_id",
                "tiles": [dict(info, tile=tile, file=files[tile].name)
                          for tile, info in enumerate(identities)]}
    return outputs, mapping, metadata


def validate_rct_remap_sources(files):
    """Check collection headers; return namespaces for validation while indexing."""
    identities = [read_tile_namespace(path) for path in files]
    if len(files) > 1 and any(info is None for info in identities):
        raise ValueError("Multiple RCT tiles require encoded tile namespaces; rerun merge/filter with the tree files")
    tile_ids = [info["tile_id"] for info in identities if info is not None]
    if len(tile_ids) != len(set(tile_ids)):
        raise ValueError("Duplicate RCT tile namespace in prediction collection")
    return identities
