"""Join several versions of the same originals, each with its own attributes.

Every collection holds the same original files (matched by name), for example
one remap output per segmentation run. A file must hold the same points in the
same record order in every collection; the joined file keeps the first
collection's points and standard fields and adds every collection's
ExtraBytes. An attribute in several collections is kept once when its values
agree and is an error otherwise (attributes are never renamed). Only LAS/LAZ
inputs are accepted: COPC conversion reorders points, so record order would
no longer identify a point.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import laspy
import numpy as np

from point_cloud_metadata import (copy_single_source_header, point_cloud_source_key, raw_point_cloud_files,
                                  write_retained_evlrs)

READ_POINTS = 1_000_000


def _extra_bytes_params(dimension):
    return laspy.ExtraBytesParams(name=dimension.name, type=dimension.dtype, description=dimension.description,
                                  offsets=dimension.offsets, scales=dimension.scales, no_data=dimension.no_data)


def _files_by_key(directory: Path) -> dict:
    files = raw_point_cloud_files(directory)
    if not files:
        raise ValueError(f"{directory}: no LAS/LAZ files (COPC inputs have lost their record order)")
    by_key = {point_cloud_source_key(path): path for path in files}
    if len(by_key) != len(files):
        raise ValueError(f"{directory}: several files share one original name")
    return by_key


def _joined_header(paths):
    headers = []
    for path in paths:
        with laspy.open(path) as reader:
            headers.append(reader.header)
    base = headers[0]
    for path, header in zip(paths[1:], headers[1:]):
        if header.point_count != base.point_count:
            raise ValueError(f"{path}: {header.point_count:,} points, {paths[0]} has {base.point_count:,}; "
                             "collections must hold the same points")
    dimensions = {d.name: d for d in base.point_format.extra_dimensions}
    sources = {name: [0] for name in dimensions}
    standard = set(base.point_format.standard_dimension_names)
    for index, header in enumerate(headers[1:], start=1):
        for dimension in header.point_format.extra_dimensions:
            if dimension.name in standard:
                continue
            if dimension.name in dimensions:
                if np.dtype(dimension.dtype) != np.dtype(dimensions[dimension.name].dtype):
                    raise ValueError(f"{dimension.name} has different types in {paths[0]} and {paths[index]}")
                sources[dimension.name].append(index)
                continue
            dimensions[dimension.name] = dimension
            sources[dimension.name] = [index]
    header = copy_single_source_header(base)
    names = [d.name for d in base.point_format.extra_dimensions]
    header.remove_extra_dims(names)
    for name, dimension in dimensions.items():
        header.add_extra_dim(_extra_bytes_params(dimension))
    return header, headers, sources


def _same_points(chunk, base_chunk, header, base_header):
    if np.array_equal(header.scales, base_header.scales) and np.array_equal(header.offsets, base_header.offsets):
        return all(np.array_equal(chunk[a], base_chunk[a]) for a in ("X", "Y", "Z"))
    tolerance = float(np.max([header.scales, base_header.scales])) / 2
    return all(np.abs(np.asarray(getattr(chunk, a)) - np.asarray(getattr(base_chunk, a))).max(initial=0) <= tolerance
               for a in ("x", "y", "z"))


def join_file(paths: List[Path], output: Path) -> int:
    """Write one original with the attributes of every collection; return its point count."""
    header, headers, sources = _joined_header(paths)
    standard = [n for n in header.point_format.standard_dimension_names if n not in ("X", "Y", "Z")]
    readers = [laspy.open(path) for path in paths]
    written = 0
    try:
        with laspy.open(output, mode="w", header=header, do_compress=True) as writer:
            for chunks in zip(*(reader.chunk_iterator(READ_POINTS) for reader in readers)):
                base = chunks[0]
                for path, chunk, source_header in zip(paths[1:], chunks[1:], headers[1:]):
                    if len(chunk) != len(base) or not _same_points(chunk, base, source_header, headers[0]):
                        raise ValueError(f"{path} and {paths[0]} differ in point coordinates or order "
                                         f"near record {written:,}; collections must hold the same points")
                record = laspy.ScaleAwarePointRecord.zeros(len(base), header=header)
                for axis in ("X", "Y", "Z"):
                    record[axis] = base[axis]
                for name in standard:
                    record[name] = base[name]
                for name, holders in sources.items():
                    values = np.asarray(chunks[holders[0]][name])
                    for other in holders[1:]:
                        if not np.array_equal(values, np.asarray(chunks[other][name])):
                            raise ValueError(f"{name} differs between {paths[holders[0]]} and {paths[other]}; "
                                             "rename it in one collection")
                    record[name] = values
                writer.write_points(record)
                written += len(base)
            write_retained_evlrs(writer, header)
    finally:
        for reader in readers:
            reader.close()
    if written != headers[0].point_count:
        raise RuntimeError(f"Joined {written:,} of {headers[0].point_count:,} points of {paths[0].name}")
    return written


def join_attribute_collections(directories: List[Path], output_dir: Path) -> Path:
    """Join every original across ``directories`` into ``output_dir``; return it."""
    if len(directories) < 2:
        raise ValueError("Joining needs at least two collections")
    collections = [_files_by_key(Path(d)) for d in directories]
    keys = sorted(collections[0])
    for directory, collection in zip(directories[1:], collections[1:]):
        if sorted(collection) != keys:
            raise ValueError(f"{directory} holds other originals than {directories[0]}: "
                             f"{sorted(set(keys) ^ set(collection))}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    for key in keys:
        paths = [collection[key] for collection in collections]
        count = join_file(paths, output_dir / f"{key}.laz")
        print(f"    Joined {key}: {count:,} points from {len(paths)} collections")
    return output_dir
