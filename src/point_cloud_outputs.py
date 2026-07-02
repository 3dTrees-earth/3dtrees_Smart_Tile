#!/usr/bin/env python3
"""Point-cloud output writers for SmartTile merge/remap products.

This module owns the boundary where in-memory SmartTile arrays become LAS/LAZ
products. Keeping that logic here makes the metadata contract easier to audit:
outputs that still describe one original source preserve that source header as
far as LAS/COPC allows, while merged multi-source products preserve only
run-true CRS/projection metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import laspy
import numpy as np

from instance_labels import MERGED_OUTPUT_SCALES
from point_cloud_metadata import (
    copy_single_source_header,
    point_cloud_files,
    projection_metadata_vlrs,
)


def merged_product_header(
    merged_points: np.ndarray,
    original_input_dir: Optional[Path],
    original_tiles_dir: Path,
) -> laspy.LasHeader:
    """Build a merged product header from source metadata when it remains true."""
    offsets = np.min(merged_points, axis=0)
    source_files = point_cloud_files(original_input_dir) or point_cloud_files(original_tiles_dir)

    if len(source_files) == 1:
        with laspy.open(str(source_files[0]), laz_backend=laspy.LazBackend.LazrsParallel) as source:
            header = copy_single_source_header(
                source.header,
                offsets=offsets,
                scales=MERGED_OUTPUT_SCALES,
                preserve_extra_dimensions=False,
            )
        header.point_count = 0
        return header

    header = laspy.LasHeader(point_format=6, version="1.4")
    header.offsets = offsets
    header.scales = MERGED_OUTPUT_SCALES

    if source_files:
        with laspy.open(str(source_files[0]), laz_backend=laspy.LazBackend.LazrsParallel) as source:
            source_header = source.header
            header.global_encoding = source_header.global_encoding
            header.vlrs = projection_metadata_vlrs(source_header.vlrs)
            source_evlrs = getattr(source_header, "evlrs", None)
            if source_evlrs is not None:
                header.evlrs = projection_metadata_vlrs(source_evlrs)

    return header


def write_loaded_point_cloud(
    source_file: Path,
    output_file: Path,
    points: np.ndarray,
    all_dims: Dict[str, np.ndarray],
    source_indices: Optional[np.ndarray] = None,
) -> None:
    """Write loaded points/dimensions back to LAZ while preserving source metadata."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with laspy.open(str(source_file), laz_backend=laspy.LazBackend.LazrsParallel) as reader:
        header = copy_single_source_header(reader.header, preserve_extra_dimensions=True)
        source_las = reader.read()
        source_points = source_las.points

    output_las = laspy.LasData(header)
    if source_indices is not None:
        selected_source_points = source_points[np.asarray(source_indices)]
    elif len(source_points) == len(points):
        selected_source_points = source_points
    else:
        selected_source_points = None

    output_las.points = laspy.ScaleAwarePointRecord.zeros(len(points), header=header)
    if selected_source_points is not None:
        source_dim_names = set(source_points.point_format.dimension_names)
        for dim_name in output_las.point_format.dimension_names:
            if dim_name in {"X", "Y", "Z"} or dim_name not in source_dim_names:
                continue
            output_las.points[dim_name] = selected_source_points[dim_name]

    output_las.x = points[:, 0]
    output_las.y = points[:, 1]
    output_las.z = points[:, 2]

    output_dimension_names = set(output_las.point_format.dimension_names)
    output_dimension_names.update(dim.name for dim in output_las.point_format.extra_dimensions)
    for dim_name, values in all_dims.items():
        if dim_name in ("X", "Y", "Z"):
            continue
        if dim_name not in output_dimension_names:
            output_las.add_extra_dim(laspy.ExtraBytesParams(name=dim_name, type=values.dtype))
            output_dimension_names.add(dim_name)
        setattr(output_las, dim_name, values)

    output_las.write(
        str(output_file),
        do_compress=output_file.name.lower().endswith(".laz"),
        laz_backend=laspy.LazBackend.LazrsParallel,
    )
