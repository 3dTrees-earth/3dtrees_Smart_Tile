#!/usr/bin/env python3
"""
Tile planning for the tile task: index sources and write the tile layout.

1. Reject CRSs that cannot be tiled in metres
2. Build the spatial index (tindex) from input LAZ/LAS files
3. Calculate tile bounds and the tile job list
4. Write the tile overview

The tile task reads the points once in `tile_centroids`; no full-resolution
tiles are written.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import plot_tiles_and_copc

from point_cloud_metadata import point_cloud_files
from tile_crs import require_metric_tiling_crs
from tile_tindex import (
    build_tindex,
    calculate_tile_bounds,
    parse_proj_bounds as _parse_proj_bounds,
    write_single_cloud_bounds,
)



def _point_count(path: Path) -> int:
    import laspy

    with laspy.open(path) as reader:
        return int(reader.header.point_count)


def _make_tile_header(header_snapshot, offsets=None, scales=None):
    """Create a LasHeader from a source header, preserving source metadata.

    Args:
        header_snapshot: Source laspy header to copy from.
        offsets: Optional XYZ offsets. Defaults to source header offsets.
        scales: Optional XYZ scales. Defaults to source header scales.

    Returns:
        A new laspy.LasHeader ready for writing.
    """
    import laspy
    from laspy.vlrs.vlrlist import VLRList

    # Rebuild instead of using header_snapshot.copy(): laspy marks COPC headers
    # as non-writable even after stale COPC VLRs are removed.
    hdr = laspy.LasHeader(
        point_format=header_snapshot.point_format,
        version=header_snapshot.version,
    )
    hdr.point_count = 0
    hdr.offsets = offsets if offsets is not None else header_snapshot.offsets
    hdr.scales = scales if scales is not None else header_snapshot.scales

    for attr in (
        "file_source_id",
        "global_encoding",
        "uuid",
        "system_identifier",
        "generating_software",
        "creation_date",
    ):
        if hasattr(header_snapshot, attr):
            try:
                setattr(hdr, attr, getattr(header_snapshot, attr))
            except Exception:
                pass

    # COPC hierarchy/index records describe a specific COPC container layout.
    # Tile-part LAZ files and regenerated COPC outputs must not inherit stale ones.
    def is_stale_copc_vlr(vlr) -> bool:
        return getattr(vlr, "user_id", "") == "copc"

    hdr.vlrs = VLRList([
        vlr for vlr in getattr(header_snapshot, "vlrs", [])
        if not is_stale_copc_vlr(vlr)
    ])
    source_evlrs = getattr(header_snapshot, "evlrs", None)
    if source_evlrs is not None:
        hdr.evlrs = VLRList([vlr for vlr in source_evlrs if not is_stale_copc_vlr(vlr)])

    # Copy extra dimensions
    try:
        extra_dims_src = getattr(header_snapshot.point_format, "extra_dimensions", None)
        if extra_dims_src:
            extra_dims_dst = getattr(hdr.point_format, "extra_dimensions", None)
            existing = {d.name for d in (extra_dims_dst or [])}
            for dim in extra_dims_src:
                if dim.name not in existing:
                    hdr.add_extra_dim(laspy.ExtraBytesParams(
                        name=dim.name, type=dim.dtype,
                        description=getattr(dim, "description", "") or "",
                    ))
                    existing.add(dim.name)
    except Exception:
        pass

    return hdr


@dataclass(frozen=True)
class TilingPlan:
    """Inputs and layout for the single-scan tile subsampling."""
    source_files: List[Path]
    tindex_file: Path
    bounds_json: Path
    jobs_file: Path
    skip_tiling: bool


def plan_tiling(input_dir, output_dir, tile_length, tile_buffer, tiling_threshold=None,
                grid_origin=None) -> TilingPlan:
    """Index sources and plan the tile layout; no point output.

    A single input below `tiling_threshold` MB is planned as one whole-cloud
    tile without buffer, and its overview is written here. Otherwise the plan
    keeps every grid tile: the caller's single read decides occupancy,
    publishes it and calls `plot_overview`.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    tindex_file = output_dir / f"tindex_{int(tile_length)}m.gpkg"

    source_files = point_cloud_files(input_dir)
    if not source_files:
        raise ValueError(f"No LAZ/LAS files found in {input_dir}")
    empty = [path.name for path in source_files if _point_count(path) == 0]
    if empty:
        # An empty file would otherwise put (0, 0) into the tindex extent.
        raise ValueError(f"Input file(s) without points: {', '.join(empty)}; remove them and run again")
    should_skip_tiling = False
    if tiling_threshold is not None and len(source_files) == 1:
        original_size_mb = source_files[0].stat().st_size / (1024 * 1024)
        if original_size_mb < tiling_threshold:
            should_skip_tiling = True
            print("=" * 60)
            print("Tiling Threshold Check")
            print("=" * 60)
            print(f"  Single file detected: {source_files[0].name}")
            print(f"  Original file size: {original_size_mb:.2f} MB")
            print(f"  Threshold: {tiling_threshold} MB")
            print("  Decision: one whole-cloud tile, no buffer")
            print("=" * 60)
            print()
    # Tile sizes and resolutions are metres: reject degree/ECEF/foot/mixed CRSs first.
    require_metric_tiling_crs(source_files)

    # Step 1: Build tindex from input LAZ/LAS
    tindex_file = build_tindex(input_dir, tindex_file)

    # Step 2: Calculate tile bounds
    jobs_file, bounds_json, env = calculate_tile_bounds(
        tindex_file, tile_length, tile_buffer, output_dir, grid_origin
    )

    # Symlink tindex for Galaxy if needed
    fixed_tindex = output_dir / "tindex.gpkg"
    if not fixed_tindex.exists() and tindex_file.exists():
        if fixed_tindex.is_symlink():
            fixed_tindex.unlink()
        try:
            fixed_tindex.symlink_to(tindex_file.name)
        except OSError:
            # Windows developer environments may not grant symlink privileges.
            # Galaxy only needs a stable path, so a byte-for-byte copy is safe.
            shutil.copy2(tindex_file, fixed_tindex)

    plan = TilingPlan(source_files, tindex_file, bounds_json, jobs_file, should_skip_tiling)
    if should_skip_tiling:
        write_single_cloud_bounds(bounds_json, source_files[0])
        from prepare_tile_jobs import write_job_list
        write_job_list(bounds_json, jobs_file)
        plot_overview(plan)
    return plan


def plot_overview(plan: TilingPlan) -> None:
    """Write the tile layout overview next to the bounds JSON."""
    plot_tiles_and_copc.plot_extents(
        plan.tindex_file, plan.bounds_json, plan.bounds_json.parent / "overview_copc_tiles.png"
    )


def read_tile_jobs(jobs_file: Path) -> Dict[str, Tuple[float, float, float, float]]:
    """Return {label: (xmin, ymin, xmax, ymax)} from a tile job list."""
    tiles: Dict[str, Tuple[float, float, float, float]] = {}
    for line in Path(jobs_file).read_text().splitlines():
        parts = line.strip().split("|")
        if len(parts) >= 2:
            bounds = _parse_proj_bounds(parts[1])
            if bounds:
                tiles[parts[0]] = bounds
    return tiles
