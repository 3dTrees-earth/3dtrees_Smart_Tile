#!/usr/bin/env python3
"""
Main tiling script: index building and tiling from LAZ/LAS input.

This script handles the first phase of the 3DTrees pipeline:
1. Build spatial index (tindex) from input LAZ/LAS files
2. Calculate tile bounds
3. Create overlapping tiles (laspy crop, COPC conversion via PDAL or untwine)

Uses laspy for Phase 1 (distribute/crop) and PDAL or untwine for Phase 2 (COPC).

Usage:
    python main_tile.py --input_dir /path/to/input --output_dir /path/to/output
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import plot_tiles_and_copc

from copc_metadata import (
    append_source_geotiff_projection_evlrs as _append_source_geotiff_projection_evlrs,
    copc_preserves_source_crs as _copc_preserves_source_crs,
    first_crs_source as _first_crs_source,
    laspy_laz_backend as _laspy_laz_backend,
)
from parameters import TILE_PARAMS
from point_cloud_metadata import point_cloud_files
from tile_copc import (
    convert_laz_to_copc as _convert_laz_to_copc,
    convert_laz_to_copc_pdal as _convert_laz_to_copc_pdal,
    finalize_tile_to_copc as _finalize_tile_to_copc,
    finalize_tile_to_copc_pdal as _finalize_tile_to_copc_pdal,
    finalize_tile_to_copc_untwine as _finalize_tile_to_copc_untwine,
)
from tile_tindex import (
    bounds_overlap as _bounds_overlap,
    build_tindex,
    calculate_tile_bounds,
    filter_source_files_for_tile,
    get_bounds as _get_bounds,
    get_pdal_path,
    get_pdal_wrench_path,
    get_source_bounds_from_tindex,
    get_source_files_from_tindex,
    parse_proj_bounds as _parse_proj_bounds,
    update_tile_bounds_json_from_files,
)



def _tiling_input_files(input_dir: Path) -> List[Path]:
    """Return tiling inputs, preferring COPC twins over matching raw files."""
    return point_cloud_files(input_dir)


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


def _crop_with_laspy(
    input_file: str,
    output_file: Path,
    bounds: Tuple[float, float, float, float],
) -> Tuple[bool, int, str]:
    """Crop a LAZ/COPC file to bounds using laspy + numpy.

    Reads the full file, applies a bounding box mask, and writes the
    cropped points as compressed LAZ.  This bypasses PDAL's readers.copc
    which hangs on large selections (>50M points).

    Args:
        input_file: Path to input LAZ/COPC file.
        output_file: Path for the cropped output LAZ file.
        bounds: (xmin, ymin, xmax, ymax) bounding box.

    Returns:
        (success, point_count, message)
    """
    import laspy
    import numpy as np

    xmin, ymin, xmax, ymax = bounds

    try:
        laz_backend = _laspy_laz_backend()
        kwargs = {}
        if input_file.lower().endswith(".laz") and laz_backend is not None:
            kwargs["laz_backend"] = laz_backend

        las = laspy.read(input_file, **kwargs)

        mask = (
            (np.asarray(las.x) >= xmin)
            & (np.asarray(las.x) <= xmax)
            & (np.asarray(las.y) >= ymin)
            & (np.asarray(las.y) <= ymax)
        )

        count = int(mask.sum())
        if count == 0:
            return (True, 0, "No points in bounds")

        cropped = las.points[mask]

        new_header = _make_tile_header(las.header)

        new_las = laspy.LasData(new_header)
        new_las.points = cropped

        write_kwargs = {}
        if laz_backend is not None:
            write_kwargs["laz_backend"] = laz_backend
        new_las.write(str(output_file), **write_kwargs)

        return (True, count, "OK")
    except Exception as e:
        return (False, 0, str(e))


def _distribute_source_file(args: Tuple) -> List[Tuple[str, int]]:
    """
    Phase 1: Read one source file (chunked) and write cropped parts for all
    overlapping tiles.

    Each source file is read exactly once.  Points are streamed in chunks
    of CHUNK_SIZE to limit peak memory, and for each chunk the bounding-box
    mask is applied for every overlapping tile.  Matching points are written
    immediately to per-tile LAS part files (one per source/chunk).

    When the LazrsParallel backend is used, RAYON_NUM_THREADS is set from the
    threads argument so chunk decompression uses multiple threads.

    Args:
        args: (source_idx, src_file, overlapping_tiles, tiles_dir, decompress_threads, chunk_size)
              overlapping_tiles: list of (label, (xmin, ymin, xmax, ymax))
              decompress_threads: threads for LAZ decompression (LazrsParallel / Rayon)
              chunk_size: points per chunk (smaller = less peak RAM, more overhead)

    Returns:
        list of (tile_label, point_count) for tiles that received points
    """
    import laspy
    import numpy as np

    source_idx, src_file, overlapping_tiles, tiles_dir, decompress_threads, chunk_size = args

    # So LazrsParallel (Rayon) uses N threads for chunk decompression
    if decompress_threads and decompress_threads > 0:
        os.environ["RAYON_NUM_THREADS"] = str(decompress_threads)

    if not os.path.isfile(src_file):
        return []

    try:
        laz_backend = _laspy_laz_backend()
        open_kwargs = {}
        if src_file.lower().endswith(".laz") and laz_backend is not None:
            open_kwargs["laz_backend"] = laz_backend

        # --- stream through the file in chunks --------------------------------
        # Write chunk parts immediately. The COPC finalization already merges
        # part_*.las files, so keeping one part per source/chunk avoids holding
        # all duplicated buffered-tile points in memory for large multi-tile runs.
        tile_counts: Dict[str, int] = {label: 0 for label, _ in overlapping_tiles}

        # Build a compact bounds array for vectorised overlap tests
        tile_labels = [label for label, _ in overlapping_tiles]
        tile_xmin = np.array([b[0] for _, b in overlapping_tiles])
        tile_xmax = np.array([b[2] for _, b in overlapping_tiles])
        tile_ymin = np.array([b[1] for _, b in overlapping_tiles])
        tile_ymax = np.array([b[3] for _, b in overlapping_tiles])

        with laspy.open(src_file, **open_kwargs) as reader:
            header_snapshot = reader.header
            src_scales = header_snapshot.scales
            src_offsets = header_snapshot.offsets
            tile_bounds_map = {lbl: bnds for lbl, bnds in overlapping_tiles}

            for chunk_index, chunk in enumerate(reader.chunk_iterator(chunk_size)):
                cx = np.asarray(chunk.x)
                cy = np.asarray(chunk.y)

                for i, label in enumerate(tile_labels):
                    mask = (
                        (cx >= tile_xmin[i])
                        & (cx <= tile_xmax[i])
                        & (cy >= tile_ymin[i])
                        & (cy <= tile_ymax[i])
                    )
                    cnt = int(mask.sum())
                    if cnt == 0:
                        continue

                    tile_dir = tiles_dir / label
                    tile_dir.mkdir(exist_ok=True)
                    part_file = tile_dir / f"part_{source_idx}_{chunk_index:06d}.las"

                    selected = chunk.array[mask].copy()

                    # Compute tile-centered offsets to keep scaled values within int32.
                    bxmin, bymin, bxmax, bymax = tile_bounds_map[label]
                    tile_offsets = np.array([
                        (bxmin + bxmax) / 2.0,
                        (bymin + bymax) / 2.0,
                        src_offsets[2],
                    ])

                    # Re-encode X/Y with tile-specific offsets.
                    real_x = selected['X'] * src_scales[0] + src_offsets[0]
                    real_y = selected['Y'] * src_scales[1] + src_offsets[1]
                    selected['X'] = np.round((real_x - tile_offsets[0]) / src_scales[0]).astype(np.int32)
                    selected['Y'] = np.round((real_y - tile_offsets[1]) / src_scales[1]).astype(np.int32)

                    new_header = _make_tile_header(header_snapshot, offsets=tile_offsets)
                    point_record = laspy.ScaleAwarePointRecord(
                        selected, new_header.point_format, new_header.scales, new_header.offsets,
                    )
                    new_las = laspy.LasData(new_header)
                    new_las.points = point_record
                    new_las.write(str(part_file))

                    tile_counts[label] += cnt

        results: List[Tuple[str, int]] = []
        for label in tile_labels:
            if tile_counts[label] > 0:
                results.append((label, tile_counts[label]))

        return results

    except Exception as e:
        print(f"    ⚠ Error processing {Path(src_file).name}: {e}")
        return []


def create_tiles(
    tindex_file: Path,
    tile_jobs_file: Path,
    tiles_dir: Path,
    log_dir: Path,
    threads: int = 5,
    max_parallel: int = 5,
    chunk_size: int = 20_000_000,
) -> List[Path]:
    """
    Create overlapping tiles from source LAZ/LAS files (two-phase).

    Phase 1 – Distribute: each source file is read exactly once (in chunks)
    and cropped points are written as per-tile LAZ part files.  This avoids
    the previous O(sources × tiles) read pattern.

    Phase 2 – Finalise: each tile's part files are merged and converted to
    COPC format (tries untwine, falls back to PDAL).  Fully parallelised
    across tiles.

    Args:
        tindex_file: Path to tindex GeoPackage
        tile_jobs_file: Path to tile jobs file
        tiles_dir: Output directory for tiles
        log_dir: Directory for log files
        threads: Threads used per process for LAZ chunk decompression (LazrsParallel/Rayon)
        max_parallel: Maximum parallel workers for each phase
        chunk_size: Points per chunk when reading source files (smaller = less peak RAM)

    Returns:
        List of created tile paths
    """
    print()
    print("=" * 60)
    print("Step 3: Creating tiles (two-phase)")
    print("=" * 60)

    # Create directories
    tiles_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ── Parse tile jobs ──────────────────────────────────────────────────
    all_tiles: Dict[str, Tuple[float, float, float, float]] = {}
    with open(tile_jobs_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) >= 2:
                label = parts[0]
                tb = _parse_proj_bounds(parts[1])
                if tb:
                    all_tiles[label] = tb

    if not all_tiles:
        raise ValueError("No tile jobs found")

    # ── Source files & bounds ────────────────────────────────────────────
    source_files = get_source_files_from_tindex(tindex_file)
    if not source_files:
        raise ValueError("No source files found in tindex")

    source_bounds = get_source_bounds_from_tindex(tindex_file)
    bounds_by_basename = (
        {Path(p).name: b for p, b in source_bounds.items()} if source_bounds else {}
    )
    crs_reference = _first_crs_source([Path(p) for p in source_files])

    # Skip tiles whose COPC output already exists
    pending_tiles: Dict[str, Tuple[float, float, float, float]] = {}
    already_done = 0
    for label, bounds in all_tiles.items():
        final_tile = tiles_dir / f"{label}.copc.laz"
        if final_tile.exists() and final_tile.stat().st_size > 0:
            valid_existing_crs = True
            if crs_reference is not None:
                preserved_geotiff, geotiff_message = _append_source_geotiff_projection_evlrs(
                    crs_reference, final_tile
                )
                if not preserved_geotiff:
                    print(f"  Existing tile {final_tile.name} GeoTIFF preservation failed: {geotiff_message}")
                    valid_existing_crs = False
                valid_existing_crs, crs_message = _copc_preserves_source_crs(
                    crs_reference, final_tile
                ) if valid_existing_crs else (False, geotiff_message)
                if not valid_existing_crs:
                    print(f"  Existing tile {final_tile.name} CRS validation failed: {crs_message}")
                    try:
                        final_tile.unlink(missing_ok=True)
                    except OSError:
                        pass
            if valid_existing_crs:
                already_done += 1
            else:
                pending_tiles[label] = bounds
        else:
            pending_tiles[label] = bounds

    print(f"  Source files: {len(source_files)}")
    print(f"  Total tiles: {len(all_tiles)} ({already_done} already done, {len(pending_tiles)} pending)")
    print(f"  Workers: {max_parallel}")

    if not pending_tiles:
        print("  ✓ All tiles already exist")
        return list(tiles_dir.glob("*.copc.laz"))

    # ── Phase 1: Distribute ─────────────────────────────────────────────
    # For each source file, determine which pending tiles it overlaps.
    distribute_tasks = []
    for source_idx, src_file in enumerate(source_files):
        fb = _get_bounds(src_file, source_bounds, bounds_by_basename)
        overlapping = []
        for label, tb in pending_tiles.items():
            if fb is None or _bounds_overlap(fb, tb):
                overlapping.append((label, tb))
        if overlapping:
            distribute_tasks.append((source_idx, src_file, overlapping, tiles_dir, threads, chunk_size))

    print()
    print(f"  Phase 1: Reading {len(distribute_tasks)} source file(s), "
          f"distributing to {len(pending_tiles)} tile(s)  [chunked reads, {threads} thread(s) per decompress]")
    print()

    tile_point_counts: Dict[str, int] = {}
    with ProcessPoolExecutor(max_workers=max_parallel) as executor:
        futures = {
            executor.submit(_distribute_source_file, task): Path(task[1]).name
            for task in distribute_tasks
        }
        for future in as_completed(futures):
            src_name = futures[future]
            try:
                results = future.result()
                for label, count in results:
                    tile_point_counts[label] = tile_point_counts.get(label, 0) + count
                if results:
                    total_pts = sum(c for _, c in results)
                    print(f"    ✓ {src_name}: {total_pts:,} pts → {len(results)} tile(s)")
                else:
                    print(f"    - {src_name}: no overlapping data")
            except Exception as e:
                print(f"    ✗ {src_name}: {e}")

    # ── Phase 2: Finalise ───────────────────────────────────────────────
    finalize_tasks = [
        (label, tiles_dir, log_dir, pending_tiles.get(label))
        for label in pending_tiles
    ]

    print()
    print(
        f"  Phase 2: Merging & converting {len(finalize_tasks)} tile(s) to COPC"
    )
    print()

    successful = 0
    failed = 0
    skipped = 0

    with ProcessPoolExecutor(max_workers=max_parallel) as executor:
        futures = {
            executor.submit(_finalize_tile_to_copc, task): task[0]
            for task in finalize_tasks
        }
        for future in as_completed(futures):
            label, success, message = future.result()
            pts = tile_point_counts.get(label, 0)
            if success:
                if "Already exists" in message or "No data" in message:
                    skipped += 1
                    print(f"    - {label}: {message}")
                else:
                    successful += 1
                    print(f"    ✓ {label}: {message} ({pts:,} pts)")
            else:
                failed += 1
                print(f"    ✗ {label}: {message}")

    print()
    print(f"  Tiling complete: {successful} created, {skipped} skipped, {failed} failed")

    return list(tiles_dir.glob("*.copc.laz"))


def run_tiling_pipeline(
    input_dir: Path,
    output_dir: Path,
    tile_length: float = 100,
    tile_buffer: float = 5,
    grid_offset: float = 1.0,
    num_workers: int = 4,
    threads: int = 5,
    max_tile_procs: int = 5,
    dimension_reduction: bool = True,  # Ignored (kept for API compatibility)
    tiling_threshold: float = None,
    chunk_size: int = 2_000_000,
) -> Path:
    """
    Run the complete tiling pipeline.

    Steps:
    1. Build spatial index (tindex) from input LAZ/LAS files
    2. Calculate tile bounds
    3. Create overlapping tiles (laspy crop, COPC conversion via untwine/PDAL)

    If input folder contains a single file below tiling_threshold, converts it to
    COPC and returns that directory for direct subsampling.

    Args:
        input_dir: Directory containing input LAZ/LAS files
        output_dir: Base output directory
        tile_length: Tile size in meters
        tile_buffer: Buffer overlap in meters
        grid_offset: Offset from min coordinates
        num_workers: Unused (kept for API compatibility)
        threads: Threads per PDAL writer
        max_tile_procs: Maximum parallel tile processes
        dimension_reduction: Ignored (kept for API compatibility)
        tiling_threshold: File size threshold in MB. If single file below this, skip tiling
        chunk_size: Points per chunk when reading LAZ/LAS in Phase 1 (smaller = less peak RAM)

    Returns:
        Path to tiles directory (or copc_single directory if tiling was skipped)
    """
    print("=" * 60)
    print("3DTrees Tiling Pipeline (laspy + PDAL)")
    print("=" * 60)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Tile size: {tile_length}m with {tile_buffer}m buffer")
    print()

    tiles_dir = output_dir / f"tiles_{int(tile_length)}m"
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    tindex_file = output_dir / f"tindex_{int(tile_length)}m.gpkg"

    # Check if we should skip tiling (single small file)
    should_skip_tiling = False
    if tiling_threshold is not None:
        input_files = _tiling_input_files(input_dir)

        if len(input_files) == 1:
            original_size_mb = input_files[0].stat().st_size / (1024 * 1024)
            if original_size_mb < tiling_threshold:
                should_skip_tiling = True
                print("=" * 60)
                print("Tiling Threshold Check")
                print("=" * 60)
                print(f"  Single file detected: {input_files[0].name}")
                print(f"  Original file size: {original_size_mb:.2f} MB")
                print(f"  Threshold: {tiling_threshold} MB")
                print("  Decision: Will skip tiling and use a COPC source for subsampling")
                print("=" * 60)
                print()

    # Validate input
    source_files = _tiling_input_files(input_dir)
    if not source_files:
        raise ValueError(f"No LAZ/LAS files found in {input_dir}")

    # Step 1: Build tindex from input LAZ/LAS
    tindex_file = build_tindex(input_dir, tindex_file)

    # Step 2: Calculate tile bounds
    jobs_file, bounds_json, env = calculate_tile_bounds(
        tindex_file, tile_length, tile_buffer, output_dir, grid_offset
    )

    # Symlink tindex for Galaxy if needed
    fixed_tindex = output_dir / "tindex.gpkg"
    if not fixed_tindex.exists() and tindex_file.exists():
        if fixed_tindex.is_symlink():
            fixed_tindex.unlink()
        fixed_tindex.symlink_to(tindex_file.name)

    # Plot overview
    plot_tiles_and_copc.plot_extents(
        tindex_file, bounds_json, output_dir / "overview_copc_tiles.png"
    )

    # Check if we should skip tiling (single small file)
    # Done AFTER tindex/bounds/plot so those outputs are always available for merge
    if should_skip_tiling:
        print()
        print("=" * 60)
        print("Skipping Tiling (Single Small File)")
        print("=" * 60)
        source_file = source_files[0]
        copc_single_dir = output_dir / "copc_single"
        copc_single_dir.mkdir(parents=True, exist_ok=True)
        if source_file.name.lower().endswith(".copc.laz"):
            out_copc = copc_single_dir / source_file.name
            if not out_copc.exists() or out_copc.stat().st_size == 0:
                print("  Reusing uploaded COPC without conversion...")
                shutil.copy2(source_file, out_copc)
                print(f"  ✓ Copied {out_copc.name}")
            else:
                print(f"  Using existing {out_copc.name}")
            print("  Returning COPC directory for direct subsampling")
            print("=" * 60)
            return copc_single_dir

        out_copc = copc_single_dir / f"{source_file.stem}.copc.laz"
        rebuild_copc = not out_copc.exists() or out_copc.stat().st_size == 0
        if not rebuild_copc:
            preserved_geotiff, geotiff_message = _append_source_geotiff_projection_evlrs(
                source_file, out_copc
            )
            valid_crs, crs_message = _copc_preserves_source_crs(source_file, out_copc)
            if not preserved_geotiff or not valid_crs:
                if not preserved_geotiff:
                    print(f"  Existing COPC GeoTIFF preservation failed: {geotiff_message}")
                print(f"  Existing COPC CRS validation failed: {crs_message}")
                print("  Rebuilding COPC from source LAZ...")
                try:
                    out_copc.unlink(missing_ok=True)
                except OSError:
                    pass
                rebuild_copc = True
        if rebuild_copc:
            print("  Converting LAZ to COPC...")
            if not _convert_laz_to_copc(source_file, out_copc):
                raise RuntimeError(f"LAZ→COPC conversion failed: {source_file}")
            print(f"  ✓ Created {out_copc.name}")
        else:
            print(f"  Using existing {out_copc.name}")
        print(f"  Returning COPC directory for direct subsampling")
        print("=" * 60)
        return copc_single_dir

    # Step 3: Create tiles
    tile_files = create_tiles(
        tindex_file,
        jobs_file,
        tiles_dir,
        log_dir,
        threads,
        max_tile_procs,
        chunk_size,
    )

    print()
    print("=" * 60)
    print("Tiling Pipeline Complete")
    print("=" * 60)
    print(f"  Source files: {len(source_files)}")
    print(f"  Tiles created: {len(tile_files)}")
    print(f"  Tiles directory: {tiles_dir}")

    return tiles_dir


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="3DTrees Tiling Pipeline - laspy + PDAL tiling from LAZ/LAS input",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input_dir", "-i",
        type=Path,
        required=True,
        help="Input directory containing LAZ files"
    )

    parser.add_argument(
        "--output_dir", "-o",
        type=Path,
        required=True,
        help="Output directory for all stages"
    )

    parser.add_argument(
        "--tile_length",
        type=float,
        default=TILE_PARAMS.get('tile_length', 100),
        help=f"Tile size in meters (default: {TILE_PARAMS.get('tile_length', 100)})"
    )

    parser.add_argument(
        "--tile_buffer",
        type=float,
        default=TILE_PARAMS.get('tile_buffer', 5),
        help=f"Buffer overlap in meters (default: {TILE_PARAMS.get('tile_buffer', 5)})"
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=TILE_PARAMS.get('workers', 4),
        help=f"Number of parallel workers (default: {TILE_PARAMS.get('workers', 4)})"
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=TILE_PARAMS.get('threads', 5),
        help=f"Threads per COPC writer (default: {TILE_PARAMS.get('threads', 5)})"
    )

    parser.add_argument(
        "--max_tile_procs",
        type=int,
        default=5,
        help="Maximum parallel tile processes (default: 5)"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=TILE_PARAMS.get("chunk_size", 20_000_000),
        help="Points per chunk when reading LAZ/LAS (default: 2_000_000; smaller = less peak RAM)",
    )
    args = parser.parse_args()

    # Validate input
    if not args.input_dir.exists():
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)

    # Run pipeline
    try:
        tiles_dir = run_tiling_pipeline(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            tile_length=args.tile_length,
            tile_buffer=args.tile_buffer,
            num_workers=args.num_workers,
            threads=args.threads,
            max_tile_procs=args.max_tile_procs,
            chunk_size=args.chunk_size,
        )
        print(f"\nTiles ready for subsampling: {tiles_dir}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
