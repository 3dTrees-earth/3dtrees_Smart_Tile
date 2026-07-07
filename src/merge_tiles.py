#!/usr/bin/env python3
"""
Tile Merger - Merge Segmented Point Cloud Tiles with Species ID Preservation

Merges overlapping segmented point cloud tiles using:
1. Load and filter: Centroid-based buffer zone filtering (remove instances in overlap zones)
2. Assign global IDs: Create unique instance IDs across all tiles
3. Cross-tile matching: Overlap ratio matching for cross-tile instance merging
4. Merge and deduplicate: Combine tiles and remove duplicate points
5. Small volume merging: Reassign small clusters to nearest large instance
6. Retiling: Map merged results back to original point cloud files

Key feature: Species ID is always preserved from the LARGER instance (by point count)
during all merge and reassignment operations.

Usage:
python merge_tiles.py \
    --input-dir /path/to/segmented_remapped \
    --original-tiles-dir /path/to/original_tiles \
    --output-merged /path/to/merged.laz \
    --output-tiles-dir /path/to/output_tiles
"""

import gc
import sys
import numpy as np
import laspy
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from concurrent.futures import ProcessPoolExecutor
from instance_labels import (
    MERGED_OUTPUT_SCALES,
    cast_instances_for_output,
    instance_extra_bytes_params,
    validate_merged_output_contract,
)
from point_cloud_outputs import (
    merged_product_header,
    write_loaded_point_cloud,
)
from merge_deduplication import deduplicate_points
from merge_instance_ids import global_id
from merge_instance_matching import InstanceMergeResult, assign_and_match_instances
from merge_loaded_cloud import (
    load_merged_file,
    reassign_small_instances_in_dims,
)
from merge_original_dimensions import add_original_dimensions_to_merged
from merge_small_instances import merge_small_volume_instances
from merge_tile_loading import (
    load_tile,
    load_tile_wrapper as _load_tile_wrapper,
    merge_tile_name,
)
from point_cloud_metadata import point_cloud_files
from output_remap import (
    remap_to_original_input_files,
    retile_to_original_files,
)
from tile_spatial import (
    get_tile_bounds_from_header,
)
from tile_bounds_graph import (
    build_neighbor_graph_from_bounds_json,
    match_tiles_to_json_bounds,
)
from filter_task_support import derive_tile_buffer_from_json, is_tree_sidecar_file

# Force unbuffered output for real-time progress feedback
# (especially important when running in Docker/containers)
sys.stdout.reconfigure(line_buffering=True)


def _merge_input_files(input_dir: Path) -> List[Path]:
    """Return merge input files, including mixed LAS/LAZ and preferring COPC twins."""
    return point_cloud_files(input_dir)


def input_has_tree_sidecars(input_dir: Path) -> bool:
    """Return whether a merge/filter input directory contains tree sidecar files."""
    return any(is_tree_sidecar_file(path) for path in Path(input_dir).glob("*.txt"))


def _assign_local_only_instances(
    tiles,
    kept_instances_per_tile,
) -> InstanceMergeResult:
    """Assign merged IDs without cross-tile matching or orphan recovery."""
    global_to_merged = {}
    merged_instance_sources = {}
    tile_idx_to_name = {}
    next_merged_id = 1

    print(f"\n{'=' * 60}")
    print("Stage 2: Assigning local-only instance IDs")
    print(f"{'=' * 60}")
    for tile_idx, tile in enumerate(tiles):
        tile_idx_to_name[tile_idx] = tile.name
        kept_instances = sorted(inst for inst in kept_instances_per_tile[tile.name] if inst > 0)
        print(
            f"  Processing tile {tile_idx + 1}/{len(tiles)}: "
            f"{tile.name} ({len(kept_instances)} instances)"
        )
        for local_inst in kept_instances:
            gid = global_id(tile_idx, local_inst)
            global_to_merged[gid] = next_merged_id
            merged_instance_sources[next_merged_id] = [gid]
            next_merged_id += 1
    print(f"  Total local-only merged instances: {next_merged_id - 1}")
    return InstanceMergeResult(
        global_to_merged=global_to_merged,
        merged_instance_sources=merged_instance_sources,
        tile_idx_to_name=tile_idx_to_name,
    )


def _instance_order_north_to_south(points: np.ndarray, instances: np.ndarray) -> List[int]:
    """Return positive instance IDs sorted north-to-south, then west-to-east."""
    pos_mask = instances > 0
    if not np.any(pos_mask):
        return []

    pos_instances = instances[pos_mask].astype(np.int64, copy=False)
    max_positive_id = int(pos_instances.max())
    counts = np.bincount(pos_instances, minlength=max_positive_id + 1)
    unique_inst = np.flatnonzero(counts)
    unique_inst = unique_inst[unique_inst > 0]

    min_x = np.full(max_positive_id + 1, np.inf, dtype=np.float64)
    max_x = np.full(max_positive_id + 1, -np.inf, dtype=np.float64)
    min_y = np.full(max_positive_id + 1, np.inf, dtype=np.float64)
    max_y = np.full(max_positive_id + 1, -np.inf, dtype=np.float64)

    pos_x = points[pos_mask, 0]
    pos_y = points[pos_mask, 1]
    np.minimum.at(min_x, pos_instances, pos_x)
    np.maximum.at(max_x, pos_instances, pos_x)
    np.minimum.at(min_y, pos_instances, pos_y)
    np.maximum.at(max_y, pos_instances, pos_y)

    center_y = (min_y[unique_inst] + max_y[unique_inst]) / 2.0
    center_x = (min_x[unique_inst] + max_x[unique_inst]) / 2.0
    order = np.lexsort((center_x, -center_y))
    return unique_inst[order].tolist()


# =============================================================================
# Main Merge Function
# =============================================================================


def merge_tiles(
    input_dir: Path,
    original_tiles_dir: Path,
    output_merged: Path,
    output_tiles_dir: Path,
    tile_bounds_json: Path,
    original_input_dir: Optional[Path] = None,
    overlap_threshold: float = 0.3,
    correspondence_tolerance: float = 0.1,
    max_volume_for_merge: float = 4.0,
    border_zone_width: float = 10.0,
    min_cluster_size: int = 300,
    num_threads: int = 8,
    enable_matching: bool = True,
    enable_volume_merge: bool = True,
    skip_merged_file: bool = False,
    verbose: bool = False,
    retile_buffer: float = 2.0,
    retile_max_radius: float = 2.0,
    debug_instance_ids: Optional[Set[int]] = None,
    match_all_instances: bool = False,
    instance_dimension: str = "PredInstance",
    transfer_original_dims_to_merged: bool = True,
    threedtrees_dims: Optional[List[str]] = None,
    threedtrees_suffix: str = "SAT",
    chunk_size: int = 1_000_000,
):
    """
    Main merge function implementing the tile merging pipeline.
    """
    if tile_bounds_json is None:
        raise ValueError("tile_bounds_json is required but was not provided.")
    if not tile_bounds_json.exists():
        raise FileNotFoundError(
            f"tile_bounds_tindex.json not found: {tile_bounds_json}. "
            "Merge requires this file and will not run without it."
        )
    buffer = derive_tile_buffer_from_json(tile_bounds_json)
    tree_sidecars_present = input_has_tree_sidecars(input_dir)
    if tree_sidecars_present:
        if enable_matching:
            print("Tree sidecar files detected; disabling cross-tile instance matching.")
        if enable_volume_merge:
            print("Tree sidecar files detected; disabling small cluster reassignment.")
        enable_matching = False
        enable_volume_merge = False

    print("=" * 60)
    print("Tile Merger")
    print("=" * 60)
    print(f"Input directory: {input_dir}")
    print(f"Original tiles: {original_tiles_dir}")
    print(f"Output merged: {output_merged}" + (" (SKIPPED)" if skip_merged_file else ""))
    print(f"Output tiles: {output_tiles_dir}")
    print(f"Tile bounds JSON: {tile_bounds_json}")
    print(f"Instance dimension: {instance_dimension}")
    print(f"Buffer: {buffer}m (from tile_bounds_tindex.json)")
    print(f"Workers: {num_threads}")
    print(f"Instance matching: {'ENABLED' if enable_matching else 'DISABLED'}")
    if enable_matching:
        print(f"  Overlap threshold: {overlap_threshold}")
        print(f"  Match all instances: {'YES' if match_all_instances else 'NO (border region only)'}")
    print(f"Tree sidecar files: {'PRESENT' if tree_sidecars_present else 'none'}")
    print(f"Small cluster reassignment: {'ENABLED' if enable_volume_merge else 'DISABLED'}")
    if enable_volume_merge:
        print(f"  Min cluster size: {min_cluster_size} points")
    print(f"Volume merge: {'ENABLED' if enable_volume_merge else 'DISABLED'}")
    if enable_volume_merge:
        print(f"  Max volume for merge: {max_volume_for_merge} m³")
    if original_input_dir:
        print(f"Original input dir: {original_input_dir} (Stage 7 enabled)")
    print(f"Verbose: {verbose}")
    print("=" * 60)

    # Check if merged output file already exists
    if output_merged.exists():
        print(f"\n{'=' * 60}")
        print(f"Merged file already exists: {output_merged}")
        print(f"{'=' * 60}")
        validate_merged_output_contract(output_merged, instance_dimension)
        print("  Loading merged file and proceeding to retiling stage...")

        merged_points, all_merged_dims, merged_extra_dim_params = load_merged_file(output_merged)
        # For retile we need (instances, extra_dims) split; for remap we pass all_merged_dims as-is
        if instance_dimension in all_merged_dims:
            merged_instances = all_merged_dims[instance_dimension]
            merged_extra_dims = {k: v for k, v in all_merged_dims.items() if k != instance_dimension}
        elif "treeID" in all_merged_dims:
            merged_instances = all_merged_dims["treeID"]
            merged_extra_dims = {k: v for k, v in all_merged_dims.items() if k != "treeID"}
        else:
            merged_instances = None
            merged_extra_dims = {}
            for k, v in all_merged_dims.items():
                if merged_instances is None and np.issubdtype(v.dtype, np.integer):
                    merged_instances = v
                else:
                    merged_extra_dims[k] = v
            if merged_instances is None:
                merged_instances = np.zeros(len(merged_points), dtype=np.int32)

        retile_to_original_files(
            merged_points,
            merged_instances,
            merged_extra_dims,
            merged_extra_dim_params,
            original_tiles_dir,
            output_tiles_dir,
            tolerance=0.1,
            num_threads=num_threads,
            retile_buffer=retile_buffer,
            instance_dimension=instance_dimension,
        )
        print(f"  ✓ Stage 6 completed: Retiled to original files")

        if original_input_dir is not None:
            print(f"\n{'=' * 60}")
            print("Stage 7: Remapping to Original Input Files")
            print(f"{'=' * 60}")

            original_output_dir = output_tiles_dir.parent / "original_with_predictions"
            remap_to_original_input_files(
                merged_points,
                all_merged_dims,
                merged_extra_dim_params,
                original_input_dir,
                original_output_dir,
                tolerance=retile_max_radius,
                num_threads=num_threads,
                retile_buffer=retile_buffer,
                threedtrees_dims=threedtrees_dims,
                threedtrees_suffix=threedtrees_suffix,
            )
            print(f"  ✓ Stage 7 completed: Remapped to original input files")
        else:
            print(f"\n  Note: --original-input-dir not provided, skipping Stage 7 (remap to original input files)")

        print(f"\n{'=' * 60}")
        print("Merge complete!")
        print(f"{'=' * 60}")
        return

    # Find all input point clouds, preferring COPC twins for the same source.
    laz_files = _merge_input_files(input_dir)

    if len(laz_files) == 0:
        print(f"No LAZ/LAS files found in {input_dir}")
        return

    print(f"\nFound {len(laz_files)} tiles to merge")

    # Extract tile names and get bounds from headers for neighbor detection
    print("  Extracting tile bounds from headers...")
    tile_boundaries: Dict[str, Tuple[float, float, float, float]] = {}
    tile_names: List[str] = []
    source_file_by_tile_name: Dict[str, Path] = {}

    for f in laz_files:
        name = merge_tile_name(f)
        tile_names.append(name)
        source_file_by_tile_name[name] = f

        bounds = get_tile_bounds_from_header(f)
        if bounds:
            tile_boundaries[name] = bounds
        else:
            print(f"    Warning: Could not extract bounds from {f.name}")

    if len(tile_boundaries) == 0:
        raise ValueError("Could not extract bounds from any tile files")

    print(f"  Extracted bounds from {len(tile_boundaries)} tiles")

    # Build neighbor graph from tile_bounds_tindex.json and match to loaded tiles
    print("  Loading neighbor graph from tile_bounds_tindex.json...")
    json_bounds, centers, neighbors_idx = build_neighbor_graph_from_bounds_json(tile_bounds_json)
    print(f"  JSON tiles in bounds file: {len(json_bounds)}")

    tile_to_json, json_to_tile = match_tiles_to_json_bounds(tile_boundaries, json_bounds, centers)
    print("  Matched tiles to JSON bounds successfully")

    # Build neighbors per tile name using the JSON neighbor graph.
    # JSON can contain tiles that were never created (no points in bounds); json_to_tile only
    # has entries for JSON indices that matched an actual LAZ file. So neighbor_name =
    # json_to_tile.get(n_idx) is None when the neighbor exists only in JSON (no point cloud).
    # Border detection and cross-tile matching then ignore that edge (no points filtered).
    neighbors_by_tile: Dict[str, Dict[str, Optional[str]]] = {}
    for tile_name, json_idx in tile_to_json.items():
        neighbors_for_tile: Dict[str, Optional[str]] = {"east": None, "west": None, "north": None, "south": None}
        for direction in ("east", "west", "north", "south"):
            n_idx = neighbors_idx[json_idx].get(direction)
            if n_idx is None:
                neighbors_for_tile[direction] = None
            else:
                neighbor_name = json_to_tile.get(n_idx)  # None if no LAZ for that JSON tile
                neighbors_for_tile[direction] = neighbor_name
        neighbors_by_tile[tile_name] = neighbors_for_tile

    print("  Built neighbor mapping per tile from JSON graph")

    # =========================================================================
    # Stage 1: Load and Filter
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("Stage 1: Loading tiles and filtering buffer zone instances")
    print(f"{'=' * 60}")
    print(f"  Loading {len(laz_files)} files using {num_threads} workers (--workers={num_threads})...")

    tiles = []
    filtered_instances_per_tile = {}
    kept_instances_per_tile = {}

    # Load tiles in parallel using ProcessPoolExecutor for true CPU parallelism
    # Prepare arguments for multiprocessing (must be pickleable)
    load_args = [(f, tile_boundaries, buffer, neighbors_by_tile, instance_dimension) for f in laz_files]

    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(_load_tile_wrapper, load_args))

    # Process results
    buffer_direction_per_tile = {}  # tile_name -> {inst_id -> direction}
    for result in results:
        if result is not None:
            tile_data, filtered, kept, buffer_dirs = result
            tiles.append(tile_data)
            filtered_instances_per_tile[tile_data.name] = filtered
            kept_instances_per_tile[tile_data.name] = kept
            buffer_direction_per_tile[tile_data.name] = buffer_dirs

    # Clean up loading intermediates
    del results
    gc.collect()

    if len(tiles) == 0:
        print("No tiles loaded successfully")
        return

    # Log extra dims found across tiles
    all_extra_dim_names = set()
    for tile in tiles:
        all_extra_dim_names.update(tile.extra_dims.keys())
    if all_extra_dim_names:
        print(f"  Extra dimensions (passenger data): {', '.join(sorted(all_extra_dim_names))}")

    total_points = sum(len(tile.points) for tile in tiles)
    total_kept = sum(len(kept) for kept in kept_instances_per_tile.values())
    total_filtered = sum(len(filtered) for filtered in filtered_instances_per_tile.values())
    print(f"  ✓ Stage 1 completed: {len(tiles)} tiles loaded, {total_points:,} total points")
    print(f"    Kept {total_kept} instances, filtered {total_filtered} buffer zone instances")

    # Save filtered tiles (with filtered instances removed)
    filtered_tiles_dir = output_tiles_dir / "filtered_tiles"
    filtered_tiles_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Saving filtered tiles to {filtered_tiles_dir}...")
    for tile in tiles:
        kept_instances = kept_instances_per_tile[tile.name]
        keep_mask = np.isin(tile.instances, list(kept_instances) + [0])

        filtered_points = tile.points[keep_mask]
        filtered_instances = tile.instances[keep_mask]
        filtered_extra_dims = {name: arr[keep_mask] for name, arr in tile.extra_dims.items()}

        if len(filtered_points) == 0:
            print(f"    Warning: {tile.name} has no points after filtering, skipping")
            continue

        filtered_output_path = filtered_tiles_dir / f"{tile.name}.laz"
        header = laspy.LasHeader(point_format=6, version="1.4")
        header.offsets = [filtered_points[:, 0].min(), filtered_points[:, 1].min(), filtered_points[:, 2].min()]
        header.scales = MERGED_OUTPUT_SCALES

        output_las = laspy.LasData(header)
        output_las.x = filtered_points[:, 0]
        output_las.y = filtered_points[:, 1]
        output_las.z = filtered_points[:, 2]

        extra_dims_params = [instance_extra_bytes_params(instance_dimension, filtered_instances)]
        for dim_name, dim_arr in filtered_extra_dims.items():
            extra_dims_params.append(laspy.ExtraBytesParams(name=dim_name, type=dim_arr.dtype))
        output_las.add_extra_dims(extra_dims_params)

        setattr(output_las, instance_dimension, cast_instances_for_output(filtered_instances, instance_dimension))
        for dim_name, dim_arr in filtered_extra_dims.items():
            setattr(output_las, dim_name, dim_arr)

        output_las.write(str(filtered_output_path))
    print(f"  ✓ Saved {len(tiles)} filtered tiles as .laz files (filtered instances removed)")

    # =========================================================================
    if enable_matching:
        match_result = assign_and_match_instances(
            tiles=tiles,
            tile_boundaries=tile_boundaries,
            neighbors_by_tile=neighbors_by_tile,
            kept_instances_per_tile=kept_instances_per_tile,
            filtered_instances_per_tile=filtered_instances_per_tile,
            buffer_direction_per_tile=buffer_direction_per_tile,
            buffer=buffer,
            border_zone_width=border_zone_width,
            overlap_threshold=overlap_threshold,
            correspondence_tolerance=correspondence_tolerance,
            num_threads=num_threads,
            debug_instance_ids=debug_instance_ids,
            match_all_instances=match_all_instances,
            verbose=verbose,
        )
    else:
        print(f"\n{'=' * 60}")
        print("Stage 3: Cross-tile instance matching (SKIPPED)")
        print(f"{'=' * 60}")
        if tree_sidecars_present:
            print("  Tree sidecar input keeps tile instances local; no cross-tile IDs are merged.")
        else:
            print("  Instance matching disabled by caller.")
        match_result = _assign_local_only_instances(tiles, kept_instances_per_tile)
    global_to_merged = match_result.global_to_merged
    merged_instance_sources = match_result.merged_instance_sources
    tile_idx_to_name = match_result.tile_idx_to_name
    if skip_merged_file:
        print(f"\n{'=' * 60}")
        print("Writing filtered output tiles without global merged cloud")
        print(f"{'=' * 60}")
        print("  --skip_merged_file is enabled; writing per-tile outputs directly")
        output_tiles_dir.mkdir(parents=True, exist_ok=True)

        written_tiles = 0
        total_output_points = 0
        for tile_idx, tile in enumerate(tiles):
            kept_instances = kept_instances_per_tile[tile.name]
            max_local_inst = int(tile.instances.max()) + 1
            inst_to_merged = np.full(max_local_inst, -1, dtype=np.int32)
            if max_local_inst > 0:
                inst_to_merged[0] = 0

            for local_inst in kept_instances:
                if local_inst <= 0:
                    continue
                gid = global_id(tile_idx, local_inst)
                inst_to_merged[local_inst] = global_to_merged.get(gid, -1)

            safe_instances = np.clip(tile.instances, 0, max_local_inst - 1)
            remapped_instances = inst_to_merged[safe_instances]
            valid_mask = remapped_instances != -1
            if not np.any(valid_mask):
                print(f"  Warning: {tile.name} has no valid points after filtering, skipping")
                continue

            output_dims = {
                instance_dimension: cast_instances_for_output(
                    remapped_instances[valid_mask],
                    instance_dimension,
                )
            }
            output_dims.update({
                dim_name: values[valid_mask]
                for dim_name, values in tile.extra_dims.items()
            })

            source_file = source_file_by_tile_name[tile.name]
            output_file = output_tiles_dir / f"{tile.name}.laz"
            write_loaded_point_cloud(
                source_file,
                output_file,
                tile.points[valid_mask],
                output_dims,
                source_indices=np.flatnonzero(valid_mask),
            )
            written_tiles += 1
            total_output_points += int(np.count_nonzero(valid_mask))
            print(f"  Wrote {output_file.name}: {np.count_nonzero(valid_mask):,} points")

        import csv

        csv_output_path = output_merged.parent / f"{output_merged.stem}_instance_metadata.csv"
        csv_output_path.parent.mkdir(parents=True, exist_ok=True)
        instances_with_clusters = {
            merged_id
            for merged_id, sources in merged_instance_sources.items()
            if len(sources) > 1
        }
        final_instance_ids = sorted(set(global_to_merged.values()) - {0, -1})

        with open(csv_output_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([instance_dimension, "has_added_clusters"])
            for final_id in final_instance_ids:
                writer.writerow([final_id, 1 if final_id in instances_with_clusters else 0])

        csv_copy_path = output_tiles_dir / csv_output_path.name
        if csv_copy_path != csv_output_path:
            import shutil
            shutil.copy2(str(csv_output_path), str(csv_copy_path))

        print(f"  ✓ Wrote {written_tiles} filtered output tiles, {total_output_points:,} points")
        print(f"  Instance metadata CSV: {csv_output_path}")
        print(f"\n{'=' * 60}")
        print("Merge complete!")
        print(f"{'=' * 60}")
        return

    # =========================================================================
    # Stage 4: Merge and Deduplicate
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("Stage 4: Merging tiles and deduplicating")
    print(f"{'=' * 60}")

    all_points = []
    all_instances = []
    all_extra_dims_lists: Dict[str, list] = {name: [] for name in all_extra_dim_names}

    n_input_tiles = len(tiles)
    for tile_idx, tile in enumerate(tiles):
        kept_instances = kept_instances_per_tile[tile.name]

        max_local_inst = tile.instances.max() + 1

        inst_to_merged = np.full(max_local_inst, -1, dtype=np.int32)

        if max_local_inst > 0:
            inst_to_merged[0] = 0

        for local_inst in kept_instances:
            if local_inst <= 0:
                continue
            gid = global_id(tile_idx, local_inst)
            merged_id = global_to_merged.get(gid, -1)
            inst_to_merged[local_inst] = merged_id

        safe_instances = np.clip(tile.instances, 0, max_local_inst - 1)
        remapped_instances = inst_to_merged[safe_instances]

        all_points.append(tile.points)
        all_instances.append(remapped_instances)

        # Collect extra dims (passenger data) - use zeros for dims missing from this tile
        for dim_name in all_extra_dim_names:
            if dim_name in tile.extra_dims:
                all_extra_dims_lists[dim_name].append(tile.extra_dims[dim_name])
            else:
                all_extra_dims_lists[dim_name].append(
                    np.zeros(len(tile.points), dtype=np.int32)
                )

    del tiles
    del filtered_instances_per_tile
    del kept_instances_per_tile
    gc.collect()

    merged_points = np.vstack(all_points)
    merged_instances = np.concatenate(all_instances)
    merged_extra_dims: Dict[str, np.ndarray] = {
        name: np.concatenate(arrays) for name, arrays in all_extra_dims_lists.items()
    }

    del all_points
    del all_instances
    del all_extra_dims_lists
    gc.collect()

    # Remove points from filtered buffer instances (instance_id = -1)
    valid_points_mask = merged_instances != -1
    n_filtered_removed = np.sum(merged_instances == -1)
    n_ground_points = np.sum(merged_instances == 0)

    merged_points = merged_points[valid_points_mask]
    merged_instances = merged_instances[valid_points_mask]
    merged_extra_dims = {name: arr[valid_points_mask] for name, arr in merged_extra_dims.items()}

    if n_filtered_removed > 0:
        print(f"  Removed {n_filtered_removed:,} points from filtered buffer instances")
    gc.collect()

    total_before = len(merged_points)
    print(f"  Total points: {total_before:,}")
    if n_input_tiles <= 1:
        print("  Single input tile; skipping cross-tile deduplication")
    else:
        print(f"  Deduplicating...", flush=True)
        merged_points, merged_instances, merged_extra_dims = deduplicate_points(
            merged_points, merged_instances, merged_extra_dims
        )
    gc.collect()

    n_removed = total_before - len(merged_points)
    n_tree_instances = len(np.unique(merged_instances[merged_instances > 0]))
    print(f"  Removed {n_removed:,} duplicate points ({100*n_removed/total_before:.1f}%)")
    print(f"  ✓ Stage 4 completed: {len(merged_points):,} points, {n_tree_instances} tree instances")

    del global_to_merged
    gc.collect()

    # =========================================================================
    # Stage 5: Small Volume Instance Merging
    # =========================================================================
    if enable_volume_merge:
        print(f"\n{'=' * 60}")
        print("Stage 5: Small Volume Instance Merging")
        print(f"{'=' * 60}")

        pos_mask = merged_instances > 0
        nonzero_count = pos_mask.sum()
        zero_count = len(merged_instances) - nonzero_count
        print(f"  Instance points: {nonzero_count:,}, ground points: {zero_count:,}")

        pos_instances = merged_instances[pos_mask]

        unique_inst = np.unique(pos_instances)
        print(f"  Processing {len(unique_inst):,} unique instances...", flush=True)

        merged_instances, _ = merge_small_volume_instances(
            merged_points,
            merged_instances,
            min_points_for_hull_check=1000,
            min_cluster_size=min_cluster_size,
            max_volume_for_merge=max_volume_for_merge,
            max_search_radius=5.0,
            num_threads=num_threads,
            verbose=verbose,
        )
        print(f"  ✓ Stage 5 completed")
    else:
        print(f"\n{'=' * 60}")
        print("Stage 5: Small Volume Instance Merging (SKIPPED)")
        print(f"{'=' * 60}")

    # =========================================================================
    # Renumber instances to continuous IDs (north-to-south ordering)
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("Renumbering instances (north-to-south ordering)")
    print(f"{'=' * 60}")

    print("  Computing bounding box centers...")
    sorted_by_location = _instance_order_north_to_south(merged_points, merged_instances)

    old_to_new = {0: 0}

    for new_id, old_id in enumerate(sorted_by_location, start=1):
        old_to_new[old_id] = new_id

    print(f"  Instances renumbered from north to south")

    max_old_id = int(merged_instances.max()) + 1

    instance_lookup = np.zeros(max_old_id, dtype=np.int32)
    for old_id, new_id in old_to_new.items():
        if old_id < max_old_id:
            instance_lookup[old_id] = new_id

    merged_instances = instance_lookup[merged_instances]

    print(f"  Final instance count: {len(sorted_by_location)}")

    # =========================================================================
    # Save merged output (optional - can be skipped with skip_merged_file=True)
    # =========================================================================
    if skip_merged_file:
        print(f"\n{'=' * 60}")
        print("Saving merged output (SKIPPED)")
        print(f"{'=' * 60}")
        print(f"  Skipped merged LAZ file creation (--skip_merged_file)")
        print(f"  Total points: {len(merged_points):,}")
        print(f"  Total instances: {len(sorted_by_location)}")
    else:
        print(f"\n{'=' * 60}")
        print("Saving merged output")
        print(f"{'=' * 60}")

        output_merged.parent.mkdir(parents=True, exist_ok=True)

        # Write initial merged to a temp path so we can either enrich it or use as final
        merged_init = output_merged.parent / (output_merged.stem + "_init.laz")

        header = merged_product_header(
            merged_points,
            original_input_dir,
            original_tiles_dir,
        )

        output_las = laspy.LasData(header)
        output_las.x = merged_points[:, 0]
        output_las.y = merged_points[:, 1]
        output_las.z = merged_points[:, 2]

        # Add instance dimension and passenger extra dimensions from the merge.
        extra_dims_params = [instance_extra_bytes_params(instance_dimension, merged_instances)]
        for dim_name, dim_arr in merged_extra_dims.items():
            extra_dims_params.append(laspy.ExtraBytesParams(name=dim_name, type=dim_arr.dtype))
        output_las.add_extra_dims(extra_dims_params)

        setattr(output_las, instance_dimension, cast_instances_for_output(merged_instances, instance_dimension))
        for dim_name, dim_arr in merged_extra_dims.items():
            setattr(output_las, dim_name, dim_arr)

        output_las.write(
            str(merged_init), do_compress=True, laz_backend=laspy.LazBackend.LazrsParallel
        )

        del output_las
        gc.collect()

        print(f"  Saved merged (initial): {merged_init}")
        print(f"  Total points: {len(merged_points):,}")
        print(f"  Total instances: {len(sorted_by_location)}")

        import shutil
        # Add original-file dimensions and write directly to final output (so merged file is always enriched when requested)
        if original_input_dir is not None and transfer_original_dims_to_merged:
            try:
                add_original_dimensions_to_merged(
                    merged_init,
                    original_input_dir,
                    output_merged,
                    tolerance=0.1,
                    retile_buffer=retile_buffer,
                    num_threads=num_threads,
                )
                print(f"  Enriched merged file with original-file dimensions: {output_merged}")
            except Exception as e:
                print(f"  Warning: Could not add original dimensions to merged file: {e}")
                shutil.copy2(str(merged_init), str(output_merged))
                print(f"  Wrote un-enriched merged to {output_merged}")
        else:
            shutil.copy2(str(merged_init), str(output_merged))
            print(f"  Saved merged output: {output_merged}")

        validate_merged_output_contract(output_merged, instance_dimension)

        try:
            merged_init.unlink()
        except OSError:
            pass

        # Also save a copy to output_tiles_dir for convenience
        merged_copy_path = output_tiles_dir / output_merged.name
        if merged_copy_path != output_merged:
            shutil.copy2(str(output_merged), str(merged_copy_path))
            print(f"  Copied to output tiles folder: {merged_copy_path}")

    # =========================================================================
    # Create CSV with instance metadata
    # =========================================================================
    import csv

    csv_output_path = output_merged.parent / f"{output_merged.stem}_instance_metadata.csv"

    instances_with_clusters = set()
    for old_merged_id, sources in merged_instance_sources.items():
        if len(sources) > 1:
            final_id = old_to_new.get(old_merged_id, None)
            if final_id is not None and final_id > 0:
                instances_with_clusters.add(final_id)

    print(f"\n  Writing instance metadata CSV: {csv_output_path}")
    print(f"  Found {len(instances_with_clusters)} final instances with added clusters from cross-tile merging")

    # Collect all final instance IDs
    final_instance_ids = sorted(set(old_to_new.values()) - {0})

    with open(csv_output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([instance_dimension, "has_added_clusters"])

        for final_id in final_instance_ids:
            has_clusters = final_id in instances_with_clusters
            writer.writerow([final_id, 1 if has_clusters else 0])

    csv_copy_path = output_tiles_dir / csv_output_path.name
    if csv_copy_path != csv_output_path:
        import shutil
        shutil.copy2(str(csv_output_path), str(csv_copy_path))
        print(f"  Copied CSV to output tiles folder: {csv_copy_path}")

    del merged_instance_sources
    gc.collect()

    # =========================================================================
    # Stage 6: Retile to Original Files (Required)
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("Stage 6: Retiling to Original Files")
    print(f"{'=' * 60}")

    retile_to_original_files(
        merged_points,
        merged_instances,
        merged_extra_dims,
        None,
        original_tiles_dir,
        output_tiles_dir,
        tolerance=0.1,
        num_threads=num_threads,
        retile_buffer=retile_buffer,
        instance_dimension=instance_dimension,
    )
    print(f"  ✓ Stage 6 completed: Retiled to original files")

    # =========================================================================
    # Stage 7: Remap to Original Input Files (if provided)
    # =========================================================================
    if original_input_dir is not None:
        print(f"\n{'=' * 60}")
        print("Stage 7: Remapping to Original Input Files")
        print(f"{'=' * 60}")

        original_output_dir = output_tiles_dir.parent / "original_with_predictions"
        all_merged_dims = {instance_dimension: merged_instances, **merged_extra_dims}
        remap_to_original_input_files(
            merged_points,
            all_merged_dims,
            None,
            original_input_dir,
            original_output_dir,
            tolerance=0.1,
            num_threads=num_threads,
            retile_buffer=retile_buffer,
            threedtrees_dims=threedtrees_dims,
            threedtrees_suffix=threedtrees_suffix,
            chunk_size=chunk_size,
        )
        print(f"  ✓ Stage 7 completed: Remapped to original input files")
    else:
        print(f"\n  Note: --original-input-dir not provided, skipping Stage 7 (remap to original input files)")

    print(f"\n{'=' * 60}")
    print("Merge complete!")
    print(f"{'=' * 60}")


# =============================================================================
# CLI
# =============================================================================


def main():
    from merge_tiles_cli import main as cli_main

    cli_main(merge_tiles)


if __name__ == "__main__":
    main()
