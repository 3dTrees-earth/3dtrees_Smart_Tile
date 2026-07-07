#!/usr/bin/env python3
"""
Main orchestrator script for the 3DTrees smart tiling pipeline.

Routes to appropriate task modules based on --task parameter:
- tile: XYZ reduction, COPC conversion, tiling, and subsampling (1cm and 10cm)
- merge: Remap predictions and merge tiles with instance matching
- filter: Remove duplicate buffer-zone instances from segmented/remapped tiles
- remap: Remap merged file dimensions to original input files
- create_merged_file: Create prod-merged files from original_with_predictions

Usage:
    python src/run.py --task tile --input-dir /path/to/input --output-dir /path/to/output
    python src/run.py --task merge --subsampled-10cm-folder /path/to/10cm --original-input-dir /path/to/input
    python src/run.py --task filter --input-dir /path/to/segmented_remapped --output-dir /path/to/filtered
    python src/run.py --task remap --merged-laz /path/to/merged.laz --original-laz-input-dir /path/to/originals --original-laz-output-dir /path/to/output
    python src/run.py --task create_merged_file --original-with-predictions-dir /path/to/original_with_predictions --output-dir /path/to/output
"""

import sys
import argparse
import os
from pathlib import Path

# Add src directory to path for imports when run from project root
_src_dir = Path(__file__).parent.resolve()
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

# Import Pydantic-based parameters
try:
    from parameters import Parameters, print_params, get_tile_params, get_merge_params, get_remap_params
except ImportError as e:
    print(f"Error: Could not import parameters.py: {e}")
    print("Please install required dependencies: pip install pydantic pydantic-settings")
    sys.exit(1)


def _parse_csv(value: str | None) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()] if value else []


def _semantic_dimension_for_instance(instance_dimension: str) -> str | None:
    """Infer the matching semantic prediction dimension for a PredInstance name."""
    if not instance_dimension.startswith("PredInstance"):
        return None
    return f"PredSemantic{instance_dimension[len('PredInstance'):]}"


def _effective_threedtrees_dims(params: Parameters) -> list[str] | None:
    """Return prediction dimensions to transfer for merge/remap original enrichment."""
    dims = _parse_csv(params.threedtrees_dims)
    if dims == ["PredInstance", "PredSemantic"] and params.instance_dimension != "PredInstance":
        dims = [params.instance_dimension]
        semantic_dimension = _semantic_dimension_for_instance(params.instance_dimension)
        if semantic_dimension:
            dims.append(semantic_dimension)
    return dims or None


def _create_prod_merged_outputs(
    original_with_predictions_dir: Path,
    output_dir: Path,
    params: Parameters,
) -> None:
    """Create prod-merged outputs using the shared create_merged_file implementation."""
    if not original_with_predictions_dir.exists():
        print(f"  Skipping prod-merged outputs; missing {original_with_predictions_dir}")
        return

    try:
        from main_create_merged_file import create_prod_merged_files
    except ImportError as e:
        print(f"Error: Could not import main_create_merged_file.py: {e}")
        sys.exit(1)

    print()
    print("=" * 60)
    print("Creating Prod-Merged Files")
    print("=" * 60)
    print(f"Original-with-predictions dir: {original_with_predictions_dir}")
    print(f"Output dir: {output_dir}")
    if params.staged_copc_dir:
        print(f"Staged COPC dir: {params.staged_copc_dir}")
    if params.standardization_json:
        print(f"Standardization JSON: {params.standardization_json}")
    print(f"Selected resolutions: {params.merged_resolutions}")
    print(f"Selected output formats: {params.merged_output_formats}")
    print("Product subsampling method: nearest-to-centroid")
    print()

    outputs = create_prod_merged_files(
        original_with_predictions_dir=original_with_predictions_dir,
        output_dir=output_dir,
        resolution_selector=params.merged_resolutions,
        output_format_selector=params.merged_output_formats,
        res1=params.resolution_1,
        res2=params.resolution_2,
        num_spatial_chunks=params.num_spatial_chunks or params.workers,
        chunk_workers=params.workers,
        staged_copc_dir=params.staged_copc_dir,
        standardization_json=params.standardization_json,
    )
    print("  Prod-merged outputs:")
    for output in outputs:
        print(f"    {output}")


def _raw_original_output_dir(params: Parameters, raw_input_dir: Path) -> Path:
    """Return the output directory for raw uploaded Original-with-predictions files."""
    if params.original_raw_output_dir:
        return Path(params.original_raw_output_dir)
    if params.output_dir:
        return Path(params.output_dir)
    return raw_input_dir.parent / "original_with_predictions"


def _transfer_dims_for_instance(params: Parameters) -> list[str]:
    """Return prediction dimensions to transfer, including configured model names."""
    dims = [d.strip() for d in params.threedtrees_dims.split(",") if d.strip()] if params.threedtrees_dims else []
    if params.instance_dimension and params.instance_dimension not in dims:
        dims.append(params.instance_dimension)
    if params.instance_dimension and "Instance" in params.instance_dimension:
        semantic_dim = params.instance_dimension.replace("Instance", "Semantic", 1)
        if semantic_dim not in dims:
            dims.append(semantic_dim)
    return dims


def _comma_paths(value: str | None) -> list[Path]:
    """Parse a comma-separated path list from CLI/Galaxy parameters."""
    if not value:
        return []
    return [Path(part.strip()) for part in str(value).split(",") if part.strip()]


def _prepare_merge_prediction_collection_source(
    *,
    prediction_collections: list[Path],
    reference_dir: Path | None,
    output_folder: Path,
    params: Parameters,
    retile_buffer: float,
    workers: int,
) -> Path:
    """Create one merge-ready segmented folder from one or more prediction collections."""
    from prediction_collection_remap import remap_prediction_collections_to_original_files

    if not prediction_collections:
        raise ValueError("At least one segmented collection is required")

    for collection in prediction_collections:
        if not collection.exists():
            raise FileNotFoundError(f"Segmented collection not found: {collection}")

    if reference_dir is None:
        reference_dir = prediction_collections[0]
        collections_to_add = prediction_collections[1:]
    else:
        collections_to_add = prediction_collections

    if not reference_dir.exists():
        raise FileNotFoundError(f"Reference segmented folder not found: {reference_dir}")

    if not collections_to_add:
        return reference_dir

    combined_dir = output_folder / "segmented_collections_combined"
    target_dims = {d.strip() for d in params.remap_dims.split(",") if d.strip()} if params.remap_dims else None
    print()
    print("=" * 60)
    print("Preparing multi-collection merge source")
    print("=" * 60)
    print(f"Reference geometry/source: {reference_dir}")
    print(f"Additional collections: {[str(path) for path in collections_to_add]}")
    print(f"Combined merge source: {combined_dir}")
    print(f"Remap tolerance: {params.remap_tolerance}m")
    print()

    remap_prediction_collections_to_original_files(
        collections_to_add,
        reference_dir,
        combined_dir,
        tolerance=params.remap_tolerance,
        num_threads=workers,
        retile_buffer=retile_buffer,
        target_dims=target_dims,
        chunk_size=params.chunk_size or 5_000_000,
        num_spatial_chunks=params.num_spatial_chunks or params.workers,
    )
    return combined_dir


def _validate_raw_original_lane(
    raw_input_dir: Path,
    raw_output_dir: Path,
) -> None:
    """Fail early on ambiguous raw-download lane configuration."""
    if not raw_input_dir.exists():
        print(f"Error: LAZ original input directory not found: {raw_input_dir}")
        sys.exit(1)
    if raw_output_dir.resolve() == raw_input_dir.resolve():
        print(
            "Error: --original-laz-output-dir/--output-dir must differ from --original-laz-input-dir."
        )
        sys.exit(1)


def run_tile_task(params: Parameters):
    """
    Run the tile task: COPC conversion, tiling, and subsampling.

    Pipeline:
    1. Convert LAZ/LAS inputs to intermediate COPC with standard LAS dimensions by default
    2. Build spatial index
    3. Calculate tile bounds
    4. Create overlapping tiles
    5. Subsample to resolution 1 (1cm by default)
    6. Subsample to resolution 2 (10cm)
    """
    # Import Python modules
    try:
        from main_tile import run_tiling_pipeline
        from main_subsample import run_subsample_pipeline
    except ImportError as e:
        print(f"Error: Could not import required modules: {e}")
        print("Make sure main_tile.py and main_subsample.py exist.")
        sys.exit(1)

    # Required arguments
    if not params.input_dir:
        print("Error: --input-dir is required for tile task")
        sys.exit(1)
    if not params.output_dir:
        print("Error: --output-dir is required for tile task")
        sys.exit(1)

    # Validate input directory
    input_dir = Path(params.input_dir)
    output_dir = Path(params.output_dir)

    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)

    # Get parameters from Pydantic model
    tile_length = params.tile_length
    tile_buffer = params.tile_buffer
    threads = params.threads
    workers = params.workers
    dimension_reduction = True
    num_spatial_chunks = params.num_spatial_chunks
    subsampling_chunks = num_spatial_chunks or workers
    res1 = params.resolution_1
    res2 = params.resolution_2
    output_copc_res1 = params.output_copc_res1
    output_copc_res2 = params.output_copc_res2
    subsampling_method = params.subsampling_method
    tiling_threshold = params.tiling_threshold
    chunk_size = params.chunk_size
    print("=" * 60)
    print("Running Tile Task (Python Pipeline)")
    print("=" * 60)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Tile length: {tile_length}m")
    print(f"Tile buffer: {tile_buffer}m")
    print(f"Workers: {workers}")
    print(f"Threads per writer: {threads}")
    print(f"Tile source file workers: {tile_source_workers}")
    print(f"Tile finalization workers: {tile_writer_workers}")
    print(f"Subsampling spatial chunks/window workers: {subsampling_chunks}")
    print("Subsampling dimensions: minimal (standard dims only)")
    print(f"Subsampling method: {subsampling_method}")
    print(f"Resolutions: {res1}m ({int(res1*100)}cm), {res2}m ({int(res2*100)}cm)")
    print(f"Resolution 1 output: {'COPC LAZ' if output_copc_res1 else 'LAZ'}")
    print(f"Resolution 2 output: {'COPC LAZ' if output_copc_res2 else 'LAZ'}")
    if tiling_threshold is not None:
        print(f"Tiling threshold: {tiling_threshold} MB")
    print(f"Chunk size: {chunk_size:,} points")
    print()

    try:
        # Step 1-4: Tiling pipeline
        tiles_dir = run_tiling_pipeline(
            input_dir=input_dir,
            output_dir=output_dir,
            tile_length=tile_length,
            tile_buffer=tile_buffer,
            num_workers=workers,
            threads=threads,
            max_tile_procs=tile_writer_workers,
            source_file_workers=tile_source_workers,
            dimension_reduction=dimension_reduction,
            tiling_threshold=tiling_threshold,
            chunk_size=chunk_size,
        )

        # Check if tiling was skipped (returns copc_dir instead of tiles_dir)
        tiling_skipped = tiles_dir.name.startswith("copc_")

        if tiling_skipped:
            # Single file case - create tiles_* directory structure for consistency
            # Move COPC files to tiles_* directory so subsampling creates consistent structure
            tiles_dir_normalized = output_dir / f"tiles_{int(tile_length)}m"
            original_copc_dir = output_dir / "original_copc"
            tiles_dir_normalized.mkdir(exist_ok=True)
            original_copc_dir.mkdir(exist_ok=True)

            # Copy/move COPC files to tiles directory
            import shutil
            for copc_file in tiles_dir.glob("*.copc.laz"):
                dest_file = tiles_dir_normalized / copc_file.name
                if not dest_file.exists():
                    try:
                        shutil.copy2(copc_file, dest_file)
                    except OSError as exc:
                        print(
                            "  Warning: metadata-preserving copy failed "
                            f"({exc}); retrying as data-only copy"
                        )
                        shutil.copyfile(copc_file, dest_file)
                original_copc_file = original_copc_dir / copc_file.name
                if not original_copc_file.exists():
                    try:
                        shutil.copy2(copc_file, original_copc_file)
                    except OSError as exc:
                        print(
                            "  Warning: metadata-preserving original COPC copy failed "
                            f"({exc}); retrying as data-only copy"
                        )
                        shutil.copyfile(copc_file, original_copc_file)

            # Update tiles_dir to use normalized structure
            tiles_dir = tiles_dir_normalized
            output_prefix = f"{output_dir.name}_{int(tile_length)}m"
            print(f"  Note: Tiling was skipped, using normalized directory structure: {tiles_dir}")
        else:
            # Normal tiled case
            output_prefix = f"{output_dir.name}_{int(tile_length)}m"

        # Step 5-6: Subsampling pipeline
        res1_dir, res2_dir = run_subsample_pipeline(
            tiles_dir=tiles_dir,
            res1=res1,
            res2=res2,
            num_cores=workers,
            num_threads=subsampling_chunks,
            output_prefix=output_prefix,
            output_base_dir=output_dir,  # Output directly to output_dir, not under tiles_dir
            dimension_reduction=dimension_reduction,
            subsampling_method=subsampling_method,
            output_copc_res1=output_copc_res1,
            output_copc_res2=output_copc_res2,
        )

        # Step 7: Update tile_bounds_tindex.json with actual bounds from created tiles
        # (so remap/merge matching uses file extent instead of nominal grid)
        bounds_json = output_dir / "tile_bounds_tindex.json"
        if bounds_json.exists():
            from main_tile import update_tile_bounds_json_from_files
            num_updated = update_tile_bounds_json_from_files(bounds_json, res1_dir)
            if num_updated > 0:
                print(f"  Updated tile_bounds_tindex.json with bounds from {num_updated} tile(s) in {res1_dir.name}")

        print()
        print("=" * 60)
        print("Tile Task Complete")
        print("=" * 60)
        print(f"Tiles: {tiles_dir}")
        print(f"Subsampled {int(res1*100)}cm: {res1_dir}")
        print(f"Subsampled {int(res2*100)}cm: {res2_dir}")

        # Return the input_dir for use in merge task if needed
        return input_dir

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_merge_task(params: Parameters):
    """
    Run the merge task: filter predictions, remap if needed, and merge tiles.

    Pipeline:
    1. Filter duplicate buffer-zone instances from segmented predictions
    2. Remap filtered predictions from 10cm to target resolution (via main_remap.py)
    3. Merge target-resolution tiles with instance matching (via main_merge.py)
    4. Remap to original input files with the shared remap helper (if configured)
    """
    # Import Python modules
    try:
        from instance_labels import MERGED_OUTPUT_SCALES
        from filter_buffer_instances import filter_buffer_instances_dir
        from main_remap import remap_all_tiles
        from main_merge import run_merge
        from prediction_collection_remap import remap_prediction_collections_to_original_files
    except ImportError as e:
        print(f"Error: Could not import required modules: {e}")
        print("Make sure main_remap.py and main_merge.py exist.")
        sys.exit(1)

    # Required arguments - need either a 10cm segmented source, multiple finalized
    # prediction collections, or an already-remapped segmented folder.
    # Note: subsampled_10cm_folder is populated by --subsampled-segmented-folder via alias
    if not params.subsampled_10cm_folder and not params.segmented_folders and not params.segmented_remapped_folder:
        print(
            "Error: --subsampled-segmented-folder, --segmented-folders, "
            "or --segmented-remapped-folder is required for merge task"
        )
        sys.exit(1)
    if params.subsampled_10cm_folder and params.segmented_remapped_folder:
        print(
            "Error: --subsampled-segmented-folder/--subsampled-10cm-folder and "
            "--segmented-remapped-folder are mutually exclusive for merge task"
        )
        sys.exit(1)

    # Get parameters from Pydantic model
    workers = params.workers
    buffer = None
    overlap_threshold = params.overlap_threshold
    max_centroid_distance = params.max_centroid_distance
    max_volume_for_merge = params.max_volume_for_merge
    border_zone_width = params.border_zone_width
    min_cluster_size = params.min_cluster_size
    retile_buffer = 2.0  # Fixed to 2.0m
    output_merged = params.output_merged_laz
    output_tiles_dir = params.output_tiles_folder
    original_tiles_dir = params.original_tiles_dir
    original_input_dir = params.original_raw_input_dir or params.original_input_dir

    print("=" * 60)
    print("Running Merge Task (Python Pipeline)")
    print("=" * 60)

    try:
        if params.original_copc_input_dir and original_input_dir:
            _validate_copc_original_lane(Path(params.original_copc_input_dir))
            _validate_copc_laz_source_pairs(Path(params.original_copc_input_dir), Path(original_input_dir))
        elif params.original_copc_input_dir:
            print(
                "Error: --original-laz-input-dir is required when --original-copc-input-dir "
                "is used for merge/remap-to-originals."
            )
            sys.exit(1)

        def merge_work_dir(input_folder: Path) -> Path:
            if params.output_folder:
                return Path(params.output_folder)
            if output_tiles_dir:
                return Path(output_tiles_dir).parent
            if output_merged:
                return Path(output_merged).parent
            return input_folder.parent

        def filter_predictions(input_folder: Path) -> Path:
            filtered_folder = merge_work_dir(input_folder) / "segmented_filtered"
            print()
            print("=" * 60)
            print("Filtering Segmented Predictions")
            print("=" * 60)
            filter_summary = filter_buffer_instances_dir(
                input_dir=input_folder,
                output_dir=filtered_folder,
                buffer=buffer,
                suffix="_filtered",
                instance_dimension=params.instance_dimension,
                output_extension=".laz",
            )
            if filter_summary["input_files"] == 0:
                print(f"Error: No filterable prediction files found in {input_folder}")
                sys.exit(1)
            return filtered_folder

        # Step 1: Filter predictions, then remap if a source-resolution folder is provided.
        segmented_for_merge_folder = None
        segmented_source_folder = None
        filtered_segmented_folder = None

        if params.subsampled_10cm_folder:
            subsampled_10cm_dir = Path(params.subsampled_10cm_folder)
            segmented_source_folder = subsampled_10cm_dir

        if params.subsampled_10cm_folder or prediction_collections:
            subsampled_10cm_dir = Path(params.subsampled_10cm_folder) if params.subsampled_10cm_folder else None

            if subsampled_10cm_dir is not None and not subsampled_10cm_dir.exists():
                print(f"Error: Input directory does not exist: {subsampled_10cm_dir}")
                sys.exit(1)

            print(f"Input (10cm): {subsampled_10cm_dir or prediction_collections}")
            print()

            # Derive target folder and output folder
            # The resolution folders are now at: tiles_*/subsampled_res1 and tiles_*/subsampled_res2
            # For backward compatibility, also check old naming: subsampled_{resolution}cm
            parent_dir = (
                subsampled_10cm_dir.parent
                if subsampled_10cm_dir is not None
                else prediction_collections[0].parent
            )
            target_folder = params.subsampled_target_folder

            if target_folder is None:
                # Try new naming first (subsampled_res1) as default target
                target_folder_res1 = parent_dir / "subsampled_res1"
                if target_folder_res1.exists():
                    target_folder = target_folder_res1
                else:
                    # Fallback or error
                    pass

            output_folder = params.output_folder
            if output_folder is None:
                output_folder = merge_work_dir(subsampled_10cm_dir) / "segmented_remapped"

            if target_folder is None or not target_folder.exists():
                print(f"Error: Target resolution folder does not exist or not specified")
                if target_folder:
                    print(f"Path: {target_folder}")
                print(f"Please provide --subsampled-target-folder")
                sys.exit(1)

            filtered_segmented_folder = filter_predictions(subsampled_10cm_dir)

            # Optional: tile_bounds_tindex.json for remap matching (use --tile_bounds_json first)
            remap_tile_bounds_json = None
            if tile_bounds_json.exists():
                remap_tile_bounds_json = tile_bounds_json
            if remap_tile_bounds_json is None:
                remap_json_candidates = [
                    parent_dir / "tile_bounds_tindex.json",
                    subsampled_10cm_dir / "tile_bounds_tindex.json",
                ]
                if params.original_tiles_dir:
                    remap_json_candidates.insert(0, Path(params.original_tiles_dir) / "tile_bounds_tindex.json")
                for p in remap_json_candidates:
                    if p.exists():
                        remap_tile_bounds_json = p
                        break

            # Remap - source is filtered segmented predictions, target is the configured resolution-1 subsample
            segmented_for_merge_folder = remap_all_tiles(
                source_folder=filtered_segmented_folder,
                target_folder=target_folder,
                output_folder=output_folder,
                tile_bounds_json=remap_tile_bounds_json,
                verbose=bool(params.verbose),
                num_workers=workers,
                instance_dimension=params.instance_dimension,
                output_scales=tuple(MERGED_OUTPUT_SCALES),
            )

        elif params.segmented_remapped_folder:
            segmented_source_folder = Path(params.segmented_remapped_folder)
            if not segmented_source_folder.exists():
                print(f"Error: Segmented folder does not exist: {segmented_source_folder}")
                sys.exit(1)
            filtered_segmented_folder = filter_predictions(segmented_source_folder)
            segmented_for_merge_folder = filtered_segmented_folder

        if segmented_for_merge_folder is None:
            print("Error: No segmented folder available for merge")
            sys.exit(1)

        print()
        if segmented_source_folder is not None:
            print(f"Segmented folder: {segmented_source_folder}")
        print(f"Filtered segmented folder: {filtered_segmented_folder}")
        print(f"Merge input folder: {segmented_for_merge_folder}")
        print(f"Buffer: {buffer}m")
        print(f"Overlap threshold: {overlap_threshold}")
        print(f"Workers: {workers}")
        if params.original_raw_input_dir or params.original_input_dir:
            print(f"LAZ original input dir: {params.original_raw_input_dir or params.original_input_dir}")
        print()

        # Auto-derive paths if not provided
        parent_dir = Path(segmented_for_merge_folder).parent
        if output_tiles_dir is None:
            # Use segmented folder's parent, but ensure it's writable
            # If parent is root or not writable, use segmented folder itself
            if parent_dir == Path('/') or not os.access(parent_dir, os.W_OK):
                output_tiles_dir = Path(segmented_for_merge_folder) / "output_tiles"
            else:
                output_tiles_dir = parent_dir / "output_tiles"
        if original_tiles_dir is None:
            # Try to find the tiles directory (parent of subsampled folders)
            original_tiles_dir = Path(segmented_for_merge_folder).parent
        original_with_predictions_dir = None
        if original_input_dir:
            original_with_predictions_dir = (
                Path(params.original_raw_output_dir)
                if params.original_raw_output_dir
                else Path(output_tiles_dir).parent / "original_with_predictions"
            )
            _validate_raw_original_lane(
                Path(original_input_dir),
                original_with_predictions_dir,
            )

        # Parse 3DTrees dimension branding params
        threedtrees_dims = _effective_threedtrees_dims(params)
        threedtrees_suffix = params.threedtrees_suffix

        merged_output = run_merge(
            segmented_dir=segmented_for_merge_folder,
            output_tiles_dir=output_tiles_dir,
            original_tiles_dir=original_tiles_dir,
            tile_bounds_json=tile_bounds_json,
            original_input_dir=None,
            output_merged=output_merged,
            overlap_threshold=overlap_threshold,
            max_centroid_distance=max_centroid_distance,
            max_volume_for_merge=max_volume_for_merge,
            border_zone_width=border_zone_width,
            min_cluster_size=min_cluster_size,
            num_threads=workers,
            enable_matching=not params.disable_matching,
            require_overlap=True,
            enable_volume_merge=not params.disable_volume_merge,
            skip_merged_file=params.skip_merged_file,
            verbose=params.verbose,
            retile_buffer=retile_buffer,
            instance_dimension=params.instance_dimension,
            transfer_original_dims_to_merged=False,
            threedtrees_dims=threedtrees_dims,
            threedtrees_suffix=threedtrees_suffix,
            chunk_size=params.chunk_size or 1_000_000,
        )

        if original_input_dir:
            product_output_dir = Path(merged_output).parent if merged_output else Path(output_tiles_dir).parent
            print()
            print("=" * 60)
            print("Remapping merged 1cm tiles to uploaded originals")
            print("=" * 60)
            remap_prediction_collections_to_original_files(
                [Path(output_tiles_dir)],
                Path(original_input_dir),
                original_with_predictions_dir,
                tolerance=0.1,
                num_threads=workers,
                retile_buffer=retile_buffer,
                target_dims=set(threedtrees_dims) if threedtrees_dims else None,
                chunk_size=params.chunk_size or 5_000_000,
                num_spatial_chunks=params.num_spatial_chunks or params.workers,
                prefer_copc_sources=False,
            )

        if original_input_dir and params.transfer_original_dims_to_merged:
            _create_prod_merged_outputs(
                original_with_predictions_dir=original_with_predictions_dir,
                output_dir=product_output_dir,
                params=params,
            )
        elif original_input_dir:
            print("  Skipping prod-merged output creation (disabled).")

        print()
        print("=" * 60)
        print("Merge Task Complete")
        print("=" * 60)
        if params.skip_merged_file:
            print("Merged output: skipped")
        else:
            print(f"Merged output: {merged_output}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_remap_task(params: Parameters):
    """
    Run the remap task.

    Supported modes:
    - --segmented-folders: finalized prediction collections -> original files folder.
      Extra dimensions are preserved as-is and duplicate prediction names fail.
    - --merged-laz: one merged LAZ/COPC LAZ file -> original files folder.
      3DTrees dimensions are suffixed during transfer.
    """
    try:
        from prediction_collection_remap import remap_prediction_collections_to_original_files
        from merge_tiles import (
            load_merged_file,
            reassign_small_instances_in_dims,
            remap_to_original_input_files,
        )
        from output_remap import remap_merged_file_to_original_input_files
        from point_cloud_outputs import write_loaded_point_cloud
    except ImportError as e:
        print(f"Error: Could not import required modules: {e}")
        sys.exit(1)

    if not params.original_raw_input_dir and not params.original_input_dir:
        print("Error: --original-laz-input-dir is required for remap task")
        print("       Legacy --original-input-dir is still accepted as a LAZ/LAS source for compatibility.")
        sys.exit(1)
    if not params.segmented_folders and not params.merged_laz:
        print("Error: --segmented-folders or --merged-laz is required for remap task")
        sys.exit(1)
    if params.segmented_folders and params.merged_laz:
        print("Error: --segmented-folders and --merged-laz are mutually exclusive for remap task")
        sys.exit(1)
    if params.segmented_folders and params.pre_remap_reassign_instances:
        print("Error: --pre-remap-reassign-instances is only supported with --merged-laz")
        sys.exit(1)
    if params.merged_laz and params.remap_dims:
        print("Error: --remap-dims is only supported with --segmented-folders; use --threedtrees-dims with --merged-laz")
        sys.exit(1)

    laz_input_dir = Path(params.original_raw_input_dir or params.original_input_dir)
    laz_output_dir = _raw_original_output_dir(params, laz_input_dir)
    _validate_raw_original_lane(laz_input_dir, laz_output_dir)

    workers = max(1, params.workers)
    retile_buffer = 2.0
    tolerance = params.remap_tolerance

    if params.segmented_folders:
        collections = [Path(p.strip()) for p in params.segmented_folders.split(",") if p.strip()]
        target_dims = {d.strip() for d in params.remap_dims.split(",") if d.strip()} if params.remap_dims else None

        print("=" * 60)
        print("Remap: prediction collections -> original files")
        print("=" * 60)
        print(f"Collections: {[str(c) for c in collections]}")
        print(f"LAZ original input dir: {laz_input_dir}")
        print(f"LAZ output dir: {laz_output_dir}")
        print(f"Remap dims: {sorted(target_dims) if target_dims else 'all extra dimensions'}")
        print()

        try:
            print()
            print("=" * 60)
            print("Remap: prediction collections -> uploaded LAZ originals")
            print("=" * 60)
            remap_prediction_collections_to_original_files(
                collections,
                laz_input_dir,
                laz_output_dir,
                tolerance=tolerance,
                num_threads=workers,
                retile_buffer=retile_buffer,
                target_dims=target_dims,
                chunk_size=params.chunk_size or 5_000_000,
                num_spatial_chunks=params.num_spatial_chunks or params.workers,
            )
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

        if params.transfer_original_dims_to_merged:
            product_output_dir = laz_output_dir.parent
            _create_prod_merged_outputs(
                original_with_predictions_dir=laz_output_dir,
                output_dir=product_output_dir,
                params=params,
            )
        else:
            print("  Skipping prod-merged output creation (disabled).")

        print()
        print("Remap complete.")
        return

    merged_laz = Path(params.merged_laz)
    if not merged_laz.exists():
        print(f"Error: Merged file not found: {merged_laz}")
        sys.exit(1)

    print("=" * 60)
    print("Remap: merged file -> original files")
    print("=" * 60)
    print(f"Merged file: {merged_laz}")
    print(f"LAZ original input dir: {laz_input_dir}")
    print(f"LAZ output dir: {laz_output_dir}")
    print()

    # Parse 3DTrees dimension branding params
    threedtrees_dims = _effective_threedtrees_dims(params)
    threedtrees_suffix = params.threedtrees_suffix

    if not params.pre_remap_reassign_instances:
        remap_merged_file_to_original_input_files(
            merged_laz,
            laz_input_dir,
            laz_output_dir,
            tolerance=tolerance,
            num_threads=workers,
            retile_buffer=retile_buffer,
            threedtrees_dims=threedtrees_dims,
            threedtrees_suffix=threedtrees_suffix,
            num_spatial_chunks=params.num_spatial_chunks or params.workers,
            chunk_size=params.chunk_size or 5_000_000,
        )
    else:
        merged_points, merged_extra_dims, merged_extra_dim_params = load_merged_file(merged_laz)
        candidate_dims = threedtrees_dims or []
        instance_dimension = params.pre_remap_reassign_instance_dimension
        if instance_dimension is None:
            instance_dimension = next((d for d in candidate_dims if "instance" in d.lower()), None)
        if instance_dimension is None:
            print("Error: --pre-remap-reassign-instances requires an instance dimension")
            sys.exit(1)

        print()
        print("Pre-remap small instance reassignment")
        print(f"  Instance dimension: {instance_dimension}")
        print(f"  Reassign point-count clusters below: {params.pre_remap_reassign_min_cluster_size}")
        print(f"  Hull check for clusters below: {params.pre_remap_reassign_hull_point_threshold}")
        print(f"  Reassign hull volume below: {params.pre_remap_reassign_max_volume} m3")
        reassignment_stats = reassign_small_instances_in_dims(
            merged_points,
            merged_extra_dims,
            instance_dimension=instance_dimension,
            min_cluster_size=params.pre_remap_reassign_min_cluster_size,
            hull_point_threshold=params.pre_remap_reassign_hull_point_threshold,
            max_volume_for_merge=params.pre_remap_reassign_max_volume,
            max_search_radius=float("inf"),
            num_threads=workers,
            verbose=bool(params.verbose),
        )
        print(
            "  Reassignment result: "
            f"{reassignment_stats['changed_points']:,} points changed, "
            f"{reassignment_stats['instances_before']:,} -> "
            f"{reassignment_stats['instances_after']:,} instances"
        )
        if params.pre_remap_reassigned_laz:
            reassigned_laz = Path(params.pre_remap_reassigned_laz)
            print(f"  Saving reassigned segmented point cloud: {reassigned_laz}")
            write_loaded_point_cloud(
                source_file=merged_laz,
                output_file=reassigned_laz,
                points=merged_points,
                all_dims=merged_extra_dims,
            )

        remap_to_original_input_files(
            merged_points,
            merged_extra_dims,
            merged_extra_dim_params,
            laz_input_dir,
            laz_output_dir,
            tolerance=tolerance,
            num_threads=workers,
            retile_buffer=retile_buffer,
            threedtrees_dims=threedtrees_dims,
            threedtrees_suffix=threedtrees_suffix,
            num_spatial_chunks=params.num_spatial_chunks or params.workers,
            chunk_size=params.chunk_size or 1_000_000,
        )

    # Create prod-merged files from the enriched original outputs (optional).
    if params.transfer_original_dims_to_merged:
        product_output_dir = laz_output_dir.parent
        _create_prod_merged_outputs(
            original_with_predictions_dir=laz_output_dir,
            output_dir=product_output_dir,
            params=params,
        )
    else:
        print("  Skipping prod-merged output creation (disabled).")

    print()
    print("Remap complete.")


def run_create_merged_file_task(params: Parameters):
    """Create prod-merged files from Original-with-predictions files."""
    try:
        from main_create_merged_file import create_prod_merged_files
    except ImportError as e:
        print(f"Error: Could not import main_create_merged_file.py: {e}")
        sys.exit(1)

    input_dir = params.original_with_predictions_dir or params.input_dir
    if input_dir is None:
        print("Error: --original-with-predictions-dir or --input-dir is required for create_merged_file task")
        sys.exit(1)

    output_dir = params.output_dir
    if output_dir is None:
        output_dir = Path(input_dir).parent
    else:
        output_dir = Path(output_dir)

    print("=" * 60)
    print("Create Prod-Merged Files")
    print("=" * 60)
    print(f"Original-with-predictions dir: {input_dir}")
    print(f"Output dir: {output_dir}")
    if params.staged_copc_dir:
        print(f"Staged COPC dir: {params.staged_copc_dir}")
    if params.standardization_json:
        print(f"Standardization JSON: {params.standardization_json}")
    print(f"Selected resolutions: {params.merged_resolutions}")
    print(f"Selected output formats: {params.merged_output_formats}")
    print(f"Resolution 1: {params.resolution_1:g}m")
    print(f"Resolution 2: {params.resolution_2:g}m")
    print("Product subsampling method: nearest-to-centroid")
    print()

    try:
        outputs = create_prod_merged_files(
            original_with_predictions_dir=Path(input_dir),
            output_dir=output_dir,
            resolution_selector=params.merged_resolutions,
            output_format_selector=params.merged_output_formats,
            res1=params.resolution_1,
            res2=params.resolution_2,
            num_spatial_chunks=params.num_spatial_chunks or params.workers,
            chunk_workers=params.workers,
            staged_copc_dir=params.staged_copc_dir,
            standardization_json=params.standardization_json,
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print()
    print("Create merged file complete.")
    for output in outputs:
        print(f"  {output}")


def run_filter_task(params: Parameters):
    """Filter buffer-zone duplicate instances from segmented/remapped tiles."""
    try:
        from filter_buffer_instances import filter_buffer_instances_dir
    except ImportError as e:
        print(f"Error: Could not import filter_buffer_instances.py: {e}")
        sys.exit(1)

    if not params.input_dir:
        print("Error: --input-dir is required for filter task")
        sys.exit(1)
    if not params.output_dir:
        print("Error: --output-dir is required for filter task")
        sys.exit(1)

    input_dir = Path(params.input_dir)
    output_dir = Path(params.output_dir)
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)
    if output_dir.resolve() == input_dir.resolve() and params.filter_suffix == "":
        print("Error: --output-dir must differ from --input-dir when --filter-suffix is empty")
        sys.exit(1)

    print("=" * 60)
    print("Running Filter Task")
    print("=" * 60)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Buffer: {params.buffer}m")
    print(f"Instance dimension: {params.instance_dimension}")
    print(f"Output suffix: {params.filter_suffix!r}")
    if params.filter_output_extension:
        print(f"Output extension: {params.filter_output_extension}")
    print()

    try:
        summary = filter_buffer_instances_dir(
            input_dir=input_dir,
            output_dir=output_dir,
            buffer=params.buffer,
            suffix=params.filter_suffix,
            instance_dimension=params.instance_dimension,
            output_extension=params.filter_output_extension,
        )
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    print()
    print("Filter task complete.")
    print(f"  Files processed: {summary['input_files']}")
    print(f"  Output files: {len(summary['output_files'])}")


def preprocess_boolean_flags(args_list):
    """
    Preprocess CLI args to convert boolean flags to explicit True/False for Pydantic.
    Pydantic expects --flag True/False, but we want --flag to work like argparse.
    """
    boolean_flags = [
        '--show-params', '--show_params',
        '--disable-matching', '--disable_matching',
        '--disable-volume-merge', '--disable_volume_merge',
        '--pre-remap-reassign-instances', '--pre_remap_reassign_instances',
        '--output-copc-res1', '--output_copc_res1',
        '--output-copc-res2', '--output_copc_res2',
        '--skip-merged-file', '--skip_merged_file',
        '--transfer-original-dims-to-merged', '--transfer_original_dims_to_merged',
        '--verbose', '-v'
    ]

    processed = []
    i = 0
    while i < len(args_list):
        arg = args_list[i]
        if arg in boolean_flags:
            # Check if next arg is already True/False
            if i + 1 < len(args_list) and args_list[i + 1].lower() in ['true', 'false']:
                processed.extend([arg, args_list[i + 1]])
                i += 2
            else:
                # Add explicit True for boolean flag
                processed.extend([arg, 'True'])
                i += 1
        else:
            processed.append(arg)
            i += 1
    return processed


def _accepted_cli_flags() -> set[str]:
    """Return long CLI flags accepted by Parameters or the run.py preprocessor."""
    accepted = {
        "show-params",
        "show_params",
        "produce-merged-file",
        "produce_merged_file",
        "no-produce-merged-file",
        "no-produce_merged_file",
        "no-transfer-original-dims-to-merged",
        "no-transfer_original_dims_to_merged",
    }
    for field_name, field in Parameters.model_fields.items():
        accepted.add(field_name)
        accepted.add(field_name.replace("_", "-"))
        validation_alias = field.validation_alias
        if validation_alias is None:
            continue
        choices = getattr(validation_alias, "choices", None)
        if choices is None:
            accepted.add(str(validation_alias))
        else:
            accepted.update(str(choice) for choice in choices)
    return accepted


def _unknown_cli_flags(args_list) -> list[str]:
    """Return unknown long CLI flags from args_list."""
    accepted = _accepted_cli_flags()
    unknown = []
    for raw_arg in args_list:
        if not raw_arg.startswith("--") or raw_arg == "--":
            continue
        flag = raw_arg[2:].split("=", 1)[0]
        if flag and flag not in accepted:
            unknown.append(flag)
    return sorted(set(unknown))


def _validate_known_cli_flags(args_list) -> None:
    """Fail fast on typoed or unsupported long CLI flags."""
    unknown = _unknown_cli_flags(args_list)
    if unknown:
        print(
            "Error: Unknown SmartTile CLI argument(s): "
            + ", ".join(f"--{flag}" for flag in unknown)
        )
        sys.exit(1)


def _field_cli_flags(field_name: str) -> list[str]:
    """Return concise user-facing CLI flags for a Parameters field."""
    field = Parameters.model_fields[field_name]
    flags = [field_name.replace("_", "-")]
    if field_name == "transfer_original_dims_to_merged":
        flags = ["produce-merged-file", "no-produce-merged-file", *flags]
    validation_alias = field.validation_alias
    choices = getattr(validation_alias, "choices", None)
    if choices is not None:
        flags.extend(str(choice) for choice in choices)
    elif validation_alias is not None:
        flags.append(str(validation_alias))

    seen = set()
    public_flags = []
    for flag in flags:
        normalized = flag.replace("_", "-")
        if normalized in seen:
            continue
        seen.add(normalized)
        public_flags.append(f"--{normalized}")
    return public_flags


def _print_cli_help() -> None:
    """Print a compact SmartTile CLI help page."""
    option_groups = [
        ("Common", ["task", "input_dir", "output_dir", "workers", "num_spatial_chunks", "chunk_size"]),
        (
            "Tile",
            [
                "tile_length",
                "tile_buffer",
                "threads",
                "resolution_1",
                "resolution_2",
                "output_copc_res1",
                "output_copc_res2",
                "subsampling_method",
                "tiling_threshold",
            ],
        ),
        (
            "Merge",
            [
                "subsampled_10cm_folder",
                "subsampled_target_folder",
                "segmented_remapped_folder",
                "original_raw_input_dir",
                "original_raw_output_dir",
                "original_copc_input_dir",
                "output_merged_laz",
                "output_tiles_folder",
            ],
        ),
        (
            "Remap / Prod-Merged",
            [
                "segmented_folders",
                "merged_laz",
                "original_with_predictions_dir",
                "staged_copc_dir",
                "standardization_json",
                "transfer_original_dims_to_merged",
                "merged_resolutions",
                "merged_output_formats",
                "threedtrees_dims",
            ],
        ),
        (
            "Filtering / Matching",
            [
                "instance_dimension",
                "filter_suffix",
                "filter_output_extension",
                "buffer",
                "overlap_threshold",
                "max_centroid_distance",
                "max_volume_for_merge",
                "min_cluster_size",
                "disable_matching",
                "disable_volume_merge",
                "pre_remap_reassign_instances",
                "verbose",
            ],
        ),
    ]

    print("SmartTile point-cloud processing pipeline")
    print()
    print("Usage:")
    print("  python src/run.py --task tile --input-dir INPUT --output-dir OUTPUT")
    print("  python src/run.py --task merge --subsampled-10cm-folder SEGMENTED_10CM --output-dir OUTPUT")
    print("  python src/run.py --task remap --segmented-folders COLLECTIONS --original-laz-input-dir ORIGINALS --original-laz-output-dir OUTPUT")
    print("  python src/run.py --task create_merged_file --original-with-predictions-dir INPUT --output-dir OUTPUT")
    print("  python src/run.py --task filter --input-dir INPUT --output-dir OUTPUT")
    print()
    print("Tasks: tile, merge, filter, remap, create_merged_file")
    print("Use --show-params to print resolved defaults and environment overrides.")
    print()
    print("Options:")
    for title, field_names in option_groups:
        print(f"  {title}:")
        for field_name in field_names:
            field = Parameters.model_fields.get(field_name)
            if field is None:
                continue
            flags = ", ".join(_field_cli_flags(field_name))
            description = field.description or ""
            print(f"    {flags}")
            if description:
                print(f"      {description}")
        print()


def main():
    if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
        _print_cli_help()
        sys.exit(0)

    # Handle --show-params flag first using argparse (before Pydantic parsing)
    # This avoids Pydantic's boolean flag parsing issues
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--show-params', '--show_params', action='store_true')
    pre_args, remaining_args = pre_parser.parse_known_args()

    # If --show-params was found, add it back to remaining_args for Pydantic
    if pre_args.show_params:
        remaining_args = ['--show-params'] + remaining_args

    # Manually map aliases that Pydantic might not generate flags for
    # --subsampled-segmented-folder -> --subsampled-10cm-folder
    # --produce-merged-file -> --transfer-original-dims-to-merged
    # --no-produce-merged-file -> --transfer-original-dims-to-merged False
    # --no-transfer-original-dims-to-merged -> --transfer-original-dims-to-merged False
    mapped_args = []
    for arg in remaining_args:
        if arg == '--subsampled-segmented-folder':
            mapped_args.append('--subsampled-10cm-folder')
        elif arg in ('--produce-merged-file', '--produce_merged_file'):
            mapped_args.append('--transfer-original-dims-to-merged')
        elif arg in ('--no-produce-merged-file', '--no-produce_merged_file'):
            mapped_args.extend(['--transfer-original-dims-to-merged', 'False'])
        elif arg in ('--no-transfer-original-dims-to-merged', '--no-transfer_original_dims_to_merged'):
            mapped_args.extend(['--transfer-original-dims-to-merged', 'False'])
        else:
            mapped_args.append(arg)
    remaining_args = mapped_args

    _validate_known_cli_flags(remaining_args)

    # Preprocess boolean flags for Pydantic
    processed_args = [sys.argv[0]] + preprocess_boolean_flags(remaining_args)

    # Temporarily replace sys.argv for Pydantic parsing
    original_argv = sys.argv
    sys.argv = processed_args

    # Parse parameters using Pydantic (handles CLI automatically)
    try:
        params = Parameters()
    except Exception as e:
        print(f"Error parsing parameters: {e}")
        sys.exit(1)
    finally:
        # Restore original argv
        sys.argv = original_argv

    # Show parameters if requested (flag handled by pre-parser; not in Parameters)
    if pre_args.show_params:
        print_params(params)
        sys.exit(0)

    # Task is required if not showing params
    if not params.task:
        print("Error: --task is required (unless using --show-params)")
        print("Usage: python run.py --task tile --input-dir /path/to/input --output-dir /path/to/output")
        print("       python run.py --task merge --subsampled-10cm-folder /path/to/10cm")
        print("       python run.py --task filter --input-dir /path/to/segmented_remapped --output-dir /path/to/filtered")
        print("       python run.py --task remap --merged-laz /path/to/merged.laz --original-input-dir /path/to/originals")
        print("       python run.py --task create_merged_file --original-with-predictions-dir /path/to/original_with_predictions --output-dir /path/to/output")
        print("       python run.py --show-params")
        sys.exit(1)

    # Route to appropriate task function
    if params.task == "tile":
        run_tile_task(params)
    elif params.task == "merge":
        run_merge_task(params)
    elif params.task == "filter":
        run_filter_task(params)
    elif params.task == "remap":
        run_remap_task(params)
    elif params.task == "create_merged_file":
        run_create_merged_file_task(params)
    else:
        print(f"Error: Unknown task: {params.task}")
        print("Valid tasks: tile, merge, filter, remap, create_merged_file")
        sys.exit(1)


if __name__ == "__main__":
    main()
