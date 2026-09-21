#!/usr/bin/env python3
"""
Main orchestrator script for the 3DTrees smart tiling pipeline.

Routes to appropriate task modules based on --task parameter:
- tile: XYZ reduction, COPC conversion, tiling, and subsampling (1cm and 10cm)
- merge: Remap predictions and merge tiles with instance matching
- filter: Deduplicate overlapping dense tile points after instance reconciliation
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
    from merge_prediction_collections import comma_paths
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
        num_spatial_chunks=params.num_spatial_chunks,
        chunk_workers=params.num_spatial_chunks,
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


def _validate_copc_original_lane(copc_input_dir: Path) -> None:
    """Fail early when an explicit COPC-original lane is missing COPC files."""
    if not copc_input_dir.exists():
        print(f"Error: COPC original input directory not found: {copc_input_dir}")
        sys.exit(1)

    from point_cloud_metadata import copc_files

    if not copc_files(copc_input_dir):
        print(f"Error: No COPC LAZ files found in original COPC input dir: {copc_input_dir}")
        sys.exit(1)


def _validate_copc_laz_source_pairs(copc_input_dir: Path, raw_input_dir: Path) -> None:
    """Validate that raw uploaded originals have matching COPC twins when both lanes are used."""
    import laspy
    from point_cloud_metadata import copc_files, point_cloud_source_key, raw_point_cloud_files

    raw_by_key = {point_cloud_source_key(path): path for path in raw_point_cloud_files(raw_input_dir)}
    copc_by_key = {point_cloud_source_key(path): path for path in copc_files(copc_input_dir)}
    if not raw_by_key:
        print(f"Error: No raw LAZ/LAS files found in original input dir: {raw_input_dir}")
        sys.exit(1)
    missing = sorted(set(raw_by_key) - set(copc_by_key))
    if missing:
        print(
            "Error: COPC original input dir is missing COPC twins for raw originals: "
            + ", ".join(missing)
        )
        sys.exit(1)

    for key, raw_path in raw_by_key.items():
        copc_path = copc_by_key[key]
        with laspy.open(str(raw_path), laz_backend=laspy.LazBackend.LazrsParallel) as raw_reader:
            raw_header = raw_reader.header
            raw_count = int(raw_header.point_count)
            raw_bounds = (
                float(raw_header.x_min),
                float(raw_header.x_max),
                float(raw_header.y_min),
                float(raw_header.y_max),
                float(raw_header.z_min),
                float(raw_header.z_max),
            )
        with laspy.open(str(copc_path), laz_backend=laspy.LazBackend.LazrsParallel) as copc_reader:
            copc_header = copc_reader.header
            copc_count = int(copc_header.point_count)
            copc_bounds = (
                float(copc_header.x_min),
                float(copc_header.x_max),
                float(copc_header.y_min),
                float(copc_header.y_max),
                float(copc_header.z_min),
                float(copc_header.z_max),
            )
        if raw_count != copc_count:
            print(
                f"Error: COPC/raw source pair point-count mismatch for {key}: "
                f"{copc_path.name} has {copc_count:,}, {raw_path.name} has {raw_count:,}"
            )
            sys.exit(1)
        if any(abs(a - b) > 0.02 for a, b in zip(raw_bounds, copc_bounds)):
            print(
                f"Error: COPC/raw source pair bounds mismatch for {key}: "
                f"{copc_path.name} {copc_bounds} vs {raw_path.name} {raw_bounds}"
            )
            sys.exit(1)


def _require_tile_bounds_json(tile_bounds_json: Path | None) -> Path:
    """Return a valid tile bounds JSON path or exit with a production-safe error."""
    if tile_bounds_json is None:
        print("Error: merge task requires --tile-bounds-json /path/to/tile_bounds_tindex.json")
        sys.exit(1)
    tile_bounds_json = Path(tile_bounds_json)
    if not tile_bounds_json.exists():
        print(f"Error: tile_bounds_tindex.json not found: {tile_bounds_json}")
        sys.exit(1)
    return tile_bounds_json


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
    tile_source_workers = params.tile_source_workers or workers
    tile_writer_workers = params.tile_writer_workers or workers
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
    """Transfer to dense geometry, reconcile per model, then deduplicate points."""
    from strict_prediction_pipeline import merge_collections

    modes = [bool(params.subsampled_10cm_folder), bool(params.segmented_folders),
             bool(params.segmented_remapped_folder)]
    if sum(modes) != 1:
        print("Error: provide exactly one segmented input mode")
        sys.exit(1)
    tile_bounds = _require_tile_bounds_json(params.tile_bounds_json)
    collections = (comma_paths(params.segmented_folders) if params.segmented_folders else
                   [Path(params.subsampled_10cm_folder or params.segmented_remapped_folder)])
    work = Path(params.output_folder or
                (Path(params.output_tiles_folder).parent if params.output_tiles_folder else
                 Path(params.output_merged_laz).parent if params.output_merged_laz else
                 collections[0].parent))
    output_tiles = Path(params.output_tiles_folder or work / "output_tiles")
    target = params.subsampled_target_folder or params.original_tiles_dir
    if target is None and params.subsampled_10cm_folder:
        target = collections[0].parent / "subsampled_res1"
    ready = bool(params.segmented_remapped_folder) or (bool(params.segmented_folders) and target is None)
    originals = params.original_raw_input_dir or params.original_input_dir
    original_output = Path(params.original_raw_output_dir or output_tiles.parent / "original_with_predictions")
    try:
        if params.original_copc_input_dir:
            if not originals:
                raise ValueError("--original-laz-input-dir is required with --original-copc-input-dir")
            _validate_copc_original_lane(Path(params.original_copc_input_dir))
            _validate_copc_laz_source_pairs(Path(params.original_copc_input_dir), Path(originals))
        if originals:
            _validate_raw_original_lane(Path(originals), original_output)
        merged = None if params.skip_merged_file else params.output_merged_laz
        if merged is None and not params.skip_merged_file and len(collections) == 1:
            merged = work / "merged.laz"
        report = merge_collections(
            collections=collections, target_dir=Path(target) if target else None,
            output_tiles=output_tiles, tile_bounds_json=tile_bounds,
            originals=Path(originals) if originals else None, original_output=original_output,
            merged_output=Path(merged) if merged else None,
            transfer_radius=params.prediction_transfer_tolerance,
            overlap_threshold=params.overlap_threshold, matching=not params.disable_matching,
            ready=ready, instance_dimension=params.instance_dimension,
            target_dims=set(_parse_csv(params.remap_dims)) if params.remap_dims else None,
        )
        if originals and params.transfer_original_dims_to_merged:
            _create_prod_merged_outputs(original_output, output_tiles.parent, params)
        print(f"Merge complete: {report['state']}; report: {output_tiles.parent / 'remap_first_report.json'}")
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)


def run_remap_task(params: Parameters):
    """Validate baseline and final coverage independently for every model."""
    from strict_prediction_pipeline import strict_remap

    originals = params.original_raw_input_dir or params.original_input_dir
    if not originals or bool(params.segmented_folders) == bool(params.merged_laz):
        print("Error: remap requires raw originals and exactly one of --segmented-folders or --merged-laz")
        sys.exit(1)
    if params.pre_remap_reassign_instances:
        print("Error: pre-remap label reassignment is incompatible with the strict reconciled-label contract")
        sys.exit(1)
    original_dir = Path(originals)
    output = _raw_original_output_dir(params, original_dir)
    try:
        _validate_raw_original_lane(original_dir, output)
        if params.original_copc_input_dir:
            _validate_copc_original_lane(Path(params.original_copc_input_dir))
            _validate_copc_laz_source_pairs(Path(params.original_copc_input_dir), original_dir)
        collections = comma_paths(params.segmented_folders) if params.segmented_folders else [Path(params.merged_laz)]
        strict_remap(
            collections=collections, originals=original_dir, output=output,
            baseline_collections=comma_paths(params.baseline_1cm_folders) or None,
            instance_dimension=params.instance_dimension,
            target_dims=set(_parse_csv(params.remap_dims)) if params.remap_dims else None,
        )
        if params.transfer_original_dims_to_merged:
            _create_prod_merged_outputs(output, output.parent, params)
        print(f"Strict original remap complete: {output}")
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)


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
            num_spatial_chunks=params.num_spatial_chunks,
            chunk_workers=params.num_spatial_chunks,
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
    """Deduplicate already-dense tiles under the same strict merge contract."""
    from strict_prediction_pipeline import merge_collections

    if not params.input_dir or not params.output_dir:
        print("Error: --input-dir and --output-dir are required for filter")
        sys.exit(1)
    if Path(params.input_dir).resolve() == Path(params.output_dir).resolve():
        print("Error: filter input and output must differ")
        sys.exit(1)
    tile_bounds = _require_tile_bounds_json(params.tile_bounds_json)
    try:
        merge_collections(collections=[Path(params.input_dir)], target_dir=None,
                          output_tiles=Path(params.output_dir), tile_bounds_json=tile_bounds,
                          ready=True, matching=not params.disable_matching,
                          instance_dimension=params.instance_dimension,
                          overlap_threshold=params.overlap_threshold)
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)


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
        ("Common", ["task", "input_dir", "output_dir", "workers", "num_spatial_chunks", "chunk_size", "memory_gb"]),
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
