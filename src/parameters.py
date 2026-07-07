"""
Centralized parameter configuration for the 3DTrees smart tiling pipeline.

Uses Pydantic BaseSettings for CLI argument parsing, environment variable support,
and parameter validation.

Usage:
    python run.py --task tile --input-dir /path/to/input --output-dir /path/to/output
    python run.py --task merge --subsampled-10cm-folder /path/to/10cm --original-input-dir /path/to/input
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, AliasChoices, field_validator, model_validator
from pathlib import Path
from collections.abc import Iterable
from typing import Optional

from worker_budget import DEFAULT_FILE_WORKERS, available_cpu_count


class Parameters(BaseSettings):
    """
    Pipeline parameters with CLI and environment variable support.

    All parameters can be passed via:
    - CLI arguments: --param-name value
    - Environment variables: PARAM_NAME=value
    """

    # ==========================================================================
    # Common parameters
    # ==========================================================================

    task: str = Field(
        "tile",
        description=(
            "Task to perform: 'tile' (tiling + subsampling), 'merge' (remap + merge), "
            "'filter' (remove duplicate buffer-zone instances), 'remap' "
            "(prediction source -> original files), or 'create_merged_file' "
            "(prod-merged from original_with_predictions)"
        ),
    )

    input_dir: Optional[Path] = Field(
        default=None,
        description="Input directory with LAZ/LAS files (required for 'tile' task)",
        validation_alias=AliasChoices("input-dir", "input_dir"),
    )

    output_dir: Optional[Path] = Field(
        default=None,
        description="Output directory (required for 'tile' task)",
        validation_alias=AliasChoices("output-dir", "output_dir"),
    )

    workers: int = Field(
        DEFAULT_FILE_WORKERS,
        description="File-level parallelism. Defaults to two files processed concurrently.",
        validation_alias=AliasChoices("workers", "number-of-threads", "number_of_threads"),
    )

    # ==========================================================================
    # Tile task parameters
    # ==========================================================================

    tile_length: Optional[int] = Field(
        100,
        description="Tile size in meters (only for 'tile' task)",
        validation_alias=AliasChoices("tile-length", "tile_length"),
    )

    tile_buffer: Optional[int] = Field(
        20,
        description="Buffer overlap in meters (only for 'tile' task)",
        validation_alias=AliasChoices("tile-buffer", "tile_buffer"),
    )

    threads: Optional[int] = Field(
        10,
        description="Threads per COPC writer (only for 'tile' task)",
    )

    tile_source_workers: Optional[int] = Field(
        default=None,
        description=(
            "Maximum source files processed in parallel during tile Phase 1. "
            "Defaults to --workers when unset."
        ),
        validation_alias=AliasChoices("tile-source-workers", "tile_source_workers"),
    )

    tile_writer_workers: Optional[int] = Field(
        default=None,
        description=(
            "Maximum tiles finalized to COPC in parallel during tile Phase 2. "
            "Defaults to --workers when unset."
        ),
        validation_alias=AliasChoices(
            "tile-writer-workers",
            "tile_writer_workers",
            "tile-finalize-workers",
            "tile_finalize_workers",
        ),
    )

    resolution_1: Optional[float] = Field(
        0.01,
        description="First subsampling resolution in meters (1cm) (only for 'tile' task)",
        validation_alias=AliasChoices("resolution-1", "resolution_1"),
    )

    resolution_2: Optional[float] = Field(
        0.1,
        description="Second subsampling resolution in meters (10cm) (only for 'tile' task)",
        validation_alias=AliasChoices("resolution-2", "resolution_2"),
    )

    output_copc_res1: bool = Field(
        True,
        description="Write first-resolution subsampled outputs as COPC LAZ (default: True for 1cm products)",
        validation_alias=AliasChoices("output-copc-res1", "output_copc_res1"),
    )

    output_copc_res2: bool = Field(
        False,
        description="Write second-resolution subsampled outputs as COPC LAZ (default: False, regular LAZ for 10cm products)",
        validation_alias=AliasChoices("output-copc-res2", "output_copc_res2"),
    )


    instance_dimension: str = Field(
        "PredInstance",
        description="Name of the instance ID dimension in input files (default: PredInstance, fallback: treeID)",
        validation_alias=AliasChoices("instance-dimension", "instance_dimension"),
    )

    filter_suffix: str = Field(
        "_filtered",
        description="Suffix added to output filenames for the filter task",
        validation_alias=AliasChoices("filter-suffix", "filter_suffix", "suffix"),
    )

    filter_output_extension: Optional[str] = Field(
        default=None,
        description="Optional output extension override for the filter task, e.g. .laz",
        validation_alias=AliasChoices("filter-output-extension", "filter_output_extension", "output-extension", "output_extension"),
    )

    filter_anchor: str = Field(
        "centroid",
        description="Representative point used by filter task border ownership: centroid, highest_point, or lowest_point",
        validation_alias=AliasChoices("filter-anchor", "filter_anchor"),
    )

    num_spatial_chunks: Optional[int] = Field(
        default_factory=available_cpu_count,
        description="Per-file spatial/chunk parallelism. Defaults to available CPU count.",
        validation_alias=AliasChoices("num-spatial-chunks", "num_spatial_chunks"),
    )

    subsampling_method: str = Field(
        default="center-of-mass",
        description="Subsampling method: center-of-mass (default) or nearest-to-centroid",
        validation_alias=AliasChoices("subsampling-method", "subsampling_method"),
    )

    tiling_threshold: Optional[float] = Field(
        default=None,
        description="File size threshold in MB. If input folder has single file below this size, skip tiling (only for 'tile' task)",
        validation_alias=AliasChoices("tiling-threshold", "tiling_threshold"),
    )

    chunk_size: Optional[int] = Field(
        default=20_000_000,
        description="Points per chunk when reading LAZ/LAS in tiling Phase 1 or multi-collection remap (smaller = less peak RAM, more overhead)",
        validation_alias=AliasChoices("chunk-size", "chunk_size"),
    )

    # ==========================================================================
    # Merge task parameters
    # ==========================================================================

    subsampled_10cm_folder: Optional[Path] = Field(
        default=None,
        description="Path to subsampled 10cm folder with segmented results (for 'merge' task)",
        validation_alias=AliasChoices("subsampled-10cm-folder", "subsampled_10cm_folder", "subsampled-segmented-folder"),
    )

    subsampled_target_folder: Optional[Path] = Field(
        default=None,
        description="Path to target resolution subsampled folder (auto-derived if not specified)",
        validation_alias=AliasChoices("subsampled-target-folder", "subsampled_target_folder"),
    )

    segmented_remapped_folder: Optional[Path] = Field(
        default=None,
        description="Path to segmented remapped folder (for 'merge' task, skip remap step)",
        validation_alias=AliasChoices("segmented-remapped-folder", "segmented_remapped_folder"),
    )

    original_tiles_dir: Optional[Path] = Field(
        default=None,
        description="Directory with original tile files for retiling (for 'merge' task)",
        validation_alias=AliasChoices("original-tiles-dir", "original_tiles_dir"),
    )

    tile_bounds_json: Optional[Path] = Field(
        default=None,
        description="Path to tile_bounds_tindex.json for neighbor graph and remap matching (merge task). If set, used instead of auto-derived paths.",
        validation_alias=AliasChoices("tile-bounds-json", "tile_bounds_json"),
    )

    original_input_dir: Optional[Path] = Field(
        default=None,
        description=(
            "Legacy directory with original input LAZ/LAS files for final remap. "
            "Prefer --original-laz-input-dir for the explicit production workflow. "
            "COPC-only original enrichment is no longer supported."
        ),
        validation_alias=AliasChoices("original-input-dir", "original_input_dir"),
    )

    original_raw_input_dir: Optional[Path] = Field(
        default=None,
        description=(
            "Directory with uploaded LAZ/LAS originals to enrich. This is the "
            "production Original-with-predictions writer/metadata lane and ignores "
            "COPC twins so raw source metadata/VLRs remain the metadata source."
        ),
        validation_alias=AliasChoices(
            "original-laz-input-dir",
            "original_laz_input_dir",
            "original-raw-input-dir",
            "original_raw_input_dir",
            "original-download-input-dir",
            "original_download_input_dir",
        ),
    )

    original_copc_input_dir: Optional[Path] = Field(
        default=None,
        description=(
            "Optional directory with COPC twins for validating/routing merge or remap "
            "jobs that also enrich uploaded LAZ/LAS originals."
        ),
        validation_alias=AliasChoices(
            "original-copc-input-dir",
            "original_copc_input_dir",
        ),
    )

    original_raw_output_dir: Optional[Path] = Field(
        default=None,
        description=(
            "Output directory for enriched uploaded LAZ/LAS originals. Defaults to "
            "--output-dir when set, otherwise original_with_predictions next to the "
            "raw input directory."
        ),
        validation_alias=AliasChoices(
            "original-laz-output-dir",
            "original_laz_output_dir",
            "original-raw-output-dir",
            "original_raw_output_dir",
            "original-download-output-dir",
            "original_download_output_dir",
        ),
    )

    output_merged_laz: Optional[Path] = Field(
        default=None,
        description="Output path for merged LAZ file (auto-derived if not specified)",
        validation_alias=AliasChoices("output-merged-laz", "output_merged_laz"),
    )

    output_tiles_folder: Optional[Path] = Field(
        default=None,
        description="Output folder for per-tile results (auto-derived if not specified)",
        validation_alias=AliasChoices("output-tiles-folder", "output_tiles_folder"),
    )

    output_folder: Optional[Path] = Field(
        default=None,
        description="Output folder for remapped files (auto-derived if not specified)",
        validation_alias=AliasChoices("output-folder", "output_folder"),
    )

    original_with_predictions_dir: Optional[Path] = Field(
        default=None,
        description="Directory with Original-with-predictions files for create_merged_file task",
        validation_alias=AliasChoices("original-with-predictions-dir", "original_with_predictions_dir"),
    )

    staged_copc_dir: Optional[Path] = Field(
        default=None,
        description=(
            "Optional directory with already converted Original-with-predictions COPC files "
            "for create_merged_file/prod-merged outputs. Matching COPCs are reused after "
            "a readable-header check."
        ),
        validation_alias=AliasChoices("staged-copc-dir", "staged_copc_dir"),
    )

    standardization_json: Optional[Path] = Field(
        default=None,
        description=(
            "Optional tool_standard collection_summary.json. When provided, SmartTile "
            "validates that Original-with-predictions COPCs and LAS/COPC prod-merged "
            "outputs still expose the expected standardized source dimensions."
        ),
        validation_alias=AliasChoices("standardization-json", "standardization_json"),
    )

    merged_resolutions: str = Field(
        "res1,res2",
        description=(
            "Comma-separated prod-merged output resolutions for create_merged_file. "
            "Use res1/res2, numeric meters, or centimeter labels such as 1cm,10cm."
        ),
        validation_alias=AliasChoices("merged-resolutions", "merged_resolutions"),
    )

    merged_output_formats: str = Field(
        "copc.laz",
        description="Comma-separated prod-merged output formats: laz, copc.laz, or ply.",
        validation_alias=AliasChoices("merged-output-formats", "merged_output_formats"),
    )

    # ==========================================================================
    # Remap-to-originals task parameters
    # ==========================================================================

    merged_laz: Optional[Path] = Field(
        default=None,
        description="Path to merged LAZ/COPC LAZ file (for 'remap' task). Selected dimensions from this file are added to original files.",
        validation_alias=AliasChoices("merged-laz", "merged_laz"),
    )

    segmented_folders: Optional[str] = Field(
        default=None,
        description=(
            "Comma-separated list of finalized prediction collection folders/files "
            "for multi-collection 'remap'. Extra dimensions are copied as-is and "
            "duplicate prediction dimension names fail."
        ),
        validation_alias=AliasChoices("segmented-folders", "segmented_folders"),
    )

    remap_dims: Optional[str] = Field(
        default=None,
        description=(
            "Optional comma-separated allowlist of extra dimension names to transfer "
            "from prediction collections during multi-collection remap."
        ),
        validation_alias=AliasChoices("remap-dims", "remap_dims"),
    )

    remap_tolerance: float = Field(
        0.125,
        gt=0,
        description=(
            "Maximum nearest-neighbor distance in meters when remapping "
            "prediction collections to original points."
        ),
        validation_alias=AliasChoices("remap-tolerance", "remap_tolerance"),
    )

    output_merged_with_originals: Optional[Path] = Field(
        default=None,
        description="Legacy path for the old merged-with-originals remap output. Prod-merged outputs now use --merged-resolutions.",
        validation_alias=AliasChoices("output-merged-with-originals", "output_merged_with_originals"),
    )

    produce_merged_file: bool = Field(
        False,
        description="For filter/remap tails, also create a merged output product when supported.",
        validation_alias=AliasChoices("produce-merged-file", "produce_merged_file"),
    )

    transfer_original_dims_to_merged: bool = Field(
        False,
        description="Create prod-merged files from Original-with-predictions after merge/remap. Uses the create_merged_file implementation.",
        validation_alias=AliasChoices("transfer-original-dims-to-merged", "transfer_original_dims_to_merged"),
    )

    threedtrees_dims: str = Field(
        "PredInstance,PredSemantic",
        description="Comma-separated list of dimension names produced by 3DTrees to transfer to original files. These are renamed to {name}_{suffix} in the output (e.g. PredInstance_SAT).",
        validation_alias=AliasChoices("threedtrees-dims", "threedtrees_dims"),
    )

    threedtrees_suffix: str = Field(
        "SAT",
        description="Suffix for 3DTrees dimension names (e.g. SAT -> PredInstance_SAT).",
        validation_alias=AliasChoices("threedtrees-suffix", "threedtrees_suffix"),
    )

    pre_remap_reassign_instances: bool = Field(
        False,
        description="Before remapping to originals, reassign small instances in the segmented/merged point cloud.",
        validation_alias=AliasChoices("pre-remap-reassign-instances", "pre_remap_reassign_instances"),
    )

    pre_remap_reassign_instance_dimension: Optional[str] = Field(
        default=None,
        description="Instance dimension to update during pre-remap reassignment. Defaults to the first transferred dimension containing 'Instance'.",
        validation_alias=AliasChoices("pre-remap-reassign-instance-dimension", "pre_remap_reassign_instance_dimension"),
    )

    pre_remap_reassign_min_cluster_size: int = Field(
        250,
        description="Pre-remap reassignment: instances below this point count are reassigned to the nearest larger instance.",
        validation_alias=AliasChoices("pre-remap-reassign-min-cluster-size", "pre_remap_reassign_min_cluster_size"),
    )

    pre_remap_reassign_hull_point_threshold: int = Field(
        5000,
        description="Pre-remap reassignment: compute convex hulls for instances below this point count.",
        validation_alias=AliasChoices("pre-remap-reassign-hull-point-threshold", "pre_remap_reassign_hull_point_threshold"),
    )

    pre_remap_reassign_max_volume: float = Field(
        5.0,
        description="Pre-remap reassignment: instances below the hull point threshold and this hull volume in m3 are reassigned.",
        validation_alias=AliasChoices("pre-remap-reassign-max-volume", "pre_remap_reassign_max_volume"),
    )

    pre_remap_reassigned_laz: Optional[Path] = Field(
        default=None,
        description="Optional path to save the segmented/merged point cloud after pre-remap reassignment.",
        validation_alias=AliasChoices("pre-remap-reassigned-laz", "pre_remap_reassigned_laz"),
    )

    # ==========================================================================
    # Remap task parameters
    # ==========================================================================

    source_folder: Optional[Path] = Field(
        default=None,
        description="Path to source LAZ files (e.g., segmented files) for 'remap' task",
        validation_alias=AliasChoices("source-folder", "source_folder"),
    )

    target_folder: Optional[Path] = Field(
        default=None,
        description="Path to target LAZ files (e.g., subsampled files) for 'remap' task",
        validation_alias=AliasChoices("target-folder", "target_folder"),
    )

    # Merge algorithm parameters
    buffer: Optional[float] = Field(
        10.0,
        description="Legacy standalone filter-task buffer. Merge derives its buffer from tile_bounds_tindex.json.",
    )

    overlap_threshold: Optional[float] = Field(
        0.3,
        description="Overlap ratio threshold for instance matching (0.3 = 30%)",
        validation_alias=AliasChoices("overlap-threshold", "overlap_threshold"),
    )

    max_centroid_distance: Optional[float] = Field(
        3.0,
        description="Max centroid distance to merge instances in meters",
        validation_alias=AliasChoices("max-centroid-distance", "max_centroid_distance"),
    )

    max_volume_for_merge: Optional[float] = Field(
        4.0,
        description="Max convex hull volume for small instance merging in m³",
        validation_alias=AliasChoices("max-volume-for-merge", "max_volume_for_merge"),
    )

    border_zone_width: Optional[float] = Field(
        10.0,
        description="Width of border zone beyond buffer for instance matching (meters)",
        validation_alias=AliasChoices("border-zone-width", "border_zone_width"),
    )

    min_cluster_size: Optional[int] = Field(
        300,
        description="Minimum cluster size in points for reassignment",
        validation_alias=AliasChoices("min-cluster-size", "min_cluster_size"),
    )

    enable_volume_merge: bool = Field(
        True,
        description="Enable small-instance volume reassignment in filter task.",
        validation_alias=AliasChoices("enable-volume-merge", "enable_volume_merge"),
    )

    remap_merge: bool = Field(
        False,
        description="After filter task, remap filtered dimensions to originals and/or subsampled targets.",
        validation_alias=AliasChoices("remap-merge", "remap_merge"),
    )

    disable_matching: bool = Field(
        False,
        description="Disable cross-tile instance matching",
        validation_alias=AliasChoices("disable-matching", "disable_matching"),
    )

    disable_volume_merge: bool = Field(
        False,
        description="Disable small volume instance merging",
        validation_alias=AliasChoices("disable-volume-merge", "disable_volume_merge"),
    )

    skip_merged_file: bool = Field(
        False,
        description="Skip creating merged LAZ file (only create retiled outputs)",
        validation_alias=AliasChoices("skip-merged-file", "skip_merged_file"),
    )

    verbose: bool = Field(
        False,
        description="Print detailed merge decisions",
    )

    # ==========================================================================
    # Validators
    # ==========================================================================

    @field_validator(
        "input_dir",
        "output_dir",
    )
    @classmethod
    def validate_tile_required_params(cls, v, info):
        """Validate that tile task required parameters are provided."""
        # Note: Actual validation happens in run.py after instantiation
        # since we need to check the task value
        return v

    @field_validator(
        "tile_length",
        "resolution_1",
        "resolution_2",
        "threads",
        "chunk_size",
    )
    @classmethod
    def validate_tile_params(cls, v, info):
        """Validate tile parameters are positive when provided."""
        if v is not None and v <= 0:
            raise ValueError(f"{info.field_name} must be positive")
        return v

    @field_validator("tile_buffer")
    @classmethod
    def validate_tile_buffer(cls, v, info):
        """Validate tile buffer is non-negative so zero-overlap tiling is allowed."""
        if v is not None and v < 0:
            raise ValueError(f"{info.field_name} must be non-negative")
        return v

    @field_validator(
        "buffer",
        "overlap_threshold",
        "max_centroid_distance",
        "max_volume_for_merge",
        "pre_remap_reassign_max_volume",
    )
    @classmethod
    def validate_merge_params(cls, v, info):
        """Validate merge parameters are positive when provided."""
        if v is not None and v < 0:
            raise ValueError(f"{info.field_name} must be non-negative")
        return v

    @field_validator("overlap_threshold")
    @classmethod
    def validate_overlap_threshold(cls, v):
        """Validate overlap threshold is between 0 and 1."""
        if v is not None and (v < 0 or v > 1):
            raise ValueError("overlap_threshold must be between 0 and 1")
        return v

    @field_validator(
        "workers",
        "num_spatial_chunks",
        "tile_source_workers",
        "tile_writer_workers",
        "min_cluster_size",
        "pre_remap_reassign_min_cluster_size",
        "pre_remap_reassign_hull_point_threshold",
    )
    @classmethod
    def validate_positive_int(cls, v, info):
        """Validate integer parameters are positive."""
        if v is not None and v <= 0:
            raise ValueError(f"{info.field_name} must be positive")
        return v

    @field_validator("subsampling_method")
    @classmethod
    def validate_subsampling_method(cls, v):
        """Validate and normalize the subsampling method."""
        normalized = (v or "center-of-mass").strip().lower()
        aliases = {
            "com": "center-of-mass",
            "center_of_mass": "center-of-mass",
            "centroid": "nearest-to-centroid",
            "voxelcentroidnearestneighbor": "nearest-to-centroid",
            "voxel-centroid-nearest-neighbor": "nearest-to-centroid",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in {"center-of-mass", "nearest-to-centroid"}:
            raise ValueError("subsampling_method must be 'center-of-mass' or 'nearest-to-centroid'")
        return normalized

    @field_validator("filter_anchor")
    @classmethod
    def validate_filter_anchor(cls, v):
        """Validate and normalize filter anchor selection."""
        normalized = (v or "centroid").strip().lower()
        if normalized not in {"centroid", "highest_point", "lowest_point"}:
            raise ValueError("filter_anchor must be 'centroid', 'highest_point', or 'lowest_point'")
        return normalized

    @model_validator(mode="after")
    def sync_prod_merged_flags(self):
        """Keep legacy Galaxy prod-merged selection and implementation flag aligned."""
        if self.produce_merged_file or self.transfer_original_dims_to_merged:
            self.produce_merged_file = True
            self.transfer_original_dims_to_merged = True
        return self

    @field_validator("merged_output_formats", mode="before")
    @classmethod
    def validate_merged_output_formats(cls, v):
        """Validate and normalize prod-merged output formats."""
        aliases = {
            "las": "laz",
            "laz": "laz",
            ".laz": "laz",
            "copc": "copc.laz",
            "copc_laz": "copc.laz",
            "copc-laz": "copc.laz",
            "copc.laz": "copc.laz",
            ".copc.laz": "copc.laz",
            "ply": "ply",
            ".ply": "ply",
        }
        parsed = []
        seen = set()

        def tokens(value):
            if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
                for item in value:
                    yield from tokens(item)
                return
            text = str(value or "")
            if text.startswith("[") and text.endswith("]"):
                text = text[1:-1]
            for token in text.split(","):
                yield token.strip().strip("'\"")

        for raw_token in tokens(v or "copc.laz"):
            token = raw_token.strip().lower()
            if not token:
                continue
            output_format = aliases.get(token)
            if output_format is None:
                raise ValueError("merged_output_formats must contain only 'laz', 'copc.laz', or 'ply'")
            if output_format in seen:
                continue
            seen.add(output_format)
            parsed.append(output_format)
        if not parsed:
            raise ValueError("merged_output_formats must contain at least one format")
        return ",".join(parsed)

    # ==========================================================================
    # Model configuration
    # ==========================================================================

    model_config = SettingsConfigDict(
        case_sensitive=False,
        cli_parse_args=True,
        cli_ignore_unknown_args=True,
        env_prefix="",  # No prefix for env vars
        extra="ignore",  # Ignore unknown fields
    )


def print_params(params: Parameters):
    """Print current parameter configuration."""
    print("=" * 60)
    print("Current Parameters")
    print("=" * 60)

    print("\nCommon:")
    print(f"  task: {params.task}")
    print(f"  input_dir: {params.input_dir}")
    print(f"  output_dir: {params.output_dir}")
    print(f"  workers: {params.workers}")
    print(f"  num_spatial_chunks: {params.num_spatial_chunks}")
    print(f"  instance_dimension: {params.instance_dimension}")
    print(f"  filter_suffix: {params.filter_suffix}")
    print(f"  filter_output_extension: {params.filter_output_extension}")
    print(f"  filter_anchor: {params.filter_anchor}")

    print("\nTile Task:")
    print(f"  tile_length: {params.tile_length}")
    print(f"  tile_buffer: {params.tile_buffer}")
    print(f"  threads: {params.threads}")
    print(f"  tile_source_workers: {params.tile_source_workers}")
    print(f"  tile_writer_workers: {params.tile_writer_workers}")
    print(f"  chunk_size: {params.chunk_size}")
    print(f"  resolution_1: {params.resolution_1}")
    print(f"  resolution_2: {params.resolution_2}")
    print(f"  output_copc_res1: {params.output_copc_res1}")
    print(f"  output_copc_res2: {params.output_copc_res2}")
    print(f"  subsampling_method: {params.subsampling_method}")

    print("\nMerge Task:")
    print(f"  subsampled_10cm_folder: {params.subsampled_10cm_folder}")
    print(f"  original_input_dir: {params.original_input_dir}")
    print(f"  original_raw_input_dir: {params.original_raw_input_dir}")
    print(f"  original_raw_output_dir: {params.original_raw_output_dir}")
    print(f"  overlap_threshold: {params.overlap_threshold}")
    print(f"  max_centroid_distance: {params.max_centroid_distance}")
    print(f"  max_volume_for_merge: {params.max_volume_for_merge}")
    print(f"  min_cluster_size: {params.min_cluster_size}")
    print(f"  disable_matching: {params.disable_matching}")
    print(f"  enable_volume_merge: {params.enable_volume_merge}")
    print(f"  remap_merge: {params.remap_merge}")
    print(f"  verbose: {params.verbose}")

    print("\nCreate Merged File Task:")
    print(f"  original_with_predictions_dir: {params.original_with_predictions_dir}")
    print(f"  staged_copc_dir: {params.staged_copc_dir}")
    print(f"  standardization_json: {params.standardization_json}")
    print(f"  merged_resolutions: {params.merged_resolutions}")
    print(f"  merged_output_formats: {params.merged_output_formats}")
    print(f"  produce_merged_file: {params.produce_merged_file}")

    print("\nRemap Task:")
    print(f"  merged_laz: {params.merged_laz}")
    print(f"  segmented_folders: {params.segmented_folders}")
    print(f"  remap_dims: {params.remap_dims}")
    print(f"  remap_tolerance: {params.remap_tolerance}")
    print(f"  original_raw_input_dir: {params.original_raw_input_dir}")
    print(f"  original_raw_output_dir: {params.original_raw_output_dir}")
    print(f"  threedtrees_dims: {params.threedtrees_dims}")
    print(f"  threedtrees_suffix: {params.threedtrees_suffix}")
    print(f"  transfer_original_dims_to_merged: {params.transfer_original_dims_to_merged}")
    print(f"  merged_resolutions: {params.merged_resolutions}")
    print(f"  merged_output_formats: {params.merged_output_formats}")

    print("=" * 60)


# Legacy compatibility: provide dict-like access for modules that need it
def get_tile_params(params: Parameters) -> dict:
    """Get tile parameters as a dictionary for legacy compatibility."""
    return {
        'tile_length': params.tile_length,
        'tile_buffer': params.tile_buffer,
        'threads': params.threads,
        'workers': params.workers,
        'resolution_1': params.resolution_1,
        'resolution_2': params.resolution_2,
        'output_copc_res1': params.output_copc_res1,
        'output_copc_res2': params.output_copc_res2,
        'subsampling_method': params.subsampling_method,
        'chunk_size': params.chunk_size,
    }


def get_merge_params(params: Parameters) -> dict:
    """Get merge parameters as a dictionary for legacy compatibility."""
    return {
        'overlap_threshold': params.overlap_threshold,
        'max_centroid_distance': params.max_centroid_distance,
        'max_volume_for_merge': params.max_volume_for_merge,
        'min_cluster_size': params.min_cluster_size,
        'workers': params.workers,
        'verbose': params.verbose,
        'instance_dimension': params.instance_dimension,
    }


def get_remap_params(params: Parameters) -> dict:
    """Get remap parameters as a dictionary for legacy compatibility."""
    return {
        'workers': params.workers,
        'instance_dimension': params.instance_dimension,
    }


# Legacy dict exports for backwards compatibility with modules that import them directly
TILE_PARAMS = {
    'tile_length': 100,
    'tile_buffer': 20,
    'threads': 10,
    'workers': DEFAULT_FILE_WORKERS,
    'resolution_1': 0.01,
    'resolution_2': 0.1,
    'output_copc_res1': True,
    'output_copc_res2': False,
    'subsampling_method': 'center-of-mass',
    'chunk_size': 20_000_000,
}

REMAP_PARAMS = {
    'target_resolution_cm': 2,
    'workers': DEFAULT_FILE_WORKERS,
}

MERGE_PARAMS = {
    'overlap_threshold': 0.3,
    'max_centroid_distance': 3.0,
    'max_volume_for_merge': 4.0,
    'min_cluster_size': 300,
    'workers': DEFAULT_FILE_WORKERS,
    'verbose': True,
    'retile_buffer': 2.0,  # Fixed to 2.0m
}


if __name__ == "__main__":
    """CLI for viewing/testing parameter configuration."""
    params = Parameters()
    print_params(params)
