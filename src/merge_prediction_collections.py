"""Helpers for preparing merge-ready prediction collection inputs."""

from pathlib import Path
from typing import Any


def comma_paths(value: str | None) -> list[Path]:
    """Parse a comma-separated path list from CLI/Galaxy parameters."""
    if not value:
        return []
    return [Path(part.strip()) for part in str(value).split(",") if part.strip()]


def prepare_merge_prediction_collection_source(
    *,
    prediction_collections: list[Path],
    reference_dir: Path | None,
    output_folder: Path,
    params: Any,
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
