# 3DTrees Smart Tiling Pipeline

**A high-performance point cloud processing pipeline for 3D tree segmentation: intelligent tiling, multi-resolution subsampling, prediction remapping, and cross-tile instance merging with species ID preservation.**

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Pipeline Architecture](#pipeline-architecture)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Detailed Usage](#detailed-usage)
7. [Pipeline Stages](#pipeline-stages)
8. [Parameters Reference](#parameters-reference)
9. [Input/Output Formats](#inputoutput-formats)
10. [Advanced Configuration](#advanced-configuration)
11. [Docker & Automation](#docker--automation)
12. [Troubleshooting](#troubleshooting)
13. [Project Structure](#project-structure)
14. [Dependencies](#dependencies)
15. [License](#license)

---

## Overview

The **3DTrees Smart Tiling Pipeline** is a production-ready system designed to process large-scale LiDAR point clouds for individual tree segmentation. It addresses the fundamental challenge of processing massive datasets that exceed memory limits by intelligently dividing point clouds into manageable tiles, processing them independently, and then seamlessly merging the results.

### The Problem

Modern airborne and terrestrial LiDAR surveys can produce datasets with billions of points covering entire forests. Deep learning-based tree segmentation models typically operate on limited spatial extents due to memory constraints. Processing such data requires:

1. **Spatial partitioning** - Dividing large datasets into manageable tiles
2. **Buffer zones** - Handling tree instances that span tile boundaries
3. **Multi-resolution processing** - Subsampling for efficient neural network inference
4. **Prediction upscaling** - Remapping low-resolution predictions back to high-resolution data
5. **Instance merging** - Reconnecting tree instances split across tiles

### The Solution

This pipeline provides an end-to-end solution with five user-facing task modes:
`tile`, `merge`, `filter`, `remap`, and `create_merged_file`.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              TILE TASK                                          │
│                                                                                 │
│  Input LAZ/LAS Files                                                            │
│         │                                                                       │
│         ▼                                                                       │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                    │
│  │ Spatial      │     │ Tile Grid    │     │ Two-Phase    │                    │
│  │ Index        │────▶│ Calculation  │────▶│ Tiling       │                    │
│  │ (tindex)     │     │ (bounds)     │     │ (laspy+COPC) │                    │
│  └──────────────┘     └──────────────┘     └──────────────┘                    │
│                                                   │                            │
│                                                   ▼                            │
│                                          ┌───────────────────┐                 │
│                                          │ Multi-Resolution  │                 │
│                                          │ Subsampling       │                 │
│                                          │ (1cm + 10cm)      │                 │
│                                          └───────────────────┘                 │
│                                                   │                            │
│                                                   ▼                            │
│                                           Outputs: tiles_100m/                 │
│                                                    ├─ c00_r00.copc.laz         │
│                                                    ├─ subsampled_1cm/          │
│                                                    └─ subsampled_10cm/         │
└─────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
                              [External Segmentation]
                         (e.g., ForAINet, SegmentAnyTree)
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              MERGE TASK                                         │
│                                                                                 │
│  Segmented 10cm Tiles (with PredInstance attribute)                            │
│         │                                                                       │
│         ▼                                                                       │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                    │
│  │ Prediction   │     │ Buffer       │     │ Cross-tile   │                    │
│  │ Remapping    │────▶│ Filtering    │────▶│ Instance     │                    │
│  │ (10cm→1cm)   │     │              │     │ Matching     │                    │
│  └──────────────┘     └──────────────┘     └──────────────┘                    │
│                                                   │                            │
│                                                   ▼                            │
│                              ┌───────────────────────────────────┐             │
│                              │ Deduplication + Small Volume Merge │            │
│                              └───────────────────────────────────┘             │
│                                                   │                            │
│                                                   ▼                            │
│                              ┌───────────────────────────────────┐             │
│                              │ Remap to Original Input Files     │             │
│                              └───────────────────────────────────┘             │
│                                                   │                            │
│                                                   ▼                            │
│                                      Unified Point Cloud                       │
│                               (Consistent Instance IDs Across Tiles)           │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Features

### High Performance
- **Parallel processing** using Python's `ProcessPoolExecutor` for multi-core utilization
- **COPC format output** via untwine (with automatic PDAL fallback) for efficient spatial queries and streaming access
- **Memory-efficient chunking** - tiles are processed independently to minimize memory footprint
- **Single-source range parallelism** - one large input file is split by point range during tile distribution so `--workers` can be used before COPC finalization
- **Parallel subsampling** - each tile is spatially divided into chunks processed concurrently

### Intelligent Tiling
- **Configurable tile size** - default 100m × 100m, adjustable for different use cases
- **Buffer zones** - overlapping regions (default 20m) ensure trees at boundaries are fully captured
- **Spatial indexing** - uses PDAL tindex for efficient data retrieval
- **Data-aligned grid** - tiles start from actual data extent, minimizing empty tiles
- **Smart tiling threshold** - optional single-file bypass for small datasets

### Multi-Resolution Processing
- **Dual subsampling** - generates both 1cm and 10cm resolution outputs (configurable)
- **Selectable voxel subsampling** - defaults to SmartTile `center-of-mass` XYZ averaging; `nearest-to-centroid` preserves the previous PDAL voxel centroid nearest-neighbor behavior
- **Explicit dimension policy** - intermediate COPC conversion strips extra dimensions by default, trying Untwine `--dims Classification` first and falling back to PDAL if validation shows extras remain; prod-merged creation preserves enriched dimensions

### Smart Instance Merging
- **Centroid-based filtering** - removes duplicate instances in buffer zones
- **Overlap ratio matching** - identifies same trees across tile boundaries using point correspondence
- **Union-Find algorithm** - efficiently groups matched instances into unified trees
- **Species ID preservation** - always preserves species from the larger instance fragment
- **Small volume merging** - reassigns orphaned tree fragments to nearby larger instances
- **Original file remapping** - maps predictions back to original input files

---

## Pipeline Architecture

### Stage-by-Stage Breakdown

#### TILE TASK: Data Preparation

| Stage | Component | Description |
|-------|-----------|-------------|
| 1 | **Spatial Index** | Creates a GeoPackage tindex using `pdal tindex` for efficient spatial queries across all input files. |
| 2 | **Tile Bounds** | Calculates optimal tile grid based on data extent, tile size (default: 100m), and buffer (default: 20m) parameters. |
| 3a | **Phase 1: Distribute** | Reads each source LAZ/LAS file once (in memory-efficient chunks via laspy), distributes points to overlapping tiles as intermediate part files. Per-tile offsets prevent int32 overflow. |
| 3b | **Phase 2: COPC Conversion** | Merges part files and converts each tile to COPC format using untwine (fast, automatic fallback to PDAL). |
| 4 | **Subsampling R1** | Downsamples COPC tiles to resolution 1 (default: 1cm) using parallel spatial chunk processing. |
| 5 | **Subsampling R2** | Further downsamples to resolution 2 (default: 10cm) for neural network inference. |

#### MERGE TASK: Result Integration

| Stage | Component | Description |
|-------|-----------|-------------|
| 0 | **Prediction Remapping** | Transfers PredInstance labels from 10cm predictions to 1cm resolution using KDTree nearest-neighbor lookup. |
| 1 | **Load and Filter** | Loads tiles, applies centroid-based buffer zone filtering to remove duplicate instances. |
| 2 | **Global ID Assignment** | Creates unique instance IDs across all tiles using tile-specific offsets. |
| 3 | **Cross-tile Matching** | Identifies matching instances in tile overlaps using overlap ratio (Union-Find grouping). |
| 3b | **Orphan Recovery** | Recovers filtered instances that no neighbor "covers", so no trees are lost. |
| 4 | **Merge and Deduplicate** | Combines all tiles, removes duplicate points from overlapping buffer regions. |
| 5 | **Small Volume Merge** | Reassigns tree fragments with volume < 4m³ to nearest large instance. |
| 6 | **Retiling** | Maps final instance IDs back to original tile boundaries for per-tile output. |
| 7 | **Original Remap** | Maps final instance IDs back to original input LAZ files (pre-tiling, optional). |

---

## Installation

### Conda Environment

```bash
# Create conda environment
mamba create -n 3dtrees -c conda-forge \
    python=3.10 \
    pdal=2.6 \
    untwine \
    gdal \
    laspy \
    lazrs-python \
    numpy \
    scipy \
    matplotlib-base \
    fiona \
    pyproj \
    geopandas \
    pydantic \
    pydantic-settings

# Activate environment
conda activate 3dtrees

# Verify installation
python src/run.py --show-params
```

### System Requirements

- **Operating System**: Linux (tested on Ubuntu 20.04+), macOS, Windows with WSL2
- **Memory**: Minimum 8GB RAM, 16GB+ recommended for large datasets
- **CPU**: Multi-core processor recommended (parallel processing scales with cores)
- **Storage**: SSD recommended for I/O-intensive operations
- **PDAL**: Version 2.5 or higher
- **GDAL**: Version 3.0 or higher

---

## Quick Start

### Basic Tile Task

Process a directory of LAZ files into tiled, multi-resolution outputs:

```bash
python src/run.py --task tile \
    --input-dir /path/to/input \
    --output-dir /path/to/output
```

### Basic Merge Task

Filter duplicate buffer-zone instances from segmented predictions, remap the
filtered predictions to the target resolution, then merge the remapped tiles.
The task always writes merged per-tile outputs for downstream processing and can
also write the current processed merged LAZ (requires **tile_bounds_tindex.json**
from the Tile task):

```bash
python src/run.py --task merge \
    --subsampled-segmented-folder /path/to/subsampled_10cm \
    --subsampled-target-folder /path/to/subsampled_1cm \
    --tile_bounds_json /path/to/tile_bounds_tindex.json \
    --output-folder /path/to/out \
    --output-merged-laz /path/to/out/merged.laz
```

Optional: add `--original-laz-input-dir /path/to/original` to enrich uploaded
originals from the merged 1cm tile outputs. Use `--original-laz-output-dir` to
choose that folder.

### Basic Filter Task

Filter segmented or remapped tiles by removing instances whose centroids sit in
overlap buffers facing neighboring tiles:

```bash
python src/run.py --task filter \
    --input-dir /path/to/segmented_remapped \
    --output-dir /path/to/filtered_tiles \
    --buffer 10.0 \
    --instance-dimension PredInstance
```

The filter task writes one output file per input file and preserves all point
dimensions in the kept points. Use `--filter-suffix` to change the default
`_filtered` filename suffix. Use `--filter-output-extension .laz` when mixed
LAZ/LAS inputs should be collected as LAZ outputs.

### Create Prod-Merged Products

Create user-facing prod-merged files from `original_with_predictions/`. This mode serves two
production goals:

1. Write a final merged point-cloud product next to the enriched original files after remap.
2. Merge multiple uploaded source files into one point-cloud product while preserving CRS,
   source dimensions, scales, offsets, and truthful LAS/COPC metadata as far as the output
   format allows.

Inputs are first staged through preservation-mode COPC, then real original points are
selected with nearest-to-centroid product downsampling:

```bash
python src/run.py --task create_merged_file \
    --original-with-predictions-dir /path/to/original_with_predictions \
    --output-dir /path/to/products \
    --staged-copc-dir /path/to/already_converted_original_with_predictions_copc \
    --standardization-json /path/to/collection_summary.json \
    --merged-resolutions res1,res2 \
    --merged-output-formats copc.laz
```

By default this writes `prod_merged_1cm.copc.laz` and `prod_merged_10cm.copc.laz`.
Use `--merged-output-formats laz,copc.laz,ply` to write multiple formats for each selected resolution.
The intermediate COPC files are written under `original_with_predictions_copc/` in the output directory.
If matching `.copc.laz` files are already present for an Original-with-predictions source, they are reused and the matching raw LAZ/LAS file is not staged a second time.
Use `--staged-copc-dir` to reuse an explicit COPC cache from a previous product or validation run. SmartTile checks that each staged COPC has a readable header before using it, so interrupted partial conversions are ignored and rebuilt in the current output directory.
Use `--standardization-json` with the tool_standard `collection_summary.json` to validate that staged Original-with-predictions COPCs and LAS/COPC prod-merged outputs still expose the expected standardized source dimensions. This restores the v2.1 schema guard; it does not filter prediction dimensions.

### Basic Remap Task (merged file → original files)

Add 3Dtrees prediction dimensions from a merged LAZ file to the original files,
then create prod-merged products from those Original-with-predictions files:

```bash
python src/run.py --task remap \
    --merged-laz /path/to/merged.laz \
    --original-input-dir /path/to/original/files \
    --output-dir /path/to/original_with_predictions \
    --merged-resolutions res1,res2 \
    --merged-output-formats laz,copc.laz
```

When `--merged-laz` points to a `.copc.laz` file, SmartTile uses a bounded
streaming remap path: uploaded original LAZ/LAS files are read in
`--chunk-size` point chunks, each original chunk queries the merged COPC by its
XY bounds plus remap buffer, a local KDTree is built only for that window, and
the enriched original chunk is written immediately. This keeps the uploaded
LAZ/LAS file as the metadata and geometry source while avoiding a full-original
or full-merged KDTree in memory. Plain merged `.laz` inputs fall back to the
legacy loaded-merged-cloud path.

### Multi-Collection Remap Task (prediction collections → original files)

Finalized prediction collections can be remapped together onto the original
files. This is intended for model outputs that have already been filtered and
merged independently. SmartTile preserves prediction dimension names exactly as
provided; model-specific names such as `PredInstance_SAT`,
`PredInstance_ForestMamba`, `species_id_sat`, and `species_prob_foma` must be
present before this step.

```bash
python src/run.py --task remap \
    --segmented-folders /path/to/sat_predictions,/path/to/foma_predictions,/path/to/species_predictions \
    --original-copc-input-dir /path/to/original_copc_files \
    --original-laz-input-dir /path/to/uploaded/raw_laz_files \
    --original-laz-output-dir /path/to/original_with_predictions_raw \
    --remap-dims PredInstance_SAT,PredSemantic_SAT,PredInstance_ForestMamba,species_id_sat,species_prob_sat,species_id_foma,species_prob_foma \
    --chunk-size 10000000 \
    --produce-merged-file \
    --merged-resolutions 1cm \
    --merged-output-formats laz
```

If `--remap-dims` is omitted, all extra dimensions from every prediction
collection are transferred. Duplicate extra-dimension names across prediction
collections fail early; SmartTile does not auto-rename them to `_2`, `_3`, or
add late model suffixes during final remap.

Multi-collection remap writes each Original-with-predictions file in one pass:
for every original-point chunk it loads the overlapping prediction points from
each finalized collection, transfers all selected dimensions, and writes the
enriched original chunk once. This avoids temporary per-collection LAZ rewrites.
Use `--chunk-size` to tune the memory/speed tradeoff. Larger chunks reduce
repeated prediction-window scans but increase peak memory; for large two-file
datasets under a 30GB container cap, `10000000` points per chunk has been a
useful validation setting.
When there is only one raw LAZ/LAS original, SmartTile can still use the worker
budget by enriching multiple original chunks concurrently. `--num-spatial-chunks`
caps that raw chunk concurrency; if omitted, it defaults to `--workers`.

Prod-merged output creation is controlled by `--produce-merged-file` /
`--no-produce-merged-file` (aliases for the existing
`--transfer-original-dims-to-merged` toggle). When enabled, the outputs are
created from `original_with_predictions_raw/` using `--merged-resolutions`,
`--merged-output-formats`, `--staged-copc-dir`, and `--standardization-json`.

Prediction collections stored as COPC are loaded with bounded COPC spatial
queries when remap needs a local prediction window.

For production downloads, pass the uploaded LAZ/LAS files to
`--original-laz-input-dir`; this is the writer and metadata source for
`original_with_predictions_raw/`. `--original-input-dir` remains accepted as a
legacy alias for the same LAZ/LAS source. If original COPCs are available, pass
them to `--original-copc-input-dir` only as a matching/validation lane. SmartTile
validates that COPC and LAZ sources match, but it does not write an enriched
COPC-original intermediate. Selected prediction dimensions are written directly
onto the uploaded LAZ/LAS files, so headers, GeoTIFF/GeoKey VLRs, scales,
offsets, point format, and non-prediction dimensions come from the uploaded file
itself. Prod-merged `copc.laz`, `laz`, and `ply` outputs are then created from
the enriched uploaded LAZ/LAS files. COPC-derived products preserve CRS
semantically, but a LAZ -> COPC conversion may represent the same CRS as WKT VLR
rather than the original GeoKey VLRs. Therefore SmartTile does not promise that a
COPC-derived LAZ is byte-identical to enriching the raw uploaded LAZ directly.

### View Current Parameters

```bash
python src/run.py --show-params
```

---

## Detailed Usage

### Tile Task Options

```bash
python src/run.py --task tile \
    --input-dir /path/to/input \           # Required: Directory with LAZ/LAS files
    --output-dir /path/to/output \         # Required: Output directory
    --tile-length 100 \                    # Tile size in meters (default: 100)
    --tile-buffer 20 \                     # Buffer overlap in meters (default: 20)
    --resolution-1 0.01 \                  # First resolution (default: 1cm)
    --resolution-2 0.1 \                   # Second resolution (default: 10cm)
    --output-copc-res1 True \              # 1cm output as COPC LAZ (default: True)
    --output-copc-res2 False \             # 10cm output as regular LAZ (default: False)
    --workers 8 \                          # Parallel workers (default: 4)
    --threads 10                           # Threads per COPC writer (default: 10)
```

### Create Merged File Task Options

```bash
python src/run.py --task create_merged_file \
    --original-with-predictions-dir /path/to/original_with_predictions \
    --output-dir /path/to/products \
    --resolution-1 0.01 \
    --resolution-2 0.1 \
    --merged-resolutions res1,res2 \
    --merged-output-formats laz,copc.laz,ply \
    --staged-copc-dir /path/to/products/original_with_predictions_copc \
    --standardization-json /path/to/collection_summary.json \
    --num-spatial-chunks 10
```

`--merged-resolutions` accepts `res1`, `res2`, numeric meter values such as `0.05`, or centimeter labels such as `1cm,10cm`.
`--merged-output-formats` accepts `laz`, `copc.laz`, and `ply`; it can contain one or several comma-separated formats.
The task stages LAZ/LAS inputs to COPC in preservation mode with untwine when available, falling back to PDAL `writers.copc`, before merging and product downsampling. Existing matching `.copc.laz` files are reused so one source is not merged twice.
`--staged-copc-dir` points to a reusable cache of already converted Original-with-predictions COPCs. This is recommended for repeat validation runs and production reruns where the enriched originals have not changed.
`--standardization-json` points to the standardization `collection_summary.json` and validates that expected non-constant source dimensions survived into the staged COPCs and final LAS/COPC prod-merged products.
LAZ and COPC outputs use LAS/COPC metadata forwarding. PLY outputs carry point dimensions as PLY properties, but do not preserve LAS/COPC VLR metadata such as CRS records.
`--num-spatial-chunks` controls bounded COPC reads for prod-merged creation. For large UTM datasets, prefer setting it to the available CPU budget (for example `10`) instead of using a single global merge.
For COPC output, SmartTile first writes bounded LAZ chunks and then prefers direct `untwine` chunk-to-COPC finalization. This keeps RAM bounded and avoids a giant merged temporary LAZ, but it still needs scratch disk for the chunk files and untwine hierarchy/output staging. Direct untwine output is accepted only when its point count exactly matches the source chunk total; otherwise SmartTile falls back to the PDAL merge/conversion path.
When multiple product formats are selected for the same resolution, SmartTile generates one canonical set of nearest-to-centroid chunk LAZ files and writes all selected formats from those same chunks. This avoids repeated chunk computation and keeps LAZ, COPC LAZ, and PLY point counts aligned for a given resolution.
Chunked nearest-to-centroid product generation is designed to preserve product metadata, CRS, scales, offsets, and dimensions. It should not be treated as a bit-for-bit reproducible operation: different chunking, PDAL/untwine versions, or parallel execution details may change the selected representative point at voxel boundaries while preserving the same spatial extent and metadata contract.

### Local Validation

Run the unit suite from the tool directory:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

The suite covers output format validation, metadata/header preservation helpers, COM subsampling method selection, scientific-notation bounds parsing, one-pass multi-collection remap behavior, and instance-label dtype rules.

### Merge Task Options

**Required:** `--subsampled-segmented-folder`, `--subsampled-target-folder`, `--tile_bounds_json` (from Tile task).

```bash
python src/run.py --task merge \
    --subsampled-segmented-folder /path/to/10cm \   # Segmented 10cm tiles
    --subsampled-target-folder /path/to/1cm \      # Subsampled 1cm tiles (remap target)
    --tile_bounds_json /path/to/tile_bounds_tindex.json \
    --output-folder /path/to/out \                  # Optional; default: parent of segmented
    --output-merged-laz /path/to/out/merged.laz \  # Optional; merged LAZ path
    --original-copc-input-dir /path/to/original_copc \ # Optional; matching/validation source
    --original-laz-input-dir /path/to/raw_original \   # Optional; uploaded LAZ/LAS source to enrich
    --merged-resolutions res1,res2 \                # Prod-merged outputs from original_with_predictions
    --buffer 10.0 \                                # Buffer zone distance (default: 10m)
    --overlap-threshold 0.3 \                      # Instance matching (default: 0.3)
    --max-centroid-distance 3.0 \                   # Max centroid distance (default: 3m)
    --workers 8 \                                  # Parallel workers (default: 4)
    --disable-matching                             # Disable cross-tile matching
```

---

## Pipeline Stages

### Stage 1: Spatial Indexing

**Purpose**: Create a spatial index (tindex) for efficient querying across all input files.

**Technology**: Uses `pdal tindex` to create a GeoPackage with file boundaries.

**Output**: `tindex_100m.gpkg` containing polygons representing each input file's extent.

### Stage 2: Tile Grid Calculation

**Purpose**: Compute optimal tile boundaries based on data extent and parameters.

**Algorithm**:
1. Load extent from tindex
2. Apply grid offset to starting coordinates
3. Create tiles of `tile_length` × `tile_length` meters
4. Add `tile_buffer` meters to each side
5. Generate job list with projected and geographic bounds

**Output Files**:
- `tile_bounds_tindex.json`: Complete tile metadata
- `tile_jobs_100m.txt`: Per-tile processing instructions
- `overview_copc_tiles.png`: Visualization of tiles and input files

### Stage 3: Tile Creation (Two-Phase)

**Purpose**: Distribute points from source LAZ/LAS files into spatially partitioned tiles in COPC format.

#### Phase 1: Distribute

Each source file is read with laspy in memory-efficient chunks controlled by `--chunk-size`. Points are distributed to all overlapping tiles as intermediate `.las` part files. Per-tile offsets are computed from tile bounds to prevent int32 overflow in scaled coordinates.

**Parallelization**: Multiple source files are distributed in parallel. If there is only one large source file, SmartTile splits it into bounded point ranges so `--workers` can be used before Untwine/PDAL finalizes the tile COPCs. The per-worker point chunk is reduced from `--chunk-size` so peak concurrent memory stays close to the caller's chunk budget.

#### Phase 2: COPC Conversion

All part files for each tile are merged and converted to COPC format. Untwine is preferred for COPC writing. Intermediate COPC conversion uses Untwine's `--dims Classification` path by default while forwarding header metadata and CRS records, then validates that no extra byte dimensions remain. If Untwine fails, CRS validation fails, or extra dimensions remain, SmartTile falls back to PDAL `writers.copc` without `extra_dims=all`. Prod-merged creation is the explicit exception and preserves enriched dimensions for final products.

The SmartTile Docker image has been validated against Untwine 1.5.1: `--dims ""` is rejected; `--dims Classification` can leave prediction or source extras on enriched files; and long explicit standard-dimension lists such as `--dims Red,Green,Blue` are not used because they can crash this build. SmartTile therefore treats Untwine dimension limiting as the first attempt and accepts it only after output inspection confirms there are no extra byte dimensions.

**Parallelization**: Multiple tiles converted concurrently (controlled by `--workers`).

**Output**: `c{col}_r{row}.copc.laz` files in `tiles_{tile_length}m/`.

**Options**:
- **Default**: Try Untwine `--dims Classification`, inspect the COPC output, and fall back to PDAL standard-dimension output if extra dimensions remain.
- **Prod-merged exception**: `create_merged_file` preserves extra dimensions in staged COPCs and final LAZ/COPC products.

### Stage 4-5: Multi-Resolution Subsampling

**Purpose**: Create downsampled versions for efficient neural network processing.

**Algorithm**: Voxel-based downsampling using the selected `--subsampling-method`.

- `center-of-mass` (default): averages only X/Y/Z within each populated voxel. Non-coordinate dimensions are copied from the real point nearest to that averaged XYZ when dimensions are preserved; they are never averaged.
- `nearest-to-centroid`: uses PDAL `filters.voxelcentroidnearestneighbor`, preserving the previous SmartTile behavior.

**Process**:
1. COPC `center-of-mass` runs voxel-aligned COPC windows in parallel, controlled by `--num-spatial-chunks`
2. Other subsampling paths split each tile spatially into chunks along X-axis
3. Chunks/windows are processed in parallel
4. Results are merged back into single file

**Outputs**:
- `subsampled_res1/`: Resolution_1 files, default 1cm COPC LAZ (`*.copc.laz`)
- `subsampled_res2/`: Resolution_2 files, default 10cm regular LAZ (`*.laz`)

### The Merge Process in Detail

The merge process takes segmented tiles (each with local `PredInstance` IDs) and produces a single unified point cloud where every tree has a globally unique ID, even trees that were split across tile boundaries. It also writes per-tile outputs and optionally maps predictions back to the original input files.

The diagram below shows the spatial layout. Tiles overlap by a configurable buffer (default 10 m). Trees near tile edges appear in both tiles with **different** local instance IDs. The merge must figure out which IDs in adjacent tiles refer to the same physical tree and unify them.

```
Tile A                           Tile B
┌──────────────────────┐  ┌──────────────────────┐
│                      │  │                      │
│   Tree 42            │  │            Tree 17   │
│      ╲               │  │               ╱      │
│       ╲   buffer     │  │    buffer    ╱       │
│        ╲  zone ──────┤  ├──── zone    ╱        │
│         ╲   ▓▓▓▓▓▓▓▓▓│  │▓▓▓▓▓▓▓▓▓  ╱         │
│          ╲  ▓overlap▓│  │▓overlap▓  ╱          │
│           ╲ ▓▓▓▓▓▓▓▓▓│  │▓▓▓▓▓▓▓▓▓ ╱           │
│            ╲──────────┤  ├──────────╱            │
│                      │  │                      │
│   Tree 42 and Tree 17 are the SAME physical    │
│   tree — the merge must unify them.            │
└──────────────────────┘  └──────────────────────┘
```

Below is each stage explained.

---

### Stage 0: Prediction Remapping (10 cm to 1 cm)

**What it does**: The segmentation model ran on 10 cm subsampled tiles. This stage transfers the `PredInstance` labels from the coarse 10 cm points to the finer 1 cm subsampled points using nearest-neighbor lookup, so subsequent stages work at 1 cm resolution.

**How it works**:
1. For each tile, load the 10 cm segmented file and the corresponding 1 cm subsampled file.
2. Build a cKDTree from the 10 cm points.
3. For every 1 cm point, find the closest 10 cm point and copy its `PredInstance` (and `species_id` if present).
4. Save the result as `{tile_id}_segmented_remapped.laz`.

**Why**: Segmentation at 10 cm is fast but loses detail. Remapping to 1 cm gives the merge much denser point clouds to work with, which improves overlap detection and deduplication accuracy.

---

### Stage 0b: Pre-Merge Buffer Filtering

**What it does**: Filters segmented prediction tiles before remap. In the
standard merge lane, `task merge` writes filtered 10 cm predictions to
`segmented_filtered/`, remaps that directory to the target resolution, and feeds
the remapped output into merge.

**Why**: Filtering is part of the production merge lane, so operators do not
need to run a separate `filter` task before merge.

---

### Stage 1: Load and Merge Filtered Predictions

**What it does**: Loads filtered-then-remapped target-resolution prediction tiles and prepares
kept instances, border-region candidates, and recoverable filtered instances for
cross-tile matching.

**How it works**:

1. Load each LAZ tile; read XYZ coordinates, `PredInstance`, and any extra dimensions (e.g. `species_id`).
2. Use the `tile_bounds_tindex.json` file (from the Tile task) to determine which tiles are neighbors (east/west/north/south).
3. For each tile, compute the centroid of every instance.
4. An instance is **filtered** (removed) if its centroid falls in the buffer zone on a side that has a neighbor. The rule is: the tile with the lower column index (west) or lower row index (south) "owns" instances in the overlap.
5. Instances **not** in any buffer zone are **kept**.

**Result**: Each tile now has a set of "kept" instances and a set of "filtered" instances. Filtered instances are candidates for orphan recovery later.

```
Tile boundary:
┌──────────────────────────────────────┐
│ buffer │                     │ buffer│
│  zone  │     CORE AREA       │  zone │
│ (west) │  (instances here     │(east) │
│        │   are always kept)   │       │
│ ▓▓▓▓▓▓ │                     │▓▓▓▓▓▓│
│ filtered│                     │filtered│
│ if west │                     │if east│
│ neighbor│                     │neighbor│
└──────────────────────────────────────┘
```

---

### Stage 2: Global ID Assignment

**What it does**: Assigns globally unique IDs to every kept instance across all tiles.

**How it works**:
- Each tile gets an offset: `tile_idx * 100000`.
- A local instance ID `42` in tile 3 becomes global ID `300042`.
- A Union-Find data structure is initialized with one entry per kept instance, tracking its point count (used later to preserve the species ID from the larger fragment).

**Why**: Local IDs are only unique within one tile. Subsequent stages need IDs that are unique across the entire dataset.

---

### Stage 3: Border Region Instance Matching

**What it does**: Finds instances that represent the **same physical tree** across tile boundaries and groups them using Union-Find, so they end up with the same final ID.

**How it works**:

1. **Identify border instances**: For each tile, find kept instances whose centroid lies in the "border region" (the strip from `buffer` to `buffer + border_zone_width` meters from the tile edge, on sides with a neighbor). Only these instances can possibly match a counterpart in a neighbor tile.

2. **For each neighbor pair** (e.g. tile A east <-> tile B west): compare border instances from tile A facing east with border instances from tile B facing west:
   - **Bounding box check** (fast filter): Skip pairs whose 2D bounding boxes don't overlap or come within 10 cm. This eliminates most non-matching pairs cheaply.
   - **FF3D overlap ratio** (expensive check): For surviving pairs, compute point-to-point correspondence using hash-based grid matching (O(n) per pair). The overlap ratio is `max(intersection/size_a, intersection/size_b)` -- this handles asymmetric cases where one fragment is much larger than the other.
   - **Match decision**: If the overlap ratio exceeds `overlap_threshold` (default 0.3 = 30%), the pair is matched. Union-Find merges their global IDs into one group.

3. **Species ID preservation**: When instances are grouped, the species ID is always taken from the **larger** instance (by point count).

**Result**: A mapping from every global ID to a "merged ID". Matched instances share the same merged ID.

```
Tile A (east border)         Tile B (west border)
┌──────────┐                 ┌──────────┐
│          │   border zone   │          │
│ inst 42 ─┼─────────────────┼─ inst 17 │  overlap ratio = 0.65
│ (500 pts)│                 │ (480 pts)│  > threshold 0.3 → MATCH
│          │                 │          │
│ inst 99 ─┼─────────────────┼─ (none)  │  no counterpart → no match
│          │                 │          │
└──────────┘                 └──────────┘
```

---

### Stage 3b: Orphan Recovery

**What it does**: Recovers filtered instances that would otherwise be lost because no neighbor tile "owns" them (e.g. a tree whose centroid landed in the buffer zone of **both** tiles due to slightly different segmentation results).

**How it works**:

1. Compute bounding boxes **only** for filtered instances and kept instances in the border region (not all instances -- this is fast).
2. Build a cKDTree of the centers of all border-region kept instances.
3. For each filtered (orphan) instance:
   - Query the tree for kept instances within 30 m.
   - For each nearby kept instance, check if their points overlap (>50% of the orphan's points are within 1 m of the neighbor's points, using a per-neighbor cKDTree that is cached so each neighbor tree is built at most once).
   - If a covering instance is found, the orphan is skipped (it already exists in a neighbor tile).
   - If **no** covering instance is found, the orphan is **recovered**: it gets a new unique merged ID and is added back to the kept set.

**Why**: Without this step, trees at tile corners (where buffer zones of multiple tiles overlap) can be lost entirely.

---

### Stage 4: Merge and Deduplicate

**What it does**: Concatenates all kept points from all tiles into a single point cloud and removes duplicate points that exist in overlapping buffer regions.

**How it works**:

1. For each tile, remap local instance IDs to their final merged IDs (from Stage 3). Points belonging to filtered instances that were not recovered get ID `-1` and are discarded.
2. Concatenate all tile points, instance arrays, and extra dimension arrays.
3. **Deduplication**: Uses grid-based spatial hashing (not KDTree) for O(n) performance:
   - Divide the point cloud into 50 m grid cells.
   - Within each cell, hash each point's coordinates at 1 cm resolution.
   - Points with the same hash in the same cell are duplicates; keep the one with the higher instance ID.

**Result**: A single merged point cloud with no duplicate points and consistent instance IDs.

---

### Stage 5: Small Volume Instance Merging

**What it does**: Reassigns tiny tree fragments (orphaned clusters) to the nearest large instance.

**How it works**:

1. For each instance with a positive ID, compute its 3D convex hull volume (in parallel using ProcessPoolExecutor).
2. Classify instances as "small" if: volume < `max_volume_for_merge` (default 4 m3) **or** point count < `min_cluster_size` (default 300).
3. Build a cKDTree from the centroids of all "large" instances.
4. For each small instance, find the nearest large instance by centroid distance.
5. Reassign all points of the small instance to the nearest large instance (vectorized lookup table, no per-point loop).

**Species preservation**: The species ID of the receiving (larger) instance is kept unchanged. Small fragments do not overwrite species.

---

### Stage 6: Retiling to Original Tile Files

**What it does**: Maps the final merged instance IDs back onto each original tile's point cloud, so you get per-tile output files with globally consistent IDs.

**How it works**:

1. For each original tile file:
   - Read the tile's bounding box from its header.
   - Filter the merged point cloud to points within `tile bounds + spatial buffer`.
   - Build a cKDTree from those filtered merged points.
   - Read the original tile's points and query the tree for the nearest merged point for each original point.
   - Write the tile with the matched `PredInstance` (and any extra dims like `species_id`).

**Parallelism**: Can process multiple tiles concurrently using `--workers`.

---

### Stage 7: Uploaded Original Remapping (Optional)

**What it does**: If `--original-laz-input-dir` is provided, maps the final
merged 1 cm tile prediction dimensions back to the uploaded original LAZ/LAS
files before tiling. `--original-input-dir` remains a legacy alias for this
raw-LAZ lane. If matching original COPCs are available, pass them via
`--original-copc-input-dir` for source matching/validation; SmartTile still
writes enriched originals from the uploaded LAZ/LAS files so the original
metadata/VLRs remain the writer source.

**How it works**: Merge uses the same prediction-collection-to-original remap
helper as the `remap` task. The post-merge 1 cm `output_tiles/` directory is the
prediction collection, so uploaded originals receive the globally merged IDs from
the per-tile products. `--skip-merged-file` can be used with original remap when
only per-tile outputs, enriched originals, or prod-merged products are needed.

---

## Parameters Reference

### Tile Task Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--tile-length` | 100 | Tile size in meters |
| `--tile-buffer` | 20 | Buffer overlap in meters |
| `--threads` | 10 | Threads per COPC writer |
| `--workers` | 4 | Parallel file/tile processing |
| `--num-spatial-chunks` | `--workers` | Per-file subsampling parallelism for COM windows or stripe chunks |
| `--resolution-1` | 0.01 | First subsampling resolution (1cm) |
| `--resolution-2` | 0.1 | Second subsampling resolution (10cm) |
| `--output-copc-res1` | True | Write first-resolution subsampled outputs as COPC LAZ (`*.copc.laz`) |
| `--output-copc-res2` | False | Write second-resolution subsampled outputs as COPC LAZ; default keeps 10cm as regular LAZ |
| `--subsampling-method` | center-of-mass | Subsampling method: `center-of-mass` or `nearest-to-centroid` |
| `--chunk-size` | 20000000 | Points per chunk when reading LAZ/LAS in tiling Phase 1, multi-collection remap, and merged-COPC-to-original remap (smaller = less peak RAM; larger = fewer scans) |
| `--tiling-threshold` | None | File size threshold in MB for skipping tiling on single small files |

### Merge Task Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--tile_bounds_json` | **Required** | Path to tile_bounds_tindex.json from Tile task |
| `--buffer` | 10.0 | Buffer distance for filtering (meters) |
| `--border-zone-width` | 10.0 | Width of border zone beyond buffer for instance matching (meters) |
| `--overlap-threshold` | 0.3 | Overlap ratio for instance matching (30%) |
| `--max-centroid-distance` | 3.0 | Max centroid distance to merge (meters) |
| `--max-volume-for-merge` | 4.0 | Max volume for small instance merge (m³) |
| `--min-cluster-size` | 300 | Minimum cluster size in points for reassignment |
| `--original-laz-input-dir` | None | Optional: uploaded original LAZ/LAS files to enrich; `--original-input-dir` is a legacy alias |
| `--original-laz-output-dir` | None | Optional output directory for enriched uploaded originals |
| `--original-copc-input-dir` | None | Optional matching original COPC LAZ files for source matching/validation |
| `--merged-resolutions` | res1,res2 | Prod-merged output resolutions when original inputs are available |
| `--merged-output-formats` | copc.laz | Prod-merged output formats: `laz`, `copc.laz`, `ply` |
| `--skip-merged-file` | False | Skip creating the processed merged LAZ; per-tile outputs and optional original/prod-merged outputs can still be written |
| `--disable-matching` | False | Disable cross-tile instance matching |
| `--disable-volume-merge` | False | Disable small volume instance merging |
| `--workers` | 4 | Parallel processing (tile loading, KDTree queries) |
| `--tolerance` | 5.0 | Max difference in meters for bounds matching (remap task) |

*Retile buffer is fixed internally at 2.0 m; correspondence tolerance is no longer a user parameter.*

### Filter Task Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input-dir` | **Required** | Directory with segmented/remapped LAZ/LAS tile files |
| `--output-dir` | **Required** | Directory for filtered tile outputs |
| `--buffer` | 10.0 | Buffer distance in meters |
| `--instance-dimension` | PredInstance | Instance ID dimension to filter; falls back to `treeID` when absent |
| `--filter-suffix` | _filtered | Suffix added to output filenames |
| `--filter-output-extension` | None | Optional extension override such as `.laz`; by default the input extension is preserved |

### Understanding `--workers`, `--threads`, and `--num-spatial-chunks`

These parameters control different aspects of parallelism:

#### `--workers` (Global Parallelism)

Controls how many files/tasks run simultaneously using Python's `ProcessPoolExecutor`:

| Task | What `--workers` Controls |
|------|---------------------------|
| **Tile Task** | Parallel source-file distribution; one large source is split into point ranges; tile COPC finalization runs in parallel |
| **Merge Task** | Parallel tile loading, parallel convex hull computation, KDTree queries |
| **Remap Task** | Parallel files/tiles; for one raw original, parallel original chunks; KDTree query workers are divided across the active workers |

**Memory impact**: Higher values = more files or remap chunks in memory simultaneously. Remap keeps the total KDTree CPU budget bounded by sharing `--workers` across outer remap workers, raw chunk workers, and inner SciPy query workers.

For tile distribution of a single large source, SmartTile caps each worker's laspy chunk size to approximately `--chunk-size / --workers` with a 100k-point floor. This prevents `--workers=20 --chunk-size=20_000_000` from trying to hold 20 full 20M-point chunks at once.

#### `--threads` (COPC Writer Threads)

Controls tiling/COPC writer threading. It does not control subsampling parallelism.

#### `--num-spatial-chunks` (Subsampling Parallelism)

Controls per-file or per-product spatial chunking:

- COPC `center-of-mass` uses `--num-spatial-chunks` voxel-aligned COPC window workers
- Other paths split each tile into `--num-spatial-chunks` spatial chunks along the X-axis
- `create_merged_file` uses `--num-spatial-chunks` for bounded COPC product reads before final COPC/LAZ/PLY writing
- Remap uses `--num-spatial-chunks` as the number of native COPC spatial-query windows when original files are COPC
- Remap uses `--num-spatial-chunks` as the maximum number of concurrent raw LAZ/LAS original chunks when only one original file is being enriched
- Subsampling chunks/windows are processed in parallel using `ProcessPoolExecutor`
- Product chunks are currently processed sequentially to keep memory and disk pressure predictable

```
Example with --num-spatial-chunks=5:
  tile.laz → [chunk/window 0..4] → parallel subsample → merge
  original_with_predictions/*.copc.laz → [bounds 0..4] → product chunks → prod_merged_*.copc.laz
  original.laz → [chunk 0..4] → parallel prediction remap → original_with_predictions/original.laz
```

**Memory and disk impact**: Higher values increase subsampling worker concurrency and reduce per-product read windows. Tune down if memory or storage I/O becomes the bottleneck; tune up when bounded product chunks are still too large.

If not specified, this defaults to `--workers`.



---

## Input/Output Formats

### Input Requirements

#### Tile Task
- **File formats**: LAZ (compressed) or LAS (uncompressed)
- **Coordinate system**: Should be in a projected CRS (e.g., UTM)
- **Directory structure**: Flat directory with LAZ/LAS files

#### Merge Task
- **Required attribute**: `PredInstance` (integer instance IDs)
- **Optional attributes**: `PredSemantic`, `species_id`
- **File naming**: `c{col}_r{row}*.laz` pattern

### Output Structure

```
output_dir/
├── tiles_100m/                  # Tiled point clouds (100m default)
│   ├── c00_r00.copc.laz         # COPC tiles (Phase 2 output)
│   ├── c00_r01.copc.laz
│   └── c01_r00.copc.laz
│
├── subsampled_res1/             # Resolution 1 subsamples (1cm COPC LAZ by default)
│   ├── output_100m_c00_r00_subsampled_1cm.copc.laz
│   └── ...
│
├── subsampled_res2/             # Resolution 2 subsamples (10cm regular LAZ by default)
│   ├── output_100m_c00_r00_subsampled_10cm.laz
│   └── ...
│
├── segmented_filtered/          # Filtered segmented predictions before remap (merge task)
│   ├── c00_r00_segmented_filtered.laz
│   └── ...
│
├── segmented_remapped/          # Filtered predictions remapped to target resolution (merge task)
│   ├── c00_r00_segmented_remapped.laz
│   └── ...
│
├── output_tiles/                # Final per-tile outputs with merged IDs
│   ├── c00_r00.copc.laz
│   └── ...
│
├── original_with_predictions/   # Original files with PredInstance (merge task)
│   ├── input_file_1.laz
│   └── input_file_2.laz
│
├── products/
│   ├── original_with_predictions_copc/
│   │   ├── input_file_1.copc.laz
│   │   └── input_file_2.copc.laz
│   ├── prod_merged_1cm.copc.laz  # Default prod-merged product
│   └── prod_merged_10cm.copc.laz
│
├── tindex_100m.gpkg             # Spatial index
├── tile_bounds_tindex.json      # Tile metadata
├── tile_jobs_100m.txt           # Processing jobs
├── overview_copc_tiles.png      # Visualization
│
└── logs/                        # Processing logs
```

### Point Cloud Attributes

#### Tile Task Output
- `X`, `Y`, `Z`: 3D coordinates
- Intermediate COPC conversion strips extra dimensions by default; Untwine `--dims Classification` is accepted only when output inspection confirms no extra byte dimensions remain.
- `create_merged_file` preserves enriched dimensions for prod-merged outputs.

#### Merge Task Output
- `X`, `Y`, `Z`: 3D coordinates
- `PredInstance`: Global tree instance ID (consistent across tiles)
- `PredSemantic`: Semantic class (if present in input)
- `species_id`: Tree species ID (if present in input)

---

## Advanced Configuration

### Large Dataset Processing

For datasets exceeding 100GB with larger tiles:

```bash
python src/run.py --task tile \
    --input-dir /data/input \
    --output-dir /data/output \
    --tile-length 500 \
    --tile-buffer 30 \
    --workers 32 \
    --threads 10 \
    --num-spatial-chunks 10
```

### High-Precision Processing

For research applications requiring maximum fidelity (already default):

```bash
python src/run.py --task tile \
    --input-dir /data/input \
    --output-dir /data/output \
    --tile-length 300 \
    --tile-buffer 20 \
    --resolution-1 0.01 \
    --resolution-2 0.05
```

### Memory-Constrained Systems

For systems with limited RAM (smaller tiles, lower resolution):

```bash
python src/run.py --task tile \
    --input-dir /data/input \
    --output-dir /data/output \
    --tile-length 100 \
    --tile-buffer 10 \
    --workers 2 \
    --threads 2 \
    --num-spatial-chunks 2 \
    --resolution-1 0.02 \
    --resolution-2 0.15
```

### Single Small File Processing

For processing single files without tiling:

```bash
python src/run.py --task tile \
    --input-dir /data/input \
    --output-dir /data/output \
    --tiling-threshold 1000  # Skip tiling if single file < 1000 MB
```

---

## Docker & Automation

The pipeline includes Docker support for containerized execution and automated workflows with resource monitoring.

### Docker Setup

Build the Docker image:

```bash
docker build -t 3dtrees_smart_tile .
```

Run a task in Docker by mounting input and output folders and calling `run.py`:

```bash
docker run --rm --user "$(id -u):$(id -g)" --cpus=20 --memory=50g \
  -v /path/to/input:/in:ro \
  -v /path/to/output:/out \
  3dtrees_smart_tile \
  python -u /src/run.py --task tile --input-dir /in --output-dir /out
```

Merge requires `tile_bounds_tindex.json` from the Tile task output. The Galaxy
wrapper uses the same `run.py` task surface inside
`ghcr.io/3dtrees-earth/3dtrees_smart_tile:2.2`.

### Additional Documentation

- [Dockerfile](Dockerfile) - Container configuration

---

## Troubleshooting

### Common Issues

#### "No LAZ/LAS files found"
- Ensure input files have `.laz` or `.las` extension (lowercase)
- Check that `input_dir` points to the correct directory
- Verify file permissions

#### "pdal: command not found"
- Install PDAL: `conda install -c conda-forge pdal`
- Verify installation: `pdal --version`
- Check PATH environment variable

#### "untwine: command not found"
- The pipeline automatically falls back to PDAL's `writers.copc` if untwine is not installed or if the preferred untwine path fails validation, so this is not an error — just slower COPC conversion.
- To install for better performance: `conda install -c conda-forge untwine`
- Verify: `untwine --help`

#### "Memory allocation failed"
- Reduce `--tile-length` for smaller tiles
- Decrease `--workers` to limit concurrent memory usage
- Decrease `--chunk-size` if single-source tile distribution still uses too much memory
- Decrease `--num-spatial-chunks` to limit per-file subsampling workers
- Use `--resolution-1` and `--resolution-2` with larger values

#### "No space left on device" during `create_merged_file`
- Use an output directory on local scratch storage, not network storage
- Increase free scratch space; direct untwine COPC finalization can temporarily need many times the final COPC size
- Reduce simultaneous jobs writing to the same disk
- Keep `--num-spatial-chunks` enabled so SmartTile avoids one giant temporary merged LAZ

#### "CRS mismatch" or "Coordinates appear projected"
- Ensure all input files are in the same coordinate reference system
- Use projected CRS (e.g., UTM) not geographic (WGS84)

#### "No PredInstance attribute found"
- Verify segmentation output includes `PredInstance` dimension
- Check attribute names (case-sensitive): `PredInstance`, not `predinstance`

### Debugging

Enable verbose output:

```bash
python src/run.py --task merge --verbose ...
```

Check processing logs in `output_dir/logs/`:

```bash
ls -la output_dir/logs/
cat output_dir/logs/c00_r00_convert.log
```

### Performance Tuning

#### Optimize for SSD
```bash
# Use more workers when I/O is fast
python src/run.py --task tile --workers 16 --threads 10 --num-spatial-chunks 10 ...
```

#### Optimize for HDD
```bash
# Reduce parallel I/O
python src/run.py --task tile --workers 2 --threads 2 --num-spatial-chunks 2 ...
```

#### Monitor resource usage
```bash
# Run with htop in another terminal
htop -p $(pgrep -f "python src/run.py")
```

---

## Project Structure

```
3dtrees_smart_tile/
├── src/                                 # Python source code
│   ├── run.py                          # Main CLI orchestrator
│   ├── parameters.py                   # Parameter configuration (Pydantic)
│   ├── main_tile.py                    # Tiling pipeline
│   ├── main_subsample.py               # Subsampling pipeline
│   ├── main_remap.py                   # Prediction remapping
│   ├── main_create_merged_file.py      # Prod-merged product creation
│   ├── main_merge.py                   # Merge wrapper
│   ├── merge_tiles.py                  # Merge compatibility/core entry points
│   ├── merge_*.py                      # Merge internals
│   ├── tile_*.py                       # Tiling internals
│   ├── subsample_*.py                  # Subsampling internals
│   ├── copc_*.py                       # COPC metadata and staging helpers
│   ├── filter_buffer_instances.py      # Buffer zone filtering
│   ├── prepare_tile_jobs.py            # Tile job generation
│   ├── get_bounds_from_tindex.py       # Extent calculation
│   └── plot_tiles_and_copc.py          # Visualization
│
├── README.md                           # This documentation
├── CONTEXT.md                          # Agent/developer context and invariants
├── Dockerfile                          # Container configuration
├── tests/                              # Unit and integration tests
├── tool_appendix.txt                   # Paper/tool appendix table
└── .gitignore                          # Git ignore rules
```

### Module Descriptions

| Module | Purpose |
|--------|---------|
| `run.py` | CLI entry point, task routing, parameter handling |
| `parameters.py` | Pydantic-based parameter definitions with CLI support |
| `main_tile.py` | Two-phase tiling (distribute + COPC conversion), tindex creation |
| `main_subsample.py` | Parallel voxel-based subsampling |
| `main_remap.py` | KDTree-based prediction remapping |
| `main_create_merged_file.py` | Prod-merged product creation from Original-with-predictions files |
| `main_merge.py` | Merge task orchestration |
| `merge_tiles.py`, `merge_*.py` | Merge orchestration and internals |
| `tile_*.py`, `subsample_*.py` | Extracted tiling and subsampling helpers |
| `copc_*.py`, `point_cloud_*.py` | COPC staging, metadata preservation, and output helpers |
| `filter_buffer_instances.py` | Centroid-based buffer zone filtering |
| `prepare_tile_jobs.py` | Tile grid calculation and job list generation |
| `get_bounds_from_tindex.py` | Extent extraction from spatial index |
| `plot_tiles_and_copc.py` | Matplotlib visualization of tiles |

---

## Dependencies

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| Python | ≥3.10 | Runtime |
| PDAL | ≥2.5 | Point cloud processing, subsampling |
| untwine | Latest | Fast COPC conversion (auto-fallback to PDAL if unavailable) |
| laspy | Latest | LAZ/LAS file I/O |
| lazrs-python | Latest | LAZ compression |
| NumPy | Latest | Array operations |
| SciPy | Latest | KDTree spatial queries |
| pydantic | ≥2.0 | Parameter validation |
| pydantic-settings | Latest | CLI and env var support |

### Optional Dependencies

| Package | Purpose |
|---------|---------|
| matplotlib | Visualization |
| fiona | Vector file handling |
| pyproj | CRS transformations |
| geopandas | Geospatial operations |

### External Tools

| Tool | Purpose |
|------|---------|
| [untwine](https://github.com/hobuinc/untwine) | COPC conversion (preferred, auto-fallback to PDAL) |

---

## License

MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Contributing

Contributions are welcome! Please open an issue or pull request on the project repository.

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{3dtrees_smart_tile,
  title = {3DTrees Smart Tiling Pipeline},
  author = {},
  year = {2026},
  url = {https://github.com/your-org/3dtrees_smart_tile}
}
```

---

**Questions or issues?** Open an issue on GitHub or contact the maintainers.
