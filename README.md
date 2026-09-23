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
- **Explicit dimension policy** - intermediate COPC conversion strips extra dimensions by default, streaming standard fields to temporary LAS before Untwine when the input contains extras; prod-merged creation preserves enriched dimensions

### Smart Instance Merging
- **Core instance ownership** - removes whole non-owned instances using the selected dense anchor
- **Point-level buffer deduplication** - retains points of owned instances without a final, label-consistent survivor within 1 cm
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
| 1 | Dense transfer | Assign every 1 cm target point a prediction from its own model tile, within the explicit transfer radius (default 0.1732 m). |
| 2 | Core ownership | Remove whole instances whose dense centroid (or selected anchor) lies outside the core on a side with a declared neighbor. Keep points of owned instances, resolving shared claims by nearest claimant core below. |
| 3 | Instance reconciliation | Match retained local IDs across declared tile overlaps, independently for each model. |
| 4 | Point deduplication | Assign shared points of distinct trees to the retained claimant nearest its core; otherwise remove only label-consistent duplicates within 0.01 m XYZ. |
| 5 | Coverage validation | Require complete original-to-unfiltered coverage. Require complete original-to-final coverage for other models; unmatched RCT final labels become background 0 and are counted in the report. |
| 6 | Publication | Publish validated staged outputs. On failure retain diagnostics, never partial final products. |


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

Transfer predictions to the 1 cm target geometry first, remove non-owned instances,
reconcile retained instance IDs, then deduplicate cross-tile buffer points.
Transfer and unfiltered original coverage require 100% assignment. Final original
coverage also requires 100% for non-RCT models; unmatched RCT points become
background 0.
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

### RayCloudTools tree files

Pass the RCT `*_segmented.laz`, `*_trees.txt`, and `*_trees_info.txt` files together in
one segmented input folder. SmartTile detects `PredInstance_RCT` and requires both
text files for every tile. Use the matching 1 cm tiles and tile-bounds JSON:

```bash
python src/run.py --task merge \
    --subsampled-segmented-folder /path/to/rct_predictions \
    --subsampled-target-folder /path/to/subsampled_res1 \
    --tile-bounds-json /path/to/tile_bounds_tindex.json \
    --output-tiles-folder /path/to/out/output_tiles \
    --instance-dimension PredInstance_RCT --skip-merged-file
```

This mode transfers the RCT labels to the 1 cm tiles, removes whole trees whose
selected anchor is outside the owning core, and leaves every retained positive
instance ID unchanged. It does not reconcile, merge, or renumber RCT trees. The
filtered LAZ tiles are in `output_tiles/`; matching tree and treeinfo tables are
in `segmented_filtered/` beside that folder. Each retained row gets a leading
`predinstance` column equal to its original local ID, so gaps left by removed
trees do not change the ID-to-row relationship. `instance_metadata.csv`
records each retained `(tile, PredInstance_RCT)` pair. Missing or inconsistent
tree files fail before outputs are published. RCT IDs may repeat between tiles,
so keep the LAZ tiles and their named text tables together; a single merged LAZ
cannot identify the tree table for a repeated ID. During final original
remap, an RCT point with no surviving prediction inside the matching radius
receives `PredInstance_RCT=0` (and zero for any other RCT prediction fields).
The report records its unmatched count and coordinates. Unfiltered 1 cm
baseline coverage remains mandatory.

### Basic Filter Task

Deduplicate already-remapped 1 cm tiles using the same point-level contract.
Coarse predictions must enter through the merge task with a target folder first:

```bash
python src/run.py --task filter \
    --input-dir /path/to/segmented_remapped \
    --output-dir /path/to/filtered_tiles \
    --tile-bounds-json /path/to/tile_bounds_tindex.json \
    --instance-dimension PredInstance
```

The filter task writes regular LAZ tiles in deterministic order, preserves all
kept-point dimensions, and includes ownership decisions and an ID/source mapping
in the report. `--filter-anchor centroid` (default), `highest_point`, or
`lowest_point` selects instance ownership for both merge and filter. Non-owned
instances are removed entirely; retained instances keep their full crowns,
including unshared buffer points. Shared points claimed by distinct retained trees use the nearest claimant core’s instance and attributes; equal distances use stable source filename order. Background ID 0 uses point-wise core ownership and retains the owning tile’s semantics. Legacy output naming
options remain unused.

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
    --baseline-1cm-folders /path/to/unfiltered_1cm_tiles \
    --original-input-dir /path/to/original/files \
    --output-dir /path/to/original_with_predictions \
    --merged-resolutions res1,res2 \
    --merged-output-formats laz,copc.laz
```

LAS, LAZ and COPC prediction sources are streamed into a disk-backed spatial
index. Each query and stored batch contains at most 32,768 points. Enriched
originals retain their raw coordinate records, source fields and source header
metadata. No complete dense tile or merged cloud is loaded into one KDTree.

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
    --baseline-1cm-folders /path/to/unfiltered_1cm_tiles \
    --original-copc-input-dir /path/to/original_copc_files \
    --original-laz-input-dir /path/to/uploaded/raw_laz_files \
    --original-laz-output-dir /path/to/original_with_predictions_raw \
    --remap-dims PredInstance_SAT,PredSemantic_SAT,PredInstance_ForestMamba,species_id_sat,species_prob_sat,species_id_foma,species_prob_foma \
    --chunk-size 10000000 \
    --produce-merged-file \
    --merged-resolutions 1cm \
    --merged-output-formats laz
```

If `--remap-dims` is omitted, prediction extra dimensions from every collection
are transferred. Native LAS fields carried as ExtraBytes after point-format
conversion (for example SAT `scan_angle`) are excluded: the dense target and
original files remain authoritative for those values. Duplicate extra-dimension
names across prediction
collections fail early; SmartTile does not auto-rename them to `_2`, `_3`, or
add late model suffixes during final remap.

Final remap requires one shared unfiltered 1 cm baseline folder, or one baseline
folder per collection in the same order. For locally preserved merge outputs,
`smarttile_merge.json` resolves the sibling baseline folder automatically. Galaxy
wrappers must pass `--baseline-1cm-folders` explicitly when packaging only the LAZ
collection and dropping its manifest. The merge manifest records the first-stage resolution. Standalone remap reads it
when available; older manifests default to 1 cm. `--resolution-1` can specify a
finer or coarser first-stage resolution, and `--remap-tolerance` can explicitly
override the derived final radius. Baseline coverage still requires 100%, and a
`--min-remap-match-fraction 0.99` request is rejected.

Every original file and model has separate baseline and final coverage metrics.
For non-RCT models, final coverage requires **100% within the 3D first-stage
voxel diagonal** (17.32 mm at 1 cm resolution) in Euclidean XYZ. Missing
coverage fails the entire remap before any enriched original is published.
For RCT, final misses within that same search radius become prediction value 0;
`background_assigned_points` and per-file final metrics record their count.

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

For a larger synthetic remap-first check, install the optional `psutil` package
and run the following with a new output directory:

```bash
python benchmarks/remap_first_smoke.py --output-root out/remap-first-smoke
```

This generates three overlapping 1 cm tiles (300,000 points) and 200,000
original points, then checks dense transfer, reconciliation, survivor-only
deduplication, strict baseline/final coverage, and unchanged source fields.
Inputs, coverage/checksum reports, timing, and sampled peak RSS remain in the
output directory. This synthetic check does not replace the exact dataset
3110/3111 replays required by 3DT-2101 or validate the Linux container/toolchain.

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

All part files for each tile are merged and converted to COPC format. Untwine is preferred for COPC writing. Intermediate COPC conversion streams native standard fields into a temporary LAS in 32,768-point batches when ExtraBytes are present, then runs Untwine and validates the output schema and CRS. If Untwine fails, CRS validation fails, or extra dimensions remain, SmartTile falls back to PDAL `writers.copc` without `extra_dims=all`. Prod-merged creation is the explicit exception and preserves enriched dimensions for final products.

The SmartTile Docker image has been validated against Untwine 1.5.1: `--dims ""` is rejected; `--dims Classification` can leave prediction or source extras on enriched files; and long explicit standard-dimension lists such as `--dims Red,Green,Blue` are not used because they can crash this build. SmartTile therefore strips unwanted ExtraBytes before conversion, keeps all standard fields and retained metadata, and omits `--dims` on the reduced input. This avoids converting the complete enriched cloud twice when dimension limiting fails. Temporary staging is removed when conversion finishes.

**Parallelization**: Multiple tiles converted concurrently (controlled by `--workers`).

**Output**: `c{col}_r{row}.copc.laz` files in `tiles_{tile_length}m/`.

**Options**:
- **Default**: Stream away unwanted ExtraBytes when present, run Untwine on the standard-field input, and validate the output schema/CRS. PDAL remains a fallback for converter or validation failure.
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

### The Merge Process in Detail (v2.4 / 3DT-2101)

Each model collection is processed independently. Do not combine SAT and
ForestMamba into one instance space. When merging several collections, output
subdirectories `model_000`, `model_001`, etc. preserve their separate geometry.
Use model-specific prediction dimension names before the final original remap.

1. **Transfer before filtering.** Match source/target tiles through the tile
   layout and require a source for every target. For each 1 cm point, copy the
   nearest prediction from its own coarse tile within
   `--prediction-transfer-tolerance` (default **0.1732 m XYZ**). Empty or distant
   prediction coverage fails. Zero is a valid model background label; a missing
   match or declared label no-data value is never accepted as background.
2. **Filter instance ownership on the dense tiles.** Compute each positive
   instance's `--filter-anchor` (`centroid`, `highest_point`, or `lowest_point`)
   from its remapped 1 cm points. Remove the entire instance if its anchor lies
   outside the declared core on a side with a declared neighbor, even when that
   neighbor is absent from a partial replay. Core boundaries are inclusive;
   dataset exterior edges are retained. Older layouts without explicit cores
   derive them from bounds and `tile_buffer`. A single-file bypass with no
   declared grid owns its entire extent. Keep all points of owned instances
   with their owning tile’s per-point semantic attributes. Background ID 0 has
   no instance anchor: retain it only in its spatial core, with east/north
   neighbors owning shared upper edges. Retain dataset exterior background.
   Preserve full-instance anchors, bounds, and decisions before filtering.
   Keep the unfiltered dense geometry separately for coverage validation.
   Before reconciliation, scan rejected positive instances in declared core
   overlaps. Where no retained tree point supports a positive 1 cm sample in
   XYZ, admit the whole rejected instance with the most newly covered distinct
   core samples. Recheck coverage after each admission; equal scores use stable
   source filename and local ID order. Background never counts as tree support.
   Other rejected instances do not participate in matching or conflicts.
   `orphan_recovery` reports candidates, admissions and the final support check.
3. **Reconcile retained IDs and select their owners.** Within declared tile overlaps,
   use 0.05 m correspondences and mutual-best instance pairs meeting the configured
   overlap threshold (default 30% of the smaller instance). Group matches
   deterministically; a group cannot contain different instances from one tile.
   Ambiguous pairs stay distinct. Recovery cannot bridge two groups that were
   distinct among normally retained instances; rejected bridges are reported.
   For a group with recovered members, select
   the member that supplies the most newly covered core samples; otherwise
   select the first retained tile in stable source-filename order. Remove
   secondary copies. If this loses required recovered geometry, fail explicitly.
   Preserve the owner’s
   per-point semantic values and other attributes without voting or blending.
   Record owners and removals in `semantic_ownership`. Labels use uint16,
   promoted to uint32 when needed.
4. **Assign shared points, then deduplicate.** Within declared buffered tile
   overlaps, resolve points claimed by distinct positive instances using the
   nearest claimant core: among retained tree claims within **0.01 m XYZ**,
   choose the tile with the smallest XY distance to its closed core rectangle.
   Break equal-distance ties by stable source filename order, including shared
   tree-core edges. A background-only tile or removed instance cannot win, even
   when its core contains the disputed point. Preserve the winner's instance ID,
   per-point semantics and other attributes. Nearby distinct points across a
   nearest-core bisector can retain their respective owners. Unshared buffer
   tails remain intact; this rule does not merge instance IDs or crop whole
   crowns to the core.
   Decisions examine both earlier and later tiles and are recorded in
   `shared_point_ownership`. Final original coverage remains mandatory.
   Next, a retained tree wins over a neighboring background point within
   **0.01 m XYZ**, keeping the tree owner's per-point semantic prediction and
   other attributes. Use the surviving tree index after tree/tree resolution;
   only retained or explicitly recovered tree instances can override background.
   Record these removals
   in `tree_background_ownership`. Unshared background remains unchanged.
   Adjacent points in separate owning cores may retain different tree labels
   or background semantics within 1 cm. Other unresolved label conflicts still
   fail, including a non-nearest conflicting neighbor. Finally, process tiles in stable source-filename order and remove
   label-consistent cross-tile duplicates only with an actual final survivor
   within 0.01 m. Same-tile points never delete one another.
5. **Measure original coverage directly.** Check every uploaded original against
   both the unfiltered 1 cm geometry and final survivors, independently for every
   model. Unfiltered 1 cm coverage requires 100% within the first-stage
   voxel diagonal in XYZ (17.32 mm for 1 cm voxels); so does final coverage
   for non-RCT models. A point and its voxel center of mass can be separated
   by up to that distance. RCT final points without a surviving prediction
   inside the radius receive background 0, with counts in the coverage report.
   Orphan recovery never invents points or fills missing predictions.
6. **Publish after validation.** Stage all model and enriched-original outputs
   privately. A failed model/file discards the staged products and preserves a
   JSON diagnostic report. Existing nonempty outputs are refused rather than
   mixed with a new run. The processed merged file is an intermediate; final
   prod-merged products still derive from the enriched raw originals.

Distance comparisons are inclusive and use an explicit eight-ULP float64
roundoff guard after removing the common coordinate offset. This numerical guard
is shared by transfer, deduplication and coverage; it is not a configurable
spatial relaxation. Nearest ties choose the earlier tile and then source point.

The dense implementation is sequential and disk-backed. A fixed 32,768-point
limit applies to stored/query batches, while small instance correspondence maps
remain in memory. SQLite RTree indexes batch bounds; exact distances use each
batch's float64 XYZ. Batch payloads are fixed binary records viewed directly
from SQLite blobs, avoiding ZIP/NPY parsing on every query. Scratch space holds unfiltered tiles, final staged tiles and
spatial indexes. Production-scale runtime and disk use need corpus benchmarking.

`remap_first_report.json` contains source-to-tile mappings, transfer counts,
reconciled IDs, duplicate/conflict counts, sample survivor references,
per-original/per-model coverage and distance histograms, checksums, and CPU/wall
time. `output_tiles_unfiltered_1cm/` preserves baseline geometry.
`smarttile_merge.json` points to that baseline for a later strict remap.

Strict merge uses contract `3DT-2183/v7-orphan-recovery` (the standalone
filter still uses normal core ownership). Full-instance anchor positions,
retained/removed IDs, source and point counts are recorded in
`instance_ownership`; admissions and post-dedup support are recorded in
`orphan_recovery`. The final original coverage gate remains 100% within the
first-stage voxel diagonal for non-RCT models; unmatched RCT final points
become background 0 and are counted in the report. Small-cluster reassignment remains disabled;
`--pre-remap-reassign-instances` is rejected because it would change the
reconciled-label contract. Tree-instance ID changes are recorded in
the reconciliation map; auxiliary tree text tables are not rewritten.

### Single-file tiling bypass metadata

When the size threshold bypasses tiling, the bounds JSON describes one cloud
with its actual extent and zero internal buffer. Subsampling updates that one
entry using the output header, including filenames without grid labels.

Merge/filter can also recover historical bypass outputs whose JSON still contains
an unused multi-tile plan. Recovery requires exactly one cloud matching the full
`proj_extent` within 1 cm, unchanged `bounds`/`planned_bounds`, and an extent that
cannot fit inside any individual planned tile. It uses the actual cloud bounds as
one core with no internal neighbors and records `layout_recovery` in the ownership
report. Each model output includes the effective `tile_bounds.json`, referenced by
filename and SHA-256 in `smarttile_merge.json`. A recovered layout also records
the original JSON checksum. The input JSON is preserved. Partial tile collections retain their declared
neighbors; unrelated or already-updated layouts are not recovered this way.

## Parameters Reference

### Tile Task Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--tile-length` | 100 | Tile size in meters |
| `--tile-buffer` | 20 | Buffer overlap in meters |
| `--threads` | 10 | Threads per COPC writer |
| `--workers` | 2 | Tile file workers; CPU-capped query threads for merge/filter and batch processes for final remap |
| `--num-spatial-chunks` | `--workers` | Per-file subsampling parallelism for COM windows or stripe chunks |
| `--resolution-1` | 0.01 | First subsampling resolution (1cm) |
| `--resolution-2` | 0.1 | Second subsampling resolution (10cm) |
| `--output-copc-res1` | True | Write first-resolution subsampled outputs as COPC LAZ (`*.copc.laz`) |
| `--output-copc-res2` | False | Write second-resolution subsampled outputs as COPC LAZ; default keeps 10cm as regular LAZ |
| `--subsampling-method` | center-of-mass | Subsampling method: `center-of-mass` or `nearest-to-centroid` |
| `--chunk-size` | 20000000 | Points per chunk when reading LAZ/LAS in tiling Phase 1, multi-collection remap, and merged-COPC-to-original remap (smaller = less peak RAM; larger = fewer scans) |
| `--tiling-threshold` | None | File size threshold in MB for skipping tiling on single small files |

### Merge, Filter and Remap Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--tile-bounds-json` | Required for merge/filter | Declared buffered tile layout |
| `--subsampled-target-folder` | Sibling subsampled_res1 | 1 cm target geometry for coarse predictions |
| `--prediction-transfer-tolerance` | 0.1732 | Earlier model-to-dense transfer radius in XYZ metres |
| `--overlap-threshold` | 0.3 | Required instance correspondence fraction |
| `--disable-matching` | False | Leave IDs tile-local; shared tree points use nearest claimant core ownership |
| `--baseline-1cm-folders` | Merge manifest | Shared baseline or folders in model order for final remap |
| `--remap-tolerance` | Auto: sqrt(3) × `--resolution-1` | Optional final original coverage radius override in XYZ metres |
| `--min-remap-match-fraction` | 1.0, fixed | Required coverage for every original and model |
| `--original-laz-input-dir` | None | Uploaded raw originals; legacy alias: original-input-dir |
| `--original-laz-output-dir` | Derived | Enriched raw originals |
| `--original-copc-input-dir` | None | Optional twin validation, not product geometry |
| `--skip-merged-file` | False | Omit the processed intermediate; use for multiple models |
| `--merged-resolutions` | res1,res2 | Product resolutions after strict original validation |
| `--merged-output-formats` | copc.laz | Final product formats |

The dense merge/remap path uses fixed memory batches and one processing worker.
The worker settings below continue to apply to tiling, subsampling and
prod-merged generation, not the new disk-backed merge loop.

### Understanding `--workers`, `--threads`, and `--num-spatial-chunks`

These parameters control different aspects of parallelism:

#### `--workers` (Global Parallelism)

Controls file concurrency for tiling, native queries for merge/filter, and batch processes for final remap:

| Task | What `--workers` Controls |
|------|---------------------------|
| **Tile Task** | Parallel source-file distribution; one large source is split into point ranges; tile COPC finalization runs in parallel |
| **Merge / Filter Tasks** | Parallel KDTree queries within each bounded batch; model/tile order and index writes remain serial |
| **Remap Task** | Independent query batches in worker processes, each with read-only index connections and one query thread; ordered output writer |

For example, `--workers 4` requests up to four query threads for merge/filter or
four processes for final remap.
The effective limit is capped by `GALAXY_SLOTS`, CPU affinity and detected cgroup
CPU quotas, and is recorded in the report's `parallelism` field. Queries use at
most one thread per 1,024 query points to avoid thread startup overhead on tiny
spatial cells. A Docker container capped at one CPU will therefore use one query
thread even when more workers are requested.

**Memory impact**: Merge/filter share batches and KDTrees between query threads.
Final remap workers share completed disk indexes through separate read-only
SQLite connections; each worker has its own bounded query arrays and SQLite
cache (16 MiB per index), plus Python/library memory. The parent admits at most
two 32,768-point batches per process and writes results in original order.
Inputs with only one full batch remain serial; larger inputs require at least
one full batch per process. `parallelism.enrichment_processes` records the actual
process budget, and `timings` separates indexing from enrichment.
Index construction and LAS/LAZ I/O remain serial, so total speedup depends on how
much time the dataset spends querying. `--num-spatial-chunks` does not change
this final-remap process budget.

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

### Vector ExtraBytes in COPC

COPC conversion scalarizes vector ExtraBytes automatically and retains their
logical descriptors in an embedded schema and JSON sidecar. Source EVLRs are
retained during scalarization. Generated field names avoid existing scalar
names regardless of descriptor order; component descriptions fit the LAS byte
limit while the schema retains the complete original description.

Before publishing a vector COPC, SmartTile validates every component's storage
type, scale, offset, and no-data descriptor, then compares all raw component
values together with their XYZ coordinates. Point reordering is allowed;
changed values, missing fields, and changed duplicate counts fail. Validation
uses at most 32,768 points per batch and an exact SQLite multiset on the output
filesystem, so it adds two streaming reads and disk space proportional to the
number of distinct XYZ/vector records. Rejected outputs and sidecars remain
private and are removed. Writer versions that normalize vector component types
or descriptors are rejected until they satisfy this contract.

### Point Cloud Attributes

#### Tile Task Output
- `X`, `Y`, `Z`: 3D coordinates
- Intermediate COPC conversion strips ExtraBytes in bounded batches before Untwine, then validates that none remain.
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
`ghcr.io/3dtrees-earth/3dtrees_smart_tile:v2.2`.

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
│   ├── filter_buffer_instances.py      # Legacy standalone buffer filtering
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
| `dense_instance_ownership.py` | Dense instance ownership by centroid, highest point, or lowest point |
| `filter_buffer_instances.py` | Legacy standalone centroid filter |
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

COPC subsampling window bounds clamp the last grid edge to the actual LAS maximum.
A non-advancing floating-point step raises an error rather than repeatedly appending
windows (the dataset 3083 memory failure).
