# SmartTile

SmartTile is the 3DTrees point-cloud tiling and reconciliation pipeline. It
splits large LiDAR uploads into buffered tiles for tree-segmentation models,
reconciles the per-tile predictions into dataset-wide instances, transfers them
back onto every original point and builds merged download products.

| Task | Purpose |
| --- | --- |
| `tile` | Plan a buffered tile grid and write subsampled (or original-resolution) tiles in one read of every source point. |
| `merge` | Transfer model predictions to dense tile geometry, keep each instance in the tile that owns it, reconcile IDs across tiles and remove duplicate buffer points. |
| `remap` | Copy prediction dimensions from processed collections onto the uploaded LAS/LAZ originals (`original_with_predictions`). |
| `create_merged_file` | Combine originals (enriched or not) into `prod_merged_*` products at selected resolutions and formats. |

Segmentation, species inference, visualization and ForestStructure are
separate tools; the Galaxy wrapper runs these same `run.py` tasks. The usual
order is `tile` → segmentation → `merge` → species → `remap` →
`create_merged_file`; [docs/task-workflow.md](docs/task-workflow.md) shows the
artifacts and matching distances. `python src/run.py --help` lists
every flag (unknown flags are rejected); `--show-params` prints resolved values.

## Installation

The container runtime is the explicit conda lock (Python 3.12, PDAL, GDAL,
Untwine, laspy/lazrs, NumPy, SciPy, pydantic):

```bash
mamba create -p ./smarttile-env --file conda-linux-64.lock
mamba activate ./smarttile-env   # PDAL and Untwine must be on PATH
python src/run.py --help
```

`environment.yml` is the unpinned development specification.

## Input requirements

- LAS, LAZ or COPC (`.las`, `.laz`, `.copc.laz`, case-insensitive) in one flat
  directory. A COPC is preferred over a LAS/LAZ with the same name.
- One projected CRS in metres shared by all files, or no CRS at all (local
  metres). Tile sizes, buffers, voxel sizes and radii are metres, so tiling
  rejects geographic (degrees), geocentric and non-metre CRSs, vertical units
  other than metres, unreadable CRS records, files with different CRSs, and a
  mix of files with and without a CRS. WKT stored in EVLRs is read.
- No empty files, and unique names ignoring case and extension (`Plot1.las` and
  `plot1.laz` fail rather than one being ignored).
- Header bounds must cover the points (the grid is planned from headers).
- Instance labels: `0` is background, positive values are instances, negative
  values are invalid.

## Tasks

Every task writes into a new destination (existing non-empty outputs are
refused) and publishes only after all checks pass, so a failed run leaves no
partial products.

### tile

```bash
python src/run.py --task tile \
    --input-dir /data/upload --output-dir /data/tiles \
    --tile-length 100 --tile-buffer 20 \
    --tile-outputs 0.01:laz,0.1:laz
```

`--tile-outputs` lists every output as `resolution:format` (`laz`, `copc.laz`,
`ply`; `original` keeps every point). Output *i* goes to `subsampled_res{i}/`.
Without it, SmartTile writes 1 cm COPC (`--resolution-1`, `--output-copc-res1`)
and 10 cm LAZ (`--resolution-2`, `--output-copc-res2`). Voxels keep the exact
mean of their points (`--subsampling-method center-of-mass`, XYZ only) or the
original point nearest that mean (`nearest-to-centroid`). Every resolution is
computed from the original points; results do not depend on `--workers`.

```
tiles/
├── subsampled_res1/c00_r00_subsampled_1cm.laz   # one file per occupied tile
├── subsampled_res2/c00_r00_subsampled_10cm.laz
├── tile_bounds_tindex.json   # cores, buffers, neighbours: required by merge
├── tile_jobs_100m.txt        # job list
├── tindex_100m.gpkg          # source index (tindex.gpkg links to it)
└── overview_copc_tiles.png
```

Tiles whose core holds no source point are dropped (see
`core_occupancy.skipped_tiles`). A single input below `--tiling-threshold` MB
becomes one whole-cloud tile without buffer; `--grid-origin-x/-y` fix the grid.

**Instance tiles.** For inputs that already carry predictions,
`--instance-tiles PredInstance_SAT,PredInstance_FM --instance-outputs original:copc.laz,0.1:laz`
writes each instance whole into the tile whose core holds its XY centroid, so
per-instance tools see every tree once (first dimension in `subsampled_res*/`,
later ones in `instance_dim2/subsampled_res*/`, …). `--instance-tile-dims` adds
attributes. Instance IDs must be unique across files
(`--allow-reused-instance-ids` skips that check).

### merge

Coarse predictions onto the dense tiles of the same tile run:

```bash
python src/run.py --task merge \
    --subsampled-segmented-folder /data/sat_10cm_predictions \
    --subsampled-target-folder /data/tiles/subsampled_res1 \
    --tile-bounds-json /data/tiles/tile_bounds_tindex.json \
    --output-folder /data/merge_sat
```

Tiles that already hold dense predictions are only filtered (no transfer):

```bash
python src/run.py --task merge \
    --segmented-remapped-folder /data/dense_predictions \
    --tile-bounds-json /data/tiles/tile_bounds_tindex.json \
    --output-dir /data/filtered
```

Pass exactly one of `--subsampled-segmented-folder` (alias
`--subsampled-10cm-folder`), `--segmented-remapped-folder` or
`--segmented-folders a,b` (several models; one `model_000/`, `model_001/`, …
each; add `--subsampled-target-folder` to transfer). The target defaults to
`subsampled_res1/` beside the segmented folder. For coarse-only filtering use
`--segmented-remapped-folder` with `--resolution-1` set to the input spacing
(e.g. `0.1`); the inputs must keep the coordinates of the same tile run.

Per model, merge transfers each tile's predictions to its dense target within
`--prediction-transfer-tolerance` (0.1732 m; every target point must match),
removes whole instances whose `--filter-anchor` (`centroid`, `highest_point`,
`lowest_point`) lies outside the core towards a declared neighbour, recovers
rejected instances that supply uncovered core geometry, reconciles instances
across overlaps (5 cm correspondences, `--overlap-threshold` 0.3;
`--disable-matching` turns it off), gives shared points to the nearest core,
removes label-consistent duplicates within 1 cm, optionally reassigns small
instances (`--reassign-small-instances`) and moves every instance whole into one
output tile ([decision diagrams](docs/filter-decision-flow.md)). With
`--original-laz-input-dir` it also enriches the originals like `remap`.

**RayCloudTools.** Put each tile's `*_trees.txt` and `*_trees_info.txt` (and
optional `*_trees_mesh.ply`) beside its LAZ and pass
`--instance-dimension PredInstance_RCT --skip-merged-file`. RCT trees are not
reconciled or deduplicated; retained IDs become `tile_id * 100000 + local_id`,
and the tables and meshes are filtered to the same IDs. See
[docs/raycloud-filter-only.md](docs/raycloud-filter-only.md).

```
work_root/                         # --output-folder / --output-dir, else parent of the input
├── output_tiles/                  # c00_r00.laz, …: every instance ID in exactly one tile
│   ├── instance_metadata.csv
│   ├── PredInstance_summary.json  # <instance dimension>_summary.json
│   ├── tile_bounds.json           # effective layout (checksummed in the manifest)
│   └── smarttile_merge.json       # manifest: baseline path, first-stage resolution
├── output_tiles_unfiltered_1cm/   # dense baseline (omit with --skip-unfiltered-tiles)
├── segmented_filtered/            # RCT tree tables and meshes
├── merged_tiles.laz               # single non-RCT collection, unless --skip-merged-file
├── merged_tiles_instance_metadata.csv
├── original_with_predictions/     # with --original-laz-input-dir
└── remap_first_report.json        # decisions and timings; also written on failure
```

### remap

```bash
python src/run.py --task remap \
    --segmented-folders /data/merge_sat/output_tiles,/data/species_tiles \
    --original-laz-input-dir /data/upload \
    --original-laz-output-dir /data/original_with_predictions \
    --remap-dims PredInstance_SAT,PredSemantic_SAT,species_id,species_prob
```

Use `--segmented-folders` (comma-separated collections) or `--merged-laz` (one
file). Every extra dimension is transferred unless `--remap-dims` restricts it;
names must already be unique across collections (duplicates fail, nothing is
renamed). Each original point takes the values of the nearest prediction within
√3 × `--resolution-1` (17.32 mm at 1 cm; the merge manifest supplies the
resolution, `--remap-tolerance` overrides the radius). Non-RCT models need 100%
coverage or nothing is published; unmatched RCT points become 0.
`--baseline-1cm-folders` (or a preserved merge manifest) adds a baseline check
that tells tiling gaps from filtering gaps.

`--match-instances` (originals already carry each collection's instance field)
takes values only from the point's own instance and keeps its ID.
`--prediction-resolution-original` matches original-resolution predictions
within LAS encoding precision. The uploaded LAS/LAZ files stay the metadata
source; `--original-copc-input-dir` only validates COPC twins, and a COPC without
its LAS/LAZ source fails.

The output folder (`--original-laz-output-dir`, else `--output-dir`, else
`original_with_predictions/` beside the input) receives one enriched file per
original and `instance_mapping.json` (final IDs compacted to 1..N per model across
the dataset; RCT adds `rct_instance_mapping.json` and per-original tree tables).
`<folder>_coverage.json` is written beside it, and `--produce-merged-file` puts
products in its parent.

### create_merged_file

```bash
python src/run.py --task create_merged_file \
    --original-with-predictions-dir /data/original_with_predictions \
    --output-dir /data/products \
    --merged-resolutions full,10cm --merged-output-formats copc.laz,laz
```

`--merged-resolutions` takes `res1`, `res2`, metres (`0.05`), labels (`1cm`) or
`full` (default `res1,res2`); `--merged-output-formats` takes `laz`, `copc.laz`,
`ply` (default `copc.laz`).

```
products/
├── merged.copc.laz                 # full: every point, in sorted-file record order
├── prod_merged_10cm.copc.laz       # subsampled: one real point per voxel
├── prod_merged_10cm.laz
└── original_with_predictions_copc/ # staged COPCs reused by later runs
```

- `full` adds `OriginalPointIndex` (uint64 first ExtraByte; uint32 in PLY) so
  ForestStructure can restore source order (`--no-original-point-index` disables it).
- Subsampled products keep the original point nearest each voxel mean on an
  absolute grid (GPS-time order); they do not depend on `--workers`.
- Inputs must share point format, ExtraBytes and CRS; products use the finest
  input scale. Instance IDs must already be unique across files (true for
  `merge`/`remap` outputs); nothing is renumbered.
- Comma-separated `--original-with-predictions-dir` folders with the same
  originals are joined by record order first (shared attributes must agree).
- `--staged-copc-dir` reuses COPCs only when the staging manifest records their
  source; `--standardization-json` checks that expected dimensions survive.

## Key parameters

| Flag | Default | Notes |
| --- | --- | --- |
| `--workers` | allocated CPUs | See [CPU budget](#cpu-budget) |
| `--tile-length` / `--tile-buffer` | 100 / 20 m | Core size and buffer per side |
| `--tile-outputs` | 1 cm COPC, 10 cm LAZ | `res:format,…`; replaces `--resolution-1/2`, `--output-copc-res1/2` |
| `--resolution-1` | 0.01 m | Also sets the remap radius (√3 × value) |
| `--tile-bounds-json` | required for merge | `tile_bounds_tindex.json` of the same tile run |
| `--instance-dimension` | `PredInstance` | Fallback `treeID`; `PredInstance_RCT` for RayCloudTools |
| `--reassign-small-instances` | off | With `--max-cluster-size` 3000, `--max-volume-for-merge` 4 m³; not for RCT |
| `--skip-merged-file` | off | No `merged_tiles.laz`; required for RCT |
| `--skip-unfiltered-tiles` | off | Do not publish the dense baseline |
| `--remap-tolerance` | √3 × `--resolution-1` | Final coverage radius override |
| `--produce-merged-file` | off | Products after enrichment (merge, remap) |
| `--merged-resolutions` | `res1,res2` | `full` = every point |
| `--merged-output-formats` | `copc.laz` | `laz`, `copc.laz`, `ply` |

## CPU budget

`--workers` (alias `--number-of-threads`) is the only CPU setting. It defaults
to the allocated CPUs, the minimum of `GALAXY_SLOTS`, CPU affinity and the
cgroup CPU quota (e.g. `docker --cpus`), never the host core count, and larger
requests are capped to it. It sets the tiling read/reduce processes (at most two
outputs are converted to COPC at once), merge per-tile and query processes,
remap enrichment processes and product chunks. Outputs are identical for any
value; the report's `parallelism` field records the effective budget.

## Scratch disk and memory

- Put the output on local SSD scratch. Tiling spills about 16 B per source point
  (plus buffer overlap) and about 20 B per output point; enrichment spills 16 B
  per original point. Merge keeps about two stages on disk at a time. COPC
  finalization of products can need several times the final COPC size.
- Merge and remap use disk-backed spatial indexes in 32,768-point batches; no
  complete cloud is loaded into one tree. Product chunks average 2 M points, so
  memory does not grow with the dataset, but very dense areas can exceed that.
- `SMARTTILE_SPATIAL_CACHE_MB` sets the nearest-query cache (default 512 MiB,
  reduced to the cgroup memory limit; `0` disables it).
  `SMARTTILE_KEEP_FAILED_CHUNKS=1` keeps failed product chunks for diagnosis.

## Metadata contract

- Enriched originals keep their header, CRS records, scales, offsets, point
  format and other dimensions; only predictions are added.
- LAS 1.4/COPC outputs carry exactly one standardized WKT CRS record, so a LAZ
  derived from COPC is not byte-identical to the enriched original.
- ExtraBytes `no_data` declarations survive re-encoding; ambiguous ones fail.
- Vector ExtraBytes (2–3 elements) are scalarized for COPC; the schema is kept in
  a `3DTrees` VLR/EVLR and a `*.vector-extra-bytes.json` sidecar, and every
  component value is verified before publication.
- PLY (binary) has no VLRs: the CRS goes into a `comment crs: EPSG:<code>` (or
  WKT) header line, which QGIS reads; attribute names keep their case.

## Docker

The image has no `ENTRYPOINT`; pass the full command. Outputs must be new or
empty directories, so mount a parent and name a subdirectory:

```bash
docker build -t smarttile .
docker run --rm --user "$(id -u):$(id -g)" --cpus=8 --memory=32g \
    -v /data/upload:/in:ro -v /data/work:/work \
    smarttile python -u /src/run.py --task tile --input-dir /in --output-dir /work/tiles
```

## Tests

In the locked environment, from the repository root (CI runs the same suite).
The lock has no pip, so install the test tools beside it:

```bash
python3 -m pip install --target .pytest-deps pytest==9.1.1 pytest-subtests==0.15.0
PYTHONPATH=src:.pytest-deps python -m pytest -q -p no:cacheprovider tests
```

## Troubleshooting

| Message | Fix |
| --- | --- |
| `No LAZ/LAS files found` | Check `--input-dir` and the file extensions. |
| `… and … name the same source; rename one of them` | File names differ only in case or extension; rename one. |
| `Input file(s) without points` | Remove the empty files. |
| `Unsupported input CRS`, `different CRSs`, `have no CRS`, `Unreadable CRS records` | Reproject all files to one projected CRS in metres (e.g. the UTM zone). |
| `source points lie outside every planned tile core` | Header bounds are stale; rewrite the files (e.g. `pdal translate`). |
| `Refusing to mix this run with existing output`, `Refusing to replace existing output` | Use a new output path. |
| `COPC-only originals cannot be enriched` | Provide the uploaded LAS/LAZ for every COPC. |
| Coverage failure in merge or remap | See `remap_first_report.json` or `<folder>_coverage.json` (missing examples per file and model); check that predictions and originals come from the same tile run. |
| `No space left on device` | Use larger local scratch; reduce concurrent jobs on that disk. |

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

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{3dtrees_smart_tile,
  title = {3DTrees Smart Tiling Pipeline},
  author = {},
  year = {2026},
  url = {https://github.com/3dTrees-earth/3dtrees_Smart_Tile}
}
```
