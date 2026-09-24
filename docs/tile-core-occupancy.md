# Skip tiles with empty cores

The tile task checks original source XYZ coordinates before distributing points
or writing COPC tiles. A tile is created only if at least one source point lies
inside its core, including its boundary. Points in the buffer alone do not
qualify. All input files contribute; a later file or chunk can populate a core.
The single-cloud tiling bypass is unchanged.

The check uses bounded chunks (at most one million points) and stops checking a
core after its first hit. It stops reading when all cores have points. Sparse
layouts can require one additional complete source scan; this avoids writing,
subsampling, and running segmentation on tiles with only buffer points.

Retained tiles still contain their full buffers. Original grid column/row
identities and core bounds are preserved. `tile_bounds_tindex.json` lists only
retained tiles in `tiles`, with rejected cores documented separately under
`core_occupancy.skipped_tiles`. The tile job list and overview use this retained
layout. Neighbor discovery therefore does not treat an omitted core as a tile.

The rule is geometric, not semantic: a core containing only ground points still
qualifies. Shared core-boundary points count in both adjacent cores, consistent
with the existing inclusive tree-anchor convention.

Use a fresh output directory when migrating an existing run that contains an
empty-core tile. The task rejects such stale outputs instead of deleting them
or allowing subsequent directory-based subsampling to reuse them. No existing
test outputs are migrated automatically.
