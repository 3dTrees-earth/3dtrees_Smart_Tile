# Skip tiles with empty cores

The tile task publishes a tile only if at least one original source point lies
inside its core, including its boundary. Points in the buffer alone do not
qualify. All input files contribute; a later file or point range can populate a
core. The single-cloud tiling bypass is unchanged.

Occupancy is counted in the same single read that writes the subsampled tiles
(`tile_centroids.write_tile_centroids`): `plan_tiling(..., defer_occupancy=True)`
keeps every planned grid tile, each read worker counts the source points inside
every core, and the spilled blocks of tiles with an empty core are discarded
before reduction. No separate occupancy scan is needed, and tiles with
only buffer points get no output and no segmentation job.
`tile_core_occupancy.publish_occupied_tiles` then rewrites the layout and job
list.

Retained tiles still contain their full buffers. Original grid column/row
identities and core bounds are preserved. `tile_bounds_tindex.json` lists only
retained tiles in `tiles`, with rejected cores documented separately under
`core_occupancy.skipped_tiles`. The tile job list and overview use this retained
layout. Neighbor discovery therefore does not treat an omitted core as a tile.

The rule is geometric, not semantic: a core containing only ground points still
qualifies. Shared core-boundary points count in both adjacent cores, consistent
with the existing inclusive tree-anchor convention.

Use a fresh output directory for every run. The single read publishes its
outputs only after every tile passed its checks, but it does not remove files
from an older layout already present in `subsampled_res*/`.
