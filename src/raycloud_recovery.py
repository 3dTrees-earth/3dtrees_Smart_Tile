"""Admit whole RCT trees only when no surviving tree competes for their points."""
import numpy as np
from scipy.spatial import cKDTree

from bounded_point_index import distance_limit, spatial_batches
from dense_tile_merge import DUPLICATE_RADIUS


class RayCloudRecoveryGate:
    """Check full dense geometry against retained and previously admitted trees.

    Instance identity is (source tile, local ID). Rejected candidates do not
    supply ownership; admitting one makes it a blocker for later candidates.
    All reads use bounded spatial batches from the existing dense index.
    """

    def __init__(self, model, dense, owned, admitted, ownership, origin):
        self.dimension = model.instance
        self.dense, self.owned, self.admitted = dense, owned, admitted
        self.bounds = {(tile, row['instance']): np.asarray(row['bbox_xyz']) - origin
                       for tile, item in enumerate(ownership['tiles'])
                       for row in item['instances'] if not row['kept']}
        self.origin = origin
        self.blocked = []

    def _geometry(self, tile, uid):
        for _, data in self.dense.candidates(self.bounds[tile, uid], 0, tile=tile):
            mask = data['values'][self.dimension] == uid
            if np.any(mask):
                yield data['xyz'][mask]

    def _conflict(self, index, xyz):
        for group in spatial_batches(xyz):
            points = xyz[group]
            limit = distance_limit(points, DUPLICATE_RADIUS)
            for other_tile, data in index.candidates(points, DUPLICATE_RADIUS):
                labels = data['values'][self.dimension]
                positive = labels > 0
                if not np.any(positive):
                    continue
                distances, positions = index.query_tree(cKDTree(data['xyz'][positive]), points)
                hits = np.flatnonzero(distances <= limit)
                if len(hits):
                    first = hits[0]
                    return {'conflicting_tile': other_tile,
                            'conflicting_local_instance': int(labels[positive][positions[first]]),
                            'xyz': (points[first] + self.origin).tolist()}
        return None

    def __call__(self, tile, uid):
        # Check every point, including buffer tails and already-supported parts.
        # Do not publish any candidate geometry until its full check succeeds.
        for xyz in self._geometry(tile, uid):
            for index in (self.owned, self.admitted):
                conflict = self._conflict(index, xyz)
                if conflict is not None:
                    self.blocked.append({'tile': tile, 'local_instance': uid, **conflict})
                    return False
        offset = 0
        for xyz in self._geometry(tile, uid):
            self.admitted.add(tile, xyz, {self.dimension: np.full(len(xyz), uid, dtype=np.uint32)},
                              np.arange(offset, offset + len(xyz), dtype=np.int64))
            offset += len(xyz)
        self.admitted.flush()
        return True
