"""3DT-1899: distance-bounded, semantic, two-way cross-tile matching; genuine trees still merge."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bounded_point_index import PointIndex  # noqa: E402
from merge_stages import MatchLimits, describe_model, index_file, reconcile_instances  # noqa: E402
from test_merge_stages import write_cloud  # noqa: E402


def cluster(center, n=40, spread=0.4, seed=0):
    rng = np.random.default_rng(seed)
    return np.asarray(center, float) + rng.uniform(-spread, spread, (n, 3))


def reconcile(tmp_path, tiles, limits=MatchLimits(), overlaps=None):
    """tiles: per tile a list of (points, instance id, semantic); returns ({(tile, id): group}, report)."""
    source = tmp_path / "source"
    source.mkdir(parents=True)
    files, counts = [], {}
    for t, instances in enumerate(tiles):
        xyz = np.concatenate([p for p, _, _ in instances])
        ids = np.concatenate([[i] * len(p) for p, i, _ in instances])
        sem = np.concatenate([[s] * len(p) for p, _, s in instances])
        files.append(write_cloud(source / f"{t}.las", xyz[:, 0], ids, sem, ys=xyz[:, 1], zs=xyz[:, 2]))
        counts.update({(t, i): len(p) for p, i, _ in instances})
    model = describe_model(source, "PredInstance")
    dims = {n: p.type for n, p in model.dimensions.items()}
    with PointIndex(tmp_path / "index.db", dims) as index:
        for t, f in enumerate(files):
            index_file(index, f, t, np.zeros(3), model)
        report = {}
        mapping = reconcile_instances(model, files, index, np.zeros(3), counts, .3, .05, report,
                                      overlaps=overlaps, limits=limits)
    return mapping, report["reconciliation"]


def test_a_genuine_tree_seen_by_two_tiles_merges(tmp_path):
    tree = cluster((10, 10, 8))
    mapping, report = reconcile(tmp_path, [[(tree, 1, 1)], [(tree, 7, 1)]])
    assert mapping[(0, 1)] == mapping[(1, 7)] and not report["rejected_candidates"]


def test_same_xy_different_z_stays_distinct(tmp_path):
    # Shared points between an understorey and a crown 12 m above it.
    low, high, shared = cluster((10, 10, 2), seed=1), cluster((10, 10, 14), seed=2), cluster((10, 10, 8), n=20, seed=3)
    mapping, report = reconcile(tmp_path, [[(np.r_[low, shared], 1, 1)], [(np.r_[high, shared], 2, 1)]])
    assert mapping[(0, 1)] != mapping[(1, 2)]
    assert report["rejected_candidates"] == {"centroid_distance": 1}


def test_far_centroids_stay_distinct_even_with_full_overlap_of_the_smaller(tmp_path):
    big, part = cluster((0, 0, 5), n=200, spread=6, seed=4), cluster((5, 0, 5), n=20, spread=.2, seed=5)
    mapping, report = reconcile(tmp_path, [[(np.r_[big, part], 1, 1)], [(part, 2, 1)]],
                                limits=MatchLimits(max_centroid_distance=2.0, min_points=10))
    assert mapping[(0, 1)] != mapping[(1, 2)]


def test_a_fragment_of_the_same_tree_joins_it(tmp_path):
    # A tile edge leaves only part of a tree in the neighbor; ownership, not matching, keeps its points apart.
    tree, tip = cluster((10, 10, 8), n=200, spread=2, seed=6), cluster((10, 10, 9), n=15, spread=.2, seed=7)
    mapping, report = reconcile(tmp_path, [[(np.r_[tree, tip], 1, 1)], [(tip, 2, 1)]])
    assert mapping[(0, 1)] == mapping[(1, 2)] and not report["rejected_candidates"]


def test_too_few_shared_points_do_not_match(tmp_path):
    a = cluster((10, 10, 8), n=8, seed=8)
    mapping, report = reconcile(tmp_path, [[(a, 1, 1)], [(a, 2, 1)]])
    assert mapping[(0, 1)] != mapping[(1, 2)] and report["rejected_candidates"] == {"support": 1}


def test_different_semantics_do_not_match(tmp_path):
    a = cluster((10, 10, 1))
    mapping, report = reconcile(tmp_path, [[(a, 1, 1)], [(a, 2, 4)]])
    assert mapping[(0, 1)] != mapping[(1, 2)] and report["rejected_candidates"] == {"semantic": 1}


def test_non_neighbor_tiles_are_not_compared(tmp_path):
    tree = cluster((10, 10, 8))
    box = [[0, 0], [20, 20]]
    overlaps = [{}, {}, {1: box}]  # tile 2 neighbors tile 1 only
    mapping, _ = reconcile(tmp_path, [[(tree, 1, 1)], [(cluster((100, 100, 5)), 5, 1)], [(tree, 3, 1)]], overlaps=overlaps)
    assert mapping[(0, 1)] != mapping[(2, 3)]


def test_transitive_bridge_is_refused(tmp_path):
    # Centroids 1.75 m apart along the chain, but the ends 3.5 m apart: no group of all three under 3 m.
    a, b, c = (cluster((x, 0, 6), n=60, spread=2.5, seed=s) for x, s in ((0, 9), (3.5, 10), (7, 11)))
    ab, bc = np.r_[a, b], np.r_[b, c]
    mapping, report = reconcile(tmp_path, [[(ab, 1, 1)], [(np.r_[ab, bc], 2, 1)], [(bc, 3, 1)]],
                                limits=MatchLimits(max_centroid_distance=3.0, min_points=10))
    assert len({mapping[(0, 1)], mapping[(1, 2)], mapping[(2, 3)]}) == 2 and report["rejected_candidates"].get("group_limits") == 1


def test_order_does_not_decide_a_contest(tmp_path):
    # Stronger correspondences join first, whatever their tile order.
    tree = cluster((10, 10, 8), n=100, seed=12)
    first = reconcile(tmp_path / "a", [[(tree, 1, 1)], [(tree, 2, 1)], [(tree[:60], 3, 1)]])[0]
    second = reconcile(tmp_path / "b", [[(tree[:60], 3, 1)], [(tree, 2, 1)], [(tree, 1, 1)]])[0]
    assert (first[(0, 1)] == first[(1, 2)]) and (second[(1, 2)] == second[(2, 1)])


def test_a_tree_cut_by_the_tile_edge_merges_on_its_overlap_part(tmp_path):
    # The tree spans x 0-20 m; tile 0 sees x <= 12, tile 1 x >= 8, the overlap is x 8-12. Whole-copy
    # centroids lie 8 m apart, but in the overlap both copies are the same points.
    rng = np.random.default_rng(13)
    tree = np.c_[rng.uniform(0, 20, 600), rng.uniform(8, 12, 600), rng.uniform(2, 14, 600)]
    left, right = tree[tree[:, 0] <= 12], tree[tree[:, 0] >= 8]
    overlaps = [{}, {0: (np.array([8.0, -100.0]), np.array([12.0, 100.0]))}]
    mapping, report = reconcile(tmp_path, [[(left, 1, 1)], [(right, 2, 1)]], overlaps=overlaps)
    assert mapping[(0, 1)] == mapping[(1, 2)] and not report["rejected_candidates"]
    assert report["accepted_pairs"][0]["overlap_centroid_distance_m"] < 0.5
