import sys,tempfile,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from bounded_point_index import PointIndex
from spatial_query_cache import SpatialQueryCache

class QueryCacheTests(unittest.TestCase):
    def test_cached_matches_disk_for_ties_filters_and_vector_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool=SpatialQueryCache(8*1024**2)
            dims={'id':np.uint32,'vector':np.dtype((np.int16,3))}
            with PointIndex(Path(tmp)/'i.db',dims,query_cache=pool) as idx:
                xyz=np.array([[-.01,0,0],[0,0,0],[0,0,0],[.02,0,0],[8.,1,2]])
                for tile in (1,0):
                    idx.add(tile,xyz,{'id':np.array([0,2,3,4,5],dtype=np.uint32)+tile,'vector':np.arange(15,dtype=np.int16).reshape(5,3)},np.array([5,4,2,1,0]))
                idx.flush();q=np.array([[0,0,0],[.01,0,0],[8.,1,2],[99,0,0]])
                for kw in ({},{'positive_dimension':'id'},{'tile':1},{'before_tile':1},{'overlaps':{0:(np.array([-.1,-1]),np.array([.01,1]))}}):
                    got=idx.nearest(q,.01,**kw);expected=idx._nearest_batches(q,.01,**kw)
                    np.testing.assert_array_equal(got[0],expected[0]);np.testing.assert_array_equal(got[2],expected[2])
                    for k in dims:np.testing.assert_array_equal(got[1][k],expected[1][k])
                hits=pool.hits;idx.nearest(q,.01);idx.nearest(q,.01);self.assertGreater(pool.hits,hits)
                self.assertLessEqual(pool.bytes_used,pool.max_bytes)
            self.assertEqual(pool.bytes_used,0)

    def test_entry_is_charged_at_kept_size_not_halo_blob_size(self):
        # Blobs touching a region's halo hold more points than the region keeps.
        # Charging the read size let only a few dense regions fit, so they were
        # rebuilt repeatedly; the kept entry is charged at its real size.
        from spatial_query_cache import TREE_BYTES_PER_POINT, region_entry
        with tempfile.TemporaryDirectory() as tmp:
            pool=SpatialQueryCache(64*1024**2)
            with PointIndex(Path(tmp)/'i.db',{'id':np.uint32},query_cache=pool) as idx:
                rng=np.random.default_rng(3)
                for _ in range(20):
                    xyz=np.c_[rng.uniform(0,6,1000),rng.uniform(0,2,1000),np.zeros(1000)]
                    idx.add(0,xyz,{'id':np.ones(1000,dtype=np.uint32)},np.arange(1000))
                idx.flush()
                tree,payload=region_entry(idx,0,(0,0),.01,None,None)
                self.assertTrue(np.all(payload['xyz'][:,0]<=4.01))
                read=sum(len(b) for (b,) in idx.db.execute('SELECT b.data FROM bounds r JOIN batches b ON b.id=r.id WHERE r.x0<=4.01'))
                self.assertLess(payload.nbytes,read)
                (_,charged),=pool.entries.values()
                self.assertEqual(charged,payload.nbytes+len(payload)*TREE_BYTES_PER_POINT+1024**2)
                tree,payload=region_entry(idx,0,(1,1),.01,None,None)   # far region: no points
                self.assertIsNone(tree);self.assertEqual(len(payload),0)
                self.assertLessEqual(pool.bytes_used,pool.max_bytes)

    def test_append_invalidates_cached_region(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool=SpatialQueryCache(8*1024**2)
            with PointIndex(Path(tmp)/'i.db',{'id':np.uint32},query_cache=pool) as idx:
                idx.add(0,np.array([[.02,0,0]]),{'id':np.array([1])},np.array([0]));idx.flush()
                q=np.zeros((1,3));self.assertEqual(idx.nearest(q,.05)[1]['id'][0],1)
                idx.add(0,q,{'id':np.array([9])},np.array([1]));idx.flush()
                self.assertEqual(idx.nearest(q,.05)[1]['id'][0],9)

    def test_tiny_budget_falls_back_and_pool_is_shared(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool=SpatialQueryCache(1)
            with PointIndex(Path(tmp)/'a.db',{'id':np.uint32},query_cache=pool) as a, PointIndex(Path(tmp)/'b.db',{'id':np.uint32},query_cache=pool) as b:
                for idx in (a,b):
                    idx.add(0,np.zeros((1,3)),{'id':np.array([4])},np.array([0]));idx.flush()
                    self.assertEqual(idx.nearest(np.zeros((1,3)),.01)[1]['id'][0],4)
                self.assertEqual(pool.bytes_used,0)

    def test_read_only_reopen_and_radius_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'a.db';pool=SpatialQueryCache(8*1024**2)
            with PointIndex(path,{'id':np.uint32},query_cache=pool) as idx:
                idx.add(0,np.array([[8.01,0,0]]),{'id':np.array([4])},np.array([0]));idx.flush()
            with PointIndex(path,{'id':np.uint32},query_cache=pool,read_only=True) as idx:
                q=np.array([[8.,0,0]])
                np.testing.assert_array_equal(idx.nearest(q,.01)[2],idx._nearest_batches(q,.01)[2])

    def test_two_collections_keep_separate_ids_with_identical_cached_output(self):
        from unittest.mock import patch
        from remap_first_pipeline import merge_collections
        from test_merge_stages import write_cloud
        from test_remap_first_pipeline import layout
        import laspy
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);collections=[root/'SAT',root/'FM'];originals=root/'originals';originals.mkdir()
            write_cloud(originals/'a.las',[0,.5,1,1.5,2])
            for model,folder in zip(('SAT','FM'),collections):
                folder.mkdir()
                write_cloud(folder/'a.las',[0,.5,1],[7,7,7],instance='PredInstance_'+model)
                write_cloud(folder/'b.las',[1,1.5,2],[12,12,12],instance='PredInstance_'+model)
            for label,pool in [('disk',SpatialQueryCache(0)),('cache',SpatialQueryCache(16*1024**2))]:
                with patch('spatial_query_cache.shared_cache',return_value=pool):
                    report=merge_collections(collections=collections,target_dir=None,output_tiles=root/label/'tiles',
                        tile_bounds_json=layout(root,2),originals=originals,ready=True)
                self.assertEqual(report['state'],'validated')
            for rel in ('tiles/model_000/c00_r00.laz','tiles/model_001/c01_r00.laz','original_with_predictions/a.las'):
                a,b=laspy.read(root/'disk'/rel),laspy.read(root/'cache'/rel)
                np.testing.assert_array_equal(a.points.array,b.points.array)
            cloud=laspy.read(root/'cache/original_with_predictions/a.las')
            self.assertIn('PredInstance_SAT',cloud.point_format.dimension_names)
            self.assertIn('PredInstance_FM',cloud.point_format.dimension_names)

    def test_eviction_and_randomized_queries_match_disk(self):
        rng=np.random.default_rng(2183)
        with tempfile.TemporaryDirectory() as tmp:
            pool=SpatialQueryCache(8*1024**2)
            with PointIndex(Path(tmp)/'i.db',{'id':np.uint32},query_cache=pool) as idx:
                clouds=[]
                for tile in (2,0,1):
                    xyz=rng.uniform([-20,-10,-2],[20,10,2],size=(300,3))
                    xyz[:3]=[[0,0,0],[0,0,0],[.01,0,0]]
                    clouds.append(xyz)
                    labels=rng.integers(0,10,len(xyz),dtype=np.uint32)
                    for start in range(0,len(xyz),50):
                        idx.add(tile,xyz[start:start+50],{'id':labels[start:start+50]},np.arange(start,start+50))
                idx.flush()
                q=np.concatenate(clouds)+rng.uniform(-.005,.005,(900,3))
                for radius in (.01,.05):
                    for kw in ({},{'tile':0},{'before_tile':2},{'positive_dimension':'id'},
                               {'overlaps':{0:(np.array([-20.,-10.]),np.array([0.,10.])),1:(np.array([0.,-10.]),np.array([20.,10.]))}}):
                        actual=idx.nearest(q,radius,**kw);expected=idx._nearest_batches(q,radius,**kw)
                        np.testing.assert_array_equal(actual[0],expected[0]);np.testing.assert_array_equal(actual[1]['id'],expected[1]['id']);np.testing.assert_array_equal(actual[2],expected[2])
                        self.assertLessEqual(pool.bytes_used,pool.max_bytes)
                self.assertLess(len(pool.entries),20)


def test_dense_region_is_cached_per_query_cell_with_identical_results(tmp_path):
    # A 4 m region too large for the cache: its 2 m query cells are cached instead,
    # and nearest results equal the bounded disk path, ties included.
    rng = np.random.default_rng(7)
    region_points = 40_000
    xyz = np.c_[rng.uniform(0, 4, region_points), rng.uniform(0, 4, region_points), rng.uniform(0, 10, region_points)]
    xyz = np.round(xyz, 3)
    xyz[:50] = xyz[50:100]  # exact duplicates: ties go to the lowest source index
    queries = np.round(np.c_[rng.uniform(0, 4, 3000), rng.uniform(0, 4, 3000), rng.uniform(0, 10, 3000)], 3)
    small = SpatialQueryCache(6 * 1024**2)  # the region needs more than half of it, a 2 m cell does not
    with PointIndex(tmp_path / 'i.db', {'id': np.uint32}, query_cache=small) as idx:
        for start in range(0, region_points, 20_000):  # inserts are bounded batches
            part = slice(start, start + 20_000)
            idx.add(0, xyz[part], {'id': np.arange(1, region_points + 1, dtype=np.uint32)[part]},
                    np.arange(region_points)[part])
        idx.flush()
        got = idx.nearest(queries, .05)
        stats = small.stats()
        expected = idx._nearest_batches(queries, .05)
        np.testing.assert_array_equal(got[0], expected[0]); np.testing.assert_array_equal(got[2], expected[2])
        np.testing.assert_array_equal(got[1]['id'], expected[1]['id'])
        # Only the sparse neighbor regions reached by queries on x or y = 4 are built whole.
        assert stats['cell_builds'] >= 4 and stats['disk_fallbacks'] == 0
        assert small.peak_bytes <= small.max_bytes


def test_regions_go_to_workers_by_point_count(tmp_path):
    from parallel_remap import balanced_regions, region_owners
    with PointIndex(tmp_path / 'i.db', {'id': np.uint32}) as idx:
        sizes = {(0, 0): 9000, (1, 0): 3000, (2, 0): 3000, (3, 0): 3000, (4, 0): 100}  # each under the batch bound
        for (rx, ry), n in sizes.items():
            pts = np.c_[np.full(n, rx * 4 + 1.0), np.full(n, ry * 4 + 1.0), np.arange(n) * .01]
            idx.add(0, pts, {'id': np.ones(n, dtype=np.uint32)}, np.arange(n))
        idx.flush()
        assigned = balanced_regions([idx, None], 2)
        again = balanced_regions([idx], 2)  # deterministic
        assert all(np.array_equal(a, b) for kind in assigned for a, b in zip(assigned[kind], again[kind]))
        # The dense spot gets a worker of its own; the three medium regions share the other.
        owners = region_owners(np.array([[rx * 4 + 1., 1., 0.] for rx in range(5)] + [[401., 1., 0.]]), 2, assigned)
        assert owners[0] != owners[1] and owners[1] == owners[2] == owners[3] and owners[5] in (0, 1)


def test_tiles_out_of_reach_are_skipped_with_identical_results(tmp_path):
    # Six tiles in a grid; each query group can reach only its neighbours.
    rng = np.random.default_rng(2802)
    for budget in (0, 1, 8 * 1024**2):
        pool = SpatialQueryCache(budget)
        with PointIndex(tmp_path / f'{budget}.db', {'id': np.uint32}, query_cache=pool) as idx:
            clouds = []
            for tile in (4, 1, 0, 5, 3, 2):
                x0, y0 = 10.0 * (tile % 3), 10.0 * (tile // 3)
                xyz = rng.uniform([x0 - 2, y0 - 2, 0], [x0 + 12, y0 + 12, 3], (300, 3))
                xyz[:2] = [[x0 + 10, y0 + 10, 1], [x0 + 10, y0 + 10, 1]]  # coincident corners across tiles
                labels = rng.integers(0, 4, len(xyz), dtype=np.uint32)
                for start in range(0, len(xyz), 60):
                    idx.add(tile, xyz[start:start + 60], {'id': labels[start:start + 60]}, np.arange(start, start + 60))
                clouds.append(xyz)
            idx.flush()
            q = np.concatenate(clouds + [rng.uniform([-5, -5, 0], [35, 25, 3], (200, 3))]) + rng.uniform(-.006, .006, (2000, 3))
            for kw in ({}, {'tile': 4}, {'before_tile': 3}, {'positive_dimension': 'id'},
                       {'overlaps': {0: (np.array([8., -2.]), np.array([12., 12.])), 2: (np.array([18., 8.]), np.array([22., 12.]))}}):
                actual, expected = idx.nearest(q, .01, **kw), idx._nearest_batches(q, .01, **kw)
                np.testing.assert_array_equal(actual[0], expected[0])
                np.testing.assert_array_equal(actual[1]['id'], expected[1]['id'])
                np.testing.assert_array_equal(actual[2], expected[2])
            distances = idx.nearest_distances(q, .01)
            np.testing.assert_array_equal(np.isfinite(distances), np.isfinite(idx._nearest_batches(q, .01)[0]))
            assert pool.bytes_used <= pool.max_bytes
