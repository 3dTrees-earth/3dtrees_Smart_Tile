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
            for rel in ('tiles/model_000/tile_00000.laz','tiles/model_001/tile_00001.laz','original_with_predictions/a.las'):
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
