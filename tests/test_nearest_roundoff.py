"""Regression from the full GFZ combined/standalone remap discrepancy."""
import tempfile
from pathlib import Path
import laspy
import numpy as np
from bounded_point_index import PointIndex,coordinates
from spatial_query_cache import SpatialQueryCache


def point(offset,raw):
    a=laspy.LasData(laspy.LasHeader(point_format=3,version='1.2'))
    a.header.scales=np.full(3,.001);a.header.offsets=np.asarray(offset)
    for n,v in zip(('X','Y','Z'),raw):a[n]=[v]
    return a


def test_gfz_remap_roundoff_uses_stable_tile_tie_across_origins():
    query=point([399157.564,5646649.995,368.34000000000003],[-424917,-462321,20428])
    sources=[point([398693.428,5646237.674,368.34000000000003],[39219,-50000,20428]),
             point([398753.428,5646177.674,368.34000000000003],[-20781,10000,20428])]
    for origin in (query.header.offsets,[398693.428,5646177.674,368.34000000000003]):
        for budget in (0,32*1024**2):
            with tempfile.TemporaryDirectory() as tmp:
                with PointIndex(Path(tmp)/'points.db',{'score':np.float32},query_cache=SpatialQueryCache(budget)) as index:
                    for tile,source in enumerate(sources):
                        index.add(tile,coordinates(source.points,source.header,origin),{'score':np.array([.76 if tile==0 else .88],np.float32)},np.array([0]))
                    index.flush()
                    _,values,refs=index.nearest(coordinates(query.points,query.header,origin),.01732)
                    assert refs.tolist()==[[0,0]], 'Numerical tie must keep stable source-tile priority'
                    assert values['score'][0]==np.float32(.76)


def test_numerical_ties_are_anchored_to_minimum_across_batch_partitions():
    epsilon=8*np.spacing(1.)
    xyz=np.array([[.125+2*epsilon,0,0],[.125+epsilon,0,0],[.125,0,0]])
    for cache in (0,32*1024**2):
        for split in (False,True):
            with tempfile.TemporaryDirectory() as tmp:
                with PointIndex(Path(tmp)/'points.db',{'score':np.int32},query_cache=SpatialQueryCache(cache)) as index:
                    for group in ([np.array([0,1,2])] if not split else [np.array([i]) for i in range(3)]):
                        index.add(0,xyz[group],{'score':group},group)
                    index.flush()
                    _,values,refs=index.nearest(np.zeros((1,3)),.2)
                    assert refs.tolist()==[[0,1]]
                    assert values['score'].tolist()==[1]


def test_baseline_distances_do_not_need_source_identity_or_attributes():
    from scipy.spatial import cKDTree
    rng=np.random.default_rng(25)
    xyz=rng.uniform(-3,3,(500,3));queries=np.vstack([xyz[:100],xyz[100:200]+.01,[[99,99,99]]])
    expected,_=cKDTree(xyz).query(queries,distance_upper_bound=.02)
    for budget in (0,64*1024**2):
        with tempfile.TemporaryDirectory() as tmp:
            with PointIndex(Path(tmp)/'points.db',{},query_cache=SpatialQueryCache(budget)) as index:
                for tile in (0,1):index.add(tile,xyz,{},np.arange(len(xyz)))
                index.flush()
                np.testing.assert_array_equal(index.nearest_distances(queries,.02),expected)
                assert index.nearest_distances(np.zeros((0,3)),.02).size==0
