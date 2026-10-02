import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from bounded_point_index import PointIndex
from dense_instance_ownership import ownership_candidates
from spatial_query_cache import SpatialQueryCache


def test_ownership_cache_matches_disk_and_filters_source_side_of_core_boundary():
    regions=[{'core':[[0,1],[-1,1]]},{'core':[[1,2],[-1,1]]},{'core':[[0,2],[1,2]]}]
    bounds=(np.array([-1.,-1.]),np.array([3.,3.]))
    model=SimpleNamespace(instance='instance')
    queries=np.array([[.9995,0,0],[.5,.9995,0],[.3,0,0]])
    xyz=np.array([[1.001,0,0],[.99,0,0],[.5,1.001,0],[.5,.99,0],[.3,0,0]])
    labels=np.array([1,1,2,2,0],np.uint32)
    results=[]
    for budget in (0,64*1024**2):
        with tempfile.TemporaryDirectory() as tmp:
            cache=SpatialQueryCache(budget)
            with PointIndex(Path(tmp)/'points.db',{'instance':np.uint32},query_cache=cache) as index:
                index.add(0,xyz,{'instance':labels},np.arange(5));index.flush()
                mapping={(0,1):10,(0,2):20}
                case=[]
                for current in (1,2):
                    for background in (False,True):
                        found=np.zeros(len(queries),bool)
                        for _,_,tree in ownership_candidates(index,queries,current,{0:bounds},regions,np.zeros(3),model,mapping,background_only=background):
                            d,_=tree.query(queries);found|=d<=.01
                        case.append(found.tolist())
                results.append(case)
                if budget:
                    hits=cache.hits
                    list(ownership_candidates(index,queries,1,{0:bounds},regions,np.zeros(3),model,mapping,background_only=False))
                    assert cache.hits>hits
                    assert cache.bytes_used<=cache.max_bytes
    assert results[0]==results[1]
    assert results[1][0]==[True,True,False]
