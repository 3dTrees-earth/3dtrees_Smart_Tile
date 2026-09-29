from collections import Counter
from pathlib import Path
from unittest.mock import patch
import laspy
import numpy as np
from bounded_point_index import PointIndex
from dense_tile_merge import describe_model,index_file,reconcile_instances
from test_dense_tile_merge import write_cloud


def test_three_tile_reconciliation_reads_each_query_file_once(tmp_path):
    source=tmp_path/'source';source.mkdir()
    files=[write_cloud(source/f'{t}.las',[0,2,4],[t+1,0,t+1]) for t in range(3)]
    model=describe_model(source,'PredInstance');dims={n:p.type for n,p in model.dimensions.items()}
    with PointIndex(tmp_path/'all.db',dims) as index:
        for t,f in enumerate(files):index_file(index,f,t,np.zeros(3),model)
        report={}
        with patch('dense_tile_merge.laspy.open',wraps=laspy.open) as opened:
            mapping=reconcile_instances(model,files,index,np.zeros(3),{(t,t+1):2 for t in range(3)},.3,.05,report)
        reads=Counter(Path(c.args[0]).name for c in opened.call_args_list)
        assert reads=={'1.las':1,'2.las':1}
        assert set(mapping.values())=={1}
        assert [p['matches'] for p in report['reconciliation']['accepted_pairs']]==[2,2,2]
        assert report['reconciliation']['query_points']==6
        assert report['reconciliation']['skipped_query_points']==3


def test_query_pruning_retains_group_roundoff_and_nearest_background(tmp_path):
    source=tmp_path/'source';source.mkdir()
    paths=[]
    for tile in (0,1):
        h=laspy.LasHeader(point_format=3,version='1.2');h.scales=[1e-12,.001,.001];h.offsets=[.01+5e-12 if tile==0 else 0,0,0]
        a=laspy.LasData(h);a.X=[0] if tile==0 else [0,0];a.Y=[0]*len(a.points);a.Z=[0] if tile==0 else [0,16384000]
        a.add_extra_dim(laspy.ExtraBytesParams(name='PredInstance',type=np.uint32));a.PredInstance=[1] if tile==0 else [2,0]
        p=source/f'{tile}.las';a.write(p);paths.append(p)
    model=describe_model(source,'PredInstance');dims={n:p.type for n,p in model.dimensions.items()}
    with PointIndex(tmp_path/'all.db',dims) as index:
        for t,p in enumerate(paths):index_file(index,p,t,np.zeros(3),model)
        report={};mapping=reconcile_instances(model,paths,index,np.zeros(3),{(0,1):1,(1,2):1},.3,.01,report)
        assert set(mapping.values())=={1}, 'Do not drop the high-Z background query that sets the original XY-group roundoff'
    source2=tmp_path/'source2';source2.mkdir()
    files=[write_cloud(source2/'0.las',[.01,0],[1,0]),write_cloud(source2/'1.las',[0],[2])]
    model=describe_model(source2,'PredInstance')
    with PointIndex(tmp_path/'other.db',dims) as index:
        for t,p in enumerate(files):index_file(index,p,t,np.zeros(3),model)
        report={};mapping=reconcile_instances(model,files,index,np.zeros(3),{(0,1):1,(1,2):1},.3,.05,report)
        assert len(set(mapping.values()))==2, 'A closer background point must block a false positive correspondence'
