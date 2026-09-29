import json
from pathlib import Path
import laspy
import numpy as np
import pytest
from test_dense_tile_merge import write_cloud
from strict_prediction_pipeline import merge_collections


@pytest.mark.parametrize('missing',[False,True])
def test_combined_remap_uses_worker_budget_with_identical_outputs(tmp_path,monkeypatch,missing):
    import strict_prediction_pipeline as pipeline
    monkeypatch.setattr(pipeline,'MAX_BATCH_POINTS',8)
    monkeypatch.setattr(pipeline,'spatial_query_worker_count',lambda n:n)
    originals=tmp_path/'originals';predictions=tmp_path/'predictions'
    originals.mkdir();predictions.mkdir()
    xs=np.arange(35)*.02
    write_cloud(originals/'raw.las',xs)
    px=xs[:-1] if missing else xs
    write_cloud(predictions/'a.las',px,np.full(len(px),7),np.full(len(px),3))
    layout=tmp_path/'layout.json';layout.write_text(json.dumps({'tiles':[]}))
    reports=[]
    for workers in (1,4):
        out=tmp_path/f'case{workers}'
        args=dict(collections=[predictions],target_dir=None,output_tiles=out/'tiles',tile_bounds_json=layout,ready=True,originals=originals,workers=workers)
        if missing:
            with pytest.raises(ValueError,match='100% original coverage'):merge_collections(**args)
            assert not (out/'original_with_predictions').exists()
            assert not (out/'tiles').exists()
            reports.append(json.loads((out/'remap_first_report.json').read_text()))
        else:
            reports.append(merge_collections(**args))
    assert reports[1]['parallelism']['enrichment_processes']==4
    assert reports[1]['parallelism']['max_pending_batches']==8
    assert reports[0]['original_coverage']==reports[1]['original_coverage']
    if not missing:
        np.testing.assert_array_equal(laspy.read(tmp_path/'case1/original_with_predictions/raw.las').points.array,laspy.read(tmp_path/'case4/original_with_predictions/raw.las').points.array)
