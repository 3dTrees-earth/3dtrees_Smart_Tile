from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import threading
import json
import os
from runner import ROOT, run as run_stage

failed = threading.Event()


def run(*args, **kwargs):
    if os.environ.get('GFZ_RESUME_PASSED') == '1':
        previous = json.loads((ROOT / 'manifest.json').read_text())['stages'].get(args[0], {})
        if previous.get('state') == 'passed':
            print(args[0], 'reusing validated stage', flush=True)
            return
    if failed.is_set():
        raise RuntimeError("An earlier postprocessing stage failed; further launches stopped")
    try:
        return run_stage(*args, **kwargs)
    except Exception:
        failed.set()
        raise


def filtering(model):
    run('dense_' + model.lower(), 'smart', ['python','/run/commands/artifacts.py','dense',model])
    args = ['python','-u','/src/run.py','--task','filter',
        '--input-dir','/run/dense/' + model, '--output-dir',f'/run/03_filter/{model}/output_tiles',
        '--tile-bounds-json','/run/01_tiles/tile_bounds_tindex.json',
        '--instance-dimension','PredInstance_' + model,'--filter-anchor','centroid',
        '--overlap-threshold','0.3','--workers','10']
    if model == 'RCT': args += ['--disable-matching','True']
    run('filter_' + model.lower(),'smart',args,output=f'03_filter/{model}')


def detailviews(models, gpu):
    for model in models:
        for source in sorted((ROOT / '03_filter' / model / 'output_tiles').glob('*.laz')):
            tile = source.stem
            output=f'04_detailview/{model}/{tile}'
            run(f'dv_{model.lower()}_{tile}','dv',[
                'python3','-u','/app/run.py','--dataset-path','/run/' + str(source.relative_to(ROOT)),
                '--output-dir','/run/' + output,'--tree-id-col','PredInstance_'+model,'--n-aug','10',
                '--projection-backend','torch','--model-path','/app/model_europe_v1',
                '--output-species-id-dim','species_id_'+model,'--output-species-prob-dim','species_prob_'+model,
                '--output-type','both'],gpu=gpu,output=output)
            run(f'validate_dv_{model.lower()}_{tile}','smart',[
                'python','/run/commands/final_artifacts.py','detailview',model,tile,'/run/'+output])


def main():
    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(filtering,m) for m in ('SAT','FM','RCT')]):future.result()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(detailviews,['SAT','RCT'],0),pool.submit(detailviews,['FM'],1)]
        for future in as_completed(futures):future.result()
    fields = ['PredInstance_SAT','PredSemantic_SAT','PredInstance_FM','PredSemantic_FM','PredScore_FM','PredInstance_RCT']
    for model in ('SAT','FM','RCT'):
        fields += ['species_id_'+model,'species_prob_'+model]
    run('final_remap','smart',['python','-u','/src/run.py','--task','remap',
        '--segmented-folders',','.join('/run/04_detailview/'+m+'/collection' for m in ('SAT','FM','RCT')),
        '--original-laz-input-dir','/input','--output-dir','/run/outputs/original_with_predictions',
        '--remap-dims',','.join(fields),'--resolution-1','0.01','--workers','10','--chunk-size','1000000','--memory-gb','50'])
    run('compact_species_tables','smart',['python','/run/commands/final_artifacts.py','tables'])
    run('validate_final','smart',['python','/run/commands/final_artifacts.py','final'])

if __name__ == '__main__':main()
