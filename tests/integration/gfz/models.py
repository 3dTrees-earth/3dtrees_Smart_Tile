from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import os
import threading
from runner import ROOT, run

FM_CONFIG = os.environ.get('GFZ_FM_CONFIG', '/workspace/configs/ForAINetv2/forestmamba_chm_radius16_qp300_2many_v6_expand_1.py')
FM_CHECKPOINT = os.environ.get('GFZ_FM_CHECKPOINT', '/workspace/work_dirs/forestmamba/v6_expand_1_epoch_3000_fix.pth')

failed = threading.Event()


def branch(model):
    try:
        for source in sorted((ROOT / '01_tiles/subsampled_res2').glob('*.laz')):
            if failed.is_set():
                return
            tile = source.name[:7]  # c00_r00
            dataset = '/run/' + str(source.relative_to(ROOT))
            attempt = os.environ.get('GFZ_FM_ATTEMPT','') if model == 'FM' else ''
            manifest = json.loads((ROOT / 'manifest.json').read_text())
            key = f'{model.lower()}_{tile}{attempt}'
            if manifest['stages'].get(key,{}).get('state') == 'passed':
                validation = f'validate_{model.lower()}_{tile}'
                if manifest['stages'].get(validation,{}).get('state') != 'passed':
                    run(validation,'smart',['python','/run/commands/artifacts.py','prediction',model,tile,
                        f'/run/02_models/{model}{attempt}/{tile}'])
                continue
            output = f'02_models/{model}{attempt}/{tile}'
            out_path = '/run/' + output
            if model == 'SAT':
                args = ['python3.8','-u','/src/run.py','--dataset-path',dataset,'--output-dir',out_path,
                        '--log_file','true','--predinstance_name','PredInstance_SAT','--predsemantic_name','PredSemantic_SAT']
                image, gpu = 'sat', 0
            elif model == 'FM':
                args = ['python','-u','/workspace/src/run.py','--dataset-path',dataset,'--output-dir',out_path,
                        '--work-dir','/work/model','--repo-dir','/workspace',
                        '--checkpoint',FM_CHECKPOINT,
                        '--config-base',FM_CONFIG,
                        '--preprocess-workers','10','--chunk-size','1000000',
                        '--semantic-dim','PredSemantic_FM','--instance-dim','PredInstance_FM','--score-dim','PredScore_FM',
                        '--bluepoint-iterations','0','--bluepoint-second-pass-threshold','0.01','--spatial-match-tolerance','0.01']
                image, gpu = 'fm', 1
            else:
                args = ['python3','-u','/src/run.py','--dataset-path',dataset,'--output-dir',out_path,
                        '--gradient','1','--max-diameter','0.9','--crop-length','1','--distance-limit','1',
                        '--height-min','2','--girth-height-ratio','0.12','--global-taper','0.024',
                        '--global-taper-factor','0.3','--gravity-factor','0.3','--treeinfo']
                image, gpu = 'rct', None
            run(f'{model.lower()}_{tile}{attempt}', image, args, gpu=gpu, output=output)
            if failed.is_set(): return
            run(f'validate_{model.lower()}_{tile}', 'smart',
                ['python','/run/commands/artifacts.py','prediction',model,tile,out_path])
    except Exception:
        failed.set()
        raise


def main():
    manifest = json.loads((ROOT/'manifest.json').read_text())
    if manifest['stages'].get('validate_tiles',{}).get('state') != 'passed':
        run('validate_tiles', 'smart', ['python','/run/commands/artifacts.py','tiles'])
    # Reject partial checkpoint loads before expensive preprocessing/inference.
    run('fm_checkpoint_preflight'+os.environ.get('GFZ_FM_ATTEMPT',''),'fm',
        ['python','/run/commands/check_fm_weights.py','--config',FM_CONFIG,'--checkpoint',FM_CHECKPOINT])
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(branch, m) for m in ('SAT','FM','RCT')]
        for future in as_completed(futures): future.result()

if __name__ == '__main__': main()
