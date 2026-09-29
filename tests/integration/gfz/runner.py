"""Local Docker stage runner with bounded concurrency and reproducible evidence."""
import concurrent.futures
import json
import os
import pwd
from pathlib import Path
import subprocess
import threading
import time

ROOT = Path(os.environ['GFZ_RUN_ROOT']).resolve()
LOCK = threading.Lock()
IMAGES = {
 'smart': os.environ.get('GFZ_SMART_IMAGE', 'smarttile:v2.4a'),
 'sat': 'ghcr.io/3dtrees-earth/3dtrees_sat:1.2.3',
 'fm': os.environ.get('GFZ_FM_IMAGE', 'forestmamba:upstream-f0c5239-expand1-epoch3000-gfz-20260924-kg281'),
 'rct': 'rct:v1.2.2-pr3-cgroup',
 'dv': os.environ.get('GFZ_DV_IMAGE', 'ghcr.io/3dtrees-earth/3dtrees_detailview:1.1.1'),
}


def update(stage, value):
    with LOCK:
        manifest = json.loads((ROOT / 'manifest.json').read_text())
        previous = manifest['stages'].get(stage)
        if previous and previous.get('command') != value.get('command'):
            manifest.setdefault('stage_history', {}).setdefault(stage, []).append(previous)
        manifest['stages'][stage] = value
        manifest['updated_at'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        (ROOT / 'manifest.json.tmp').write_text(json.dumps(manifest, indent=2) + '\n')
        (ROOT / 'manifest.json.tmp').replace(ROOT / 'manifest.json')


def run(stage, image, args, *, gpu=None, output=None):
    import hashlib
    prefix = os.environ.get('GFZ_CONTAINER_PREFIX', 'gfz-' + hashlib.sha256(str(ROOT).encode()).hexdigest()[:8])
    attempt = stage + os.environ.get('GFZ_RUN_ATTEMPT', '')
    name = prefix + '-' + attempt.replace('_', '-')
    uid, gid = os.getuid(), os.getgid()
    if uid == 0:
        raise RuntimeError('Launch the suite as the intended non-root owner')
    username = pwd.getpwuid(uid).pw_name
    work = ROOT / 'work' / attempt
    work.mkdir(parents=True, exist_ok=False)
    for sub in ['tmp', 'cache']:
        (work / sub).mkdir()
    out = ROOT / (output or ('outputs/' + stage))
    out.mkdir(parents=True, exist_ok=True)
    image_id = subprocess.check_output(['docker', 'image', 'inspect', IMAGES[image], '--format', '{{.Id}}'], text=True).strip()
    cmd = ['docker', 'run', '--name', name, '--cpus', '10', '--memory', '50g', '--memory-swap', '50g', '--shm-size', '4g', '--user', f'{uid}:{gid}',
           '-e', f'USER={username}', '-e', f'LOGNAME={username}',
           '-e', 'XDG_CACHE_HOME=/work/cache', '-e', 'HF_HOME=/work/cache/huggingface',
           '-e', 'TRANSFORMERS_CACHE=/work/cache/huggingface', '-e', 'TORCH_HOME=/work/cache/torch',
           '-e', 'TORCH_EXTENSIONS_DIR=/work/cache/torch_extensions',
           '-e', 'GALAXY_SLOTS=10', '-e', 'OMP_NUM_THREADS=10', '-e', 'OPENBLAS_NUM_THREADS=1', '-e', 'MKL_NUM_THREADS=10',
           '-e', 'NUMBA_NUM_THREADS=10', '-e', 'TMPDIR=/work/tmp', '-e', 'NUMBA_CACHE_DIR=/work/cache',
           '-e', 'MPLCONFIGDIR=/work/cache', '-v', f'{ROOT}:/run', '-v', f'{ROOT}/input:/input:ro',
           '-v', f'{work}:/work', '--workdir', '/run/' + str(out.relative_to(ROOT))]
    if gpu is not None:
        cmd += ['--gpus', f'device={gpu}']
    cmd += ['--entrypoint', args[0], image_id, *args[1:]]
    (ROOT / 'commands' / (attempt + '.json')).write_text(json.dumps(cmd, indent=2))
    started = time.time()
    status = {'state': 'running', 'started_at': started, 'image': IMAGES[image], 'image_id': image_id,
              'gpu': gpu, 'uid': uid, 'gid': gid, 'user': username,
              'command': 'commands/' + attempt + '.json', 'log': 'logs/' + attempt + '.log'}
    update(stage, status)
    print(stage, 'running', flush=True)
    peak = cpu_usec = 0
    cgroup = None
    with (ROOT / 'logs' / (attempt + '.log')).open('w') as log, (ROOT / 'logs' / (attempt + '-resources.jsonl')).open('w') as metrics:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            sample = {'elapsed_seconds': time.time() - started}
            try:
                if cgroup is None:
                    pid = int(subprocess.check_output(['docker', 'inspect', name, '--format', '{{.State.Pid}}'], stderr=subprocess.DEVNULL, text=True))
                    if pid:
                        relative = Path(f'/proc/{pid}/cgroup').read_text().strip().split('::')[-1]
                        cgroup = Path('/sys/fs/cgroup') / relative.lstrip('/')
                if cgroup is not None:
                    mem = int((cgroup / 'memory.peak').read_text())
                    usage = dict(line.split() for line in (cgroup / 'cpu.stat').read_text().splitlines())
                    peak, cpu_usec = max(peak, mem), max(cpu_usec, int(usage['usage_usec']))
                    sample.update(memory_peak_bytes=peak, cpu_seconds=cpu_usec / 1e6)
                if gpu is not None:
                    sample['gpu'] = subprocess.check_output(['nvidia-smi', f'--id={gpu}', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip()
                metrics.write(json.dumps(sample) + '\n'); metrics.flush()
            except (OSError, ValueError, subprocess.CalledProcessError):
                pass
            time.sleep(2)
        code = proc.wait()
    status.update(state='passed' if code == 0 else 'failed', exit_code=code,
                  wall_seconds=time.time() - started, observed_memory_peak_bytes=peak,
                  observed_cpu_seconds=cpu_usec / 1e6)
    update(stage, status)
    print(stage, status['state'], round(status['wall_seconds'], 1), 'seconds', flush=True)
    if code:
        raise RuntimeError(f'{stage} failed with exit {code}; see {status["log"]}')
    return out


def tile():
    return run('tile', 'smart', ['python', '-u', '/src/run.py', '--task', 'tile', '--input-dir', '/input',
        '--output-dir', '/run/01_tiles', '--tile-length', '60', '--tile-buffer', '20', '--tiling-threshold', '0',
        '--grid-origin-x', '398663.428', '--grid-origin-y', '5646147.674',
        '--resolution-1', '0.01', '--resolution-2', '0.1', '--subsampling-method', 'center-of-mass',
        '--output-copc-res1', 'True', '--output-copc-res2', 'False', '--chunk-size', '1000000',
        '--workers', '2', '--tile-source-workers', '2', '--tile-writer-workers', '2',
        '--num-spatial-chunks', '10', '--threads', '5', '--memory-gb', '50'], output='01_tiles')

if __name__ == '__main__':
    tile()
