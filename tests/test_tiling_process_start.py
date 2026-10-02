"""Exercise compressed occupancy -> crop -> COPC in fresh, bounded processes."""
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import textwrap

import laspy
import pytest


def test_compressed_core_scan_then_parallel_tiling_completes(tmp_path):
    if laspy.LazBackend.LazrsParallel not in laspy.LazBackend.detect_available():
        pytest.skip('parallel lazrs required to exercise inherited Rayon state')
    if not shutil.which('untwine') or not shutil.which('pdal'):
        pytest.skip('COPC tools required for the complete tiling worker path')
    source_dir = Path(__file__).resolve().parents[1] / 'src'
    script = tmp_path / 'tiling_probe.py'
    script.write_text(textwrap.dedent('''
        import sys
        from pathlib import Path
        import laspy
        import numpy as np
        import main_tile
        from tile_core_occupancy import occupied_cores

        if __name__ == '__main__':
            root = Path(sys.argv[1])
            source = root / 'source.laz'
            cloud = laspy.LasData(laspy.LasHeader(point_format=0, version='1.2'))
            # Multiple compressed chunks are needed to initialize Rayon's pool.
            n = 100_019
            cloud.x = 1 + np.arange(n) % 19
            cloud.y = np.full(n, 5.)
            cloud.z = np.arange(n, dtype=float) * .01
            cloud.intensity = (np.arange(n) % 200 + 100).astype(np.uint16)
            cloud.write(source, laz_backend=laspy.LazBackend.Lazrs)
            tiles = [{'core': [[0, 10], [0, 10]]}, {'core': [[10, 20], [0, 10]]}]
            assert occupied_cores([source], tiles, chunk_size=4096).tolist() == [True, True]
            main_tile.get_source_files_from_tindex = lambda _: [str(source)]
            main_tile.get_source_bounds_from_tindex = lambda _: {str(source): (1, 5, 19, 5)}
            jobs = root / 'jobs.txt'
            jobs.write_text('c00_r00|([-2,12],[-2,12])\\nc01_r00|([8,22],[-2,12])\\n')
            outputs = main_tile.create_tiles(root/'unused.gpkg', jobs, root/'tiles', root/'logs',
                threads=2, max_parallel=2, source_parallel=2, parallel_tiles=2, chunk_size=4096)
            assert len(outputs) == 2
            for path in outputs:
                actual = laspy.read(path)
                mask = np.asarray(cloud.x) <= 12 if path.name.startswith('c00') else np.asarray(cloud.x) >= 8
                # COPC can reorder points and change point format; preserve values.
                order = np.argsort(actual.z)
                for dim in ('x', 'y', 'z', 'intensity', 'classification'):
                    assert np.array_equal(np.asarray(getattr(actual, dim))[order],
                                          np.asarray(getattr(cloud, dim))[mask]), dim
            print('PASS: occupied LAZ cores -> both crop workers -> both COPC outputs', flush=True)
    '''))
    env = dict(os.environ, PYTHONPATH=str(source_dir), RAYON_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
    proc = subprocess.Popen([sys.executable, '-u', str(script), str(tmp_path)],
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, start_new_session=True)
    try:
        output, _ = proc.communicate(timeout=25)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        output, _ = proc.communicate()
        pytest.fail('Tiling hung after compressed core-occupancy scan:\n' + output)
    assert proc.returncode == 0, output
    assert 'PASS: occupied LAZ cores' in output
