"""Reproducible local GFZ integration suite; run only against local Docker."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--stage', choices=['all','tile','models','postprocess'], default='all')
    args = parser.parse_args()
    root = args.run_root.resolve()
    os.environ['GFZ_RUN_ROOT'] = str(root)
    if args.stage in ('all','tile'):
        if not args.archive:
            parser.error('--archive is required for a new run')
        root.mkdir(parents=True, exist_ok=False)
        for name in ('input','logs','commands','outputs','work','archive'):
            (root/name).mkdir()
        for file in Path(__file__).parent.glob('*.py'):
            shutil.copy2(file,root/'commands'/file.name)
        archive=args.archive.resolve()
        with zipfile.ZipFile(archive) as z:
            for entry in z.infolist():
                if entry.is_dir():continue
                name=Path(entry.filename).name
                if not name.lower().endswith('.laz'):raise ValueError(entry.filename)
                destination=root/'input'/name
                if destination.exists():raise ValueError('Duplicate input name: '+name)
                with z.open(entry) as src,destination.open('wb') as dst:shutil.copyfileobj(src,dst)
        (root/'manifest.json').write_text(json.dumps({'input_archive':str(archive),
            'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
            'resources':{'cpus_per_container':10,'memory_gib_per_container':50,'gpus':[0,1],'max_containers':5},
            'state':'processing','stages':{}},indent=2))
    from runner import tile
    import models
    import postprocess
    if args.stage in ('all','tile'):tile()
    if args.stage in ('all','models'):models.main()
    if args.stage in ('all','postprocess'):postprocess.main()

if __name__ == '__main__':main()
