"""Artifact gates and dense transfer for the GFZ integration suite (inside image)."""
import json
from pathlib import Path
import shutil
import sys
import tempfile

import laspy
import numpy as np

ROOT = Path('/run')


def validate_tiles():
    layout = json.loads((ROOT / '01_tiles/tile_bounds_tindex.json').read_text())
    assert len(layout['tiles']) == 4, 'Expected exactly four tiles'
    report = {}
    for resolution in ('res1', 'res2'):
        files = sorted((ROOT / '01_tiles' / ('subsampled_' + resolution)).glob('*.laz'))
        assert len(files) == 4, (resolution, files)
        report[resolution] = []
        for file, tile in zip(files, layout['tiles']):
            core_count = 0
            with laspy.open(file) as reader:
                assert reader.header.point_count > 0
                assert reader.header.parse_crs() is not None
                (x0, x1), (y0, y1) = tile['core']
                for points in reader.chunk_iterator(1000000):
                    core_count += int(np.count_nonzero((points.x >= x0) & (points.x < x1) &
                                                      (points.y >= y0) & (points.y < y1)))
                report[resolution].append({'file':file.name,'points':reader.header.point_count,'core_points':core_count})
            assert core_count > 0, file
    (ROOT / 'outputs/tile_validation.json').write_text(json.dumps(report, indent=2))


def stage_prediction(model, tile, output):
    output = Path(output)
    matches = []
    dimension = 'PredInstance_' + model
    for file in sorted(output.glob('*.laz')):
        with laspy.open(file) as reader:
            if dimension in reader.header.point_format.dimension_names:
                matches.append(file)
    assert len(matches) == 1, (model, tile, matches)
    target = ROOT / 'collections' / model
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(matches[0], target / (tile + '.laz'))
    source = next(f for f in (ROOT / '01_tiles/subsampled_res2').glob('*.laz') if f.name.startswith(tile))
    with laspy.open(matches[0]) as pred, laspy.open(source) as original:
        assert pred.header.point_count == original.header.point_count, 'Model changed point count'
        count = pred.header.point_count
    if model == 'RCT':
        for kind in ('trees', 'trees_info'):
            files = list(output.rglob('*_' + kind + '.txt'))
            assert len(files) == 1, files
            shutil.copy2(files[0], target / (tile + '_' + kind + '.txt'))
        assert (output / (source.stem + '_mesh.ply')).is_file(), 'RCT terrain mesh missing'
    print(json.dumps({'model':model,'tile':tile,'points':count,'prediction':str(matches[0])}))


def dense(model_name):
    from dense_tile_merge import describe_model, prepare_dense
    from strict_prediction_pipeline import origin_for
    folder = ROOT / 'collections' / model_name
    model = describe_model(folder, 'PredInstance_' + model_name)
    # Source ExtraBytes may be present in predictions, but only model outputs
    # belong in the prediction schema; originals supply their own source fields.
    model.dimensions = {n: p for n, p in model.dimensions.items() if n.startswith(('PredInstance', 'PredSemantic', 'PredScore'))}
    pairs = []
    for file in sorted(folder.glob('*.laz')):
        targets = list((ROOT / '01_tiles/subsampled_res1').glob(file.stem + '*.laz'))
        assert len(targets) == 1, (file, targets)
        pairs.append((file, targets[0], file.stem))
    assert len(pairs) == 4
    output = ROOT / 'dense' / model_name
    output.parent.mkdir(exist_ok=True)
    report = {'model': model_name}
    # The separate filter CLI creates its own dense index. Count the validated
    # transfer records here instead of building a disposable duplicate index.
    class TransferCensus:
        query_workers = 10
        count = 0

        def add(self, tile, xyz, values, references):
            self.count += len(xyz)

        def flush(self):
            pass

    census = TransferCensus()
    files, _ = prepare_dense(model, pairs, output, census, origin_for([p[0] for p in pairs]), .1732, report)
    report['validated_dense_points'] = census.count
    if model_name == 'RCT':
        for target, (source, _, _) in zip(files, pairs):
            for kind in ('trees','trees_info'):
                shutil.copy2(folder / (source.stem + '_' + kind + '.txt'), output / (target.stem + '_' + kind + '.txt'))
    (ROOT / 'outputs' / ('transfer_' + model_name + '.json')).write_text(json.dumps(report, indent=2))

if __name__ == '__main__':
    if sys.argv[1] == 'tiles': validate_tiles()
    elif sys.argv[1] == 'prediction': stage_prediction(*sys.argv[2:])
    elif sys.argv[1] == 'dense': dense(sys.argv[2])
    else: raise ValueError(sys.argv[1])
