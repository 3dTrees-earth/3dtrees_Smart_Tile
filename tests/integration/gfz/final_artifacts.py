"""Validate DetailView identity preservation and enriched original products."""
import csv
import json
import os
import re
from pathlib import Path
import shutil
import sys

import laspy
import numpy as np

ROOT = Path('/run')


def detailview(model, tile, output):
    source = ROOT / '03_filter' / model / 'output_tiles' / (tile + '.laz')
    target = Path(output) / 'pc_with_species.laz'
    expected = {f'species_id_{model}',f'species_prob_{model}'}
    with laspy.open(source) as a, laspy.open(target) as b:
        assert a.header.point_count == b.header.point_count
        assert expected.issubset(set(b.header.point_format.dimension_names))
        for aa, bb in zip(a.chunk_iterator(1000000), b.chunk_iterator(1000000)):
            for name in a.header.point_format.dimension_names:
                np.testing.assert_array_equal(aa[name],bb[name],err_msg=f'DetailView changed {name}')
        if model == 'RCT':
            original = [v.record_data_bytes() for v in a.header.vlrs if v.user_id == '3DTrees' and v.record_id == 24002]
            final = [v.record_data_bytes() for v in b.header.vlrs if v.user_id == '3DTrees' and v.record_id == 24002]
            assert original == final, 'DetailView lost RCT tile namespace'
    collection = ROOT / '04_detailview' / model / 'collection'
    collection.mkdir(parents=True, exist_ok=True)
    shutil.copy2(target, collection / source.name)
    metadata = json.loads((source.parent / 'smarttile_merge.json').read_text())
    for key in ('baseline','rct_tree_sidecars','tile_bounds_json'):
        if key in metadata:
            absolute = (source.parent / metadata[key]).resolve()
            metadata[key] = os.path.relpath(absolute, collection)
    (collection / 'smarttile_merge.json').write_text(json.dumps(metadata, indent=2))
    for table in ('predictions.csv','predictions_probs.csv'):
        assert (Path(output)/table).is_file(), table


def export_compact_species_tables():
    output = ROOT / 'outputs/original_with_predictions'
    metadata = json.loads((output / 'instance_mapping.json').read_text())
    destination = ROOT / 'outputs/species_tables'
    destination.mkdir(exist_ok=False)
    counts = {}
    for model in ('SAT','FM','RCT'):
        mapping = {row['source_instance_id']: row['instance_id']
                   for row in metadata['models']['PredInstance_'+model]['instances']}
        for source in sorted((ROOT/'04_detailview'/model).glob('tile_*/*.csv')):
            with source.open(newline='') as stream:
                reader = csv.DictReader(stream)
                columns = reader.fieldnames
                key = 'filename' if 'filename' in columns else 'File'
                assert key in columns, (source,columns)
                rows = []
                for row in reader:
                    match = re.search(r'(\d+)$', row[key])
                    assert match, (source,row[key])
                    uid = int(match.group(1))
                    if uid not in mapping: continue
                    row['source_instance_id'] = uid
                    row['instance_id'] = mapping[uid]
                    row['source_filename'] = row[key]
                    row[key] = row[key][:match.start()] + str(mapping[uid])
                    rows.append(row)
            target = destination / (model+'_'+source.parent.name+'_'+source.name)
            with target.open('w',newline='') as stream:
                writer=csv.DictWriter(stream, fieldnames=columns+['instance_id','source_instance_id','source_filename'])
                writer.writeheader();writer.writerows(rows)
            counts[target.name]=len(rows)
    (destination/'manifest.json').write_text(json.dumps(counts,indent=2))


def validate_final():
    output = ROOT / 'outputs/original_with_predictions'
    mapping = json.loads((output / 'instance_mapping.json').read_text())
    seen = {n:set() for n in mapping['models']}
    report = {'files':[], 'models':{}, 'source_fields_preserved':True}
    for source in sorted((ROOT / 'input').glob('*.laz')):
        with laspy.open(source) as a, laspy.open(output / source.name) as b:
            assert a.header.point_count == b.header.point_count
            assert a.header.version == b.header.version
            assert a.header.point_format.id == b.header.point_format.id
            np.testing.assert_array_equal(a.header.scales,b.header.scales)
            np.testing.assert_array_equal(a.header.offsets,b.header.offsets)
            # Projection and other source metadata must remain represented.
            for v in a.header.vlrs:
                if v.user_id == 'laszip encoded': continue
                if v.user_id == 'LASF_Spec' and v.record_id == 4: continue
                assert any((w.user_id,w.record_id,w.record_data_bytes()) == (v.user_id,v.record_id,v.record_data_bytes()) for w in b.header.vlrs)
            local = {n:set() for n in seen}
            for aa, bb in zip(a.chunk_iterator(1000000), b.chunk_iterator(1000000)):
                for dimension in a.header.point_format.dimension_names:
                    np.testing.assert_array_equal(aa[dimension],bb[dimension],err_msg=f'{source.name}/{dimension}')
                for dimension in seen:
                    local[dimension].update(int(x) for x in np.unique(bb[dimension]) if x > 0)
                for model in ('SAT','FM','RCT'):
                    assert f'species_id_{model}' in bb.point_format.dimension_names
                    assert f'species_prob_{model}' in bb.point_format.dimension_names
            for dimension, values in local.items():seen[dimension].update(values)
            for kind in ('trees','trees_info'):
                with (output / f'{source.stem}_{kind}.txt').open() as f:
                    next(f); heading=next(f)
                    ids=[int(line.split(',',1)[0]) for line in f if line.strip()]
                assert len(ids)==len(set(ids))
                assert set(ids)==local['PredInstance_RCT']
            report['files'].append({'file':source.name,'points':a.header.point_count,
                                    'instances':{n:len(ids) for n,ids in local.items()}})
    assert len(report['files']) == 8
    for name, values in seen.items():
        expected = mapping['models'][name]['instance_count']
        assert values == set(range(1, expected+1)), (name, expected)
        report['models'][name] = expected
    (ROOT / 'outputs/final_validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__ == '__main__':
    if sys.argv[1] == 'detailview': detailview(*sys.argv[2:])
    elif sys.argv[1] == 'final': validate_final()
    elif sys.argv[1] == 'tables': export_compact_species_tables()
    else: raise ValueError(sys.argv[1])
