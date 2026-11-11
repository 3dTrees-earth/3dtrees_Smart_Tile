"""The QSM octree builder: original RayCloudTools files in, one octree out (3DT-2626).

Fixtures are cut from the merged 3753 outputs (fixtures/make_rct_fixture.py): trees 1, 2 and 93.
"""

import json
import struct
from pathlib import Path

from qsm_octree import build

FIXTURE = Path(__file__).parent / 'fixtures' / 'rct-3753'
INFO = FIXTURE / '11505_trees_info.txt'
MESH = FIXTURE / '11505_trees_mesh.ply'


def tree_records(out: Path) -> list[dict]:
    """The tree table as the octree describes it (the layout comes from hierarchy.json)."""
    hierarchy = json.loads((out / 'hierarchy.json').read_text())
    table = hierarchy['trees']
    fields = table['record']
    fmt = '<' + ''.join(kind for _, kind in fields)
    data = (out / 'octree.bin').read_bytes()[table['offset']:table['offset'] + table['bytes']]
    size = struct.calcsize(fmt)
    assert size == table['record_bytes'] and len(data) == size * table['count']
    names = [name for name, _ in fields]
    return [dict(zip(names, struct.unpack_from(fmt, data, i * size))) for i in range(table['count'])]


def test_each_tree_record_points_at_exactly_its_line_of_trees_info(tmp_path):
    build([(INFO, MESH)], tmp_path)
    lines = INFO.read_bytes().split(b'\n')[2:-1]
    by_id = {int(line.split(b',', 1)[0]): line + b'\n' for line in lines}
    info = INFO.read_bytes()
    records = tree_records(tmp_path)
    assert [r['id'] for r in records] == [1, 2, 93]
    for r in records:
        assert info[r['info_offset']:r['info_offset'] + r['info_length']] == by_id[r['id']]


def mesh_body():
    """The fixture mesh parsed in full: header end, vertex size, faces (count, a, b, c, tree_id)."""
    data = MESH.read_bytes()
    end = data.index(b'end_header\n') + len(b'end_header\n')
    header = data[:end].decode('ascii').splitlines()
    n_vertex = int(next(h for h in header if h.startswith('element vertex')).split()[2])
    n_face = int(next(h for h in header if h.startswith('element face')).split()[2])
    vertex_size = 3 * 8 + 4  # double x, y, z; uchar r, g, b, a
    face_start = end + n_vertex * vertex_size
    faces = [struct.unpack_from('<iiiiI', data, face_start + 20 * i) for i in range(n_face)]
    return data, end, vertex_size, faces


def test_each_tree_record_points_at_exactly_its_faces_and_vertices_in_the_mesh(tmp_path):
    build([(INFO, MESH)], tmp_path)
    data, vertex_start, vertex_size, faces = mesh_body()
    for r in tree_records(tmp_path):
        own = [f for f in faces if f[4] == r['id']]
        chunk = data[r['face_offset']:r['face_offset'] + r['face_length']]
        read = [struct.unpack_from('<iiiiI', chunk, i) for i in range(0, len(chunk), 20)]
        assert read == own and len(own) > 0
        used = {v for f in own for v in f[1:4]}
        assert min(used) == r['vertex_base']
        assert r['vertex_offset'] == vertex_start + r['vertex_base'] * vertex_size
        assert r['vertex_length'] == (max(used) - min(used) + 1) * vertex_size


def test_tree_records_carry_treeinfo_attributes_and_the_originals_sizes(tmp_path):
    hierarchy = build([(INFO, MESH)], tmp_path)
    tree1 = next(r for r in tree_records(tmp_path) if r['id'] == 1)
    # From the fixture's line of tree 1: height, crown radius, DBH, then the first (base) segment.
    assert abs(tree1['height'] - 18.6547) < 1e-4
    assert abs(tree1['crown_radius'] - 1.2301) < 1e-4
    assert abs(tree1['dbh'] - 0.2278) < 1e-4
    ox, oy, oz = hierarchy['origin']
    assert abs(tree1['base_x'] + ox - 437150.4379) < 1e-3
    assert abs(tree1['base_y'] + oy - 5842032.9962) < 1e-3
    assert abs(tree1['base_z'] + oz - 1118.0640) < 1e-3
    assert hierarchy['originals'] == [{
        'info': {'name': '11505_trees_info.txt', 'size': INFO.stat().st_size},
        'mesh': {'name': '11505_trees_mesh.ply', 'size': MESH.stat().st_size},
    }]


def test_tree_records_carry_every_per_tree_value_and_the_root_segment_totals(tmp_path):
    build([(INFO, MESH)], tmp_path)
    # Tree 1's line: tree columns, then its first (root) segment, whose volume, diameter, dominance,
    # angle and children are the tree's total volume, maximum diameter and means (treeinfo's help).
    line = next(l for l in INFO.read_text().splitlines()[2:] if l.startswith('1,'))
    parts = line.split(', ')
    tree_values = [float(v) for v in parts[0].split(',')]
    root = dict(zip(['x', 'y', 'z', 'radius', 'parent_id', 'section_id', 'volume', 'diameter', 'length', 'strength',
                     'min_strength', 'dominance', 'angle', 'children'], [float(v) for v in parts[1].split(',')]))
    tree1 = next(r for r in tree_records(tmp_path) if r['id'] == 1)
    for name, value in zip(['dimension', 'monocotal', 'bend', 'branch_slope'], [tree_values[3], tree_values[4], tree_values[6], tree_values[7]]):
        assert abs(tree1[f'tree.{name}'] - value) < 1e-4, name
    for name in ('volume', 'diameter', 'dominance', 'angle', 'children'):
        assert abs(tree1[f'root.{name}'] - root[name]) < 1e-4, name


def test_a_result_without_mesh_builds_without_mesh_ranges(tmp_path):
    hierarchy = build([(INFO, None)], tmp_path)
    assert 'mesh' not in hierarchy['originals'][0]
    for r in tree_records(tmp_path):
        assert (r['vertex_offset'], r['vertex_length'], r['face_offset'], r['face_length']) == (0, 0, 0, 0)
        assert r['info_length'] > 0


def test_a_mesh_whose_tree_faces_are_split_is_refused(tmp_path):
    data, _, _, faces = mesh_body()
    face_start = len(data) - 20 * len(faces)
    # Move the first face of the file to the end: tree 1's faces are no longer one run.
    broken = data[:face_start] + data[face_start + 20:] + data[face_start:face_start + 20]
    (tmp_path / 'mesh.ply').write_bytes(broken)
    try:
        build([(INFO, tmp_path / "mesh.ply")], tmp_path / "out")
    except ValueError as error:
        assert 'one run' in str(error)
    else:
        raise AssertionError('a split tree was accepted')


def test_every_cylinder_of_the_originals_is_in_one_node(tmp_path):
    hierarchy = build([(INFO, MESH)], tmp_path, capacity=100)
    # Cylinders = segments with a parent (each line's segments after the first, which is the root here).
    lines = INFO.read_bytes().split(b'\n')[2:-1]
    expected = {int(line.split(b',', 1)[0]): line.count(b', ') - 1 for line in lines}
    data = (tmp_path / 'octree.bin').read_bytes()
    counted: dict[int, int] = {}
    for node in hierarchy['nodes']:
        for i in range(node['count']):
            tree = struct.unpack_from('<I', data, node['offset'] + 18 * i + 14)[0]
            counted[tree] = counted.get(tree, 0) + 1
    assert counted == expected
    assert hierarchy['totals']['levels'] > 1
    assert {r['id']: r['cylinders'] for r in tree_records(tmp_path)} == expected


def test_the_octree_names_the_columns_of_the_info_lines(tmp_path):
    hierarchy = build([(INFO, None)], tmp_path)
    # Header line 2 of the fixture: tree columns, then (after ", ") segment columns.
    assert hierarchy['info_columns']['tree'][:3] == ['predinstance', 'height', 'crown_radius']
    assert hierarchy['info_columns']['segment'][:5] == ['x', 'y', 'z', 'radius', 'parent_id']
    assert len(hierarchy['info_columns']['segment']) == 14


def test_the_committed_fixture_octree_is_what_the_builder_writes(tmp_path):
    # The frontend tests read fixtures/rct-3753/octree: it must stay the builder's output.
    fresh = build([(INFO, MESH)], tmp_path, capacity=100)
    committed = json.loads((FIXTURE / 'octree' / 'hierarchy.json').read_text())
    fresh.pop('build_seconds')
    committed.pop('build_seconds')
    assert committed == fresh
    assert (FIXTURE / 'octree' / 'octree.bin').read_bytes() == (tmp_path / 'octree.bin').read_bytes()
