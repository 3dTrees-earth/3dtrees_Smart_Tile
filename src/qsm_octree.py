"""The QSM octree the viewer reads beside RayCloudTools files per original: the
only file the viewer adds to the original outputs (3DT-2626). SmartTile writes
it with `merge_qsm` (QSMs for an existing segmentation); the code is the
results viewer's `scripts/results_view/qsm_octree.py`, unchanged, so both build
identical octrees.

Reads the original `_trees_info.txt` (and `_trees_mesh.ply` where there is one)
as SmartTile stores them and writes into `--out`:

- `octree.bin`:
  - the octree's nodes, one after another (18 B per cylinder), so a node is one
    Range request. Every cylinder sits in one node: a node keeps the most
    important cylinders (radius × length: trunks first) that fall in its cube,
    up to `--capacity`; the rest go to its eight children by midpoint. Starts
    are relative to the node's cube (u16 over its size grown by half on every
    side), so precision grows with depth:

        x, y, z      u16  start, q × (2 × size / 65535) + (node min − size / 2)
        oct_u, oct_v i16  direction, octahedral (× 32767)
        length       u16  mm
        radius       u16  0.1 mm
        tree_id      u32

  - then the tree table, one record per tree, sorted by id (layout in
    hierarchy.json): the tree's attributes as treeinfo wrote them, its base and
    QSM volume, its line in `_trees_info.txt` (byte offset and length, newline
    included) and its vertex and face ranges in `_trees_mesh.ply`.
- `hierarchy.json`: the nodes (name, level, cube, byte range, count, radii,
  child mask), the tree table's place and layout, the origin (absolute) the
  cubes and bases are relative to, and the names and sizes of the originals the
  ranges point into.

A cylinder runs from a segment's parent to the segment, with the segment's
radius (treetools convention). The originals are only read.

    python3 qsm_octree.py --info 11172_trees_info.txt --mesh 11172_trees_mesh.ply --out octree
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

RECORD = np.dtype([('x', '<u2'), ('y', '<u2'), ('z', '<u2'), ('ou', '<i2'), ('ov', '<i2'), ('length', '<u2'),
                   ('radius', '<u2'), ('id', '<u4')])
# The tree table: (name, struct format), little-endian, no padding.
TREE_FIELDS = [
    ('id', 'I'), ('base_x', 'f'), ('base_y', 'f'), ('base_z', 'f'),
    ('height', 'f'), ('dbh', 'f'), ('crown_radius', 'f'), ('volume', 'f'), ('cylinders', 'I'),
    ('info_offset', 'Q'), ('info_length', 'I'),
    ('vertex_offset', 'Q'), ('vertex_length', 'I'), ('face_offset', 'Q'), ('face_length', 'I'), ('vertex_base', 'I'),
]
# The rest of treeinfo's per-tree values, and the tree totals its root segment carries instead of its own
# (total volume, maximum diameter, mean dominance, branch angle and children): NaN where a file lacks one.
TREE_VALUES = ['dimension', 'monocotal', 'bend', 'branch_slope']
ROOT_VALUES = ['volume', 'diameter', 'dominance', 'angle', 'children']
TREE_FIELDS += [(f'tree.{name}', 'f') for name in TREE_VALUES] + [(f'root.{name}', 'f') for name in ROOT_VALUES]
TREE = np.dtype([(name, '<' + {'I': 'u4', 'f': 'f4', 'Q': 'u8'}[kind]) for name, kind in TREE_FIELDS])


def octahedral(d: np.ndarray) -> np.ndarray:
    n = d / np.abs(d).sum(axis=1, keepdims=True)
    u, v = n[:, 0].copy(), n[:, 1].copy()
    below = n[:, 2] < 0
    u[below], v[below] = ((1 - np.abs(n[below, 1])) * np.sign(n[below, 0] + 1e-30),
                          (1 - np.abs(n[below, 0])) * np.sign(n[below, 1] + 1e-30))
    return np.stack([u, v], axis=1)


def read_info(path: Path):
    """Columns, trees (attributes, line range) and cylinders from an original `_trees_info.txt`.

    Header line 2 names the per-tree columns, then (after ", ") the per-segment
    columns; each line holds a tree's attributes, then its segments, ", " apart.
    The tree id is the instance column where there is one (SmartTile's
    `predinstance`, a `PredInstance_*` name), else the line number.
    """
    with path.open('rb') as f:
        f.readline()
        columns = f.readline().decode('ascii').strip()
        offset = f.tell()
        tree_part, segment_part = columns.split(', ', 1)
        tree_cols = tree_part.split(',')
        seg_cols = segment_part.split(',')
        id_col = next((i for i, c in enumerate(tree_cols) if 'instance' in c.lower()), None)
        col = {c: i for i, c in enumerate(tree_cols)}
        s = {c: i for i, c in enumerate(seg_cols)}
        trees, starts, ends, radii, owners = [], [], [], [], []
        for row, line in enumerate(f, start=1):
            values = np.fromstring(line.decode('ascii'), sep=',')
            attrs, segments = values[:len(tree_cols)], values[len(tree_cols):].reshape(-1, len(seg_cols))
            tree_id = int(np.rint(attrs[id_col])) if id_col is not None else row
            xyz = segments[:, [s['x'], s['y'], s['z']]]
            parent = np.round(segments[:, s['parent_id']]).astype(np.int64)
            child = np.flatnonzero(parent >= 0)
            start, end, r = xyz[parent[child]], xyz[child], segments[child, s['radius']]
            length = np.linalg.norm(end - start, axis=1)
            trees.append({
                'id': tree_id, 'base': xyz[0],
                'height': attrs[col['height']] if 'height' in col else np.nan,
                'dbh': attrs[col['DBH']] if 'DBH' in col else np.nan,
                'crown_radius': attrs[col['crown_radius']] if 'crown_radius' in col else np.nan,
                'volume': float(np.sum(np.pi * r ** 2 * length)), 'cylinders': len(child),
                'info_offset': offset, 'info_length': len(line),
                'values': [attrs[col[n]] if n in col else np.nan for n in TREE_VALUES]
                + [segments[0, s[n]] if n in s else np.nan for n in ROOT_VALUES],
            })
            starts.append(start)
            ends.append(end)
            radii.append(r)
            owners.append(np.full(len(child), tree_id, dtype=np.uint32))
            offset += len(line)
    cat = (lambda parts, shape: np.concatenate(parts) if parts else np.zeros(shape))
    columns = {'tree': tree_cols, 'segment': seg_cols}
    return columns, trees, cat(starts, (0, 3)), cat(ends, (0, 3)), cat(radii, (0,)), cat(owners, (0,)).astype(np.uint32)


PLY_TYPES = {'char': 'i1', 'uchar': 'u1', 'short': '<i2', 'ushort': '<u2', 'int': '<i4', 'uint': '<u4',
             'float': '<f4', 'double': '<f8', 'int8': 'i1', 'uint8': 'u1', 'int32': '<i4', 'uint32': '<u4',
             'float32': '<f4', 'float64': '<f8'}


def index_mesh(path: Path) -> dict[int, tuple[int, int, int, int, int]]:
    """Each tree's (vertex offset, vertex bytes, face offset, face bytes, first vertex) in a `_trees_mesh.ply`.

    The mesh must be binary little-endian triangles with a per-face `tree_id`
    (RCT's mesh as SmartTile stores it), each tree's faces one run, and each
    tree's vertices one run that only its faces use; anything else is refused.
    """
    with path.open('rb') as f:
        header = []
        while (line := f.readline().decode('ascii').strip()) != 'end_header':
            header.append(line)
        start = f.tell()
    if 'format binary_little_endian 1.0' not in header:
        raise ValueError(f'{path.name}: not a binary little-endian PLY')
    elements, current = [], None
    for line in header:
        word = line.split()
        if word[0] == 'element':
            current = {'name': word[1], 'count': int(word[2]), 'props': []}
            elements.append(current)
        elif word[0] == 'property':
            current['props'].append(word[1:])
    vertex, face = elements[0], elements[1]
    if vertex['name'] != 'vertex' or face['name'] != 'face':
        raise ValueError(f'{path.name}: expected vertex then face elements')
    vertex_size = sum(np.dtype(PLY_TYPES[p[0]]).itemsize for p in vertex['props'])
    if [p[0] for p in face['props']] != ['list', 'uint'] or face['props'][1][1] != 'tree_id':
        raise ValueError(f'{path.name}: faces need a vertex list and a per-face tree_id')
    count_type, index_type = PLY_TYPES[face['props'][0][1]], PLY_TYPES[face['props'][0][2]]
    face_dtype = np.dtype([('n', count_type), ('i', index_type, 3), ('tree', '<u4')])
    face_start = start + vertex['count'] * vertex_size
    faces = np.memmap(path, mode='r', offset=face_start, dtype=face_dtype, shape=(face['count'],))
    if face['count'] and not np.all(faces['n'] == 3):
        raise ValueError(f'{path.name}: only triangles are supported')
    tree = np.asarray(faces['tree'])
    runs = np.flatnonzero(np.diff(tree)) + 1
    run_start = np.concatenate([[0], runs]) if len(tree) else np.zeros(0, dtype=np.int64)
    run_end = np.concatenate([runs, [len(tree)]]) if len(tree) else np.zeros(0, dtype=np.int64)
    run_tree = tree[run_start]
    if len(np.unique(run_tree)) != len(run_tree):
        raise ValueError(f'{path.name}: a tree\'s faces are not one run')
    index = np.asarray(faces['i'])
    low = np.minimum.reduceat(index.min(axis=1), run_start) if len(tree) else np.zeros(0)
    high = np.maximum.reduceat(index.max(axis=1), run_start) if len(tree) else np.zeros(0)
    order = np.argsort(low)
    if np.any(low[order][1:] <= high[order][:-1]):
        raise ValueError(f'{path.name}: trees share vertices')
    return {int(t): (start + int(lo) * vertex_size, int(hi - lo + 1) * vertex_size,
                     face_start + int(a) * face_dtype.itemsize, int(b - a) * face_dtype.itemsize, int(lo))
            for t, a, b, lo, hi in zip(run_tree, run_start, run_end, low, high)}


def build(info: Path, out: Path, mesh: Path | None = None, capacity: int = 12000) -> dict:
    started = time.time()
    columns, trees, p0, p1, r, ids = read_info(Path(info))
    axis = p1 - p0
    length = np.linalg.norm(axis, axis=1)
    keep = length > 0
    p0, axis, length, r, ids = p0[keep], axis[keep], length[keep], r[keep], ids[keep]
    points = np.concatenate([p0, p0 + axis, np.array([t['base'] for t in trees]).reshape(-1, 3)])
    origin = np.floor(points.min(axis=0)) if len(points) else np.zeros(3)
    p0 = p0 - origin
    mid = p0 + axis / 2
    order = np.argsort(-(r * length), kind='stable')
    lo = np.minimum(p0, p0 + axis).min(axis=0) if len(p0) else np.zeros(3)
    hi = np.maximum(p0, p0 + axis).max(axis=0) if len(p0) else np.ones(3)
    size = float((hi - lo).max()) * 1.001 or 1.0
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    nodes, chunks, offset = [], [], 0
    queue = [('r', 0, lo, size, order)]
    while queue:
        name, level, nmin, nsize, cand = queue.pop(0)
        mine, rest = cand[:capacity], cand[capacity:]
        rec = np.zeros(len(mine), dtype=RECORD)
        # Starts are quantized over the node's cube grown by half its size on every side (a cylinder's start can lie
        # outside the cube: its midpoint decides the node), so the step is 2 × size / 65535.
        step = 2 * nsize / 65535
        local = p0[mine] - (nmin - nsize / 2)
        rec['x'] = np.clip(np.round(local[:, 0] / step), 0, 65535)
        rec['y'] = np.clip(np.round(local[:, 1] / step), 0, 65535)
        rec['z'] = np.clip(np.round(local[:, 2] / step), 0, 65535)
        oct_ = np.round(octahedral(axis[mine] / length[mine, None]) * 32767).astype(np.int16)
        rec['ou'], rec['ov'] = oct_[:, 0], oct_[:, 1]
        rec['length'] = np.clip(np.round(length[mine] * 1000), 1, 65535)
        rec['radius'] = np.clip(np.round(r[mine] * 10000), 1, 65535)
        rec['id'] = ids[mine]
        raw = rec.tobytes()
        chunks.append(raw)
        mask = 0
        children = []
        if len(rest):
            half = nsize / 2
            octant = ((mid[rest] - nmin) >= half).astype(int)
            code = octant[:, 0] * 4 + octant[:, 1] * 2 + octant[:, 2]
            for k in range(8):
                sel = rest[code == k]  # keeps importance order
                if len(sel):
                    mask |= 1 << k
                    cmin = nmin + half * np.array([(k >> 2) & 1, (k >> 1) & 1, k & 1])
                    children.append((f'{name}{k}', level + 1, cmin, half, sel))
        nodes.append({
            'name': name, 'level': level, 'min': [round(float(v), 4) for v in nmin], 'size': round(nsize, 4),
            'offset': offset, 'bytes': len(raw), 'count': len(mine),
            'max_radius': round(float(r[mine].max()), 4) if len(mine) else 0.0,
            'min_radius': round(float(r[mine].min()), 4) if len(mine) else 0.0,
            'children': mask,
        })
        offset += len(raw)
        queue.extend(children)

    ranges = index_mesh(Path(mesh)) if mesh else {}
    table = np.zeros(len(trees), dtype=TREE)
    for i, t in enumerate(sorted(trees, key=lambda t: t['id'])):
        base = t['base'] - origin
        table[i] = (t['id'], base[0], base[1], base[2], t['height'], t['dbh'], t['crown_radius'], t['volume'],
                    t['cylinders'], t['info_offset'], t['info_length'], *ranges.get(t['id'], (0, 0, 0, 0, 0)), *t['values'])
    tree_bytes = table.tobytes()
    (out / 'octree.bin').write_bytes(b''.join(chunks) + tree_bytes)
    originals = {'info': {'name': Path(info).name, 'size': Path(info).stat().st_size}}
    if mesh:
        originals['mesh'] = {'name': Path(mesh).name, 'size': Path(mesh).stat().st_size}
    hierarchy = {
        'contract': '3dtrees-qsm-octree-v1', 'record_bytes': RECORD.itemsize,
        'record': [[n, RECORD.fields[n][0].str] for n in RECORD.names],
        'origin': [float(v) for v in origin], 'capacity': capacity,
        'originals': originals,
        'info_columns': columns,
        'nodes': nodes,
        'trees': {'offset': offset, 'bytes': len(tree_bytes), 'count': len(trees), 'record_bytes': TREE.itemsize,
                  'record': [[name, kind] for name, kind in TREE_FIELDS]},
        'totals': {'nodes': len(nodes), 'cylinders': int(sum(n['count'] for n in nodes)), 'trees': len(trees),
                   'bytes': offset + len(tree_bytes), 'levels': int(max(n['level'] for n in nodes)) + 1,
                   'root_step_mm': round(2 * size / 65535 * 1000, 2)},
        'build_seconds': round(time.time() - started, 1),
    }
    (out / 'hierarchy.json').write_text(json.dumps(hierarchy))
    return hierarchy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--info', type=Path, required=True)
    parser.add_argument('--mesh', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--capacity', type=int, default=12000)
    args = parser.parse_args()
    hierarchy = build(args.info, args.out, mesh=args.mesh, capacity=args.capacity)
    print(json.dumps(hierarchy['totals']), hierarchy['build_seconds'], 's')


if __name__ == '__main__':
    main()
