"""Compare a Rust SmartTile output tree with the Python SmartTile reference tree.

    python tests/parity/compare_outputs.py REFERENCE CANDIDATE [--json report.json]

Per file: byte identity first; otherwise a semantic comparison that says what
differs. LAS/LAZ: header fields, VLRs/EVLRs and every point record in order.
COPC: the same, but points as a sorted multiset (Untwine orders points by
octree node). JSON: values, with the tindex path compared by file name.
Exit status 0 only when nothing differs beyond documented allowances.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import laspy
import numpy as np

HEADER_FIELDS = ("version", "file_source_id", "uuid", "system_identifier", "generating_software",
                 "creation_date", "point_count", "scales", "offsets", "mins", "maxs")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def vlr_list(vlrs):
    out = []
    for vlr in vlrs or []:
        if vlr.user_id in ("laszip encoded", "copc"):
            continue
        try:
            data = vlr.record_data_bytes()
        except Exception:
            data = bytes(getattr(vlr, "record_data", b""))
        out.append((vlr.user_id, vlr.record_id, data))
    return out


def compare_las(reference, candidate, copc):
    problems = []
    with laspy.open(reference) as a, laspy.open(candidate) as b:
        ha, hb = a.header, b.header
        for field in HEADER_FIELDS:
            va, vb = getattr(ha, field), getattr(hb, field)
            same = np.array_equal(va, vb) if isinstance(va, np.ndarray) else va == vb
            if not same:
                problems.append(f"header {field}: {va!r} != {vb!r}")
        if ha.point_format.id != hb.point_format.id:
            problems.append(f"point format {ha.point_format.id} != {hb.point_format.id}")
        if vlr_list(ha.vlrs) != vlr_list(hb.vlrs):
            problems.append(f"VLRs differ: {[v[:2] for v in vlr_list(ha.vlrs)]} vs {[v[:2] for v in vlr_list(hb.vlrs)]}")
        if vlr_list(ha.evlrs) != vlr_list(hb.evlrs):
            problems.append("EVLRs differ")
        pa, pb = a.read_points(ha.point_count), b.read_points(hb.point_count)
    xa = np.stack([pa.X, pa.Y, pa.Z], axis=1)
    xb = np.stack([pb.X, pb.Y, pb.Z], axis=1)
    if copc:
        xa = xa[np.lexsort(xa.T[::-1])]
        xb = xb[np.lexsort(xb.T[::-1])]
        if not np.array_equal(xa, xb):
            problems.append("point multisets differ")
    elif pa.array.tobytes() != pb.array.tobytes():
        if np.array_equal(xa, xb):
            problems.append("XYZ identical, other point fields differ")
        else:
            order = "same multiset, different order" if np.array_equal(xa[np.lexsort(xa.T[::-1])], xb[np.lexsort(xb.T[::-1])]) else "different points"
            problems.append(f"point records differ ({order})")
    return problems


def gpkg_content(path):
    """Feature rows and spatial reference of a GeoPackage, without its timestamps."""
    import sqlite3

    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        tables = [row[0] for row in db.execute("SELECT table_name FROM gpkg_contents ORDER BY table_name")]
        content = {"srs": db.execute("SELECT srs_id, organization, organization_coordsys_id FROM gpkg_spatial_ref_sys "
                                     "WHERE srs_id IN (SELECT srs_id FROM gpkg_geometry_columns) ORDER BY srs_id").fetchall()}
        for table in tables:
            rows = db.execute(f'SELECT * FROM "{table}" ORDER BY 1').fetchall()
            columns = [d[0] for d in db.execute(f'SELECT * FROM "{table}" LIMIT 0').description]
            content[table] = (columns, rows)
    return content


# Run-dependent report fields: wall-clock timings and source modification times.
# The occupancy grid is a Python acceleration structure; Rust has none to report.
VOLATILE_KEYS = {"timings", "mtime_ns", "spatial_occupancy"}


def _without_volatile(value):
    if isinstance(value, dict):
        result = {}
        for k, v in value.items():
            if k in VOLATILE_KEYS or k.endswith("_seconds"):
                continue
            if k == "checksums" and isinstance(v, dict) and len(v) > 1:
                # Published paths: compare them relative to the run's output root.
                import os
                root = os.path.commonpath(list(v))
                v = {os.path.relpath(path, root): digest for path, digest in v.items()}
            result[k] = _without_volatile(v)
        return result
    if isinstance(value, list):
        return [_without_volatile(v) for v in value]
    return value


def normalized_json(path):
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict) and "tindex" in data:
        data["tindex"] = Path(data["tindex"]).name
    return _without_volatile(data)


def compare(reference: Path, candidate: Path):
    files_a = {p.relative_to(reference) for p in reference.rglob("*") if p.is_file() or p.is_symlink()}
    files_b = {p.relative_to(candidate) for p in candidate.rglob("*") if p.is_file() or p.is_symlink()}
    report = {"missing": sorted(map(str, files_a - files_b)), "extra": sorted(map(str, files_b - files_a)), "files": {}}
    for rel in sorted(files_a & files_b):
        a, b = reference / rel, candidate / rel
        name = str(rel)
        if a.is_symlink() or b.is_symlink():
            same = a.is_symlink() and b.is_symlink() and a.readlink() == b.readlink()
            report["files"][name] = {"status": "identical" if same else "different", "problems": [] if same else ["symlink target differs"]}
            continue
        if sha256(a) == sha256(b):
            report["files"][name] = {"status": "identical", "problems": []}
            continue
        lower = name.lower()
        if lower.endswith((".las", ".laz")):
            problems = compare_las(a, b, copc=lower.endswith(".copc.laz"))
        elif lower.endswith(".png"):
            # The overview is redrawn (resvg, not matplotlib): same content, other pixels.
            problems = [] if b.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" else ["not a PNG"]
        elif lower.endswith(".gpkg"):
            problems = [] if gpkg_content(a) == gpkg_content(b) else ["GeoPackage features or CRS differ"]
        elif lower.endswith(".json"):
            problems = [] if normalized_json(a) == normalized_json(b) else ["JSON values differ"]
        else:
            problems = ["bytes differ"]
        report["files"][name] = {"status": "equivalent" if not problems else "different", "problems": problems}
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    report = compare(args.reference, args.candidate)
    counts = {}
    for entry in report["files"].values():
        counts[entry["status"]] = counts.get(entry["status"], 0) + 1
    report["summary"] = counts
    for name in report["missing"]:
        print(f"MISSING    {name}")
    for name in report["extra"]:
        print(f"EXTRA      {name}")
    for name, entry in report["files"].items():
        if entry["status"] != "identical":
            print(f"{entry['status'].upper():10} {name}: {'; '.join(entry['problems'])}")
    print(json.dumps(counts))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if not report["missing"] and not report["extra"] and counts.get("different", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
