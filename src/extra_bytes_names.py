"""Keep every ExtraBytes name through PDAL and untwine.

PDAL's LAS reader, which untwine also uses, changes some ExtraBytes names:
names of PDAL dimensions take PDAL's spelling ("height above ground" becomes
HeightAboveGround, PointID becomes PointId), characters other than letters,
digits, space and underscore become "_", and a name that matches a standard
field of the point format ("classification", "Withheld") is folded into that
field, so its values are lost. A float named like a colour field ("Red") is
written as that field, names that differ only in case are one dimension to
PDAL, and PDAL refuses non-ASCII names. A task whose inputs have such names
runs on staged copies whose ExtraBytes VLR carries placeholder names instead;
the placeholders get their source names back in every output and report before
the outputs are published. Only the fixed 32-byte name fields change, so no
point is rewritten.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List

import laspy
import numpy as np

PLACEHOLDER = "SmartTileEB{:03d}"
STAGING = ".smarttile-names-staging"
POINT_CLOUD_SUFFIXES = (".las", ".laz")
_VLR_HEADER = struct.Struct("<H16sHH32s")
_ENTRY_SIZE = 192
_NAME = slice(4, 36)
# Point format of the COPC written for a source format (COPC allows 6, 7 and 8).
_COPC_FORMAT = {0: 6, 1: 6, 2: 7, 3: 7, 4: 6, 5: 7, 6: 6, 7: 7, 8: 8, 9: 6, 10: 8}


def _extra_bytes_record(path: Path):
    """(offset, bytes) of the LASF_Spec record 4 payload among the regular VLRs, or None."""
    with Path(path).open("rb") as handle:
        handle.seek(94)
        header_size, point_offset, count = struct.unpack("<HII", handle.read(10))
        handle.seek(header_size)
        for _ in range(count):
            _, user_id, record_id, length, _ = _VLR_HEADER.unpack(handle.read(_VLR_HEADER.size))
            position = handle.tell()
            if position + length > point_offset:
                raise ValueError(f"{Path(path).name}: VLR extends beyond the point data")
            if user_id.rstrip(b"\0 ") == b"LASF_Spec" and record_id == 4:
                return position, handle.read(length)
            handle.seek(length, 1)
    return None


def raw_names(path: Path) -> List[bytes]:
    """The 32-byte name field of every ExtraBytes descriptor, NULs stripped."""
    record = _extra_bytes_record(path)
    if record is None:
        return []
    payload = record[1]
    return [payload[i:i + _ENTRY_SIZE][_NAME].rstrip(b"\0")
            for i in range(0, len(payload) - len(payload) % _ENTRY_SIZE, _ENTRY_SIZE)]


def rename(path: Path, names: Dict[bytes, bytes]) -> bool:
    """Rewrite ExtraBytes names in place (``names`` maps old to new); True if any changed."""
    record = _extra_bytes_record(path)
    if record is None:
        return False
    position, payload = record
    updated = bytearray(payload)
    for start in range(0, len(payload) - len(payload) % _ENTRY_SIZE, _ENTRY_SIZE):
        field = slice(start + _NAME.start, start + _NAME.stop)
        old = bytes(updated[field]).rstrip(b"\0")
        if old in names:
            new = names[old]
            if len(new) > 32:
                raise ValueError(f"ExtraBytes name longer than 32 bytes: {new!r}")
            updated[field] = new.ljust(32, b"\0")
    if bytes(updated) == payload:
        return False
    with Path(path).open("r+b") as handle:
        handle.seek(position)
        if handle.read(len(payload)) != payload:
            raise ValueError(f"{Path(path).name}: ExtraBytes VLR changed during rename")
        handle.seek(position)
        handle.write(bytes(updated))
    return True


def _pdal_dimensions(path: Path) -> List[str]:
    result = subprocess.run(["pdal", "info", "--schema", str(path)], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"pdal info failed: {result.stderr.strip()[:300]}")
    return [dim["name"] for dim in json.loads(result.stdout)["schema"]["dimensions"]]


@lru_cache(maxsize=None)
def _pdal_keeps(names: tuple, point_format: int, data_types: tuple) -> bool:
    """True if PDAL reads ExtraBytes dimensions ``names`` back unchanged, in order,
    beside the standard fields of ``point_format`` (no rename, no folding, no
    clash between names that differ only in case)."""
    with tempfile.TemporaryDirectory(prefix=".smarttile-name-probe-") as tmp:
        dims = {}
        placeholders = tuple(f"SmartTileProbe{i:03d}".encode() for i in range(len(names)))
        for label, fields in (("reference", placeholders), ("probe", names)):
            header = laspy.LasHeader(point_format=point_format, version="1.4")
            header.add_extra_dims([laspy.ExtraBytesParams(name=p.decode(), type=np.dtype(t))
                                   for p, t in zip(placeholders, data_types)])
            cloud = laspy.LasData(header)
            cloud.x = cloud.y = cloud.z = np.zeros(1)
            path = Path(tmp) / f"{label}.las"
            cloud.write(path)
            rename(path, dict(zip(placeholders, fields)))
            try:
                dims[label] = _pdal_dimensions(path)
            except RuntimeError:
                if label == "reference":
                    raise
                return False  # PDAL refuses a name ("Can't create dimension with invalid name")
    standard = len(dims["reference"]) - len(names)
    if dims["reference"][standard:] != [p.decode() for p in placeholders]:
        raise RuntimeError("PDAL changed SmartTile's placeholder ExtraBytes names")
    try:
        expected = dims["reference"][:standard] + [name.decode("utf-8") for name in names]
    except UnicodeDecodeError:
        return False
    return dims["probe"] == expected


@lru_cache(maxsize=None)
def _round_trip_keeps(names: tuple, point_format: int, data_types: tuple) -> bool:
    """True if a small file with these ExtraBytes comes out of untwine (COPC) and
    PDAL (LAS) with the same names and types. Catches what the schema alone does
    not show: a float "Red" written as the colour field, or two names that PDAL
    treats as one."""
    with tempfile.TemporaryDirectory(prefix=".smarttile-name-probe-") as tmp:
        tmp = Path(tmp)
        placeholders = tuple(f"SmartTileProbe{i:03d}".encode() for i in range(len(names)))
        header = laspy.LasHeader(point_format=point_format, version="1.4")
        header.add_extra_dims([laspy.ExtraBytesParams(name=p.decode(), type=np.dtype(t))
                               for p, t in zip(placeholders, data_types)])
        cloud = laspy.LasData(header)
        cloud.x, cloud.y, cloud.z = np.arange(3.0), np.arange(3.0), np.arange(3.0)
        for p, t in zip(placeholders, data_types):
            cloud[p.decode()] = np.array([1, 2, 3]).astype(t)
        source = tmp / "probe.las"
        cloud.write(source)
        rename(source, dict(zip(placeholders, names)))
        expected = sorted(zip(names, (np.dtype(t).str for t in data_types)))
        runs = [(["untwine", "-i", str(source), "-o", str(tmp / "u.copc.laz"), "--single_file"], tmp / "u.copc.laz"),
                (["pdal", "translate", str(source), str(tmp / "p.las"), "--writers.las.extra_dims=all",
                  "--writers.las.minor_version=4"], tmp / "p.las")]
        for command, output in runs:
            if shutil.which(command[0]) is None:
                continue
            if subprocess.run(command, capture_output=True, text=True).returncode != 0 or not output.exists():
                return False
            if sorted(zip(raw_names(output), _raw_types(output))) != expected:
                return False
    return True


def _raw_types(path: Path) -> List[str]:
    """numpy type string of every ExtraBytes descriptor, in VLR order."""
    record = _extra_bytes_record(path)
    if record is None:
        return []
    kinds = {1: "u1", 2: "i1", 3: "u2", 4: "i2", 5: "u4", 6: "i4", 7: "u8", 8: "i8", 9: "f4", 10: "f8"}
    payload = record[1]
    types = []
    for start in range(0, len(payload) - len(payload) % _ENTRY_SIZE, _ENTRY_SIZE):
        data_type = payload[start + 2]
        types.append(np.dtype("<" + kinds[data_type]).str if data_type in kinds else f"undocumented{data_type}")
    return types


def _key(name: str) -> str:
    """How PDAL and Potree compare names: case-insensitive; spaces, underscores and
    hyphens do not count ("gps-time", "GpsTime", "gps_time" are one field)."""
    return re.sub(r"[\s_\-]", "", name).lower()


# Standard LAS fields of every point format, in laspy's and PDAL's spelling.
STANDARD_FIELDS = {_key(name) for fmt in range(11)
                   for name in laspy.PointFormat(fmt).standard_dimension_names} | {
    _key(name) for name in ("Classification", "ScanChannel", "ScannerChannel", "KeyPoint", "GpsTime",
                            "ScanAngleRank", "ScanAngle", "PointSourceId", "UserData", "ReturnNumber",
                            "NumberOfReturns", "ScanDirectionFlag", "EdgeOfFlightLine", "Infrared",
                            "ClassFlags", "ClassificationFlags")}


@lru_cache(maxsize=None)
def _pdal_folds(name: bytes, point_format: int, data_type: str) -> bool:
    """True if PDAL folds an ExtraBytes ``name`` into one of its fields, so the
    extra column disappears from what PDAL reads (e.g. "Class")."""
    if shutil.which("pdal") is None:
        return False
    with tempfile.TemporaryDirectory(prefix=".smarttile-name-probe-") as tmp:
        counts = []
        for label, field in (("reference", b"SmartTileProbe000"), ("probe", name)):
            header = laspy.LasHeader(point_format=point_format, version="1.4")
            header.add_extra_dims([laspy.ExtraBytesParams(name="SmartTileProbe000", type=np.dtype(data_type))])
            cloud = laspy.LasData(header)
            cloud.x = cloud.y = cloud.z = np.zeros(1)
            path = Path(tmp) / f"{label}.las"
            cloud.write(path)
            rename(path, {b"SmartTileProbe000": field})
            try:
                counts.append(len(_pdal_dimensions(path)))
            except RuntimeError:
                return False
    return counts[1] < counts[0]


def clashes_with_standard_field(name: bytes, point_format: int, data_type: str) -> bool:
    """True if an ExtraBytes ``name`` is a standard LAS field (as PDAL and Potree
    compare names) or a name PDAL folds into one; such columns are published as
    "<name>_extra" so they never shadow the standard field."""
    try:
        text = name.decode("utf-8")
    except UnicodeDecodeError:
        return False
    if RESERVED.match(text):
        return False
    if _key(text) in STANDARD_FIELDS:
        return True
    return any(_pdal_folds(name, fmt, data_type) for fmt in {point_format, _COPC_FORMAT[point_format]})


def _extra_name(name: bytes, taken: set) -> bytes:
    """``name`` + "_extra", shortened to 32 bytes and numbered if the name is taken."""
    number = 1
    while True:
        suffix = b"_extra" if number == 1 else f"_extra{number}".encode()
        base = name[:32 - len(suffix)]
        while True:  # do not cut a UTF-8 character in half
            try:
                base.decode("utf-8")
                break
            except UnicodeDecodeError:
                base = base[:-1]
        candidate = base + suffix
        if candidate not in taken:
            return candidate
        number += 1


# Names SmartTile writes itself; they are plain identifiers and keep their spelling.
RESERVED = re.compile(r"^(OriginalPointIndex|Pred\w*|species_\w*|coloring_id\w*)$")
_TOKEN = re.compile(r"SmartTileEB\d{3}")


def names_to_stage(path: Path) -> List[bytes]:
    """ExtraBytes names of ``path`` that PDAL or untwine would rename, fold into a
    standard field, retype, refuse, or confuse with another name of the file."""
    if shutil.which("pdal") is None and shutil.which("untwine") is None:
        return []
    names = raw_names(path)
    if not names:
        return []
    with laspy.open(path) as reader:
        point_format = reader.header.point_format
        laspy_types = {dim.name: np.dtype(dim.dtype).base.str for dim in point_format.extra_dimensions}
    types = {}
    staged = set()
    for name in names:
        try:
            types[name] = laspy_types[name.decode("utf-8")]
        except (UnicodeDecodeError, KeyError):
            staged.add(name)
    # The originals and the COPC files made from them (formats 6-8) are both read by PDAL.
    formats = sorted({point_format.id, _COPC_FORMAT[point_format.id]})
    for name in names:
        if name in staged:
            continue
        for fmt in formats:
            if (shutil.which("pdal") and not _pdal_keeps((name,), fmt, (types[name],))) \
                    or not _round_trip_keeps((name,), fmt, (types[name],)):
                staged.add(name)
                break
    # Columns named like standard fields are published as "<name>_extra".
    for name in names:
        if name not in staged and name in types and clashes_with_standard_field(name, point_format.id, types[name]):
            staged.add(name)
    # PDAL treats names that differ only in case as one dimension.
    by_case = {}
    for name in names:
        by_case.setdefault(name.lower(), []).append(name)
    for group in by_case.values():
        if len(group) > 1:
            staged.update(name for name in group if not RESERVED.match(name.decode("utf-8", "replace")))
    kept = tuple(name for name in names if name not in staged)
    if kept and not all(_round_trip_keeps(kept, fmt, tuple(types[n] for n in kept)) for fmt in formats):
        staged.update(name for name in kept if not RESERVED.match(name.decode("utf-8", "replace")))
        kept = tuple(name for name in names if name not in staged)
        if kept and not all(_round_trip_keeps(kept, fmt, tuple(types[n] for n in kept)) for fmt in formats):
            raise ValueError(f"{Path(path).name}: PDAL does not keep the ExtraBytes names "
                             f"{[n.decode('utf-8', 'replace') for n in kept]}")
    return [name for name in names if name in staged]


def _point_format(path: Path) -> int:
    with Path(path).open("rb") as handle:
        handle.seek(104)
        return handle.read(1)[0] & 0x3F


def _point_clouds(folder: Path) -> List[Path]:
    return sorted(p for p in Path(folder).iterdir()
                  if p.suffix.lower() in POINT_CLOUD_SUFFIXES and not p.name.startswith("."))


class StagedNames:
    """Placeholders for the ExtraBytes names that PDAL would change, and staged
    copies of the task's input folders that use them.

    ``problem_folders`` decide which names need placeholders; the same mapping
    applies to every folder of the task (originals and prediction collections
    alike), so a name compared across inputs stays one name. ``restore`` puts
    the source names back into the task's outputs and reports."""

    def __init__(self, problem_folders: Iterable[Path], work_root: Path, also: Iterable[Path] = ()):
        self.names: Dict[bytes, bytes] = {}
        self.final: Dict[bytes, bytes] = {}
        self.staged: Dict[Path, Path] = {}
        problem_folders = [Path(f) for f in problem_folders if f is not None]
        folders = list(dict.fromkeys(problem_folders + [Path(f) for f in also if f is not None]))
        problems = sorted({name for folder in problem_folders if folder.is_dir()
                           for path in _point_clouds(folder) for name in names_to_stage(path)})
        if not problems:
            return
        taken = {name for folder in folders if folder.is_dir()
                 for path in _point_clouds(folder) for name in raw_names(path)}
        index = 0
        for name in problems:
            while PLACEHOLDER.format(index).encode() in taken:
                index += 1
            self.names[name] = PLACEHOLDER.format(index).encode()
            index += 1
        # Output names: the source name, or "<name>_extra" for a standard field's name.
        clashing = {name for folder in problem_folders if folder.is_dir() for path in _point_clouds(folder)
                    for name, data_type in zip(raw_names(path), _raw_types(path))
                    if name in self.names and clashes_with_standard_field(name, _point_format(path), data_type)}
        used = set(taken)
        for name in problems:
            self.final[name] = _extra_name(name, used) if name in clashing else name
            used.add(self.final[name])
        self.root = Path(work_root) / STAGING
        self.root.mkdir(parents=True, exist_ok=True)
        for number, folder in enumerate(folders):
            if not folder.is_dir():
                continue
            clouds = [path for path in _point_clouds(folder) if set(raw_names(path)) & set(self.names)]
            if not clouds:
                continue
            staged = self.root / f"{number}_{folder.name}"
            staged.mkdir()
            for entry in sorted(folder.iterdir()):
                target = staged / entry.name
                if entry in clouds:
                    shutil.copyfile(entry, target)
                    rename(target, self.names)
                else:
                    os.symlink(entry.absolute(), target)
            self.staged[folder] = staged
        print(f"  ExtraBytes names staged for PDAL: "
              f"{', '.join(repr(n.decode('utf-8', 'replace')) for n in self.names)}")
        for name, final in self.renamed().items():
            print(f"  ExtraBytes {name!r} is a standard field name; published as {final!r}")

    def renamed(self) -> Dict[str, str]:
        """Source name -> published name for the columns renamed to "<name>_extra"."""
        return {name.decode("utf-8", "replace"): final.decode("utf-8", "replace")
                for name, final in self.final.items() if final != name}

    def __bool__(self) -> bool:
        return bool(self.names)

    def folder(self, folder):
        return self.staged.get(Path(folder), Path(folder)) if folder is not None else folder

    def name(self, name):
        """A dimension name as the staged inputs spell it (for names given on the command line)."""
        if name is None:
            return name
        return self.names.get(name.encode("utf-8"), name.encode("utf-8")).decode("utf-8")

    def restore(self, *roots: Path) -> None:
        """Source names back in every LAS/LAZ/COPC and PLY below ``roots``; JSON,
        CSV and text reports get the source names, input folders and checksums."""
        if not self.names:
            return
        back = {placeholder: self.final.get(name, name) for name, placeholder in self.names.items()}
        text = {placeholder.decode(): name.decode("utf-8", "replace") for placeholder, name in back.items()}
        folders = {str(staged.absolute()): str(folder.absolute()) for folder, staged in self.staged.items()}
        digests, reports, seen = {}, [], set()
        for root in roots:
            if root is None or not Path(root).exists():
                continue
            for path in sorted(Path(root).rglob("*")):
                if STAGING in path.parts or not path.is_file() or path.is_symlink() or path in seen:
                    continue
                seen.add(path)
                suffix = path.suffix.lower()
                if suffix in POINT_CLOUD_SUFFIXES and set(raw_names(path)) & set(back):
                    before = _sha256(path)
                    rename(path, back)
                    digests[before] = _sha256(path)
                elif suffix == ".ply":
                    before = _sha256(path)
                    if _restore_ply(path, text):
                        digests[before] = _sha256(path)
                elif suffix in (".json", ".csv", ".txt"):
                    reports.append(path)
        for report in reports:
            try:
                content = report.read_text()
            except UnicodeDecodeError:
                continue
            if report.suffix.lower() == ".json":
                try:
                    data = json.loads(content)
                except json.JSONDecodeError:
                    continue
                fixed = _replace_strings(data, text, folders, digests)
                if fixed != data:
                    report.write_text(json.dumps(fixed, indent=2) + "\n")
            else:
                fixed = _replace_text(content, text, folders, digests)
                if fixed != content:
                    report.write_text(fixed)
        if self.renamed():
            for root in roots:
                if root is not None and Path(root).is_dir():
                    (Path(root) / "extra_bytes_names.json").write_text(json.dumps(
                        {"renamed_extra_bytes": self.renamed(),
                         "reason": "ExtraBytes named like a standard LAS field are published as <name>_extra"},
                        indent=2, ensure_ascii=False) + "\n")
                    break

    def cleanup(self) -> None:
        if self.names:
            shutil.rmtree(self.root, ignore_errors=True)


def _restore_ply(path: Path, text: Dict[str, str]) -> bool:
    """Source names in a PLY header's property lines. PLY names cannot contain
    whitespace, so whitespace in a source name becomes "_" there."""
    with Path(path).open("rb") as handle:
        head = handle.read(1 << 16)
    end = head.find(b"end_header")
    if end < 0 or not _TOKEN.search(head[:end].decode("latin-1")):
        return False
    stop = head.index(b"\n", end) + 1
    header = head[:stop].decode("latin-1")
    fixed = _TOKEN.sub(lambda m: "_".join(text.get(m.group(0), m.group(0)).split()), header)
    temporary = Path(path).with_name(f".{Path(path).name}.names")
    with Path(path).open("rb") as source, temporary.open("wb") as target:
        source.seek(stop)
        target.write(fixed.encode("utf-8"))
        shutil.copyfileobj(source, target, 1 << 24)
    temporary.replace(path)
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def _replace_text(value: str, names: Dict[str, str], folders: Dict[str, str], digests: Dict[str, str]) -> str:
    for staged, folder in folders.items():
        value = value.replace(staged, folder)
    for old, new in digests.items():
        value = value.replace(old, new)
    return _TOKEN.sub(lambda m: names.get(m.group(0), m.group(0)), value)


def _replace_strings(value, names, folders, digests):
    if isinstance(value, dict):
        return {_replace_strings(k, names, folders, digests): _replace_strings(v, names, folders, digests)
                for k, v in value.items()}
    if isinstance(value, list):
        return [_replace_strings(v, names, folders, digests) for v in value]
    if isinstance(value, str):
        return _replace_text(value, names, folders, digests)
    return value
