"""Validate destinations and publish a complete set of staged files without overwrites.

Writers own content validation. This module owns the filesystem boundary: inputs
and existing outputs are never replaced, and an ordinary publication failure
removes every new output from that attempt. It does not promise crash atomicity.
"""
from __future__ import annotations

from contextlib import ExitStack
import os
from pathlib import Path
import shutil
import tempfile


def _identity(path: Path):
    try:
        stat = path.stat()
        return stat.st_dev, stat.st_ino
    except FileNotFoundError:
        return None


def validate_output_paths(inputs, outputs) -> None:
    """Reject aliases, duplicate destinations, and pre-existing output paths."""
    inputs, outputs = [Path(p) for p in inputs], [Path(p) for p in outputs]
    source_paths = {path.resolve() for path in inputs}
    source_ids = {_identity(path) for path in inputs} - {None}
    destinations, destination_ids = set(), set()
    for path in outputs:
        resolved, identity = path.resolve(), _identity(path)
        if resolved in source_paths or identity is not None and identity in source_ids:
            raise ValueError(f"Output aliases an input: {path}")
        if resolved in destinations or identity is not None and identity in destination_ids:
            raise ValueError(f"Duplicate output destination: {path}")
        destinations.add(resolved)
        if identity is not None:
            destination_ids.add(identity)
    for path in outputs:
        # exists() alone misses dangling symlinks, which must not be replaced.
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to replace existing output: {path}")


def publish_files(products, *, inputs=()) -> None:
    """Install validated (staged, destination) files, rolling back late failures.

    Staging must share the destination filesystem. A hard link installs a file
    atomically without replacing a destination created after preflight; staged
    files remain available until the complete set is installed successfully.
    """
    products = [(Path(source), Path(destination)) for source, destination in products]
    validate_output_paths(inputs, [destination for _, destination in products])
    for source, _ in products:
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"Staged output is not a regular file: {source}")
    installed = []
    try:
        for source, destination in products:
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(source, destination)
            installed.append((source, destination))
    except Exception:
        for source, destination in reversed(installed):
            # Do not delete a file another process substituted during failure.
            if _identity(destination) == _identity(source):
                destination.unlink()
        raise
    # The caller owns scratch cleanup. Keeping the staged links also makes
    # cleanup errors unable to turn a complete publication into a partial one.


class OutputTransaction:
    """Private per-filesystem output paths and one explicit publication step."""

    def __init__(self, outputs, *, inputs=(), sidecar_suffixes=(), keep_failed=False):
        self.outputs = [Path(path) for path in outputs]
        self.inputs = [Path(path) for path in inputs]
        self.sidecar_suffixes = tuple(sidecar_suffixes)
        self.keep_failed = keep_failed
        self._paths = {}
        self._directories = []
        self._stack = ExitStack()
        reserved = [path for output in self.outputs for path in
                    (output, *(Path(str(output) + suffix) for suffix in self.sidecar_suffixes))]
        validate_output_paths(self.inputs, reserved)

    def __enter__(self):
        parents = {}
        try:
            for output in self.outputs:
                parent = output.parent.resolve()
                if parent not in parents:
                    parent.mkdir(parents=True, exist_ok=True)
                    directory = Path(tempfile.mkdtemp(prefix=".smarttile-output-", dir=parent))
                    self._directories.append(directory)
                    parents[parent] = directory
                    self._stack.callback(shutil.rmtree, directory)
                self._paths[output] = parents[parent] / output.name
        except Exception:
            self._stack.close()
            raise
        return self

    def path(self, destination):
        return self._paths[Path(destination)]

    def commit(self):
        products = []
        for output in self.outputs:
            staged = self.path(output)
            if not staged.is_file() or not staged.stat().st_size:
                raise RuntimeError(f"Output was not created: {output}")
            for suffix in self.sidecar_suffixes:
                sidecar = Path(str(staged) + suffix)
                if sidecar.exists():
                    products.append((sidecar, Path(str(output) + suffix)))
            products.append((staged, output))
        publish_files(products, inputs=self.inputs)

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is not None and self.keep_failed:
            self._stack.pop_all()
            print("  Retained failed product scratch: " + ", ".join(map(str, self._directories)))
        else:
            self._stack.close()


class DirectoryTransaction:
    """Publish a complete task directory, allowing only an empty destination."""

    def __init__(self, destination, *, inputs=()):
        self.destination = Path(destination)
        self.inputs = [Path(path) for path in inputs]
        self._temporary = None
        self._validate()

    def _validate(self):
        destination = self.destination.resolve()
        for source in self.inputs:
            source = source.resolve()
            if source == destination or source in destination.parents or destination in source.parents:
                raise ValueError(f"Input and output directories overlap: {source}, {self.destination}")
        if self.destination.is_symlink():
            raise FileExistsError(f"Refusing a symlink output directory: {self.destination}")
        if self.destination.exists() and (
                not self.destination.is_dir() or any(self.destination.iterdir())):
            raise FileExistsError(f"Refusing to mix this run with existing output: {self.destination}")

    def __enter__(self):
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self._temporary = tempfile.TemporaryDirectory(prefix=".smarttile-task-", dir=self.destination.parent)
        self.path = Path(self._temporary.name) / self.destination.name
        self.path.mkdir()
        return self

    def commit(self):
        self._validate()
        # Replacing an existing empty directory is atomic; a concurrent nonempty
        # destination causes rename to fail without removing that destination.
        os.replace(self.path, self.destination)

    def __exit__(self, exc_type, exc, traceback):
        if self._temporary is not None:
            self._temporary.cleanup()
