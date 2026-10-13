"""Output safety across input aliases, writer failures, and late publication failures."""
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from output_publication import OutputTransaction, publish_files, validate_output_paths
from main_create_merged_file import create_prod_merged_files, create_chunked_prod_merged_files_for_resolution


@pytest.mark.parametrize("alias", ["exact", "symlink", "hardlink"])
def test_full_product_cannot_replace_an_input_alias(tmp_path, alias):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    source = inputs / ("merged.laz" if alias == "exact" else "source.laz")
    source.write_bytes(b"irreplaceable input")
    output_dir = inputs if alias == "exact" else tmp_path / "outputs"
    output_dir.mkdir(exist_ok=True)
    output = output_dir / "merged.laz"
    if alias == "symlink":
        output.symlink_to(source)
    elif alias == "hardlink":
        os.link(source, output)

    with pytest.raises(ValueError, match="aliases an input"):
        create_prod_merged_files(inputs, output_dir, "full", "laz", .01, .1)
    assert source.read_bytes() == b"irreplaceable input"
    assert output.read_bytes() == b"irreplaceable input"
    assert not list(output_dir.glob(".smarttile-output-*"))


def test_duplicate_destinations_through_a_symlink_directory_are_rejected(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="Duplicate output"):
        validate_output_paths([], [real / "result.laz", alias / "result.laz"])


def test_duplicate_hardlink_destinations_are_rejected(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"previous")
    os.link(a, b)
    with pytest.raises(ValueError, match="Duplicate output"):
        validate_output_paths([], [a, b])
    assert a.read_bytes() == b.read_bytes() == b"previous"


def test_chunked_duplicate_outputs_are_rejected_before_work(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("chunk work ran before destination validation")
    monkeypatch.setattr("main_create_merged_file._prod_merged_chunk_bounds", unexpected)
    target = tmp_path / "result.laz"
    with pytest.raises(ValueError, match="Duplicate output"):
        create_chunked_prod_merged_files_for_resolution(
            [tmp_path / "source.copc.laz"], [(target, "laz"), (target, "laz")], .1, 1)


def test_late_publication_failure_rolls_back_earlier_files(tmp_path, monkeypatch):
    first, second = tmp_path / "first.laz", tmp_path / "second.laz"
    link = os.link

    def fail_second(source, destination):
        if Path(destination) == second:
            raise OSError("injected second publication failure")
        return link(source, destination)

    with pytest.raises(OSError, match="second publication"):
        with OutputTransaction([first, second]) as publication:
            publication.path(first).write_bytes(b"validated first")
            publication.path(second).write_bytes(b"validated second")
            monkeypatch.setattr("output_publication.os.link", fail_second)
            publication.commit()
    assert not first.exists() and not second.exists()
    assert not list(tmp_path.glob(".smarttile-output-*"))


def test_a_destination_created_during_publication_is_never_overwritten(tmp_path, monkeypatch):
    first, second = tmp_path / "first.laz", tmp_path / "second.laz"
    link = os.link

    def concurrent_writer(source, destination):
        if Path(destination) == second:
            second.write_bytes(b"another writer's result")
        return link(source, destination)

    with pytest.raises(FileExistsError):
        with OutputTransaction([first, second]) as publication:
            publication.path(first).write_bytes(b"validated first")
            publication.path(second).write_bytes(b"validated second")
            monkeypatch.setattr("output_publication.os.link", concurrent_writer)
            publication.commit()
    assert not first.exists()
    assert second.read_bytes() == b"another writer's result"


def test_sidecar_and_cloud_publish_together(tmp_path, monkeypatch):
    output = tmp_path / "result.copc.laz"
    suffix = ".vector-extra-bytes.json"
    link = os.link

    def fail_cloud(source, destination):
        if Path(destination) == output:
            raise OSError("cloud publication failed")
        return link(source, destination)

    with pytest.raises(OSError, match="cloud publication"):
        with OutputTransaction([output], sidecar_suffixes=(suffix,)) as publication:
            staged = publication.path(output)
            staged.write_bytes(b"validated cloud")
            Path(str(staged) + suffix).write_text('{"validated": true}')
            monkeypatch.setattr("output_publication.os.link", fail_cloud)
            publication.commit()
    assert not output.exists() and not Path(str(output) + suffix).exists()


def test_failed_full_product_validation_publishes_no_selected_formats(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    source = inputs / "source.laz"
    source.write_bytes(b"original input")
    outputs = tmp_path / "outputs"
    calls = []

    def write_product(inputs, path, fmt, **kwargs):
        calls.append(fmt)
        path.write_bytes(b"new product")

    def validate(path, fmt, *args):
        if fmt == "ply":
            raise ValueError("invalid second-format schema")

    monkeypatch.setattr("full_resolution_merge.create_full_resolution_product", write_product)
    monkeypatch.setattr("main_create_merged_file._finalize_product", validate)
    with pytest.raises(ValueError, match="second-format schema"):
        create_prod_merged_files(inputs, outputs, "full", "laz,ply", .01, .1)
    assert calls == ["laz", "ply"]
    assert source.read_bytes() == b"original input"
    assert not (outputs / "merged.laz").exists()
    assert not (outputs / "merged.ply").exists()
    assert not list(outputs.glob(".smarttile-output-*"))


def test_existing_tile_destination_is_rejected_before_any_publication(tmp_path):
    staged = tmp_path / "stage"
    staged.mkdir()
    a, b = staged / "a.laz", staged / "b.laz"
    a.write_bytes(b"new a")
    b.write_bytes(b"new b")
    final_a, final_b = tmp_path / "a.laz", tmp_path / "b.laz"
    final_a.write_bytes(b"previous a")
    final_b.mkdir()
    with pytest.raises(FileExistsError):
        publish_files([(a, final_a), (b, final_b)])
    assert final_a.read_bytes() == b"previous a"
    assert final_b.is_dir()
