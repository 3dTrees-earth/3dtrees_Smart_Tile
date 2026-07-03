#!/usr/bin/env python3
"""Regression tests for single-tile remap execution."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import main_remap  # noqa: E402


class TestSingleTileRemapInline(unittest.TestCase):
    def test_single_tile_runs_without_process_pool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            target = root / "target"
            output = root / "output"
            source.mkdir()
            target.mkdir()

            source_file = source / "c00_r00_segmented.laz"
            target_file = target / "c00_r00.laz"
            matches = [(source_file, target_file, "c00_r00")]

            with (
                mock.patch("main_remap.find_matching_files", return_value=matches),
                mock.patch(
                    "main_remap._remap_worker_item",
                    return_value=("c00_r00", True, "ok", 123),
                ) as worker,
                mock.patch("main_remap.ProcessPoolExecutor") as pool,
            ):
                result = main_remap.remap_all_tiles(
                    source,
                    target,
                    output,
                    num_workers=20,
                )

            self.assertEqual(output, result)
            worker.assert_called_once()
            pool.assert_not_called()

    def test_multiple_tiles_use_process_pool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            target = root / "target"
            output = root / "output"
            source.mkdir()
            target.mkdir()

            matches = [
                (source / "c00_r00_segmented.laz", target / "c00_r00.laz", "c00_r00"),
                (source / "c00_r01_segmented.laz", target / "c00_r01.laz", "c00_r01"),
            ]

            executor = mock.Mock()
            executor.__enter__ = mock.Mock(return_value=executor)
            executor.__exit__ = mock.Mock(return_value=None)
            executor.map.return_value = [
                ("c00_r00", True, "ok", 123),
                ("c00_r01", True, "ok", 456),
            ]

            with (
                mock.patch("main_remap.find_matching_files", return_value=matches),
                mock.patch("main_remap.ProcessPoolExecutor", return_value=executor) as pool,
            ):
                result = main_remap.remap_all_tiles(
                    source,
                    target,
                    output,
                    num_workers=20,
                )

            self.assertEqual(output, result)
            pool.assert_called_once_with(max_workers=2)
            executor.map.assert_called_once()

    def test_multiple_tiles_with_one_worker_run_serially(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            target = root / "target"
            output = root / "output"
            source.mkdir()
            target.mkdir()

            matches = [
                (source / "c00_r00_segmented.laz", target / "c00_r00.laz", "c00_r00"),
                (source / "c00_r01_segmented.laz", target / "c00_r01.laz", "c00_r01"),
            ]

            with (
                mock.patch("main_remap.find_matching_files", return_value=matches),
                mock.patch(
                    "main_remap._remap_worker_item",
                    side_effect=[
                        ("c00_r00", True, "ok", 123),
                        ("c00_r01", True, "ok", 456),
                    ],
                ) as worker,
                mock.patch("main_remap.ProcessPoolExecutor") as pool,
            ):
                result = main_remap.remap_all_tiles(
                    source,
                    target,
                    output,
                    num_workers=1,
                )

            self.assertEqual(output, result)
            self.assertEqual(worker.call_count, 2)
            pool.assert_not_called()


if __name__ == "__main__":
    unittest.main()
