import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from parameters import Parameters  # noqa: E402
    import run  # noqa: E402
except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
    if exc.name not in {"pydantic_settings", "pydantic"}:
        raise
    Parameters = None
    run = None


class RunFilterTaskTests(unittest.TestCase):
    @unittest.skipIf(Parameters is None, "pydantic_settings is not installed")
    def test_filter_task_forwards_directory_parameters(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            params = Parameters(
                task="filter",
                input_dir=input_dir,
                output_dir=output_dir,
                buffer=12.5,
                filter_suffix="_kept",
                filter_output_extension=".laz",
                instance_dimension="FomaInstance",
                _cli_parse_args=False,
            )

            with mock.patch(
                "filter_buffer_instances.filter_buffer_instances_dir",
                return_value={"input_files": 2, "output_files": [output_dir / "a.laz"]},
            ) as filter_dir:
                run.run_filter_task(params)

            filter_dir.assert_called_once_with(
                input_dir=input_dir,
                output_dir=output_dir,
                buffer=12.5,
                suffix="_kept",
                instance_dimension="FomaInstance",
                output_extension=".laz",
            )

    @unittest.skipIf(Parameters is None, "pydantic_settings is not installed")
    def test_filter_task_requires_input_and_output_dirs(self):
        params = Parameters(task="filter", input_dir=None, output_dir=None, _cli_parse_args=False)

        with self.assertRaises(SystemExit):
            run.run_filter_task(params)

    @unittest.skipIf(Parameters is None, "pydantic_settings is not installed")
    def test_filter_task_rejects_in_place_overwrite_without_suffix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir)
            params = Parameters(
                task="filter",
                input_dir=input_dir,
                output_dir=input_dir,
                filter_suffix="",
                _cli_parse_args=False,
            )

            with self.assertRaises(SystemExit):
                run.run_filter_task(params)

    @unittest.skipIf(Parameters is None, "pydantic_settings is not installed")
    def test_filter_task_reports_invalid_output_extension(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir)
            params = Parameters(
                task="filter",
                input_dir=input_dir,
                output_dir=input_dir / "out",
                filter_output_extension=".copc.laz",
                _cli_parse_args=False,
            )

            with self.assertRaises(SystemExit):
                run.run_filter_task(params)


if __name__ == "__main__":
    unittest.main()
