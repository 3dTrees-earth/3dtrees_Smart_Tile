import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import run  # noqa: E402

try:
    from parameters import Parameters  # noqa: E402
except Exception:  # pragma: no cover
    Parameters = None


class RunCreateMergedFileTaskTests(unittest.TestCase):
    @unittest.skipIf(Parameters is None, "pydantic_settings is not installed")
    def test_create_merged_file_task_passes_chunk_worker_budget(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            original_with_predictions = root / "original_with_predictions"
            output_dir = root / "out"
            original_with_predictions.mkdir()
            params = Parameters(
                task="create_merged_file",
                original_with_predictions_dir=original_with_predictions,
                output_dir=output_dir,
                merged_resolutions="10cm",
                merged_output_formats="laz,copc.laz",
                workers=6,
                num_spatial_chunks=12,
                _cli_parse_args=False,
            )

            with mock.patch(
                "main_create_merged_file.create_prod_merged_files",
                return_value=[output_dir / "prod_merged_10cm.laz"],
            ) as create_prod:
                run.run_create_merged_file_task(params)

            create_prod.assert_called_once()
            _, kwargs = create_prod.call_args
            self.assertEqual(kwargs["original_with_predictions_dir"], original_with_predictions)
            self.assertEqual(kwargs["output_dir"], output_dir)
            self.assertEqual(kwargs["num_spatial_chunks"], 12)
            self.assertEqual(kwargs["chunk_workers"], 12)


if __name__ == "__main__":
    unittest.main()
