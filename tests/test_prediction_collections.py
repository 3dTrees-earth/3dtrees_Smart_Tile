import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prediction_collections import prediction_collection_files  # noqa: E402


class PredictionCollectionTests(unittest.TestCase):
    def test_prediction_collection_files_prefer_copc_twins(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            collection = Path(tmpdir) / "collection"
            collection.mkdir()
            (collection / "source.laz").write_text("placeholder")
            (collection / "source.copc.laz").write_text("placeholder")
            (collection / "other.las").write_text("placeholder")
            files = [path.name for path in prediction_collection_files(collection)]
        self.assertEqual(files, ["other.las", "source.copc.laz"])


if __name__ == "__main__":
    unittest.main()
