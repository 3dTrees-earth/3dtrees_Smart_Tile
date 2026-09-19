import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import laspy
import numpy as np
from laspy.vlrs.vlrlist import VLRList


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import tile_copc  # noqa: E402
from vector_extra_bytes import (  # noqa: E402
    VECTOR_SCHEMA_RECORD_ID,
    VECTOR_SCHEMA_USER_ID,
    read_vector_schema,
    reconstruct_vector_values,
    scalarize_vector_extra_bytes,
    vector_schema_sidecar_path,
)


def _write_vector_las(path: Path) -> np.ndarray:
    header = laspy.LasHeader(point_format=3, version="1.4")
    las = laspy.LasData(header)
    las.x = [1.0, 2.0, 3.0]
    las.y = [4.0, 5.0, 6.0]
    las.z = [7.0, 8.0, 9.0]
    las.add_extra_dim(laspy.ExtraBytesParams(name="Vector__0", type=np.uint8))
    las.Vector__0 = [90, 91, 92]
    las.add_extra_dim(
        laspy.ExtraBytesParams(
            name="Vector",
            type=np.dtype((np.int16, 3)),
            description="three component test vector",
            scales=[0.5, 0.25, 2.0],
            offsets=[10.0, 20.0, 30.0],
            no_data=[-999, -998, -997],
        )
    )
    values = np.array(
        [[11.0, 21.0, 32.0], [12.0, 22.0, 34.0], [13.0, 23.0, 36.0]],
        dtype=np.float64,
    )
    las.Vector = values
    las.write(path)
    return values


def _drop_schema_vlr(source: Path, output: Path) -> None:
    las = laspy.read(source)
    las.header.vlrs = VLRList(
        [
            vlr
            for vlr in las.header.vlrs
            if not (
                vlr.user_id == VECTOR_SCHEMA_USER_ID
                and vlr.record_id == VECTOR_SCHEMA_RECORD_ID
            )
        ]
    )
    las.write(output)


class VectorExtraBytesTests(unittest.TestCase):
    def test_scalarization_is_deterministic_reversible_and_writes_both_metadata_copies(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.las"
            output = root / "scalarized.laz"
            expected = _write_vector_las(source)

            scalarized, schema = scalarize_vector_extra_bytes(source, output, chunk_size=2)

            self.assertEqual(scalarized, output)
            self.assertIsNotNone(schema)
            self.assertTrue(vector_schema_sidecar_path(output).exists())
            embedded = read_vector_schema(output)
            self.assertEqual(embedded, schema)
            vector = schema["dimensions"][0]
            self.assertEqual(vector["original_name"], "Vector")
            self.assertEqual(vector["component_count"], 3)
            self.assertEqual(vector["no_data"], [-999, -998, -997])
            self.assertEqual(vector["scales"], [0.5, 0.25, 2.0])
            self.assertEqual(vector["offsets"], [10.0, 20.0, 30.0])
            self.assertNotEqual(vector["component_names"][0], "Vector__0")
            self.assertEqual(vector["component_names"][1:], ["Vector__1", "Vector__2"])

            with laspy.open(output) as reader:
                dims = list(reader.header.point_format.extra_dimension_names)
                self.assertNotIn("Vector", dims)
                self.assertIn("Vector__0", dims)
                for component in vector["component_names"]:
                    self.assertIn(component, dims)
            reconstructed = reconstruct_vector_values(output)["Vector"]
            np.testing.assert_array_equal(reconstructed, expected)

    def test_copc_wrapper_embeds_schema_and_sidecar_after_writer_drops_vlr(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.las"
            output = root / "output.copc.laz"
            _write_vector_las(source)

            def fake_convert(scalar_source, output_copc, preserve_extra_dims=False):
                self.assertTrue(preserve_extra_dims)
                _drop_schema_vlr(Path(scalar_source), output_copc)
                return True

            with mock.patch.object(
                tile_copc,
                "_convert_laz_to_copc_impl",
                side_effect=fake_convert,
            ):
                self.assertTrue(
                    tile_copc.convert_laz_to_copc(
                        source,
                        output,
                        preserve_extra_dims=True,
                    )
                )

            self.assertIsNotNone(read_vector_schema(output))
            self.assertTrue(vector_schema_sidecar_path(output).exists())
            np.testing.assert_array_equal(
                reconstruct_vector_values(output)["Vector"],
                np.array(
                    [[11.0, 21.0, 32.0], [12.0, 22.0, 34.0], [13.0, 23.0, 36.0]],
                    dtype=np.float64,
                ),
            )


if __name__ == "__main__":
    unittest.main()
