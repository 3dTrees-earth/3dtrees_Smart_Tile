import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from copy import deepcopy
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
    validate_vector_conversion,
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
    las.header.vlrs[:] = VLRList(
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

    def test_scalarization_preserves_evlrs_and_compresses_transport(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root / "source.las", root / "scalar.laz"
            _write_vector_las(source)
            cloud = laspy.read(source)
            cloud.header.evlrs = VLRList([
                laspy.VLR(user_id="test", record_id=9, record_data=b"extended source metadata")])
            cloud.write(source)
            scalarize_vector_extra_bytes(source, output, chunk_size=2)
            result = laspy.read(output)
            self.assertTrue(result.header.are_points_compressed)
            self.assertEqual([(v.user_id, v.record_id, v.record_data_bytes()) for v in result.header.evlrs],
                             [("test", 9, b"extended source metadata")])

    def test_component_names_avoid_scalar_fields_after_the_vector(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cloud = laspy.LasData(laspy.LasHeader(point_format=3, version="1.4"))
            cloud.x, cloud.y, cloud.z = [1], [2], [3]
            cloud.add_extra_dim(laspy.ExtraBytesParams(name="Vector", type="3i2"))
            cloud.add_extra_dim(laspy.ExtraBytesParams(name="Vector__0", type="u2"))
            cloud.Vector = [[4, 5, 6]]
            cloud.Vector__0 = [99]
            source, output = root / "source.las", root / "scalar.laz"
            cloud.write(source)
            _, schema = scalarize_vector_extra_bytes(source, output)
            self.assertNotEqual(schema["dimensions"][0]["component_names"][0], "Vector__0")
            self.assertEqual(laspy.read(output).Vector__0.tolist(), [99])
            np.testing.assert_array_equal(reconstruct_vector_values(output)["Vector"], [[4, 5, 6]])

    def test_long_descriptions_and_fallback_names_fit_las_field(self):
        for description, name in [("d" * 32, "Vector"), ("", "V" * 32)]:
            with self.subTest(description=description), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                cloud = laspy.LasData(laspy.LasHeader(point_format=3, version="1.4"))
                cloud.x, cloud.y, cloud.z = [1], [2], [3]
                cloud.add_extra_dim(laspy.ExtraBytesParams(name=name, type="3i2", description=description))
                cloud[name] = [[4, 5, 6]]
                source, output = root / "source.las", root / "scalar.laz"
                cloud.write(source)
                _, schema = scalarize_vector_extra_bytes(source, output)
                self.assertEqual(schema["dimensions"][0]["description"], description)
                result = laspy.read(output)
                for i, component in enumerate(schema["dimensions"][0]["component_names"]):
                    actual = result.point_format.dimension_by_name(component).description
                    self.assertLessEqual(len(actual.encode("utf-8")), 32)
                    self.assertTrue(actual.endswith(f"[{i}]"))
                np.testing.assert_array_equal(reconstruct_vector_values(output)[name], [[4, 5, 6]])

    def test_round_trip_accepts_point_reordering_and_coordinate_reencoding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, scalar, output = root / "source.las", root / "scalar.laz", root / "out.laz"
            _write_vector_las(source)
            scalarize_vector_extra_bytes(source, scalar)
            cloud = laspy.read(scalar)
            cloud.points = cloud.points[[2, 0, 1]]
            cloud.change_scaling(scales=[.001] * 3, offsets=[1000, 2000, 3000])
            cloud.write(output)
            valid, message = validate_vector_conversion(scalar, output, chunk_size=1)
            self.assertTrue(valid, message)

    def test_round_trip_rejects_corruption_and_wrong_point_associations(self):
        for change in ("value", "swapped", "missing", "scale", "dtype", "no_data", "count", "duplicate"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source, output = root / "source.las", root / "out.copc.laz"
                _write_vector_las(source)

                def broken_writer(scalar_source, candidate, preserve_extra_dims=False):
                    cloud = laspy.read(scalar_source)
                    descriptors = {d.format_name(): deepcopy(d) for v in cloud.header.vlrs
                                   for d in getattr(v, "extra_bytes_structs", ())}
                    if change == "value":
                        cloud.Vector__1 = cloud.Vector__1 + 1
                    elif change == "swapped":
                        cloud.Vector__1 = np.asarray(cloud.Vector__1)[::-1]
                    elif change == "missing":
                        cloud.remove_extra_dim("Vector__1")
                    elif change in ("scale", "dtype", "no_data"):
                        dim = cloud.point_format.dimension_by_name("Vector__1")
                        values = cloud.points.array["Vector__1"].copy()
                        cloud.remove_extra_dim("Vector__1")
                        cloud.add_extra_dim(laspy.ExtraBytesParams(
                            name="Vector__1", type="i4" if change == "dtype" else dim.dtype,
                            scales=[.5] if change == "scale" else [.25], offsets=[20.],
                            no_data=[-1] if change == "no_data" else [-998]))
                        cloud.points.array["Vector__1"] = values
                    elif change == "count":
                        cloud.points = cloud.points[:2]
                    else:
                        cloud.points = cloud.points[[0, 0, 2]]
                    # laspy rebuilds every descriptor when one field changes;
                    # restore unrelated descriptors to isolate this corruption.
                    for vlr in cloud.header.vlrs:
                        if hasattr(vlr, "extra_bytes_structs"):
                            vlr.extra_bytes_structs = [
                                d if d.format_name() == "Vector__1" else descriptors[d.format_name()]
                                for d in vlr.extra_bytes_structs]
                    cloud.write(candidate)
                    return True

                expected = {
                    "value": "XYZ associations changed", "swapped": "XYZ associations changed",
                    "missing": "Missing vector component: Vector__1",
                    "scale": "Changed vector component scales: Vector__1",
                    "dtype": "Changed vector component dtype: Vector__1",
                    "no_data": "Changed vector component no_data: Vector__1",
                    "count": "changed the point count", "duplicate": "XYZ associations changed",
                }[change]
                with mock.patch.object(tile_copc, "_convert_laz_to_copc_impl", side_effect=broken_writer), mock.patch("builtins.print") as report:
                    self.assertFalse(tile_copc.convert_laz_to_copc(source, output, preserve_extra_dims=True))
                self.assertIn(expected, report.call_args.args[0])
                self.assertFalse(output.exists())
                self.assertFalse(vector_schema_sidecar_path(output).exists())
                self.assertFalse(any(root.glob(".smarttile-vector-extra-*")))

    def test_failed_validation_preserves_existing_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root / "source.las", root / "out.copc.laz"
            _write_vector_las(source)
            output.write_bytes(b"existing product")
            vector_schema_sidecar_path(output).write_bytes(b"existing schema")

            def broken_writer(scalar_source, candidate, preserve_extra_dims=False):
                cloud = laspy.read(scalar_source)
                cloud.Vector__1 = cloud.Vector__1 + 1
                cloud.write(candidate)
                return True

            with mock.patch.object(tile_copc, "_convert_laz_to_copc_impl", side_effect=broken_writer):
                self.assertFalse(tile_copc.convert_laz_to_copc(source, output, preserve_extra_dims=True))
            self.assertEqual(output.read_bytes(), b"existing product")
            self.assertEqual(vector_schema_sidecar_path(output).read_bytes(), b"existing schema")

    def test_failed_cloud_publication_restores_previous_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root / "source.las", root / "out.copc.laz"
            _write_vector_las(source)
            output.write_bytes(b"previous cloud")
            vector_schema_sidecar_path(output).write_bytes(b"previous schema")
            original_replace = Path.replace

            def fail_cloud_rename(path, destination):
                if path.parent.name == "candidate" and path.name == output.name:
                    raise OSError("simulated publication failure")
                return original_replace(path, destination)

            def writer(scalar_source, candidate, preserve_extra_dims=False):
                shutil.copyfile(scalar_source, candidate)
                return True

            with mock.patch.object(tile_copc, "_convert_laz_to_copc_impl", side_effect=writer), mock.patch.object(Path, "replace", fail_cloud_rename):
                with self.assertRaisesRegex(OSError, "simulated publication failure"):
                    tile_copc.convert_laz_to_copc(source, output, preserve_extra_dims=True)
            self.assertEqual(output.read_bytes(), b"previous cloud")
            self.assertEqual(vector_schema_sidecar_path(output).read_bytes(), b"previous schema")

    def test_round_trip_preserves_duplicate_rows_and_float_nan_no_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, scalar, output = root / "source.las", root / "scalar.laz", root / "out.laz"
            cloud = laspy.LasData(laspy.LasHeader(point_format=3, version="1.4"))
            cloud.x, cloud.y, cloud.z = [1, 1, 2], [0, 0, 0], [0, 0, 0]
            cloud.add_extra_dim(laspy.ExtraBytesParams(name="Vector", type="3f4", no_data=[np.nan] * 3))
            cloud.Vector = [[1, np.nan, 3], [1, np.nan, 3], [4, 5, 6]]
            cloud.write(source)
            scalarize_vector_extra_bytes(source, scalar)
            result = laspy.read(scalar)
            result.points = result.points[[2, 1, 0]]
            result.write(output)
            valid, message = validate_vector_conversion(scalar, output, chunk_size=1)
            self.assertTrue(valid, message)


if __name__ == "__main__":
    unittest.main()
