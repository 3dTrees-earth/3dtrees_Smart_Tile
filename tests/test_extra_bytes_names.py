"""ExtraBytes names survive PDAL and untwine (extra_bytes_names)."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import laspy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import run  # noqa: E402
from extra_bytes_names import STAGING, StagedNames, names_to_stage, raw_names, rename  # noqa: E402
from parameters import Parameters  # noqa: E402
from test_merge_stages import write_cloud  # noqa: E402

# Renamed by PDAL (its own dimension names and punctuation), or folded into a
# standard field of the point format; the last three PDAL keeps as they are.
CHANGED = ["Red", "height above ground", "PointID", "normal x", "Coord. Z", "Roughness (0.5)",
           "C2C absolute distances[<2]", "reflectance", "Withheld", "classification",
           "Ünïcode name", "a name that is exactly 32 bytes!"]
KEPT = ["Pulse width", "plain_name", "Amplitude"]
# Named like standard LAS fields: published as "<name>_extra".
EXTRA = {"Red": "Red_extra", "Withheld": "Withheld_extra", "classification": "classification_extra"}


def published(name):
    return EXTRA.get(name, name)
NEEDS_PDAL = unittest.skipIf(shutil.which("pdal") is None, "needs PDAL")


def write_named_cloud(path, names, point_format=6, n=4000, seed=0):
    """A cloud with one float32 ExtraBytes dimension per name (written under
    placeholders, then renamed in the VLR, since laspy refuses some names)."""
    header = laspy.LasHeader(point_format=point_format, version="1.4")
    header.scales = np.array([0.001] * 3)
    header.offsets = np.zeros(3)
    header.add_extra_dims([laspy.ExtraBytesParams(name=f"tmp{i}", type=np.float32) for i in range(len(names))])
    cloud = laspy.LasData(header)
    rng = np.random.default_rng(seed)
    cloud.x, cloud.y, cloud.z = rng.uniform(0, 30, n), rng.uniform(0, 30, n), rng.uniform(0, 20, n)
    cloud.intensity = rng.integers(0, 1000, n).astype(np.uint16)
    values = {}
    for i, name in enumerate(names):
        values[name] = rng.uniform(-50, 50, n).astype(np.float32)
        cloud[f"tmp{i}"] = values[name]
    cloud.write(path)
    rename(path, {f"tmp{i}".encode(): name.encode() for i, name in enumerate(names)})
    return values


def read_by_name(path):
    """Points sorted by position, with each ExtraBytes column under its raw VLR name.

    laspy cannot open a format 6-10 file with an ExtraBytes "classification"
    (it clashes with the standard field), so a copy with neutral names is read."""
    names = raw_names(path)
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / Path(path).name
        shutil.copyfile(path, copy)
        rename(copy, {name: f"col{i}".encode() for i, name in enumerate(names)})
        cloud = laspy.read(copy)
    order = np.lexsort((cloud.Z, cloud.Y, cloud.X)) if len(cloud.points) else np.arange(0)
    return {raw.decode("utf-8"): np.asarray(cloud[f"col{i}"])[order] for i, raw in enumerate(names)}, cloud, order


class RenameTests(unittest.TestCase):
    def test_rename_rewrites_only_the_name_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.las"
            write_named_cloud(path, ["plain_name", "other"])
            before = path.read_bytes()
            self.assertTrue(rename(path, {b"plain_name": b"height above ground"}))
            after = path.read_bytes()
            self.assertEqual(len(before), len(after))
            self.assertEqual(raw_names(path), [b"height above ground", b"other"])
            changed = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
            self.assertLess(max(changed) - min(changed), 32)
            self.assertFalse(rename(path, {b"missing": b"x"}))


@NEEDS_PDAL
class NamesToStageTests(unittest.TestCase):
    def test_pdal_changes_are_detected_and_plain_names_are_left_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.las"
            write_named_cloud(path, KEPT)
            self.assertEqual(names_to_stage(path), [])
            for name in CHANGED:
                write_named_cloud(path, [name] + KEPT)
                self.assertEqual(names_to_stage(path), [name.encode()], name)

    def test_names_that_differ_only_in_case_are_staged(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.las"
            write_named_cloud(path, ["treeID", "TreeID", "plain_name"])
            self.assertEqual(names_to_stage(path), [b"treeID", b"TreeID"])

    def test_names_folded_only_in_the_copc_point_format_are_staged(self):
        # Format 1 has no scanner channel; its COPC (format 6) has, and PDAL folds
        # an ExtraBytes "ScanChannel" into that field.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.las"
            write_named_cloud(path, ["ScanChannel", "plain_name"], point_format=1)
            self.assertEqual(names_to_stage(path), [b"ScanChannel"])

    def test_staging_keeps_the_input_untouched_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            originals = root / "originals"
            originals.mkdir()
            write_named_cloud(originals / "a.las", ["PointID", "plain_name"])
            write_named_cloud(originals / "b.las", ["plain_name"])
            before = (originals / "a.las").read_bytes()
            names = StagedNames([originals], root / "out")
            staged = names.folder(originals)
            self.assertNotEqual(staged, originals)
            self.assertEqual(raw_names(staged / "a.las"), [b"SmartTileEB000", b"plain_name"])
            self.assertTrue((staged / "b.las").is_symlink())
            self.assertEqual(names.name("PointID"), "SmartTileEB000")
            self.assertEqual(names.name("plain_name"), "plain_name")
            self.assertEqual((originals / "a.las").read_bytes(), before)
            names.cleanup()
            self.assertFalse((root / "out" / STAGING).exists())


@NEEDS_PDAL
class TaskNameTests(unittest.TestCase):
    def assert_same_names_and_values(self, source_values, output, full):
        values, cloud, _ = read_by_name(output)
        values.pop("OriginalPointIndex", None)  # added to full-resolution products
        self.assertEqual(sorted(values), sorted(published(n) for n in source_values))
        if full:
            for name, expected in source_values.items():
                np.testing.assert_array_equal(np.sort(values[published(name)]), np.sort(expected), err_msg=name)

    def test_create_merged_file_keeps_every_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            originals = root / "originals"
            originals.mkdir()
            source_values = write_named_cloud(originals / "plot.las", CHANGED + KEPT)
            params = Parameters(task="create_merged_file", original_with_predictions_dir=originals,
                                output_dir=root / "out", merged_resolutions="full,0.1",
                                merged_output_formats="copc.laz,laz", collection_dir=root / "collections",
                                workers=2, _cli_parse_args=False)
            run.run_create_merged_file_task(params)
            products = sorted([*(root / "out").glob("merged.*"), *(root / "out").glob("prod_merged_*")])
            self.assertEqual(len(products), 4)
            for product in products:
                self.assert_same_names_and_values(source_values, product, full=product.name.startswith("merged."))
            # Published collections are hard links of the renamed products.
            for linked in (root / "collections").rglob("*.laz"):
                self.assertEqual(sorted(set(read_by_name(linked)[0]) - {"OriginalPointIndex"}),
                                 sorted(published(n) for n in source_values))
            renamed = json.loads((root / "out" / "extra_bytes_names.json").read_text())["renamed_extra_bytes"]
            self.assertEqual(renamed, EXTRA)
            self.assertFalse((root / "out" / STAGING).exists())
            self.assertEqual(raw_names(originals / "plot.las"), [n.encode() for n in source_values])
            for report in (root / "out").rglob("*.json"):
                self.assertNotIn("SmartTileEB", report.read_text())

    def test_remap_keeps_every_name_in_originals_and_products(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions, originals, baseline = (root / n for n in ("predictions", "originals", "baseline"))
            for folder in (predictions, originals, baseline):
                folder.mkdir()
            xs = np.linspace(0, 0.1, 50)
            write_cloud(predictions / "tile.las", xs, np.ones(50, np.uint32), np.ones(50, np.uint8))
            write_cloud(baseline / "tile.las", xs)
            header = laspy.LasHeader(point_format=3, version="1.2")
            header.scales = np.array([0.000001] * 3)
            header.add_extra_dims([laspy.ExtraBytesParams(name=f"tmp{i}", type=np.float64)
                                   for i in range(len(CHANGED + KEPT))])
            cloud = laspy.LasData(header)
            cloud.x, cloud.y, cloud.z = xs, np.zeros(50), np.zeros(50)
            source_values = {}
            for i, name in enumerate(CHANGED + KEPT):
                source_values[name] = np.arange(50, dtype=np.float64) * (i + 1)
                cloud[f"tmp{i}"] = source_values[name]
            cloud.write(originals / "original.las")
            rename(originals / "original.las", {f"tmp{i}".encode(): n.encode() for i, n in enumerate(CHANGED + KEPT)})
            params = Parameters(task="remap", segmented_folders=str(predictions), original_laz_input_dir=originals,
                                original_laz_output_dir=root / "out/original_with_predictions",
                                baseline_1cm_folders=str(baseline), workers=1,
                                transfer_original_dims_to_merged=True, merged_resolutions="full",
                                merged_output_formats="copc.laz", _cli_parse_args=False)
            run.run_remap_task(params)
            enriched = root / "out/original_with_predictions/original.las"
            names = [n.decode() for n in raw_names(enriched)]
            self.assertEqual(names[:len(source_values)], [published(n) for n in source_values])
            products = list((root / "out").glob("merged.*"))
            self.assertTrue(products)
            for product in products:
                values = read_by_name(product)[0]
                for name, expected in source_values.items():
                    np.testing.assert_array_equal(np.sort(values[published(name)]), np.sort(expected), err_msg=name)
            coverage = json.loads((root / "out/original_with_predictions_coverage.json").read_text())
            self.assertNotIn("SmartTileEB", json.dumps(coverage))
            self.assertFalse((root / "out" / STAGING).exists())


@NEEDS_PDAL
class CoexistingNameTests(unittest.TestCase):
    """A renamed and a folded name beside names SmartTile itself uses."""

    def write_original(self, path, xs):
        header = laspy.LasHeader(point_format=3, version="1.2")
        header.scales = np.array([0.000001] * 3)
        header.add_extra_dims([laspy.ExtraBytesParams(name=n, type=t) for n, t in
                               (("tmp0", np.float64), ("tmp1", np.int32), ("treeID", np.int32))])
        cloud = laspy.LasData(header)
        cloud.x, cloud.y, cloud.z = xs, np.zeros(len(xs)), np.zeros(len(xs))
        cloud["tmp0"] = np.arange(len(xs)) * 0.5
        cloud["tmp1"] = np.arange(len(xs)) % 7
        cloud["treeID"] = np.arange(len(xs)) // 10
        cloud.write(path)
        rename(path, {b"tmp0": b"height above ground", b"tmp1": b"Withheld"})

    def check_outputs(self, root, original_output):
        enriched = original_output / "original.las"
        names = [n.decode() for n in raw_names(enriched)]
        for expected in ("height above ground", "Withheld_extra", "treeID", "PredInstance_SAT"):
            self.assertIn(expected, names)
        self.assertNotIn("Withheld", names)
        values = read_by_name(enriched)[0]
        np.testing.assert_array_equal(values["height above ground"], np.arange(50) * 0.5)
        np.testing.assert_array_equal(values["Withheld_extra"], np.arange(50) % 7)
        np.testing.assert_array_equal(values["treeID"], np.arange(50) // 10)
        for product in [*root.rglob("merged.*"), *root.rglob("prod_merged_*")]:
            product_names = [n.decode() for n in raw_names(product)]
            for expected in ("height above ground", "Withheld_extra", "treeID", "PredInstance_SAT"):
                self.assertIn(expected, product_names, product.name)
        for report in [*root.rglob("*.json"), *root.rglob("*.csv"), *root.rglob("*.txt")]:
            self.assertNotIn("SmartTileEB", report.read_text(), report.name)
        self.assertFalse(any(STAGING in p.parts for p in root.rglob("*")))

    def test_remap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions, originals, baseline = (root / n for n in ("predictions", "originals", "baseline"))
            for folder in (predictions, originals, baseline):
                folder.mkdir()
            xs = np.linspace(0, 0.1, 50)
            write_cloud(predictions / "tile.las", xs, np.ones(50, np.uint32), np.ones(50, np.uint8),
                        instance="PredInstance_SAT")
            write_cloud(baseline / "tile.las", xs)
            self.write_original(originals / "original.las", xs)
            params = Parameters(task="remap", segmented_folders=str(predictions), original_laz_input_dir=originals,
                                original_laz_output_dir=root / "out/original_with_predictions",
                                baseline_1cm_folders=str(baseline), workers=1,
                                remap_dims="PredInstance_SAT,PredSemantic_SAT",
                                transfer_original_dims_to_merged=True, merged_resolutions="full,0.1",
                                merged_output_formats="copc.laz,laz", collection_dir=root / "collections",
                                _cli_parse_args=False)
            run.run_remap_task(params)
            self.check_outputs(root / "out", root / "out/original_with_predictions")

    def test_merge(self):
        from test_remap_first_pipeline import layout
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, target, originals = (root / n for n in ("source", "target", "originals"))
            for folder in (source, target, originals):
                folder.mkdir()
            xs = np.linspace(0, 0.1, 50)
            write_cloud(source / "original.las", xs, np.full(50, 7, np.uint32), np.full(50, 2, np.uint8),
                        instance="PredInstance_SAT")
            write_cloud(target / "original.las", xs)
            self.write_original(originals / "original.las", xs)
            params = Parameters(task="merge", output_tiles_folder=root / "output/tiles",
                                tile_bounds_json=layout(root, 1), workers=1,
                                subsampled_10cm_folder=source, subsampled_target_folder=target,
                                original_laz_input_dir=originals, instance_dimension="PredInstance_SAT",
                                transfer_original_dims_to_merged=True, merged_resolutions="full,0.1",
                                merged_output_formats="copc.laz,laz", _cli_parse_args=False)
            run.run_merge_task(params)
            self.check_outputs(root / "output", root / "output/original_with_predictions")


class ExtraNameTests(unittest.TestCase):
    def test_extra_names_are_unique_and_fit_32_bytes(self):
        from extra_bytes_names import _extra_name
        self.assertEqual(_extra_name(b"classification", set()), b"classification_extra")
        self.assertEqual(_extra_name(b"classification", {b"classification_extra"}), b"classification_extra2")
        long = b"a_standard_name_of_thirty_two_by"
        self.assertEqual(len(_extra_name(long, set())), 32)
        self.assertTrue(_extra_name(long, set()).endswith(b"_extra"))

    @NEEDS_PDAL
    def test_names_pdal_folds_into_standard_fields_clash(self):
        from extra_bytes_names import clashes_with_standard_field
        for name in (b"classification", b"Class", b"Withheld", b"KeyPoint", b"Overlap", b"Synthetic",
                     b"Red", b"gps_time", b"Intensity", b"Z"):
            self.assertTrue(clashes_with_standard_field(name, 7, "<i4"), name)
        for name in (b"PointID", b"height above ground", b"Amplitude", b"Reflectance", b"treeID",
                     b"PredInstance_SAT", b"plain_name"):
            self.assertFalse(clashes_with_standard_field(name, 7, "<i4"), name)


if __name__ == "__main__":
    unittest.main()
