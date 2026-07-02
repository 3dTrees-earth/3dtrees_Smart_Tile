import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dimension_transfer import (  # noqa: E402
    next_available_suffix,
    plan_dimension_transfer,
    suffixes_for_collision,
)


class DimensionTransferTests(unittest.TestCase):
    def test_plans_add_overwrite_rename_and_skip_core_dimensions(self):
        source_dims = {
            "X": np.dtype(np.float64),
            "OriginalOnly": np.dtype(np.uint16),
            "EmptyTarget": np.dtype(np.uint8),
            "RealCollision": np.dtype(np.float32),
        }
        target_names = {"X", "Y", "Z", "EmptyTarget", "RealCollision"}
        target_arrays = {
            "EmptyTarget": np.zeros(4, dtype=np.uint8),
            "RealCollision": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        }

        plan = plan_dimension_transfer(
            source_dims,
            target_names,
            lambda name: target_arrays.get(name),
        )

        self.assertEqual(plan.add_new, {"OriginalOnly": np.dtype(np.uint16)})
        self.assertEqual(plan.overwrite, {"EmptyTarget": np.dtype(np.uint8)})
        self.assertEqual(plan.renamed, {"RealCollision": "RealCollision_1"})
        self.assertEqual(
            plan.output_dtypes,
            {
                "OriginalOnly": np.dtype(np.uint16),
                "EmptyTarget": np.dtype(np.uint8),
                "RealCollision_1": np.dtype(np.float32),
            },
        )
        self.assertEqual(
            plan.output_to_source,
            {
                "OriginalOnly": "OriginalOnly",
                "EmptyTarget": "EmptyTarget",
                "RealCollision_1": "RealCollision",
            },
        )

    def test_constant_nonzero_target_is_overwritten(self):
        plan = plan_dimension_transfer(
            {"Classification": np.dtype(np.uint8)},
            {"Classification"},
            lambda _: np.array([2, 2, 2], dtype=np.uint8),
        )

        self.assertEqual(plan.overwrite, {"Classification": np.dtype(np.uint8)})
        self.assertFalse(plan.renamed)

    def test_empty_target_dimension_is_overwritten(self):
        plan = plan_dimension_transfer(
            {"PredInstance": np.dtype(np.uint16)},
            {"PredInstance"},
            lambda _: np.array([], dtype=np.uint16),
        )

        self.assertEqual(plan.overwrite, {"PredInstance": np.dtype(np.uint16)})
        self.assertFalse(plan.renamed)

    def test_renamed_collision_uses_next_available_suffix(self):
        plan = plan_dimension_transfer(
            {"Intensity": np.dtype(np.uint16)},
            {"Intensity", "Intensity_1"},
            lambda _: np.array([1, 2, 3], dtype=np.uint16),
        )

        self.assertEqual(plan.renamed, {"Intensity": "Intensity_2"})

    def test_suffix_helpers_allocate_stable_names(self):
        used = {"Dim_1"}
        self.assertEqual(next_available_suffix("Dim", used), "Dim_2")
        self.assertEqual(suffixes_for_collision("Dim", used), ("Dim_2", "Dim_3"))
        self.assertEqual(used, {"Dim_1", "Dim_2", "Dim_3"})


if __name__ == "__main__":
    unittest.main()
