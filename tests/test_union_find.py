import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from union_find import UnionFind  # noqa: E402


class UnionFindTests(unittest.TestCase):
    def test_union_keeps_larger_component_as_root(self):
        uf = UnionFind()
        uf.make_set(1, size=10)
        uf.make_set(2, size=2)

        root = uf.union(1, 2)

        self.assertEqual(root, 1)
        self.assertEqual(uf.find(2), 1)
        self.assertEqual(uf.size[1], 12)

    def test_union_can_promote_larger_second_root(self):
        uf = UnionFind()
        uf.make_set(1, size=1)
        uf.make_set(2, size=5)

        root = uf.union(1, 2)

        self.assertEqual(root, 2)
        self.assertEqual(uf.find(1), 2)
        self.assertEqual(uf.size[2], 6)

    def test_get_components(self):
        uf = UnionFind()
        uf.make_set(1)
        uf.make_set(2)
        uf.make_set(3)
        uf.union(1, 2)

        components = {root: sorted(members) for root, members in uf.get_components().items()}

        self.assertIn([1, 2], components.values())
        self.assertIn([3], components.values())


if __name__ == "__main__":
    unittest.main()
