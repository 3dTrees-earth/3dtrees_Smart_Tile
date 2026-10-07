"""Every `from <src module> import name` resolves, including imports inside functions.

Lazy imports in task functions (e.g. run_tile_task) are not exercised by the
unit tests; a removed or moved helper must fail here instead of at runtime.
"""
import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def top_level_names(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)) and node in tree.body:
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names |= {alias.asname or alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
    return names | {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


class ImportTargetTests(unittest.TestCase):
    def test_from_imports_of_src_modules_resolve(self):
        modules = {p.stem: p for p in SRC.glob("*.py")}
        cache, missing = {}, []
        for path in [*SRC.glob("*.py"), *(ROOT / "tests").rglob("*.py")]:
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module in modules:
                    available = cache.setdefault(node.module, top_level_names(modules[node.module]))
                    missing += [f"{path.relative_to(ROOT)}:{node.lineno} {node.module}.{alias.name}"
                                for alias in node.names if alias.name != "*" and alias.name not in available]
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
