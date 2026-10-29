# smarttile-native

Optional Rust implementation of the tiling voxel reducer
(`tile_centroids.reduce_records`). Python keeps the spill blocks, masks,
resolution 0, scaling and every output; the extension only returns one
representative per voxel on the LAS integer grid, in voxel-key order. Selected
with `SMARTTILE_REDUCER=native` (see the main README).

It has no dependency beyond PyO3 and rust-numpy (MIT/Apache-2.0) and is MIT
licensed like SmartTile.

## Build and test

Rust stable and Python 3.12 with NumPy:

```bash
cd native
cargo test --release                      # reducer unit tests, no Python needed
pip install maturin
maturin build --release -o dist           # abi3 wheel for CPython >= 3.12
pip install dist/smarttile_native-*.whl
cd .. && PYTHONPATH=src python -m pytest -q tests/test_native_reducer.py
```

`tests/test_native_reducer.py` compares both reducers byte for byte (it is
skipped when the extension is not installed). The whole suite also runs with
`SMARTTILE_REDUCER=native`.
