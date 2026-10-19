"""The tile overview names the CRS briefly: a full WKT made a 30,000 px PNG that Galaxy rejected."""
import importlib.util
from pathlib import Path

import pytest
from pyproj import CRS

SPEC = importlib.util.spec_from_file_location(
    "plot_tiles_and_copc_labels", Path(__file__).resolve().parents[1] / "src" / "plot_tiles_and_copc.py")
plot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plot)


@pytest.mark.parametrize("srs, label", [
    ("EPSG:25832", "EPSG:25832"),
    (CRS.from_epsg(25832).to_wkt(), "EPSG:25832"),
    ('LOCAL_CS["plot grid",UNIT["metre",1]]', "plot grid"),
    ("x" * 500, "x" * 57 + "..."),
])
def test_crs_label_is_short(srs, label):
    assert plot.crs_label(srs) == label
