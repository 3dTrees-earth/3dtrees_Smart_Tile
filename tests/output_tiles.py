"""Find a merge output tile by its pair number (outputs are named cXX_rYY)."""
import json
from pathlib import Path


def output_tile(directory, number, model=0):
    """``directory`` is the output tiles folder (or one model's subfolder)."""
    directory = Path(directory)
    report = next(p / "remap_first_report.json" for p in (directory.parent, directory.parent.parent)
                  if (p / "remap_first_report.json").exists())
    names = json.loads(report.read_text())["models"][model]["tile_names"]
    return directory / f"{names[number]}.laz"
