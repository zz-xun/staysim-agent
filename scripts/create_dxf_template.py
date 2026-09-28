"""Generate an editable normalized DXF template and a millimetre example."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ezdxf
from staysim.scene_import import BOUNDARY_LAYER, FACILITY_LAYERS, NOTE_LAYER


def create_template(path: Path, *, millimetres=False):
    doc = ezdxf.new("R2010")
    doc.units = 4 if millimetres else 6
    scale = 1000 if millimetres else 1
    origin = (250 * scale, 120 * scale) if millimetres else (0, 0)
    for layer in (BOUNDARY_LAYER, NOTE_LAYER, *FACILITY_LAYERS):
        doc.layers.new(layer)
    model = doc.modelspace()
    points = [(origin[0] + x * scale, origin[1] + y * scale) for x, y in ((0, 0), (60, 0), (60, 40), (0, 40))]
    model.add_lwpolyline(points, close=True, dxfattribs={"layer": BOUNDARY_LAYER})
    for kind, x, y, radius in (("ENTRANCE", 2, 20, 1.5), ("ENTRANCE", 58, 20, 1.5),
                                ("BENCH", 20, 10, 2), ("TREE", 30, 20, 3),
                                ("CHESS_TABLE", 40, 30, 2)):
        model.add_circle((origin[0] + x * scale, origin[1] + y * scale), radius * scale,
                         dxfattribs={"layer": f"STAYSIM_{kind}"})
    model.add_text("STAYSIM: axis-aligned boundary + facility circles; no obstacles.",
                   dxfattribs={"layer": NOTE_LAYER, "height": scale, "insert": (origin[0], origin[1] - 3 * scale)})
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(path)
    return path


if __name__ == "__main__":
    for filename, millimetres in (("staysim_template_m.dxf", False), ("staysim_example_mm.dxf", True)):
        print(create_template(ROOT / "examples" / filename, millimetres=millimetres))
