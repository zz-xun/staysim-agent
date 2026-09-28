"""Create a small, human-readable CAD drawing for the DXF acceptance run."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ezdxf

from staysim.scene_import import BOUNDARY_LAYER, FACILITY_LAYERS, NOTE_LAYER


def create(path: Path) -> Path:
    doc = ezdxf.new("R2010")
    doc.units = 6  # metres
    for layer in (BOUNDARY_LAYER, NOTE_LAYER, *FACILITY_LAYERS):
        if layer not in doc.layers:
            doc.layers.new(layer)
    model = doc.modelspace()

    # 80 m x 50 m open plaza, with no obstacle geometry.
    model.add_lwpolyline([(0, 0), (80, 0), (80, 50), (0, 50)],
                         close=True, dxfattribs={"layer": BOUNDARY_LAYER})

    facilities = [
        ("ENTRANCE", "西入口", 2, 25, 2),
        ("ENTRANCE", "东入口", 78, 25, 2),
        ("BENCH", "北侧长椅", 25, 12, 2),
        ("BENCH", "南侧长椅", 30, 40, 2),
        ("BENCH", "中心长椅", 55, 25, 2),
        ("TREE", "大树一", 18, 20, 3),
        ("TREE", "大树二", 38, 15, 3),
        ("TREE", "大树三", 62, 35, 3),
        ("TREE", "大树四", 45, 42, 3),
        ("CHESS_TABLE", "中心棋桌", 40, 25, 2.5),
        ("PAVILION", "社区凉亭", 65, 15, 4),
    ]
    for kind, label, x, y, radius in facilities:
        model.add_circle((x, y), radius,
                         dxfattribs={"layer": f"STAYSIM_{kind}"})
        model.add_text(label, dxfattribs={
            "layer": NOTE_LAYER, "height": 1.2, "insert": (x + radius + 0.5, y),
        })
    model.add_text("StaySim acceptance CAD | units=m | boundary + facility circles only",
                   dxfattribs={"layer": NOTE_LAYER, "height": 1.2, "insert": (1, -3)})
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(path)
    return path


if __name__ == "__main__":
    print(create(ROOT / "examples" / "community_square_acceptance.dxf"))
