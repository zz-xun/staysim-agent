from copy import deepcopy

import ezdxf
import pytest
import yaml

from staysim import domain
from staysim.scene_import import parse_dxf, preview_scene, save_scene


def make_dxf(tmp_path, *, units=6, scale=1, origin=(0, 0), edit=None):
    doc = ezdxf.new("R2010")
    doc.units = units
    model = doc.modelspace()
    for layer in ("STAYSIM_BOUNDARY", "STAYSIM_ENTRANCE", "STAYSIM_BENCH", "STAYSIM_NOTE"):
        doc.layers.new(layer)
    points = [(origin[0] + x * scale, origin[1] + y * scale) for x, y in ((0, 0), (60, 0), (60, 40), (0, 40))]
    boundary = model.add_lwpolyline(points, close=True, dxfattribs={"layer": "STAYSIM_BOUNDARY"})
    entrance = model.add_circle((origin[0] + 2 * scale, origin[1] + 20 * scale), scale,
                                dxfattribs={"layer": "STAYSIM_ENTRANCE"})
    model.add_circle((origin[0] + 30 * scale, origin[1] + 10 * scale), 2 * scale,
                     dxfattribs={"layer": "STAYSIM_BENCH"})
    if edit:
        edit(doc, model, boundary, entrance)
    path = tmp_path / "site.dxf"
    doc.saveas(path)
    return path


def test_mm_translation_save_reuse_and_simulator_compatibility(tmp_path, monkeypatch):
    source = make_dxf(tmp_path, units=4, scale=1000, origin=(250000, 120000))
    draft = parse_dxf(source, name="My site")
    assert draft["scene"]["width"] == 60
    assert draft["scene"]["height"] == 40
    assert draft["scene"]["elements"][0]["x"] == 2
    assert draft["scene"]["elements"][0]["radius"] == 1
    assert draft["source_import"]["origin"] == [250000, 120000]
    monkeypatch.setattr(domain, "SCENARIO_DIR", tmp_path / "scenes")
    first = save_scene(draft)
    assert save_scene(draft) == first
    assert len(list(domain.SCENARIO_DIR.glob("*.yaml"))) == 1
    assert not list(domain.SCENARIO_DIR.glob("*.tmp"))
    loaded = domain.load_scenario(str(domain.SCENARIO_DIR / first))
    assert loaded.width == 60 and loaded.objects[0].x == 2
    persisted = yaml.safe_load((domain.SCENARIO_DIR / first).read_text(encoding="utf-8"))
    assert persisted["source_import"]["sha256"] == draft["source_import"]["sha256"]
    assert preview_scene(draft).axes[0].get_xlim() == (0, 60)


def test_units_must_be_known_or_explicit(tmp_path):
    path = make_dxf(tmp_path, units=0)
    with pytest.raises(ValueError, match="单位"):
        parse_dxf(path)
    assert parse_dxf(path, units="m")["warnings"]
    with pytest.raises(ValueError, match="单位"):
        parse_dxf(path, units="feet")


@pytest.mark.parametrize("edit,match", [
    (lambda d, m, b, e: setattr(b, "closed", False), "闭合"),
    (lambda d, m, b, e: b.set_points([(0, 0), (60, 40), (60, 0), (0, 40)]), "交叉"),
    (lambda d, m, b, e: b.set_points([(0, 0), (60, 0), (55, 40), (0, 40)]), "矩形"),
    (lambda d, m, b, e: b.set_points([(0, 0, 0, 0, 1), (60, 0), (60, 40), (0, 40)]), "bulge"),
    (lambda d, m, b, e: setattr(b.dxf, "elevation", 1), "二维"),
    (lambda d, m, b, e: setattr(e.dxf, "center", (2, 20, 1)), "Z=0"),
    (lambda d, m, b, e: setattr(e.dxf, "extrusion", (0, 0, -1)), "extrusion"),
    (lambda d, m, b, e: setattr(e.dxf, "center", (61, 20)), "设施 x"),
    (lambda d, m, b, e: setattr(e.dxf, "radius", 0), "半径"),
    (lambda d, m, b, e: m.delete_entity(e), "入口"),
    (lambda d, m, b, e: m.add_line((0, 0), (1, 1)), "不支持图层"),
    (lambda d, m, b, e: m.add_circle((1, 1), 1, dxfattribs={"layer": "WALL"}), "WALL"),
    (lambda d, m, b, e: m.add_lwpolyline([(0, 0), (1, 0), (1, 1), (0, 1)], close=True,
                                       dxfattribs={"layer": "STAYSIM_BOUNDARY"}), "一个"),
])
def test_rejects_unsupported_or_invalid_geometry(tmp_path, edit, match):
    with pytest.raises(ValueError, match=match):
        parse_dxf(make_dxf(tmp_path, edit=edit))


def test_explicit_note_layer_allowed(tmp_path):
    path = make_dxf(tmp_path, edit=lambda d, m, b, e: m.add_text("note", dxfattribs={"layer": "STAYSIM_NOTE"}))
    assert len(parse_dxf(path)["scene"]["elements"]) == 2


def test_confirm_revalidates_and_never_overwrites(tmp_path, monkeypatch):
    draft = parse_dxf(make_dxf(tmp_path))
    monkeypatch.setattr(domain, "SCENARIO_DIR", tmp_path / "scenes")
    invalid = deepcopy(draft)
    invalid["scene"]["elements"][0]["x"] = float("nan")
    with pytest.raises(ValueError, match="有限"):
        save_scene(invalid)
    invalid = deepcopy(draft)
    invalid["scene"]["elements"][0]["type"] = "wall"
    with pytest.raises(ValueError, match="未知"):
        save_scene(invalid)
    filename = save_scene(draft)
    path = domain.SCENARIO_DIR / filename
    path.write_text("user-owned content", encoding="utf-8")
    with pytest.raises(ValueError, match="未覆盖"):
        save_scene(draft)
    assert path.read_text(encoding="utf-8") == "user-owned content"


def test_resource_limits_and_bad_file(tmp_path):
    path = tmp_path / "bad.dxf"
    path.write_text("not a dxf", encoding="utf-8")
    with pytest.raises(ValueError, match="无法解析"):
        parse_dxf(path)
    with pytest.raises(ValueError, match="不能大于"):
        parse_dxf(make_dxf(tmp_path, scale=100))


def test_facility_entity_and_file_limits(tmp_path, monkeypatch):
    from staysim import scene_import

    path = make_dxf(tmp_path)
    monkeypatch.setattr(scene_import, "MAX_FACILITIES", 1)
    with pytest.raises(ValueError, match="设施数量"):
        parse_dxf(path)
    monkeypatch.setattr(scene_import, "MAX_ENTITIES", 2)
    with pytest.raises(ValueError, match="实体不得超过"):
        parse_dxf(path)
    monkeypatch.setattr(scene_import, "MAX_FILE_BYTES", 1)
    with pytest.raises(ValueError, match="10 MB"):
        parse_dxf(path)


def test_no_block_expansion_or_paper_space_ignored(tmp_path):
    def insert_block(doc, model, boundary, entrance):
        block = doc.blocks.new("OBSTACLE")
        block.add_circle((0, 0), 1)
        model.add_blockref("OBSTACLE", (5, 5), dxfattribs={"layer": "STAYSIM_BENCH"})

    with pytest.raises(ValueError, match="INSERT"):
        parse_dxf(make_dxf(tmp_path, edit=insert_block))
    with pytest.raises(ValueError, match="图纸空间"):
        parse_dxf(make_dxf(tmp_path, edit=lambda d, m, b, e: d.layout().add_circle((1, 1), 1)))
