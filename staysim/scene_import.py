"""Strict, deterministic import of a small two-dimensional DXF convention."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import yaml

from . import domain

MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_ENTITIES = 1000
MAX_FACILITIES = 200
MAX_EXTENT_M = 1000.0
BOUNDARY_LAYER = "STAYSIM_BOUNDARY"
NOTE_LAYER = "STAYSIM_NOTE"
FACILITY_LAYERS = {f"STAYSIM_{kind.upper()}": kind for kind in domain.TYPE_MAP}
ASSUMPTIONS = [
    "设施圆代表作用范围，不代表建筑、墙体或碰撞障碍。",
    "attractiveness=0.5、capacity=10、shade=false 为统一模型假设，未从 CAD 推断。",
    "坐标保持 CAD 的 x 向右、y 向上；矩形左下角平移为 (0,0)，长度统一为米。",
]


def _number(value, label, *, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} 必须为有限数值。")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} 必须为有限数值。")
    if minimum is not None and result < minimum:
        raise ValueError(f"{label} 不能小于 {minimum}。")
    if maximum is not None and result > maximum:
        raise ValueError(f"{label} 不能大于 {maximum}。")
    return result


def _flat(entity):
    extrusion = tuple(entity.dxf.get("extrusion", (0, 0, 1)))
    if len(extrusion) != 3 or any(abs(_number(v, "extrusion") - ref) > 1e-9
                                  for v, ref in zip(extrusion, (0, 0, 1))):
        raise ValueError("仅支持世界坐标系 XY 平面，不能使用倾斜或反向 extrusion。")
    if abs(_number(entity.dxf.get("thickness", 0), "thickness")) > 1e-9:
        raise ValueError("仅支持无厚度的二维实体。")


def _validate_draft(draft):
    """Revalidate at the write boundary; never trust browser-held draft data."""
    if not isinstance(draft, dict) or not isinstance(draft.get("scene"), dict):
        raise ValueError("场景草稿无效，请重新解析 DXF。")
    scene = draft["scene"]
    name = scene.get("name")
    if not isinstance(name, str) or not name.strip() or len(name) > 100:
        raise ValueError("场景名称需为 1–100 个字符。")
    width = _number(scene.get("width"), "场地宽度", minimum=0.1, maximum=MAX_EXTENT_M)
    height = _number(scene.get("height"), "场地高度", minimum=0.1, maximum=MAX_EXTENT_M)
    elements = scene.get("elements")
    if not isinstance(elements, list) or not 1 <= len(elements) <= MAX_FACILITIES:
        raise ValueError(f"设施数量须为 1–{MAX_FACILITIES}。")
    cleaned = []
    for item in elements:
        if not isinstance(item, dict) or item.get("type") not in domain.TYPE_MAP:
            raise ValueError("场景含未知设施类型。")
        element = dict(item)
        element["x"] = _number(item.get("x"), "设施 x", minimum=0, maximum=width)
        element["y"] = _number(item.get("y"), "设施 y", minimum=0, maximum=height)
        element["radius"] = _number(item.get("radius"), "设施半径", minimum=0.001, maximum=max(width, height))
        element["attractiveness"] = _number(item.get("attractiveness"), "吸引力", minimum=0, maximum=1)
        if type(item.get("capacity")) is not int or not 1 <= item["capacity"] <= 10000:
            raise ValueError("设施容量须为 1–10000 的整数。")
        if type(item.get("shade")) is not bool:
            raise ValueError("shade 必须为布尔值。")
        if not isinstance(item.get("label"), str) or not 1 <= len(item["label"]) <= 100:
            raise ValueError("设施名称须为 1–100 个字符。")
        cleaned.append({key: element[key] for key in
                        ("type", "x", "y", "radius", "attractiveness", "shade", "capacity", "label")})
    if not any(item["type"] == "entrance" for item in cleaned):
        raise ValueError("至少需要一个 STAYSIM_ENTRANCE 入口圆。")
    source = draft.get("source_import")
    if not isinstance(source, dict):
        raise ValueError("缺少导入来源，请重新解析 DXF。")
    digest = source.get("sha256", "")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("DXF 来源哈希无效。")
    if source.get("format") != "dxf" or source.get("units") not in ("m", "mm"):
        raise ValueError("DXF 来源格式或单位无效。")
    scale = _number(source.get("scale_to_m"), "单位比例")
    if scale != {"m": 1.0, "mm": 0.001}[source["units"]]:
        raise ValueError("导入单位与转换比例不一致。")
    origin = source.get("origin")
    if not isinstance(origin, list) or len(origin) != 2:
        raise ValueError("DXF 原点记录无效。")
    origin = [_number(value, "原始坐标", minimum=-1e9, maximum=1e9) for value in origin]
    source_name = source.get("source_name")
    if not isinstance(source_name, str) or not 1 <= len(source_name) <= 255:
        raise ValueError("DXF 来源文件名无效。")
    return {
        "name": name.strip(), "width": width, "height": height, "elements": cleaned,
        "source_import": {
            "format": "dxf", "source_name": Path(source_name).name,
            "sha256": digest, "units": source["units"], "scale_to_m": scale,
            "origin": origin, "coordinate_system": "XY, x right, y up, metres",
            "assumptions": list(ASSUMPTIONS),
        },
    }


def parse_dxf(path, units="auto", name="") -> dict:
    """Parse rectangle + facility circles, with explicit metres/millimetres."""
    import ezdxf

    path = Path(path)
    if path.suffix.lower() != ".dxf" or not path.is_file():
        raise ValueError("请选择存在的 .dxf 文件；DWG 请先另存为 DXF。")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("DXF 文件不得超过 10 MB。")
    if units not in ("auto", "m", "mm"):
        raise ValueError("单位只能选择 auto、m 或 mm。")
    if not isinstance(name, str):
        raise ValueError("场景名称必须是文本。")
    data = path.read_bytes()
    try:
        doc = ezdxf.readfile(path)
    except Exception as exc:
        raise ValueError("无法解析 DXF，请使用有效的二维 DXF 文件。") from exc
    raw_unit = doc.header.get("$INSUNITS", 0)
    detected = {4: "mm", 6: "m"}.get(raw_unit)
    if units == "auto" and detected is None:
        raise ValueError("DXF 未声明米或毫米，请明确选择绘图单位。")
    selected = detected if units == "auto" else units
    scale = {"m": 1.0, "mm": 0.001}[selected]
    warnings = []
    if units != "auto" and detected != units:
        warnings.append(f"以用户选择的 {units} 为准，覆盖 DXF INSUNITS={raw_unit}。")
    entities = list(doc.modelspace())
    if len(entities) > MAX_ENTITIES:
        raise ValueError(f"模型空间实体不得超过 {MAX_ENTITIES} 个。")
    # Paper-space geometry cannot silently become part of a simulation scene.
    if any(len(layout) for layout in doc.layouts if layout.name != "Model"):
        raise ValueError("请仅保留模型空间实体，删除布局/图纸空间中的实体后导出。")
    boundaries, facilities = [], []
    for entity in entities:
        layer = entity.dxf.layer.upper()
        kind = entity.dxftype()
        if layer == NOTE_LAYER and kind in ("TEXT", "MTEXT"):
            continue
        if layer == BOUNDARY_LAYER and kind == "LWPOLYLINE":
            boundaries.append(entity)
        elif layer in FACILITY_LAYERS and kind == "CIRCLE":
            facilities.append((entity, FACILITY_LAYERS[layer]))
        else:
            raise ValueError(f"不支持图层 {entity.dxf.layer} 上的 {kind}。边界须为闭合 LWPOLYLINE，设施须为对应 STAYSIM_* 图层的 CIRCLE；注释仅放 STAYSIM_NOTE。")
    if len(boundaries) != 1:
        raise ValueError("必须且只能有一个 STAYSIM_BOUNDARY 矩形边界。")
    if not 1 <= len(facilities) <= MAX_FACILITIES:
        raise ValueError(f"设施数量须为 1–{MAX_FACILITIES}。")
    boundary = boundaries[0]
    _flat(boundary)
    if not boundary.closed or abs(_number(boundary.dxf.elevation, "边界 elevation")) > 1e-9:
        raise ValueError("边界须为 Z=0 的闭合二维 LWPOLYLINE。")
    if abs(_number(boundary.dxf.get("const_width", 0), "边界线宽")) > 1e-9:
        raise ValueError("边界必须使用零线宽。")
    vertices = list(boundary.get_points("xyseb"))
    if len(vertices) == 5 and vertices[0][:2] == vertices[-1][:2]:
        if any(abs(_number(value, "边界线宽/弧度")) > 1e-9 for value in vertices[-1][2:]):
            raise ValueError("边界不能含曲线 bulge 或线宽。")
        vertices = vertices[:-1]
    if len(vertices) != 4:
        raise ValueError("边界须为四顶点、与坐标轴平行的矩形。")
    points = []
    for x, y, start_width, end_width, bulge in vertices:
        if any(abs(_number(value, "边界线宽/弧度")) > 1e-9 for value in (start_width, end_width, bulge)):
            raise ValueError("边界不能含曲线 bulge 或线宽。")
        points.append((_number(x, "边界 x", minimum=-1e9, maximum=1e9),
                       _number(y, "边界 y", minimum=-1e9, maximum=1e9)))
    xs, ys = sorted(set(p[0] for p in points)), sorted(set(p[1] for p in points))
    if len(xs) != 2 or len(ys) != 2 or len(set(points)) != 4:
        raise ValueError("边界须为四个不同顶点构成的轴对齐矩形。")
    for first, second in zip(points, points[1:] + points[:1]):
        if (first[0] == second[0]) == (first[1] == second[1]):
            raise ValueError("矩形边界不能交叉或包含斜边。")
    elements, counts = [], {}
    for entity, kind in facilities:
        _flat(entity)
        center = entity.dxf.center
        if abs(_number(center.z, "设施 Z")) > 1e-9:
            raise ValueError("设施圆心必须位于 Z=0 平面。")
        counts[kind] = counts.get(kind, 0) + 1
        elements.append({
            "type": kind,
            "x": (_number(center.x, "圆心 x") - xs[0]) * scale,
            "y": (_number(center.y, "圆心 y") - ys[0]) * scale,
            "radius": _number(entity.dxf.radius, "圆半径") * scale,
            "attractiveness": 0.5, "shade": False, "capacity": 10,
            "label": f"{domain.TYPE_MAP[kind].value}{counts[kind]}",
        })
    draft = {
        "scene": {"name": name.strip() or path.stem, "width": (xs[1] - xs[0]) * scale,
                  "height": (ys[1] - ys[0]) * scale, "elements": elements},
        "source_import": {"format": "dxf", "source_name": path.name,
                          "sha256": hashlib.sha256(data).hexdigest(), "units": selected,
                          "scale_to_m": scale, "origin": [xs[0], ys[0]]},
        "warnings": warnings,
    }
    verified = _validate_draft(draft)
    draft["source_import"] = verified.pop("source_import")
    draft["scene"] = verified
    return draft


def save_scene(draft) -> str:
    """Validate and atomically publish a content-addressed YAML; never overwrite."""
    scene = _validate_draft(draft)
    canonical = json.dumps(scene, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    filename = f"dxf_{digest[:16]}.yaml"
    directory = domain.SCENARIO_DIR
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / filename
    payload = yaml.safe_dump(scene, allow_unicode=True, sort_keys=False)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".tmp", dir=directory, delete=False) as stream:
        temp_path = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        try:
            os.link(temp_path, destination)
        except FileExistsError:
            if destination.is_symlink() or destination.read_text(encoding="utf-8") != payload:
                raise ValueError("同名场景文件已存在且内容不同，未覆盖；请保留文件并检查场景目录。")
    finally:
        temp_path.unlink(missing_ok=True)
    return filename


def preview_scene(draft):
    """Draw exactly the normalized geometry that will be serialized."""
    from matplotlib.figure import Figure
    from matplotlib.patches import Circle

    scene = _validate_draft(draft)
    fig = Figure(figsize=(8, 5), layout="constrained")
    axis = fig.subplots()
    palette = {"entrance": "#16a34a", "tree": "#65a30d", "bench": "#4f46e5"}
    for index, element in enumerate(scene["elements"], start=1):
        axis.add_patch(Circle((element["x"], element["y"]), element["radius"],
                              facecolor=palette.get(element["type"], "#d97706"), alpha=0.4))
        axis.annotate(f"{index}: {element['type']}", (element["x"], element["y"]), fontsize=8)
    axis.set(xlim=(0, scene["width"]), ylim=(0, scene["height"]), xlabel="x (m)", ylabel="y (m)")
    axis.set_aspect("equal")
    axis.grid(alpha=0.15)
    return fig
