"""Scene loading and simulation data; independent of UI and Agent frameworks."""

import math
import copy
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

SCENARIO_DIR = Path(__file__).resolve().parents[1] / "scenarios"

class SpaceElement(Enum):
    BENCH = "座椅/长椅"
    TREE = "树荫"
    EXERCISE = "健身器材"
    CHESS_TABLE = "棋桌"
    PLAYGROUND = "儿童活动区"
    PATH = "步道"
    ENTRANCE = "入口"
    FOUNTAIN = "水景"
    PAVILION = "凉亭"
    GREEN_LAWN = "草坪"
    FLOWER_BED = "花坛"

# 类型名到枚举的映射
TYPE_MAP = {
    "bench": SpaceElement.BENCH, "tree": SpaceElement.TREE,
    "chess_table": SpaceElement.CHESS_TABLE, "exercise": SpaceElement.EXERCISE,
    "playground": SpaceElement.PLAYGROUND, "fountain": SpaceElement.FOUNTAIN,
    "pavilion": SpaceElement.PAVILION, "flower_bed": SpaceElement.FLOWER_BED,
    "green_lawn": SpaceElement.GREEN_LAWN, "entrance": SpaceElement.ENTRANCE,
    "path": SpaceElement.PATH,
}

@dataclass
class SpaceObject:
    x: float; y: float; radius: float
    element: SpaceElement; attractiveness: float
    shade: bool = False; capacity: int = 10; label: str = ""

@dataclass
class Environment:
    width: float; height: float
    objects: List[SpaceObject] = field(default_factory=list)
    name: str = ""

    def get_nearby(self, x, y, max_dist=15.0):
        nearby = []
        for obj in self.objects:
            if math.sqrt((x-obj.x)**2 + (y-obj.y)**2) <= max_dist:
                nearby.append(obj)
        nearby.sort(key=lambda o: math.sqrt((x-o.x)**2 + (y-o.y)**2))
        return nearby

@dataclass
class PersonaProfile:
    name: str; age_range: Tuple[int, int]; speed: float
    stay_preference: float; social_preference: float
    activity_interests: Dict; description: str = ""

class Action(Enum):
    WALK_TO = "走向目标"; STAY = "停留"; WANDER = "漫游"; LEAVE = "离开"

@dataclass
class Decision:
    action: Action; target: Optional[SpaceObject] = None
    target_pos: Optional[Tuple] = None; stay_duration: float = 0.0; reason: str = ""

@dataclass
class Agent:
    id: int; persona: PersonaProfile; x: float; y: float
    energy: float = 1.0; mood: float = 0.7
    current_action: Action = Action.WANDER
    current_target: Optional[SpaceObject] = None
    stay_remaining: float = 0.0
    trajectory: List = field(default_factory=list)
    decisions_log: List = field(default_factory=list)
    stay_points: List = field(default_factory=list)
    stay_events: List = field(default_factory=list)

# --- 场景加载器 ---

def load_scenario(yaml_path: str) -> Environment:
    """从 YAML 文件加载并校验场景，不猜测未知设施类型。"""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return environment_from_dict(data)


def environment_from_dict(data) -> Environment:
    """Validate the shared YAML/history scene contract without changing numeric types."""
    if not isinstance(data, dict):
        raise ValueError("场景必须是 YAML 对象")

    def number(value, field, *, positive=False):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field} 必须是数值")
        if not math.isfinite(value) or (positive and value <= 0):
            raise ValueError(f"{field} 必须是有限{'正' if positive else ''}数")
        return value

    width = number(data.get('width'), 'width', positive=True)
    height = number(data.get('height'), 'height', positive=True)
    if not isinstance(data.get('name', ''), str):
        raise ValueError("name 必须是文本")
    elements = data.get('elements', [])
    if not isinstance(elements, list):
        raise ValueError("elements 必须是列表")
    env = Environment(width=data['width'], height=data['height'], name=data.get('name', ''))
    for index, elem in enumerate(elements):
        prefix = f"elements[{index}]"
        if not isinstance(elem, dict):
            raise ValueError(f"{prefix} 必须是对象")
        if not isinstance(elem.get('type'), str) or elem['type'] not in TYPE_MAP:
            raise ValueError(f"{prefix}.type 未知：{elem.get('type')!r}")
        x = number(elem.get('x'), f'{prefix}.x')
        y = number(elem.get('y'), f'{prefix}.y')
        if not (0 <= x <= width and 0 <= y <= height):
            raise ValueError(f"{prefix} 中心坐标必须在场景边界内")
        radius = number(elem.get('radius', 2.0), f'{prefix}.radius', positive=True)
        attractiveness = number(elem.get('attractiveness', 0.5), f'{prefix}.attractiveness')
        if not 0 <= attractiveness <= 1:
            raise ValueError(f"{prefix}.attractiveness 必须在 0 到 1 之间")
        capacity = elem.get('capacity', 10)
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError(f"{prefix}.capacity 必须是正整数")
        if not isinstance(elem.get('shade', False), bool):
            raise ValueError(f"{prefix}.shade 必须是布尔值")
        if not isinstance(elem.get('label', ''), str):
            raise ValueError(f"{prefix}.label 必须是文本")
        env.objects.append(SpaceObject(
            x=x, y=y, radius=radius,
            element=TYPE_MAP[elem['type']],
            attractiveness=attractiveness,
            shade=elem.get('shade', False),
            capacity=capacity,
            label=elem.get('label', '')
        ))
    if 'source_import' in data:
        if not isinstance(data['source_import'], dict):
            raise ValueError("source_import 必须是对象")
        try:
            json.dumps(data['source_import'], allow_nan=False)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("source_import 必须是可保存的有限 JSON 数据") from exc
        env.source_import = copy.deepcopy(data['source_import'])
    return env

def list_scenarios() -> List[str]:
    """列出所有可用的场景文件"""
    if not SCENARIO_DIR.exists():
        return []
    return sorted([f.name for f in SCENARIO_DIR.glob("*.yaml")])

# --- Persona 工厂 ---

def get_default_personas() -> Dict[str, PersonaProfile]:
    return {
        "买菜老人": PersonaProfile("买菜老人", (60,75), 0.8, 0.8, 0.7, {
            SpaceElement.BENCH: 0.9, SpaceElement.CHESS_TABLE: 0.7, SpaceElement.TREE: 0.8,
            SpaceElement.PAVILION: 0.6, SpaceElement.EXERCISE: 0.3, SpaceElement.PLAYGROUND: 0.1,
            SpaceElement.GREEN_LAWN: 0.2, SpaceElement.FOUNTAIN: 0.4, SpaceElement.FLOWER_BED: 0.5,
        }, "腿脚不便，喜欢坐着聊天、看棋，怕晒"),
        "带娃家长": PersonaProfile("带娃家长", (28,40), 1.2, 0.5, 0.6, {
            SpaceElement.PLAYGROUND: 0.95, SpaceElement.BENCH: 0.7, SpaceElement.GREEN_LAWN: 0.6,
            SpaceElement.TREE: 0.5, SpaceElement.EXERCISE: 0.2, SpaceElement.CHESS_TABLE: 0.1,
            SpaceElement.FOUNTAIN: 0.4, SpaceElement.PAVILION: 0.3, SpaceElement.FLOWER_BED: 0.2,
        }, "带孩子来玩，需要安全、能看到孩子的地方"),
        "放学儿童": PersonaProfile("放学儿童", (7,12), 1.8, 0.3, 0.9, {
            SpaceElement.PLAYGROUND: 0.95, SpaceElement.GREEN_LAWN: 0.85, SpaceElement.EXERCISE: 0.6,
            SpaceElement.FOUNTAIN: 0.7, SpaceElement.CHESS_TABLE: 0.1, SpaceElement.BENCH: 0.2,
            SpaceElement.TREE: 0.3, SpaceElement.PAVILION: 0.2, SpaceElement.FLOWER_BED: 0.3,
        }, "精力旺盛，喜欢跑动玩耍，容易被有趣的东西吸引"),
        "通勤青年": PersonaProfile("通勤青年", (22,35), 1.5, 0.2, 0.3, {
            SpaceElement.BENCH: 0.3, SpaceElement.PATH: 0.9, SpaceElement.TREE: 0.2,
            SpaceElement.FOUNTAIN: 0.4, SpaceElement.GREEN_LAWN: 0.2, SpaceElement.EXERCISE: 0.2,
            SpaceElement.PLAYGROUND: 0.05, SpaceElement.CHESS_TABLE: 0.05,
            SpaceElement.PAVILION: 0.15, SpaceElement.FLOWER_BED: 0.2,
        }, "赶时间路过，很少停留，主要是穿行"),
        "社区闲散": PersonaProfile("社区闲散", (45,65), 1.0, 0.7, 0.8, {
            SpaceElement.CHESS_TABLE: 0.85, SpaceElement.BENCH: 0.8, SpaceElement.EXERCISE: 0.7,
            SpaceElement.PAVILION: 0.7, SpaceElement.TREE: 0.5, SpaceElement.GREEN_LAWN: 0.4,
            SpaceElement.FOUNTAIN: 0.5, SpaceElement.PLAYGROUND: 0.2, SpaceElement.FLOWER_BED: 0.4,
        }, "社区熟人，四处逛逛，碰到熟人停下来聊几句"),
    }

# --- 模拟结果数据结构 ---



@dataclass
class SimResult:
    trajectories: Dict[str, List] = field(default_factory=dict)
    stay_points: Dict[str, List] = field(default_factory=dict)
    stay_events: Dict[str, List] = field(default_factory=dict)
    decision_logs: Dict[str, List] = field(default_factory=dict)
    final_positions: List = field(default_factory=list)
    env_name: str = ""
    n_agents: int = 0
    elapsed: float = 0.0
    engine_mode: str = "rule"
    llm_calls: int = 0
    total_tokens: int = 0
    spatial_entropy: float = 0.0
    facility_utilization: Dict[str, float] = field(default_factory=dict)
    cold_zone_ratio: float = 0.0
    stay_pass_ratio: float = 0.0
    mean_stay_duration: float = 0.0
    total_stay_events: int = 0
    top_facility: str = ""
