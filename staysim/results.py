"""Metrics, run artifacts and optional plotting. Plot imports are lazy."""

import csv
import datetime
import json
import math
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

import numpy as np

from .domain import (
    Environment, SCENARIO_DIR, SimResult, SpaceElement, TYPE_MAP,
    environment_from_dict, load_scenario,
)

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"
RUNS_DIR = OUTPUT_DIR / "runs"


@lru_cache(maxsize=1)
def _plot_modules():
    """Load matplotlib only when rendering; leave simulation imports lightweight."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt

    font_path = next((
        font.fname for font in fm.fontManager.ttflist
        if font.name == "Microsoft YaHei" and getattr(font, "weight", 400) == 400
    ), None)
    if font_path is None:
        font_path = next((
            font.fname for font in fm.fontManager.ttflist
            if "Microsoft YaHei" in font.name
        ), None)
    matplotlib.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "Microsoft JhengHei", "DejaVu Sans"
    ]
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["axes.unicode_minus"] = False
    return plt, fm, font_path

def _compute_spatial_metrics(agents, env, grid_size=2.0):
    """计算空间使用量化指标"""
    # 网格化空间
    n_cols = int(env.width / grid_size) + 1
    n_rows = int(env.height / grid_size) + 1
    grid = np.zeros((n_rows, n_cols))

    total_pts = 0
    for agent in agents:
        for x, y in agent.trajectory:
            if hasattr(env, 'source_import') and not (0 <= x <= env.width and 0 <= y <= env.height):
                # Leaving pedestrians must not populate cells on the opposite edge
                # through NumPy negative indexing or be counted as still on site.
                continue
            col = min(int(x / grid_size), n_cols - 1)
            row = min(int(y / grid_size), n_rows - 1)
            grid[row, col] += 1
            total_pts += 1

    # 1. 空间使用熵：-Σ(p_i × log(p_i))，越高越均匀
    flat = grid.flatten()
    flat = flat[flat > 0]  # 去掉零格子
    if total_pts > 0:
        probs = flat / flat.sum()
        entropy = -np.sum(probs * np.log(probs + 1e-10))
    else:
        entropy = 0.0

    # 2. 冷区比率：0停留格子占比
    total_cells = n_rows * n_cols
    cold_cells = np.sum(grid == 0)
    cold_ratio = cold_cells / total_cells if total_cells > 0 else 1.0

    all_stay_events = [event for agent in agents for event in agent.stay_events]

    # 3. 设施利用率：各设施旁停留事件数
    facility_util = {}
    for obj in env.objects:
        pts_nearby = 0
        for agent in agents:
            for event in agent.stay_events:
                x, y = event["x"], event["y"]
                if math.sqrt((x-obj.x)**2 + (y-obj.y)**2) < obj.radius + 2:
                    pts_nearby += 1
        label = obj.label or obj.element.value
        facility_util[label] = pts_nearby

    # 4. 停留-通行比
    total_decisions = sum(1 for a in agents for _ in a.decisions_log)
    stay_pass_ratio = len(all_stay_events) / max(1, total_decisions)

    # 5. 平均停留事件时长
    total_stay_duration = sum(event["duration"] for event in all_stay_events)
    mean_stay = total_stay_duration / max(1, len(all_stay_events))
    top_facility = ""
    if facility_util:
        top_facility = max(facility_util.items(), key=lambda item: item[1])[0]

    return {
        'entropy': round(entropy, 3),
        'cold_ratio': round(cold_ratio, 3),
        'facility_util': facility_util,
        'stay_pass_ratio': round(stay_pass_ratio, 3),
        'mean_stay': round(mean_stay, 1),
        'total_stay_events': len(all_stay_events),
        'top_facility': top_facility,
    }


def build_summary_row(env, result, config_dict=None):
    """Return one paper-ready summary row for CSV export."""
    config_dict = config_dict or {}
    return {
        "scenario": config_dict.get("scenario_file") or result.env_name or env.name,
        "engine_mode": result.engine_mode,
        "requested_engine_mode": getattr(result, "requested_engine_mode", result.engine_mode),
        "effective_engine_mode": getattr(result, "effective_engine_mode", result.engine_mode),
        "seed": config_dict.get("seed", ""),
        "duration": config_dict.get("duration", ""),
        "dt": config_dict.get("dt", 2.0),
        "total_agents": result.n_agents,
        "total_stay_events": result.total_stay_events,
        "mean_stay_duration": result.mean_stay_duration,
        "stay_pass_ratio": result.stay_pass_ratio,
        "spatial_entropy": result.spatial_entropy,
        "cold_zone_ratio": result.cold_zone_ratio,
        "top_facility": result.top_facility,
        "llm_calls": result.llm_calls,
        "total_tokens": result.total_tokens,
        "llm_failure_count": getattr(result, "llm_failure_count", 0),
        "fallback_count": getattr(result, "fallback_count", 0),
        "simulation_quality": getattr(result, "simulation_quality", "ok"),
        "elapsed_seconds": round(result.elapsed, 3),
    }


def build_validation_report(env, result, baseline_result=None):
    """Build the five behavior-reasonableness checks used in the paper."""
    def all_events():
        events = []
        for values in result.stay_events.values():
            events.extend(values)
        return events

    def events_by_keywords(*keywords, fallback_index=None):
        for pname, events in result.stay_events.items():
            lower = str(pname).lower()
            if any(keyword.lower() in lower for keyword in keywords):
                return events, pname
        items = list(result.stay_events.items())
        if fallback_index is not None and fallback_index < len(items):
            return items[fallback_index][1], items[fallback_index][0]
        return [], ""

    def count_near(events, elements=None, min_attr=None):
        hits = 0
        for event in events:
            for obj in env.objects:
                if elements and obj.element not in elements:
                    continue
                if min_attr is not None and obj.attractiveness < min_attr:
                    continue
                dist = math.sqrt((event["x"] - obj.x) ** 2 + (event["y"] - obj.y) ** 2)
                if dist <= obj.radius + 3:
                    hits += 1
                    break
        return hits, len(events)

    elderly_events, elderly_name = events_by_keywords("老人", "老", "elder", "senior", "买菜", fallback_index=0)
    child_events, child_name = events_by_keywords("儿童", " child", "kid", "学生", "放学", fallback_index=2)
    commuter_events, commuter_name = events_by_keywords("青年", "通勤", "commuter", "young", fallback_index=3)
    other_totals = [len(events) for pname, events in result.stay_events.items() if pname != commuter_name]

    elderly_hits, elderly_total = count_near(
        elderly_events,
        {SpaceElement.BENCH, SpaceElement.TREE, SpaceElement.PAVILION},
    )
    child_hits, child_total = count_near(
        child_events,
        {SpaceElement.PLAYGROUND, SpaceElement.GREEN_LAWN},
    )
    high_attr_hits, event_count = count_near(all_events(), min_attr=0.6)
    commuter_total = len(commuter_events)

    before_after_score = None
    before_after_pass = None
    if baseline_result is not None:
        before = max(1, baseline_result.total_stay_events)
        after = result.total_stay_events
        before_after_score = round(after / before, 3)
        before_after_pass = after > before

    checks = [
        {
            "name": "elderly_prefer_shade_and_seating",
            "pass": elderly_total == 0 or elderly_hits / max(1, elderly_total) >= 0.4,
            "score": round(elderly_hits / max(1, elderly_total), 3),
            "evidence": f"{elderly_name}: {elderly_hits}/{elderly_total} stay events near bench/tree/pavilion",
            "explanation": "Elderly agents should prefer shaded seating and resting places.",
        },
        {
            "name": "children_prefer_playground_and_lawn",
            "pass": child_total == 0 or child_hits / max(1, child_total) >= 0.35,
            "score": round(child_hits / max(1, child_total), 3),
            "evidence": f"{child_name}: {child_hits}/{child_total} stay events near playground/lawn",
            "explanation": "Child agents should respond to playable open-space facilities.",
        },
        {
            "name": "commuters_stay_less",
            "pass": not other_totals or commuter_total <= max(other_totals),
            "score": round(commuter_total / max(1, np.mean(other_totals) if other_totals else 1), 3),
            "evidence": f"{commuter_name}: stay events={commuter_total}, other persona mean={np.mean(other_totals) if other_totals else 0:.2f}",
            "explanation": "Commuter agents should be more pass-through oriented than leisure personas.",
        },
        {
            "name": "after_has_more_stay_events_than_before",
            "pass": before_after_pass,
            "score": before_after_score,
            "evidence": "requires baseline_result for before/after comparison",
            "explanation": "Improved facilities should increase potential stay events.",
        },
        {
            "name": "hotspots_near_high_attractiveness_facilities",
            "pass": event_count == 0 or high_attr_hits / max(1, event_count) >= 0.35,
            "score": round(high_attr_hits / max(1, event_count), 3),
            "evidence": f"{high_attr_hits}/{event_count} stay events near attractiveness >= 0.6",
            "explanation": "Stay hotspots should concentrate around high-attractiveness elements.",
        },
    ]
    valid_scores = [c["score"] for c in checks if isinstance(c.get("score"), (int, float))]
    pass_values = [c["pass"] for c in checks if c["pass"] is not None]
    return {
        "title": "StaySim paper validation report",
        "scenario": result.env_name or env.name,
        "engine_mode": result.engine_mode,
        "total_stay_events": result.total_stay_events,
        "mean_score": round(float(np.mean(valid_scores)), 3) if valid_scores else None,
        "pass_count": sum(1 for item in pass_values if item),
        "check_count": len(pass_values),
        "checks": checks,
    }


def save_run_history(env, result, config_dict, personas_dict, counts_dict, *, run_id=None):
    """每次 run_sim 完成后，自动保存结果到 output/runs/ 下带时间戳的文件夹"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    # A unique identifier prevents rapid runs from overwriting one another.
    if run_id is not None:
        # Persistent Agent executions reserve one UUID before running.
        from uuid import UUID
        if str(UUID(run_id)) != run_id:
            raise ValueError("run_id 必须是标准 UUID")
    folder_name = run_id or f"{timestamp}_{uuid4().hex}"
    run_dir = RUNS_DIR / folder_name
    run_dir.mkdir(parents=True, exist_ok=False)

    # Persist a known experiment contract, never arbitrary provider credentials.
    public_fields = {
        "scenario_file", "engine_requested", "engine_mode", "duration", "dt",
        "seed", "counts", "model", "llm_provider", "model_name",
        "duration_minutes", "random_seed",
    }
    config_dict = {key: value for key, value in config_dict.items() if key in public_fields}

    # 1. config.json —— 本次参数快照
    with open(run_dir / "config.json", 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=2)

    # 2. trajectories.csv —— 每条轨迹
    with open(run_dir / "trajectories.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["persona", "agent_idx", "step", "x", "y"])
        for pname, trajs in result.trajectories.items():
            for i, traj in enumerate(trajs):
                for step, (x, y) in enumerate(traj):
                    writer.writerow([pname, i, step, f"{x:.3f}", f"{y:.3f}"])

    # 3. decisions.jsonl —— 每条决策记录
    with open(run_dir / "decisions.jsonl", 'w', encoding='utf-8') as f:
        for pname, logs in result.decision_logs.items():
            for log in logs:
                f.write(json.dumps({"persona": pname, "decision": log}, ensure_ascii=False) + "\n")

    # 4. stay_events.csv —— paper metrics use full stay events, not only final states
    with open(run_dir / "stay_events.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["persona", "event_idx", "x", "y", "duration", "target_label", "target_type", "reason"])
        for pname, events in result.stay_events.items():
            for idx, event in enumerate(events):
                writer.writerow([
                    pname, idx, event.get("x", ""), event.get("y", ""),
                    event.get("duration", ""), event.get("target_label", ""),
                    event.get("target_type", ""), event.get("reason", ""),
                ])

    # 5. summary.csv —— one-row paper-ready result table
    summary_row = build_summary_row(env, result, config_dict)
    with open(run_dir / "summary.csv", 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
        writer.writeheader()
        writer.writerow(summary_row)

    # 6. validation_report.json —— five behavior-reasonableness checks
    validation_report = build_validation_report(env, result)
    with open(run_dir / "validation_report.json", 'w', encoding='utf-8') as f:
        json.dump(validation_report, f, ensure_ascii=False, indent=2)

    # 7. stats.json —— 汇总统计
    stay_counts = {}
    for pname, events in result.stay_events.items():
        stay_counts[pname] = len(events)
    stats = {
        "env_name": env.name,
        "env_width": env.width,
        "env_height": env.height,
        "n_agents": result.n_agents,
        "elapsed_seconds": result.elapsed,
        "engine_mode": result.engine_mode,
        "requested_engine_mode": getattr(result, "requested_engine_mode", result.engine_mode),
        "effective_engine_mode": getattr(result, "effective_engine_mode", result.engine_mode),
        "llm_calls": result.llm_calls,
        "total_tokens": result.total_tokens,
        "llm_failure_count": getattr(result, "llm_failure_count", 0),
        "fallback_count": getattr(result, "fallback_count", 0),
        "fallback_reasons": getattr(result, "fallback_reasons", []),
        "simulation_quality": getattr(result, "simulation_quality", "ok"),
        "total_stay_events": result.total_stay_events,
        "mean_stay_duration": result.mean_stay_duration,
        "stay_pass_ratio": result.stay_pass_ratio,
        "spatial_entropy": result.spatial_entropy,
        "cold_zone_ratio": result.cold_zone_ratio,
        "top_facility": result.top_facility,
        "facility_utilization": result.facility_utilization,
        "stay_counts": stay_counts,
        "persona_counts": counts_dict,
        "timestamp": timestamp,
    }
    if hasattr(env, 'source_import'):
        stats['source_import'] = env.source_import
    with open(run_dir / "stats.json", 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # 8. full_result.json —— 完整 SimResult 序列化（用于恢复图表）
    full_data = {
        "env_name": result.env_name,
        "n_agents": result.n_agents,
        "elapsed": result.elapsed,
        "engine_mode": result.engine_mode,
        "requested_engine_mode": getattr(result, "requested_engine_mode", result.engine_mode),
        "effective_engine_mode": getattr(result, "effective_engine_mode", result.engine_mode),
        "llm_calls": result.llm_calls,
        "total_tokens": result.total_tokens,
        "llm_failure_count": getattr(result, "llm_failure_count", 0),
        "fallback_count": getattr(result, "fallback_count", 0),
        "fallback_reasons": getattr(result, "fallback_reasons", []),
        "simulation_quality": getattr(result, "simulation_quality", "ok"),
        "spatial_entropy": result.spatial_entropy,
        "facility_utilization": result.facility_utilization,
        "cold_zone_ratio": result.cold_zone_ratio,
        "stay_pass_ratio": result.stay_pass_ratio,
        "mean_stay_duration": result.mean_stay_duration,
        "total_stay_events": result.total_stay_events,
        "top_facility": result.top_facility,
        "trajectories": {pname: [[(x, y) for (x, y) in traj] for traj in trajs]
                         for pname, trajs in result.trajectories.items()},
        "stay_points": {pname: [(x, y, d) for (x, y, d) in stays]
                        for pname, stays in result.stay_points.items()},
        "stay_events": result.stay_events,
        "decision_logs": result.decision_logs,
        "final_positions": [(x, y, p) for (x, y, p) in result.final_positions],
    }
    # Keep the exact environment with the run; historical charts must not depend
    # on a scene file that can subsequently change or disappear.
    type_names = {value: key for key, value in TYPE_MAP.items()}
    full_data['environment'] = {
        'name': env.name, 'width': env.width, 'height': env.height,
        'elements': [
            {'type': type_names[obj.element], 'x': obj.x, 'y': obj.y,
             'radius': obj.radius, 'attractiveness': obj.attractiveness,
             'shade': obj.shade, 'capacity': obj.capacity, 'label': obj.label}
            for obj in env.objects
        ],
    }
    if hasattr(env, 'source_import'):
        full_data['environment']['source_import'] = env.source_import
        full_data['source_import'] = env.source_import
    with open(run_dir / "full_result.json", 'w', encoding='utf-8') as f:
        json.dump(full_data, f, ensure_ascii=False)

    return str(run_dir)


def list_run_histories():
    """列出所有历史运行文件夹，按时间倒序"""
    if not RUNS_DIR.exists():
        return []
    dirs = sorted(
        [d for d in RUNS_DIR.iterdir() if d.is_dir()],
        key=lambda d: d.name,
        reverse=True
    )
    return [d.name for d in dirs]


def load_history_result(run_folder_name):
    """从历史文件夹加载 SimResult"""
    run_dir = RUNS_DIR / run_folder_name
    full_path = run_dir / "full_result.json"
    if not full_path.exists():
        return None, None, None

    with open(full_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    result = SimResult(
        env_name=data.get("env_name", ""),
        n_agents=data.get("n_agents", 0),
        elapsed=data.get("elapsed", 0),
        engine_mode=data.get("engine_mode", "rule"),
        llm_calls=data.get("llm_calls", 0),
        total_tokens=data.get("total_tokens", 0),
        spatial_entropy=data.get("spatial_entropy", 0.0),
        facility_utilization=data.get("facility_utilization", {}),
        cold_zone_ratio=data.get("cold_zone_ratio", 0.0),
        stay_pass_ratio=data.get("stay_pass_ratio", 0.0),
        mean_stay_duration=data.get("mean_stay_duration", 0.0),
        total_stay_events=data.get("total_stay_events", 0),
        top_facility=data.get("top_facility", ""),
    )
    result.requested_engine_mode = data.get("requested_engine_mode", result.engine_mode)
    result.effective_engine_mode = data.get("effective_engine_mode", result.engine_mode)
    result.llm_failure_count = data.get("llm_failure_count", 0)
    result.fallback_count = data.get("fallback_count", 0)
    result.fallback_reasons = data.get("fallback_reasons", [])
    result.simulation_quality = data.get("simulation_quality", "ok")
    if 'source_import' in data:
        result.source_import = data['source_import']
    result.trajectories = data.get("trajectories", {})
    result.stay_points = data.get("stay_points", {})
    result.stay_events = data.get("stay_events", {})
    result.decision_logs = data.get("decision_logs", {})
    result.final_positions = data.get("final_positions", [])

    # 加载 stats.json 获取场景信息
    stats_path = run_dir / "stats.json"
    env = environment_from_dict(data['environment']) if 'environment' in data else None
    config = {}
    if stats_path.exists():
        with open(stats_path, 'r', encoding='utf-8') as f:
            stats = json.load(f)
        config_path = run_dir / "config.json"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        # 尝试从场景文件重建 Environment
        scenario_name = config.get("scenario_file", "")
        if env is None and scenario_name:
            scenario_path = SCENARIO_DIR / scenario_name
            if scenario_path.exists():
                env = load_scenario(str(scenario_path))
        if env is None:
            # 用 stats 里的基本尺寸创建简单环境
            env = Environment(
                width=stats.get("env_width", 100),
                height=stats.get("env_height", 80),
                name=stats.get("env_name", run_folder_name)
            )

    return result, env, config


def _cn(size=10):
    """Return Chinese font properties if an installed font is available."""
    _, fm, font_path = _plot_modules()
    return fm.FontProperties(fname=font_path, size=size) if font_path else None


def _patch_axes(ax):
    """给 axes 打中文补丁 - 让 set_title/set_xlabel/text 自动用中文字体"""
    _, _, font_path = _plot_modules()
    if not font_path:
        return
    _orig_title = ax.set_title
    _orig_xlabel = ax.set_xlabel
    _orig_ylabel = ax.set_ylabel
    _orig_text = ax.text
    _orig_annotate = ax.annotate

    def _with_cn(fn):
        def wrapper(*args, **kw):
            if 'fontproperties' not in kw and 'fontdict' not in kw:
                kw['fontproperties'] = _cn(kw.pop('fontsize', 10))
            return fn(*args, **kw)
        return wrapper

    ax.set_title = _with_cn(_orig_title)
    ax.set_xlabel = _with_cn(_orig_xlabel)
    ax.set_ylabel = _with_cn(_orig_ylabel)
    ax.text = _with_cn(_orig_text)
    ax.annotate = _with_cn(_orig_annotate)


PERSONA_COLORS = {
    "买菜老人": "#E74C3C", "带娃家长": "#3498DB", "放学儿童": "#F39C12",
    "通勤青年": "#2ECC71", "社区闲散": "#9B59B6",
}


def plot_trajectories(env, result):
    """运动轨迹图"""
    plt, _, _ = _plot_modules()
    fig, ax = plt.subplots(figsize=(12, 9))
    _patch_axes(ax)
    ax.set_title("智能体运动轨迹与空间设施", fontsize=18, fontweight='bold', pad=15)
    ax.set_xlim(0, env.width); ax.set_ylim(0, env.height); ax.set_aspect('equal')

    element_styles = {
        SpaceElement.BENCH: ('#8B4513', '座椅'),
        SpaceElement.TREE: ('#228B22', '树荫'),
        SpaceElement.CHESS_TABLE: ('#8B6914', '棋桌'),
        SpaceElement.PLAYGROUND: ('#FF69B4', '儿童区'),
        SpaceElement.EXERCISE: ('#FF6347', '健身'),
        SpaceElement.PAVILION: ('#DAA520', '凉亭'),
        SpaceElement.FOUNTAIN: ('#00CED1', '水景'),
        SpaceElement.GREEN_LAWN: ('#90EE90', '草坪'),
        SpaceElement.FLOWER_BED: ('#FFB6C1', '花坛'),
        SpaceElement.ENTRANCE: ('#999999', '入口'),
    }

    for obj in env.objects:
        style = element_styles.get(obj.element, ('gray', '?'))
        ax.add_patch(plt.Circle((obj.x, obj.y), obj.radius, color=style[0],
                                 alpha=0.22, ec=style[0], linewidth=1.5))
        label = obj.label or obj.element.value
        ax.annotate(label, (obj.x, obj.y), fontsize=11, ha='center', va='center',
                   fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                             edgecolor=style[0], alpha=0.85))

    for pname, trajs in result.trajectories.items():
        color = PERSONA_COLORS.get(pname, 'black')
        for i, traj in enumerate(trajs[:5]):
            if len(traj) > 1:
                xs, ys = zip(*traj)
                ax.plot(xs, ys, color=color, alpha=0.45, linewidth=1.2,
                       label=pname if i == 0 else "")
        for traj in trajs[:5]:
            if traj:
                ax.scatter(traj[-1][0], traj[-1][1], color=color, s=50, zorder=3,
                          edgecolors='white', linewidth=1.5)

    ax.legend(fontsize=11, loc='upper right', ncol=2,
             framealpha=0.9, edgecolor='gray')
    ax.set_xlabel("X (米)", fontsize=13); ax.set_ylabel("Y (米)", fontsize=13)
    ax.grid(True, alpha=0.15, linestyle='--')
    plt.tight_layout(pad=2)
    return fig


def plot_heatmap(env, result):
    """热力图 —— 上方密度图满宽，下方解读"""
    plt, _, _ = _plot_modules()
    fig, (ax, ax_info) = plt.subplots(2, 1, figsize=(14, 14),
                                       gridspec_kw={'height_ratios': [3, 1]})
    _patch_axes(ax); _patch_axes(ax_info)
    ax.set_title("人群停留密度热力图", fontsize=22, fontweight='bold', pad=18)
    ax.set_xlim(0, env.width); ax.set_ylim(0, env.height); ax.set_aspect('equal')

    all_points = []
    for trajs in result.trajectories.values():
        for traj in trajs:
            all_points.extend(traj)

    if all_points:
        xs, ys = zip(*all_points)
        h = ax.hist2d(xs, ys, bins=50, range=[[0, env.width], [0, env.height]],
                       cmap='YlOrRd', alpha=0.9)
        cbar = plt.colorbar(h[3], ax=ax, shrink=0.8, pad=0.02)
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label('人次密度  (红色越深 → 停留越多)', fontsize=16)

    for obj in env.objects:
        lw = 2.5 if obj.attractiveness > 0.5 else 1.2
        ls = '-' if obj.attractiveness > 0.5 else '--'
        ax.add_patch(plt.Circle((obj.x, obj.y), obj.radius, color='none',
                                 ec='white', linewidth=lw, linestyle=ls))
        if obj.attractiveness > 0.5 or obj.element in [SpaceElement.BENCH, SpaceElement.CHESS_TABLE,
                                                         SpaceElement.PLAYGROUND, SpaceElement.PAVILION]:
            ax.annotate(obj.label or obj.element.value, (obj.x, obj.y + obj.radius + 3),
                       fontsize=13, ha='center', color='darkred', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85))

    ax.set_xlabel("X (米)", fontsize=16); ax.set_ylabel("Y (米)", fontsize=16)
    ax.tick_params(labelsize=13)
    ax.grid(True, alpha=0.1)

    # ── 下方：解读栏 ──
    ax_info.axis('off')
    ax_info.set_xlim(0, 1); ax_info.set_ylim(0, 1)

    hot_zones = []
    for obj in env.objects:
        pts = sum(1 for x, y in all_points
                  if math.sqrt((x-obj.x)**2+(y-obj.y)**2) < obj.radius+2)
        if pts > 0:
            hot_zones.append((obj, pts))
    hot_zones.sort(key=lambda t: t[1], reverse=True)

    # 横向拉宽内容
    inner_lines = []
    inner_lines.append("")
    inner_lines.append("    ═══════════ 人气排名（设施旁轨迹点数） ═══════════")
    inner_lines.append("")

    rank_line = ""
    for i, (obj, pts) in enumerate(hot_zones[:8]):
        rank_line += f"      {i+1}. {obj.label or obj.element.value}: {pts}人次       "
        if i % 2 == 1:
            inner_lines.append(rank_line)
            rank_line = ""
    if rank_line:
        inner_lines.append(rank_line)

    inner_lines.append("")
    inner_lines.append("    💡 热力集中在高吸引力设施周围，冷区即微更新可改善之处。")
    inner_lines.append("        如热力分布与实地观察的「人气点」一致 → 模拟可信。")
    inner_lines.append("")

    # 黄色框：竖向缩短、横向拉长
    ax_info.text(0.5, 0.95, "\n".join(inner_lines), transform=ax_info.transAxes,
                 fontsize=20, ha='center', va='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.6', facecolor='lightyellow',
                           edgecolor='orange', linewidth=2.5, alpha=0.9))

    # 标题在黄框上方
    ax_info.text(0.5, 1.03, "热力图解读", transform=ax_info.transAxes,
                 fontsize=24, fontweight='bold', ha='center', va='bottom',
                 fontfamily='monospace')

    plt.tight_layout(pad=3)
    return fig


def plot_stay_barchart(result):
    """停留柱状图 —— 显示全部5类人群，含平均停留时长"""
    plt, _, _ = _plot_modules()
    fig, ax = plt.subplots(figsize=(12, 7))
    _patch_axes(ax)
    ax.set_title("各人群停留事件统计（含平均停留时长）", fontsize=22, fontweight='bold', pad=20)

    # 确保全部5类人群都显示（即使停留数为0）
    all_personas = list(PERSONA_COLORS.keys())
    counts = [len(result.stay_events.get(n, [])) for n in all_personas]
    colors = [PERSONA_COLORS.get(n, 'gray') for n in all_personas]

    # 计算每类Persona的平均停留时长
    mean_durations = []
    for pname in all_personas:
        events = result.stay_events.get(pname, [])
        if events:
            mean_durations.append(sum(e.get("duration", 0) for e in events) / len(events))
        else:
            mean_durations.append(0)

    bars = ax.bar(range(len(all_personas)), counts, color=colors, edgecolor='white',
                  linewidth=2.5, width=0.55)
    ax.set_xticks(range(len(all_personas)))
    ax.set_xticklabels(all_personas, fontsize=15)
    ax.set_ylabel("停留事件数", fontsize=16)
    ax.tick_params(labelsize=14)
    ymax = max(counts) if max(counts) > 0 else 10
    for bar, c, md in zip(bars, counts, mean_durations):
        cx = bar.get_x() + bar.get_width() / 2
        # 停留事件数
        y_count = bar.get_height() + ymax * 0.03
        ax.text(cx, y_count, str(c),
                ha='center', fontweight='bold', fontsize=20,
                color=bar.get_facecolor())
        # 平均停留时长（放在次数上方）
        if c > 0:
            dur_str = f"均{md:.0f}s" if md >= 1 else "—"
            color_dur = '#666666'
            ax.text(cx, y_count + ymax * 0.06, dur_str,
                    ha='center', fontsize=12, color=color_dur)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.set_ylim(0, ymax * 1.38)
    plt.tight_layout(pad=2)
    return fig


def plot_decision_log(result):
    """决策日志 —— 引擎和内容同行，大字"""
    plt, _, _ = _plot_modules()
    llm_count = sum(1 for logs in result.decision_logs.values()
                    for l in logs if 'LLM' in l and 'LLM失败' not in l)
    llm_failed_count = sum(1 for logs in result.decision_logs.values()
                           for l in logs if 'LLM失败' in l)
    rule_count = sum(1 for logs in result.decision_logs.values()
                     for l in logs if '规则' in l)

    total_lines = 10
    for pname in PERSONA_COLORS:
        logs = result.decision_logs.get(pname, [])
        if logs:
            unique = list(dict.fromkeys(logs))
            total_lines += min(len(unique), 5) * 2 + 2
        else:
            total_lines += 2

    # 总行数 × 每行实际高度（18px字号≈0.25英寸/行）
    fig_height = max(4.5, total_lines * 0.25)
    fig, ax = plt.subplots(figsize=(18, fig_height))
    _patch_axes(ax)
    ax.axis('off')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    title = "智能体行为决策日志"
    if result.engine_mode == "llm":
        title += f"      🧠 LLM 决策 {llm_count} 条      ⚠️ LLM失败 {llm_failed_count} 条      ⚡ 规则决策 {rule_count} 条"
    ax.set_title(title, fontsize=24, fontweight='bold', pad=45, loc='center')

    COL1, COL2, COL3 = 10, 12, 45
    SEP = "─" * (COL1 + COL2 + COL3 + 8)

    def fmt(c1, c2, c3):
        return f"  {c1:<{COL1}}  {c2:<{COL2}}  {c3}"

    lines = []
    lines.append("")
    lines.append(SEP)
    lines.append(fmt("人群", "引擎", "决策内容 —— AI 实时思考过程"))
    lines.append(SEP)

    for pname in PERSONA_COLORS:
        logs = result.decision_logs.get(pname, [])
        if logs:
            unique = list(dict.fromkeys(logs))
            llm_logs = [l for l in unique if 'LLM' in l]
            rule_logs = [l for l in unique if '规则' in l]
            show = (llm_logs[-3:] + rule_logs[-2:])[-5:]

            for log in show:
                is_llm = 'LLM' in log
                src = "🧠 LLM 认知" if is_llm else "⚡ 规则 公式"
                content = log.replace("🧠LLM: ", "").replace("⚡规则: ", "")
                max_len = COL3 + 3
                if len(content) > max_len:
                    content = content[:max_len-3] + "..."
                lines.append(fmt(pname, src, content))
        else:
            lines.append(fmt(pname, "—", "（暂无决策记录）"))
        lines.append(SEP)

    if len(lines) <= 6:
        lines = ["", SEP, "    (暂无决策记录，请先运行模拟)", SEP]

    lines.append("")
    lines.append("    💡  🧠 LLM = 大语言模型认知推理    ⚡ 规则 = 数学公式决策    超时记录为 LLM失败")

    # 黄框居中（ha='center'），内容左对齐（monospace < 格式）
    ax.text(0.06, 0.93, "\n".join(lines), transform=ax.transAxes,
            fontsize=20, ha='left', va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=1.0', facecolor='lightyellow',
                      edgecolor='darkorange', linewidth=2, alpha=0.92))
    # 顶部留空间给标题，底部贴近黄框
    fig.subplots_adjust(top=0.90, bottom=0.0, left=0.02, right=0.98)
    return fig


def plot_summary(env, result):
    """统计摘要 + 验证 —— 标题在色块外，格式统一"""
    plt, _, _ = _plot_modules()
    fig = plt.figure(figsize=(22, 13))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.6, 2.5], hspace=0.38, wspace=0.12,
                          left=0.04, right=0.96, top=0.93, bottom=0.03)

    ax_top = fig.add_subplot(gs[0, :]); ax_top.axis('off')
    ax_tbl = fig.add_subplot(gs[1, 0]); ax_tbl.axis('off')
    ax_val = fig.add_subplot(gs[1, 1]); ax_val.axis('off')

    _patch_axes(ax_top); _patch_axes(ax_tbl); _patch_axes(ax_val)

    # ── 上部：左右对齐下栏边界 ──
    cost_str = ""
    if result.engine_mode == "llm":
        cost = result.total_tokens / 1000000 * 0.15
        cost_str = f"      LLM {result.llm_calls}次  Token {result.total_tokens}  费 ≈ ¥{cost*7.2:.3f}"

    info = (
        f"场景: {env.name}      {env.width}m×{env.height}m      设施 {len(env.objects)} 个      智能体 {result.n_agents} 人\n"
        f"引擎: {'🧠 LLM认知' if result.engine_mode=='llm' else '⚡ 规则'}      耗时 {result.elapsed:.2f}s{cost_str}"
    )
    ax_top.text(0.5, 0.48, info, transform=ax_top.transAxes,
                fontsize=24, ha='center', va='center', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=1.5', facecolor='lightblue',
                          edgecolor='steelblue', linewidth=2, alpha=0.7))
    ax_top.text(0.5, 0.94, "模拟统计摘要", transform=ax_top.transAxes,
                fontsize=26, fontweight='bold', ha='center', va='bottom',
                fontfamily='monospace')

    # ── 左下：人群行为表格 ──
    header = f"  {'人群':<10} {'特征':<10} {'均移动':<7} {'停留':<6} {'在场':<5} {'离场':<5}"
    sep_line = "  " + "─" * 42

    tbl = []
    tbl.append(sep_line)
    tbl.append(header)
    tbl.append(sep_line)
    for pname in PERSONA_COLORS:
        trajs = result.trajectories.get(pname, [[]])
        total_dist = 0
        for traj in trajs:
            for i in range(1, len(traj)):
                total_dist += math.sqrt((traj[i][0]-traj[i-1][0])**2+(traj[i][1]-traj[i-1][1])**2)
        avg_dist = total_dist / len(trajs) if trajs else 0
        n_stay = len(result.stay_events.get(pname, []))
        active = sum(1 for x,_,p in result.final_positions if p==pname and -5<=x<=env.width+5)
        left = sum(1 for x,_,p in result.final_positions if p==pname and (x<-5 or x>env.width+5))
        tag = "高停留" if n_stay > 3 else "高移动" if avg_dist > 80 else "多离场"
        tbl.append(f"  {pname:<10} {tag:<10} {avg_dist:>4.0f}m   {n_stay:>3}次  {active:>3}人  {left:>3}人")
    tbl.append(sep_line)

    ax_tbl.text(0.5, 0.50, "\n".join(tbl), transform=ax_tbl.transAxes,
                fontsize=22, ha='center', va='center', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=1.5', facecolor='honeydew',
                          edgecolor='seagreen', linewidth=2, alpha=0.8))
    ax_tbl.text(0.5, 0.94, "各人群行为统计", transform=ax_tbl.transAxes,
                fontsize=26, fontweight='bold', ha='center', va='bottom',
                fontfamily='monospace')

    # ── 右下：行为验证 ──
    validations = []
    e_stay = len(result.stay_events.get("买菜老人", []))
    y_stay = len(result.stay_events.get("通勤青年", []))
    validations.append((e_stay > y_stay, "老人停留 > 青年",
        f"老人{e_stay}次 vs 青年{y_stay}次"))
    c_pts = sum(len(t) for t in result.trajectories.get("放学儿童", [[]]))
    validations.append((c_pts > 50, "儿童活跃度高", f"轨迹点{c_pts} (>50)"))
    total_s = sum(len(v) for v in result.stay_events.values())
    validations.append((total_s > 3, "产生有效停留", f"共{total_s}次 (>3)"))
    hi_stay = 0
    for pname, events in result.stay_events.items():
        for ev in events:
            sx, sy = ev.get("x", 0), ev.get("y", 0)
            for obj in env.objects:
                if obj.attractiveness > 0.6 and math.sqrt((sx-obj.x)**2+(sy-obj.y)**2) < obj.radius+3:
                    hi_stay += 1; break
    validations.append((hi_stay > 0, "倾向高吸引力设施", f"{hi_stay}次"))
    leave_y = sum(1 for x,_,p in result.final_positions if p=="通勤青年" and (x<-5 or x>env.width+5))
    validations.append((leave_y > 0, "青年穿行离开", f"{leave_y}人离场"))

    val_sep = "  " + "─" * 42
    val_lines = ["", "  ═══════ 五项合理性检查 ═══════", ""]
    val_lines.append(val_sep)
    for ok, name, detail in validations:
        icon = "✅" if ok else "⚠️"
        val_lines.append(f"  {icon}  {name:<18} {detail}")
    val_lines.append(val_sep)
    val_lines.append("")
    passed = sum(1 for ok, _, _ in validations if ok)
    val_lines.append(f"  通过: {passed}/{len(validations)}")
    verdict = ("✅ 行为合理" if passed >= len(validations)-1 else
              "⚠️ 部分异常" if passed >= len(validations)-2 else "❌ 多项异常")
    val_lines.append(f"  {verdict}")
    val_lines.extend(["", "  判断标准: 行为差异合常识 | 热力与观察吻合 | 决策推理合理"])

    ax_val.text(0.5, 0.50, "\n".join(val_lines), transform=ax_val.transAxes,
                fontsize=22, ha='center', va='center', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=1.5', facecolor='lightyellow',
                          edgecolor='orange', linewidth=2, alpha=0.85))
    ax_val.text(0.5, 0.94, "行为合理性验证", transform=ax_val.transAxes,
                fontsize=26, fontweight='bold', ha='center', va='bottom',
                fontfamily='monospace')

    fig.tight_layout(pad=2)
    return fig
