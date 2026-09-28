"""StaySim Studio: Gradio UI and callbacks over the independent staysim package."""

import csv
import datetime
import json
import logging
import math
import os
import random
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

try:
    from scipy import stats as scipy_stats
    _HAS_SCIPY = True
except ImportError:
    scipy_stats = None
    _HAS_SCIPY = False

from staysim.domain import (
    Action, Agent, Decision, Environment, PersonaProfile, SCENARIO_DIR,
    SimResult, SpaceElement, SpaceObject, TYPE_MAP,
    get_default_personas, list_scenarios, load_scenario,
)
from staysim.cognition import CognitiveEngine
from staysim.simulation import apply_decision, run_sim, update_agent
from staysim.results import (
    OUTPUT_DIR, RUNS_DIR, PERSONA_COLORS, _cn, _patch_axes,
    build_summary_row, build_validation_report,
    list_run_histories, load_history_result, save_run_history,
    plot_trajectories, plot_heatmap, plot_stay_barchart, plot_decision_log, plot_summary,
)

CONFIG_FILE = Path(__file__).parent / "config.json"
LOG_DIR = Path(__file__).parent / "logs"

def load_config() -> dict:
    """加载保存的配置"""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_config(data: dict):
    """保存配置到文件"""
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_saved_api_config() -> tuple:
    """读取已保存的 API 设置 → (provider, api_key, model_name)"""
    cfg = load_config().get("llm_settings", {})
    env_api_key = os.environ.get("STAYSIM_API_KEY", "")
    return (
        os.environ.get("STAYSIM_API_PROVIDER", cfg.get("provider", "DeepSeek (V3)")),
        env_api_key or cfg.get("api_key", ""),
        os.environ.get("STAYSIM_MODEL_NAME", cfg.get("model_name", "")),
    )


def run_simulation_gui(scenario_file, n_elderly, n_parent, n_child, n_commuter, n_wanderer,
                        duration_minutes, random_seed, engine_mode, llm_provider, api_key, model_name,
                        ratio_mode, ratio_total,
                        gr_progress=gr.Progress()):
    """Gradio 回调：运行单次模拟（带进度条）"""
    if not scenario_file:
        return None, None, None, None, None, "❌ 请先选择一个场景", gr.update()

    yaml_path = SCENARIO_DIR / scenario_file
    env = load_scenario(str(yaml_path))
    personas = get_default_personas()

    # 按比例模式 → 将百分比转为人数
    if ratio_mode == "按比例":
        total = int(ratio_total)
        pct_sum = max(1, int(n_elderly) + int(n_parent) + int(n_child) + int(n_commuter) + int(n_wanderer))
        counts = {
            "买菜老人": max(0, round(total * int(n_elderly) / pct_sum)),
            "带娃家长": max(0, round(total * int(n_parent) / pct_sum)),
            "放学儿童": max(0, round(total * int(n_child) / pct_sum)),
            "通勤青年": max(0, round(total * int(n_commuter) / pct_sum)),
            "社区闲散": max(0, round(total * int(n_wanderer) / pct_sum)),
        }
    else:
        counts = {
            "买菜老人": int(n_elderly), "带娃家长": int(n_parent),
            "放学儿童": int(n_child), "通勤青年": int(n_commuter),
            "社区闲散": int(n_wanderer),
        }

    duration = int(duration_minutes) * 60
    seed = int(random_seed) if random_seed > 0 else None

    # 回调函数：更新进度条
    def progress_cb(ratio, msg):
        gr_progress(ratio, desc=msg)

    gr_progress(0, desc="正在初始化...")

    # 配置引擎模式
    if engine_mode == "llm":
        if not api_key or api_key.strip() == "":
            return None, None, None, None, None, "❌ LLM 模式需要填写 API Key", gr.update()
        api_base_map = {
            "OpenAI (GPT-4o-mini)": "https://api.openai.com/v1",
            "DeepSeek (V3)": "https://api.deepseek.com",
            "通义千问 (Qwen)": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "自定义": "https://api.openai.com/v1",
        }
        api_base = api_base_map.get(llm_provider, "https://api.openai.com/v1")
        model_map = {
            "OpenAI (GPT-4o-mini)": "gpt-4o-mini",
            "DeepSeek (V3)": "deepseek-chat",
            "通义千问 (Qwen)": "qwen-turbo",
            "自定义": model_name,
        }
        model = model_map.get(llm_provider, model_name)
        gr_progress(0.02, desc="🧠 LLM模式，初始化认知引擎...")
        result = run_sim(env, personas, counts, duration=duration, seed=seed,
                        engine_mode="llm", api_key=api_key.strip(),
                        api_base=api_base, model=model, progress=progress_cb)
    elif engine_mode == "local":
        gr_progress(0.02, desc="🖥 本地模型，连接 Ollama...")
        result = run_sim(env, personas, counts, duration=duration, seed=seed,
                        engine_mode="local", model="llama3", progress=progress_cb)
    else:
        gr_progress(0.05, desc="⚡ 规则模式，秒级计算...")
        result = run_sim(env, personas, counts, duration=duration, seed=seed,
                        progress=progress_cb)

    gr_progress(0.85, desc="🎨 生成可视化图表...")

    # 生成图表
    fig1 = plot_trajectories(env, result)
    gr_progress(0.90, desc="📊 生成热力图...")
    fig2 = plot_heatmap(env, result)
    fig3 = plot_stay_barchart(result)
    fig4 = plot_decision_log(result)
    fig5 = plot_summary(env, result)

    gr_progress(0.95, desc="💾 保存运行历史...")

    # ── 持久化：保存运行历史 ──
    config_dict = {
        "scenario_file": scenario_file,
        "engine_mode": engine_mode,
        "llm_provider": llm_provider if engine_mode == "llm" else "",
        "model_name": model_name if engine_mode == "llm" else "",
        "duration_minutes": int(duration_minutes),
        "random_seed": int(random_seed),
        "counts": counts,
    }
    saved_dir = save_run_history(env, result, config_dict, personas, counts)
    history_list = list_run_histories()

    gr_progress(0.98, desc="✅ 完成！")

    # 统计文本
    total_stay = sum(len(v) for v in result.stay_events.values())
    active_count = sum(1 for x,_,_ in result.final_positions if -5<x<105)
    mode_label = "🧠 LLM认知驱动" if result.engine_mode == "llm" else "⚡ 规则驱动"
    token_info = ""
    if result.engine_mode == "llm":
        cost_est = result.total_tokens / 1000000 * 0.15
        token_info = f"""
    · LLM 调用: {result.llm_calls} 次
    · Token 消耗: {result.total_tokens} tokens
    · 估算费用: ${cost_est:.4f} (≈¥{cost_est*7.2:.3f})"""

    stats_text = f"""
    📊 模拟完成！[{mode_label}]
    · 场景: {env.name}
    · 智能体: {result.n_agents} 人
    · 模拟时长: {duration_minutes} 分钟
    · 计算耗时: {result.elapsed:.2f} 秒
    · 总停留事件: {total_stay} 个
    · 最终在场: {active_count} 人{token_info}
    · 💾 已保存到: {saved_dir}
    """

    gr_progress(1.0, desc="✅ 模拟完成！")
    return fig1, fig2, fig3, fig4, fig5, stats_text, gr.update(choices=history_list, value=history_list[0] if history_list else None)


def compare_scenarios_gui(scenario_a, scenario_b, n_elderly, n_parent, n_child,
                           n_commuter, n_wanderer, duration_minutes, seed,
                           engine_mode, llm_provider, api_key, model_name):
    """Gradio 回调：方案对比"""
    if not scenario_a or not scenario_b:
        return None

    personas = get_default_personas()
    counts = {
        "买菜老人": int(n_elderly), "带娃家长": int(n_parent),
        "放学儿童": int(n_child), "通勤青年": int(n_commuter),
        "社区闲散": int(n_wanderer),
    }
    duration = int(duration_minutes) * 60
    seed_val = int(seed) if seed > 0 else None

    # 配置 LLM
    if engine_mode == "llm" and not (api_key and api_key.strip()):
        raise ValueError("LLM 模式需要 API Key，不能自动改用规则模式")
    engine_kwargs = {"engine_mode": engine_mode}
    if engine_mode == "llm" and api_key and api_key.strip():
        api_base_map = {
            "OpenAI (GPT-4o-mini)": "https://api.openai.com/v1",
            "DeepSeek (V3)": "https://api.deepseek.com",
            "通义千问 (Qwen)": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "自定义": "https://api.openai.com/v1",
        }
        model_map = {
            "OpenAI (GPT-4o-mini)": "gpt-4o-mini",
            "DeepSeek (V3)": "deepseek-chat",
            "通义千问 (Qwen)": "qwen-turbo",
            "自定义": model_name,
        }
        engine_kwargs = {
            "engine_mode": "llm", "api_key": api_key.strip(),
            "api_base": api_base_map.get(llm_provider, "https://api.openai.com/v1"),
            "model": model_map.get(llm_provider, model_name),
        }

    # 场景A
    env_a = load_scenario(str(SCENARIO_DIR / scenario_a))
    result_a = run_sim(env_a, personas, counts, duration=duration, seed=seed_val, **engine_kwargs)

    # 场景B
    if seed_val: random.seed(seed_val); np.random.seed(seed_val)
    env_b = load_scenario(str(SCENARIO_DIR / scenario_b))
    result_b = run_sim(env_b, personas, counts, duration=duration, seed=seed_val, **engine_kwargs)

    # 对比图
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for _ax in axes.flat: _patch_axes(_ax)
    fig.suptitle(f"方案对比: {env_a.name} vs {env_b.name}", fontsize=14, fontweight='bold',
                 fontproperties=_cn(14))

    # (0,0) 场景A轨迹
    ax = axes[0,0]
    ax.set_title(f"A: {env_a.name}", fontsize=12)
    ax.set_xlim(0, max(env_a.width, env_b.width))
    ax.set_ylim(0, max(env_a.height, env_b.height))
    ax.set_aspect('equal')
    for obj in env_a.objects:
        ax.add_patch(plt.Circle((obj.x, obj.y), obj.radius, color='gray', alpha=0.2, ec='gray'))
    for pname, trajs in result_a.trajectories.items():
        for traj in trajs[:3]:
            if len(traj) > 1:
                xs, ys = zip(*traj)
                ax.plot(xs, ys, color=PERSONA_COLORS.get(pname,'black'), alpha=0.4, linewidth=0.7)

    # (0,1) 场景B轨迹
    ax = axes[0,1]
    ax.set_title(f"B: {env_b.name}", fontsize=12)
    ax.set_xlim(0, max(env_a.width, env_b.width))
    ax.set_ylim(0, max(env_a.height, env_b.height))
    ax.set_aspect('equal')
    for obj in env_b.objects:
        ax.add_patch(plt.Circle((obj.x, obj.y), obj.radius, color='gray', alpha=0.2, ec='gray'))
    for pname, trajs in result_b.trajectories.items():
        for traj in trajs[:3]:
            if len(traj) > 1:
                xs, ys = zip(*traj)
                ax.plot(xs, ys, color=PERSONA_COLORS.get(pname,'black'), alpha=0.4, linewidth=0.7)

    # (1,0) 停留事件对比
    ax = axes[1,0]
    ax.set_title("停留事件数对比", fontsize=12)
    persona_names = list(counts.keys())
    x = np.arange(len(persona_names))
    w = 0.35
    b1 = [len(result_a.stay_events.get(p,[])) for p in persona_names]
    b2 = [len(result_b.stay_events.get(p,[])) for p in persona_names]
    ax.bar(x-w/2, b1, w, label=env_a.name[:8], color='salmon', alpha=0.8)
    ax.bar(x+w/2, b2, w, label=env_b.name[:8], color='lightgreen', alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(persona_names, fontsize=9)
    ax.legend(fontsize=8)
    for i,(v1,v2) in enumerate(zip(b1,b2)):
        if v1>0:
            pct = (v2-v1)/v1*100
            ax.text(i, max(v1,v2)+0.5, f"{'+'if pct>0 else ''}{pct:.0f}%", ha='center', fontsize=9,
                   color='green' if pct>0 else 'red', fontweight='bold')

    # (1,1) 统计摘要
    ax = axes[1,1]
    ax.set_title("对比摘要", fontsize=12)
    ax.axis('off')
    total_a = sum(len(v) for v in result_a.stay_events.values())
    total_b = sum(len(v) for v in result_b.stay_events.values())
    lines = [
        f"场景A: {total_a} 次停留 | 场景B: {total_b} 次停留",
        f"改善幅度: {(total_b-total_a)/max(1,total_a)*100:+.0f}%",
        "", "场景A空间元素:", *[f"  · {o.label or o.element.value}" for o in env_a.objects[:8]],
        "", "场景B空间元素:", *[f"  · {o.label or o.element.value}" for o in env_b.objects[:8]],
        "", f"结论: 场景B停留事件{'+'if total_b>total_a else ''}{(total_b-total_a)/max(1,total_a)*100:.0f}%"
    ]
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes, fontsize=9,
            va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    plt.tight_layout()
    return fig


# ============================================================
# 主界面构建
# ============================================================

def save_llm_settings(provider, api_key, model_name):
    """保存 LLM 非敏感设置到配置文件。API Key 请使用 STAYSIM_API_KEY 环境变量。"""
    cfg = load_config()
    cfg["llm_settings"] = {
        "provider": provider,
        "model_name": model_name,
    }
    save_config(cfg)
    return gr.update(visible=True, value="✅ 已保存供应商和模型名。API Key 不会写入配置文件")


def load_history_gui(folder_name):
    """从历史记录加载并恢复所有图表"""
    if not folder_name:
        return None, None, None, None, None, "请选择一个历史记录"

    result, env, config = load_history_result(folder_name)
    if result is None:
        return None, None, None, None, None, "❌ 历史数据加载失败"

    # 重新生成图表
    fig1 = plot_trajectories(env, result)
    fig2 = plot_heatmap(env, result)
    fig3 = plot_stay_barchart(result)
    fig4 = plot_decision_log(result)
    fig5 = plot_summary(env, result)

    # 恢复统计文本
    total_stay = sum(len(v) for v in result.stay_events.values())
    active_count = sum(1 for x, _, _ in result.final_positions if -5 < x < 105)
    mode_label = "🧠 LLM认知驱动" if result.engine_mode == "llm" else "⚡ 规则驱动"
    token_info = ""
    if result.engine_mode == "llm" and result.total_tokens > 0:
        cost_est = result.total_tokens / 1000000 * 0.15
        token_info = f"""
    · LLM 调用: {result.llm_calls} 次
    · Token 消耗: {result.total_tokens} tokens
    · 估算费用: ${cost_est:.4f} (≈¥{cost_est * 7.2:.3f})"""

    stats_text = f"""
    📂 历史记录回放 [{mode_label}]
    · 场景: {env.name}
    · 智能体: {result.n_agents} 人
    · 计算耗时: {result.elapsed:.2f} 秒
    · 总停留事件: {total_stay} 个
    · 最终在场: {active_count} 人{token_info}
    · 📁 来源: {folder_name}
    """

    # 尝试恢复参数到界面（场景、人数等）
    cfg = config or {}
    scenario_val = cfg.get("scenario_file", None)
    counts = cfg.get("counts", {})
    engine_val = cfg.get("engine_mode", "rule")
    duration_val = cfg.get("duration_minutes", 10)
    seed_val = cfg.get("random_seed", 0)

    return (
        fig1, fig2, fig3, fig4, fig5, stats_text,
        gr.update(value=scenario_val) if scenario_val else gr.update(),
        gr.update(value=counts.get("买菜老人", 8)),
        gr.update(value=counts.get("带娃家长", 8)),
        gr.update(value=counts.get("放学儿童", 8)),
        gr.update(value=counts.get("通勤青年", 8)),
        gr.update(value=counts.get("社区闲散", 8)),
        gr.update(value=duration_val),
        gr.update(value=seed_val),
        gr.update(value=engine_val),
    )


# ─── 批量实验预设配比 ───

# ─── 时间段配比预设 ───
# 早晨老人多、傍晚青年多

TIME_PERIOD_PRESETS = {
    "早晨 (7:00-9:00)": (12, 3, 2, 2, 8),   # 老人买菜晨练、闲散多
    "上午 (9:00-12:00)": (8, 10, 2, 5, 6),   # 带娃家长出没、青年上班
    "下午 (12:00-17:00)": (6, 6, 10, 3, 6),  # 儿童放学、均衡
    "傍晚 (17:00-20:00)": (4, 4, 4, 14, 6),  # 青年下班通勤
}


def apply_time_period(time_period):
    """根据所选时间段，返回推荐的人群配比滑块值"""
    preset = TIME_PERIOD_PRESETS.get(time_period)
    if preset is None:
        # 未选择 → 返回默认均衡值
        return [gr.update(value=8)] * 5
    return [
        gr.update(value=preset[0]),  # 买菜老人
        gr.update(value=preset[1]),  # 带娃家长
        gr.update(value=preset[2]),  # 放学儿童
        gr.update(value=preset[3]),  # 通勤青年
        gr.update(value=preset[4]),  # 社区闲散
    ]


RATIO_PRESETS = {
    "均衡": (6, 6, 6, 6, 6),
    "老人为主": (15, 5, 3, 3, 4),
    "带娃家长为主": (4, 14, 8, 2, 2),
    "儿童为主": (2, 3, 18, 2, 5),
    "青年为主": (2, 2, 2, 20, 4),
    "闲散为主": (5, 3, 2, 4, 16),
    "高密度": (12, 12, 12, 12, 12),
    "低密度": (2, 2, 2, 2, 2),
}


def run_batch_experiment(scenario_files, ratio_presets, repeat_count,
                          duration_minutes, engine_mode, llm_provider, api_key, model_name,
                          gr_progress=gr.Progress()):
    """批量实验：遍历 场景×配比×重复 的所有组合"""
    if not scenario_files:
        return None, None, "❌ 请至少选择一个场景"
    if not ratio_presets:
        return None, None, "❌ 请至少选择一个配比方案"

    personas = get_default_personas()
    duration = int(duration_minutes) * 60

    # 构建引擎配置
    engine_kwargs = {}
    if engine_mode == "llm":
        if not api_key or not api_key.strip():
            return None, None, "❌ LLM 模式需要填写 API Key"
        api_base_map = {
            "OpenAI (GPT-4o-mini)": "https://api.openai.com/v1",
            "DeepSeek (V3)": "https://api.deepseek.com",
            "通义千问 (Qwen)": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "自定义": "https://api.openai.com/v1",
        }
        model_map = {
            "OpenAI (GPT-4o-mini)": "gpt-4o-mini",
            "DeepSeek (V3)": "deepseek-chat",
            "通义千问 (Qwen)": "qwen-turbo",
            "自定义": model_name,
        }
        engine_kwargs = {
            "engine_mode": "llm",
            "api_key": api_key.strip(),
            "api_base": api_base_map.get(llm_provider, "https://api.openai.com/v1"),
            "model": model_map.get(llm_provider, model_name),
        }
    elif engine_mode == "local":
        engine_kwargs = {
            "engine_mode": "local",
            "model": "llama3",
        }

    total_combos = len(scenario_files) * len(ratio_presets) * int(repeat_count)
    rows = []
    combo_idx = 0

    for scenario_file in scenario_files:
        env = load_scenario(str(SCENARIO_DIR / scenario_file))

        for preset_name in ratio_presets:
            preset = RATIO_PRESETS.get(preset_name)
            if preset is None:
                continue  # 跳过自定义（由用户从界面输入）
            counts = {
                "买菜老人": preset[0], "带娃家长": preset[1],
                "放学儿童": preset[2], "通勤青年": preset[3],
                "社区闲散": preset[4],
            }

            for rep in range(int(repeat_count)):
                combo_idx += 1
                seed = combo_idx * 42  # 可复现且不同
                msg = f"🧪 组合 {combo_idx}/{total_combos}: {scenario_file} × {preset_name} × 第{rep+1}次"
                gr_progress(combo_idx / total_combos, desc=msg)

                result = run_sim(env, personas, counts, duration=duration, seed=seed, **engine_kwargs)

                stay_per_persona = {}
                for pname in ["买菜老人", "带娃家长", "放学儿童", "通勤青年", "社区闲散"]:
                    stay_per_persona[pname] = len(result.stay_events.get(pname, []))

                total_stay = sum(stay_per_persona.values())

                rows.append({
                    "场景": scenario_file.replace(".yaml", ""),
                    "配比": preset_name,
                    "引擎": result.engine_mode,
                    "总停留": total_stay,
                    "老人停留": stay_per_persona["买菜老人"],
                    "家长停留": stay_per_persona["带娃家长"],
                    "儿童停留": stay_per_persona["放学儿童"],
                    "青年停留": stay_per_persona["通勤青年"],
                    "闲散停留": stay_per_persona["社区闲散"],
                    "Token": result.total_tokens,
                    "耗时(s)": f"{result.elapsed:.1f}",
                })

    gr_progress(1.0, desc="📊 生成汇总...")

    # 构建汇总表格 DataFrame
    import pandas as pd
    df = pd.DataFrame(rows)

    # 生成统计检验图
    fig = _plot_batch_stats(df, scenario_files, ratio_presets)

    stats_text = f"✅ 批量实验完成！共 {len(rows)} 次运行（{len(scenario_files)}场景 × {len(ratio_presets)}配比 × {int(repeat_count)}重复）"

    return df, fig, stats_text


def _plot_batch_stats(df, scenario_files, ratio_presets):
    """批量实验统计图：按场景×配比 分组对比停留数 + Mann-Whitney U 统计检验"""
    import matplotlib.pyplot as plt

    n_groups = len(scenario_files) * len(ratio_presets)
    # 三行布局：上排两个图，下排统计检验结果
    fig = plt.figure(figsize=(max(16, n_groups * 1.5), 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1.2], hspace=0.35, wspace=0.25,
                          left=0.06, right=0.97, top=0.94, bottom=0.04)

    ax = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax_mw = fig.add_subplot(gs[1, :])
    _patch_axes(ax); _patch_axes(ax2); ax_mw.axis('off')

    # ── 左图：按场景+配比 总停留箱线图 ──
    ax.set_title("各场景×配比 总停留事件分布", fontsize=16, fontweight='bold')

    group_labels = []
    group_data = []
    for sf in scenario_files:
        for rp in ratio_presets:
            subset = df[(df["场景"] == sf.replace(".yaml", "")) & (df["配比"] == rp)]
            if len(subset) > 0:
                label = f"{sf.replace('.yaml','')[:6]}\n{rp}"
                group_labels.append(label)
                group_data.append(subset["总停留"].values)

    if group_data:
        bp = ax.boxplot(group_data, labels=group_labels, patch_artist=True,
                        showmeans=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        ax.tick_params(axis='x', fontsize=8, rotation=45)
        ax.set_ylabel("总停留事件数", fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')

    # ── 右图：各 Persona 平均停留堆叠柱状图 ──
    ax2.set_title("各场景×配比 各人群平均停留", fontsize=16, fontweight='bold')

    persona_cols = ["老人停留", "家长停留", "儿童停留", "青年停留", "闲散停留"]
    persona_labels = ["买菜老人", "带娃家长", "放学儿童", "通勤青年", "社区闲散"]
    bar_colors = [PERSONA_COLORS[l] for l in persona_labels]

    stacked_labels = []
    stacked_data = {p: [] for p in persona_cols}
    for sf in scenario_files:
        for rp in ratio_presets:
            subset = df[(df["场景"] == sf.replace(".yaml", "")) & (df["配比"] == rp)]
            if len(subset) > 0:
                label = f"{sf.replace('.yaml','')[:6]}\n{rp}"
                stacked_labels.append(label)
                for pc in persona_cols:
                    stacked_data[pc].append(subset[pc].mean())

    if stacked_labels:
        x = range(len(stacked_labels))
        bottom = np.zeros(len(stacked_labels))
        for i, pc in enumerate(persona_cols):
            vals = stacked_data[pc]
            ax2.bar(x, vals, bottom=bottom, color=bar_colors[i],
                    label=persona_labels[i], alpha=0.8, width=0.6)
            bottom += np.array(vals)

        ax2.set_xticks(list(x))
        ax2.set_xticklabels(stacked_labels, fontsize=8, rotation=45)
        ax2.set_ylabel("平均停留事件数", fontsize=14)
        ax2.legend(fontsize=9, loc='upper right')
        ax2.grid(True, alpha=0.3, axis='y')

    # ── 下方：Mann-Whitney U 统计检验 ──
    mw_lines = []
    mw_lines.append("  ═══════════ Mann-Whitney U 统计检验 ═══════════")
    mw_lines.append("")
    mw_lines.append(f"  {'对比方案':<28} {'U统计量':>8} {'p值':>10}  {'显著性':>10}")
    mw_lines.append("  " + "─" * 60)

    # 场景间两两对比（同配比下）
    if _HAS_SCIPY and len(scenario_files) >= 2:
        for rp in ratio_presets:
            for i in range(len(scenario_files)):
                for j in range(i + 1, len(scenario_files)):
                    d1 = df[(df["场景"] == scenario_files[i].replace(".yaml", "")) & (df["配比"] == rp)]["总停留"].values
                    d2 = df[(df["场景"] == scenario_files[j].replace(".yaml", "")) & (df["配比"] == rp)]["总停留"].values
                    if len(d1) >= 3 and len(d2) >= 3:
                        try:
                            u_stat, p_val = scipy_stats.mannwhitneyu(d1, d2, alternative='two-sided')
                            sig = "*** p<0.001" if p_val < 0.001 else "** p<0.01" if p_val < 0.01 else "* p<0.05" if p_val < 0.05 else "n.s."
                            label = f"{scenario_files[i].replace('.yaml','')[:10]} vs {scenario_files[j].replace('.yaml','')[:10]} [{rp}]"
                            mw_lines.append(f"  {label:<28} {u_stat:>8.1f} {p_val:>10.4f}  {sig:>10}")
                        except Exception:
                            pass
    elif not _HAS_SCIPY and len(scenario_files) >= 2:
        mw_lines.append("  （scipy 未安装，无法进行 Mann-Whitney U 检验）")

    if len(mw_lines) <= 4:
        mw_lines.append("  （数据不足，无法进行统计检验 → 请增加重复次数 ≥3）")

    mw_lines.append("")
    mw_lines.append("  判断标准: p < 0.05(*) p < 0.01(**) p < 0.001(***) n.s.(不显著)")

    ax_mw.text(0.5, 0.5, "\n".join(mw_lines), transform=ax_mw.transAxes,
               fontsize=14, ha='center', va='center', fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=1.0', facecolor='lightyellow',
                        edgecolor='darkorange', linewidth=2, alpha=0.9))

    return fig


def validate_with_field_data(csv_file, history_folder, gr_progress=gr.Progress()):
    """实地数据导入验证：上传 CSV，与模拟结果对比热力图分布"""
    if csv_file is None:
        return None, "❌ 请上传实地数据 CSV 文件"
    if not history_folder:
        return None, "❌ 请选择一个模拟历史记录用于对比"

    # 加载模拟结果
    result, env, config = load_history_result(history_folder)
    if result is None:
        return None, "❌ 历史数据加载失败"

    # 读取实地 CSV
    try:
        field_data = pd.read_csv(csv_file.name)
    except Exception as e:
        return None, f"❌ CSV 读取失败: {e}"

    # 检查必要列
    required_cols = ["x", "y"]
    for col in required_cols:
        if col not in field_data.columns:
            return None, f"❌ CSV 缺少必要列: {col}（需要: 时间,x坐标,y坐标,人群类型）"

    # 构建模拟热力图数据
    sim_points = []
    for trajs in result.trajectories.values():
        for traj in trajs:
            sim_points.extend(traj)

    if not sim_points:
        return None, "❌ 模拟数据无轨迹点"

    # 构建实地数据点
    field_xs = field_data["x"].values
    field_ys = field_data["y"].values

    # 公共网格（用于计算分布重合度）
    bins = 20
    x_range = [0, env.width]
    y_range = [0, env.height]

    sim_hist, x_edges, y_edges = np.histogram2d(
        [p[0] for p in sim_points], [p[1] for p in sim_points],
        bins=bins, range=[x_range, y_range])
    field_hist, _, _ = np.histogram2d(
        field_xs, field_ys, bins=bins, range=[x_range, y_range])

    # 归一化 → 概率分布
    sim_flat = sim_hist.flatten()
    field_flat = field_hist.flatten()
    sim_flat = sim_flat / (sim_flat.sum() + 1e-10)
    field_flat = field_flat / (field_flat.sum() + 1e-10)

    # ── Jensen-Shannon Divergence ──
    m_flat = (sim_flat + field_flat) / 2
    kl_sim = np.sum(sim_flat * np.log((sim_flat + 1e-10) / (m_flat + 1e-10)))
    kl_field = np.sum(field_flat * np.log((field_flat + 1e-10) / (m_flat + 1e-10)))
    jsd = (kl_sim + kl_field) / 2

    # ── 余弦相似度 ──
    cos_sim = np.dot(sim_flat, field_flat) / (
        np.linalg.norm(sim_flat) * np.linalg.norm(field_flat) + 1e-10)

    # ── 生成叠加对比图 ──
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    for _ax in axes: _patch_axes(_ax)

    # 左：模拟热力图
    ax = axes[0]
    ax.set_title(f"模拟热力图\n({env.name})", fontsize=14, fontweight='bold')
    ax.set_xlim(0, env.width); ax.set_ylim(0, env.height); ax.set_aspect('equal')
    if sim_points:
        xs, ys = zip(*sim_points)
        ax.hist2d(xs, ys, bins=30, range=[x_range, y_range], cmap='YlOrRd')
    for obj in env.objects:
        ax.add_patch(plt.Circle((obj.x, obj.y), obj.radius, color='none',
                                ec='white', linewidth=0.8, linestyle='--', alpha=0.5))

    # 中：实地数据热力图
    ax2 = axes[1]
    ax2.set_title(f"实地观测热力图\n({len(field_xs)}个数据点)", fontsize=14, fontweight='bold')
    ax2.set_xlim(0, env.width); ax2.set_ylim(0, env.height); ax2.set_aspect('equal')
    ax2.hist2d(field_xs, field_ys, bins=30, range=[x_range, y_range], cmap='Blues')
    for obj in env.objects:
        ax2.add_patch(plt.Circle((obj.x, obj.y), obj.radius, color='none',
                                 ec='white', linewidth=0.8, linestyle='--', alpha=0.5))

    # 右：叠加对比 + 指标
    ax3 = axes[2]
    ax3.set_title("分布重合度评估", fontsize=14, fontweight='bold')
    ax3.axis('off')
    ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)

    # 重合度评级
    if jsd < 0.1:
        grade = "🟢 优秀 — 模拟与实地高度吻合"
    elif jsd < 0.3:
        grade = "🟡 良好 — 趋势一致，局部有偏差"
    elif jsd < 0.5:
        grade = "🟠 一般 — 需要调整参数"
    else:
        grade = "🔴 较差 — 模拟与实地分布差异大"

    info_lines = [
        "  ═══════════ 分布重合度评估 ═══════════",
        "",
        f"  Jensen-Shannon Divergence (JSD):  {jsd:.4f}",
        f"  Cosine Similarity (余弦相似度):   {cos_sim:.4f}",
        "",
        f"  重合度评级:  {grade}",
        "",
        "  ─────────── 解读 ───────────",
        "  JSD ∈ [0, ln2]: 越接近0越相似",
        "  余弦相似度 ∈ [0, 1]: 越接近1越相似",
        f"  模拟数据点: {len(sim_points)}  实地数据点: {len(field_xs)}",
    ]
    ax3.text(0.05, 0.92, "\n".join(info_lines), transform=ax3.transAxes,
             fontsize=13, ha='left', va='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=1.0', facecolor='lightyellow',
                      edgecolor='orange', linewidth=2, alpha=0.9))

    plt.tight_layout(pad=3)
    return fig, f"✅ 验证完成 | JSD={jsd:.4f} | 余弦相似度={cos_sim:.4f} | {grade}"


def save_scene_editor(scene_name, width, height, elements_text, existing_scenes):
    """场景编辑器：从表单数据生成并保存 YAML 文件"""
    if not scene_name or not scene_name.strip():
        return gr.update(), "❌ 请输入场景名称", gr.update()
    if not elements_text or not elements_text.strip():
        return gr.update(), "❌ 请至少添加一个设施", gr.update()

    try:
        lines = [l.strip() for l in elements_text.strip().split('\n') if l.strip()]
        elements = []
        for line in lines:
            # 格式: type, x, y, radius, attractiveness, shade, capacity, label
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 5:
                continue
            elem = {
                "type": parts[0],
                "x": float(parts[1]),
                "y": float(parts[2]),
                "radius": float(parts[3]) if len(parts) > 3 else 2.0,
                "attractiveness": float(parts[4]) if len(parts) > 4 else 0.5,
                "shade": parts[5].strip().lower() == 'true' if len(parts) > 5 else False,
                "capacity": int(parts[6]) if len(parts) > 6 else 10,
                "label": parts[7] if len(parts) > 7 else parts[0],
            }
            elements.append(elem)

        yaml_data = {
            "name": scene_name.strip(),
            "description": f"用户自定义场景: {scene_name}",
            "width": float(width),
            "height": float(height),
            "elements": elements,
        }

        # 生成安全文件名
        safe_name = scene_name.strip().replace(' ', '_').replace('/', '_')
        file_path = SCENARIO_DIR / f"{safe_name}.yaml"

        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

        # 刷新场景列表
        updated_scenes = list_scenarios()
        return (
            gr.update(choices=updated_scenes, value=updated_scenes[-1] if updated_scenes else None),
            f"✅ 场景已保存: {file_path}",
            gr.update(choices=updated_scenes, value=None),
        )
    except Exception as e:
        return gr.update(), f"❌ 保存失败: {e}", gr.update()


def generate_animation_gif(history_folder, fps=8):
    """从历史记录生成智能体运动动画 GIF"""
    if not history_folder:
        return None, "请先运行模拟或选择历史记录"

    result, env, _ = load_history_result(history_folder)
    if result is None:
        return None, "❌ 历史数据加载失败"

    import matplotlib.animation as animation

    # 准备数据：每个智能体的轨迹按时间步展开
    all_trajs = []
    for pname, trajs in result.trajectories.items():
        for traj in trajs[:10]:  # 最多显示10个智能体的轨迹
            all_trajs.append((pname, traj))

    if not all_trajs:
        return None, "❌ 无轨迹数据"

    # 找最大步数
    max_steps = max(len(t) for _, t in all_trajs)

    fig, ax = plt.subplots(figsize=(10, 8))
    _patch_axes(ax)
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect('equal')
    ax.set_title(f"智能体运动动画 - {env.name}", fontsize=16, fontweight='bold')

    # 绘制静态设施
    for obj in env.objects:
        ax.add_patch(plt.Circle((obj.x, obj.y), obj.radius, color='gray',
                                alpha=0.2, ec='gray', linewidth=1))

    # 创建散点（每个智能体一个点）
    scatters = []
    for pname, _ in all_trajs:
        color = PERSONA_COLORS.get(pname, 'black')
        sc = ax.scatter([], [], s=80, color=color, edgecolors='white',
                        linewidth=1.5, label=pname, zorder=3)
        scatters.append(sc)

    # 去重图例
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=9, loc='upper right')

    def update(frame):
        for i, (_, traj) in enumerate(all_trajs):
            if frame < len(traj):
                scatters[i].set_offsets([traj[frame]])
            else:
                scatters[i].set_offsets(np.empty((0, 2)))
        return scatters

    # 生成 GIF
    gif_path = OUTPUT_DIR / f"animation_{history_folder[:20]}.gif"
    anim = animation.FuncAnimation(fig, update, frames=max_steps,
                                   interval=1000//fps, blit=True, repeat=True)
    anim.save(str(gif_path), writer='pillow', fps=fps, dpi=80)
    plt.close(fig)

    return str(gif_path), f"✅ GIF 已生成: {gif_path}"


def create_app(*, agent_db=None, selected_tab=None):
    scenarios = list_scenarios()
    default_scenario = scenarios[0] if scenarios else None

    # 加载已保存的配置
    saved_provider, saved_key, saved_model = get_saved_api_config()

    with gr.Blocks(title="StaySim Studio · 驻留行为智能模拟器") as app:
        gr.Markdown("""
        # 🚶 StaySim Studio
        ## 驻留行为智能模拟器

        **无需编程知识。** 用 Agent 描述需求并确认计划，或手动选择场景与参数，完成模拟和结果分析。

        ---
        ### 🧠 两种决策引擎可选：
        - **⚡ 规则引擎**（默认）：基于数学模型的认知决策，0 token 消耗，秒出结果，适合快速实验
        - **🧠 LLM 引擎**：调用大语言模型生成行人决策；模拟行为需要进一步验证，失败兜底会记录在结果中
        """)

        with gr.Tabs(selected=selected_tab):
            with gr.TabItem("🤖 Agent 工作台", id="agent"):
                from staysim_agent.ui import render_agent_tab
                from staysim_agent.session import DEFAULT_DB
                agent_scene_dd = render_agent_tab(agent_db if agent_db is not None else DEFAULT_DB)
            with gr.TabItem("📐 场景导入", id="scene_import"):
                from staysim_agent.ui import render_scene_import_tab
                bind_scene_catalog = render_scene_import_tab()
            # ======== Tab 1: 单次模拟 ========
            with gr.TabItem("🔬 单次模拟", id="manual"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### ⚙️ 场景与人群")

                        scenario_dd = gr.Dropdown(
                            choices=scenarios, value=default_scenario,
                            label="📐 选择场景", interactive=True
                        )

                        gr.Markdown("### 👥 人群配比（拖动滑块）")
                        ratio_mode_toggle = gr.Radio(
                            choices=["按人数", "按比例"],
                            value="按人数",
                            label="配比模式",
                            info="按人数：直接设各类人群数量 | 按比例：设总人数+各类百分比"
                        )
                        ratio_total = gr.Slider(10, 100, 40, step=5, label="👥 总人数（按比例模式）", visible=False)

                        with gr.Row():
                            n_elderly = gr.Slider(0, 30, 8, step=1, label="🧓 买菜老人")
                            n_parent = gr.Slider(0, 30, 8, step=1, label="👩 带娃家长")
                        with gr.Row():
                            n_child = gr.Slider(0, 30, 8, step=1, label="🧒 放学儿童")
                            n_commuter = gr.Slider(0, 30, 8, step=1, label="👨 通勤青年")
                        with gr.Row():
                            n_wanderer = gr.Slider(0, 30, 8, step=1, label="👴 社区闲散")

                        duration_slider = gr.Slider(1, 60, 10, step=1, label="⏱ 模拟时长（分钟）")
                        seed_input = gr.Number(0, label="🎲 随机种子（0=随机）", precision=0)

                        # ──── 时间段 ────
                        time_period_dd = gr.Dropdown(
                            choices=list(TIME_PERIOD_PRESETS.keys()),
                            value=None,
                            label="🕐 时间段（可选）",
                            info="选择时间段后自动调整人群配比。早晨老人多、傍晚青年多"
                        )

                        # ──── 历史记录 ────
                        gr.Markdown("---")
                        gr.Markdown("### 📋 历史记录")
                        history_dd = gr.Dropdown(
                            choices=list_run_histories(),
                            value=None,
                            label="📋 选择历史运行",
                            info="选择后自动恢复之前的所有图表"
                        )

                        # ──── LLM 设置 ────
                        gr.Markdown("---")
                        gr.Markdown("### 🧠 决策引擎设置")

                        engine_mode_r = gr.Radio(
                            choices=["rule", "llm"],
                            value="rule",
                            label="决策引擎模式",
                            info="规则引擎：无需模型 API | LLM 引擎：模型生成行为决策"
                        )

                        llm_provider_r = gr.Dropdown(
                            choices=["OpenAI (GPT-4o-mini)", "DeepSeek (V3)", "通义千问 (Qwen)", "自定义"],
                            value=saved_provider,
                            label="🤖 LLM 提供商",
                            visible=False,
                            info="推荐 DeepSeek：便宜且中文好"
                        )

                        api_key_r = gr.Textbox(
                            label="🔑 API Key",
                            value=saved_key,
                            placeholder="sk-... 或设置 STAYSIM_API_KEY 环境变量",
                            type="password",
                            visible=False,
                            info="Key 仅用于本次运行；保存设置不会把 Key 写入配置文件"
                        )

                        model_name_r = gr.Textbox(
                            label="📝 自定义模型名",
                            value=saved_model,
                            placeholder="gpt-4o-mini / deepseek-chat / qwen-turbo",
                            visible=False,
                            info="仅当选择「自定义」时需要填写"
                        )

                        # 保存按钮和状态提示
                        with gr.Row():
                            save_btn = gr.Button("💾 保存设置", variant="secondary", size="sm", visible=False)
                            save_status = gr.Textbox(label="", visible=False, show_label=False)

                        save_btn.click(
                            fn=lambda prov, key, mdl: save_llm_settings(prov, key, mdl),
                            inputs=[llm_provider_r, api_key_r, model_name_r],
                            outputs=[save_status]
                        )

                        # 切换 LLM 面板显隐
                        def toggle_llm_panel(mode):
                            is_llm = (mode == "llm")
                            return [
                                gr.update(visible=is_llm),  # llm_provider
                                gr.update(visible=is_llm),  # api_key
                                gr.update(visible=is_llm),  # model_name
                                gr.update(visible=is_llm),  # save_btn
                                gr.update(visible=False, value=""),  # save_status
                            ]

                        # 配比模式切换 → 切换滑块范围
                        def toggle_ratio_mode(mode):
                            if mode == "按比例":
                                return [
                                    gr.update(visible=True),   # ratio_total
                                    gr.update(minimum=0, maximum=100, value=20, step=1, label="🧓 买菜老人 %"),
                                    gr.update(minimum=0, maximum=100, value=20, step=1, label="👩 带娃家长 %"),
                                    gr.update(minimum=0, maximum=100, value=20, step=1, label="🧒 放学儿童 %"),
                                    gr.update(minimum=0, maximum=100, value=20, step=1, label="👨 通勤青年 %"),
                                    gr.update(minimum=0, maximum=100, value=20, step=1, label="👴 社区闲散 %"),
                                ]
                            else:
                                return [
                                    gr.update(visible=False),
                                    gr.update(minimum=0, maximum=30, value=8, step=1, label="🧓 买菜老人"),
                                    gr.update(minimum=0, maximum=30, value=8, step=1, label="👩 带娃家长"),
                                    gr.update(minimum=0, maximum=30, value=8, step=1, label="🧒 放学儿童"),
                                    gr.update(minimum=0, maximum=30, value=8, step=1, label="👨 通勤青年"),
                                    gr.update(minimum=0, maximum=30, value=8, step=1, label="👴 社区闲散"),
                                ]

                        ratio_mode_toggle.change(
                            toggle_ratio_mode,
                            inputs=[ratio_mode_toggle],
                            outputs=[ratio_total, n_elderly, n_parent, n_child, n_commuter, n_wanderer]
                        )

                        # 时间段切换 → 自动调整人群配比滑块
                        time_period_dd.change(
                            apply_time_period,
                            inputs=[time_period_dd],
                            outputs=[n_elderly, n_parent, n_child, n_commuter, n_wanderer]
                        )

                        engine_mode_r.change(
                            toggle_llm_panel,
                            inputs=[engine_mode_r],
                            outputs=[llm_provider_r, api_key_r, model_name_r, save_btn, save_status]
                        )

                        run_btn = gr.Button("▶️ 开始模拟", variant="primary", size="lg")

                        stats_display = gr.Textbox(label="📊 模拟摘要", lines=12)

                    with gr.Column(scale=2):
                        gr.Markdown("### 📈 分析结果")
                        with gr.Tabs():
                            with gr.TabItem("运动轨迹"):
                                traj_plot = gr.Plot()
                            with gr.TabItem("热力图"):
                                heat_plot = gr.Plot()
                            with gr.TabItem("停留统计"):
                                stay_plot = gr.Plot()
                            with gr.TabItem("决策日志"):
                                log_plot = gr.Plot()
                            with gr.TabItem("统计摘要"):
                                summary_plot = gr.Plot()
                            with gr.TabItem("🎬 动画"):
                                anim_history_dd = gr.Dropdown(
                                    choices=list_run_histories(),
                                    label="选择历史记录生成动画",
                                    info="选择一个运行过的模拟结果"
                                )
                                anim_btn = gr.Button("🎬 生成动画 GIF", variant="secondary")
                                anim_output = gr.File(label="下载动画")
                                anim_status = gr.Textbox(label="状态", lines=1)
                                anim_btn.click(
                                    generate_animation_gif,
                                    inputs=[anim_history_dd],
                                    outputs=[anim_output, anim_status]
                                )

                run_btn.click(
                    run_simulation_gui,
                    inputs=[scenario_dd, n_elderly, n_parent, n_child, n_commuter,
                           n_wanderer, duration_slider, seed_input,
                           engine_mode_r, llm_provider_r, api_key_r, model_name_r,
                           ratio_mode_toggle, ratio_total],
                    outputs=[traj_plot, heat_plot, stay_plot, log_plot, summary_plot, stats_display, history_dd],
                    concurrency_id="staysim_execution", concurrency_limit=1,
                )

                # 历史记录回放：选择后恢复所有图表
                history_dd.change(
                    load_history_gui,
                    inputs=[history_dd],
                    outputs=[traj_plot, heat_plot, stay_plot, log_plot, summary_plot, stats_display,
                            scenario_dd, n_elderly, n_parent, n_child, n_commuter, n_wanderer,
                            duration_slider, seed_input, engine_mode_r]
                )

            # ======== Tab 2: 方案对比 ========
            with gr.TabItem("⚖️ 方案对比"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### ⚙️ 对比设置")
                        scenario_a_dd = gr.Dropdown(
                            choices=scenarios, value=scenarios[0] if len(scenarios)>0 else None,
                            label="📐 方案A（改造前/原方案）"
                        )
                        scenario_b_dd = gr.Dropdown(
                            choices=scenarios, value=scenarios[-1] if len(scenarios)>1 else None,
                            label="📐 方案B（改造后/新方案）"
                        )

                        gr.Markdown("### 👥 人群配比（两方案相同）")
                        with gr.Row():
                            cn_elderly = gr.Slider(0, 30, 8, step=1, label="🧓 买菜老人")
                            cn_parent = gr.Slider(0, 30, 8, step=1, label="👩 带娃家长")
                        with gr.Row():
                            cn_child = gr.Slider(0, 30, 8, step=1, label="🧒 放学儿童")
                            cn_commuter = gr.Slider(0, 30, 8, step=1, label="👨 通勤青年")
                        with gr.Row():
                            cn_wanderer = gr.Slider(0, 30, 8, step=1, label="👴 社区闲散")

                        cd_duration = gr.Slider(1, 60, 10, step=1, label="⏱ 模拟时长（分钟）")
                        cd_seed = gr.Number(42, label="🎲 随机种子", precision=0)

                        # LLM 对比控制
                        gr.Markdown("---")
                        gr.Markdown("### 🧠 决策引擎")
                        compare_engine_mode = gr.Radio(
                            choices=["rule", "llm"], value="rule",
                            label="决策引擎模式"
                        )
                        compare_llm_provider = gr.Dropdown(
                            choices=["OpenAI (GPT-4o-mini)", "DeepSeek (V3)", "通义千问 (Qwen)", "自定义"],
                            value="DeepSeek (V3)", label="LLM 提供商", visible=False
                        )
                        compare_api_key = gr.Textbox(
                            label="API Key", type="password", visible=False
                        )
                        compare_model_name = gr.Textbox(
                            label="自定义模型名", visible=False
                        )

                        compare_engine_mode.change(
                            lambda mode: toggle_llm_panel(mode)[:3],
                            inputs=[compare_engine_mode],
                            outputs=[compare_llm_provider, compare_api_key, compare_model_name]
                        )

                        compare_btn = gr.Button("⚖️ 开始对比", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        compare_plot = gr.Plot()

                compare_btn.click(
                    compare_scenarios_gui,
                    inputs=[scenario_a_dd, scenario_b_dd, cn_elderly, cn_parent, cn_child,
                           cn_commuter, cn_wanderer, cd_duration, cd_seed,
                           compare_engine_mode, compare_llm_provider, compare_api_key, compare_model_name],
                    outputs=[compare_plot],
                    concurrency_id="staysim_execution", concurrency_limit=1,
                )

            # ======== Tab 3: 使用帮助 ========
            with gr.TabItem("📖 使用帮助"):
                gr.Markdown("""
                ## 如何使用 StaySim Studio

                ### 两种决策引擎
                - **规则引擎**：0 Token，秒出结果，适合快速实验
                - **LLM 引擎**：大模型生成行人决策，需通过观测数据验证有效性

                ### 使用步骤
                1. 选择场景 → 调人群滑块 → 选引擎 → 点开始模拟
                2. 右侧5个子标签查看分析图
                3. 方案对比标签页做改造前后对比

                ### Agent 使用步骤
                在 Agent 工作台描述需求 → 核对计划 → 确认执行 → 查看结果与下载文件。
                在“API 配置”中填写 DeepSeek Key；需求解析和行人 LLM 决策可以分别配置。
                会话 Key 不写入实验记录，手动模拟页的 Key 单独配置。
                导入自己的场景：进入“场景导入”，下载 DXF 模板、编辑上传、预览确认并保存。
                Agent 和手动模拟的场景列表会自动同步；已有任务仍保留原计划。
                Token 记录位于“工程记录”，费用以供应商账单为准。
                """)

        bind_scene_catalog([agent_scene_dd, scenario_dd, scenario_a_dd, scenario_b_dd])

    return app


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(LOG_DIR / "staysim.log"), level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 7860

    print("=" * 60)
    print("  StaySim Studio")
    print("  ZhuLiu XingWei ZhiNeng MoNiQi")
    print("=" * 60)
    print(f"\n  Scenario dir: {SCENARIO_DIR}")
    print(f"  Available: {list_scenarios()}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"\n  Starting web interface...")
    print(f"  Port: {port}")
    print(f"  Open the URL below in your browser\n")

    app = create_app()
    app.queue().launch(share=False, server_name="127.0.0.1", inbrowser=True,
                       server_port=port, theme=gr.themes.Soft())
