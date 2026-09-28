"""Small Gradio front end for the durable one-simulation Agent workflow."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any
from uuid import UUID

from .session import AgentSession, DEFAULT_DB
from .tools import ARTIFACT_NAMES
from .model import runtime_settings
from staysim import domain, results


DEMO_REQUEST = {
    "scenario_file": "community_square.yaml",
    "engine_mode": "rule",
    "duration": 20,
    "counts": {"买菜老人": 1},
    "dt": 2.0,
    "seed": 42,
}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _plan_preview(state: dict[str, Any]) -> str:
    config = state.get("config") or state.get("request") or {}
    if not config:
        return "<p>计划尚未生成。先描述场景、人群、时长和模拟模式。</p>"
    modes = {"rule": "规则模式", "llm": "LLM 模式", "local": "本地模型"}
    fields = [
        ("场景", config.get("scenario_file", "待补充")),
        ("人群", "、".join(f"{name} × {count} 人" for name, count in (config.get("counts") or {}).items()) or "待补充"),
        ("模拟时长", f"{config['duration']} 秒" if config.get("duration") is not None else "待补充"),
        ("行为模式", modes.get(config.get("engine_mode"), config.get("engine_mode") or "待补充")),
        ("时间步长", f"{config['dt']} 秒" if config.get("dt") is not None else "待补充"),
        ("随机种子", config.get("seed", "待补充")),
    ]
    rows = "".join(f'<tr><th style="text-align:left;padding:8px 16px 8px 0;white-space:nowrap">{label}</th>'
                   f'<td style="padding:8px 0">{escape(str(value))}</td></tr>' for label, value in fields)
    defaults = state.get("defaults_applied") or {}
    note = "；".join(f"{key}={value}" for key, value in defaults.items())
    return f'<table style="width:100%;font-size:14px">{rows}</table>' + (
        f'<p style="opacity:.7;font-size:12px">自动补全：{escape(note)}</p>' if note else "")


def _status_preview(state: dict[str, Any]) -> str:
    labels = {"idle": "等待填写需求", "awaiting_approval": "计划已就绪 · 等待你确认",
              "awaiting_input": "需要补充信息", "completed": "模拟已完成 · 可查看分析图和下载结果",
              "cancelled": "任务已取消", "failed": "执行失败 · 请查看操作提示",
              "blocked": "暂时无法继续 · 请查看操作提示", "running": "正在执行模拟"}
    status = state.get("status", "idle")
    return f"**当前进度** · {labels.get(status, escape(str(status)))}"


def _render(state: dict[str, Any] | None):
    import gradio as gr

    state = state or {"status": "idle", "error": ""}
    pending = state.get("pending") or []
    wait_text = ""
    if pending:
        wait = pending[0]
        wait_text = wait.get("question", "") or "\n".join(wait.get("questions", []))
    plan = {
        "config": state.get("config") or state.get("request", {}),
        "defaults_applied": state.get("defaults_applied", {}),
    }
    result = {
        "result": state.get("result"),
        "validation": state.get("result_validation"),
        "report": state.get("report"),
        "explanation": state.get("explanation"),
        "execution": state.get("execution"),
    }
    trace = state.get("tool_trace", [])
    report = state.get("report") or {}
    explanation = state.get("explanation") or {}
    summary = explanation.get("summary", "生成计划后，请核对场景、人群、时长和行为模式，再确认执行。")
    if explanation.get("findings"):
        summary += "\n\n" + "\n\n".join(explanation["findings"])
    warnings = list(dict.fromkeys(report.get("warnings", []) + explanation.get("warnings", [])))
    if warnings:
        summary += "\n\n注意：" + "；".join(warnings)
    downloads = []
    run_id = state.get("run_id", "")
    if (state.get("result_validation") or {}).get("valid") and run_id:
        # Never trust paths emitted by a model or arbitrary checkpoint fields.
        if str(UUID(run_id)) == run_id:
            folder = (results.RUNS_DIR / run_id).resolve()
            if folder.parent == results.RUNS_DIR.resolve():
                downloads = [str(folder / name) for name in (*ARTIFACT_NAMES, "execution_receipt.json")
                             if (folder / name).is_file() and not (folder / name).is_symlink()]
    waiting_for_approval = bool(pending and pending[0].get("kind") == "approval")
    return (
        state.get("task_id", ""), state.get("status", ""), _json(plan), wait_text,
        _json(result), _json(trace), state.get("ui_notice") or state.get("error", ""), state,
        summary, downloads, _json(state.get("observability", [])),
        gr.update(interactive=waiting_for_approval), gr.update(interactive=bool(pending)),
        gr.update(interactive=bool(pending)),
        _plan_preview(state), _status_preview(state),
        gr.update(interactive=bool((state.get("result_validation") or {}).get("valid"))),
    )


def _with_session(db_path: str, operation, credentials=None):
    path = Path(db_path or DEFAULT_DB)
    with runtime_settings(credentials), AgentSession(path) as session:
        return operation(session)


def configure_api(key, model, simulator_key, simulator_model, use_same):
    """Return session-only settings; never return a key in visible output."""
    key, simulator_key = (key or "").strip(), (simulator_key or "").strip()
    settings = {}
    if key:
        settings.update(STAYSIM_AGENT_API_KEY=key, STAYSIM_AGENT_MODEL=(model or "").strip() or "deepseek-chat")
    if use_same:
        simulator_key = key
    if simulator_key:
        settings.update(STAYSIM_API_KEY=simulator_key, STAYSIM_API_BASE="https://api.deepseek.com",
                        STAYSIM_MODEL_NAME=(simulator_model or "").strip() or "deepseek-chat")
    message = "已应用到当前会话（尚未测试连接）。解析 API：{}；行人决策 API：{}。".format(
        "已输入" if key else "使用本机配置", "已输入" if simulator_key else "使用本机配置")
    return settings, "", "", message


def render_scene_import_tab():
    """One deterministic local import route, sharing the simulator's scene catalog."""
    import gradio as gr
    from staysim.scene_import import parse_dxf, preview_scene, save_scene

    draft = gr.State(None)
    gr.Markdown("### DXF 场景导入\n上传规范化二维 DXF → 检查预览 → 确认保存 → 在 Agent 中生成模拟计划。")
    with gr.Row():
        with gr.Column(scale=1):
            source = gr.File(label="上传二维 DXF", file_types=[".dxf"], type="filepath")
            name = gr.Textbox(label="场景名称", placeholder="例如：我的社区广场")
            units = gr.Dropdown(choices=[("读取 DXF 单位", "auto"), ("米", "m"), ("毫米", "mm")], value="auto", label="图纸单位")
            inspect = gr.Button("解析并预览", variant="primary")
            publish = gr.Button("确认场景并保存 YAML", interactive=False)
            notice = gr.Textbox(label="导入状态", interactive=False, lines=3)
            yaml_file = gr.File(label="下载场景 YAML", interactive=False)
            template = Path(__file__).resolve().parents[1] / "examples" / "staysim_template_m.dxf"
            if template.exists():
                gr.File(value=str(template), label="下载规范 DXF 模板（米）", interactive=False)
            with gr.Accordion("绘图规范与默认参数", open=False):
                gr.Markdown("""- 场地：`STAYSIM_BOUNDARY` 图层，一个闭合、无圆弧、轴对齐的矩形 `LWPOLYLINE`。
- 设施：用 `CIRCLE` 标记位置和作用半径，图层使用 `STAYSIM_ENTRANCE`、`STAYSIM_BENCH`、`STAYSIM_TREE` 等规范名称。
- 至少一个入口。全部图形在 XY 平面，单位仅支持米或毫米。标注只放在 `STAYSIM_NOTE`。
- 不支持建筑障碍、块、复杂多边形、图片和 DWG；圆代表设施作用范围，不是障碍物。
- 吸引力、容量和遮荫是程序默认假设，预览时列出；不是从 CAD 中推断的实测值。
- 保存后场景列表自动更新。已有模拟任务保持原计划；请生成新计划使用导入场景。""")
        with gr.Column(scale=2):
            plot = gr.Plot(label="场景预览（米）")
            details = gr.JSON(label="尺寸、设施与导入假设")

    events = dict(concurrency_id="staysim_execution", concurrency_limit=1)

    def preview(path, unit, scene_name):
        try:
            if not path:
                raise ValueError("请先上传 DXF 文件。")
            value = parse_dxf(path, units=unit, name=scene_name or "")
            return value, preview_scene(value), value, "解析成功。请核对尺寸、设施和默认参数，再确认保存。", gr.update(interactive=True), None
        except (ValueError, OSError) as exc:
            return None, None, None, str(exc), gr.update(interactive=False), None

    inspect.click(preview, [source, units, name], [draft, plot, details, notice, publish, yaml_file], api_name="preview_dxf", **events)
    for control in (source, units, name):
        control.change(lambda: (None, None, None, "输入已变更，请重新解析预览。", gr.update(interactive=False), None),
                       None, [draft, plot, details, notice, publish, yaml_file], api_name=False, **events)

    def bind_catalog(selectors):
        def commit(value, path, unit, scene_name):
            try:
                if not value or not path:
                    raise ValueError("请先解析并检查预览。")
                if parse_dxf(path, units=unit, name=scene_name or "") != value:
                    raise ValueError("图纸或设置已改变，请重新预览后确认。")
                filename = save_scene(value)
                choices = domain.list_scenarios()
                return (f"已保存 {filename}。场景列表已同步，请生成新计划后确认模拟。",
                        str(domain.SCENARIO_DIR / filename),
                        *[gr.update(choices=choices, value=filename) for _ in selectors])
            except (ValueError, OSError) as exc:
                return (str(exc), None, *[gr.update() for _ in selectors])
        publish.click(commit, [draft, source, units, name], [notice, yaml_file, *selectors], api_name="save_dxf", **events)
    return bind_catalog


def render_agent_tab(db_path: str | Path = DEFAULT_DB):
    """Render within app.py's Studio tabs, using the same controls and plots."""
    import gradio as gr

    db_text = str(db_path)

    def plan_request(text, credentials, selected_scene):
        if not isinstance(text, str) or not text.strip():
            return _render({"status": "blocked", "error": "请先输入一次模拟需求。"})
        if not isinstance(selected_scene, str) or not selected_scene.strip():
            return _render({"status": "blocked", "error": "请先在“本次场景”中选择一个 YAML 场景，再生成计划。"})
        try:
            state = _with_session(db_text, lambda session: session.start(user_input=text, selected_scenario=selected_scene), credentials)
        except Exception as exc:
            state = {"status": "blocked", "error": f"创建任务失败（{type(exc).__name__}）。"}
        return _render(state)

    def create_demo():
        try:
            state = _with_session(db_text, lambda session: session.start(request=DEMO_REQUEST))
        except Exception as exc:
            state = {"status": "blocked", "error": f"创建演示任务失败（{type(exc).__name__}）。"}
        return _render(state)

    def resume_task(task_id, action, text, old_state, credentials=None):
        if not task_id:
            return _render({"status": "blocked", "error": "当前没有任务。"})
        try:
            state = _with_session(
                db_text,
                lambda session: session.resume(task_id, action=action, text=text or None),
                credentials,
            )
        except (ValueError, RuntimeError) as exc:
            state = {**(old_state or {}), "ui_notice": str(exc)}
        return _render(state)

    def refresh(task_id, old_state):
        if not task_id:
            return _render(old_state)
        try:
            state = _with_session(db_text, lambda session: session.show(task_id))
        except (ValueError, RuntimeError) as exc:
            state = {**(old_state or {}), "ui_notice": str(exc)}
        return _render(state)

    def recover(task_id, old_state, credentials):
        try:
            state = _with_session(db_text, lambda session: session.recover(task_id), credentials)
        except (ValueError, RuntimeError) as exc:
            state = {**(old_state or {}), "ui_notice": str(exc)}
        return _render(state)

    def show_plots(task_id):
        state = _with_session(db_text, lambda session: session.show(task_id))
        if not (state.get("result_validation") or {}).get("valid"):
            raise gr.Error("请先完成并验证一次模拟。")
        run_id = str(UUID(state["run_id"]))
        result, env, _ = results.load_history_result(run_id)
        if result is None or env is None:
            raise gr.Error("无法读取本次结果。")
        return (results.plot_trajectories(env, result), results.plot_heatmap(env, result),
                results.plot_stay_barchart(result), results.plot_decision_log(result),
                results.plot_summary(env, result))

    with gr.Column():
        memory = gr.State({})
        credentials = gr.State({}, time_to_live=3600)
        gr.Markdown("### 一次需求，一次完整模拟\n填写需求 → 核对计划 → 确认执行 → 查看分析与结果")
        progress = gr.Markdown(_status_preview({}))
        with gr.Row():
            with gr.Column(scale=1, min_width=340):
                with gr.Accordion("🔑 API 配置 · DeepSeek", open=False):
                    gr.Markdown("仅用于 Agent 工作台，保存在当前会话内存中；刷新页面后请重新输入。留空使用本机配置。")
                    outer_key = gr.Textbox(label="需求解析 API Key", type="password")
                    outer_model = gr.Textbox(label="需求解析模型", value="deepseek-chat")
                    same_key = gr.Checkbox(label="行人 LLM 决策也使用此 Key", value=False)
                    inner_key = gr.Textbox(label="行人决策 API Key（规则模式无需填写）", type="password")
                    inner_model = gr.Textbox(label="行人决策模型", value="deepseek-chat")
                    with gr.Row():
                        apply_api = gr.Button("应用 API 配置")
                        clear_api = gr.Button("清除会话配置")
                    api_notice = gr.Markdown("")
                gr.Markdown("### 01 · 描述模拟需求")
                available_scenes = domain.list_scenarios()
                scene_choice = gr.Dropdown(
                    choices=available_scenes,
                    label="本次场景（已默认选择；可切换）",
                    value=available_scenes[0] if available_scenes else None,
                    interactive=True,
                )
                request = gr.Textbox(
                    label="自然语言需求",
                    placeholder="例如：模拟3位买菜老人活动5分钟，使用规则模式。",
                    lines=3,
                )
                gr.Markdown("场景以上方选择为准，无需重复填写。请说明：**人群、人数、时长和模式**。\n\n可用人群：买菜老人、带娃家长、放学儿童、通勤青年、社区闲散。")
                with gr.Row():
                    plan = gr.Button("生成计划", variant="primary")
                    demo = gr.Button("离线规则演示")
                gr.Markdown("### 02 · 核对并确认")
                preview = gr.HTML(_plan_preview({}))
                status = gr.Textbox(label="状态", interactive=False, visible=False)
                with gr.Accordion("参数原始数据", open=False):
                    plan_json = gr.Code(label="计划与默认值", language="json", interactive=False)
                waiting = gr.Textbox(label="当前需要处理", interactive=False, lines=2)
                reply = gr.Textbox(label="修改或补充需求（确认前）", placeholder="例如：人数改成10人，其他条件不变。", lines=2)
                approve = gr.Button("确认并执行", variant="primary", size="lg", interactive=False)
                with gr.Row():
                    revise = gr.Button("提交修改", interactive=False)
                    cancel = gr.Button("取消任务", interactive=False)
                    refresh_button = gr.Button("刷新状态")
                error = gr.Textbox(label="操作提示", interactive=False)
                with gr.Accordion("📋 保存编号 / 恢复历史任务", open=False):
                    task_id = gr.Textbox(label="当前任务编号（可复制保存）", interactive=False)
                    restore_id = gr.Textbox(label="粘贴以前保存的 Task ID")
                    load_button = gr.Button("载入任务")
                    recover_button = gr.Button("恢复中断的任务")
            with gr.Column(scale=2):
                gr.Markdown("### 03 · 查看模拟结果")
                summary = gr.Markdown("尚无模拟结果。左侧生成计划并确认执行后，这里将展示本次分析。")
                plot_button = gr.Button("加载本次分析图", variant="secondary", interactive=False)
                gr.Markdown("完成后点击加载图表；读取已保存结果，不会重新运行模拟。")
                with gr.Tabs():
                    with gr.TabItem("运动轨迹"):
                        traj = gr.Plot()
                    with gr.TabItem("热力图"):
                        heat = gr.Plot()
                    with gr.TabItem("停留统计"):
                        stay = gr.Plot()
                    with gr.TabItem("决策日志"):
                        log = gr.Plot()
                    with gr.TabItem("统计摘要"):
                        stats = gr.Plot()
                    with gr.TabItem("结果报告"):
                        result_json = gr.Code(label="结果与验证报告", language="json", interactive=False)
                downloads = gr.File(label="下载本次结果", file_count="multiple", interactive=False)
                with gr.Accordion("工程记录 · 工具调用与模型用量", open=False):
                    trace_json = gr.Code(label="工具调用记录", language="json", interactive=False)
                    observations = gr.Code(label="操作耗时与外层模型 Token（费用未知时为 null）", language="json", interactive=False)

        outputs = [task_id, status, plan_json, waiting, result_json, trace_json, error, memory,
                   summary, downloads, observations, approve, revise, cancel, preview, progress, plot_button]
        events = dict(concurrency_id="staysim_execution", concurrency_limit=1)
        apply_api.click(configure_api, [outer_key, outer_model, inner_key, inner_model, same_key],
                        [credentials, outer_key, inner_key, api_notice], api_name=False, **events)
        clear_api.click(lambda: ({}, "", "", "会话配置已清除；后续使用本机配置。"), None,
                        [credentials, outer_key, inner_key, api_notice], api_name=False, **events)
        plan.click(plan_request, [request, credentials, scene_choice], outputs, api_name="plan", **events)
        demo.click(create_demo, None, outputs, api_name="demo", **events)
        approve.click(lambda tid, old, creds: resume_task(tid, "approve", None, old, creds), [task_id, memory, credentials], outputs, api_name="approve", **events)
        revise.click(lambda tid, text, old, creds: resume_task(tid, "reply", text, old, creds), [task_id, reply, memory, credentials], outputs, api_name="reply", **events)
        cancel.click(lambda tid, old, creds: resume_task(tid, "cancel", None, old, creds), [task_id, memory, credentials], outputs, api_name="cancel", **events)
        refresh_button.click(refresh, [task_id, memory], outputs, api_name="refresh", **events)
        load_button.click(refresh, [restore_id, memory], outputs, api_name="load", **events)
        recover_button.click(recover, [task_id, memory, credentials], outputs, api_name="recover", **events)
        plot_button.click(show_plots, task_id, [traj, heat, stay, log, stats], api_name="agent_plots", **events)
        # New task/loading/cancelling clears plots so an earlier run is not mislabelled.
        for button in (plan, demo, load_button, revise, cancel):
            button.click(lambda: (None,) * 5, None, [traj, heat, stay, log, stats], api_name=False, **events)
    return scene_choice


def create_agent_app(db_path: str | Path = DEFAULT_DB):
    """Use the original Studio shell; the domain/CLI remains independent of UI."""
    from app import create_app

    app = create_app(agent_db=db_path, selected_tab="agent")
    app.queue(default_concurrency_limit=1)
    return app


def launch_agent_app(db_path: str | Path = DEFAULT_DB, **launch_kwargs):
    """Launch the Agent UI explicitly from the CLI or a notebook."""
    import gradio as gr

    return create_agent_app(db_path).launch(server_name="127.0.0.1", share=False,
                                           theme=gr.themes.Soft(), **launch_kwargs)
