"""Exercise the actual Gradio endpoints and persisted model usage offline."""

import json

from gradio_client import Client, handle_file

from staysim import domain, results
from staysim_agent.session import AgentSession
from staysim_agent.ui import create_agent_app
from test_request_schema import COMPLETE, mock_deepseek


def test_model_usage_survives_restart_without_credentials(tmp_path):
    model, calls = mock_deepseek(json.dumps(COMPLETE))
    db = tmp_path / "agent.sqlite"
    with AgentSession(db, model=model) as session:
        pending = session.start(user_input="社区广场20位买菜老人，规则模式10分钟")
    with AgentSession(db) as session:
        restored = session.show(pending["task_id"])
    usage = restored["observability"][0]
    assert usage["model_calls"] == len(calls) == 4
    assert usage["total_tokens"] == 120
    assert usage["cost_estimate"] is None
    assert usage["usage_missing"] == 0
    assert usage["elapsed_seconds"] > 0
    assert b"test-only" not in db.read_bytes()


def test_gradio_demo_confirm_download_reload_cancel(tmp_path, monkeypatch):
    monkeypatch.setattr(results, "RUNS_DIR", tmp_path / "runs")
    app = create_agent_app(tmp_path / "agent.sqlite")
    labels = [item.get("props", {}).get("label") for item in app.config["components"]]
    assert "🤖 Agent 工作台" in labels and "🔬 单次模拟" in labels
    try:
        _, url, _ = app.launch(server_name="127.0.0.1", prevent_thread_lock=True, quiet=True)
        client = Client(url, verbose=False, download_files=False)
        no_scene = client.predict('模拟3位买菜老人活动5分钟，使用规则模式。', None, api_name='/plan')
        assert no_scene[1] == 'blocked' and '选择一个 YAML 场景' in no_scene[6]
        pending = client.predict(api_name="/demo")
        task_id = pending[0]
        assert pending[1] == "awaiting_approval"
        assert not list((tmp_path / "runs").glob("*"))
        done = client.predict(task_id, api_name="/approve")
        assert done[1] == "completed"
        assert len(done[8]) == 10
        assert len(json.loads(done[9])) == 2
        plots = client.predict(task_id, api_name="/agent_plots")
        assert len(plots) == 5 and all(plot["type"] == "matplotlib" for plot in plots)
        assert client.predict(task_id, api_name="/approve")[4] == done[4]
        restored = Client(url, verbose=False, download_files=False).predict(task_id, api_name="/load")
        assert restored[1] == "completed"
        assert len(list((tmp_path / "runs").iterdir())) == 1
        cancel_id = client.predict(api_name="/demo")[0]
        assert client.predict(cancel_id, api_name="/cancel")[1] == "cancelled"
        assert client.predict(cancel_id, api_name="/approve")[1] == "cancelled"
        # Invalid task IDs cannot load arbitrary filesystem files.
        invalid = client.predict("../bad", api_name="/load")
        assert "任务编号" in invalid[6]
    finally:
        app.close()


def test_dxf_upload_to_agent_run_and_catalog_sync(tmp_path, monkeypatch):
    from pathlib import Path
    from staysim_agent.model import request_catalog

    scenes = tmp_path / "scenarios"
    scenes.mkdir()
    monkeypatch.setattr(domain, "SCENARIO_DIR", scenes)
    monkeypatch.setattr(results, "RUNS_DIR", tmp_path / "runs")
    app = create_agent_app(tmp_path / "agent.sqlite")
    source = Path(__file__).resolve().parents[1] / "examples" / "staysim_example_mm.dxf"
    try:
        _, url, _ = app.launch(server_name="127.0.0.1", prevent_thread_lock=True, quiet=True)
        client = Client(url, verbose=False, download_files=False)
        file = handle_file(str(source))
        preview = client.predict(file, "auto", "DXF integration", api_name="/preview_dxf")
        assert preview[1]["scene"]["width"] == 60
        assert preview[1]["scene"]["height"] == 40
        assert not list(scenes.glob("*.yaml"))
        saved = client.predict(file, "auto", "DXF integration", api_name="/save_dxf")
        filename = next(scenes.glob("*.yaml")).name
        assert all(update["value"] == filename and filename in [choice[1] for choice in update["choices"]]
                   for update in saved[2:])
        assert filename in [item["scenario_file"] for item in request_catalog()["scenarios"]]
        assert client.predict(file, "auto", "DXF integration", api_name="/save_dxf")[0] == saved[0]
        assert len(list(scenes.glob("*.yaml"))) == 1
        stale = client.predict(file, "mm", "Changed name", api_name="/save_dxf")
        assert "重新预览" in stale[0]
        # The Agent uses one natural-language route; the scenario selector now includes the imported YAML.
        assert filename in [choice[1] for choice in saved[2]["choices"]]
    finally:
        app.close()


def test_ui_credentials_are_request_local_and_not_checkpointed(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from staysim_agent import graph
    from staysim_agent.model import runtime_setting, runtime_settings
    from staysim_agent.ui import _with_session, configure_api

    credentials, outer_field, inner_field, notice = configure_api("private-test-key", "parser", "", "walker", True)
    assert not outer_field and not inner_field and "private-test-key" not in notice
    model, calls = mock_deepseek(json.dumps({**COMPLETE, "engine_mode": "llm"}))
    def request_model():
        assert runtime_setting("STAYSIM_AGENT_API_KEY") == "private-test-key"
        return model
    monkeypatch.setattr(graph, "create_request_model", request_model)
    monkeypatch.delenv("STAYSIM_API_KEY", raising=False)
    db = tmp_path / "private.sqlite"
    state = _with_session(str(db), lambda session: session.start(user_input="广场 LLM 模式20人600秒"), credentials)
    assert state["status"] == "awaiting_approval"  # Inner-key validation inside graph also saw this scope.
    assert len(calls) == 4
    assert b"private-test-key" not in db.read_bytes()
    assert "private-test-key" not in json.dumps(state)
    assert runtime_setting("STAYSIM_API_KEY") == ""
    def in_scope(key):
        with runtime_settings({"STAYSIM_API_KEY": key}):
            return runtime_setting("STAYSIM_API_KEY")
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(in_scope, ["first", "second"])) == ["first", "second"]
