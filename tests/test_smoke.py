import json

import matplotlib

matplotlib.use("Agg")

import app


def _one_agent_counts(personas):
    counts = {name: 0 for name in personas}
    counts[next(iter(personas))] = 1
    return counts


def test_load_default_scenario():
    env = app.load_scenario(str(app.SCENARIO_DIR / "community_square.yaml"))

    assert env.name
    assert env.width > 0
    assert env.height > 0
    assert env.objects
    assert any(obj.element == app.SpaceElement.ENTRANCE for obj in env.objects)


def test_rule_mode_short_simulation_runs():
    env = app.load_scenario(str(app.SCENARIO_DIR / "community_square.yaml"))
    personas = app.get_default_personas()
    counts = _one_agent_counts(personas)

    result = app.run_sim(env, personas, counts, duration=20, dt=2.0, seed=42, engine_mode="rule")

    assert result.n_agents == 1
    assert result.engine_mode == "rule"
    assert result.llm_calls == 0
    assert result.trajectories
    assert result.final_positions
    assert hasattr(result, "total_stay_events")
    assert hasattr(result, "stay_events")


def test_rule_engine_does_not_call_local_or_llm():
    env = app.load_scenario(str(app.SCENARIO_DIR / "community_square.yaml"))
    persona = next(iter(app.get_default_personas().values()))
    engine = app.CognitiveEngine(mode="rule")

    decision = engine.decide((50, 40), 1.0, 0.7, persona, env, 15.0)

    assert decision.action is not None
    assert "Ollama" not in decision.reason
    assert "LLM" not in decision.reason
    assert engine.llm_call_count == 0


def test_plot_heatmap_returns_figure():
    env = app.load_scenario(str(app.SCENARIO_DIR / "community_square.yaml"))
    personas = app.get_default_personas()
    counts = _one_agent_counts(personas)
    result = app.run_sim(env, personas, counts, duration=20, dt=2.0, seed=42, engine_mode="rule")

    fig = app.plot_heatmap(env, result)

    assert fig is not None
    assert len(fig.axes) >= 1


def test_paper_summary_and_validation_report():
    env = app.load_scenario(str(app.SCENARIO_DIR / "community_square.yaml"))
    personas = app.get_default_personas()
    counts = _one_agent_counts(personas)
    result = app.run_sim(env, personas, counts, duration=20, dt=2.0, seed=42, engine_mode="rule")

    summary = app.build_summary_row(env, result, {"scenario_file": "community_square.yaml", "seed": 42, "duration": 20})
    assert "total_stay_events" in summary
    assert "spatial_entropy" in summary
    assert "cold_zone_ratio" in summary

    report = app.build_validation_report(env, result)
    assert report["title"] == "StaySim paper validation report"
    assert len(report["checks"]) == 5


def test_save_llm_settings_does_not_persist_api_key(tmp_path, monkeypatch):
    config_file = tmp_path / "config.json"
    monkeypatch.setattr(app, "CONFIG_FILE", config_file)

    app.save_llm_settings("DeepSeek (V3)", "sk-secret", "deepseek-chat")

    data = json.loads(config_file.read_text(encoding="utf-8"))
    assert data["llm_settings"]["provider"] == "DeepSeek (V3)"
    assert data["llm_settings"]["model_name"] == "deepseek-chat"
    assert "api_key" not in data["llm_settings"]
