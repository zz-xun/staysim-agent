"""Regression checks for experiment identity, without network requests."""
import matplotlib.pyplot as plt
import pytest

import app
from scripts import paper_experiments


def test_comparison_runs_each_scenario_once_with_requested_engine(monkeypatch):
    calls = []
    original = app.run_sim

    def simulate(env, personas, counts, **kwargs):
        calls.append((env.name, kwargs["engine_mode"]))
        return original(env, personas, counts, duration=20, seed=42, engine_mode="rule")

    monkeypatch.setattr(app, "run_sim", simulate)
    fig = app.compare_scenarios_gui(
        "old_community_before.yaml", "community_square.yaml",
        1, 0, 0, 0, 0, 1, 42, "llm", "自定义", "test-only", "test-model",
    )
    assert len(calls) == 2
    assert calls[0][0] != calls[1][0]
    assert [mode for _, mode in calls] == ["llm", "llm"]
    plt.close(fig)


def test_comparison_rejects_missing_key_before_running(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Simulation must not start without a required key")

    monkeypatch.setattr(app, "run_sim", unexpected)
    with pytest.raises(ValueError, match="API Key"):
        app.compare_scenarios_gui(
            "old_community_before.yaml", "community_square.yaml",
            1, 0, 0, 0, 0, 1, 42, "llm", "自定义", "", "test-model",
        )


def test_batch_skips_unavailable_llm_without_mislabeled_rule_runs(monkeypatch, tmp_path):
    import csv

    calls = []
    original = app.run_sim

    def simulate(env, personas, counts, **kwargs):
        calls.append(kwargs["engine_mode"])
        return original(env, personas, counts, duration=20, seed=42, engine_mode="rule")

    monkeypatch.setattr(app, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(app, "run_sim", simulate)
    monkeypatch.setattr(app, "get_saved_api_config", lambda: ("test", "", "test"))
    monkeypatch.setattr(app, "save_run_history", lambda *args: None)
    monkeypatch.setattr(paper_experiments, "SEEDS", [42])
    monkeypatch.setattr(paper_experiments, "COUNT_VALUES", [1, 0, 0, 0, 0])
    monkeypatch.setattr(paper_experiments, "_save_bar_chart", lambda *args: None)
    paper_experiments.main()
    assert calls == ["rule", "rule"]
    summary = next(tmp_path.rglob("summary.csv"))
    with summary.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert all(row["engine_requested"] == row["engine_mode"] == "rule" for row in rows)
