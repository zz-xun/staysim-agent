"""Behavioral parity and the UI-free domain boundary after stage 1 extraction."""

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
import subprocess
import sys

import pytest

from staysim import domain, results, simulation
from staysim_agent.tools import (
    inspect_scenario, run_experiment, validate_experiment_config,
)


# Captured from the pre-extraction app.py on 2026-09-27, Python 3.12.
# Each case runs all five personas, two pedestrians each, for 600s at dt=2.
# Hashes cover all result fields except nondeterministic wall-clock elapsed time:
# trajectories, decisions, stay events, final positions and every metric.
@pytest.mark.parametrize("scenario,seed,expected", [
    ("community_square.yaml", 42, "b9765d28276492707836bc85651e0ffe765afc844d7cde262e768890bb686ab7"),
    ("community_square.yaml", 7, "a0076248200477b4ff37b5bbed4f24e3aa467ac34dc04eda1f976dbaa31ffccd"),
    ("old_community_before.yaml", 42, "1cccedd9e728404532a672bf380a781dd3237ba22b66a0f8acfc8948d831eb56"),
    ("old_community_before.yaml", 7, "e9b73deb32984448b412a7fcf74776155d8f29ad9b0f1123783a166c3bb51c6f"),
])
def test_seeded_rule_results_match_before_extraction(scenario, seed, expected):
    env = domain.load_scenario(str(domain.SCENARIO_DIR / scenario))
    personas = domain.get_default_personas()
    result = simulation.run_sim(
        env, personas, {name: 2 for name in personas},
        duration=600, dt=2.0, seed=seed, engine_mode="rule",
    )
    data = asdict(result)
    data.pop("elapsed")
    payload = json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")
    assert hashlib.sha256(payload).hexdigest() == expected


def test_agent_runs_without_importing_ui_or_plotting(tmp_path):
    """A fresh interpreter avoids false passes from pytest import order."""
    root = Path(__file__).resolve().parents[1]
    code = """
import sys
from pathlib import Path
from staysim import results
from staysim_agent.session import AgentSession
results.RUNS_DIR = Path.cwd() / 'runs'
with AgentSession(Path.cwd() / 'checkpoints.sqlite') as session:
    pending = session.start(request={
    'scenario_file': 'community_square.yaml', 'engine_mode': 'rule',
    'duration': 20, 'dt': 2, 'seed': 42,
    'counts': {'买菜老人': 1},
    })
    state = session.resume(pending['task_id'], action='approve')
assert state['status'] == 'completed', state
assert Path(state['result']['run_dir'], 'full_result.json').is_file()
assert not {'app', 'gradio', 'matplotlib'} & set(sys.modules)
print('headless graph completed')
"""
    process = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(root)},
        capture_output=True, text=True, timeout=60,
    )
    assert process.returncode == 0, process.stderr
    assert "headless graph completed" in process.stdout


def test_scene_tool_uses_domain_catalog():
    info = inspect_scenario.invoke({"scenario_file": "community_square.yaml"})
    assert info["scenario_file"] == "community_square.yaml"
    assert info["element_count"] > 0


def test_scene_tool_rejects_path_escape():
    with pytest.raises(ValueError, match="YAML"):
        inspect_scenario.invoke({"scenario_file": "../app.py"})


def test_result_roundtrip_and_distinct_run_directories(tmp_path, monkeypatch):
    monkeypatch.setattr(results, "RUNS_DIR", tmp_path / "runs")
    env = domain.load_scenario(str(domain.SCENARIO_DIR / "community_square.yaml"))
    personas = domain.get_default_personas()
    counts = {name: 1 for name in personas}
    original = simulation.run_sim(env, personas, counts, duration=60, seed=42)
    config = {"scenario_file": "community_square.yaml", "seed": 42, "duration": 60}
    first = Path(results.save_run_history(env, original, config, personas, counts))
    second = Path(results.save_run_history(env, original, config, personas, counts))
    assert first != second
    restored, restored_env, restored_config = results.load_history_result(first.name)
    # JSON persistence intentionally converts tuples into lists.
    assert json.loads(json.dumps(asdict(restored))) == json.loads(json.dumps(asdict(original)))
    assert restored_env.name == env.name
    assert restored_config == config
    assert {path.name for path in first.iterdir()} == {
        "config.json", "trajectories.csv", "decisions.jsonl", "stay_events.csv",
        "summary.csv", "validation_report.json", "stats.json", "full_result.json",
    }


def test_runtime_key_is_not_in_tool_output_or_artifacts(tmp_path, monkeypatch):
    secret = "test-only-runtime-secret"
    monkeypatch.setenv("STAYSIM_API_KEY", secret)
    monkeypatch.setattr(results, "RUNS_DIR", tmp_path / "runs")
    observed = {}

    def fake_simulation(env, personas, counts, **kwargs):
        observed.update(kwargs)
        return domain.SimResult(env_name=env.name, n_agents=1, engine_mode="llm")

    monkeypatch.setattr(simulation, "run_sim", fake_simulation)
    config = {
        "scenario_file": "community_square.yaml", "engine_mode": "llm",
        "duration": 20, "counts": {"买菜老人": 1},
    }
    checked = validate_experiment_config.invoke({"experiment_config": config})
    assert checked["valid"]
    assert "api_key" not in checked["config"]
    outcome = run_experiment.invoke({"experiment_config": checked["config"]})
    assert observed["api_key"] == secret
    assert secret not in json.dumps([checked, outcome])
    for path in Path(outcome["run_dir"]).iterdir():
        assert secret not in path.read_text(encoding="utf-8-sig")


def test_history_drops_credentials_even_for_direct_callers(tmp_path, monkeypatch):
    monkeypatch.setattr(results, "RUNS_DIR", tmp_path / "runs")
    env = domain.Environment(100, 80, name="test")
    result = domain.SimResult(env_name=env.name)
    folder = Path(results.save_run_history(
        env, result,
        {"seed": 42, "api_key": "test-only-secret", "headers": {"Authorization": "secret"}},
        {}, {},
    ))
    assert json.loads((folder / "config.json").read_text(encoding="utf-8")) == {"seed": 42}
