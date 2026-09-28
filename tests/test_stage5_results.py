"""Stage 5: post-run evidence and degraded LLM execution are explicit."""

import json
from pathlib import Path

from staysim import results
from staysim_agent.session import AgentSession
from staysim_agent.tools import run_experiment


REQUEST = {
    "scenario_file": "community_square.yaml", "engine_mode": "rule",
    "duration": 20, "dt": 2.0, "seed": 42, "counts": {"买菜老人": 1},
}


def test_completed_task_contains_verified_report(tmp_path, monkeypatch):
    monkeypatch.setattr(results, "RUNS_DIR", tmp_path / "runs")
    with AgentSession(tmp_path / "agent.sqlite") as session:
        pending = session.start(request=REQUEST)
        done = session.resume(pending["task_id"], action="approve")

    assert done["status"] == "completed"
    assert done["result_validation"]["valid"] is True
    assert done["report"]["effective_engine"]["requested"] == "rule"
    report_path = Path(done["result"]["report_path"])
    assert report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "completed"
    assert report["evidence"]["actual_agents"] == 1


def test_llm_failure_is_recorded_as_degraded(tmp_path, monkeypatch):
    monkeypatch.setattr(results, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setenv("STAYSIM_API_KEY", "test-only-key")

    class FailedResponse:
        status_code = 500
        text = "provider unavailable"

        def json(self):
            return {}

    monkeypatch.setattr("staysim.cognition.requests.post", lambda *args, **kwargs: FailedResponse())
    outcome = run_experiment.invoke({"experiment_config": {
        **REQUEST, "engine_mode": "llm",
    }})

    assert outcome["requested_engine_mode"] == "llm"
    assert outcome["effective_engine_mode"] == "llm_with_fallback"
    assert outcome["fallback_count"] >= 1
    assert outcome["simulation_quality"] == "degraded"
    assert outcome["result_validation"]["valid"] is True
