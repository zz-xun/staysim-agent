from staysim_agent import build_simulation_graph
from staysim_agent.session import AgentSession


def test_one_complete_rule_simulation_workflow(tmp_path, monkeypatch):
    from staysim import results

    monkeypatch.setattr(results, "RUNS_DIR", tmp_path / "runs")

    with AgentSession(tmp_path / "checkpoints.sqlite") as session:
        pending = session.start(request={
            "scenario_file": "community_square.yaml",
            "engine_mode": "rule",
            "duration": 20,
            "dt": 2.0,
            "seed": 42,
            "counts": {"买菜老人": 1},
        })
        assert pending["status"] == "awaiting_approval"
        state = session.resume(pending["task_id"], action="approve")
    assert state["status"] == "completed"
    assert state["result"]["engine_mode"] == "rule"
    assert state["result"]["n_agents"] == 1


def test_llm_without_key_stops_before_execution(monkeypatch):
    monkeypatch.delenv("STAYSIM_API_KEY", raising=False)
    state = build_simulation_graph().invoke({
        "approved": True,
        "request": {
            "scenario_file": "community_square.yaml",
            "engine_mode": "llm",
            "duration": 20,
            "counts": {"买菜老人": 1},
        }
    })
    assert state["status"] == "blocked"
    assert "API Key" in state["error"]
    assert state.get("result") is None


def test_without_approval_waits_before_execution():
    state = build_simulation_graph().invoke({
        "request": {
            "scenario_file": "community_square.yaml",
            "engine_mode": "rule",
            "duration": 20,
            "counts": {"买菜老人": 1},
        }
    })
    assert state["status"] == "awaiting_approval"
    assert state.get("result") is None
