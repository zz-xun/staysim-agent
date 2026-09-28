"""Durable interruptions and crash recovery, with real rule simulations."""

import json
from pathlib import Path
import subprocess
import sys

import httpx
import pytest
from langchain_deepseek import ChatDeepSeek
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from staysim import domain, results, simulation
from staysim_agent import build_simulation_graph
from staysim_agent import tools as tool_module
from staysim_agent.session import AgentSession


REQUEST = {
    "scenario_file": "community_square.yaml", "engine_mode": "rule",
    "duration": 20, "dt": 2.0, "seed": 42, "counts": {"买菜老人": 1},
}


@pytest.fixture
def storage(tmp_path, monkeypatch):
    monkeypatch.setattr(results, "RUNS_DIR", tmp_path / "runs")
    return tmp_path / "agent.sqlite"


def count_runs(monkeypatch):
    calls = []
    original = simulation.run_sim

    def counted(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(simulation, "run_sim", counted)
    return calls


def test_reopen_confirm_and_duplicate_confirm_only_runs_once(storage, monkeypatch):
    calls = count_runs(monkeypatch)
    with AgentSession(storage) as session:
        pending = session.start(request=REQUEST)
        task_id = pending["task_id"]
        assert pending["status"] == "awaiting_approval"
        assert pending["pending"][0]["kind"] == "approval"
        assert pending["execution"]["status"] == "not_started"
        assert calls == []
    with AgentSession(storage) as session:
        assert session.show(task_id)["plan_id"] == pending["plan_id"]
        completed = session.resume(task_id, action="approve")
        assert completed["status"] == "completed"
        assert completed["execution"]["status"] == "completed"
        assert completed["pending"] == []
        assert completed["next_nodes"] == []
    with AgentSession(storage) as session:
        repeated = session.resume(task_id, action="approve")
        assert repeated["result"] == completed["result"]
        assert session.recover(task_id)["result"] == completed["result"]
    assert len(calls) == 1
    assert len(list(results.RUNS_DIR.iterdir())) == 1


def test_input_approved_true_cannot_skip_real_interrupt():
    graph = build_simulation_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "test"}}
    state = graph.invoke({"request": REQUEST, "approved": True}, config)
    assert state["status"] == "awaiting_approval"
    assert not state["approved"]
    assert state["__interrupt__"]
    blocked = graph.invoke(Command(resume={"action": "approve", "plan_id": "stale"}), config)
    assert blocked["status"] == "blocked"
    assert blocked["result"] is None


def test_cancel_survives_restart_and_cannot_be_approved(storage, monkeypatch):
    calls = count_runs(monkeypatch)
    with AgentSession(storage) as session:
        task_id = session.start(request=REQUEST)["task_id"]
    with AgentSession(storage) as session:
        state = session.resume(task_id, action="cancel")
        assert state["status"] == "cancelled"
    with AgentSession(storage) as session:
        assert session.resume(task_id, action="approve")["status"] == "cancelled"
    assert not calls


def test_scene_changed_after_confirmation_plan_blocks_run(storage, monkeypatch, tmp_path):
    scenes = tmp_path / "scenarios"
    scenes.mkdir()
    scene = scenes / REQUEST["scenario_file"]
    scene.write_bytes((domain.SCENARIO_DIR / scene.name).read_bytes())
    monkeypatch.setattr(domain, "SCENARIO_DIR", scenes)
    calls = count_runs(monkeypatch)
    with AgentSession(storage) as session:
        pending = session.start(request=REQUEST)
    scene.write_text(scene.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
    with AgentSession(storage) as session:
        state = session.resume(pending["task_id"], action="approve")
        assert state["status"] == "failed"
        assert state["execution"]["status"] == "blocked"
        assert "改变" in state["execution"]["error"]
    assert calls == []


def test_crash_after_artifacts_recovers_receipt_without_rerun(storage, monkeypatch):
    calls = count_runs(monkeypatch)
    original = tool_module._run_experiment

    def crash_after_save(*args, **kwargs):
        original(*args, **kwargs)
        raise KeyboardInterrupt("simulate process termination after receipt")

    monkeypatch.setattr(tool_module, "_run_experiment", crash_after_save)
    with AgentSession(storage) as session:
        task_id = session.start(request=REQUEST)["task_id"]
        with pytest.raises(KeyboardInterrupt):
            session.resume(task_id, action="approve")
    monkeypatch.setattr(tool_module, "_run_experiment", original)
    with AgentSession(storage) as session:
        assert session.show(task_id)["execution"]["status"] == "running"
        state = session.recover(task_id)
        assert state["status"] == "completed"
        assert state["execution"]["status"] == "completed"
    assert len(calls) == 1


def test_crash_before_receipt_is_uncertain_and_never_auto_retried(storage, monkeypatch):
    calls = []

    def crash(*args, **kwargs):
        calls.append(1)
        raise KeyboardInterrupt("simulate process termination during simulation")

    monkeypatch.setattr(tool_module, "_run_experiment", crash)
    with AgentSession(storage) as session:
        task_id = session.start(request=REQUEST)["task_id"]
        with pytest.raises(KeyboardInterrupt):
            session.resume(task_id, action="approve")
    with AgentSession(storage) as session:
        state = session.recover(task_id)
        assert state["status"] == "failed"
        assert state["execution"]["status"] == "uncertain"
        assert session.resume(task_id, action="approve")["status"] == "failed"
    assert len(calls) == 1


def test_corrupted_receipt_artifact_is_not_accepted(storage, monkeypatch):
    original = tool_module._run_experiment
    calls = count_runs(monkeypatch)

    def corrupt_and_crash(*args, **kwargs):
        outcome = original(*args, **kwargs)
        Path(outcome["run_dir"], "full_result.json").write_text("{}", encoding="utf-8")
        raise KeyboardInterrupt()

    monkeypatch.setattr(tool_module, "_run_experiment", corrupt_and_crash)
    with AgentSession(storage) as session:
        task_id = session.start(request=REQUEST)["task_id"]
        with pytest.raises(KeyboardInterrupt):
            session.resume(task_id, action="approve")
    with AgentSession(storage) as session:
        state = session.recover(task_id)
        assert state["status"] == "failed"
        assert state["result"] is None
        assert state["execution"]["status"] == "uncertain"
    assert len(calls) == 1


def test_missing_fields_are_persistent_interrupts(storage):
    with AgentSession(storage) as session:
        state = session.start(request={"scenario_file": REQUEST["scenario_file"]})
        assert state["pending"][0]["kind"] == "clarification"
        task_id = state["task_id"]
    with AgentSession(storage) as session:
        assert session.show(task_id)["status"] == "needs_clarification"
        with pytest.raises(ValueError, match="补充"):
            session.resume(task_id, action="approve")
        assert session.resume(task_id, action="cancel")["status"] == "cancelled"


def parser_model(payloads):
    """Use actual structured/tool calling, supplying deterministic service output."""
    def respond(request):
        body = json.loads(request.content)
        payloads.append(body)
        if body.get("response_format"):
            # The resumed task explicitly asks for two pedestrians.
            message = {"role": "assistant", "content": json.dumps({**REQUEST, "counts": {"买菜老人": 2}})}
        elif not any(msg["role"] == "tool" for msg in body["messages"]):
            config = json.loads(body["messages"][1]["content"])["request"]
            calls = [
                ("inspect_scenario", {"scenario_file": config["scenario_file"]}),
                ("validate_experiment_config", {"experiment_config": config}),
            ]
            message = {"role": "assistant", "content": None, "tool_calls": [
                {"id": f"call-{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
                for i, (name, args) in enumerate(calls)
            ]}
        else:
            message = {"role": "assistant", "content": '{"action":"ready","question":null}'}
        return httpx.Response(200, json={
            "id": "test", "object": "chat.completion", "created": 0, "model": "deepseek-flash",
            "choices": [{"index": 0, "message": message,
                         "finish_reason": "tool_calls" if message.get("tool_calls") else "stop"}],
        })
    client = httpx.Client(transport=httpx.MockTransport(respond))
    return ChatDeepSeek(api_key="test-only-key", model="deepseek-flash", http_client=client, max_retries=0), client


@pytest.mark.parametrize("initial", [REQUEST, {"scenario_file": REQUEST["scenario_file"]}])
def test_reply_rechecks_and_requires_new_confirmation(storage, monkeypatch, initial):
    calls = count_runs(monkeypatch)
    with AgentSession(storage) as session:
        original = session.start(request=initial)
    payloads = []
    model, client = parser_model(payloads)
    try:
        with AgentSession(storage, model=model) as session:
            updated = session.resume(original["task_id"], action="reply", text="改为2位买菜老人，规则模式20秒，步长2秒，种子42")
            assert updated["status"] == "awaiting_approval"
            assert updated["pending"][0]["request"]["counts"] == {"买菜老人": 2}
            assert updated["plan_id"] != original.get("plan_id")
            assert updated["approved"] is False
            assert not calls
            assert len(updated["tool_trace"]) == 2
            assert "改为2位" in payloads[0]["messages"][-1]["content"]
        # Confirmation after restart needs no outer model calls.
        with AgentSession(storage) as session:
            done = session.resume(original["task_id"], action="approve")
            assert done["status"] == "completed"
            assert done["result"]["n_agents"] == 2
        assert b"test-only-key" not in storage.read_bytes()
    finally:
        client.close()


def test_database_lock_prevents_competing_commands(storage):
    with AgentSession(storage):
        with pytest.raises(RuntimeError, match="另一个"):
            with AgentSession(storage):
                pytest.fail("second process/session must not acquire lock")


def test_cli_separate_processes_create_show_approve_and_repeat(storage, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    # Override the domain artifact location only inside the CLI subprocess.
    launcher = "from pathlib import Path; from staysim import results; import run_agent; results.RUNS_DIR=Path(__import__('sys').argv[2]).parent/'runs'; raise SystemExit(run_agent.main())"

    def cli(*args):
        process = subprocess.run(
            [sys.executable, "-X", "utf8", "-c", launcher, "--db", str(storage), *args],
            cwd=root, capture_output=True, encoding="utf-8", timeout=60,
        )
        assert process.returncode == 0, process.stderr + process.stdout
        return json.loads(process.stdout)

    pending = cli("--demo")
    task_id = pending["task_id"]
    assert pending["status"] == "awaiting_approval"
    assert cli("--show", task_id)["pending"] == pending["pending"]
    done = cli("--approve", task_id)
    assert done["status"] == "completed"
    assert cli("--approve", task_id)["result"] == done["result"]
    assert len(list((storage.parent / "runs").iterdir())) == 1
