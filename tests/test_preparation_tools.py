"""Real DeepSeek tool-call wire protocol with deterministic HTTP responses."""

import json

import httpx
import pytest
from langchain_core.messages import ToolMessage
from langchain_deepseek import ChatDeepSeek

from staysim import simulation
from staysim_agent import build_simulation_graph
from staysim_agent.graph import MAX_PREPARATION_ROUNDS


REQUEST = {
    "scenario_file": "community_square.yaml", "engine_mode": "rule",
    "duration": 600, "counts": {"买菜老人": 20}, "dt": 2.0, "seed": 42,
}
READY = {"role": "assistant", "content": '{"action":"ready","question":null}'}


def tool_call(name, args, call_id="call-1"):
    return {
        "id": call_id, "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def batch(*calls):
    return {"role": "assistant", "content": None, "tool_calls": list(calls)}


def run_script(messages, monkeypatch, request=None):
    """Only provider responses are fake; graph and domain tools execute normally."""
    monkeypatch.setattr(simulation, "run_sim", lambda *a, **k: pytest.fail("preparation must not simulate"))
    payloads = []
    responses = [{"role": "assistant", "content": json.dumps(request or REQUEST)}, *messages]

    def respond(http_request):
        body = json.loads(http_request.content)
        index = len(payloads)
        payloads.append(body)
        message = responses[min(index, len(responses) - 1)]
        return httpx.Response(200, json={
            "id": f"mock-{index}", "object": "chat.completion", "created": 0,
            "model": "deepseek-flash",
            "choices": [{"index": 0, "message": message,
                         "finish_reason": "tool_calls" if message.get("tool_calls") else "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        })

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = ChatDeepSeek(api_key="test-only", model="deepseek-flash", http_client=client, max_retries=0)
        state = build_simulation_graph(model).invoke({"user_input": "社区广场20位买菜老人，规则模式10分钟"})
    return state, payloads


def inspect(call_id="inspect"):
    return tool_call("inspect_scenario", {"scenario_file": REQUEST["scenario_file"]}, call_id)


def validate(call_id="validate", request=None):
    return tool_call("validate_experiment_config", {"experiment_config": request or REQUEST}, call_id)


def test_model_can_choose_reverse_order_and_receives_matching_observations(monkeypatch):
    state, payloads = run_script([batch(validate()), batch(inspect()), READY], monkeypatch)
    assert state["status"] == "awaiting_approval"
    assert state["result"] is None
    assert state["preparation_rounds"] == 3
    assert [row["tool"] for row in state["tool_trace"]] == ["validate_experiment_config", "inspect_scenario"]
    for payload in payloads[1:]:
        assert {tool["function"]["name"] for tool in payload["tools"]} == {
            "inspect_scenario", "validate_experiment_config",
        }
        assert "response_format" not in payload
    observation = next(msg for msg in payloads[2]["messages"] if msg["role"] == "tool")
    assert observation["tool_call_id"] == "validate"
    assert json.loads(observation["content"])["valid"] is True
    assert [msg.tool_call_id for msg in state["messages"] if isinstance(msg, ToolMessage)] == ["validate", "inspect"]


def test_two_tool_calls_in_one_response_are_both_answered(monkeypatch):
    state, payloads = run_script([batch(inspect(), validate()), READY], monkeypatch)
    assert state["status"] == "awaiting_approval"
    replies = [msg for msg in payloads[-1]["messages"] if msg["role"] == "tool"]
    assert [msg["tool_call_id"] for msg in replies] == ["inspect", "validate"]


def test_incorrect_argument_name_can_be_corrected_after_observation(monkeypatch):
    wrong = tool_call("validate_experiment_config", {"config": REQUEST}, "wrong")
    state, payloads = run_script([batch(wrong), batch(inspect(), validate()), READY], monkeypatch)
    assert state["status"] == "awaiting_approval"
    assert state["tool_trace"][0]["ok"] is False
    assert "experiment_config" in state["tool_trace"][0]["observation"]["error"]
    assert any(msg["role"] == "tool" and msg["tool_call_id"] == "wrong" for msg in payloads[2]["messages"])


def test_claimed_success_cannot_skip_tools_and_loop_is_bounded(monkeypatch):
    state, payloads = run_script([READY], monkeypatch)
    assert state["status"] == "blocked"
    assert state["preparation_rounds"] == MAX_PREPARATION_ROUNDS
    assert state["tool_trace"] == []
    assert state["config"] is None
    assert len(payloads) == 1 + MAX_PREPARATION_ROUNDS


def test_model_recovers_after_premature_ready(monkeypatch):
    state, _ = run_script([READY, batch(inspect(), validate()), READY], monkeypatch)
    assert state["status"] == "awaiting_approval"


@pytest.mark.parametrize("call", [
    tool_call("run_experiment", {"experiment_config": REQUEST}),
    tool_call("inspect_scenario", {"scenario_file": "old_community_before.yaml"}),
    validate(request={**REQUEST, "counts": {"买菜老人": 1}}),
    validate(request={**REQUEST, "approved": True}),
])
def test_unauthorized_tools_or_changed_parameters_are_rejected(call, monkeypatch):
    clarify = {"role": "assistant", "content": '{"action":"clarify","question":"请检查原始参数。"}'}
    state, _ = run_script([batch(call), clarify], monkeypatch)
    assert state["status"] == "needs_clarification"
    assert not state["tool_trace"][0]["ok"]
    assert state["config"] is None
    assert state["request"] == REQUEST


def test_missing_simulator_key_is_seen_by_model_without_mode_downgrade(monkeypatch):
    monkeypatch.delenv("STAYSIM_API_KEY", raising=False)
    request = {**REQUEST, "engine_mode": "llm"}
    clarify = {"role": "assistant", "content": '{"action":"clarify","question":"请在本机配置 STAYSIM_API_KEY 后重新提交。"}'}
    state, payloads = run_script([batch(inspect(), validate(request=request)), clarify], monkeypatch, request)
    assert state["status"] == "needs_clarification"
    assert state["request"]["engine_mode"] == "llm"
    assert state["validation"]["valid"] is False
    assert "STAYSIM_API_KEY" in state["questions"][0]
    observed = [json.loads(msg["content"]) for msg in payloads[-1]["messages"] if msg["role"] == "tool"]
    assert any("API Key" in item.get("error", "") for item in observed)


def test_failed_validation_cannot_be_overridden_by_ready(monkeypatch):
    monkeypatch.delenv("STAYSIM_API_KEY", raising=False)
    request = {**REQUEST, "engine_mode": "llm"}
    state, _ = run_script([batch(inspect(), validate(request=request)), READY], monkeypatch, request)
    assert state["status"] == "blocked"
    assert state["config"] is None


@pytest.mark.parametrize("message", [
    {"role": "assistant", "content": "检查完成，我已经运行了模拟。"},
    {"role": "assistant", "content": '{"action":"ready","approved":true}'},
    batch(inspect("duplicate"), validate("duplicate")),
    batch(inspect("1"), inspect("2"), validate("3")),
    batch({"id": "bad", "type": "function", "function": {"name": "inspect_scenario", "arguments": "broken JSON"}}),
])
def test_bad_completion_or_tool_protocol_stops(message, monkeypatch):
    state, _ = run_script([message], monkeypatch)
    assert state["status"] in ("failed", "blocked")
    assert state["config"] is None
    assert state["result"] is None
