"""Request extraction uses real LangChain/DeepSeek code with a mocked HTTP service."""

import json

import httpx
import pytest
from langchain_deepseek import ChatDeepSeek
from pydantic import ValidationError

from staysim_agent import build_simulation_graph
from staysim_agent.model import (
    ExperimentRequest, ModelConfigurationError, RequestDraft,
    build_request_parser, complete_request, create_request_model,
)
from staysim_agent.tools import validate_experiment_config


COMPLETE = {
    "scenario_file": "community_square.yaml",
    "engine_mode": "rule", "duration": 600, "counts": {"买菜老人": 20},
}


def mock_deepseek(content, *, finish_reason="stop", status=200):
    calls = []

    def respond(request):
        body = json.loads(request.content)
        calls.append(body)
        if status != 200:
            return httpx.Response(status, json={"error": {"message": "private-provider-error"}})
        message = {"role": "assistant", "content": content}
        reason = finish_reason
        if body.get("tools"):
            task = json.loads(body["messages"][1]["content"])["request"]
            seen = [item for item in body["messages"] if item["role"] == "tool"]
            if len(seen) < 2:
                name = "inspect_scenario" if not seen else "validate_experiment_config"
                arguments = {"scenario_file": task["scenario_file"]} if not seen else {"experiment_config": task}
                message = {"role": "assistant", "content": None, "tool_calls": [{
                    "id": f"check-{len(seen)}", "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }]}
                reason = "tool_calls"
            else:
                message = {"role": "assistant", "content": '{"action":"ready","question":null}'}
                reason = "stop"
        return httpx.Response(200, json={
            "id": "offline-test", "object": "chat.completion", "created": 0,
            "model": "deepseek-flash",
            "choices": [{
                "index": 0, "finish_reason": reason, "message": message,
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        })

    client = httpx.Client(transport=httpx.MockTransport(respond))
    model = ChatDeepSeek(
        api_key="test-only", model="deepseek-flash", http_client=client,
        base_url="https://api.deepseek.com", max_retries=0,
    )
    return model, calls


def test_real_lcel_chain_sends_catalog_json_mode_and_validates_response():
    model, calls = mock_deepseek(json.dumps(COMPLETE, ensure_ascii=False))
    draft = build_request_parser(model).invoke({
        "user_request": "社区广场，20位买菜老人，规则模式模拟10分钟",
    })
    assert isinstance(draft, RequestDraft)
    assert draft.duration == 600
    assert draft.seed is None
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert "community_square.yaml" in calls[0]["messages"][0]["content"]
    assert "old_community_before.yaml" in calls[0]["messages"][0]["content"]
    assert "买菜老人" in calls[0]["messages"][0]["content"]
    assert "不要执行模拟" in calls[0]["messages"][0]["content"]


def test_natural_language_graph_produces_plan_and_invalidates_prior_approval():
    model, calls = mock_deepseek(json.dumps(COMPLETE))
    state = build_simulation_graph(model).invoke({
        "user_input": "社区广场，20位买菜老人，规则模式模拟10分钟",
        "approved": True, "config": {"stale": True}, "result": {"stale": True},
    })
    assert state["status"] == "awaiting_approval"
    assert state["approved"] is False
    assert state["result"] is None
    assert state["config"] == {**COMPLETE, "dt": 2.0, "seed": 42}
    assert state["defaults_applied"] == {"dt": 2.0, "seed": 42}
    assert len(calls) == 4
    assert [item["tool"] for item in state["tool_trace"]] == [
        "inspect_scenario", "validate_experiment_config",
    ]


def test_missing_parameters_return_questions_without_simulation(monkeypatch):
    model, _ = mock_deepseek(json.dumps({"scenario_file": "community_square.yaml"}))
    from staysim import simulation
    monkeypatch.setattr(simulation, "run_sim", lambda *a, **k: pytest.fail("must not run"))
    state = build_simulation_graph(model).invoke({"user_input": "模拟社区广场"})
    assert state["status"] == "needs_clarification"
    assert set(state["missing_fields"]) == {"counts", "duration", "engine_mode"}
    assert len(state["questions"]) == 3
    assert state["result"] is None
    assert state["catalog"]["scenarios"]
    assert "counts" not in state["request"]


def test_unsupported_or_ambiguous_requirement_still_needs_clarification():
    draft = RequestDraft(**COMPLETE, clarification="本期仅支持一次模拟，请选择一个方案。")
    prepared = complete_request(draft)
    assert prepared["status"] == "needs_clarification"
    assert len(prepared["questions"]) == 1


@pytest.mark.parametrize("response,finish_reason", [
    ("", "stop"),
    ("not JSON", "stop"),
    (json.dumps(COMPLETE)[:-1], "stop"),
    (json.dumps(COMPLETE), "length"),
    (json.dumps({**COMPLETE, "counts": {"买菜老人": 1.5}}), "stop"),
    (json.dumps({**COMPLETE, "approved": True}), "stop"),
    (json.dumps({**COMPLETE, "api_key": "model-must-not-set-this"}), "stop"),
    (json.dumps({**COMPLETE, "api_base": "https://untrusted.invalid"}), "stop"),
])
def test_invalid_model_output_stops_before_execution(response, finish_reason):
    model, _ = mock_deepseek(response, finish_reason=finish_reason)
    state = build_simulation_graph(model).invoke({"user_input": "模拟一次"})
    assert state["status"] == "failed"
    assert state["result"] is None
    assert state["config"] is None


@pytest.mark.parametrize('status,expected', [(401, '认证失败'), (402, '余额不足'),
                         (403, '拒绝访问'), (404, '模型或接口'), (429, '请求受限'), (503, '服务暂时异常')])
def test_provider_errors_are_actionable_and_redacted(status, expected):
    model, _ = mock_deepseek('', status=status)
    state = build_simulation_graph(model).invoke({'user_input': '模拟3人5分钟'})
    assert state['status'] == 'failed'
    assert expected in state['error']
    assert 'private-provider-error' not in state['error']
    assert not state.get('result')


def test_network_error_message_does_not_expose_private_exception():
    from staysim_agent.model import model_error_message
    from openai import APIConnectionError
    exc = APIConnectionError(message='secret key detail', request=httpx.Request('POST', 'https://api.deepseek.com'))
    message = model_error_message(exc, '需求解析失败')
    assert '联网权限' in message and 'secret' not in message


def test_service_error_is_not_replaced_by_a_rule_simulation():
    model, _ = mock_deepseek("", status=503)
    state = build_simulation_graph(model).invoke({"user_input": "模拟一次"})
    assert state["status"] == "failed"
    assert state["result"] is None
    assert "private-provider-error" not in state["error"]


@pytest.mark.parametrize("change", [
    {"duration": 20.5}, {"duration": True}, {"dt": float("nan")},
    {"dt": float("inf")}, {"dt": 700}, {"seed": -1},
    {"seed": 2**32}, {"seed": True}, {"counts": {}},
    {"counts": {"买菜老人": 0}}, {"counts": {"买菜老人": -1}},
    {"counts": {"买菜老人": True}}, {"counts": {"买菜老人": 1.5}},
    {"approved": True}, {"engine_mode": "chat"},
])
def test_execution_contract_rejects_invalid_or_privileged_fields(change):
    with pytest.raises(ValidationError):
        ExperimentRequest.model_validate({**COMPLETE, **change})


def test_missing_counts_no_longer_uses_one_pedestrian_default():
    config = {key: value for key, value in COMPLETE.items() if key != "counts"}
    checked = validate_experiment_config.invoke({"experiment_config": config})
    assert not checked["valid"]
    assert any(error["field"] == "counts" for error in checked["details"])


@pytest.mark.parametrize("change", [
    {"scenario_file": "invented.yaml"}, {"counts": {"不存在的人群": 20}},
])
def test_schema_valid_but_unknown_domain_values_do_not_run(change):
    model, _ = mock_deepseek(json.dumps({**COMPLETE, **change}))
    state = build_simulation_graph(model).invoke({"user_input": "模拟一次"})
    assert state["status"] in ("failed", "blocked")
    assert state["result"] is None


def test_non_dict_request_stops_before_inspection():
    state = build_simulation_graph().invoke({"request": "incorrect shape"})
    assert state["status"] == "failed"
    assert state["scenario"] is None
    assert state["result"] is None


def test_outer_model_credential_is_required_without_reusing_other_services(monkeypatch):
    monkeypatch.delenv("STAYSIM_AGENT_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-unrelated")
    monkeypatch.setenv("STAYSIM_API_KEY", "test-simulator-only")
    with pytest.raises(ModelConfigurationError):
        create_request_model()
    state = build_simulation_graph().invoke({"user_input": "模拟社区广场"})
    assert state["status"] == "blocked"
    assert "STAYSIM_AGENT_API_KEY" in state["error"]
    assert "test-unrelated" not in state["error"]
    assert "test-simulator-only" not in state["error"]


def test_approval_and_credentials_are_absent_from_model_schema():
    properties = RequestDraft.model_json_schema()["properties"]
    assert not {"approved", "api_key", "api_base"} & properties.keys()
    assert not {"approved", "api_key"} & ExperimentRequest.model_fields.keys()
