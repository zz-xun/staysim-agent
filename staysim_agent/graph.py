"""LangGraph workflow for one complete StaySim simulation."""

from __future__ import annotations

import json
from typing import Any, Literal
from uuid import uuid4

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import ValidationError
from typing_extensions import TypedDict

from .model import (
    ExperimentRequest, ModelConfigurationError, PreparationDecision, RequestDraft,
    SimulationExplanation,
    build_request_parser, complete_request, create_request_model,
    preparation_messages, request_catalog, model_error_message,
)
from .tools import (
    inspect_scenario,
    validate_experiment_config,
    plan_fingerprint,
    validate_execution_result,
)


class SimulationState(TypedDict, total=False):
    """State passed between the simulation workflow nodes."""

    request: dict[str, Any]
    user_input: str
    selected_scenario: str
    approved: bool
    scenario: dict[str, Any] | None
    config: dict[str, Any] | None
    validation: dict[str, Any] | None
    result: dict[str, Any] | None
    result_validation: dict[str, Any] | None
    report: dict[str, Any] | None
    explanation: dict[str, Any] | None
    missing_fields: list[str]
    questions: list[str]
    defaults_applied: dict[str, Any]
    catalog: dict[str, Any]
    status: str
    error: str
    messages: list[BaseMessage]
    tool_trace: list[dict[str, Any]]
    preparation_rounds: int
    plan_id: str
    run_id: str
    approved_plan_id: str


MAX_PREPARATION_ROUNDS = 6
MAX_TOOLS_PER_ROUND = 2
PREPARATION_TOOLS = {
    tool.name: tool for tool in (inspect_scenario, validate_experiment_config)
}


def parse_request(state: SimulationState, model: BaseChatModel | None = None) -> dict[str, Any]:
    # A new input invalidates outputs and approvals from any previous request.
    cleared = {
        "error": "", "scenario": None, "config": None, "validation": None,
        "result": None, "missing_fields": [], "questions": [],
        "result_validation": None, "report": None, "explanation": None,
        "defaults_applied": {}, "catalog": {},
        "messages": [], "tool_trace": [], "preparation_rounds": 0,
        "approved": False, "plan_id": "", "approved_plan_id": "", "run_id": "",
    }
    natural_language = "user_input" in state
    if natural_language:
        cleared.update({"approved": False, "request": {}})
    try:
        if natural_language:
            text = state["user_input"]
            if not isinstance(text, str) or not text.strip():
                return {**cleared, "status": "needs_clarification", "questions": ["请描述希望运行的一次模拟。"]}
            parser = build_request_parser(model if model is not None else create_request_model())
            selected = state.get("selected_scenario")
            if selected:
                text += "\n用户在场景选择器中指定的本次场景文件：" + json.dumps(selected, ensure_ascii=False)
            draft = parser.invoke({"user_request": text})
            if not isinstance(draft, RequestDraft):
                raise ValueError("模型未返回请求对象")
            if selected:
                if draft.scenario_file and draft.scenario_file != selected:
                    draft = draft.model_copy(update={"clarification": "文字中的场景与选择器不一致，请核对后重新生成计划。"})
                draft = draft.model_copy(update={"scenario_file": selected})
        else:
            raw = state.get("request")
            if not isinstance(raw, dict):
                return {**cleared, "status": "failed", "error": "request 必须是字典，或通过 user_input 提供自然语言"}
            draft = RequestDraft.model_validate(raw)
        prepared = complete_request(draft)
        if prepared["status"] == "needs_clarification":
            prepared["catalog"] = request_catalog()
        return {**cleared, **prepared}
    except ModelConfigurationError as exc:
        return {**cleared, "status": "blocked", "error": str(exc)}
    except ValidationError as exc:
        problems = [
            f"{'.'.join(map(str, issue['loc'])) or 'request'}: {issue['msg']}"
            for issue in exc.errors(include_input=False, include_url=False)
        ]
        return {**cleared, "status": "failed", "error": "参数校验失败：" + "；".join(problems)}
    except OutputParserException:
        return {**cleared, "status": "failed", "error": "模型输出为空、格式错误或不符合字段约束，请重新描述。模拟未执行。"}
    except Exception as exc:
        # Provider error text can contain request contents. Do not return it.
        return {**cleared, "status": "failed", "error": model_error_message(exc, "需求解析失败")}


def prepare_node(state: SimulationState, model: BaseChatModel | None = None) -> dict[str, Any]:
    """Ask the model for a tool call or a final preparation decision."""
    rounds = state.get("preparation_rounds", 0)
    if rounds >= MAX_PREPARATION_ROUNDS:
        return {"status": "blocked", "error": "准备阶段达到 6 轮上限，请检查工具反馈后重新提交。模拟未执行。"}
    messages = list(state.get("messages") or preparation_messages(
        state["request"], state.get("defaults_applied", {}),
    ))
    try:
        llm = model if model is not None else create_request_model()
        response = llm.bind_tools(list(PREPARATION_TOOLS.values())).invoke(messages)
        if not isinstance(response, AIMessage):
            raise ValueError("需要 AIMessage")
        if response.invalid_tool_calls:
            raise ValueError("工具参数不是完整 JSON")
        if response.response_metadata.get("finish_reason") not in (None, "stop", "tool_calls"):
            raise ValueError("输出未正常结束")
        messages.append(response)
        update = {"messages": messages, "preparation_rounds": rounds + 1}
        if response.tool_calls:
            ids = [call.get("id") for call in response.tool_calls]
            if not all(ids) or len(set(ids)) != len(ids):
                raise ValueError("工具调用标识缺失或重复")
            if len(ids) > MAX_TOOLS_PER_ROUND:
                return {**update, "status": "blocked", "error": "一轮最多调用两个检查工具。模拟未执行。"}
            return {**update, "status": "calling_tools"}

        decision = PreparationDecision.model_validate_json(response.content)
        if decision.action == "clarify":
            return {**update, "status": "needs_clarification", "questions": [decision.question]}

        # A model's claim never substitutes for actual, successful observations.
        if (state.get("scenario") or {}).get("scenario_file") == state["request"]["scenario_file"]:
            validation = state.get("validation") or {}
            if validation.get("valid") is True and validation.get("config") == state["request"]:
                return {**update, "status": "prepared"}
        validation = state.get("validation") or {}
        if validation.get("valid") is False:
            return {**update, "status": "blocked", "error": validation.get("error", "配置检查未通过")}
        messages.append(HumanMessage(content="程序检查：缺少成功的场景查看或完整配置校验记录。请调用缺少的检查工具；不能直接宣布通过。"))
        return {**update, "status": "preparing"}
    except ModelConfigurationError as exc:
        return {"status": "blocked", "error": str(exc)}
    except Exception as exc:
        return {"status": "failed", "error": model_error_message(exc, "工具准备失败")}


def preparation_tools_node(state: SimulationState) -> dict[str, Any]:
    """Execute an allowlisted tool call, return its observation to the model."""
    messages = list(state["messages"])
    trace = list(state.get("tool_trace", []))
    update: dict[str, Any] = {"status": "preparing"}
    for call in messages[-1].tool_calls:
        name, args = call["name"], call["args"]
        try:
            if name not in PREPARATION_TOOLS:
                raise ValueError("准备阶段只能查看场景或校验配置，不能执行其他工具")
            if name == "inspect_scenario":
                if args != {"scenario_file": state["request"]["scenario_file"]}:
                    raise ValueError("场景参数必须与当前 request 完全一致")
            else:
                if set(args) != {"experiment_config"}:
                    raise ValueError("工具参数只能是 experiment_config")
                supplied = args["experiment_config"]
                if not isinstance(supplied, dict) or set(supplied) != set(state["request"]):
                    raise ValueError("experiment_config 必须包含完整 request，不能增删参数")
                checked = ExperimentRequest.model_validate(supplied)
                if checked.model_dump() != state["request"]:
                    raise ValueError("不能更改用户参数；请使用当前 request")
            # Passing the full call makes LangChain produce ToolMessage with the same id.
            observation = PREPARATION_TOOLS[name].invoke(call)
            output = json.loads(observation.content)
            if name == "inspect_scenario":
                update["scenario"] = output
            else:
                update["validation"] = output
        except (ValueError, TypeError, KeyError, OSError) as exc:
            # Never reflect credential-bearing provider errors or raw invalid inputs.
            message = str(exc) if isinstance(exc, ValueError) and not isinstance(exc, ValidationError) else "工具参数或场景数据无效，请按 schema 检查后重试"
            output = {"valid": False, "error": message}
            observation = ToolMessage(
                content=json.dumps(output, ensure_ascii=False),
                tool_call_id=call["id"], name=name, status="error",
            )
        messages.append(observation)
        trace.append({
            "round": state["preparation_rounds"], "tool_call_id": call["id"],
            "tool": name, "arguments": args, "observation": output,
            "ok": output.get("valid", True) is not False,
        })
    return {**update, "messages": messages, "tool_trace": trace}


def inspect_node(state: SimulationState) -> dict[str, Any]:
    try:
        scenario = inspect_scenario.invoke(
            str(state["request"].get("scenario_file", ""))
        )
    except (KeyError, TypeError, ValueError) as exc:
        return {"status": "failed", "error": str(exc)}
    return {"scenario": scenario, "status": "inspected"}


def validate_node(state: SimulationState) -> dict[str, Any]:
    validation = validate_experiment_config.invoke({"experiment_config": state["request"]})
    if not validation.get("valid"):
        return {
            "validation": validation,
            "status": "blocked",
            "error": validation.get("error", "实验配置不可用"),
        }
    try:
        fingerprint = plan_fingerprint(validation["config"])
    except (OSError, ValueError):
        return {"status": "blocked", "error": "场景已不可用，请重新检查。"}
    return {
        "validation": validation, "config": validation["config"],
        "plan_id": fingerprint, "run_id": str(uuid4()),
        "approved": False, "approved_plan_id": "", "status": "awaiting_approval",
        "questions": ["请核对本次配置和默认值，确认后执行；也可修改需求或取消。"],
    }


def validation_error_node(state: SimulationState) -> dict[str, Any]:
    return {"status": "blocked", "error": state.get("error", "实验配置不可用")}


def approval_node(state: SimulationState) -> dict[str, Any]:
    reply = interrupt({
        "kind": "approval", "plan_id": state["plan_id"],
        "request": state["config"], "defaults_applied": state.get("defaults_applied", {}),
        "question": "确认运行这一版配置？可以 approve、reply 修改需求或 cancel。",
    })
    if isinstance(reply, dict) and reply.get("action") == "reply":
        return _apply_reply(state, reply)
    if isinstance(reply, dict) and reply.get("action") == "cancel":
        return {"status": "cancelled", "approved": False, "questions": []}
    if not isinstance(reply, dict) or reply != {"action": "approve", "plan_id": state["plan_id"]}:
        return {"status": "blocked", "error": "确认无效或不属于当前计划。", "approved": False}
    return {
        "status": "approved", "approved": True,
        "approved_plan_id": state["plan_id"], "questions": [],
    }


def _apply_reply(state: SimulationState, reply: dict) -> dict[str, Any]:
    text = reply.get("text")
    if not isinstance(text, str) or not text.strip():
        return {"status": "blocked", "error": "补充内容不能为空"}
    previous = state.get("user_input") or json.dumps(state.get("request", {}), ensure_ascii=False)
    return {
        "user_input": previous + "\n用户最新补充（冲突时以此为准）：" + text.strip(),
        "approved": False, "approved_plan_id": "", "config": None, "result": None,
        "result_validation": None, "report": None, "explanation": None,
        "status": "revising", "error": "", "questions": [],
    }


def await_input_node(state: SimulationState) -> dict[str, Any]:
    reply = interrupt({
        "kind": "clarification", "questions": state.get("questions", []),
        "request": state.get("request", {}), "catalog": state.get("catalog", {}),
    })
    if isinstance(reply, dict) and reply.get("action") == "cancel":
        return {"status": "cancelled", "approved": False, "questions": []}
    if isinstance(reply, dict) and reply.get("action") == "reply":
        return _apply_reply(state, reply)
    return {"status": "blocked", "error": "当前任务需要补充需求，不能直接确认执行。"}


def run_node(state: SimulationState, executor=None) -> dict[str, Any]:
    try:
        if not state.get("approved") or state.get("approved_plan_id") != state.get("plan_id"):
            return {"status": "blocked", "error": "缺少当前计划的用户确认。"}
        if executor is None:
            return {"status": "blocked", "error": "执行模拟需使用带持久化执行记录的 AgentSession。"}
        result = executor(state["config"], state["run_id"], state["plan_id"])
    except Exception as exc:
        return {"status": "failed", "error": f"模拟未交付成功（{type(exc).__name__}）。请查看 execution 运行记录；不会自动重复执行。"}
    return {"result": result, "status": "executed"}


def validate_result_node(state: SimulationState) -> dict[str, Any]:
    """Verify the receipt, artifacts, metrics and engine audit after execution."""
    try:
        checked = validate_execution_result(state["config"], state["result"], state["plan_id"])
    except (OSError, KeyError, ValueError, TypeError) as exc:
        return {"status": "failed", "error": f"结果验证失败（{type(exc).__name__}），请查看运行目录。"}
    if not checked.get("valid"):
        validation = checked.get("validation") or {}
        errors = validation.get("errors") or ["结果未通过完整性检查"]
        return {
            "result_validation": validation, "report": checked.get("report"),
            "status": "failed", "error": "；".join(errors),
        }
    return {
        "result_validation": checked["validation"], "report": checked.get("report"),
        "status": "validated",
    }


def _fallback_explanation(report: dict[str, Any]) -> dict[str, Any]:
    metrics = report.get("metrics", {})
    engine = report.get("effective_engine", {})
    warnings = list(report.get("warnings", []))
    if engine.get("fallback_count"):
        warnings.append("部分行人决策使用兜底逻辑，不能将本次结果描述为纯 LLM 结果。")
    return {
        "summary": f"已完成 {metrics.get('n_agents', 0)} 位行人的一次 StaySim 模拟，结果文件已通过完整性检查。",
        "findings": [
            f"停留事件 {metrics.get('total_stay_events', 0)} 次，空间熵 {metrics.get('spatial_entropy', 0)}。",
            f"请求模式为 {engine.get('requested', 'unknown')}，实际有效模式为 {engine.get('effective', 'unknown')}。",
        ],
        "warnings": warnings,
    }


def explain_result_node(state: SimulationState, model: BaseChatModel | None = None) -> dict[str, Any]:
    """Explain verified facts; the explanation model has no tools or execution access."""
    report = state.get("report") or {}
    explanation = _fallback_explanation(report)
    if model is not None:
        try:
            prompt = [
                SystemMessage(content="你是 StaySim 结果解读助手。只能根据给定 JSON 报告解释结果，不能修改指标、虚构事实、调用工具或建议重新运行。只输出符合 schema 的 JSON。"),
                HumanMessage(content=json.dumps(report, ensure_ascii=False)),
            ]
            response = model.with_structured_output(SimulationExplanation, method="json_mode").invoke(prompt)
            if isinstance(response, SimulationExplanation):
                explanation = response.model_dump()
        except Exception:
            # A report remains useful when the optional explanation model is unavailable.
            pass
    return {"explanation": explanation, "status": "completed"}


def route_after_inspect(state: SimulationState) -> Literal["validate", "failed"]:
    return "validate" if state.get("status") != "failed" else "failed"


def route_after_validate(state: SimulationState) -> Literal["approve", "blocked"]:
    return "approve" if state.get("status") != "blocked" else "blocked"


def route_after_approval(state: SimulationState) -> str:
    return {"approved": "run", "revising": "revise"}.get(state.get("status"), "waiting")


def build_simulation_graph(model: BaseChatModel | None = None, *, checkpointer=None, executor=None):
    """Build and compile the one-simulation workflow."""
    graph = StateGraph(SimulationState)
    graph.add_node("parse_request", lambda state: parse_request(state, model=model))
    graph.add_node("prepare", lambda state: prepare_node(state, model=model))
    graph.add_node("preparation_tools", preparation_tools_node)
    graph.add_node("inspect_scenario", inspect_node)
    graph.add_node("validate_config", validate_node)
    graph.add_node("approval", approval_node)
    graph.add_node("await_input", await_input_node)
    graph.add_node("run_experiment", lambda state: run_node(state, executor=executor))
    graph.add_node("validate_result", validate_result_node)
    graph.add_node("explain_result", lambda state: explain_result_node(state, model=model))
    graph.add_node("blocked", validation_error_node)
    graph.add_edge(START, "parse_request")
    graph.add_conditional_edges(
        "parse_request",
        lambda state: ("prepare" if "user_input" in state else "inspect") if state.get("status") == "planned" else "clarify" if state.get("status") == "needs_clarification" else "stop",
        {"prepare": "prepare", "inspect": "inspect_scenario", "clarify": "await_input", "stop": END},
    )
    graph.add_conditional_edges(
        "prepare",
        lambda state: {
            "calling_tools": "tools", "preparing": "retry", "prepared": "verify",
            "needs_clarification": "clarify",
        }.get(state.get("status"), "stop"),
        {"tools": "preparation_tools", "retry": "prepare", "verify": "validate_config", "clarify": "await_input", "stop": END},
    )
    graph.add_edge("preparation_tools", "prepare")
    graph.add_conditional_edges("inspect_scenario", route_after_inspect, {"validate": "validate_config", "failed": END})
    graph.add_conditional_edges("validate_config", route_after_validate, {"approve": "approval", "blocked": "blocked"})
    graph.add_conditional_edges("approval", route_after_approval, {"run": "run_experiment", "revise": "parse_request", "waiting": END})
    graph.add_conditional_edges("await_input", lambda state: "revise" if state.get("status") == "revising" else "stop", {"revise": "parse_request", "stop": END})
    graph.add_edge("blocked", END)
    graph.add_conditional_edges(
        "run_experiment",
        lambda state: "validate" if state.get("status") == "executed" else "stop",
        {"validate": "validate_result", "stop": END},
    )
    graph.add_conditional_edges(
        "validate_result",
        lambda state: "explain" if state.get("status") == "validated" else "stop",
        {"explain": "explain_result", "stop": END},
    )
    graph.add_edge("explain_result", END)
    return graph.compile(checkpointer=checkpointer)
