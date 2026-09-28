"""Offline workflow evaluation for the portfolio demo.

This measures deterministic state transitions and execution latency; it does not
claim to measure DeepSeek extraction quality because it makes no network calls.
"""

from __future__ import annotations

import json
import argparse
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from staysim import results
from staysim_agent.session import AgentSession, UsageObserver
from staysim_agent.model import build_request_parser, complete_request, create_request_model


VALID_REQUEST = {
    "scenario_file": "community_square.yaml", "engine_mode": "rule",
    "duration": 20, "dt": 2.0, "seed": 42, "counts": {"买菜老人": 1},
}

# These are expected answers for evaluation, never returned to the parser model.
LANGUAGE_CASES = [
    ("中文分钟", "在社区广场模拟20位买菜老人活动10分钟，使用规则模式。",
     {"scenario_file": "community_square.yaml", "counts": {"买菜老人": 20}, "duration": 600, "engine_mode": "rule"}, []),
    ("秒", "community_square.yaml，买菜老人2人，规则模式，持续90秒。",
     {"duration": 90, "counts": {"买菜老人": 2}, "engine_mode": "rule"}, []),
    ("半分钟", "社区广场，1位买菜老人，规则模式，模拟半分钟。", {"duration": 30}, []),
    ("技术参数", "community_square.yaml，买菜老人3人，规则模式20秒，dt为1秒，seed为7。",
     {"dt": 1.0, "seed": 7, "duration": 20}, []),
    ("旧场景", "old_community_before.yaml，买菜老人2人，规则模拟1分钟。",
     {"scenario_file": "old_community_before.yaml", "duration": 60}, []),
    ("LLM模式", "community_square.yaml，买菜老人1人，LLM行为，模拟20秒。", {"engine_mode": "llm"}, []),
    ("本地模式", "community_square.yaml，买菜老人1人，使用本地模型，模拟20秒。", {"engine_mode": "local"}, []),
    ("缺模式", "community_square.yaml，买菜老人2人，模拟20秒。", {}, ["engine_mode"]),
    ("缺时长", "community_square.yaml，买菜老人2人，规则模式。", {}, ["duration"]),
    ("缺人群", "community_square.yaml，20人，规则模式模拟20秒。", {}, ["counts"]),
    ("缺场景", "模拟2位买菜老人，规则模式20秒。", {}, ["scenario_file"]),
    ("缺全部", "帮我模拟一下。", {}, ["scenario_file", "counts", "duration", "engine_mode"]),
]


def evaluate_language(model=None):
    """Explicit opt-in: real parser quality, no simulation or preparation tools."""
    parser = build_request_parser(model if model is not None else create_request_model())
    cases = []
    for name, text, expected, missing in LANGUAGE_CASES:
        observer = UsageObserver()
        started = time.perf_counter()
        try:
            draft = parser.invoke({"user_request": text}, config={"callbacks": [observer]})
            actual = complete_request(draft)
            matched = all(actual["request"].get(key) == value for key, value in expected.items())
            matched = matched and set(actual.get("missing_fields", [])) == set(missing)
            matched = matched and actual["status"] == ("needs_clarification" if missing else "planned")
            row = {"passed": matched, "actual": actual}
        except Exception as exc:
            row = {"passed": False, "error_type": type(exc).__name__}
        cases.append({"name": name, **row, "elapsed_seconds": round(time.perf_counter() - started, 4),
                      "usage": observer.summary()})
    return {"mode": "live_parser", "case_count": len(cases),
            "passed": all(case["passed"] for case in cases),
            "accuracy": sum(case["passed"] for case in cases) / len(cases), "cases": cases}


def evaluate() -> dict:
    cases = []
    with tempfile.TemporaryDirectory(prefix="staysim-agent-eval-") as directory:
        old_runs = results.RUNS_DIR
        results.RUNS_DIR = Path(directory) / "runs"
        try:
            started = time.perf_counter()
            with AgentSession(Path(directory) / "agent.sqlite") as session:
                pending = session.start(request=VALID_REQUEST)
                done = session.resume(pending["task_id"], action="approve")
                repeated = session.resume(pending["task_id"], action="approve")
            cases.append({
                "name": "valid_rule_workflow",
                "passed": done["status"] == "completed" and done["result_validation"]["valid"],
                "status": done["status"],
                "elapsed_seconds": round(time.perf_counter() - started, 4),
            })
            cases.append({"name": "duplicate_approval_no_rerun", "passed": repeated["result"] == done["result"]
                          and len(list(results.RUNS_DIR.iterdir())) == 1})
            with AgentSession(Path(directory) / "cancel.sqlite") as session:
                task = session.start(request=VALID_REQUEST)
                cancelled = session.resume(task["task_id"], action="cancel")
                repeated_cancel = session.resume(task["task_id"], action="approve")
            cases.append({"name": "cancel_is_terminal", "passed": cancelled["status"] == "cancelled"
                          and repeated_cancel["status"] == "cancelled" and repeated_cancel["result"] is None})

            started = time.perf_counter()
            with AgentSession(Path(directory) / "missing.sqlite") as session:
                missing = session.start(request={"scenario_file": VALID_REQUEST["scenario_file"]})
            cases.append({
                "name": "missing_fields_interrupt",
                "passed": missing["status"] == "needs_clarification" and bool(missing["pending"]),
                "status": missing["status"],
                "elapsed_seconds": round(time.perf_counter() - started, 4),
            })

            started = time.perf_counter()
            with AgentSession(Path(directory) / "invalid.sqlite") as session:
                invalid = session.start(request={
                    **VALID_REQUEST, "counts": {"不存在的人群": 1},
                })
            cases.append({
                "name": "unknown_persona_blocked",
                "passed": invalid["status"] == "blocked",
                "status": invalid["status"],
                "elapsed_seconds": round(time.perf_counter() - started, 4),
            })
        finally:
            results.RUNS_DIR = old_runs
    return {
        "mode": "offline_workflow",
        "network_calls": 0,
        "case_count": len(cases),
        "passed": all(case["passed"] for case in cases),
        "cases": cases,
    }


if __name__ == "__main__":
    options = argparse.ArgumentParser(description="StaySim Agent 工作流及需求解析评估")
    options.add_argument("--live", action="store_true", help="显式调用 DeepSeek 评估12条自然语言；产生 API 费用，不运行模拟")
    options.add_argument("--output", type=Path, help="保存本次评估 JSON")
    args = options.parse_args()
    result = evaluate_language() if args.live else evaluate()
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(payload)
    raise SystemExit(0 if result["passed"] else 1)
