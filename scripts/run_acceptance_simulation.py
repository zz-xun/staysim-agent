"""Run the imported CAD scene through the durable Agent workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from staysim import domain, results
from staysim_agent.session import AgentSession


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="dxf_c7f6e53b78e09c07.yaml")
    parser.add_argument("--duration", type=int, default=600)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--db", default=str(ROOT / "output" / "agent" / "acceptance_agent.sqlite"))
    args = parser.parse_args()

    scene_path = domain.SCENARIO_DIR / args.scenario
    if not scene_path.is_file():
        raise SystemExit(f"找不到导入场景: {scene_path}")
    env = domain.load_scenario(str(scene_path))
    request = {
        "scenario_file": args.scenario,
        "engine_mode": "rule",
        "duration": args.duration,
        "counts": {"买菜老人": args.count},
        "dt": 2.0,
        "seed": 42,
    }

    with AgentSession(args.db) as session:
        pending = session.start(request=request)
        if pending.get("status") != "awaiting_approval" or pending.get("result") is not None:
            raise SystemExit(f"确认前状态异常: {pending.get('status')}")
        completed = session.resume(pending["task_id"], action="approve")
        repeated = session.resume(pending["task_id"], action="approve")

    with AgentSession(args.db) as session:
        restored = session.show(pending["task_id"])
    result, restored_env, _ = results.load_history_result(completed["run_id"])
    if completed.get("status") != "completed" or result is None:
        raise SystemExit(f"模拟未完成: {completed.get('status')}")
    evidence = {
        "status": "passed",
        "scenario_file": args.scenario,
        "cad_scene": {"name": env.name, "width_m": env.width, "height_m": env.height,
                       "facility_count": len(env.objects),
                       "source_import": getattr(env, "source_import", None)},
        "request": request,
        "task_id": pending["task_id"],
        "run_id": completed["run_id"],
        "before_approval": {"status": pending["status"], "result": pending.get("result")},
        "after_approval": {"status": completed["status"], "result": completed.get("result")},
        "restored_status": restored["status"],
        "repeated_approval_same_run": repeated["run_id"] == completed["run_id"],
        "simulator_result": {"n_agents": result.n_agents, "env_name": result.env_name,
                              "engine_mode": result.engine_mode,
                              "total_stay_events": result.total_stay_events,
                              "spatial_entropy": result.spatial_entropy},
        "history_scene_dimensions_m": [restored_env.width, restored_env.height],
        "artifacts": [str(p) for p in sorted((results.RUNS_DIR / completed["run_id"]).iterdir())],
    }
    out = ROOT / "output" / "agent" / "cad_agent_acceptance.json"
    out.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "task_id": pending["task_id"],
                      "run_id": completed["run_id"], "evidence": str(out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
