"""Reproduce the local DXF -> approved Agent -> saved simulation result workflow."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from staysim import domain, results
from staysim.scene_import import parse_dxf, preview_scene, save_scene
from staysim_agent.model import request_catalog
from staysim_agent.session import AgentSession


def main():
    output = ROOT / "output" / "agent"
    output.mkdir(parents=True, exist_ok=True)
    source = ROOT / "examples" / "staysim_example_mm.dxf"
    draft = parse_dxf(source, name="DXF验收广场（60×40米）")
    scene = save_scene(draft)
    assert scene in [item["scenario_file"] for item in request_catalog()["scenarios"]]
    assert save_scene(draft) == scene
    figure = preview_scene(draft)
    figure.savefig(output / "dxf-scene-preview.png", dpi=140)
    request = {"scenario_file": scene, "engine_mode": "rule", "duration": 600,
               "counts": {"买菜老人": 20}, "dt": 2.0, "seed": 42}
    # This script is an explicit offline acceptance run: confirm only this fixed fixture.
    with AgentSession() as session:
        pending = session.start(request=request)
        assert pending["status"] == "awaiting_approval" and not pending.get("result")
        completed = session.resume(pending["task_id"], action="approve")
        assert completed["status"] == "completed"
        repeated = session.resume(pending["task_id"], action="approve")
        assert repeated["run_id"] == completed["run_id"]
    with AgentSession() as session:
        restored = session.show(pending["task_id"])
        assert restored["status"] == "completed"
    result, env, _ = results.load_history_result(completed["run_id"])
    assert env.width == 60 and env.height == 40 and result.n_agents == 20
    assert env.source_import["sha256"] == draft["source_import"]["sha256"]
    figure = results.plot_trajectories(env, result)
    figure.savefig(output / "dxf-trajectories.png", dpi=140)
    results._plot_modules()[0].close(figure)
    evidence = {"status": "passed", "source_file": str(source), "scenario_file": scene,
                "scene_dimensions_m": [env.width, env.height], "facility_count": len(env.objects),
                "task_id": completed["task_id"], "run_id": completed["run_id"],
                "request": request, "result": completed["result"],
                "checks": ["DXF毫米与偏移原点转换", "统一场景目录自动识别", "重复导入复用",
                           "确认前不运行", "规则模拟完成", "重复确认不重跑", "SQLite重开恢复",
                           "历史场景快照与来源一致", "预览与轨迹图生成"],
                "live_model_calls": False}
    destination = output / "dxf_evaluation.json"
    destination.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "task_id": completed["task_id"], "evidence": str(destination)}, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
