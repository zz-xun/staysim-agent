"""Three Agent capabilities calling StaySim modules directly, without the UI."""

from __future__ import annotations

import os
import hashlib
import json
import math
from pathlib import Path
from uuid import UUID
from typing import Annotated, Any

from langchain_core.tools import InjectedToolArg, tool
from pydantic import ValidationError

from staysim import domain, results, simulation
from .model import ExperimentRequest, runtime_setting


def _inspect_scenario(scenario_file: str) -> dict[str, Any]:
    path = _scenario_path(scenario_file)
    env = domain.load_scenario(str(path))
    return {
        "scenario_file": scenario_file,
        "name": env.name,
        "width": env.width,
        "height": env.height,
        "element_count": len(env.objects),
        "source_import": getattr(env, "source_import", None),
        "elements": [
            {
                "type": obj.element.name,
                "label": obj.label,
                "x": obj.x,
                "y": obj.y,
                "attractiveness": obj.attractiveness,
            }
            for obj in env.objects
        ],
    }


def _scenario_path(scenario_file: str):
    """Only allow a YAML scene directly inside the configured scene directory."""
    root = domain.SCENARIO_DIR.resolve()
    path = (root / scenario_file).resolve()
    if path.parent != root or path.suffix.lower() != ".yaml":
        raise ValueError("scenario_file 必须是 scenarios 目录下的 YAML 文件")
    if not path.is_file():
        raise ValueError(f"找不到场景文件: {scenario_file}")
    return path


@tool
def inspect_scenario(scenario_file: str) -> dict[str, Any]:
    """Inspect one YAML scenario before an experiment is run."""
    return _inspect_scenario(scenario_file)


def _validate_experiment_config(config: dict[str, Any]) -> dict[str, Any]:
    if "api_key" in config:
        raise ValueError("API Key 不能放入实验请求，请设置 STAYSIM_API_KEY 环境变量")
    request = ExperimentRequest.model_validate(config)
    _inspect_scenario(request.scenario_file)
    known_personas = domain.get_default_personas()
    for name in request.counts:
        if name not in known_personas:
            raise ValueError(f"未知人群类型: {name}")
    normalized = request.model_dump()
    if request.engine_mode == "llm":
        api_key = runtime_setting("STAYSIM_API_KEY")
        if not api_key.strip():
            raise ValueError("LLM 模式缺少 API Key，请先配置 STAYSIM_API_KEY")

    return normalized


@tool
def validate_experiment_config(experiment_config: dict[str, Any]) -> dict[str, Any]:
    """Validate one complete simulation configuration without running it."""
    try:
        normalized = _validate_experiment_config(experiment_config)
    except ValidationError as exc:
        errors = [
            {"field": ".".join(map(str, item["loc"])), "message": item["msg"]}
            for item in exc.errors(include_input=False, include_url=False)
        ]
        return {"valid": False, "error": "实验参数缺失或不合法", "details": errors}
    except (TypeError, ValueError) as exc:
        return {"valid": False, "error": str(exc)}
    return {"valid": True, "config": normalized}


def plan_fingerprint(config: dict[str, Any]) -> str:
    """Bind confirmation to parameters, scene bytes and public engine settings."""
    path = _scenario_path(config["scenario_file"])
    payload = {
        "config": config,
        "scene_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "engine_settings": {
            key: runtime_setting(key) for key in (
                ("STAYSIM_API_BASE", "STAYSIM_MODEL_NAME") if config["engine_mode"] == "llm"
                else ("STAYSIM_LOCAL_MODEL",) if config["engine_mode"] == "local" else ()
            )
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


ARTIFACT_NAMES = (
    "config.json", "trajectories.csv", "decisions.jsonl", "stay_events.csv",
    "summary.csv", "validation_report.json", "stats.json", "full_result.json",
    "simulation_report.json",
)


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_saved_result(config: dict[str, Any], outcome: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Check persisted artifacts and the semantic contract after one simulation."""
    errors: list[str] = []
    warnings: list[str] = []
    required = [name for name in ARTIFACT_NAMES if name != "simulation_report.json"]
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        errors.append("缺少结果文件: " + ", ".join(missing))
    stats: dict[str, Any] = {}
    full: dict[str, Any] = {}
    try:
        stats = json.loads((run_dir / "stats.json").read_text(encoding="utf-8"))
        full = json.loads((run_dir / "full_result.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"结果 JSON 无法读取（{type(exc).__name__}）")

    expected_agents = sum(config.get("counts", {}).values())
    actual_agents = stats.get("n_agents", full.get("n_agents"))
    if actual_agents != expected_agents:
        errors.append(f"实际人数 {actual_agents} 与配置人数 {expected_agents} 不一致")
    requested = config.get("engine_mode")
    actual_requested = stats.get("requested_engine_mode", stats.get("engine_mode"))
    if actual_requested != requested:
        errors.append("结果中的请求模式与配置不一致")
    for key in ("spatial_entropy", "stay_pass_ratio", "mean_stay_duration", "elapsed_seconds"):
        if key in stats and not _finite(stats[key]):
            errors.append(f"指标 {key} 不是有限数")
    trajectories = full.get("trajectories", {})
    for persona, count in config.get("counts", {}).items():
        if len(trajectories.get(persona, [])) != count:
            errors.append(f"人群 {persona} 的轨迹数量与配置不一致")
        if count and not trajectories.get(persona):
            errors.append(f"人群 {persona} 没有轨迹数据")
    fallback_count = int(stats.get("fallback_count", outcome.get("fallback_count", 0)) or 0)
    quality = stats.get("simulation_quality", outcome.get("simulation_quality", "ok"))
    if requested == "llm" and fallback_count:
        warnings.append("LLM 决策发生失败并使用了兜底，结果不能视为纯 LLM 模拟")
    if requested == "llm" and quality == "ok" and not stats.get("llm_calls", 0):
        warnings.append("请求了 LLM，但没有记录到 LLM 调用")
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "evidence": {
            "artifact_count": len(required),
            "actual_agents": actual_agents,
            "requested_engine_mode": requested,
            "effective_engine_mode": stats.get("effective_engine_mode", requested),
            "llm_calls": stats.get("llm_calls", 0),
            "llm_failure_count": stats.get("llm_failure_count", 0),
            "fallback_count": fallback_count,
            "simulation_quality": quality,
        },
    }


def build_simulation_report(config: dict[str, Any], outcome: dict[str, Any], validation: dict[str, Any]) -> dict[str, Any]:
    """Create a model-readable report whose facts come from the simulator."""
    return {
        "run_id": outcome.get("run_id"),
        "status": "completed" if validation.get("valid") else "invalid",
        "scenario": outcome.get("scenario"),
        "source_import": outcome.get("source_import"),
        "requested_config": config,
        "effective_engine": {
            "requested": outcome.get("requested_engine_mode", config.get("engine_mode")),
            "effective": outcome.get("effective_engine_mode", outcome.get("engine_mode")),
            "llm_calls": outcome.get("llm_calls", 0),
            "llm_failure_count": outcome.get("llm_failure_count", 0),
            "fallback_count": outcome.get("fallback_count", 0),
            "fallback_reasons": outcome.get("fallback_reasons", []),
            "simulation_quality": outcome.get("simulation_quality", "ok"),
        },
        "metrics": {key: outcome[key] for key in (
            "n_agents", "total_stay_events", "spatial_entropy", "elapsed_seconds", "total_tokens"
        ) if key in outcome},
        "warnings": validation.get("warnings", []),
        "artifact_paths": {name: str(Path(outcome["run_dir"]) / name) for name in ARTIFACT_NAMES},
        "evidence": validation.get("evidence", {}),
    }


def read_execution_receipt(run_id: str, fingerprint: str) -> dict | None:
    """A receipt is valid only after all persisted artifact hashes were committed."""
    if str(UUID(run_id)) != run_id:
        raise ValueError("运行编号无效")
    folder = results.RUNS_DIR / run_id
    receipt_path = folder / "execution_receipt.json"
    if not receipt_path.exists():
        return None
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt["fingerprint"] != fingerprint or receipt["run_id"] != run_id:
        raise ValueError("运行记录与确认参数不一致")
    receipt_artifacts = receipt.get("artifacts", {})
    legacy_names = tuple(name for name in ARTIFACT_NAMES if name != "simulation_report.json")
    names_to_check = ARTIFACT_NAMES if "simulation_report.json" in receipt_artifacts else legacy_names
    for name in names_to_check:
        digest = hashlib.sha256((folder / name).read_bytes()).hexdigest()
        if digest != receipt_artifacts.get(name):
            raise ValueError("运行文件不完整或已改变")
    return {**receipt["result"], "run_dir": str(folder),
            "legacy_receipt": "simulation_report.json" not in receipt_artifacts}


def _run_experiment(config: dict[str, Any], *, run_id: str | None = None) -> dict[str, Any]:
    normalized = _validate_experiment_config(config)
    fingerprint = plan_fingerprint(normalized)
    env = domain.load_scenario(str(_scenario_path(normalized["scenario_file"])))
    personas = domain.get_default_personas()
    result = simulation.run_sim(
        env,
        personas,
        normalized["counts"],
        duration=normalized["duration"],
        dt=normalized["dt"],
        seed=normalized["seed"],
        engine_mode=normalized["engine_mode"],
        api_key=(runtime_setting("STAYSIM_API_KEY").strip()
                 if normalized["engine_mode"] == "llm" else None),
        api_base=runtime_setting("STAYSIM_API_BASE") or None if normalized["engine_mode"] == "llm" else None,
        model=(runtime_setting("STAYSIM_MODEL_NAME") or None if normalized["engine_mode"] == "llm"
               else runtime_setting("STAYSIM_LOCAL_MODEL", "llama3")
               if normalized["engine_mode"] == "local" else None),
    )
    run_dir = results.save_run_history(
        env, result, normalized, personas, normalized["counts"], run_id=run_id,
    )
    outcome = {
        "run_id": run_id,
        "run_dir": run_dir,
        "scenario": result.env_name,
        "source_import": getattr(env, "source_import", None),
        "engine_mode": result.engine_mode,
        "requested_engine_mode": getattr(result, "requested_engine_mode", result.engine_mode),
        "effective_engine_mode": getattr(result, "effective_engine_mode", result.engine_mode),
        "n_agents": int(result.n_agents),
        "total_stay_events": int(result.total_stay_events),
        "spatial_entropy": float(result.spatial_entropy),
        "elapsed_seconds": float(result.elapsed),
        "llm_calls": int(result.llm_calls),
        "total_tokens": int(result.total_tokens),
        "llm_failure_count": int(getattr(result, "llm_failure_count", 0)),
        "fallback_count": int(getattr(result, "fallback_count", 0)),
        "fallback_reasons": list(getattr(result, "fallback_reasons", [])),
        "simulation_quality": getattr(result, "simulation_quality", "ok"),
    }
    folder = Path(run_dir)
    validation = validate_saved_result(normalized, outcome, folder)
    report = build_simulation_report(normalized, outcome, validation)
    (folder / "simulation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    outcome["result_validation"] = validation
    outcome["report_path"] = str(folder / "simulation_report.json")
    if run_id is not None:
        receipt = {
            "run_id": run_id, "fingerprint": fingerprint, "result": outcome,
            "artifacts": {
                name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                for name in ARTIFACT_NAMES
            },
        }
        temp = folder / "execution_receipt.tmp"
        with temp.open("w", encoding="utf-8") as stream:
            json.dump(receipt, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        temp.replace(folder / "execution_receipt.json")
    return outcome


def validate_execution_result(config: dict[str, Any], outcome: dict[str, Any], fingerprint: str) -> dict[str, Any]:
    """Validate the receipt and return the immutable report for downstream explanation."""
    run_id = outcome.get("run_id")
    if not run_id:
        run_dir = Path(outcome.get("run_dir", ""))
        validation = outcome.get("result_validation") or validate_saved_result(config, outcome, run_dir)
        report_path = run_dir / "simulation_report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
        else:
            report = build_simulation_report(config, outcome, validation)
        return {"valid": validation.get("valid", False), "validation": validation, "report": report}
    receipt = read_execution_receipt(run_id, fingerprint)
    if receipt is None:
        return {"valid": False, "validation": {"valid": False, "errors": ["缺少执行回执"]}}
    run_dir = Path(receipt["run_dir"])
    validation = validate_saved_result(config, receipt, run_dir)
    try:
        report = json.loads((run_dir / "simulation_report.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        if receipt.get("legacy_receipt"):
            report = build_simulation_report(config, receipt, validation)
        else:
            validation["valid"] = False
            validation.setdefault("errors", []).append("simulation_report.json 无法读取")
            report = None
    return {"valid": validation.get("valid", False), "validation": validation, "report": report}


@tool
def run_experiment(
    experiment_config: dict[str, Any],
    run_id: Annotated[str | None, InjectedToolArg] = None,
) -> dict[str, Any]:
    """Run one validated simulation and save its reproducible run artifacts."""
    return _run_experiment(experiment_config, run_id=run_id)
