import csv
import datetime
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import app


SCENARIOS = ["old_community_before.yaml", "community_square.yaml"]
ENGINES = ["rule", "llm"]
SEEDS = [42, 43, 44, 45, 46]
COUNT_VALUES = [8, 8, 8, 8, 8]


def _save_bar_chart(rows, out_dir):
    labels = []
    values = []
    grouped = {}
    for row in rows:
        key = (row["scenario"], row["engine_mode"])
        grouped.setdefault(key, []).append(float(row["total_stay_events"]))
    for key, vals in grouped.items():
        labels.append(f"{key[0].replace('.yaml', '')}\n{key[1]}")
        values.append(sum(vals) / len(vals))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(values)), values, color=["#9CA3AF", "#4F46E5", "#10B981", "#F59E0B"][: len(values)])
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean stay events")
    ax.set_title("Stay events: before/after x rule/LLM")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "paper_stay_events_comparison.png", dpi=220)
    plt.close(fig)


def main():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = app.OUTPUT_DIR / "paper_experiments" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    personas = app.get_default_personas()
    counts = dict(zip(personas.keys(), COUNT_VALUES))
    rows = []
    baseline_by_engine_seed = {}
    results_by_key = {}

    for scenario in SCENARIOS:
        env = app.load_scenario(str(app.SCENARIO_DIR / scenario))
        for engine in ENGINES:
            for seed in SEEDS:
                kwargs = {"engine_mode": engine}
                if engine == "llm":
                    provider, api_key, model = app.get_saved_api_config()
                    if not api_key:
                        # Keep a requested LLM condition distinguishable from a rule run.
                        # A missing key means this condition is unavailable, not that
                        # the experiment silently changed its engine.
                        print(f"Skipping unavailable LLM condition: {scenario}, seed={seed}")
                        continue
                    else:
                        kwargs.update({
                            "api_key": api_key,
                            "api_base": "https://api.deepseek.com/v1",
                            "model": model or "deepseek-chat",
                        })
                result = app.run_sim(env, personas, counts, duration=600, seed=seed, **kwargs)
                config = {
                    "scenario_file": scenario,
                    "engine_requested": engine,
                    "engine_mode": result.engine_mode,
                    "seed": seed,
                    "duration": 600,
                    "dt": 2.0,
                    "counts": counts,
                }
                app.save_run_history(env, result, config, personas, counts)
                row = app.build_summary_row(env, result, config)
                row["engine_requested"] = engine
                rows.append(row)

                key = (scenario, engine, seed)
                results_by_key[key] = (env, result)
                if scenario == "old_community_before.yaml":
                    baseline_by_engine_seed[(engine, seed)] = result

    summary_path = out_dir / "summary.csv"
    with open(summary_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    validation_reports = []
    for (scenario, engine, seed), (env, result) in results_by_key.items():
        baseline = baseline_by_engine_seed.get((engine, seed))
        report = app.build_validation_report(env, result, baseline_result=baseline if scenario != "old_community_before.yaml" else None)
        report.update({"scenario_file": scenario, "engine_requested": engine, "seed": seed})
        validation_reports.append(report)

    with open(out_dir / "validation_report.json", "w", encoding="utf-8") as f:
        json.dump(validation_reports, f, ensure_ascii=False, indent=2)

    _save_bar_chart(rows, out_dir)
    print(f"Paper experiments saved to: {out_dir}")


if __name__ == "__main__":
    main()
