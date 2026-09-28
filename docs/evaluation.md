# Evaluation Notes

The offline evaluator checks deterministic workflow behavior without a provider call. It covers a valid rule workflow, duplicate approval, terminal cancellation, missing-field interruption, and unknown-persona blocking.

The DXF acceptance flow checks unit conversion, YAML publication, preview, approval gating, rule simulation, result validation, SQLite restoration, and duplicate-run reuse.

These checks establish application behavior, not LLM accuracy or real-world pedestrian prediction. A live DeepSeek evaluation must report its fixed test set, model name, Prompt version, sample count, latency, token usage, and per-case result. Rule fallback is reported as degraded and is not counted as pure LLM behavior.
