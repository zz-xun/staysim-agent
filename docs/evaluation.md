# Evaluation Notes / 评估说明

The offline evaluator checks deterministic workflow behavior without a provider call. It covers a valid rule workflow, duplicate approval, terminal cancellation, missing-field interruption, and unknown-persona blocking.

离线评估不调用模型服务，检查确定性的工作流行为，包括正常规则流程、重复确认、终态取消、缺参暂停和未知人群阻断。

The DXF acceptance flow checks unit conversion, YAML publication, preview, approval gating, rule simulation, result validation, SQLite restoration, and duplicate-run reuse.

These checks establish application behavior, not LLM accuracy or real-world pedestrian prediction. A live DeepSeek evaluation must report its fixed test set, model name, Prompt version, sample count, latency, token usage, and per-case result. Rule fallback is reported as degraded and is not counted as pure LLM behavior.

这些检查证明的是应用行为，不是 LLM 准确率或真实行人预测能力。真实 DeepSeek 评估必须报告固定测试集、模型、Prompt 版本、样本量、延迟、Token 和逐例结果。规则兜底标记为 degraded，不计入纯 LLM 行为。
