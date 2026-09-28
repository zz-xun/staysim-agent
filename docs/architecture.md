# Architecture / 架构

StaySim Agent separates the application workflow from the simulation domain.

StaySim Agent 将应用工作流与模拟领域内核分离。

```text
Gradio / CLI
    -> AgentSession
        -> LangGraph state machine
            -> LangChain request parser and preparation tools
                -> validated ExperimentRequest
                    -> human approval interrupt
                        -> controlled run_experiment
                            -> StaySim run_sim()
                                -> result validation and report
```

## Ownership boundaries

- **LangChain** owns model adapters, prompts, structured output, and tool schemas.
- **LangGraph** owns state, routing, pauses, resume, and checkpoint integration.
- **StaySim** owns personas, behavior decisions, movement, metrics, and result artifacts.
- **SQLite** owns durable task state and execution receipts. It is not a knowledge base or RAG store.

The preparation model can inspect a scenario and validate a configuration. It cannot set approval, choose credentials, or call the simulation execution tool. This boundary keeps model suggestions separate from side-effecting execution.

## Failure handling

Missing fields interrupt for clarification. A changed request invalidates the previous plan and approval. A completed execution with a valid receipt is reused on duplicate approval. An execution record without a complete receipt is marked uncertain rather than silently rerun.

缺少参数时流程会暂停并追问；需求变化会使旧计划和确认失效；已有完整回执的任务在重复确认时复用；只有执行记录但没有完整回执时标记为 uncertain，不会静默重跑。
