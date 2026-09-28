# Architecture

StaySim Agent separates the application workflow from the simulation domain.

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
