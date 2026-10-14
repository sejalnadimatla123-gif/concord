# Concord

**Adaptive Multi-Agent Orchestration & Benchmarking System**

Concord is a research and engineering project exploring when collaboration between AI agents improves task performance and when additional communication, latency, and cost outweigh the benefits.

## Current Implementation

- FastAPI backend with `/health` and `/run` endpoints
- Gemini integration through LiteLLM
- Structured agent results with latency, token usage, and cost
- Solo topology
- Solo + Reflect topology
- Automated tests using pytest and mock agents

## Current Limitations and Next Steps

### Execution Traces

`ReflectTopology` currently returns the final reflected answer and aggregate latency, token usage, and cost. It does not yet preserve the initial answer as a separately accessible execution record.

Future work will introduce structured execution traces containing topology identity, individual agent-call outputs, per-call metrics, final results, and failure information. These traces will support debugging and controlled comparisons between topologies.

Unknown cost is represented as `None` rather than reporting an incomplete aggregate as the full cost. Agent-call failures currently propagate to the caller instead of being silently treated as successful runs.

## Research Direction

Concord will compare different agent topologies using controlled experiments. Planned metrics include:

- Task accuracy
- Cost per task and cost per solved task
- Latency
- Token usage and communication overhead
- Failure rates

A contextual-bandit router using LinUCB is planned to learn which topology is most appropriate for a given task.

**Important:** Performance improvements have not yet been established experimentally.
