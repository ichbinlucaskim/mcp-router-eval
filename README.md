# mcp-router-eval

Contract-driven evaluation of GNN tool routing in agentic MCP pipelines. This project measures
the **transfer loss** from retrieval accuracy to end-to-end task completion across a three-layer
pipeline (GNN Router → executor-agnostic Contract Layer → Claude Code executor), and tests whether
a dependency-aware learned GNN router reduces that loss on dependency/composite queries. The middle
**Contract Layer** — typed I/O schemas, dependency-closure invariants, and a router/contract/executor
failure-attribution taxonomy — is the engineering centerpiece.

See [`capstone_proposal.md`](capstone_proposal.md) for the full design (architecture, data contracts,
metrics, and the 14-week task plan) and [`docs/`](docs/) for architecture decision records (ADRs)
capturing the choices made so far.

> Status: **scaffolding only** — modules are stubs; no logic is implemented yet.
