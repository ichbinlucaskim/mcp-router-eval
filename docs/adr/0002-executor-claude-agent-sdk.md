# 0002 — Executor = Claude Code via claude-agent-sdk, mock tools as MCP tools

## Status

Accepted

## Context

Layer 3 must bind tools, drive calls, and collect an ordered call trace into `ExecResult` (§3.3).
The environment audit (`docs/build-readiness-report.md` §1) found:

- The raw `anthropic` Python SDK (0.96.0) is present but exposes **no agent/MCP loop** — it cannot
  drive multi-step tool calling on its own.
- The **Claude Code CLI (2.1.197)** is installed, and the **`claude-agent-sdk`** Python package
  (PyPI) wraps it with `query()` / `ClaudeSDKClient`, programmatic MCP-tool registration, and a
  structured message/tool-call stream.

## Decision

The executor drives the installed Claude Code CLI through `claude-agent-sdk`, registering the
generic mock tools (ADR 0004) as in-process MCP tools, and maps the SDK's structured tool-call
stream into `ExecResult.call_trace`.

## Consequences

- `claude-agent-sdk` is added to dependencies; the executor depends on the CLI being installed.
- The SDK's tool-call granularity directly supplies `ToolCall {tool_id, args, ok, error, t_ms}`,
  keeping the executor thin.
- Aligns with the proposal's non-goal of not building a new agent framework (§1.4).

## Alternatives considered

- **Raw `anthropic` + hand-rolled tool loop** — rejected: re-implements the agent loop for no gain
  in trace fidelity.
- **A third-party agent framework (e.g. LangGraph)** — deferred: named only as a stretch second
  executor behind the same contract.
