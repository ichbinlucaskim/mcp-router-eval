# 0004 — Completion is a structural proxy, not semantic success

## Status

Accepted

## Context

Layer 3 reports `completion_rate`, and RQ1 depends on comparing it against retrieval recall. But
(`docs/build-readiness-report.md` §2–3):

- ToolLinkOS tools are **fictional and do not execute**.
- `instances.json` provides gold tool **names** only — **no gold arguments and no gold answer** — so
  there is nothing to check semantic task success against.

## Decision

`completed` is defined as a **structural proxy**: a query is complete iff the agent invokes exactly
the required tool set (`golden_function_names`), respects dependency order (a tool is called only
after its `depends_on` are satisfied), and every call is type-valid against the tool's built
JSONSchema. This is implemented by the generic mock executor (executor/mock_tools.py).

## Consequences

- `transfer_loss = recall@k − completion_rate` measures structural, not semantic, completion; this
  is documented wherever the metric appears.
- No promise of real functional completion is made; keeps the executor within the Week 4–5 budget.
- The mock executor synthesizes type-valid args (marked synthetic in `ExecResult` evidence).

## Alternatives considered

- **Real functional execution** — rejected: impossible with fictional, non-runnable tools.
- **LLM-judged semantic completion** — rejected: no gold answer to judge against; adds cost and
  non-determinism to a verdict that must be reproducible.
