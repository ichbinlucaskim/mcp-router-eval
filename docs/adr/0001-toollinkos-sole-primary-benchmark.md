# 0001 — ToolLinkOS as sole primary benchmark; ToolSandbox demoted to stretch

## Status

Accepted

## Context

The proposal assumed two benchmarks: ToolLinkOS and ToolSandbox. Inspection of the actual sources
(`docs/build-readiness-report.md` §2) found:

- **ToolLinkOS** ships as three JSON files under an **MIT license** with 523 regular + 50 core = 573
  tools and, critically, a gold **query set** (`instances.json`, 1,569 queries with
  `golden_function_names`). Retrieval metrics are therefore computable.
- **Apple ToolSandbox** is a different kind of benchmark — a stateful, conversational, executable
  framework with a user simulator — **not** a dependency-typed retrieval graph, and it ships **no**
  gold ranked query→tool labels. Its dependency-annotated variant used by the reference paper is
  **not** in any public repo. Its license is a custom Apple license (`NOASSERTION`), so
  **redistribution is not permitted**.

## Decision

ToolLinkOS is the sole primary and guaranteed benchmark for both the retrieval and (sampled)
execution layers. ToolSandbox is demoted to a stretch goal, usable only if its dependency
annotation is independently replicated, and is never redistributed in this repo (link/submodule only).

## Consequences

- The "two-layer reproducible benchmark" deliverable is scoped to ToolLinkOS; two-benchmark language
  in the proposal is softened.
- `data/raw/` stays gitignored (ADR-aligned): raw dataset files are fetched locally, not committed.
- Removes the risk of blocking on an unavailable/unlicensed second benchmark.

## Alternatives considered

- **Keep both as co-primary** — rejected: ToolSandbox lacks the required gold ranked labels and its
  license blocks redistribution.
- **Synthesize a second dependency benchmark** — rejected: out of scope (proposal §1.4 non-goals).
