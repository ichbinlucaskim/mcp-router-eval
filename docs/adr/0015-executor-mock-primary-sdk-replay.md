# 0015 — Executor = deterministic mock runner (primary) + claude-agent-sdk replay adapter (demonstration)

## Status

Accepted

## Context

Layer 3 (the executor) consumes an `ExecPlan` and must return an `ExecResult` (§3.3). Two facts from
earlier decisions constrain how it can be built:

- Our tools are **fictional / non-executing**: the ToolLinkOS tools have no real backends, so an agent
  cannot actually *do* anything with them. Completion is therefore a **structural proxy**, not semantic
  task success (ADR 0004).
- Execution order is defined as a **topo-sort of the `PARAMETER_*` sub-graph** (ADR 0012), and argument
  validity is checked against the **built JSON Schema** (ADR 0013/0014). The verdict we need is
  well-defined and structural — correct tool set, valid order, type-valid args — not open-ended.

Agent evaluation has a core **fidelity-vs-reproducibility trade-off**: benchmarks that connect to live
services gain realism but introduce temporal instability, while mocked/synthetic environments gain
determinism but can underspecify real behavior
([Agent-Diff](https://arxiv.org/html/2602.11224v1)). The binding problem for us is **LLM
non-determinism**: a recorded live trace cannot be replayed to reproduce identical behavior, so an
executor built directly on the live SDK would make proposal **ablation A** (isolate the router by
holding the executor fixed) impossible — the executor would not be fixed.

Two lines of evidence say the structural verdict should be computed deterministically in code, and that
a deterministic mock environment is the reproducibility standard for this kind of check:

- For **well-defined, structural verdicts** (were the required tools called, in the right order, with
  valid params), code-based deterministic checks are the recommended method — the survey frames
  code-based evaluation as the most deterministic/objective approach for tasks with well-defined
  outputs ([Evaluation and Benchmarking of LLM Agents: A
  Survey](https://arxiv.org/html/2507.21504v1)), and Tool Correctness is explicitly a **deterministic**
  measure (not an LLM judge) that verifies the required tools were called
  ([Confident AI](https://www.confident-ai.com/blog/llm-agent-evaluation-complete-guide)).
- **Deterministic mock environments** are the current reproducibility standard for agent benchmarks:
  tasks run on deterministic mock services and are scored over final state so that results reproduce
  while allowing diverse solution paths ([LiveClawBench](https://arxiv.org/html/2604.13072v1)), with
  full state management and **deterministic snapshot/restore**
  ([ClawsBench](https://arxiv.org/html/2604.05172v1)).

Separately, the **replay pattern** lets us exercise the real SDK without paying its non-determinism on
every run: record a trace once, then replay it by substituting deterministic stubs for the
non-deterministic components (LLM/tool calls), reproducing the exact execution path at near-zero cost —
"given the same trace, the agent must produce the same deterministic output every time"
([Deterministic Replay for AI Agents](https://www.sakurasky.com/blog/missing-primitives-for-trustworthy-ai-part-8/);
record-and-replay for LLM agents, [AgentRR](https://arxiv.org/html/2505.17716v1)).

## Decision

The executor is built in **two layers**:

1. **PRIMARY — a deterministic mock runner.** It consumes an `ExecPlan` and produces an `ExecResult`
   via a **structural verdict**: the correct tool set + `PARAMETER_*` topo-sorted order (ADR 0012) +
   type-valid arguments checked against the built JSON Schema (ADR 0013/0014). **All evaluation and all
   ablations run on this layer.** It is reproducible and zero-cost (no model calls).
2. **ADAPTER — a `claude-agent-sdk` replay path.** It runs a **small** set of queries through the real
   SDK (per ADR 0002) **once**, records the trace, and thereafter **replays and validates it
   deterministically**. This provides SDK-integration evidence and target-role alignment; it is **not**
   on the critical path of the full evaluation.

### Relationship to proposal §7

This **refines**, and does not contradict, §7 T2.1's literal "run end-to-end via the SDK." For
reproducibility the **mock runner is primary** and the SDK is exercised via **replay as a
demonstration**, not as the evaluation substrate. Stated explicitly here so §7 and this ADR do not read
as conflicting: §7's phrasing describes the SDK integration; the evaluation actually runs on the
deterministic mock runner, with the SDK reached through the replay adapter.

## Consequences

- **Ablation A is guaranteed:** because the primary runner is deterministic, the executor is genuinely
  held fixed, so router isolation is sound.
- **Evaluation is reproducible and cheap:** no live model calls on the critical path; identical inputs
  give identical `ExecResult`s.
- **SDK integration is demonstrated but off the critical path:** the replay adapter supplies the
  hands-on MCP/agent-SDK evidence without injecting non-determinism into the numbers.
- The **replay-trace storage format** is a later decision — flagged here, **not** decided now.

## Alternatives considered

- **Run ALL queries through the live SDK** — rejected: LLM non-determinism breaks ablation-A isolation,
  adds per-run cost, and fictional tools cannot yield semantic completion anyway (ADR 0004), so the
  extra fidelity buys nothing the structural verdict needs.
- **Pure mock, no SDK at all** — rejected: loses the target-role "hands-on MCP / agent-SDK" evidence
  and the ecological-fidelity signal that the replay adapter provides cheaply.

## Sources

- Agent-Diff — fidelity-vs-reproducibility framing: <https://arxiv.org/html/2602.11224v1>
- Evaluation and Benchmarking of LLM Agents: A Survey — code-based deterministic checks:
  <https://arxiv.org/html/2507.21504v1>
- Confident AI — Tool Correctness is a deterministic (non-judge) measure:
  <https://www.confident-ai.com/blog/llm-agent-evaluation-complete-guide>
- LiveClawBench — deterministic mock environments, outcome-driven reproducibility:
  <https://arxiv.org/html/2604.13072v1>
- ClawsBench — high-fidelity mock services with deterministic snapshot/restore:
  <https://arxiv.org/html/2604.05172v1>
- Deterministic Replay for AI Agents — record once, replay deterministically at near-zero cost:
  <https://www.sakurasky.com/blog/missing-primitives-for-trustworthy-ai-part-8/>
- AgentRR — record-and-replay for LLM agents: <https://arxiv.org/html/2505.17716v1>
