# Documentation

## Project status (as of 2026-07-05)

Groundwork: capstone proposal (with a dated post-ground-truth revision) + build-readiness report +
firsthand ground-truth inspection; **ADRs 0001–0017**; repo scaffold; `scripts/fetch_data.py`
(dataset pinned to commit `b630b98`).

**Build order is dependency-driven, not §7 phase-number order** (see the proposal's "Build order
(actual, dependency-driven)" note): contract layer → data pipeline → executor → routers/GNN → eval → gate.

- **Done — Contract layer (T1)**, on `main`:
  - **T1.1** `contracts.py` — 4 boundary contracts (RouteResult / ExecPlan / ExecResult / Attribution)
    + nested types (pydantic v2, `extra="forbid"`).
  - **T1.2** `contract_layer/invariants.py` — closure-complete / dangling-param (`PARAMETER_*` only,
    ADR 0013), deps **injected**.
  - **T1.3** `contract_layer/attribution.py` — deterministic ROUTING/CONTRACT/EXECUTION blame,
    **upstream-wins** rule; report + gold set **injected**.
- **Done — Data pipeline**, on `main`:
  - **preprocess** (ADR 0011/0014) — raw→processed JSONL/JSON + fail-loud validation hook (canonical
    types, 573 tools, referential integrity, `PARAMETER_*` acyclic).
  - **loader** (ADR 0008/0012) — processed → `ToolSpec` + injected `tool_deps` + queries (`q{index}`);
    `execution_order()` topo-sorts the `PARAMETER_*` sub-graph.
  - **graph_build** (ADR 0006/0013) — PyG `Data(x, edge_index, edge_type)`, `num_relations=4` typed
    edges + `is_core` node feature; `ToolGraph` contract frozen; **RGCNConv forward smoke test green**.
  - **70 tests green on `main`** (37 contract layer + 13 preprocess + 13 loader/integration + 7 graph_build).
- **Done — Executor primary layer (T2)**, on `main` (ADR 0015/0016/0017):
  - **`executor/mock_tools.py`** — deterministic mock runner (the PRIMARY evaluation substrate;
    SDK is off the critical path, ADR 0015). Argument synthesis honoring `enum`/`default` (ADR 0016);
    structural completion verdict (ADR 0004: all required invoked + `PARAMETER_*` order respected +
    all calls ok); **measured wall-clock latency** reconciling exactly (`total == routing+contract+
    execution`, ADR 0017 — not fabricated).
  - **Unsourced-arg availability rule** — a `PARAMETER_*`-sourced required arg is satisfied only if its
    producer is present *and* ran earlier (structural, not value-threading, ADR 0016 §5); this wires
    **Scenario B → CONTRACT** (producer absent) and **Scenario C → EXECUTION** (producer runs later).
  - **loader → mock runner → attribution proven on real `q240`** (Audible spine); `topo_order()`
    extracted from the loader and reused by the runner. Deterministic failure injection via
    `arg_overrides` (ADR 0017).
  - **83 tests green on `main`** (70 prior + 13 new: 10 `test_mock_tools` + 3 `test_integration`).
- **Cumulative done:** contract layer (T1) + data pipeline + **executor primary (T2)**.
- **Current position:** executor primary done — **entering the router / GNN (T3)**.
- **Cumulative remaining:** **routers / GNN (T3)** → **evaluation / attribution wiring** →
  **SDK replay adapter** (`executor/claude_exec.py`, demonstration only, off the critical path — ADR 0015) →
  **gate** (deferred).
- **Deferred — `gate.py` (T1.4):** consumes `confidence` / `homophily_local` (router) and is tuned
  against `completion_rate` (executor), so it is YAGNI until the router exists (the executor now does).
- Still intentional stubs (`raise NotImplementedError`): `embedding/*`, `executor/claude_exec.py`,
  `routers/*`, `eval/*`, `contract_layer/gate.py`.

## Standing rule — verify before asserting (all sessions)

When any decision, convention, or fact is uncertain or ambiguous — data-format questions, library
behavior, best-practice architecture, an unfamiliar term, whether an approach is standard — **do not
guess and do not proceed on assumption**. **Search the web first**, verify against a credible source,
**cite** what was found, then act. Prefer verified fact over plausible-sounding memory. If a claim
cannot be verified, **say so explicitly** rather than asserting it.

## Reference docs

- [`data-inspection-toollinkos.md`](data-inspection-toollinkos.md) — firsthand ground-truth of the dataset.
- [`build-readiness-report.md`](build-readiness-report.md) — point-in-time environment/dataset inspection (with dated corrections).
- [`completion-scoring-examples.md`](completion-scoring-examples.md) — worked structural-completion scenarios (test-case ready).
- [`feasibility-completion.md`](feasibility-completion.md) — cycle/DAG analysis; PARAMETER sub-graph is acyclic.

## Architecture Decision Records (ADRs)

`adr/` holds one file per significant decision, numbered sequentially (`NNNN-title.md`).

**Convention:** every future significant decision gets a new numbered ADR **before** the code that
implements it is written. Never delete an ADR — supersede it with a new one and mark the old one's
Status accordingly (`Superseded by NNNN`).

Each ADR uses the format in [`adr/0000-adr-template.md`](adr/0000-adr-template.md):
Title / Status / Context / Decision / Consequences / Alternatives considered.

### Index

| # | Decision | Status |
| --- | --- | --- |
| [0001](adr/0001-toollinkos-sole-primary-benchmark.md) | ToolLinkOS is the sole primary benchmark; ToolSandbox demoted to stretch | Accepted |
| [0002](adr/0002-executor-claude-agent-sdk.md) | Executor = Claude Code via claude-agent-sdk with mock tools as MCP tools | Accepted |
| [0003](adr/0003-embedding-provider-interface.md) | Embedding behind a provider interface; LocalEmbedder(BGE) default, ada-002 optional | Accepted |
| [0004](adr/0004-completion-structural-proxy.md) | Completion is a structural proxy, not semantic success | Accepted |
| [0005](adr/0005-closure-depth-slices.md) | Query slice = closure-depth buckets (shallow 2-3 / deep >=6) | Accepted |
| [0006](adr/0006-four-edge-types.md) | 4 real dependency edge types; "core" is a node feature | Accepted |
| [0007](adr/0007-pyg-over-dgl.md) | GNN framework = PyTorch Geometric (not DGL) | Accepted |
| [0008](adr/0008-graph-identity-conventions.md) | Tool identity = name; synthetic query_id = q{index} | Accepted |
| [0009](adr/0009-record-decisions-as-adrs.md) | Record architecture decisions as numbered ADRs before implementing them | Accepted |
| [0010](adr/0010-gnn-model-scope.md) | Limit GNN routers to R-GCN + GAT; GraphSAGE control; exclude Graph Transformer | Accepted |
| [0011](adr/0011-preprocessing-stage.md) | Normalize raw data in a dedicated preprocessing stage; loaders read processed, not raw | Accepted |
| [0012](adr/0012-execution-order-topo-sort.md) | Execution order = topo-sort of the PARAMETER_* sub-graph; golden order is not runnable order | Accepted (amended 2026-07-01) |
| [0013](adr/0013-edge-type-functional-split.md) | Edge functional split: PARAMETER_* = ordering (acyclic), TOOL_* = router representation | Accepted |
| [0014](adr/0014-processed-artifact-format.md) | Processed artifacts = JSONL + JSON metadata (not parquet) at this scale | Accepted |
| [0015](adr/0015-executor-mock-primary-sdk-replay.md) | Executor = deterministic mock runner (primary) + claude-agent-sdk replay adapter (demonstration) | Accepted |
| [0016](adr/0016-mock-arg-synthesis.md) | Mock executor synthesizes args minimally (required, type-valid) honoring enum/default, deterministically | Accepted |
| [0017](adr/0017-latency-and-failure-injection.md) | Latency = measured wall-clock (not synthetic); failure scenarios deterministic, point-injected | Accepted |
| [0018](adr/0018-router-interface-signals.md) | Router interface: normalized confidence common to all routers; model-specific signals (homophily) not forced | Accepted |
