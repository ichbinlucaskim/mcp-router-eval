# Documentation

## Project status (as of 2026-07-01)

- **Done:** capstone proposal (with a dated post-ground-truth revision) + build-readiness report +
  firsthand ground-truth inspection; **ADRs 0001–0013**; repo scaffold; `scripts/fetch_data.py`
  (dataset pinned to commit `b630b98`). **Contract layer (T1) complete** and merged to `main`:
  - **T1.1** — `contracts.py` frozen: the 4 boundary contracts (RouteResult / ExecPlan / ExecResult /
    Attribution) + nested types (pydantic v2, `extra="forbid"`).
  - **T1.2** — `contract_layer/invariants.py`: closure-complete / dangling-param checks
    (`PARAMETER_*` only, ADR 0013), dependency info **injected** (no loader coupling).
  - **T1.3** — `contract_layer/attribution.py`: deterministic ROUTING/CONTRACT/EXECUTION blame with
    the **upstream-wins** rule; `InvariantReport` + gold set **injected**, imports only `contracts`.
  - **37 tests green on `main`** (20 contracts + 8 invariants + 9 attribution).
- **Current position:** **contract layer (T1) done.**
- **Next:** the **data pipeline** — preprocess (raw→processed, ADR 0011) → loader (ADR 0008/0012) →
  graph_build (PyG graph: 4 edge types + `is_core`, ADR 0006/0013).
- **Deferred — `gate.py` (T1.4):** intentionally not built yet. The gate consumes `confidence` /
  `homophily_local` (router-produced) and is tuned against `completion_rate` (executor-produced), so
  it is YAGNI until the router and executor exist to produce those signals.
- Everything else in `src/mcp_router_eval/` (loader, graph_build, gate, routers, embedding, executor,
  eval) is still an intentional stub (`raise NotImplementedError`).

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
