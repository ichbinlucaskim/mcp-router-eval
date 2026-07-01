# Capstone Proposal — Contract-Driven Evaluation of GNN Tool Routing in Agentic MCP Pipelines

**One-line thesis:** Retrieval accuracy is *not* task completion. We build an evaluation framework that measures the loss in transfer from routing quality to end-to-end task completion in an MCP agentic pipeline, design an executor-agnostic *contract layer* with a failure-attribution taxonomy between router and executor, and show that a dependency-aware GNN router reduces this transfer loss on dependency/composite queries.

> **NORTH STAR:** Evaluate MCP tool routing by **end-to-end structural task completion**, not retrieval accuracy, and test whether a **dependency-aware GNN router** reduces the **retrieval→completion transfer loss** on **deep-dependency** queries.

> ℹ️ This document has a dated revision at the bottom — [**Revised 2026-07-01 (post-ground-truth)**](#revised-2026-07-01-post-ground-truth) — recording how the real ToolLinkOS data changed the design. The original text below is intentionally left intact; read it together with the revision.

---

## 1. Background & Problem Statement

### 1.1 Context

An LLM agent operating over hundreds of MCP (Model Context Protocol) tools must select the correct subset for a given user query, then execute them to complete a task. Prior work (Graph RAG-Tool Fusion, Agent-as-a-Graph) stops at **retrieval accuracy** (mAP / recall / nDCG over the tool set). But in practice a retrieval hit does not guarantee task success: a correctly retrieved tool can still fail end-to-end because a *dependency tool* was omitted, the call *order* was wrong, or a required *parameter* was never sourced.

### 1.2 Gap

Two gaps, one per axis:

- **Evaluation gap:** No benchmark measures the *transfer* from retrieval quality to end-to-end completion, nor attributes failure to router vs. contract vs. executor.
- **Method gap:** Existing "graph" tool retrieval uses graph *traversal* (heuristic expansion over dependency edges), not a *learned* GNN. Whether learned message passing beats traversal on dependency-heavy queries is untested.

### 1.3 Core Research Questions

- **RQ1:** How much does retrieval accuracy over-estimate end-to-end task completion in an MCP pipeline? (the "transfer loss")
- **RQ2:** Does a dependency-aware learned GNN router reduce transfer loss vs. embedding / traversal baselines, specifically on dependency/composite queries?
- **RQ3:** When the GNN is unreliable (low-homophily / low-confidence regions), can a completion-rate-tuned gate recover performance by falling back to vector retrieval?
- **RQ4:** When a task fails, can a contract layer attribute the failure to routing vs. contract violation vs. execution?

### 1.4 Scope guardrails (explicit non-goals)

- Not building a new agent framework — Claude Code (Anthropic Agent SDK) is the executor.
- Not generating a new dataset — ToolLinkOS / ToolSandbox are the benchmarks.
- Not solving test generation, UI understanding, or multi-modal (out of scope).
- Retrieval layer evaluated on the **full** benchmark; execution layer on a **bounded sample** (1–2 domains, curated composite queries) to keep cost finite.

---

## 2. System Architecture

Three layers. The middle layer (Contract) is the primary engineering contribution.

```
                          ┌──────────────────────────────────────────────────┐
                          │                  USER QUERY (q)                    │
                          │      "get the current price of Tesla stock"        │
                          └───────────────────────┬──────────────────────────┘
                                                  │ q : str
                                                  ▼
        ┌──────────────────────────────────────────────────────────────────────┐
        │ LAYER 1 — ROUTER  (GNN over the tool dependency graph)                 │
        │                                                                        │
        │   Tool Graph G = (V, E)                                                │
        │     V = tools, x_v = text embedding of tool doc                        │
        │     E = typed dependency edges (param-source / precond / core)         │
        │                                                                        │
        │   query-conditioned R-GCN / GAT  →  score(v | q)  for all v ∈ V        │
        │   select top-k + dependency closure                                    │
        └───────────────────────────────────┬────────────────────────────────────┘
                                            │ RouteResult  (see §3.1)
                                            ▼
        ┌──────────────────────────────────────────────────────────────────────┐
        │ LAYER 2 — CONTRACT  (executor-agnostic; the core contribution)         │
        │                                                                        │
        │   (a) INTERFACE   validate RouteResult schema                          │
        │   (b) INVARIANTS  check dependency-closure, no dangling param source   │
        │   (c) GATE        if low-confidence/low-homophily → vector fallback    │
        │   (d) TRACE INIT  open an attributable execution trace                 │
        └───────────────────────────────────┬────────────────────────────────────┘
                                            │ ExecPlan  (see §3.2)
                                            ▼
        ┌──────────────────────────────────────────────────────────────────────┐
        │ LAYER 3 — EXECUTOR  (Claude Code / Anthropic Agent SDK)                │
        │                                                                        │
        │   bind tools → drive calls → collect call trace → completion verdict   │
        └───────────────────────────────────┬────────────────────────────────────┘
                                            │ ExecResult  (see §3.3)
                                            ▼
        ┌──────────────────────────────────────────────────────────────────────┐
        │ EVALUATION + ATTRIBUTION                                               │
        │   retrieval metrics (mAP/recall/nDCG)                                  │
        │   execution metrics (completion / latency / context-fidelity)          │
        │   failure attribution: ROUTING | CONTRACT | EXECUTION  (see §3.4)      │
        └──────────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Contracts (exact I/O of every interface)

These schemas are the spine of the project. They are defined once and frozen by Week 3.

### 3.1 `RouteResult` — Router → Contract

```
RouteResult {
  query_id        : str
  query_text      : str
  ranked_tools    : List[ToolScore]      # full ranking, for retrieval metrics
  selected_tools  : List[tool_id]        # top-k after closure expansion
  closure_edges   : List[Edge]           # dependency edges used to expand
  confidence      : float                # router self-estimate ∈ [0,1]
  homophily_local : float                # mean neighbor similarity of selected set
  router_name     : str                  # for ablation bookkeeping
}
ToolScore { tool_id: str, score: float, rank: int }
Edge      { src: tool_id, dst: tool_id, type: ENUM[param_source|precond|core] }
```

### 3.2 `ExecPlan` — Contract → Executor

```
ExecPlan {
  query_id        : str
  query_text      : str
  bound_tools     : List[ToolSpec]       # validated, closure-complete tool set
  invariant_report: InvariantReport      # what the contract checked & guaranteed
  gate_decision   : ENUM[pass|fallback]  # did the gate swap in vector retrieval?
  trace_id        : str                  # opened trace handle
}
ToolSpec { tool_id: str, name: str, schema: JSONSchema, deps: List[tool_id] }
InvariantReport {
  closure_complete : bool                # all deps of selected tools present
  dangling_params  : List[str]           # required params with no sourcing tool
  violations       : List[str]
}
```

### 3.3 `ExecResult` — Executor → Evaluation

```
ExecResult {
  query_id        : str
  trace_id        : str
  call_trace      : List[ToolCall]       # actual calls in order
  completed       : bool                 # task-level success verdict
  latency_ms      : { routing: int, contract: int, execution: int, total: int }
  tools_used      : List[tool_id]        # subset of bound_tools actually invoked
}
ToolCall { tool_id: str, args: dict, ok: bool, error: str|null, t_ms: int }
```

### 3.4 `Attribution` — Evaluation output (the differentiator)

```
Attribution {
  query_id   : str
  outcome    : ENUM[success | failure]
  blame      : ENUM[ none | ROUTING | CONTRACT | EXECUTION ]
  evidence   : str
}
# Decision rule (applied post-hoc, deterministic):
#   retrieval missed a required tool        → blame = ROUTING
#   closure incomplete / dangling param     → blame = CONTRACT
#   tools present & valid but call failed    → blame = EXECUTION
```

---

## 4. Metrics

| Layer | Metric | Definition |
| --- | --- | --- |
| Retrieval | mAP@{10,20}, recall@{10,20}, nDCG@10 | over ranked_tools vs. gold tool set |
| Execution | completion_rate | fraction of queries with `completed=true` |
| Execution | latency | total + per-layer (routing / contract / execution) |
| Execution | context_fidelity | ` |
| Execution | call_efficiency | `1 - |
| Transfer | **transfer_loss** | `recall@k − completion_rate` on the matched query set |
| Attribution | blame distribution | % failures attributed to ROUTING / CONTRACT / EXECUTION |

**Primary result format:** every metric reported sliced by query type — **single-tool** vs. **dependency/composite** — because the GNN's expected advantage lives entirely in the composite slice.

---

## 5. Models

### 5.1 Baselines (public code; fair comparison)

- **BM25** — lexical.
- **Naïve RAG** — dense embedding cosine top-k.
- **Hybrid RAG** — lexical + dense fusion.
- **Graph RAG-Tool Fusion** — vector seed + dependency-edge traversal (the strongest published non-learned baseline).

### 5.2 Proposed router

- **GraphSAGE** — structure-light control (does message passing help at all?).
- **R-GCN / GAT (query-conditioned)** — primary. Typed messages per edge type; query injected as a conditioning vector into attention.
- **Analysis axis — homophily:** characterize where the GNN wins/loses as a function of local homophily of the selected tool set. Low-homophily dependency edges (a tool whose dependency is semantically *dissimilar*) are exactly where cosine fails and message passing should win.

### 5.3 Gate (selective invocation)

Threshold on `confidence` / `homophily_local`, **tuned against completion_rate (not retrieval accuracy)**. Below threshold → fall back to vector retrieval. Reports how much of the dependency-slice gap the gate recovers.

---

## 6. Experimental Design

- **Isolation (ablation A):** hold contract + executor fixed, swap only the router. Any Δ in completion is attributable to the router.
- **Edge-type ablation (B):** drop edge types (param_source / precond / core) one at a time; measures which dependency structure the GNN exploits.
- **Homophily ablation (C):** bucket queries by `homophily_local`; report completion per bucket.
- **Gate ablation (D):** with / without gate; recovery on the low-homophily bucket.
- **Statistics:** 5 seeds, report mean ± std; paired significance test (router vs. baseline) on completion_rate over the composite slice.

---

## 7. Work Breakdown — 14 Weeks, Task-Level

Each task lists **input → output → done-when**.

### Phase 0 — Reproduce (Weeks 1–2)

- **T0.1** Stand up ToolLinkOS + ToolSandbox locally.
    - in: public repos / datasets → out: loadable graph + query sets → done-when: dataset stats match the paper (573 tools, ~6.3 deps/tool).
- **T0.2** Reproduce published baseline retrieval numbers (BM25, naïve/hybrid RAG, Graph RAG-Tool Fusion).
    - in: benchmark + baseline code → out: mAP@10 table → done-when: numbers within ±2% of reported.
- **T0.3** Freeze a **query taxonomy**: label each query single-tool vs. dependency/composite.
    - in: query set → out: per-query type label → done-when: 100% labeled, spot-checked.

### Phase 1 — Contract Layer (Week 3) — *core contribution, built early*

- **T1.1** Define & freeze schemas §3.1–3.4 as typed dataclasses/pydantic.
    - in: §3 spec → out: `contracts.py` with validators → done-when: round-trip serialization tested.
- **T1.2** Implement invariant checks (closure-complete, dangling-param, violations).
    - in: RouteResult + tool dep table → out: InvariantReport → done-when: unit tests on hand-built pass/fail cases.
- **T1.3** Implement the deterministic attribution rule (§3.4).
    - in: RouteResult + ExecResult → out: Attribution → done-when: synthetic failures classify correctly.

### Phase 2 — Executor (Weeks 4–5)

- **T2.1** Wire Claude Code (Anthropic Agent SDK) as executor behind the contract interface.
    - in: ExecPlan → out: ExecResult with call_trace → done-when: a single-tool query runs end-to-end and returns `completed`.
- **T2.2** Implement per-layer latency + trace logging.
    - in: live run → out: latency dict + trace → done-when: latency sums reconcile to total.
- **T2.3** Bound the execution sample (pick 1–2 domains, curate composite queries).
    - in: query taxonomy → out: execution test set → done-when: set size fixed and documented.

### Phase 3 — GNN Router (Weeks 6–8)

- **T3.1** Build the tool graph: nodes=tools, node features=tool-doc embeddings, typed dependency edges.
    - in: ToolLinkOS deps → out: PyG/DGL graph object → done-when: graph stats logged, no orphan edges.
- **T3.2** Implement GraphSAGE control + query-conditioned R-GCN/GAT.
    - in: graph + query embeddings → out: score(v|q) → done-when: trains, loss decreases, retrieval metrics computed.
- **T3.3** Closure expansion + emit RouteResult (incl. confidence, homophily_local).
    - in: scores → out: RouteResult → done-when: passes contract validation.

### Phase 4 — Evaluation & Attribution (Weeks 9–10)

- **T4.1** Run full retrieval-layer eval, all routers, sliced by query type.
    - in: routers + benchmark → out: retrieval table → done-when: GNN vs. baselines table complete.
- **T4.2** Run execution-layer eval on the bounded set; compute transfer_loss.
    - in: ExecResult stream → out: completion / latency / fidelity tables → done-when: transfer_loss per slice reported.
- **T4.3** Run attribution; produce blame distribution per router.
    - in: Attribution stream → out: ROUTING/CONTRACT/EXECUTION breakdown → done-when: every failure attributed.

### Phase 5 — Gate (Week 11)

- **T5.1** Tune gate threshold against completion_rate; evaluate recovery on low-homophily bucket.
    - in: confidence/homophily + completion → out: gate curve + recovery number → done-when: with/without-gate comparison done.

### Phase 6 — Ablations (Week 12)

- **T6.1** Run ablations A–D (§6).
    - in: frozen pipeline → out: ablation tables → done-when: each ablation has a clean Δ.

### Phase 7 — Write-up (Weeks 13–14)

- **T7.1** Draft report (problem, system, contract, results, attribution, limitations).
- **T7.2** Final figures + defense slides + reproducible eval harness README.
    - done-when: a fresh clone reproduces the headline table.

---

## 8. Deliverables

1. **GNN router** (GraphSAGE control + query-conditioned R-GCN/GAT).
2. **Executor-agnostic contract layer** — schemas, invariant checks, and the failure-attribution harness. *(The portfolio centerpiece: it is the thing a framework would otherwise hide.)*
3. **Two-layer reproducible benchmark** — retrieval + execution, sliced by query type, with transfer-loss and attribution reporting.

---

## 9. Risks & Safeguards

| Risk | Safeguard |
| --- | --- |
| Executor wiring eats the schedule | Claude Code chosen for familiarity; single-tool e2e working by Week 5 before GNN starts. |
| GNN doesn't beat baselines on overall average | Claim is scoped to the **composite slice** + transfer-loss reduction; a null result is still an honest "retrieval ≠ completion" contribution. |
| Execution cost explodes over full benchmark | Retrieval on full set, execution on a bounded 1–2 domain sample. |
| Contract layer feels like plumbing | It is reframed as the contribution: explicit interface + invariants + attribution is precisely the systems-design evidence the target role wants. |

---

## 10. Alignment Note (target role)

This capstone directly exercises: MCP tool/server performance evaluation across an agentic pipeline (latency, context fidelity, task completion); building evaluation frameworks and custom benchmarks comparing routing strategies against a strong published baseline; hands-on MCP tool integration via the Anthropic Agent SDK; and experimental design with seeded, sliced, significance-tested results.

---

*Stretch (only if time remains): add LangGraph as a second executor behind the same contract to demonstrate the contract layer is executor-agnostic.*

---

## Revised 2026-07-01 (post-ground-truth)

The original design above was written before the ToolLinkOS data was inspected firsthand. Standing up
the dataset (fetch + inspection) and freezing the contract layer surfaced facts that changed several
choices. The original prose is left intact; this delta records what the real data changed and why.
Each line cites the ADR / inspection that drove it (see [`docs/`](docs/)).

**Data pipeline**

- **Preprocessing stage added (raw → processed).** Raw ToolLinkOS is dirty — type aliasing
  (`bool`/`boolean`, `int`/`integer`), 21 non-scalar `dict`/`list`/`array` params, `enum`/`default`
  side-keys, 2 malformed dependency rows, 2 malformed param objects. A dedicated stage normalizes
  `data/raw/` → `data/processed/`; loaders and all pipeline code read **processed only** (raw is
  immutable). *(ADR 0011; `docs/data-inspection-toollinkos.md`.)*

**Dependency edges & execution order**

- **Edges split by function.** `PARAMETER_*` = execution precondition; `TOOL_*` = conceptual
  association. The router/GNN sees all **4** relation types (`param_direct`, `param_indirect`,
  `tool_direct`, `tool_indirect`); *ordering* consumes `PARAMETER_*` only. *(ADR 0006/0013.)*
- **Execution order = topo-sort of the `PARAMETER_*` sub-graph.** The full 4-type graph is **cyclic**
  (485/573 nodes; **1,374/1,569 gold sets** contain a cycle) because of `TOOL_INDIRECTLY` get/set
  pairs (e.g. `get_location_service_status ⇄ set_location_service_status`). The `PARAMETER_*`
  sub-graph is acyclic across all instances, so ordering uses it; `TOOL_*` is excluded from ordering.
  The stored `golden_function_names` order is **main-first, not runnable order**. *(ADR 0012 amended,
  0013; `docs/feasibility-completion.md`.)*

**Query slicing**

- **Single-vs-composite → closure-depth buckets.** ToolLinkOS has **zero** single-tool queries (every
  instance has ≥2 gold tools; mean ≈6). The §4 headline slice becomes **shallow (2–3 gold tools)** vs
  **deep (≥6)**. *(ADR 0005; inspection.)*

**Completion & embeddings**

- **Completion is a STRUCTURAL proxy, not semantic.** Tools are fictional and instances carry no gold
  args/answers, so `completed` = correct tool set + dependency order + type-valid args — not real task
  success. *(ADR 0004.)*
- **Embeddings = local BGE behind a provider interface**, with ada-002 optional (only to reproduce the
  paper's published dense-baseline numbers). Claude has no embedding API; relative comparisons need
  only one shared embedding space. *(ADR 0003.)*

**Benchmark & deliverables**

- **ToolLinkOS is the sole primary benchmark; ToolSandbox is demoted to stretch** (Apple custom
  license = no redistribution; it is a stateful conversational eval, not a dependency-typed retrieval
  graph with gold ranked labels). This **supersedes the "two-layer / two-benchmark" wording in §8**:
  the reproducible benchmark is ToolLinkOS-only (retrieval on the full set, execution on a bounded
  sample), with ToolSandbox as an optional second benchmark only if independently annotated.
  *(ADR 0001.)*

**14-week plan**

- **Phase 0 groundwork now explicitly includes the data-verification work already done**: fetch +
  pin the dataset (`scripts/fetch_data.py`, commit `b630b98`), normalize the type vocabulary, verify
  `core ≠ leaf` (30/50 core tools have deps), and run the cycle/DAG check that scoped ordering to the
  `PARAMETER_*` sub-graph. These are prerequisites to T0.1/T3.1, not incidental.

**Executor SDK naming.** The executor is Claude Code driven via the **`claude-agent-sdk`** Python
package (the current name of what §1.4/§7 call the "Anthropic Agent SDK"); the raw `anthropic` client
cannot drive the agent loop. *(ADR 0002.)*

### Build order (actual, dependency-driven)

The §7 Phase numbering (0 reproduce → 1 contract → 2 executor → 3 GNN → …) was the **initial plan of
record** and is kept intact above. The **actual** build order follows *dependency*, not phase number:

1. **Contract layer (T1)** — built first and **frozen** (schemas → invariants → attribution). It is the
   spine every other component talks through.
2. **Data pipeline** — preprocess (raw→processed) → loader → graph_build.
3. **Executor (T2)** — wired *after* the data pipeline.
4. **Routers / GNN (T3)**.
5. **Evaluation / attribution wiring**, then the **gate**.

**Why the reorder:** a component that *consumes* a contract (executor, routers) is built only once
that contract is frozen **and** its real inputs exist — otherwise it would be written against
throwaway fixtures and reworked. The executor also consumes loader/graph_build output (real
`ToolSpec` schemas), so it follows the data pipeline rather than preceding the GNN.

**Not skipped, just resequenced:** the executor (Phase 2 / T2.1–T2.3) is **not** dropped — it comes
*after* the data pipeline instead of before the GNN. The **gate (T1.4)** stays **deferred** until the
router and executor exist to produce its inputs (`confidence` / `homophily_local` / `completion_rate`).
