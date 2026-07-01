# Capstone Proposal — Contract-Driven Evaluation of GNN Tool Routing in Agentic MCP Pipelines

**One-line thesis:** Retrieval accuracy is *not* task completion. We build an evaluation framework that measures the loss in transfer from routing quality to end-to-end task completion in an MCP agentic pipeline, design an executor-agnostic *contract layer* with a failure-attribution taxonomy between router and executor, and show that a dependency-aware GNN router reduces this transfer loss on dependency/composite queries.

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