# Documentation

## Project status (as of 2026-07-05)

Groundwork: capstone proposal (with a dated post-ground-truth revision) + build-readiness report +
firsthand ground-truth inspection; **ADRs 0001–0027**; repo scaffold; `scripts/fetch_data.py`
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
- **Done — Embedding provider (router-track infra)**, on `main` (ADR 0003):
  - **`embedding/base.py`** — frozen `Embedder` interface (providers implement `dim`/`version`/`_embed`;
    the base owns the cache-aware `encode` + on-disk cache + version tag). Cache key =
    `sha256(version + NUL + text)`; vectors segregated into a per-version sub-dir with a `meta.json`
    `{version, dim}` record so a model change never mixes with another model's vectors.
  - **`embedding/local.py`** — `LocalEmbedder` wrapping `BAAI/bge-small-en-v1.5` (384-dim), `eval()` +
    `no_grad`, **deterministic** (same text → identical vector). `openai_embed.py` stays an
    interface-satisfying **stub** (ada-002 optional, no paid API).
  - Cache lives under `data/processed/embeddings/` (regenerable, gitignored). Shared input infra for
    GNN node features (`x[:, 1:]`) and vector baselines.
  - **94 tests green on `main`** (83 prior + 11 `test_embedding`).
- **Done — Router track: first router (T3)**, on `main` (ADR 0018 + 2026-07-05 amendment):
  - **`routers/base.py`** — frozen `Router` interface: `rank(query) → RankResult` is **pure ranking**
    (full `ranked_tools` + top-k), **no closure**. Shared helpers: min-max `normalize_confidence` to
    `[0,1]` over the top-k window with the `M==m`/empty → `1.0` degenerate rule; `HOMOPHILY_NA`
    sentinel (`0.0`, n/a) for non-GNN routers (ADR 0018).
  - **`routers/closure.py`** — the **single shared post-processing stage** (ADR 0018 amendment):
    expands any router's top-k with the transitive `PARAMETER_*` closure only (TOOL_* excluded,
    ADR 0013; reuses `topo_order`), then assembles the `RouteResult`. Identical for every router →
    ablation-A isolates ranking quality.
  - **`routers/baselines.py`** — **BM25Router** over tool documents (tool_id words + param
    descriptions), **tuned `k1=0.9, b=0.4`** (not library defaults, ADR 0018); deterministic.
  - **Full real path proven:** BM25 → shared closure → `RouteResult` → invariants → mock executor →
    attribution on `q240` (SUCCESS; and a dependency-drop → **CONTRACT** through the *real* router).
  - **107 tests green on `main`** (94 prior + 13 `test_routers_bm25`).
- **Done — Router track: vector baselines (T3)**, on `main` (ADR 0003 / 0018 / 0019 / 0020):
  - **`NaiveRAGRouter`** — dense cosine over the embedding provider (LocalEmbedder BGE); **first real
    use of the embedding provider**, reusing its versioned cache so the 573 tool vectors are computed
    once. Pure ranking (ADR 0018); deterministic.
  - **`HybridRAGRouter`** — **convex-combination fusion** (ADR 0019):
    `α·norm(dense) + (1−α)·norm(sparse)` over min-max-normalized scores (ADR 0018), `α` default 0.5,
    configurable. Endpoints reduce exactly to the pure rankers (`α=0` → BM25, `α=1` → NaiveRAG).
  - **Same-text guarantee (ADR 0020):** every router (BM25, NaiveRAG, HybridRAG) embeds/indexes the
    **identical** `tool_document()` text, so the comparison isolates method, not input. Single vector
    `minmax_normalize` shared by confidence + fusion.
  - **Full real path proven** for the dense router too: NaiveRAG → shared closure → invariants → mock
    executor → attribution on `q240` (SUCCESS; dependency-drop → **CONTRACT**).
  - **119 tests green on `main`** (107 prior + 12 `test_routers_vector`).
- **Done — Router track: traversal baseline (T3)**, on `main` (ADR 0021 + 2026-07-05 amendment):
  - **`TraversalRouter`** — reproduces **Graph RAG-Tool Fusion**'s standard (no-rerank) method:
    hybrid initial retrieval → per-tool **DFS** of `PARAMETER_*` deps up to `d_limit` (TOOL_* excluded,
    ADR 0013) → **block-interleaved** order `[v1, deps(v1), v2, deps(v2), …]` de-duped, truncated to
    `final_top_k`. No LLM reranking (determinism, ADR 0015); `k`/`d_limit`/`final_top_k` recorded per run.
  - The **one** router with its own expansion (explicit exception), yet it **still passes through the
    shared closure stage** — final `selected_tools` closure-completeness identical to every router.
  - **Thesis premise shown:** traversal surfaces the low-homophily `validate_email` that NaiveRAG buries
    (rank ≈384); full path → invariants → mock executor → attribution (SUCCESS; drop → **CONTRACT**).
  - **129 tests green on `main`** (119 prior + 10 `test_routers_traversal`).
  - **Baselines COMPLETE:** {BM25, NaiveRAG, HybridRAG, Traversal}.
- **Done — GNN router complete (T3)**, on `main` — design **and** implementation across
  ADR 0010 / 0022 (+amendment) / 0023 / 0024 / 0025 / 0026 / 0027:
  - **Stage 1 — models (`routers/gnn_models.py`):** three **2-layer** backbones (R-GCN `num_relations=4`,
    GAT heads-2, SAGE control; ADR 0010/0025); node features fill graph_build's `x[:, 1:]` with BGE
    embeddings of `tool_document()` (ADR 0020); **late-cosine two-tower** scorer (per-tower L2 + optional
    per-tower projection, **no query-node fusion MLP**; ADR 0022 amendment).
  - **Stage 2 — vectorized training (`routers/gnn_train.py`):** query-level split (ADR 0024, train-only
    stats), masked **InfoNCE** (ADR 0026) with the dependency-structure false-negative mask (ADR 0023),
    **AdamW**. Built to the profiling findings: **one GNN forward per step**, **precomputed
    `[Q×N]` mask**, **batch-embedded queries**, **matmul scoring** — all identical-result fast paths.
  - **Stage 3 — router (`routers/gnn.py`):** `GNNRouter` loads a checkpoint, ranks by late-cosine
    matmul (**pure ranking** + shared closure, ADR 0018/0021), min-max confidence (ADR 0018), and
    computes a **real `homophily_local`** (ADR 0027: mean cosine to `PARAMETER_*` deps; sentinel for
    dependency-free tools). Full-pipeline integration proven (GNNRouter → closure → invariants →
    executor → attribution). Deterministic; checkpoints gitignored.
  - **162 tests green on `main`** (152 prior + 10 `test_gnn_router`; GNN stages add 33 tests total).
- **Five routers now stand behind one contract:** {BM25, NaiveRAG, HybridRAG, Traversal, GNN}.
- **Cumulative done:** contract layer (T1) + data pipeline + **executor primary (T2)** + embedding
  provider + **routers {BM25, NaiveRAG, HybridRAG, Traversal, GNN} + shared closure stage (T3)** +
  **full GNN (design ADRs 0022–0027 + implementation stages 1–3)**.
- **Current position:** **all five routers implemented behind the same contract** — **entering the
  evaluation harness** (the north-star: structural completion + retrieval→completion transfer-loss
  comparison across all five routers).
- **Cumulative remaining:** **evaluation** (retrieval metrics + closure-depth slices, ADR 0005; the
  north-star structural-completion + transfer-loss comparison across all five routers) → **SDK replay
  adapter** (`executor/claude_exec.py`, demonstration only, off the critical path — ADR 0015) →
  **gate** (deferred).
- **Deferred — `gate.py` (T1.4):** consumes `confidence` / `homophily_local` (router) and is tuned
  against `completion_rate` (executor); the full router set now exists, so it unblocks alongside the gate.
- Still intentional stubs (`raise NotImplementedError`): `embedding/openai_embed.py`,
  `executor/claude_exec.py`, `eval/*`, `contract_layer/gate.py`.

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
| [0004](adr/0004-completion-structural-proxy.md) | Completion is a structural proxy, not semantic success | Accepted (amended 2026-07-05; amendment's required-set A/B point superseded by [0030](adr/0030-completion-required-set.md)) |
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
| [0018](adr/0018-router-interface-signals.md) | Router interface: normalized confidence common to all routers; model-specific signals (homophily) not forced | Accepted (amended 2026-07-05) |
| [0019](adr/0019-hybrid-fusion-convex-combination.md) | Hybrid fusion = convex combination of normalized scores (not RRF); α tunable, default 0.5 | Accepted |
| [0020](adr/0020-uniform-router-document-text.md) | All routers index/embed the same tool document text (fair method comparison) | Accepted |
| [0021](adr/0021-traversal-router-expansion.md) | Traversal router keeps its own dependency expansion (explicit exception), then passes through the shared closure | Accepted (amended 2026-07-05) |
| [0022](adr/0022-gnn-formulation-query-conditioned-scoring.md) | GNN router formulation = query-conditioned node scoring (not link prediction / node classification) | Accepted (amended 2026-07-05) |
| [0023](adr/0023-gnn-negative-sampling.md) | GNN negative sampling = in-batch negatives + dependency-structure false-negative filter; hard negatives conservative | Accepted |
| [0024](adr/0024-gnn-split-and-leakage.md) | GNN split = query-level (transductive graph shared); leakage prevented by train-only stats + tuning-only validation | Accepted |
| [0025](adr/0025-gnn-architecture-hyperparameters.md) | GNN architecture: 2 layers fixed; hidden dim / GAT heads / dropout searched on validation | Accepted |
| [0026](adr/0026-gnn-training-setup.md) | GNN training setup: InfoNCE over in-batch negatives; AdamW; warm-up not required at our scale | Accepted |
| [0027](adr/0027-homophily-local-definition.md) | homophily_local = node-level feature homophily (mean cosine to PARAMETER_* dependency neighbors) | Accepted |
| [0028](adr/0028-evaluation-metrics.md) | Evaluation metrics = standard retrieval + structural completion + north-star transfer loss, sliced by depth, deterministic attribution | Accepted |
| [0029](adr/0029-validation-tuning-protocol.md) | Validation tuning = deterministic grid search over the small discrete space; best by validation completion_rate (mAP@10 tiebreaker); multi-seed | Accepted |
| [0030](adr/0030-completion-required-set.md) | Completion required-set = required-argument PARAMETER_* closure (variant A), decoupled from the execution/ordering closure; supersedes the ADR-0004 amendment's A/B ambiguity | Accepted |
| [0031](adr/0031-gnn-popularity-debiasing.md) | GNN popularity debiasing = logQ correction in InfoNCE (contrastive–IPW equivalence, train-only frequency); with a note that the ADR-0023 false-negative filter may have weakened popularity suppression | Accepted |
