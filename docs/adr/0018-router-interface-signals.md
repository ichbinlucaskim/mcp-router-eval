# 0018 — Router interface: normalized confidence is common to all routers; model-specific signals are not forced

## Status

Accepted

## Context

The router track produces several **heterogeneous** routers — lexical BM25, vector RAG, dependency
traversal, and the dependency-aware GNN — that must all satisfy the **same** frozen `RouteResult`
contract (§3.1). Two of that contract's fields are not equally meaningful across router families:

- `confidence` (constrained to `[0,1]`) — every router can emit *some* per-candidate score, but the
  scores live on **incomparable scales** (a BM25 term-weight sum is not a cosine similarity is not a
  GNN logit). Writing raw per-router scores into `confidence` makes the field apples-to-oranges across
  routers, which breaks both the gate (tuned against one notion of confidence) and the router-vs-router
  comparison at the heart of the thesis.
- `homophily_local` (unbounded) — a **GNN-specific** signal: the mean semantic similarity of a tool's
  selected neighborhood. It is conceptually meaningful for a graph model but **meaningless for lexical
  BM25**, which has no neighborhood notion at all.

The literature supports both a shared, normalized confidence and *not* forcing model-specific signals:

- A robust router/reranker design can rely on the **candidate order** alone and need **not** require
  retriever-specific scores or calibration: CAR "only relies on the baseline order and generator-side
  confidence changes, and it does not require retriever-specific scores, calibration, or architectural
  assumptions," which is exactly why it works across heterogeneous backbones (BM25 and Contriever)
  ([CAR, arXiv:2605.04495](https://arxiv.org/abs/2605.04495v1)).
- When BM25 and neural scores **must** be compared/fused, the standard remedy is **min-max
  normalization over the per-query top candidates**: RIRAG normalizes each score as
  `(s(q,c) − m_q) / (M_q − m_q)`, with `m_q`/`M_q` the min/max over the top candidates for `q`, before a
  convex combination of BM25 and neural scores
  ([RIRAG, arXiv:2409.05677](https://arxiv.org/html/2409.05677)). That a BM25 score needs an explicit,
  comparable **representation** (not its raw value) to combine well with neural models is also the
  finding of the BM25-injection reranker study
  ([arXiv:2301.09728](https://arxiv.org/abs/2301.09728)).
- Baselines must be **strong**, or the comparison is vacuous. BM25 is a formidable baseline — BEIR
  "reveals BM25 to be a strong baseline for generalization, generally outperforming many other, more
  complex approaches," frequently beating dense retrievers out-of-domain
  ([BEIR overview](https://www.emergentmind.com/topics/beir-benchmark)). A credible BM25 therefore uses
  **tuned** parameters `k1 = 0.9, b = 0.4` (the Pyserini defaults, distinct from the classic
  `1.2 / 0.75`) rather than arbitrary library defaults
  ([arXiv:2404.01012](https://arxiv.org/pdf/2404.01012)).

## Decision

**Common fields — every router fills them, the same way:**

- `ranked_tools` — the full per-query ranking (feeds retrieval metrics).
- `selected_tools` — the top-k after **dependency-closure expansion** (ADR 0006/0013).
- `router_name` — for ablation bookkeeping.
- `confidence` — **normalized to `[0,1]` via min-max over the per-query top candidates**, using the
  **same** normalization method for every router, so the gate and cross-router comparison are fair.
  Min-max is `(s − m_q) / (M_q − m_q)` over the query's top candidates (RIRAG); when `M_q == m_q`
  (degenerate, all-equal scores) the router emits a documented constant (e.g. `1.0`) rather than
  dividing by zero.

**Model-specific field — not forced on routers that cannot compute it meaningfully:**

- `homophily_local` — filled with a **meaningful** value **only by the GNN**. Lexical/vector baselines
  set it to a **documented neutral sentinel** (`0.0`, "not applicable") purely to satisfy the contract;
  the value is **not** computed for them and must not be read as a real neighbor-similarity signal.

**Baselines are strong, not throwaway:**

- BM25 uses the **tuned** parameters `k1 = 0.9, b = 0.4`, not library defaults. A weak baseline would
  make the thesis comparison meaningless.

## Consequences

- The gate and every router-vs-router comparison operate on a **single, comparable** confidence scale.
- The `RouteResult` contract stays valid for all routers even though `homophily_local` is GNN-only —
  no router is forced to fabricate a signal it cannot compute (the CAR principle).
- Because the baseline is genuinely strong, **BM25 may beat the GNN** on some slices — an honest,
  expected possible outcome, consistent with the proposal's risk section, not a failure of the setup.
- The `homophily_local` sentinel convention must be documented at the field's point of use so a `0.0`
  from a baseline is never confused with a computed GNN value.

## Alternatives considered

- **Raw per-router scores in `confidence`** — rejected: BM25/cosine/GNN scales are incommensurable, so
  cross-router comparison and gate tuning would be invalid.
- **Force every router to compute `homophily_local`** — rejected: it is meaningless for lexical
  retrieval (no neighborhood), and mandating a retriever-specific signal violates the CAR "no
  retriever-specific requirement" principle.

## Sources

- CAR — retriever-independent; relies on baseline order, requires no retriever-specific
  scores/calibration: <https://arxiv.org/abs/2605.04495v1>
- RIRAG — min-max normalization `(s − m_q)/(M_q − m_q)` over top candidates to fuse/compare BM25 vs
  neural scores: <https://arxiv.org/html/2409.05677>
- Injecting the BM25 Score as Text Improves BERT-Based Re-rankers — BM25-score *representation* (not
  raw value) matters when combining with neural models: <https://arxiv.org/abs/2301.09728>
- BM25 tuned parameters `k1 = 0.9, b = 0.4` (Pyserini defaults, vs classic `1.2 / 0.75`):
  <https://arxiv.org/pdf/2404.01012>
- BEIR — BM25 is a strong baseline that frequently beats dense retrievers out-of-domain:
  <https://www.emergentmind.com/topics/beir-benchmark>

## Amendment 2026-07-05 — closure expansion is a shared post-processing stage, not per-router work

This refines (does **not** overwrite) the original Decision above. The original listed `selected_tools`
"incl. dependency-closure expansion" among the fields every router fills, which reads as if each router
performs its own expansion. That is corrected here.

### Amended decision

Dependency-closure expansion is **not** performed inside each router. Routers do **pure ranking only**:
they produce `ranked_tools` (the full ranking) and their top-k. A **single shared post-processing
stage** then expands every router's top-k **identically** — adding the `PARAMETER_*` dependency closure
(reusing the loader's ordering/closure helper, ADR 0012/0013) — and fills `selected_tools` and
`closure_edges`. The `RouteResult` **contract is unchanged** (same fields, same `[0,1]` normalized
`confidence`, same GNN-only `homophily_local` sentinel rule); only **who fills** `selected_tools` /
`closure_edges` changes: **router → shared stage**.

### Rationale (verified sources)

- Retrieval pipelines are **staged** — pre-retrieval / retrieval / post-retrieval — and operations that
  refine an already-retrieved candidate set (re-ranking, filtering, and here **closure expansion**)
  belong in the **post-retrieval** stage, not inside the retriever. Modular RAG makes this explicit by
  "treating each component as an independent module," so a shared refinement step is a first-class
  pipeline stage rather than retriever-internal logic ([Retrieval-Augmented Generation for LLMs: A
  Survey — Gao et al., arXiv:2312.10997](https://arxiv.org/abs/2312.10997)). Reasoning-intensive
  retrieval systems are likewise built as **separable multi-stage modules** (document processing →
  query expansion → retriever → reranker), reinforcing that expansion is a stage, not part of the
  ranker ([DIVER, arXiv:2508.07995](https://arxiv.org/abs/2508.07995v1)).
- **Ablation-hygiene (the decisive reason).** Proposal **ablation A** swaps *only the router* and holds
  everything else fixed. If each router expanded its own closure, the expansion logic would vary with
  the router and **contaminate** the comparison — differences in `selected_tools` could come from
  expansion, not ranking. Extracting expansion into one shared stage guarantees every router receives
  the **identical** closure, so only **pure ranking quality** is compared. This mirrors how the
  contract layer already separated invariants/attribution *out of* the routers (ADR 0013 / T1.2–1.3):
  shared, router-independent work lives in shared stages.

*(Note on sourcing, per the standing rule: the specific claims "closure expansion = post-retrieval per
MDPI Information 9/12/320" and "DIVER excludes its non-shared module from the main experiments" could
not be verified this session, so they are **not** asserted here. The staged/modular-pipeline rationale
is cited from the sources above that were verified; the fairness argument rests on the project's own
ablation A.)*

### Consequence

Ablation A stays **clean** — expansion is held constant across all routers, so the experiment isolates
ranking. Every router gets the identical closure. The change is a relocation of *work*, not a contract
change: `selected_tools` / `closure_edges` remain `RouteResult` fields, now populated by the shared
post-processing stage. Baselines (BM25, vector, traversal) and the GNN all feed the same expander.
