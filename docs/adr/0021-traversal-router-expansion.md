# 0021 — Traversal router keeps its own dependency expansion (explicit exception to pure-ranking), then passes through the shared closure

## Status

Accepted

## Context

The traversal baseline reproduces **Graph RAG-Tool Fusion** — the method (and dataset, ToolLinkOS) our
whole benchmark is built on. That method is, by construction, *hybrid initial retrieval → dependency
traversal*: it "combines vector-based retrieval with efficient graph traversal to capture all relevant
tools (nodes) along with any nested dependencies (edges)" over a tool knowledge graph
([Graph RAG-Tool Fusion, arXiv:2502.07223](https://arxiv.org/abs/2502.07223)). The traversal step **is
the method** — its retrieval quality comes from walking dependency edges, not from ranking alone.

This collides with our router interface. ADR 0018 (2026-07-05 amendment) mandates that routers do
**pure ranking** and that **closure expansion lives in a single shared stage**, so ablation A swaps only
the ranker and holds expansion fixed. If we forced the traversal router to hand its bare top-k to the
shared closure and do no traversal of its own, it would **no longer be Graph RAG-Tool Fusion** and could
not reproduce that paper's numbers — the very thing a faithful baseline must do.

Three further findings shape the design:

- **Hold the graph machinery fixed as a controlled variable.** RLM-on-KG runs all its experiments over
  "the same KG construction and storage layer," varying only the controller — treating the graph
  substrate as a controlled variable so comparisons are attributable to the method, not the plumbing
  ([RLM-on-KG, arXiv:2604.17056](https://arxiv.org/abs/2604.17056)). Our analogue: the **shared closure
  stage is the fixed substrate** every router (traversal included) passes through.
- **Traversal method is a first-class, categorized dimension**, not an interchangeable detail — PolyG
  builds a taxonomy of graph-question patterns and matches a traversal strategy to each, showing the
  traversal strategy itself is a design axis worth naming and comparing
  ([PolyG, arXiv:2504.02112](https://arxiv.org/abs/2504.02112)). So the traversal router's expansion
  deserves to be preserved and labeled as method-intrinsic, not silently replaced.
- **Do not expand indiscriminately.** Connecting all neighbors regardless of edge type "introduces
  noisy data and can exceed the context window due to rapid subgraph growth"
  ([GraphRunner, arXiv:2507.08945](https://arxiv.org/abs/2507.08945)). Traversal must therefore walk the
  **typed** dependency edges, not every neighbor.

## Decision

- **The traversal router HAS its own dependency expansion.** It performs hybrid **initial retrieval**
  (reusing `HybridRAGRouter`, ADR 0019) → **traverses the dependency edges** to refine its ranking. This
  is the essence of the method and is what reproduces Graph RAG-Tool Fusion (arXiv:2502.07223).
- **It still passes through the shared closure stage afterward.** Exactly like every other router, its
  top-k then goes through the shared closure (ADR 0018 amendment) so that final `selected_tools`
  closure-completeness is guaranteed **identically** across all routers.
- **Traversal is `PARAMETER_*`-centric** (ADR 0013): it walks the typed dependency edges, **not** every
  neighbor — avoiding the noise / subgraph blow-up GraphRunner warns about.
- **This is the explicit, sole exception to the pure-ranking rule.** The traversal router is the *only*
  router that carries its own expansion, justified because expansion is intrinsic to the method and
  required to reproduce the origin paper. It is documented as such; no other router gets this exception.

## Consequences

- The traversal router's **method is preserved** — it uses dependency structure the way the paper does,
  so its numbers are comparable to Graph RAG-Tool Fusion's.
- Final closure is **commonly controlled**: every router (traversal included) ends by passing through
  the shared closure, so `selected_tools` closure-completeness is handled the same way for all.
- **Ablation A still holds.** "Initial retrieval + dependency traversal" is *this router's method*
  (the thing being swapped); the shared closure stage remains the fixed substrate (the thing held
  constant, per the RLM-on-KG controlled-variable discipline).
- The exception is **bounded and named** — one router, documented — so it does not erode the
  pure-ranking rule for the others.

## Alternatives considered

- **Replace traversal expansion with the shared closure** (make it pure-ranking like the rest) —
  rejected: it would no longer be Graph RAG-Tool Fusion and could not reproduce the origin paper's
  results; the traversal *is* the method.
- **Let the traversal router skip the shared closure** (its own expansion only) — rejected: its final
  `selected_tools` would be closure-handled differently from every other router, making the comparison
  unfair. It keeps its traversal **and** still passes through the shared closure.

## Sources

- Graph RAG-Tool Fusion — hybrid initial retrieval → dependency-edge traversal over a tool graph (the
  method + ToolLinkOS dataset this baseline reproduces): <https://arxiv.org/abs/2502.07223>
- RLM-on-KG — all experiments share the same KG construction/storage layer (graph substrate as a
  controlled variable): <https://arxiv.org/abs/2604.17056>
- PolyG — traversal strategy is a first-class, categorized dimension (a taxonomy of graph-question
  patterns matched to traversal): <https://arxiv.org/abs/2504.02112>
- GraphRunner — connecting all neighbors regardless of edge type injects noise and blows up the
  subgraph: <https://arxiv.org/abs/2507.08945>

## Amendment 2026-07-05 — exact method (block-interleaving) and reranking excluded

This refines (does **not** overwrite) the Decision above by pinning the *exact* traversal algorithm to
the origin paper and recording that the paper's optional LLM reranker is excluded. Verified against
Graph RAG-Tool Fusion's Algorithm 1 and §7.1 ([arXiv:2502.07223](https://arxiv.org/abs/2502.07223)).

### Exact method (Algorithm 1)

1. **Initial hybrid retrieval** — take the top-k tools from hybrid retrieval (reuse `HybridRAGRouter`,
   ADR 0019). The paper uses `k = 3`.
2. **Per-tool DFS for dependencies** — for each retrieved tool `t`, depth-first-search its
   `PARAMETER_*` dependencies up to a per-tool depth limit `d_limit`, appending each dependency **only
   if not already in the list**. Paper's Algorithm 1, verbatim:
   > "for each tool d ∈ DFS(t, KG) up to d_limit do / if d ∉ S_graph_list then / Append d to
   > S_graph_list"
3. **Block-interleaving order** — tools are processed sequentially and each retrieved tool is
   **immediately followed by its own dependencies**, i.e. `[vector tool 1, its deps]`, then
   `[vector tool 2, its deps]`, … concatenated into one de-duplicated list, then **truncated to the
   final top-K**:
   > "Limit S_graph_list to final_top_K"

This is **neither a plain closure add nor a score recompute**: it **preserves the initial vector
ranking order** and inserts each tool's dependencies directly after it. The block-interleaving *is* the
router's ranking output (its `ranked_tools` / top-k).

### Reranking excluded

The paper reports two configurations — **standard** ("no RR") and **+reranking** ("w/ RR", ~7–14%
absolute mAP@10 gain). The reranker is **LLM-based** (gpt-4o-2024-08-06) and **optional**. It conflicts
with our determinism / low-cost stance (ADR 0015), so this baseline reproduces the **standard,
no-reranking** version. LLM reranking is recorded as **future work only**, not implemented.

### Rationale

§7.1 states the premise our whole thesis rests on: *"Since tool dependencies are often semantically
unrelated to the main tool, naïve RAG struggles to retrieve all relevant dependencies."* — i.e.
**low-homophily dependencies** are exactly what graph traversal recovers and dense/lexical ranking
misses. Reproducing Algorithm 1's interleaved DFS traversal is therefore essential to represent that
method faithfully.

### Relationship to the original 0021 decision

Consistent with ADR 0021: the **block-interleaving is the router's method** (its ranking), and the
**shared closure stage still runs afterward** to guarantee final `selected_tools` closure-completeness
identically to every other router. Edges stay `PARAMETER_*`-centric (ADR 0013); `d_limit` is the
paper's **per-tool depth parameter** (recorded per run for reproducibility).
