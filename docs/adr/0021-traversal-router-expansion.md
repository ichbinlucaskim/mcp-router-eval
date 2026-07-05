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
