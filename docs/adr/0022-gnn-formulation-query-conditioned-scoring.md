# 0022 — GNN router formulation: query-conditioned node scoring (not link prediction / node classification)

## Status

Accepted

## Context

The router's output is a **tool ranking** (`ranked_tools`, ADR 0018). To realize the GNN router we must
choose a learning formulation that produces that ranking, and decide how the query conditions the GNN.
Three candidate formulations:

1. **Link prediction** — predict `(query, tool)` edges.
2. **Node classification** — label each tool relevant / not-relevant.
3. **Query-conditioned node scoring/ranking** — score each tool node's relevance *conditioned on the
   query*, and rank by that score.

The graph-retrieval literature consistently realizes query-relevant graph retrieval as **query-
conditioned node scoring**, over a GNN that refines node embeddings across the graph:

- A GNN can "learn to assign importance weights to nodes based on their relevance to the question and
  the relevance of their neighboring nodes," producing per-node importance scores used directly for
  retrieval ranking — this is exactly node scoring for ranking, implemented with lightweight GNNs in
  PyG ([GNN-RAG, arXiv:2405.20139](https://arxiv.org/abs/2405.20139)).
- Query-relevant retrieval is a **two-stage** "score nodes/edges *conditioned on the query* → extract a
  connected subgraph" pipeline ([G-Retriever, arXiv:2402.07630](https://arxiv.org/abs/2402.07630)) —
  the scoring stage is precisely query-conditioned node scoring; the extraction stage is the analogue
  of our shared closure stage (ADR 0021).
- Query conditioning is implemented as **query-aware attention plus a learned scoring head** (with
  query-guided pooling), built in PyTorch Geometric
  ([Query-Aware GNN for RAG, arXiv:2508.05647](https://arxiv.org/abs/2508.05647)).
- Query-conditioned GNNs, with **query conditioning as a critical ablated component**, drive
  graph-evidence retrieval ([TCAR-Gen, arXiv:2606.00029](https://arxiv.org/abs/2606.00029)).

**Honest limitation — none of these is our problem.** Every source above is a **document / knowledge /
QA graph** (text chunks, entities, temporal narratives) for retrieval-augmented *generation*. Ours is a
**typed tool-dependency graph** where the target is *routing tools by query*, and the thesis is that
graph structure recovers **low-homophily dependencies** dense retrieval misses. The **architectural
pattern** — a GNN refining node embeddings + a query-conditioned node relevance score — transfers
cleanly; the **task, graph semantics, and supervision** do not. We adopt the pattern, not any paper's
specifics, and validate on our own benchmark.

## Decision

- **Learning formulation = query-conditioned node scoring / ranking.** The GNN scores every tool node's
  relevance conditioned on the query; the ranked scores *are* `ranked_tools`. Not link prediction, not
  node classification.
- **Query conditioning = similarity between the query embedding and the graph-refined node
  embeddings.** The query is embedded with the **same BGE model** the vector baselines use (ADR 0003);
  the GNN (R-GCN / GAT / SAGE) refines each tool's node embedding over the dependency graph; relevance
  is `cos(query, node)` combined with a **learned scoring head**. Using the same BGE query embedding
  keeps the GNN comparable to the vector baselines (only the graph refinement differs).
- **Backbones reaffirmed (ADR 0010):** R-GCN + GAT, GraphSAGE as a control; Graph Transformer excluded.
- **Deferred to their own later ADRs (named now, not decided here):** the **supervision loss**, the
  **negative-sampling** scheme, and the **train / val / test split**.

## Consequences

- The output **maps directly to `RouteResult.ranked_tools`** — no conversion from an edge- or
  label-space back to a ranking (the awkwardness of the rejected formulations).
- Sharing the **same BGE query embedding** with the vector baselines makes the router comparison fair:
  the GNN's only added ingredient is graph refinement, so a win is attributable to the graph.
- Graph refinement gives the model a **path to catch low-homophily dependencies** — a tool whose text
  is semantically distant from the query can still score high via its dependency neighborhood — which
  is the thesis.
- Node scoring still feeds the **shared closure stage** (ADR 0018 amendment / 0021), like every router.

## Alternatives considered

- **Link prediction of `(query, tool)` edges** — rejected: indirect to a ranking output (needs a query
  node injected into the graph and edge scores converted back to a per-tool ranking); less natural for
  `RouteResult`.
- **Node classification (relevant / not)** — rejected: a binary label discards the ranking information
  the retrieval metrics (mAP / nDCG) and the top-k selection need.
- **Concatenating the query into node features** (instead of similarity-based conditioning) — kept
  **open as a possible ablation** (an alternative conditioning mechanism), but similarity-based
  conditioning against the shared BGE embedding is **primary**, because it keeps the GNN directly
  comparable to the dense baselines.

## Sources

- GNN-RAG — GNN assigns query-relevance importance scores to nodes for retrieval ranking (lightweight
  GNN, PyG): <https://arxiv.org/abs/2405.20139>
- G-Retriever — two-stage query-conditioned node/edge scoring → subgraph extraction:
  <https://arxiv.org/abs/2402.07630>
- Query-Aware GNN for RAG — query-aware attention + learned scoring head + query-guided pooling, in
  PyTorch Geometric: <https://arxiv.org/abs/2508.05647>
- TCAR-Gen — query-conditioned GNN with query conditioning as a critical component:
  <https://arxiv.org/abs/2606.00029>

*(All four are document / knowledge / QA graphs, not tool-dependency graphs — the architectural pattern
transfers, the task does not; see the honest-limitation note in Context.)*

## Amendment 2026-07-05 — scoring is late cosine (a two-tower design), not a query-node fusion MLP

This refines (does **not** overwrite) the Decision above by pinning down what the scoring function is,
before implementation. The original phrase "`cos(query, node)` combined with a **learned scoring head**"
could be misread as an MLP that ingests the query and node embeddings *together*. It is not.

### Scoring is late cosine

`score(query, tool) = cos( query_embedding , GNN-refined node_embedding )`. The two sides are computed
**independently** — the query tower is the BGE embedding (ADR 0003); the tool tower is the GNN encoder
over the dependency graph — and the **only** cross-tower interaction is this output cosine. There is
**no** MLP that takes the query and node embeddings jointly (no cross-attention, no query-node fusion
layer). This is a **two-tower** design.

### What "learned scoring head" actually means here

The learned parameters are: (1) the **GNN encoder weights** that produce the graph-refined node
embeddings, and *optionally* (2) a **per-tower linear projection** that maps each side into one shared
space, with **L2 normalization** before the cosine. Both are per-tower (they touch one side only) — the
towers stay independent. There is **no** query×node fusion network.

### Rationale (verified sources)

- A two-tower model is **late-interaction by construction**: "user and item features do not mix until
  the similarity stage" — the cross-tower exchange is restricted to the output similarity
  ([Two-Tower Models, Kumo.ai](https://kumo.ai/pyg/concepts/two-tower-model/); [two-tower retrieval
  overview](https://www.emergentmind.com/topics/two-tower-retrieval)).
- **Cosine keeps the towers independent** and is the standard efficient choice — it "keeps the
  two-towers independent and allows offline indexing and efficient serving via ANN"
  ([Etsy Unified Embedding, arXiv:2306.04833](https://arxiv.org/abs/2306.04833)); the key design
  principle of two-tower retrieval is precisely to keep the query and item embeddings **independent**
  after training, scored at the output ([end-to-end e-commerce two-tower,
  arXiv:2006.02282](https://arxiv.org/abs/2006.02282)).
- **Cross-attention / fusion scoring is richer but `O(N)`** and belongs to a **later ranking** stage,
  not first-stage retrieval — in practice two towers retrieve, a cross-attention re-ranker ranks
  ([Kumo.ai](https://kumo.ai/pyg/concepts/two-tower-model/)). Our router is a first-stage retriever, so
  late cosine is the right tier.
- **Decisive for us (fairness).** NaiveRAG already scores by **cosine** against the shared BGE
  embedding (ADR 0003 / 0020). If the GNN also scores by cosine against that same query embedding, the
  **only** difference between NaiveRAG and the GNN is the **graph refinement** of the tool tower — so a
  GNN win is attributable to the graph, not to a more expressive scorer. A query-node fusion MLP would
  confound that comparison (it adds scorer capacity on top of the graph).

### Honest correction

The original "cos + learned scoring head" wording is hereby read as **late cosine with the GNN encoder
(and an optional per-tower projection) as the only learned parts** — *not* a query-node fusion MLP. The
"concatenate the query into node features" alternative in the original Alternatives section remains an
open ablation; it is a *conditioning* variant, still scored by late cosine, not a fusion scorer.

### Honest limitation

The cited sources are **recommendation / search two-tower** systems, not a GNN over a tool-dependency
graph. The **"GNN-as-tower-encoder + late cosine"** pattern is established (Kumo.ai's GNN-enhanced
two-tower; the well-known PinSage GNN-embedding system is a familiar instance), but none is exactly our
tool-routing setting — we adopt the pattern and validate on our own benchmark.
