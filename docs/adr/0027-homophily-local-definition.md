# 0027 — homophily_local = node-level feature homophily (mean cosine to PARAMETER_* dependency neighbors)

## Status

Accepted

## Context

ADR 0018 assigned `homophily_local` as a **GNN-only** signal — baseline routers emit the neutral
sentinel, only the GNN computes a real value — but left the **formula** open. We fix it here before
implementing the GNNRouter (stage 3).

The obstacle: the textbook local homophily is **label-based** — the fraction of a node's neighbors that
share its class label — and our **tool nodes have no class labels**. The **feature** variant of
homophily, which replaces the label match with a feature *similarity*, does apply and is exactly what we
need. Two verified sources define it:

- **RAW-GNN** generalizes edge homophily by replacing the label indicator `1{Y_u = Y_v}` with a
  **feature similarity** between connected nodes, using **cosine**: `sim(i, j) = cos(x_i, x_j)`. The
  paper also shows that low feature-homophily (heterophilous) graphs need a **larger receptive field** —
  structure matters more precisely when neighbors are feature-dissimilar
  ([Jin et al., "RAW-GNN: RAndom Walk Aggregation based Graph Neural Network",
  arXiv:2206.13953](https://arxiv.org/abs/2206.13953)).
- **Disentangling Homophily** defines **"Local Similarity"** as a **node-level feature homophily**: the
  per-node mean cosine similarity to its neighbors, averaged over nodes —
  `h_LS-cos = (1/|V|) Σ_u (1/d_u) Σ_{v ∈ N_u} cos(X_u, X_v)` (their Local-Similarity metric)
  ([Zheng, Luan & Chen, "What Is Missing In Homophily? Disentangling Graph Homophily For Graph Neural
  Networks", arXiv:2406.18854](https://arxiv.org/abs/2406.18854)).

**Honest limitation.** Both papers are **node-classification graphs with class labels**; they define the
*feature* homophily as a label-free alternative, which is the part we adopt. We apply that feature
definition to a **tool-dependency graph** where there are no labels at all and the neighborhood of
interest is the **`PARAMETER_*` dependency** neighborhood — the formula transfers, the task does not.

## Decision

- **`homophily_local(t)` = the mean cosine similarity between tool `t`'s node embedding and the
  embeddings of `t`'s `PARAMETER_*` dependency neighbors** (ADR 0006/0013 edge types). This is the
  node-level Local Similarity (Disentangling) / generalized edge homophily (RAW-GNN), **adapted to our
  label-free graph and restricted to the `PARAMETER_*` neighborhood**:

  `homophily_local(t) = (1/|D(t)|) Σ_{d ∈ D(t)} cos( h_t , h_d )`,  where `D(t)` = `t`'s `PARAMETER_*`
  dependency neighbors and `h_·` are node embeddings.

- **Which embedding: the GNN-refined node embeddings** the router already computes (ADR 0022) — no extra
  encoder pass.

- **Sentinel-safe for dependency-free tools.** If a tool has **no** `PARAMETER_*` dependencies, its
  local homophily is **undefined** → emit the ADR-0018 **sentinel** for that tool, **not** a fabricated
  `0.0` (a real `0.0` would falsely read as maximal heterophily).

- **GNN-only.** Baseline routers still emit the ADR-0018 sentinel; only the GNNRouter computes a real
  value (ADR 0018 unchanged).

- **Interpretation — the thesis signal.** **Low** `homophily_local` = **heterophily** = the dependency
  is semantically *far* from the tool, so pure vector retrieval (which assumes homophily) is likely to
  **miss** it and graph structure is needed. This is exactly our data: dependency pairs have **mean
  lexical-Jaccard 0.08, with 54% (809/1,496) at zero overlap** (`docs/feasibility-completion.md`) — the
  low-homophily regime where RAW-GNN says the receptive field (graph structure) must do the work.

## Consequences

- Produces a **per-selected-tool, thesis-relevant** number — how heterophilous each routing decision's
  dependencies are — rather than one opaque graph-level scalar.
- Computed from **embeddings the router already has** (the GNN-refined node embeddings), so no added
  compute path.
- **Sentinel-safe**: dependency-free tools are handled by the ADR-0018 convention, not a misleading `0`.
- Ties directly to the **deep-dependency slices** (ADR 0005) in evaluation: low-homophily dependencies
  concentrate in the deep-closure bucket, where the GNN should help most.

## Alternatives considered

- **Label-based local homophily** (fraction of same-label neighbors) — rejected: we have **no labels**;
  the feature variant is the applicable definition.
- **Global feature homophily over ALL edges** (one number for the whole graph) — rejected: we want a
  **per-selected-tool** signal for the gate and attribution, not a single graph statistic; and only the
  `PARAMETER_*` neighborhood is routing-relevant (`TOOL_*` edges are representation-only, ADR 0013).
- **Euclidean local similarity** (`h_LS-euc`) — noted, but **cosine** matches our late-cosine scoring
  (ADR 0022) and is scale-free, so cosine is chosen for consistency.

## Sources

- Jin et al., "RAW-GNN: RAndom Walk Aggregation based Graph Neural Network" (arXiv:2206.13953) —
  generalized edge homophily replaces the label indicator with a feature cosine `sim(i,j)=cos(x_i,x_j)`;
  low feature-homophily needs a larger receptive field: <https://arxiv.org/abs/2206.13953>
- Zheng, Luan & Chen, "What Is Missing In Homophily? Disentangling Graph Homophily For Graph Neural
  Networks" (arXiv:2406.18854) — node-level feature homophily "Local Similarity", per-node mean cosine
  to neighbors: <https://arxiv.org/abs/2406.18854>

*(Only these two homophily papers are cited, both verified this session — arXiv ids, authors/titles, and
the feature-cosine / node-level definitions. The lexical-Jaccard figure is the project's own
`docs/feasibility-completion.md` measurement, not an external citation.)*
