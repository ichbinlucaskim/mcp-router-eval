# 0025 — GNN architecture hyperparameters: 2 layers fixed; hidden dim / GAT heads / dropout searched on validation

## Status

Accepted

## Context

The GNN backbones are set (R-GCN + GAT, GraphSAGE control; ADR 0010) and the scoring is late cosine
over a two-tower design (ADR 0022 + amendment), but the **architecture hyperparameters** — number of
layers, hidden dimension, attention heads, dropout — are unset. Choosing them arbitrarily would blur
whether a GNN loss is due to **hyperparameters** or **method** (the weak-baseline problem the router
comparison must avoid). Two facts about our setting matter: the tool graph is **small and relatively
dense** (573 nodes, ~6.3 dependencies/node), and the split is **transductive** (ADR 0024).

What the literature supports (verified this session):

- **Depth: keep it shallow.** Over-smoothing is the established failure mode of deep GNNs — "as the
  number of layers increases, node representations become nearly indistinguishable and model
  performance degrades." A graph method whose whole point is *distinguishing* nodes must therefore stay
  shallow: the degradation of "node distinctiveness due to oversmoothing in GNN-based embeddings" is
  exactly what a shallow encoder avoids
  ([Graph Alignment via Dual-Pass Spectral Encoding, arXiv:2509.09597](https://arxiv.org/abs/2509.09597)).
  For a small, dense graph a 2-layer encoder already reaches most nodes' relevant neighborhoods, so
  extra depth mostly adds smoothing, not signal.
- **Width/heads/dropout: tune per dataset, don't hardcode.** GNN architecture hyperparameters are
  dataset-dependent and are best chosen by search — e.g. a per-dataset **Bayesian optimization** of
  GCN / GAT / GIN across seven assay datasets, rather than a single fixed configuration
  ([Comparison of Optimised Geometric Deep Learning Architectures over Toxicological Assays,
  arXiv:2507.17775](https://arxiv.org/abs/2507.17775)). So width/heads/dropout belong in a **validation
  search**, not the ADR's fixed set.
- **Keep the three backbones comparable.** Standard practice benchmarks SAGE / GCN / GAT under the same
  protocol ([GLNN, arXiv:2110.08727](https://arxiv.org/abs/2110.08727); comparative GNN model
  evaluation, [arXiv:2203.12363](https://arxiv.org/abs/2203.12363)), so our three backbones share the
  same layer count and the same hidden/dropout search ranges.

**Honest limitation.** These sources are node-classification / graph-alignment / molecular / fraud
tasks — **not** query-conditioned tool scoring. The **principles** transfer (shallow to avoid
over-smoothing; search width/heads/dropout on validation; benchmark backbones uniformly); the **optimal
values are tuned on OUR validation** (ADR 0024), not read off any paper. The numeric *ranges* below are
standard GNN defaults, deliberately narrow — not claimed to be optimal from a citation.

## Decision

**Fixed (architecture):**

- **Number of layers = 2.** Standard for small, dense graphs; avoids over-smoothing / loss of node
  distinctiveness (same spirit as ADR 0010 excluding the Graph Transformer). Applies to all three
  backbones.
- **R-GCN `num_relations` = 4** (already fixed, ADR 0006/0013).
- **L2 normalization before the cosine** (ADR 0022 amendment).
- **No residual connections** — unnecessary at 2 layers (residuals matter for deep GNNs).

**Searched on the validation split (ADR 0024 tuning; standard-default ranges):**

- **Hidden dim ∈ {32, 64, 128}** (default 64).
- **GAT heads ∈ {2, 4}** (default 2) — applies to GAT only.
- **Dropout ∈ {0.0, 0.3, 0.5}** (default 0.5).

Ranges apply across all three backbones (heads to GAT only). The selected values are recorded per run.

## Deferred (optimization hyperparameters — the stage-2 training ADR)

Explicitly **not** decided here: **learning rate, LR warm-up, LR decay / schedule, weight decay,
optimizer, batch size, epochs**. These are decided in the stage-2 training ADR against current
practice (in particular, whether a small GAT benefits from warm-up).

## Consequences

- Architecture is fixed **on evidence** (over-smoothing → shallow) and searched **narrowly**, so a GNN
  result cannot be dismissed as a "hyperparameter vs. method" artifact.
- A narrow validation search (3 × 2 × 3 at most, heads only for GAT) limits overfitting on a 573-node
  graph while still adapting width/heads/dropout to our data.
- The three backbones stay **comparable** — same depth, same hidden/dropout ranges — so R-GCN vs GAT vs
  SAGE differences are architectural, not hyperparameter luck.

## Alternatives considered

- **Deeper GNN (3+ layers)** — rejected: over-smoothing on a small, dense graph collapses node
  distinctiveness (the verified failure mode), which is fatal for a node-discrimination task.
- **Large hidden dim (256+)** — rejected: overfitting risk on 573 nodes; kept out of the search range.
- **GAT heads = 8** — rejected: excess heads add capacity/averaging without benefit at our scale; the
  search caps heads at 4.

## Sources

- Graph Alignment via Dual-Pass Spectral Encoding — over-smoothing degrades node distinctiveness in GNN
  embeddings (motivates a shallow encoder): <https://arxiv.org/abs/2509.09597>
- Comparison of Optimised Geometric Deep Learning Architectures over Toxicological Assays — per-dataset
  Bayesian optimization of GCN/GAT/GIN (architecture hyperparameters are tuned, not hardcoded):
  <https://arxiv.org/abs/2507.17775>
- GLNN — standard SAGE/GCN/GAT benchmarking under one protocol: <https://arxiv.org/abs/2110.08727>
- Ethereum Fraud Detection with Heterogeneous GNNs — comparative GNN model evaluation:
  <https://arxiv.org/abs/2203.12363>

*(Honesty note: the brief also cited arXiv:2303.00995 for "2 layers best / hidden saturates ~32" — on
inspection that paper is a recommendation contrastive-learning work that does not discuss layer depth
or hidden-dim saturation, so it is **dropped**. Specific numeric anchors from the brief — hidden-dim
saturation ~32, and per-paper GAT-heads / hidden-128 / dropout-0.5 values — could **not** be verified
this session; the search ranges above are therefore stated as standard GNN defaults to be tuned on our
validation, not asserted as any paper's optimum.)*
