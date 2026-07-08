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

## Amendment 2026-07-05 — initial residual connection to counter hub-amplification collapse

This **supersedes the "No residual connections" line** of the Decision above (that one line only). The
2-layer depth, the hidden/dropout/heads search ranges, `num_relations = 4`, and L2-before-cosine are all
**unchanged**.

### Context — the "no residual" premise was refuted by measurement

The original Decision reasoned "**No residual connections** — unnecessary at 2 layers." The root-cause
diagnosis of the GNN collapse (git-clean, measured this session) refutes that premise **for our graph**:
even at 2 layers, message passing over **high-degree system-tool hubs** degrades the ranking.

- `get_wifi_status` has **in-degree 371** (it is a dependency-source of 371 tools); message passing pulls
  node embeddings toward a popularity/frequency-dominated region.
- Symptom, measured on the validation split (short-trained rgcn): **`corr(gold_freq, mean_rank) = −0.235`
  for the GNN** (it ranks by frequency) vs **≈ 0 (−0.032) for NaiveRAG** on the **identical** BGE
  features. The GNN buries the query-specific **main tool at median rank 272**, while **NaiveRAG on the
  same features ranks it at median 0** (0.970 variant-A completion vs the GNN's 0.000).
- ⇒ **The features are good; message passing degrades them.** The problem is not features (NaiveRAG wins
  on them) and not the ADR-0023 false-negative filter (refuted separately: `neg_on == neg_off`). It is MP
  hub amplification — which a residual back to the raw features directly counters. logQ (ADR 0031) targets
  a *negative-sampling* bias that is not the cause, so it fails and over-correction drops val_map.

### Decision — add a GCNII-style initial residual

Add an **initial residual** to each GNN layer, connecting every layer back to the **raw input node
features**:

> **`h^(l+1) = (1 − α_res) · MP(h^(l)) + α_res · h^(0)`**

where `h^(0)` is the raw input node representation (BGE text embedding + `is_core`, ADR 0003/0020) and
`MP(·)` is the backbone's message-passing layer. This **preserves the strong raw-feature signal** against
MP's hub amplification (unlike a plain layer-to-layer residual, the connection is always to `h^(0)`, so
the good features cannot be smoothed away over layers).

- **`α_res` is tuned on validation** (added to the ADR-0029 grid, alongside the other searched
  hyperparameters). Because our features are already strong (NaiveRAG **0.970**), a **large `α_res`**
  (strong feature preservation) is plausible, so the grid **should include high values**.
- **Identity mapping** (GCNII's *second* technique — a weighted identity in the layer transform) is kept
  as an **ablation, not primary**: its main benefit is stabilizing *very deep* stacks, and we are at 2
  layers. The initial residual alone is the primary change.

Applies to all three backbones (R-GCN / GAT / SAGE) uniformly, so they stay comparable.

### Source (verified this session)

- Chen, Wei, Huang, Ding & Li, *"Simple and Deep Graph Convolutional Networks"* (GCNII, ICML 2020,
  arXiv:2007.02133) — **verified this session:** the **initial residual** connects each layer to the
  first-layer representation `H^(0)`, and is shown to relieve over-smoothing **more effectively than a
  plain residual**: <https://arxiv.org/abs/2007.02133>

**Honest gap (depth).** GCNII targets **deep** GCNs (dozens of layers), where the initial residual fights
*over-smoothing with depth*. We apply the **initial-residual principle at 2 layers** for a different,
measured reason: to **preserve features against hub amplification**, not to enable depth. The principle
(always re-inject `h^(0)`) transfers; our motivation is shallower and hub-driven, not depth-driven.

### Evaluation link

Whether the initial residual lifts the GNN out of collapse — the previously-**xfailed** GNN completion
test (`test_full_pipeline_integration`, ADR-0030) flipping to **XPASS** — is the A-branch research
question. A **documented negative result remains acceptable** (ADR 0031): if even a feature-preserving
residual cannot beat NaiveRAG on this graph, that is itself a reportable finding about MP for tool
routing.
