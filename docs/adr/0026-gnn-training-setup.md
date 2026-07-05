# 0026 — GNN training setup: InfoNCE over in-batch negatives; AdamW; warm-up not required at our scale

## Status

Accepted

## Context

The GNN router's design is locked — query-conditioned late-cosine scoring (ADR 0022 + amendment),
in-batch negatives with a dependency-structure false-negative filter (ADR 0023), a query-level
transductive split (ADR 0024), and a fixed 2-layer architecture with searchable width/heads/dropout
(ADR 0025). To train it we still need a **loss, an optimizer, and a schedule**. ADR 0025 deferred
exactly these optimization hyperparameters to here.

**Sourcing stance (honesty).** The loss / optimizer / schedule choices below are **standard
dense-retrieval and GNN practice** combined with **this project's own logic** (they follow directly
from ADR 0022/0023) — they are **not** claims drawn from specific papers, so **no paper is cited for
them**. Earlier drafts attached arXiv identifiers to these points; on verification several of those did
not support the specific claims, so they are **omitted rather than asserted** (consistent with the
standing rule: do not manufacture citations). The **one** load-bearing external citation in this ADR is
**AdamW**, which is verified below.

## Decision

- **Loss = InfoNCE over in-batch negatives.** This is not a new choice so much as the composition of
  decisions already made: ADR 0023's **in-batch negatives** supply the negative pool; ADR 0022's
  **cosine** is exactly the similarity inside InfoNCE (a temperature-scaled softmax over cosine
  similarities, positive vs. negatives); ADR 0023's **dependency-structure filter** removes false
  negatives from that pool before the softmax. The **temperature τ** is tuned on validation (ADR 0024).
  A **triplet / margin** loss is kept as an ablation alternative.

- **Optimizer = AdamW.** For adaptive optimizers, L2 regularization and weight decay are **not**
  equivalent; AdamW decouples the weight decay from the gradient update, restoring correct weight-decay
  regularization and improving generalization — at the default learning rate `0.001` AdamW gave a ~15%
  relative test-error improvement over Adam on CIFAR-10 / ImageNet32x32
  ([Loshchilov & Hutter, "Decoupled Weight Decay Regularization", ICLR 2019,
  arXiv:1711.05101](https://arxiv.org/abs/1711.05101)). We use AdamW so weight decay regularizes the
  GNN correctly.

- **Learning rate = 1e-3** as the standard starting point (AdamW's own default), tuned on validation.

- **Warm-up = not required at our scale.** LR warm-up primarily addresses (a) large-Transformer
  training and (b) the gradient variance of **mini-batch subgraph sampling** on large graphs. Neither
  applies here: our graph is **573 nodes trained full-batch** with a **shallow 2-layer** GNN. Because
  GAT uses attention, a **short warm-up is retained as an option for GAT only**, to be enabled *only if*
  early-epoch instability is observed — not by default.

- **LR decay = `ReduceLROnPlateau` (on the validation metric) or cosine** — the specific schedule is
  chosen on validation, not fixed here.

- **Searched on validation** (ADR 0024/0025 tuning frame, **train-only statistics**, ADR 0024 leakage
  rule): temperature **τ**, weight decay (**1e-4 – 1e-2**), and learning rate around **1e-3**.
  **Gradient clipping** is available as an optional stability measure.

## Consequences

- The loss is **consistent with the scoring**: InfoNCE's similarity *is* the late cosine (ADR 0022) and
  its negatives *are* the filtered in-batch pool (ADR 0023) — no impedance mismatch between how the
  model scores and how it is trained.
- **AdamW** gives correct weight-decay regularization (the verified failure mode of plain Adam), which
  matters for generalization on a small (573-node) graph.
- **Skipping warm-up** matches our small, full-batch, shallow-GNN scale; the GAT-only option is a cheap
  safety valve without imposing warm-up everywhere.
- All training hyperparameters are **validation-tuned with train-only statistics**, preserving the
  ADR 0024 no-leakage guarantee.

## Alternatives considered

- **Pure InfoNCE requiring very large batches** — its usual weakness (needing many negatives) is
  **mitigated by in-batch negatives** (ADR 0023), so batch size is not a blocker at our scale.
- **Plain Adam** — rejected: it couples weight decay incorrectly for adaptive updates (the verified
  AdamW result); AdamW is used instead.
- **Mandatory warm-up everywhere** — rejected: unnecessary for a small, full-batch, 2-layer GNN; a
  GAT-only, opt-in warm-up is retained for attention stability.
- **BCE / pointwise loss** — rejected: it discards ranking signal and is less aligned with the
  query-conditioned *ranking* objective (ADR 0022) than a contrastive InfoNCE loss.

## Sources

- Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (ICLR 2019) — AdamW decouples weight
  decay from the Adam update, fixing Adam's weight-decay coupling and improving generalization; default
  lr 0.001: <https://arxiv.org/abs/1711.05101>

*(This is the only external citation in this ADR. The loss, in-batch-negative construction, warm-up
stance, and schedule are standard practice / project logic following ADR 0022/0023/0024/0025, not paper
claims — deliberately uncited rather than backed by unverified references.)*
