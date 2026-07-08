# 0029 — Validation tuning protocol: deterministic grid search over the small discrete space, best by validation metric, multi-seed

## Status

Accepted

## Context

ADR 0025/0026 fixed **what** to search — architecture `hidden ∈ {32, 64, 128}`, GAT `heads ∈ {2, 4}`,
`dropout ∈ {0.0, 0.3, 0.5}`, plus training `τ`, `lr`, `weight_decay` — but not **how** the search is
run. Choosing the protocol arbitrarily would let a GNN win or loss be an artifact of the *tuning
method* rather than the model, which is exactly the "hyperparameter vs. method" confound ADR 0025 exists
to avoid. This is the last design decision before the full training + evaluation run.

Two facts settle it, and they are **standard, textbook practice** (stated as such — deliberately
uncited, per the standing rule not to manufacture citations):

- **The search space is small and discrete.** Per non-GAT backbone (R-GCN, SAGE): `hidden × dropout =
  3 × 3 = 9` configurations; GAT adds `heads` → `3 × 2 × 3 = 18`. The continuous training
  hyperparameters are searched over a **small explicit discrete set** of values (not a continuous
  optimizer), keeping the whole thing a finite grid.
- **Each run is cheap** — a 573-node, 2-layer, full-batch GNN (the profiling pass timed a
  forward+backward at ~3 ms; ADR stages).

**Grid search is the standard choice for a small discrete space** — exhaustive, deterministic,
reproducible. Random search and Bayesian optimization earn their keep on **high-dimensional or
expensive** spaces (they sample rather than enumerate); ours is neither, so their advantage does not
apply here.

## Decision

- **Protocol = grid search** over the ADR 0025/0026 discrete space, run **per backbone**
  (R-GCN / GAT / SAGE) so each model is tuned on equal footing. Continuous hyperparameters (`τ`, `lr`,
  `weight_decay`) enter the grid as a **small explicit discrete set** of values, so the search is a
  finite, fully enumerated grid.
- **Selection = best validation metric.** The configuration with the best **validation** score is
  chosen (ADR 0024: validation is tuning-only; the **test split is untouched** during tuning). The
  selection metric is **pre-declared** to avoid post-hoc cherry-picking:
  - **Primary: validation `completion_rate`** — the north-star-aligned structural-completion metric
    (ADR 0004/0028), computed on the validation split through the same pipeline.
  - **Tiebreaker: validation `mAP@10`** — the fine-grained ranking metric (comparable to the ToolLinkOS
    paper), used to break ties when `completion_rate` is equal (common on a small validation set).
  This ordered pair is fixed **before** the run; no other metric may be substituted after seeing results.
- **Early stopping on validation.** Each configuration trains with early stopping when the validation
  metric plateaus (ADR 0026's `ReduceLROnPlateau`/patience), so the grid's total cost stays bounded.
- **Reproducibility + multi-seed.** The grid is deterministic; the seed is fixed and recorded. The
  **chosen** configuration is then **repeated over multiple seeds** (the ES-GNN multi-seed practice
  already in ADR 0024) and reported as **mean ± variance**, so the headline is not a single lucky run.

## Consequences

- Tuning is **exhaustive over a small space**, deterministic, and reproducible — a GNN result cannot be
  dismissed as a tuning-method artifact.
- Selection is by a **single, pre-declared** validation metric (completion_rate, mAP@10 tiebreaker) with
  **no test leakage** (ADR 0024).
- The reported number is **multi-seed** (variance, not luck), matching the ADR 0024 protocol.
- This is the concrete "checkup" the tuning script implements: enumerate the grid, early-stop each
  config on validation, pick the best by the declared metric, then multi-seed the winner.

## Alternatives considered

- **Random search** — rejected: its advantage is efficiently sampling **high-dimensional** spaces; ours
  is small and discrete, where grid search is exhaustive *and* deterministic (random would be strictly
  worse — non-exhaustive with no upside here).
- **Bayesian optimization** — rejected: it pays off for **expensive** objectives or **continuous**
  spaces; our runs are cheap and the space is discrete, so the modeling overhead buys nothing.
- **Manual tuning** — rejected: not reproducible and it invites cherry-picking the exact confound this
  ADR removes.
