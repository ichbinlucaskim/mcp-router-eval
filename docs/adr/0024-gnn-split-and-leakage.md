# 0024 — GNN split is query-level (graph structure intentionally shared, transductive); leakage prevented by train-only statistics and tuning-only validation

## Status

Accepted

## Context

The GNN router's formulation is **query-conditioned node scoring** (ADR 0022): the tool graph (573
nodes) is **fixed** and, for a given **query**, we predict which tools are relevant. The natural unit of
a split is therefore the **query** (we have 1,569, ADR 0008) — not nodes and not edges. The tool nodes
are shared prior knowledge, so "sharing the graph across splits" is a legitimate transductive design —
but that same sharing is exactly where naïve graph splits leak, so it must be handled deliberately.

The literature is clear on both the leak and the legitimate transductive setting:

- **Transductive random node/edge splits leak.** On the Amazon co-purchase graph, a transductive /
  edge-based train/test split gave "unusually high validation accuracy and AUC **before any training**,"
  which the authors diagnose as leakage — the model "memorized node identities or neighborhood
  overlaps." Their fix is to **partition the node set *before* edge filtering** so no test node is seen
  in training ([Amazon Co-purchase GNN, arXiv:2508.14059](https://arxiv.org/abs/2508.14059)). The lesson
  we take: split by a **unit**, and never let that unit straddle train and test.
- **Fixed nodes → transductive is a legitimate setting, not a bug.** MOTIVE defines it precisely: "in
  the transductive case, the nodes are fixed from the start of training and the model will predict edges
  on entities it has learned on," versus inductive cold-start where a node was unseen in training
  ([MOTIVE, arXiv:2406.08649](https://arxiv.org/abs/2406.08649)). Our 573 tools are fixed prior
  knowledge, so a transductive graph is correct — provided the *prediction target* (which tools a query
  needs) is not leaked.
- **Ratios, repeats, and the tuning/report discipline.** ES-GNN splits 60/20/20 and evaluates over **10
  random splits** per dataset ([ES-GNN, arXiv:2205.13700](https://arxiv.org/abs/2205.13700)); GNN
  training strategies are reported across both transductive and inductive settings
  ([TuneUp, arXiv:2210.14843](https://arxiv.org/abs/2210.14843)). Standard hygiene: **validation tunes,
  test is reported once.**

**Honest limitation.** These sources are **link-prediction / node-classification / recommendation**
splits. None does a **query-level** split for query-conditioned tool scoring — that unit is specific to
our formulation. We **combine** their principles (unit-level partitioning to stop leakage; transductive
graph-sharing as legitimate when nodes are fixed; multi-seed ratios; tune-on-val/report-on-test); we do
not copy any single paper's protocol.

## Decision

- **Split unit = query** (not nodes, not edges). Partition the 1,569 queries into train / val / test;
  **the same query never appears in two splits.**
- **The tool graph structure is intentionally shared** across all splits. The 573-tool dependency graph
  is fixed prior knowledge, **not** a prediction target, so this is a legitimate **transductive**
  setting (MOTIVE's definition). Stated explicitly so graph-sharing is not mistaken for leakage: tool
  **nodes** are visible in every split; the **test queries' gold tool sets** (the target) are never
  seen in training.
- **Leakage prevention:**
  1. **No query overlap** across splits.
  2. **Any fitted statistic is fit on TRAIN ONLY** and applied to val/test — min-max normalization
     parameters (ADR 0018), embedding-cache scope, any learned thresholds — **never** fit over all
     queries.
  3. **Validation is for hyperparameter tuning only; test is for the final report only** (reported
     once).
- **Ratio + repeats:** **70/15/15** (60/20/20 acceptable), evaluated over **multiple random seeds**
  (ES-GNN practice), with the **seed recorded per run** for reproducibility.

## Consequences

- Evaluation reflects **generalization to unseen queries**, which is what the routing claim needs.
- The shared tool graph is **honestly a transductive design**, documented rather than hidden; it is
  legitimate because the tools are fixed and the target (query→tools) is split.
- **Train-only statistics** close the preprocessing leak that would otherwise let test-set scale/scope
  bleed into training.
- **Multi-seed repeats** report variance across splits, not a single lucky partition.
- **Ties to the earlier build:** the closure-depth **slice buckets (ADR 0005, shallow vs deep)** are
  applied **within the test split**, so the deep-dependency claim is measured on held-out queries.

## Alternatives considered

- **Node / edge split** — rejected: the wrong unit for query-conditioned scoring (we score fixed nodes,
  we do not hold out nodes), and transductive random node/edge splits **leak** via neighborhood/identity
  memorization (Amazon Co-purchase).
- **A single split** — rejected: risk of a lucky/unlucky partition; use **multiple seeds** and report
  variance.
- **Fitting normalization (or any statistic) over all queries** — rejected: leaks val/test information
  into training; statistics are **train-only**.

## Sources

- Amazon Co-purchase GNN — transductive/edge random splits give suspiciously high pre-training AUC
  (leakage via node-identity/neighborhood memorization); fix = partition the unit before edge
  filtering: <https://arxiv.org/abs/2508.14059>
- MOTIVE — transductive = nodes fixed from the start, predicting on learned entities (legitimate);
  inductive = cold-start unseen nodes: <https://arxiv.org/abs/2406.08649>
- ES-GNN — 60/20/20 split, 10 random splits per dataset: <https://arxiv.org/abs/2205.13700>
- TuneUp — GNN training/evaluation across both transductive and inductive settings:
  <https://arxiv.org/abs/2210.14843>
