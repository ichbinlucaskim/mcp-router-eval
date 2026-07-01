# 0010 — Limit GNN routers to R-GCN + GAT; GraphSAGE as lower-bound control; exclude Graph Transformer

## Status

Accepted

## Context

Proposal §5.2 originally floated a four-rung model ladder up to a Graph Transformer. Two facts make
that ladder the wrong shape for this project:

- **The graph is small** — 573 nodes (`docs/build-readiness-report.md`, graph stats).
- **The target signal is local** — a tool's answer lives in its dependency *neighborhood* (the tools
  it depends on / that depend on it), not in global node-pair relations across the whole graph.

Recorded before any GNN code per ADR 0009. Sourced from `docs/build-readiness-report.md` (graph
stats) and the model-scope discussion.

## Decision

- **R-GCN (`num_relations=4`) = primary.** Tests whether edge-**type**-aware message passing wins.
- **GAT = comparison.** Tests whether learned neighbor importance adds over relation typing.
- **GraphSAGE = lower-bound control.** A sanity floor — does message passing help at all? — not a
  contribution model.
- **Graph Transformer = EXCLUDED.** Global O(N²) attention has little upside at 573 nodes, risks
  overfitting, and demotes the relation structure that is central to the thesis.
- **The weighted-primary ablation stays on the edge-type axis** (drop param/tool × direct/indirect),
  not on stacking more model architectures.

## Consequences

- Bounded experiment matrix: **2 models + 1 control × seeds × slices**.
- The heavy ablation is **edge-type** (ablation B), which is thesis-relevant, rather than an
  architecture bake-off.
- Graph Transformer is noted as **future work** for large-scale tool graphs where global attention
  could pay off.

## Alternatives considered

- **Full SAGE → R-GCN → R-GAT → GT ladder** — rejected: experiment-matrix blowup and GT overfit risk
  at this scale.
- **R-GCN only** — rejected: cannot separate the relation-typing effect from the attention effect
  without GAT as a comparison.
