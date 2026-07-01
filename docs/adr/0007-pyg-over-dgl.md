# 0007 — GNN framework = PyTorch Geometric (not DGL)

## Status

Accepted

## Context

The proposal leaves the GNN framework open ("PyG/DGL"). The environment audit
(`docs/build-readiness-report.md` §1) found:

- **PyTorch Geometric 2.8.0 is already installed and verified** here: `RGCNConv`, `GATConv`, and
  `SAGEConv` import and run a forward pass without `torch-scatter` on this machine.
- **DGL is not installed** and would require a second install with a heavier heterograph API.
- The tool graph is tiny (573 nodes), so framework-level performance is irrelevant.

## Decision

Use **PyTorch Geometric** for all graph construction and GNN routers (GraphSAGE control,
query-conditioned R-GCN/GAT). `torch-scatter` is optional (declared as an extra), installed only if
a segment-op error appears.

## Consequences

- No second framework install; typed-edge R-GCN and GAT are first-class.
- `data/graph_build.py` emits a PyG graph object; `routers/gnn.py` builds on PyG layers.
- `pyproject.toml` lists `torch-geometric` as a core dep and `torch-scatter` under an optional
  `scatter` extra.

## Alternatives considered

- **DGL** — rejected: not installed, heavier heterograph API, no upside on a 573-node graph.
