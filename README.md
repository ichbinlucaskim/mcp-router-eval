# mcp-router-eval

Contract-driven evaluation of GNN tool routing in agentic MCP pipelines. It measures the **transfer
loss** from retrieval accuracy to end-to-end task completion across a three-layer pipeline (Router →
executor-agnostic Contract Layer → Claude Code executor) and compares five routers — BM25, NaiveRAG,
HybridRAG, Traversal, and a dependency-aware learned GNN — on dependency/composite queries. The middle
**Contract Layer** — typed I/O schemas, dependency-closure invariants, and a router/contract/executor
failure-attribution taxonomy — is the engineering centerpiece.

**Headline result:** on the ToolLinkOS benchmark, a learning-free dense-retrieval baseline (**NaiveRAG,
0.979** structural completion) **beats the dependency-aware GNN** (**≤ 0.052**; R-GCN/SAGE 0.000, GAT
0.052). The GNN collapses via message-passing **hub amplification** compounded by a **frequency-biased
loss** — a documented **negative result**, consistent with the benchmark's own SOTA (dense retrieval +
deterministic traversal, not a learned GNN). Full write-up: [`docs/findings-gnn-collapse.md`](docs/findings-gnn-collapse.md).

## How to run

```bash
# run the full five-router comparative evaluation (test split, 5 seeds)
PYTHONPATH=src python scripts/run_full_eval.py \
  --config data/processed/eval/full_eval_gnn_config.json --seeds 5

# tests
PYTHONPATH=src pytest
```

The GNN config's provenance (and how to reproduce the gitignored JSON) is in
[`docs/full-eval-gnn-config.md`](docs/full-eval-gnn-config.md).

## Repository structure

- `src/mcp_router_eval/` — `routers/` (BM25, NaiveRAG, HybridRAG, Traversal, GNN + shared closure),
  `contract_layer/` (invariants, attribution, gate), `executor/` (mock runner + SDK replay adapter),
  `embedding/` (provider interface + local BGE), `eval/` (metrics, harness, slices, tuning), `data/`
  (preprocess, loader, graph build).
- `scripts/` — `fetch_data.py` (pinned dataset fetch), `run_full_eval.py`, `run_grid_search.py`.
- `docs/` — findings, reference reports, and the ADR index; `capstone_proposal.md` — full design.

## Where to read

- [`docs/findings-gnn-collapse.md`](docs/findings-gnn-collapse.md) — the core deliverable (GNN-collapse
  negative result: mechanism, controlled evidence, fairness audit, related work).
- [`docs/README.md`](docs/README.md) — project status and the ADR index.
- [`capstone_proposal.md`](capstone_proposal.md) — full design (architecture, data contracts, metrics).

## License

This project's own code is released under the [MIT License](LICENSE).

## Data & attribution

**ToolLinkOS** (the primary benchmark) — source:
[github.com/EliasLumer/Graph-RAG-Tool-Fusion-ToolLinkOS](https://github.com/EliasLumer/Graph-RAG-Tool-Fusion-ToolLinkOS),
**MIT-licensed**. This repository does **not** redistribute the dataset: the JSON files are
gitignored and are **fetched** on demand via [`scripts/fetch_data.py`](scripts/fetch_data.py), pinned
to upstream commit `b630b98656e25c3b83a71ea0406572add38ae46d`. See
[`data/raw/SOURCE.md`](data/raw/SOURCE.md) for the pinned provenance and checksums.

ToolLinkOS tools are **fictional and non-functional** (research/illustrative only); this project
treats task completion as a *structural proxy* accordingly (see `docs/adr/0004-completion-structural-proxy.md`).

**ToolLinkOS vs ToolSandbox:** ToolLinkOS (MIT) is the sole primary benchmark. **ToolSandbox**
([apple/ToolSandbox](https://github.com/apple/ToolSandbox)) is under an Apple custom license and is
**stretch-only, referenced by link/submodule and never redistributed** here (see
`docs/adr/0001-toollinkos-sole-primary-benchmark.md`).

If you use the ToolLinkOS dataset or the Graph RAG-Tool Fusion method, cite the upstream work
(arXiv 2502.07223):

```bibtex
@misc{lumer2025graphragtoolfusion,
      title={Graph RAG-Tool Fusion},
      author={Elias Lumer and Pradeep Honaganahalli Basavaraju and Myles Mason and James A. Burke and Vamse Kumar Subbiah},
      year={2025},
      eprint={2502.07223},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.07223},
}
```
