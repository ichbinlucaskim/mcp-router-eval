# mcp-router-eval

Contract-driven evaluation of GNN tool routing in agentic MCP pipelines. This project measures
the **transfer loss** from retrieval accuracy to end-to-end task completion across a three-layer
pipeline (GNN Router → executor-agnostic Contract Layer → Claude Code executor), and tests whether
a dependency-aware learned GNN router reduces that loss on dependency/composite queries. The middle
**Contract Layer** — typed I/O schemas, dependency-closure invariants, and a router/contract/executor
failure-attribution taxonomy — is the engineering centerpiece.

See [`capstone_proposal.md`](capstone_proposal.md) for the full design (architecture, data contracts,
metrics, and the 14-week task plan) and [`docs/`](docs/) for architecture decision records (ADRs)
capturing the choices made so far.

> Status: **scaffolding only** — modules are stubs; no logic is implemented yet.

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
