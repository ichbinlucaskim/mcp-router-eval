# mcp-router-eval

Contract-driven evaluation of GNN tool routing in agentic MCP pipelines — a comparative study of five routers behind one contract.

## Problem

An MCP agent has to route a natural-language query to the right tools, but tools have **dependencies**: one tool needs argument values produced by another, so a plan is only executable if its full dependency closure is present and correctly ordered. The hard, unsolved part is not ranking the one obvious tool — it is **dependency-closure recovery**: retrieving the low-visibility tools a query never mentions but its execution requires. On ToolLinkOS these dependencies are typically **semantically unrelated** to the main tool (lexical-Jaccard mean 0.08; 54% of dependency pairs share zero tokens), so dense retrieval plausibly misses them — and a missed dependency is a plan that cannot run.

## Approach

Five routers — **BM25**, **NaiveRAG** (dense), **HybridRAG**, **graph Traversal**, and a dependency-aware learned **GNN** — rank tools behind a single `Router` contract. A deterministic **Contract Layer** then expands each router's top-k into its required-argument dependency closure, checks structural invariants, and attributes any failure to **ROUTING / CONTRACT / EXECUTION**; a deterministic mock executor runs the plan in topological order and returns a structural-completion verdict. Learning is one stage of the system, not the whole system — the GNN only *ranks*; the closure expansion, completion gate, attribution, and metrics are deterministic and byte-identical across every router, so the comparison isolates ranking quality.

## Results — the honest finding

Measured on the leakage-safe **test** split (236 queries, `k=10`) from one full run; GNN as **mean over 5 seeds**, baselines deterministic ([ADR 0028/0029](docs/README.md)):

| class | router | overall completion | deep-slice `transfer_loss` |
|---|---|---|---|
| dense retrieval | **NaiveRAG** | **0.979** | **0.000** |
| dense + sparse | HybridRAG | 0.936 | 0.000 |
| graph traversal | Traversal | 0.877 | 0.000 |
| sparse | BM25 | 0.725 | 0.077 |
| learned GNN | R-GCN / GAT / SAGE | 0.000 / **0.052** / 0.000 | n/a |

Completion is variant-A structural completion (PRIMARY, [ADR 0030](docs/adr/0030-completion-required-set.md)); `transfer_loss` on the deep slice (closure-depth ≥ 6) is spine-conditioned ([ADR-0028 amendment](docs/adr/0028-evaluation-metrics.md)).

**The finding:** a learning-free dense-retrieval baseline (**NaiveRAG, 0.979 completion**) **beats the dependency-aware GNN** (**≤ 0.052**; R-GCN/SAGE 0.000, GAT 0.052) on identical BGE features. The GNN collapses via message-passing **hub amplification** — the hub `get_wifi_status` (in-degree **371**) dominates aggregation — compounded by a **frequency-biased loss** that is rewarded for ranking the ~80%-gold system tools regardless of the query. This is a documented **negative result** whose conclusion **aligns with the benchmark's own SOTA**: *Graph RAG-Tool Fusion* solves the dependency problem with dense retrieval + deterministic traversal, not a learned GNN ([arXiv:2502.07223](https://arxiv.org/abs/2502.07223)).

**How the collapse was pinned down** (the value is the diagnosis, not just the score): an **isolation probe** — a GNN whose only difference from NaiveRAG is message passing — collapses identically to 0.000, with node embeddings over-smoothed (mean pairwise cosine 0.501 → 0.862); a **fairness audit** confirms the hub is the data's not our construction, the control is symmetric, and the gate is uniform across routers; and standard remedies (logQ debiasing, GCNII initial residual) do not move completion off the floor. Full write-up: **[docs/findings-gnn-collapse.md](docs/findings-gnn-collapse.md)**.

Honest caveats: the result is bounded to this **graph class** (heterophilic, hub-dominated, frequency-biased labels), not a general verdict on GNNs — on homophilic graphs GNNs remain effective; GAT's 0.052 is a real but marginal micro-signal that *supports* the frequency/hub account rather than escaping the collapse; and a homophily↔`transfer_loss` correlation was not computable GNN-side (the GNN's `transfer_loss` is `n/a` under the retrieval collapse) and is not fabricated.

## Pipeline

```
┌────────────────────────────────────────────────────────────────────────────┐
│ mcp-router-eval — Evaluation Pipeline (structural-completion routing)      │
│ Route an MCP query → recover the dependency closure → run it → attribute   │
└────────────────────────────────────────────────────────────────────────────┘


[0] PROBLEM DEFINITION
    ├─ task    : select the tool set that STRUCTURALLY COMPLETES the query
    ├─ metric  : completion · retrieval(mAP@10) · spine-cond. transfer_loss
    │            ✗ full-golden recall as PRIMARY (label-noisy, unreachable@k)
    └─ success : recover the required-arg closure, not just the main tool
              │        └─ metric · gate · attribution fixed here (ADR 0028/0030)
              ▼
╔═══════════════════════════ PHASE 1 — DATA LAYER ═══════════════════════════╗
║                                                                            ║
║ [1a] INGESTION                                                             ║
║      ToolLinkOS JSON, fetched at pinned commit b630b98 (not redistributed) ║
║      └─ 573 tools · 1,569 queries · 1,496 dependency edges                 ║
║              │                                                             ║
║              ▼                                                             ║
║ [1b] PREPROCESS   ✗ read raw                                               ║
║      normalize dirty types + malformed rows → JSONL (ADR 0011/0014)        ║
║              │                                                             ║
║              ▼                                                             ║
║ [1c] GRAPH BUILD → VALIDATION GATE   ◄──────── hard gate                   ║
║      4 typed edges: PARAMETER_* (ordering) · TOOL_* (representation)       ║
║      ├─ assert PARAMETER_* ordering sub-graph is ACYCLIC (ADR 0012/0013)   ║
║      └─ FAIL loudly on any cycle       PASS → continue                     ║
║                                                                            ║
╚═══════════════════════════════════════╪════════════════════════════════════╝
                                        ▼
╔══════════════════════ PHASE 2 — SPLIT (leakage-safe) ══════════════════════╗
║                                                                            ║
║ [2a] QUERY-LEVEL SPLIT   ✗ random / row-level leakage                      ║
║      transductive graph shared; supervision is not (ADR 0024)              ║
║              │                                                             ║
║              ▼                                                             ║
║ [2b] TRAIN-ONLY STATS + TUNING-ONLY VALIDATION                             ║
║      frequencies/normalizers fit on train; tune on val, report on test     ║
║                                                                            ║
╚═══════════════════════════════════════╪════════════════════════════════════╝
                                        ▼
[3] FEATURES — BGE node embeddings of the SAME tool_document text for every
    router (ADR 0003/0020)   ✗ per-router text (would confound the comparison)
              │
              ▼
[4] ROUTER (ranking only)     BM25 · NaiveRAG · HybridRAG · Traversal · GNN
    pure top-k ranking behind one Router contract (ADR 0018)
    ├─ HybridRAG = convex combination of scores  ✗ RRF (ADR 0019)
    └─ GNN = query-conditioned node scoring  ✗ link-pred / node-clf (ADR 0022)
              │
              ▼
╔═══════════════════ TRAINING LOOP — GNN only (per epoch) ═══════════════════╗
║                                                                            ║
║ [5] FORWARD     R-GCN / GAT / SAGE encoder → two-tower late-cosine score   ║
║             ▼                                                              ║
║ [6] LOSS     masked InfoNCE · in-batch negs · FN filter (ADR 0023/0026)    ║
║             ▼                                                              ║
║ [7] DEBIAS      logQ popularity correction, removed at inference (ADR 0031)║
║             ▼                                                              ║
║ [8] OPTIMIZE    AdamW → checkpoint on val completion (mAP@10 tiebreak)     ║
║                                                                            ║
╚═══════════════════════════════════════╪════════════════════════════════════╝
                                        ▼
[9] HYPERPARAMETER TUNING   deterministic grid search, multi-seed (ADR 0029)
    hidden · dropout · heads · τ · lr · weight_decay · α(logQ) · α_res
              │
              ▼
╔════════════════ PHASE 10 — CONTRACT LAYER (deterministic) ═════════════════╗
║                                                                            ║
║ [10a] CLOSURE EXPANSION   shared, identical for every router               ║
║       pull the required-arg PARAMETER_* dependency closure (ADR 0018)      ║
║              │                                                             ║
║              ▼                                                             ║
║ [10b] INVARIANTS + COMPLETION GATE   ◄─── ML proposes, RULES dispose       ║
║       closure-complete · no dangling params · variant-A (ADR 0030)         ║
║              │                                                             ║
║              ▼                                                             ║
║ [10c] ATTRIBUTION  ROUTING | CONTRACT | EXECUTION (upstream-wins, ADR 0018)║
║                                                                            ║
╚═══════════════════════════════════════╪════════════════════════════════════╝
                                        ▼
╔════════════════ PHASE 11 — EXECUTOR (mock, deterministic) ═════════════════╗
║                                                                            ║
║ [11a] RUN PLAN in topological order   ✗ Claude-SDK on the critical path    ║
║       deterministic mock runner is PRIMARY (ADR 0015); SDK replay is a demo║
║              │                                                             ║
║              ▼                                                             ║
║ [11b] STRUCTURAL COMPLETION VERDICT + measured latency (ADR 0004/0017)     ║
║                                                                            ║
╚═══════════════════════════════════════╪════════════════════════════════════╝
                                        ▼
╔═══════════════ PHASE 12 — EVALUATION (test split, run once) ═══════════════╗
║                                                                            ║
║ [12a] METRICS   retrieval · structural completion · transfer_loss          ║
║              │                                                             ║
║              ▼                                                             ║
║ [12b] DEPTH SLICES   shallow 2-3 / deep ≥6 (ADR 0005)                      ║
║              │                                                             ║
║              ▼                                                             ║
║ [12c] ROUTER SHOWDOWN   5 routers, one contract → the headline table       ║
║                                                                            ║
╚═══════════════════════════════════════╪════════════════════════════════════╝


┌────────────────────────────────────────────────────────────────────────────┐
│ learned / deterministic boundary                                           │
│   learned      : GNN router — ranks tools (PHASES 4–9)                     │
│   deterministic: closure · invariants · completion gate · attribution ·    │
│                  mock executor · metrics  (byte-identical across routers)  │
│   deferred     : confidence/homophily gate · Claude-SDK replay (demo)      │
└────────────────────────────────────────────────────────────────────────────┘
```

## What's learned vs deterministic

The system draws a hard line between what is learned and what is not.

- **Learned** — only the **GNN router** (R-GCN / GAT / SAGE), which ranks tools from graph structure and BGE node features. It emits a ranking; it decides nothing else.
- **Non-learned rankers** — BM25, NaiveRAG, HybridRAG, and Traversal use no training.
- **Deterministic** — the Contract Layer (closure expansion, invariants, variant-A completion verdict), the ROUTING/CONTRACT/EXECUTION attribution (upstream-wins rule), the mock executor, and the metrics/slices are plain code, **byte-identical across all five routers**. The comparison therefore isolates ranking, not plumbing.

## Task & metrics

The task is **structural-completion routing**: given a query, select the tool set whose execution structurally completes it — i.e. recover the required-argument `PARAMETER_*` dependency closure (variant A, [ADR 0030](docs/adr/0030-completion-required-set.md)), not merely the obvious main tool.

Evaluation reports three groups sliced by closure depth ([ADR 0005/0028](docs/README.md)): **retrieval** (mAP@10 and rank stats), **structural completion** (the PRIMARY gate), and the north-star **`transfer_loss` = 1 − P(completion | retrieved the spine)**. Full-golden recall is explicitly rejected as the PRIMARY signal — the stored gold sets are label-noisy and unattainable at `k=10` on deep queries, which flatters or breaks every router equally; conditioning on the required-argument **spine** ([ADR-0028 amendment](docs/adr/0028-evaluation-metrics.md)) is the trustworthy measure of whether retrieval actually converts to an executable plan.

## What makes this rigorous

- **One contract, five routers** — closure expansion, the completion gate, and attribution are byte-identical across BM25 / NaiveRAG / HybridRAG / Traversal / GNN, so a result reflects ranking, not evaluation asymmetry ([ADR 0018/0020](docs/README.md)).
- **Deterministic failure attribution** — every failure is blamed ROUTING / CONTRACT / EXECUTION by a deterministic upstream-wins rule, not a heuristic, so a router's loss is explained, not just measured.
- **Leakage control** — query-level split with train-only statistics and tuning only on validation ([ADR 0024/0029](docs/README.md)); the transductive graph is shared but supervision is not.
- **Pinned, non-redistributed data** — ToolLinkOS is fetched at a pinned upstream commit and never re-hosted (see *Data & attribution*).
- **The negative result is audited, not asserted** — an isolation probe, a fairness audit, and remedy probes back the GNN-collapse claim ([docs/findings-gnn-collapse.md](docs/findings-gnn-collapse.md)).

## Quickstart

Python 3.11+ is required.

```bash
python scripts/fetch_data.py     # fetch ToolLinkOS at the pinned commit (not redistributed)
pytest                           # test suite (pyproject sets pythonpath=src)
```

The GNN router track additionally needs `torch` and `torch-geometric`; the baselines and contract/executor layers do not.

## Reproduce

The full five-router comparison is one command:

```bash
PYTHONPATH=src python scripts/run_full_eval.py \
  --config data/processed/eval/full_eval_gnn_config.json --seeds 5
```

It trains each GNN backbone over 5 seeds (baselines need no training), runs all five routers on the **test** split, and writes `full_eval.{json,txt}` to `data/processed/eval/` (gitignored).

**Determinism.** Deterministic given the seeds ([ADR 0029](docs/adr/0029-validation-tuning-protocol.md)); the dataset is pinned to upstream commit `b630b98`, never a moving `HEAD`. The GNN config JSON is gitignored — its committed provenance (reproduce it verbatim) is [`docs/full-eval-gnn-config.md`](docs/full-eval-gnn-config.md), or produce `best_configs.json` first with `scripts/run_grid_search.py`.

## Repository layout

- `src/mcp_router_eval/` — `routers/` (BM25, NaiveRAG, HybridRAG, Traversal, GNN + shared closure), `contract_layer/` (invariants, attribution, gate), `executor/` (mock runner + SDK replay adapter), `embedding/` (provider interface + local BGE), `eval/` (metrics, harness, slices, tuning), `data/` (preprocess, loader, graph build).
- `scripts/` — `fetch_data.py`, `run_grid_search.py`, `run_full_eval.py`.
- `docs/` — [`findings-gnn-collapse.md`](docs/findings-gnn-collapse.md) (the study), reference reports, and the ADR index ([`docs/README.md`](docs/README.md)); `capstone_proposal.md` is the full design.

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

This project's own code is released under the [MIT License](LICENSE).

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

## Status

All five routers stand behind one contract, plus the evaluation harness and comparative study — complete on `main`. **200 tests** cover the contract layer, data pipeline, executor, routers, and the eval metrics/slices. The headline is the documented GNN-collapse negative result above. Intentional, deferred stubs: the confidence/`homophily_local` gate (`contract_layer/gate.py`) and the Claude Code SDK replay adapter (`executor/claude_exec.py`, demonstration only, off the critical path — [ADR 0015](docs/adr/0015-executor-mock-primary-sdk-replay.md)).

## Future work

- **Gate** — tune the confidence / `homophily_local` gate against `completion_rate` (`contract_layer/gate.py`); the full router set now exists, so it is unblocked.
- **SDK replay adapter** — `executor/claude_exec.py`, a demonstration path that replays a plan through Claude Code via the agent SDK (off the critical path).

## Decision record

Every non-trivial decision is a numbered ADR in [`docs/adr/`](docs/adr/), written **before** the code it governs (ADRs 0001–0031; index in [`docs/README.md`](docs/README.md)). Highlights: sole benchmark (0001), structural-completion proxy (0004), closure-depth slices (0005), GNN scope + query-conditioned formulation (0010/0022), negative sampling + logQ debiasing (0023/0031), evaluation metrics + transfer loss (0028), and the variant-A completion required-set (**0030**).
