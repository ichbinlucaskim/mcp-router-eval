#!/usr/bin/env python3
"""Full multi-seed evaluation across all five routers (ADR 0028/0029). RUN THIS YOURSELF.

Takes the best config per backbone (from ``run_grid_search.py``), trains each GNN backbone over
``--seeds`` seeds (baselines need no training), runs all five routers {BM25, NaiveRAG, HybridRAG,
Traversal, GNN} on the **full TEST split**, and reports the three metric groups per depth slice with
the deep-slice transfer_loss as the headline — GNN as **mean ± std** across seeds (ADR 0029). Test
split only (ADR 0024); deterministic given the seeds.

Prerequisites: ``data/processed/`` present, embedding cache reachable, torch/torch-geometric installed,
and ``best_configs.json`` produced by ``run_grid_search.py`` (same ``--seed`` there as seed[0] here).

Outputs (to ``data/processed/eval/``, gitignored):
  - ``full_eval.json`` — full comparison (baselines + per-seed GNN + mean±std summary)
  - ``full_eval.txt``  — readable table

Rough runtime: a few GNN trainings per backbone × seeds + one pass of five routers over the ~236 test
queries → order of minutes on CPU.

Example:
    python scripts/run_full_eval.py --config data/processed/eval/best_configs.json --seeds 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

from mcp_router_eval.data.loader import load
from mcp_router_eval.embedding.local import LocalEmbedder
from mcp_router_eval.eval.harness import EVAL_DIR
from mcp_router_eval.eval.tuning import load_best_configs, render_full_table, run_full_eval


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=EVAL_DIR / "best_configs.json",
                    help="best_configs.json from run_grid_search.py")
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=EVAL_DIR)
    ap.add_argument("--seeds", type=int, default=5, help="number of seeds for the GNN mean±std")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    dataset = load(args.data) if args.data else load()
    embedder = LocalEmbedder(cache_dir=args.cache) if args.cache else LocalEmbedder()
    best_configs = load_best_configs(args.config)
    seeds = list(range(args.seeds))
    print(f"[full-eval] backbones={list(best_configs)} seeds={seeds} k={args.k}")

    comparison = run_full_eval(dataset, embedder, best_configs, seeds=seeds, k=args.k, out_dir=args.out)
    print(f"[full-eval] wrote {args.out/'full_eval.json'} and {args.out/'full_eval.txt'}\n")
    print(render_full_table(comparison))


if __name__ == "__main__":
    main()
