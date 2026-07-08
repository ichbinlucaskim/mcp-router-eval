#!/usr/bin/env python3
"""Grid-search tuning over the ADR 0025/0026 discrete space (ADR 0029). RUN THIS YOURSELF.

Per backbone (R-GCN / GAT / SAGE) it trains every config in the discrete grid — hidden {32,64,128},
GAT heads {2,4}, dropout {0.0,0.3,0.5}, τ {0.05,0.1,0.2}, lr {1e-3,5e-4}, weight_decay {1e-4,1e-3,1e-2},
and logQ correction strength α {0.0,0.5,1.0} (ADR 0031 amendment) — with best-validation checkpointing,
and selects the best config by **validation completion_rate** (mAP@10 tiebreaker), on the validation
split only (test untouched, ADR 0024).

Prerequisites: ``data/processed/`` present (run ``scripts/fetch_data.py`` + preprocess) and the BGE
embedding cache reachable (huggingface.co on first use); torch / torch-geometric already installed.

Outputs (to ``data/processed/eval/``, gitignored):
  - ``best_configs.json``  — the chosen config per backbone (feeds run_full_eval.py)
  - ``grid_log.jsonl``     — every config's validation score (auditable, no cherry-picking)

Rough runtime: a few hundred cheap full-batch configs per backbone → order of minutes–tens of minutes
on CPU for ``--backbone all``; use ``--backbone rgcn`` (or a larger ``--epochs``) to scope it.

Example:
    python scripts/run_grid_search.py --backbone all --data data/processed --epochs 30 --seed 0
"""
from __future__ import annotations

import argparse
from pathlib import Path

from mcp_router_eval.data.loader import load
from mcp_router_eval.embedding.local import LocalEmbedder
from mcp_router_eval.eval.harness import EVAL_DIR
from mcp_router_eval.eval.tuning import (
    ALPHAS,
    BACKBONES,
    DROPOUTS,
    GAT_HEADS,
    HIDDENS,
    LRS,
    TAUS,
    WEIGHT_DECAYS,
    grid_size,
    run_grid,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbone", choices=(*BACKBONES, "all"), default="all")
    ap.add_argument("--data", type=Path, default=None, help="processed data dir (default: repo default)")
    ap.add_argument("--cache", type=Path, default=None, help="embedding cache dir (default: repo default)")
    ap.add_argument("--out", type=Path, default=EVAL_DIR, help="output dir (gitignored)")
    ap.add_argument("--epochs", type=int, default=30, help="per-config training budget (best-val kept)")
    ap.add_argument("--seed", type=int, default=0, help="split + init seed (must match run_full_eval)")
    ap.add_argument("--k", type=int, default=10, help="retrieval cutoff for the mAP tiebreaker")
    args = ap.parse_args()

    backbones = BACKBONES if args.backbone == "all" else (args.backbone,)
    total = sum(grid_size(b) for b in backbones)
    print(f"[grid] backbones={backbones} configs={total} (per-backbone: "
          f"{ {b: grid_size(b) for b in backbones} }) epochs={args.epochs} seed={args.seed}", flush=True)
    print(f"[grid] space: hidden={HIDDENS} dropout={DROPOUTS} heads(GAT)={GAT_HEADS} "
          f"tau={TAUS} lr={LRS} weight_decay={WEIGHT_DECAYS} alpha={ALPHAS}", flush=True)

    dataset = load(args.data) if args.data else load()
    embedder = LocalEmbedder(cache_dir=args.cache) if args.cache else LocalEmbedder()

    # run_grid emits live per-config progress (counter, elapsed, ETA), per-backbone best, and paths.
    run_grid(dataset, embedder, backbones=backbones, epochs=args.epochs,
             seed=args.seed, k=args.k, out_dir=args.out)


if __name__ == "__main__":
    main()
